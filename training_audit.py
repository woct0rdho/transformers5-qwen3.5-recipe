"""Shared reporting and correctness checks for full-step training audits."""

import hashlib
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from transformers.integrations.gguf.gguf_quantized_parameter import (
    GgufQuantizedParameter,
)


def process_memory() -> dict[str, int]:
    """Read process RSS/file/private/swap and the machine's unified-pool state."""

    values: dict[str, int] = {}
    for line in Path("/proc/self/smaps_rollup").read_text().splitlines():
        key, _, rest = line.partition(":")
        if key in {"Rss", "Pss_File", "Private_Clean", "Private_Dirty", "Swap"}:
            values[f"process_{key.lower()}_bytes"] = int(rest.split()[0]) * 1024
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, rest = line.partition(":")
        # This device's memory is the machine's memory, so the machine's own accounting says how
        # much room a phase has. `MemAvailable` is the number that decides whether it will swap.
        if key in {
            "MemTotal",
            "MemFree",
            "MemAvailable",
            "Cached",
            "Shmem",
            "SwapTotal",
            "SwapFree",
        }:
            values[f"system_{key.lower()}_bytes"] = int(rest.split()[0]) * 1024
    return values


def accelerator_memory() -> dict[str, int]:
    """Synchronize and capture current and peak accelerator allocator state."""

    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "device_free_bytes": free,
        "device_total_bytes": total,
    }


def memory_snapshot() -> dict[str, int]:
    return accelerator_memory() | process_memory()


def run_phase(
    timeline: dict[str, Any],
    name: str,
    function: Callable[[], Any],
    *,
    reset_peak: bool = True,
) -> Any:
    """Run one synchronized phase and retain wall time plus peak memory."""

    torch.cuda.synchronize()
    if reset_peak:
        torch.cuda.reset_peak_memory_stats()
    before = process_memory()
    started = time.perf_counter()
    with torch.autograd.profiler.record_function(f"training_phase/{name}"):
        result = function()
    torch.cuda.synchronize()
    entry = {
        "seconds": time.perf_counter() - started,
        "memory": accelerator_memory(),
        "process_before": before,
        "process_after": process_memory(),
    }
    timeline.setdefault(name, []).append(entry)
    print(name, entry, flush=True)
    return result


def validate_loss_output(output: Any, label: str) -> None:
    """Reject a fused training output that materialized logits or auxiliary loss."""

    if getattr(output, "logits", None) is not None:
        raise RuntimeError(f"{label} materialized full logits")
    if getattr(output, "aux_loss", None) is not None:
        raise RuntimeError(f"{label} retained router auxiliary loss")
    loss = getattr(output, "loss", None)
    if loss is None or not bool(torch.isfinite(loss).item()):
        raise RuntimeError(f"{label} produced a missing or nonfinite loss")


def complete_training_update(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    *,
    label: str,
    max_grad_norm: float,
) -> dict[str, Any]:
    """Run one synchronized full update and retain phase times plus memory.

    The accelerator peak is reset after the incoming gradients are dropped, so
    every reported peak belongs to exactly this update and remains comparable
    across architectures and across extra-step and profiled executions.
    """

    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    phase_times: dict[str, Any] = {}

    started = time.perf_counter()
    with torch.autograd.profiler.record_function("training_phase/forward"):
        output = model(**batch, use_cache=False)
    torch.cuda.synchronize()
    phase_times["forward_seconds"] = time.perf_counter() - started
    validate_loss_output(output, label)

    started = time.perf_counter()
    with torch.autograd.profiler.record_function("training_phase/backward"):
        output.loss.backward()
    torch.cuda.synchronize()
    phase_times["backward_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    with torch.autograd.profiler.record_function("training_phase/gradient_clip"):
        norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            max_grad_norm,
        )
    torch.cuda.synchronize()
    phase_times["clip_seconds"] = time.perf_counter() - started
    if not bool(torch.isfinite(norm).item()):
        raise RuntimeError(f"{label} clipping norm is nonfinite")

    started = time.perf_counter()
    with torch.autograd.profiler.record_function("training_phase/optimizer"):
        optimizer.step()
    torch.cuda.synchronize()
    phase_times["optimizer_seconds"] = time.perf_counter() - started
    phase_times["loss"] = float(output.loss.detach())
    phase_times["clip_norm"] = float(norm)
    phase_times["memory"] = accelerator_memory()
    del output
    return phase_times


def clean_loading_info(loading_info: dict[str, Any]) -> dict[str, Any]:
    return {
        key: [str(item) for item in value] if isinstance(value, list) else str(value)
        for key, value in loading_info.items()
    }


def factor_family(name: str) -> str:
    factor = "A" if ".lora_A" in name else "B" if ".lora_B" in name else "other"
    if ".lora_A_down." in name or ".lora_B_down." in name:
        return f"expert_down_{factor}"
    if ".mlp.experts." in name:
        return f"expert_gate_up_{factor}"
    return f"ordinary_{factor}"


def _gradient_sample(gradient: torch.Tensor, size: int = 16) -> torch.Tensor:
    flat = gradient.detach().reshape(-1)
    stride = max(flat.numel() // size, 1)
    return flat[::stride][:size].float()


def summarize_gradients(model: torch.nn.Module, label: str) -> dict[str, Any]:
    """Summarize ownership, finiteness, sparsity, norms, and factor families."""

    groups: dict[str, list[torch.Tensor]] = defaultdict(list)
    missing: list[str] = []
    frozen_with_grad: list[str] = []
    samples: list[torch.Tensor] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            if parameter.grad is not None:
                frozen_with_grad.append(name)
            continue
        if parameter.grad is None:
            missing.append(name)
            continue
        groups[factor_family(name)].append(parameter.grad.detach())
        samples.append(_gradient_sample(parameter.grad))

    device = samples[0].device if samples else torch.device("cuda")
    all_sum_squares = torch.zeros((), device=device, dtype=torch.float64)
    all_maximum = torch.zeros((), device=device, dtype=torch.float32)
    group_report: dict[str, Any] = {}
    nonfinite_names: list[str] = []
    zero_names: list[str] = []
    for family, gradients in sorted(groups.items()):
        finite = torch.stack([torch.isfinite(gradient).all() for gradient in gradients])
        nonzero = torch.stack([torch.count_nonzero(gradient) for gradient in gradients])
        sum_squares = torch.stack(
            [gradient.float().square().sum().double() for gradient in gradients]
        ).sum()
        maximum = torch.stack(
            [gradient.abs().max().float() for gradient in gradients]
        ).max()
        finite_cpu = finite.cpu()
        nonzero_cpu = nonzero.cpu()
        group_report[family] = {
            "tensors": len(gradients),
            "finite_tensors": int(finite_cpu.sum()),
            "zero_tensors": int(torch.count_nonzero(nonzero_cpu == 0)),
            "nonzero_elements": int(nonzero_cpu.sum()),
            "elements": sum(gradient.numel() for gradient in gradients),
            "norm": float(sum_squares.sqrt()),
            "max_abs": float(maximum),
        }
        all_sum_squares += sum_squares
        all_maximum = torch.maximum(all_maximum, maximum)

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        if not bool(torch.isfinite(parameter.grad).all().item()):
            nonfinite_names.append(name)
        if torch.count_nonzero(parameter.grad).item() == 0:
            zero_names.append(name)
    sample = torch.cat(samples).cpu().numpy() if samples else None
    sample_hash = (
        hashlib.sha256(sample.tobytes()).hexdigest() if sample is not None else None
    )
    report = {
        "label": label,
        "groups": group_report,
        "missing": missing,
        "nonfinite": nonfinite_names,
        "zero": zero_names,
        "frozen_with_grad": frozen_with_grad,
        "packed_with_grad": [
            name
            for name, parameter in model.named_parameters()
            if isinstance(parameter, GgufQuantizedParameter)
            and parameter.grad is not None
        ],
        "total_norm": float(all_sum_squares.sqrt()),
        "max_abs": float(all_maximum),
        "sample_hash": sample_hash,
        "sample_values": sample.tolist() if sample is not None else [],
    }
    print(
        label,
        "missing",
        len(missing),
        "nonfinite",
        len(nonfinite_names),
        "zero",
        len(zero_names),
        "norm",
        report["total_norm"],
        flush=True,
    )
    return report


def representative_packed_state(model: torch.nn.Module) -> dict[str, Any]:
    """Sample one immutable packed parameter of every loaded quantization type."""

    selected: dict[int, tuple[str, GgufQuantizedParameter]] = {}
    for name, parameter in model.named_parameters():
        # A parameter with no payload cannot witness anything, so it never stands in for its quantization
        # type: the disk-backed PLE table is one, and the type it owns has to be sampled elsewhere.
        if isinstance(parameter, GgufQuantizedParameter) and parameter.numel():
            selected.setdefault(int(parameter.quant_type), (name, parameter))
    if not selected:
        raise RuntimeError("The audited model has no packed GGUF parameters.")

    result = {}
    for quant_type, (name, parameter) in sorted(selected.items()):
        payload = parameter.as_subclass(torch.Tensor).detach().reshape(-1)
        chunk = min(payload.numel(), 1024)
        middle = max((payload.numel() - chunk) // 2, 0)
        sample = torch.cat(
            (
                payload[:chunk].cpu(),
                payload[middle : middle + chunk].cpu(),
                payload[-chunk:].cpu(),
            )
        ).numpy()
        result[str(quant_type)] = {
            "name": name,
            "data_ptr": parameter.data_ptr(),
            "version": parameter._version,
            "sample_sha256": hashlib.sha256(sample.tobytes()).hexdigest(),
            "bytes": parameter.numel() * parameter.element_size(),
        }
    return result


def summarize_b_updates(model: torch.nn.Module) -> dict[str, Any]:
    """Audit an update against the known all-zero LoRA-B initialization."""

    names = []
    nonzero_counts = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and ".lora_B" in name:
            names.append(name)
            nonzero_counts.append(torch.count_nonzero(parameter.detach()))
    if not nonzero_counts:
        raise RuntimeError("The audited model has no trainable LoRA-B parameters.")
    counts = torch.stack(nonzero_counts).cpu().tolist()
    return {
        "tensors": len(counts),
        "changed_tensors": sum(value > 0 for value in counts),
        "changed_elements": sum(counts),
        "unchanged": [
            name for name, value in zip(names, counts, strict=True) if value == 0
        ],
    }


def audit_optimizer(
    optimizer: torch.optim.Optimizer, model: torch.nn.Module
) -> dict[str, Any]:
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer_parameters = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    trainable_ids = {id(parameter) for parameter in trainable}
    optimizer_ids = [id(parameter) for parameter in optimizer_parameters]
    report = {
        "class": f"{type(optimizer).__module__}.{type(optimizer).__name__}",
        "parameter_tensors": len(optimizer_parameters),
        "unique_parameter_tensors": len(set(optimizer_ids)),
        "missing_trainable_tensors": len(trainable_ids - set(optimizer_ids)),
        "extra_tensors": len(set(optimizer_ids) - trainable_ids),
        "groups": [
            {
                "tensors": len(group["params"]),
                "learning_rate": group["lr"],
                "weight_decay": group["weight_decay"],
            }
            for group in optimizer.param_groups
        ],
    }
    if (
        report["parameter_tensors"] != report["unique_parameter_tensors"]
        or report["missing_trainable_tensors"]
        or report["extra_tensors"]
    ):
        raise RuntimeError(f"adapter optimizer audit failed: {report}")
    return report


def validate_first_gradients(report: dict[str, Any], *, require_complete: bool) -> None:
    if (
        (require_complete and report["missing"])
        or report["nonfinite"]
        or report["frozen_with_grad"]
    ):
        raise RuntimeError("first backward violated gradient ownership or finiteness")
    if report["packed_with_grad"]:
        raise RuntimeError("packed base parameters received gradients")
    for family, values in report["groups"].items():
        expected_zero = family.endswith("_A")
        if expected_zero and values["zero_tensors"] != values["tensors"]:
            raise RuntimeError(
                f"zero-initialized first-step {family} gradients were nonzero"
            )
        if not expected_zero and values["zero_tensors"]:
            raise RuntimeError(f"first-step {family} has zero gradient tensors")


def validate_second_gradients(
    report: dict[str, Any], *, require_complete: bool
) -> None:
    if (
        (require_complete and report["missing"])
        or report["nonfinite"]
        or report["frozen_with_grad"]
    ):
        raise RuntimeError("second backward violated gradient ownership or finiteness")
    if report["packed_with_grad"]:
        raise RuntimeError("packed base parameters received gradients")
    zero_families = {
        family: values["zero_tensors"]
        for family, values in report["groups"].items()
        if values["zero_tensors"]
    }
    if zero_families:
        raise RuntimeError(
            f"second backward has zero adapter gradients: {zero_families}"
        )


def audit_training_contract(
    model: torch.nn.Module,
    *,
    decoder_type: type[torch.nn.Module],
    expected_layers: int,
) -> dict[str, Any]:
    """Verify cache/router controls and per-layer non-reentrant checkpointing."""

    get_base_model = getattr(model, "get_base_model", None)
    target = get_base_model() if callable(get_base_model) else model
    config = getattr(target, "config", None)
    if config is None:
        raise RuntimeError("The audited model has no configuration.")
    configs = [config]
    get_text_config = getattr(config, "get_text_config", None)
    if callable(get_text_config):
        text_config = get_text_config()
        if text_config is not config:
            configs.append(text_config)

    errors = []
    for current in configs:
        if getattr(current, "use_cache", False):
            errors.append("use_cache is enabled")
        if getattr(current, "output_router_logits", False):
            errors.append("output_router_logits is enabled")
        if float(getattr(current, "router_aux_loss_coef", 0.0)) != 0.0:
            errors.append("router_aux_loss_coef is nonzero")

    layers = [module for module in target.modules() if isinstance(module, decoder_type)]
    if len(layers) != expected_layers:
        errors.append(f"expected {expected_layers} decoder layers, found {len(layers)}")
    disabled = [
        index for index, layer in enumerate(layers) if not layer.gradient_checkpointing
    ]
    reentrant = [
        index
        for index, layer in enumerate(layers)
        if getattr(
            getattr(layer, "_gradient_checkpointing_func", None), "keywords", {}
        ).get("use_reentrant")
        is not False
    ]
    if disabled:
        errors.append(f"checkpointing disabled on layers {disabled}")
    if reentrant:
        errors.append(f"non-reentrant policy missing on layers {reentrant}")
    if errors:
        raise RuntimeError("training contract audit failed: " + "; ".join(errors))
    return {
        "decoder_layers": len(layers),
        "checkpointed_layers": len(layers) - len(disabled),
        "use_reentrant": False,
        "use_cache": False,
        "output_router_logits": False,
        "router_aux_loss_coef": 0.0,
    }


def summarize_timeline(timeline: dict[str, Any]) -> dict[str, Any]:
    """Rank phases by wall time and report the highest observed memory peaks."""

    rows = []
    for phase, entries in timeline.items():
        total_seconds = sum(float(entry["seconds"]) for entry in entries)
        rows.append(
            {
                "phase": phase,
                "calls": len(entries),
                "total_seconds": total_seconds,
                "mean_seconds": total_seconds / len(entries),
                "peak_allocated_bytes": max(
                    int(entry["memory"]["peak_allocated_bytes"]) for entry in entries
                ),
                "peak_reserved_bytes": max(
                    int(entry["memory"]["peak_reserved_bytes"]) for entry in entries
                ),
                "peak_process_rss_bytes": max(
                    int(entry["process_after"].get("process_rss_bytes", 0))
                    for entry in entries
                ),
            }
        )
    ranked = sorted(rows, key=lambda row: row["total_seconds"], reverse=True)
    return {
        "phases_by_total_time": ranked,
        "total_measured_seconds": sum(float(row["total_seconds"]) for row in rows),
        "peak_allocated_bytes": max(
            (row["peak_allocated_bytes"] for row in rows), default=0
        ),
        "peak_reserved_bytes": max(
            (row["peak_reserved_bytes"] for row in rows), default=0
        ),
        "peak_process_rss_bytes": max(
            (row["peak_process_rss_bytes"] for row in rows), default=0
        ),
    }
