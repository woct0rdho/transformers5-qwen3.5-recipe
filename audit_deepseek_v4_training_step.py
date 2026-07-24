#!/usr/bin/env python3
"""DeepSeek V4 full-step correctness, memory, gradient, and profiling audit."""

import argparse
import gc
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import bitsandbytes as bnb
import torch
from datasets import Dataset, load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM
from transformers.integrations.gguf import DeepseekV4GGUFExperts, GGUFGroupedLinear
from transformers.integrations.gguf_dequant import GGUFQuantizedTensor
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4DecoderLayer

from deepseek_v4_attention import (
    configure_deepseek_v4_attention,
    require_complete_deepseek_v4_attention,
)
from deepseek_v4_liger_loss import apply_deepseek_v4_liger_loss
from deepseek_v4_liger_mhc import (
    configure_deepseek_v4_liger_mhc,
    require_complete_deepseek_v4_liger_mhc,
)
from deepseek_v4_liger_rmsnorm import (
    configure_deepseek_v4_liger_rmsnorm,
    require_complete_deepseek_v4_liger_rmsnorm,
)
from deepseek_v4_lora import (
    DEEPSEEK_V4_TARGET_MODULES_PATTERN,
    audit_deepseek_v4_injection,
    configure_deepseek_v4_grouped_mmq,
    register_deepseek_v4_lora,
)
from deepseek_v4_moe_lora import (
    DeepseekV4GGUFMoeLora,
    register_deepseek_v4_moe_lora,
)
from deepseek_v4_profiler import profile_warmed_training_update
from deepseek_v4_routing import DeepseekV4RouteCollector
from fast_moe_ranking import configure_fast_moe_ranking

EXPECTED_STATE_TENSORS = 1328
EXPECTED_PACKED_PARAMETERS = 474
EXPECTED_PACKED_BYTES = 84_512_276_480
EXPECTED_GROUPED_LINEARS = 43
EXPECTED_EXPERT_MODULES = 43
EXPECTED_INTEGER_BUFFERS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir", type=Path, default=Path("~/models/ds4").expanduser()
    )
    parser.add_argument("--gguf-file", default="DeepSeek-V4-Flash-IQ2XXS.gguf")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data_tokenized_ds4",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("out_deepseek_v4"))
    parser.add_argument(
        "--report-output", type=Path, default=Path("deepseek_v4_training_report.json")
    )
    parser.add_argument("--profile-output", type=Path)
    parser.add_argument("--batch-size", type=int, choices=(1, 4, 16), default=1)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--row-start", type=int, default=0)
    parser.add_argument("--seed", type=int, default=19_260_817)
    parser.add_argument("--save-adapter", action="store_true")
    return parser.parse_args()


def process_memory() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/self/smaps_rollup").read_text().splitlines():
            key, _, rest = line.partition(":")
            if key in {"Rss", "Pss_File", "Private_Clean", "Private_Dirty", "Swap"}:
                values[f"process_{key.lower()}_bytes"] = int(rest.split()[0]) * 1024
    except OSError:
        pass
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            if key in {"SwapTotal", "SwapFree"}:
                values[f"system_{key.lower()}_bytes"] = int(rest.split()[0]) * 1024
    except OSError:
        pass
    return values


def accelerator_memory() -> dict[str, int]:
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
    function,
    *,
    reset_peak: bool = True,
):
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
    print(name, json.dumps(entry), flush=True)
    return result


def clean_loading_info(loading_info: dict[str, Any]) -> dict[str, Any]:
    return {
        key: [str(item) for item in value] if isinstance(value, list) else str(value)
        for key, value in loading_info.items()
    }


def audit_loaded_model(
    model: torch.nn.Module, loading_info: dict[str, Any]
) -> dict[str, Any]:
    packed = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if isinstance(parameter, GGUFQuantizedTensor)
    ]
    grouped = sum(isinstance(module, GGUFGroupedLinear) for module in model.modules())
    experts = sum(
        isinstance(module, DeepseekV4GGUFExperts) for module in model.modules()
    )
    integer_buffers = [
        (name, buffer)
        for name, buffer in model.named_buffers()
        if not buffer.is_floating_point()
    ]
    packed_bytes = sum(
        parameter.numel() * parameter.element_size() for _, parameter in packed
    )
    cleaned = clean_loading_info(loading_info)
    errors = []
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        if loading_info.get(key):
            errors.append(f"{key}={loading_info[key]}")
    observed = {
        "state_tensors": len(model.state_dict()),
        "packed_parameters": len(packed),
        "packed_bytes": packed_bytes,
        "grouped_linears": grouped,
        "expert_modules": experts,
        "integer_buffers": len(integer_buffers),
    }
    expected = {
        "state_tensors": EXPECTED_STATE_TENSORS,
        "packed_parameters": EXPECTED_PACKED_PARAMETERS,
        "packed_bytes": EXPECTED_PACKED_BYTES,
        "grouped_linears": EXPECTED_GROUPED_LINEARS,
        "expert_modules": EXPECTED_EXPERT_MODULES,
        "integer_buffers": EXPECTED_INTEGER_BUFFERS,
    }
    for key, value in expected.items():
        if observed[key] != value:
            errors.append(f"{key}: expected {value}, found {observed[key]}")
    non_cuda = [
        name
        for name, parameter in model.named_parameters()
        if parameter.device.type != "cuda"
    ]
    non_cuda += [
        name for name, buffer in model.named_buffers() if buffer.device.type != "cuda"
    ]
    if non_cuda:
        errors.append(f"model tensors outside cuda:0: {non_cuda[:8]}")
    if errors:
        raise RuntimeError("DeepSeek V4 load audit failed: " + "; ".join(errors))
    return observed | {
        "loading_info": cleaned,
        "model_footprint_bytes": model.get_memory_footprint(),
        "integer_buffer_paths": [name for name, _ in integer_buffers],
    }


def load_fixed_batch(
    dataset_dir: Path,
    *,
    row_start: int,
    batch_size: int,
    sequence_length: int,
) -> tuple[Dataset, dict[str, torch.Tensor], dict[str, Any]]:
    if sequence_length <= 1 or sequence_length > 2048:
        raise ValueError(
            "The fixed DeepSeek dataset supports sequence lengths in [2,2048]."
        )
    dataset = load_from_disk(str(dataset_dir))
    if row_start < 0 or row_start + batch_size > len(dataset):
        raise IndexError(
            f"requested rows [{row_start},{row_start + batch_size}) outside dataset of {len(dataset)} rows"
        )
    selected = dataset.select(range(row_start, row_start + batch_size))
    rows = [selected[index] for index in range(batch_size)]
    input_ids = torch.tensor(
        [row["input_ids"][:sequence_length] for row in rows],
        dtype=torch.long,
        device="cuda:0",
    )
    valid_lengths = torch.tensor(
        [min(int(row["num_tokens"]), sequence_length) for row in rows],
        device=input_ids.device,
    )
    positions = torch.arange(sequence_length, device=input_ids.device).unsqueeze(0)
    attention_mask = (positions < valid_lengths.unsqueeze(1)).to(torch.long)
    labels = input_ids.clone()
    labels.masked_fill_(attention_mask == 0, -100)
    return (
        selected,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        },
        {
            "dataset_rows": len(dataset),
            "selected_rows": list(range(row_start, row_start + batch_size)),
            "num_tokens": valid_lengths.cpu().tolist(),
            "input_shape": list(input_ids.shape),
        },
    )


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

    group_report: dict[str, Any] = {}
    all_sum_squares = torch.zeros((), device="cuda", dtype=torch.float64)
    all_maximum = torch.zeros((), device="cuda", dtype=torch.float32)
    nonfinite_names: list[str] = []
    zero_names: list[str] = []
    for family, gradients in sorted(groups.items()):
        finite = torch.stack([torch.isfinite(gradient).all() for gradient in gradients])
        nonzero = torch.stack([torch.count_nonzero(gradient) for gradient in gradients])
        sum_squares = sum(
            gradient.float().square().sum().double() for gradient in gradients
        )
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
            if isinstance(parameter, GGUFQuantizedTensor) and parameter.grad is not None
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
    selected: dict[int, tuple[str, GGUFQuantizedTensor]] = {}
    for name, parameter in model.named_parameters():
        if isinstance(parameter, GGUFQuantizedTensor):
            selected.setdefault(int(parameter.quant_type), (name, parameter))
    result = {}
    for quant_type in (8, 10, 16):
        name, parameter = selected[quant_type]
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
    """Audit the first update against the known all-zero LoRA-B initialization."""

    names = []
    nonzero_counts = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and ".lora_B" in name:
            names.append(name)
            nonzero_counts.append(torch.count_nonzero(parameter.detach()))
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


def main() -> None:
    args = parse_args()
    if args.max_steps < 1:
        raise ValueError("--max-steps must be positive")

    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "status": "running",
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "environment": {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "device": torch.cuda.get_device_name(0),
            "pid": os.getpid(),
        },
        "timeline": {},
        "memory_before": memory_snapshot(),
    }

    def persist() -> None:
        args.report_output.write_text(json.dumps(report, indent=2) + "\n")

    persist()
    model = None
    collector = DeepseekV4RouteCollector()
    try:
        torch.manual_seed(args.seed)

        def load_model():
            return AutoModelForCausalLM.from_pretrained(
                args.model_dir,
                gguf_file=args.gguf_file,
                gguf_mmap_policy="release",
                local_files_only=True,
                dtype=torch.bfloat16,
                device_map={"": "cuda:0"},
                attn_implementation="eager",
                output_loading_info=True,
            )

        model, loading_info = run_phase(report["timeline"], "load", load_model)
        model.config.use_cache = False
        model.config.output_router_logits = False
        model.config.router_aux_loss_coef = 0.0
        report["attention"] = configure_deepseek_v4_attention(model)
        require_complete_deepseek_v4_attention(report["attention"])
        report["router"] = configure_fast_moe_ranking(model)
        report["grouped_mmq"] = configure_deepseek_v4_grouped_mmq(model)
        if report["grouped_mmq"]["enabled"] != 43:
            raise RuntimeError(
                "expected 43 native DeepSeek grouped output-A projections, found "
                f"{report['grouped_mmq']['enabled']}"
            )
        report["liger_rmsnorm"] = configure_deepseek_v4_liger_rmsnorm(model)
        require_complete_deepseek_v4_liger_rmsnorm(report["liger_rmsnorm"])
        report["liger_mhc"] = configure_deepseek_v4_liger_mhc(model)
        require_complete_deepseek_v4_liger_mhc(report["liger_mhc"])
        report["load_audit"] = audit_loaded_model(model, loading_info)
        report["memory_after_load"] = memory_snapshot()
        persist()

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=DEEPSEEK_V4_TARGET_MODULES_PATTERN,
            r=args.rank,
            lora_alpha=args.alpha,
            lora_dropout=0.0,
            bias="none",
            use_rslora=False,
            init_lora_weights=True,
        )

        def inject_adapters():
            register_deepseek_v4_lora(lora_config)
            register_deepseek_v4_moe_lora(lora_config, model)
            wrapped = get_peft_model(model, lora_config, autocast_adapter_dtype=False)
            apply_deepseek_v4_liger_loss(wrapped)
            return wrapped

        model = run_phase(report["timeline"], "adapter_injection", inject_adapters)
        report["injection_audit"] = audit_deepseek_v4_injection(
            model,
            expert_wrapper_type=DeepseekV4GGUFMoeLora,
            rank=args.rank,
        )
        initial_b = [
            torch.count_nonzero(parameter.detach())
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and ".lora_B" in name
        ]
        initial_b_nonzero = torch.stack(initial_b).cpu()
        report["injection_audit"]["initial_b_tensors"] = len(initial_b)
        report["injection_audit"]["initial_b_nonzero_tensors"] = int(
            torch.count_nonzero(initial_b_nonzero)
        )
        if bool(torch.any(initial_b_nonzero).item()):
            raise RuntimeError(
                "LoRA-B factors must be zero-initialized before the first update"
            )
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
        model.train()
        checkpointed = [
            module
            for module in model.modules()
            if isinstance(module, DeepseekV4DecoderLayer)
            and getattr(module, "gradient_checkpointing", False)
        ]
        if len(checkpointed) != 43:
            raise RuntimeError(
                f"expected 43 checkpointed decoder layers, found {len(checkpointed)}"
            )
        report["checkpointing"] = {
            "policy": "per_decoder_layer",
            "use_reentrant": False,
            "checkpointed_layers": len(checkpointed),
        }
        report["memory_after_adapters"] = memory_snapshot()
        persist()

        selected_dataset, batch, data_report = load_fixed_batch(
            args.dataset_dir,
            row_start=args.row_start,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
        )
        report["data"] = data_report
        optimizer = run_phase(
            report["timeline"],
            "optimizer_create",
            lambda: bnb.optim.AdamW8bit(
                [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ],
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            ),
        )
        report["optimizer"] = audit_optimizer(optimizer, model)
        report["memory_after_optimizer_create"] = memory_snapshot()
        packed_before = representative_packed_state(model)
        report["packed_before"] = packed_before
        report["route_validation_policy"] = {
            "expert_identity": "implementation_defined_near_ties",
            "correctness_metric": "sorted_selected_routing_weights",
            "sample_rows_per_summary": 256,
        }
        collector.install(model)

        optimizer.zero_grad(set_to_none=True)
        collector.clear()
        output = run_phase(
            report["timeline"],
            "first_forward",
            lambda: model(**batch, use_cache=False),
        )
        if output.logits is not None or output.aux_loss is not None:
            raise RuntimeError(
                "scoped training loss must return logits=None and aux_loss=None"
            )
        if output.loss is None or not bool(torch.isfinite(output.loss).item()):
            raise RuntimeError("first loss is missing or nonfinite")
        first_loss = float(output.loss.detach())
        run_phase(report["timeline"], "first_backward", output.loss.backward)
        first_gradients = summarize_gradients(model, "first_backward")
        report["first_backward"] = {
            "loss": first_loss,
            "gradients": first_gradients,
            "routes": collector.summaries(),
        }
        persist()
        require_complete = args.sequence_length == 2048
        validate_first_gradients(first_gradients, require_complete=require_complete)
        del output

        clip_norm = run_phase(
            report["timeline"],
            "gradient_clip",
            lambda: torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ],
                args.max_grad_norm,
            ),
        )
        if not bool(torch.isfinite(clip_norm).item()):
            raise RuntimeError("gradient clipping returned a nonfinite norm")
        report["clip_norm_before"] = float(clip_norm)
        run_phase(report["timeline"], "optimizer_step", optimizer.step)
        updates = summarize_b_updates(model)
        expected_unchanged = {
            name for name in first_gradients["missing"] if ".lora_B" in name
        }
        if set(updates["unchanged"]) != expected_unchanged:
            raise RuntimeError(
                "LoRA-B update coverage differs from the first backward's missing branches: "
                f"unchanged={updates['unchanged'][:8]}, expected={sorted(expected_unchanged)[:8]}"
            )
        if require_complete and updates["changed_tensors"] != updates["tensors"]:
            raise RuntimeError(
                f"some LoRA-B tensors did not update: {updates['unchanged'][:8]}"
            )
        report["first_update"] = updates
        optimizer.zero_grad(set_to_none=True)
        gc.collect()

        collector.clear()
        output = run_phase(
            report["timeline"],
            "second_forward",
            lambda: model(**batch, use_cache=False),
        )
        if output.logits is not None or output.aux_loss is not None:
            raise RuntimeError(
                "second scoped loss must return logits=None and aux_loss=None"
            )
        if output.loss is None or not bool(torch.isfinite(output.loss).item()):
            raise RuntimeError("second loss is missing or nonfinite")
        second_loss = float(output.loss.detach())
        run_phase(report["timeline"], "second_backward", output.loss.backward)
        second_gradients = summarize_gradients(model, "second_backward")
        report["second_backward"] = {
            "loss": second_loss,
            "gradients": second_gradients,
            "routes": collector.summaries(),
        }
        persist()
        validate_second_gradients(second_gradients, require_complete=require_complete)
        del output

        packed_after = representative_packed_state(model)
        report["packed_after"] = packed_after
        if packed_before != packed_after:
            raise RuntimeError(
                "representative packed payload identity, version, or checksum changed"
            )
        grouped_gradients = [
            name
            for name, module in model.named_modules()
            if isinstance(module, GGUFGroupedLinear) and module.weight.grad is not None
        ]
        if grouped_gradients:
            raise RuntimeError(
                f"frozen grouped o_a_proj weights received gradients: {grouped_gradients[:8]}"
            )
        report["grouped_output_gradients"] = grouped_gradients
        report["memory_after_gate"] = memory_snapshot()
        persist()

        optimizer.zero_grad(set_to_none=True)
        collector.remove()

        def complete_update(label: str) -> dict[str, Any]:
            phase_times = {}
            optimizer.zero_grad(set_to_none=True)
            started = time.perf_counter()
            with torch.autograd.profiler.record_function("training_phase/forward"):
                step_output = model(**batch, use_cache=False)
            torch.cuda.synchronize()
            phase_times["forward_seconds"] = time.perf_counter() - started
            if step_output.logits is not None or not bool(
                torch.isfinite(step_output.loss).item()
            ):
                raise RuntimeError(f"{label} produced invalid scoped loss output")
            backward_started = time.perf_counter()
            with torch.autograd.profiler.record_function("training_phase/backward"):
                step_output.loss.backward()
            torch.cuda.synchronize()
            phase_times["backward_seconds"] = time.perf_counter() - backward_started
            clip_started = time.perf_counter()
            with torch.autograd.profiler.record_function(
                "training_phase/gradient_clip"
            ):
                norm = torch.nn.utils.clip_grad_norm_(
                    [
                        parameter
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ],
                    args.max_grad_norm,
                )
            torch.cuda.synchronize()
            phase_times["clip_seconds"] = time.perf_counter() - clip_started
            if not bool(torch.isfinite(norm).item()):
                raise RuntimeError(f"{label} clipping norm is nonfinite")
            step_started = time.perf_counter()
            with torch.autograd.profiler.record_function("training_phase/optimizer"):
                optimizer.step()
            torch.cuda.synchronize()
            phase_times["optimizer_seconds"] = time.perf_counter() - step_started
            phase_times["loss"] = float(step_output.loss.detach())
            phase_times["clip_norm"] = float(norm)
            phase_times["memory"] = accelerator_memory()
            del step_output
            return phase_times

        for step in range(1, args.max_steps):
            report.setdefault("extra_steps", []).append(
                complete_update(f"extra_step_{step}")
            )
            persist()

        if args.profile_output is not None:
            report["profile"] = profile_warmed_training_update(
                model,
                complete_update,
                output_path=args.profile_output,
                metadata={
                    "batch_size": args.batch_size,
                    "sequence_length": args.sequence_length,
                    "rank": args.rank,
                    "checkpointing": report["checkpointing"],
                    "route_distributions": report["second_backward"]["routes"],
                },
            )
            persist()

        if args.save_adapter:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            run_phase(
                report["timeline"],
                "save_adapter",
                lambda: model.save_pretrained(args.output_dir),
            )
            report["adapter_output"] = str(args.output_dir)

        del selected_dataset
        report["status"] = "passed"
        report["memory_final"] = memory_snapshot()
        persist()
        print(f"DEEPSEEK V4 GATE PASS: {args.report_output}", flush=True)
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        try:
            report["memory_at_failure"] = memory_snapshot()
        except BaseException as memory_error:
            report["memory_at_failure_error"] = str(memory_error)
        persist()
        raise
    finally:
        collector.remove()


if __name__ == "__main__":
    main()
