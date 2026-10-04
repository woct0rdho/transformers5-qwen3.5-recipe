#!/usr/bin/env python3
"""DeepSeek V4 full-step correctness, memory, gradient, and profiling audit."""

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import gc
import json
import re
from pathlib import Path
from typing import Any

import bitsandbytes as bnb
import torch
from datasets import Dataset, load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.integrations.gguf.gguf_quantized_parameter import (
    GgufQuantizedParameter,
)
from transformers.integrations.gguf.modules import GgufGroupedLinear
from transformers.integrations.gguf.moe import DeepseekV4GgufExperts
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
    require_complete_deepseek_v4_grouped_mmq,
)
from deepseek_v4_moe_lora import DeepseekV4GgufMoeLora, register_deepseek_v4_moe_lora
from deepseek_v4_profiler import profile_warmed_training_update
from deepseek_v4_routing import DeepseekV4RouteCollector
from fast_moe_ranking import configure_fast_moe_ranking
from training_audit import (
    audit_optimizer,
    audit_training_contract,
    clean_loading_info,
    complete_training_update,
    memory_snapshot,
    representative_packed_state,
    run_phase,
    summarize_b_updates,
    summarize_gradients,
    summarize_timeline,
    validate_first_gradients,
    validate_loss_output,
    validate_second_gradients,
)

EXPECTED_STATE_TENSORS = 1328
EXPECTED_PACKED_PARAMETERS = 474
EXPECTED_PACKED_BYTES = 84_512_276_480
EXPECTED_GROUPED_LINEARS = 43
EXPECTED_EXPERT_MODULES = 43
EXPECTED_INTEGER_BUFFERS = 3

# llama.cpp creates the vision-token routing bias with `TENSOR_NOT_REQUIRED` and only substitutes it
# for `exp_probs_b` when the batch carries mtmd image embeddings (`src/models/deepseek4.cpp`), so a
# text-only run legitimately leaves it unused. The `-Vision-Exp` checkpoint carries one per layer.
VISION_ONLY_KEY = re.compile(r"^model\.layers\.\d+\.exp_probs_b_vl\.bias$")


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


def audit_loaded_model(
    model: torch.nn.Module, loading_info: dict[str, Any]
) -> dict[str, Any]:
    packed = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if isinstance(parameter, GgufQuantizedParameter)
    ]
    grouped = sum(isinstance(module, GgufGroupedLinear) for module in model.modules())
    experts = sum(
        isinstance(module, DeepseekV4GgufExperts) for module in model.modules()
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
    for key in ("missing_keys", "mismatched_keys", "error_msgs"):
        if loading_info.get(key):
            errors.append(f"{key}={loading_info[key]}")
    ignored_vision_keys = sorted(
        key
        for key in loading_info.get("unexpected_keys", ())
        if VISION_ONLY_KEY.match(key)
    )
    unexpected_keys = sorted(
        key
        for key in loading_info.get("unexpected_keys", ())
        if not VISION_ONLY_KEY.match(key)
    )
    if unexpected_keys:
        errors.append(f"unexpected_keys={unexpected_keys}")
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
    assert isinstance(model, PreTrainedModel)
    get_memory_footprint = model.get_memory_footprint
    return observed | {
        "loading_info": cleaned,
        "ignored_vision_keys": len(ignored_vision_keys),
        "model_footprint_bytes": get_memory_footprint(),
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
    if not isinstance(dataset, Dataset):
        raise TypeError(f"expected a Dataset at {dataset_dir}, got DatasetDict")
    if row_start < 0 or row_start + batch_size > len(dataset):
        raise IndexError(
            f"requested rows [{row_start},{row_start + batch_size}) outside dataset of {len(dataset)} rows"
        )
    selected = dataset.select(range(row_start, row_start + batch_size))
    rows = [selected[index] for index in range(batch_size)]
    input_ids = torch.tensor(
        [row["input_ids"][:sequence_length] for row in rows],
        dtype=torch.int64,
        device="cuda:0",
    )
    valid_lengths = torch.tensor(
        [min(int(row["num_tokens"]), sequence_length) for row in rows],
        device=input_ids.device,
    )
    positions = torch.arange(sequence_length, device=input_ids.device).unsqueeze(0)
    valid_tokens = positions < valid_lengths.unsqueeze(1)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.masked_fill(~valid_tokens, -100)
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
        report["performance_summary"] = summarize_timeline(report["timeline"])
        args.report_output.write_text(json.dumps(report, indent=2) + "\n")

    persist()
    model = None
    collector = DeepseekV4RouteCollector()
    torch.manual_seed(args.seed)

    def load_model():
        return AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            gguf_file=args.gguf_file,
            gguf_mmap_policy="pread",
            local_files_only=True,
            dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
            attn_implementation=None,
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
    require_complete_deepseek_v4_grouped_mmq(report["grouped_mmq"])
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
        register_deepseek_v4_moe_lora(
            lora_config, model, expert_prior="deepseek-learned"
        )
        wrapped = get_peft_model(model, lora_config, autocast_adapter_dtype=False)
        apply_deepseek_v4_liger_loss(wrapped)
        return wrapped

    model = run_phase(report["timeline"], "adapter_injection", inject_adapters)
    report["injection_audit"] = audit_deepseek_v4_injection(
        model,
        expert_wrapper_type=DeepseekV4GgufMoeLora,
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
    report["checkpointing"] = audit_training_contract(
        model,
        decoder_type=DeepseekV4DecoderLayer,
        expected_layers=43,
    ) | {"policy": "per_decoder_layer"}
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
            [parameter for parameter in model.parameters() if parameter.requires_grad],
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
    validate_loss_output(output, "first forward")
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
            [parameter for parameter in model.parameters() if parameter.requires_grad],
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
    validate_loss_output(output, "second forward")
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
        if isinstance(module, GgufGroupedLinear) and module.weight.grad is not None
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
        return complete_training_update(
            model,
            optimizer,
            batch,
            label=label,
            max_grad_norm=args.max_grad_norm,
        )

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


if __name__ == "__main__":
    main()
