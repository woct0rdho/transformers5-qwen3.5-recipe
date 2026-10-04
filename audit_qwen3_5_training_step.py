#!/usr/bin/env python3
"""Qwen3.5-MoE full-step correctness, memory, gradient, and profiling audit."""

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import gc
import json
from pathlib import Path
from typing import Any, cast

import bitsandbytes as bnb
import torch
from datasets import Dataset, load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM
from transformers.integrations.gguf.gguf_quantized_parameter import (
    GgufQuantizedParameter,
)
from transformers.integrations.gguf.modules import GgufLinear
from transformers.integrations.gguf.moe import GgufExperts
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeDecoderLayer,
)

from attention_aiter_tuning import configure_qwen35_flash_attention_2
from fast_lora import FastGgufLoraLinear, FastLoraLinear, register_fast_lora
from fast_moe_lora import FastGgufMoeLora, register_fast_moe_lora
from fast_moe_ranking import configure_fast_moe_ranking
from fla_tuning import configure_qwen35_fla
from gdn_bwd_dhu import install as install_gdn_bwd_dhu
from gdn_bwd_dqkwg import install as install_gdn_bwd_dqkwg
from gdn_tiled_value_heads import (
    configure_tiled_value_heads,
    require_tiled_value_heads,
)
from gdn_wu_recompute import install as install_gdn_wu_recompute
from gguf_dequant_compile import configure_compiled_gguf_dequantize
from gguf_liger_loss import apply_gguf_liger_fused_linear_cross_entropy
from qwen3_5_fused_norms import (
    configure_qwen35_fused_norms,
    require_complete_qwen35_fused_norms,
)
from qwen3_5_profiler import profile_warmed_training_update
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

EXPECTED_STATE_TENSORS = 733
EXPECTED_LOGICAL_PARAMETERS = 34_660_610_688
EXPECTED_PACKED_PARAMETERS = 432
EXPECTED_PACKED_BYTES = 14_216_723_456
EXPECTED_GGUF_LINEARS = 351
EXPECTED_EXPERT_MODULES = 40
EXPECTED_ORDINARY_WRAPPERS = 250
EXPECTED_GATED_DELTA_NET_LAYERS = 30
EXPECTED_NATIVE_ORDINARY_WRAPPERS = 250
EXPECTED_EXPERT_WRAPPERS = 40
_GGUF_EXPERTS_TYPE = cast(type[Any], GgufExperts)
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "down_proj",
    "gate_proj",
    "up_proj",
    "in_proj_qkv",
    "in_proj_z",
    "out_proj",
    "experts",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir", type=Path, default=Path("~/models/qwen3.6").expanduser()
    )
    parser.add_argument("--gguf-file", default="Qwen3.6-35B-A3B-APEX-I-Mini.gguf")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data_tokenized_qwen3.5",
    )
    parser.add_argument(
        "--report-output", type=Path, default=Path("qwen3_5_training_report.json")
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
    return parser.parse_args()


def audit_loaded_model(
    model: torch.nn.Module, loading_info: dict[str, Any]
) -> dict[str, Any]:
    packed = [
        parameter
        for parameter in model.parameters()
        if isinstance(parameter, GgufQuantizedParameter)
    ]
    observed = {
        "state_tensors": len(model.state_dict()),
        "logical_parameters": sum(
            parameter.logical_numel
            if isinstance(parameter, GgufQuantizedParameter)
            else parameter.numel()
            for parameter in model.parameters()
        ),
        "packed_parameters": len(packed),
        "packed_bytes": sum(parameter.numel() for parameter in packed),
        "gguf_linears": sum(
            isinstance(module, GgufLinear) for module in model.modules()
        ),
        "expert_modules": sum(
            isinstance(module, _GGUF_EXPERTS_TYPE) for module in model.modules()
        ),
    }
    expected = {
        "state_tensors": EXPECTED_STATE_TENSORS,
        "logical_parameters": EXPECTED_LOGICAL_PARAMETERS,
        "packed_parameters": EXPECTED_PACKED_PARAMETERS,
        "packed_bytes": EXPECTED_PACKED_BYTES,
        "gguf_linears": EXPECTED_GGUF_LINEARS,
        "expert_modules": EXPECTED_EXPERT_MODULES,
    }
    errors = [
        f"{key}: expected {value}, found {observed[key]}"
        for key, value in expected.items()
        if observed[key] != value
    ]
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        if loading_info.get(key):
            errors.append(f"{key}={loading_info[key]}")
    outside_cuda = [
        name
        for name, tensor in (*model.named_parameters(), *model.named_buffers())
        if tensor.device.type != "cuda"
    ]
    if outside_cuda:
        errors.append(f"model tensors outside cuda:0: {outside_cuda[:8]}")
    if errors:
        raise RuntimeError("Qwen3.5 load audit failed: " + "; ".join(errors))
    return observed | {"loading_info": clean_loading_info(loading_info)}


def audit_adapter_injection(model: torch.nn.Module) -> dict[str, Any]:
    ordinary = [
        module for module in model.modules() if isinstance(module, FastLoraLinear)
    ]
    native = [
        module
        for module in ordinary
        if isinstance(module, FastGgufLoraLinear) and module.uses_packed_mmq()
    ]
    experts = [
        module for module in model.modules() if isinstance(module, FastGgufMoeLora)
    ]
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    errors = []
    counts = {
        "ordinary_wrappers": len(ordinary),
        "native_ordinary_wrappers": len(native),
        "expert_wrappers": len(experts),
        "trainable_tensors": len(trainable),
        "trainable_parameters": sum(parameter.numel() for _, parameter in trainable),
    }
    expected = {
        "ordinary_wrappers": EXPECTED_ORDINARY_WRAPPERS,
        "native_ordinary_wrappers": EXPECTED_NATIVE_ORDINARY_WRAPPERS,
        "expert_wrappers": EXPECTED_EXPERT_WRAPPERS,
    }
    errors.extend(
        f"{key}: expected {value}, found {counts[key]}"
        for key, value in expected.items()
        if counts[key] != value
    )
    invalid = [name for name, _ in trainable if ".lora_" not in name]
    non_bf16 = [
        name for name, parameter in trainable if parameter.dtype != torch.bfloat16
    ]
    non_cuda = [
        name for name, parameter in trainable if parameter.device.type != "cuda"
    ]
    packed_trainable = [
        name
        for name, parameter in model.named_parameters()
        if isinstance(parameter, GgufQuantizedParameter) and parameter.requires_grad
    ]
    if invalid:
        errors.append(f"non-adapter trainable tensors: {invalid[:8]}")
    if non_bf16:
        errors.append(f"non-BF16 adapters: {non_bf16[:8]}")
    if non_cuda:
        errors.append(f"adapters outside cuda:0: {non_cuda[:8]}")
    if packed_trainable:
        errors.append(f"trainable packed parameters: {packed_trainable[:8]}")
    if errors:
        raise RuntimeError("Qwen3.5 adapter audit failed: " + "; ".join(errors))
    return counts | {"adapter_dtypes": ["torch.bfloat16"], "device": "cuda:0"}


def load_fixed_batch(
    dataset_dir: Path,
    *,
    row_start: int,
    batch_size: int,
    sequence_length: int,
) -> tuple[Dataset, dict[str, torch.Tensor], dict[str, Any]]:
    if sequence_length <= 1 or sequence_length > 2048:
        raise ValueError(
            "The fixed Qwen dataset supports sequence lengths in [2,2048]."
        )
    dataset = load_from_disk(str(dataset_dir))
    if not isinstance(dataset, Dataset):
        raise TypeError(f"expected a Dataset at {dataset_dir}, got DatasetDict")
    if row_start < 0 or row_start + batch_size > len(dataset):
        raise IndexError(
            f"requested rows [{row_start},{row_start + batch_size}) outside "
            f"dataset of {len(dataset)} rows"
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
    attention_mask = (positions < valid_lengths.unsqueeze(1)).long()
    labels = input_ids.masked_fill(attention_mask == 0, -100)
    return (
        selected,
        {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels},
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
    torch.manual_seed(args.seed)
    report["static_configuration"] = {
        "compiled_gguf_dequant": configure_compiled_gguf_dequantize(),
        "tiled_value_heads": configure_tiled_value_heads(),
        "gdn_bwd_dhu": install_gdn_bwd_dhu(),
        "gdn_bwd_dqkwg": install_gdn_bwd_dqkwg(),
        "gdn_wu_recompute": install_gdn_wu_recompute(),
        "flash_attention": configure_qwen35_flash_attention_2(),
        "fla_cache_entries": configure_qwen35_fla(),
    }

    def load_model():
        return AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            gguf_file=args.gguf_file,
            gguf_mmap_policy="pread",
            local_files_only=True,
            dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
            attn_implementation="flash_attention_2",
            output_loading_info=True,
        )

    model, loading_info = run_phase(report["timeline"], "load", load_model)
    model.config.use_cache = False
    model.config.output_router_logits = False
    model.config.router_aux_loss_coef = 0.0
    report["router"] = configure_fast_moe_ranking(model)
    report["fused_norms"] = configure_qwen35_fused_norms(model)
    require_complete_qwen35_fused_norms(report["fused_norms"])
    report["load_audit"] = audit_loaded_model(model, loading_info)
    report["tiled_value_heads"] = require_tiled_value_heads(
        model, expected_gdn_layers=EXPECTED_GATED_DELTA_NET_LAYERS
    )
    report["memory_after_load"] = memory_snapshot()
    persist()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=TARGET_MODULES,
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.0,
        bias="none",
        use_rslora=False,
        init_lora_weights=True,
    )

    def inject_adapters():
        register_fast_lora(lora_config, model)
        register_fast_moe_lora(lora_config, model, expert_prior="qwen-learned")
        wrapped = get_peft_model(model, lora_config, autocast_adapter_dtype=False)
        apply_gguf_liger_fused_linear_cross_entropy(wrapped)
        return wrapped

    model = run_phase(report["timeline"], "adapter_injection", inject_adapters)
    report["injection_audit"] = audit_adapter_injection(model)
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
        raise RuntimeError("LoRA-B factors must be zero-initialized")
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    model.train()
    report["checkpointing"] = audit_training_contract(
        model,
        decoder_type=Qwen3_5MoeDecoderLayer,
        expected_layers=40,
    ) | {"policy": "per_decoder_layer"}
    report["memory_after_adapters"] = memory_snapshot()

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
    persist()

    require_complete = args.sequence_length == 2048
    optimizer.zero_grad(set_to_none=True)
    first_output = run_phase(
        report["timeline"], "first_forward", lambda: model(**batch, use_cache=False)
    )
    validate_loss_output(first_output, "first forward")
    first_loss = float(first_output.loss.detach())
    run_phase(report["timeline"], "first_backward", first_output.loss.backward)
    first_gradients = summarize_gradients(model, "first_backward")
    validate_first_gradients(first_gradients, require_complete=require_complete)
    report["first_backward"] = {
        "loss": first_loss,
        "gradients": first_gradients,
    }
    del first_output

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
    first_update = summarize_b_updates(model)
    expected_unchanged = {
        name for name in first_gradients["missing"] if ".lora_B" in name
    }
    if set(first_update["unchanged"]) != expected_unchanged:
        raise RuntimeError("LoRA-B update coverage differs from gradient coverage")
    if require_complete and first_update["changed_tensors"] != first_update["tensors"]:
        raise RuntimeError(
            f"some LoRA-B tensors did not update: {first_update['unchanged'][:8]}"
        )
    report["first_update"] = first_update
    persist()

    optimizer.zero_grad(set_to_none=True)
    gc.collect()
    second_output = run_phase(
        report["timeline"], "second_forward", lambda: model(**batch, use_cache=False)
    )
    validate_loss_output(second_output, "second forward")
    second_loss = float(second_output.loss.detach())
    run_phase(report["timeline"], "second_backward", second_output.loss.backward)
    second_gradients = summarize_gradients(model, "second_backward")
    validate_second_gradients(second_gradients, require_complete=require_complete)
    report["second_backward"] = {
        "loss": second_loss,
        "gradients": second_gradients,
    }
    del second_output

    packed_after = representative_packed_state(model)
    report["packed_after"] = packed_after
    if packed_before != packed_after:
        raise RuntimeError("representative packed payload identity or checksum changed")
    report["memory_after_gradient_gate"] = memory_snapshot()
    persist()

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
            },
        )

    del selected_dataset
    report["status"] = "passed"
    report["memory_final"] = memory_snapshot()
    persist()
    print(f"QWEN3.5 GATE PASS: {args.report_output}", flush=True)


if __name__ == "__main__":
    main()
