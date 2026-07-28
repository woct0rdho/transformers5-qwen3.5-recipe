import re
from types import MethodType
from typing import Any

import torch
from peft import LoraConfig
from torch_ggml_ops import fixed_grouped_mmq
from transformers.integrations.gguf import GGUFGroupedLinear, GGUFLinear
from transformers.integrations.gguf_dequant import GGUFQuantizedTensor

from fast_lora import FastGGUFLoraLinear, FastLoraLinear

ORDINARY_TARGET_MODULES = frozenset(
    {
        "q_a_proj",
        "q_b_proj",
        "kv_proj",
        "o_b_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
)
DEEPSEEK_V4_TARGET_MODULES_PATTERN = (
    r"^(?!.*\.self_attn\.compressor\.indexer\.)(?:.*\.)?"
    r"(?:q_a_proj|q_b_proj|kv_proj|o_b_proj|gate_proj|up_proj|down_proj|experts)$"
)
EXPECTED_ORDINARY_WRAPPERS = 383
EXPECTED_EXPERT_WRAPPERS = 43
EXPECTED_RANK4_PARAMETERS = 645_609_472

_FORBIDDEN_TRAINABLE_PATHS = (
    ".o_a_proj.",
    ".mlp.gate.",
    ".attn_hc.",
    ".ffn_hc.",
    ".hc_head.",
    ".input_layernorm.",
    ".post_attention_layernorm.",
    ".q_a_norm.",
    ".q_b_norm.",
    ".kv_norm.",
    ".sinks",
    ".position_bias",
    ".self_attn.compressor.indexer.",
    "embed_tokens",
    "lm_head",
)
_ADAPTER_PARAMETER = re.compile(r"\.lora_[AB](?:_down)?\.")


class DeepseekV4LoraLinear(FastLoraLinear):
    """DeepSeek-owned wrapper for ordinary floating linear modules."""


class DeepseekV4GGUFLoraLinear(FastGGUFLoraLinear):
    """DeepSeek-owned wrapper with capability-based packed base dispatch."""


class _RejectedDeepseekV4GroupedLora(DeepseekV4GGUFLoraLinear):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise RuntimeError(
            "DeepSeek V4 grouped o_a_proj LoRA is unsupported; keep GGUFGroupedLinear frozen."
        )


def register_deepseek_v4_lora(lora_config: LoraConfig) -> LoraConfig:
    """Register DeepSeek ordinary wrappers without changing installed PEFT."""

    register = getattr(lora_config, "_register_custom_module", None)
    if register is None:
        raise RuntimeError(
            "This PEFT version has no LoraConfig._register_custom_module API."
        )
    if lora_config.target_modules != DEEPSEEK_V4_TARGET_MODULES_PATTERN:
        raise ValueError(
            "DeepSeek V4 target_modules must use the exact indexer-excluding pattern "
            f"{DEEPSEEK_V4_TARGET_MODULES_PATTERN!r}."
        )
    register(
        {
            GGUFGroupedLinear: _RejectedDeepseekV4GroupedLora,
            GGUFLinear: DeepseekV4GGUFLoraLinear,
            torch.nn.Linear: DeepseekV4LoraLinear,
        }
    )
    return lora_config


def _deepseek_v4_fixed_grouped_mmq_forward(
    self: GGUFGroupedLinear, input: torch.Tensor
) -> torch.Tensor:
    """Run DeepSeek's frozen eight-group Q8_0 output-A projection natively.

    ``GGUFGroupedLinear`` exposes a public ``[... , 8, 4096] -> [..., 8,
    1024]`` projection while its packed parameter is a flattened physical
    ``[8192, 4352]`` payload. Flattening it through ``GGUFLinear`` would lose
    the group boundary and the stock Transformers autograd path materializes
    the whole logical matrix. The fixed grouped operator owns that layout.
    """

    original_input_dtype = input.dtype
    if self.compute_dtype != torch.bfloat16:
        raise RuntimeError("DeepSeek fixed grouped MMQ requires BF16 compute_dtype.")
    if self.input_permutation is not None or self.output_permutation is not None:
        raise RuntimeError(
            "DeepSeek fixed grouped MMQ does not support layout permutations."
        )

    compute_input = input.to(self.compute_dtype)
    if not compute_input.is_contiguous() or compute_input.storage_offset() != 0:
        compute_input = compute_input.contiguous()
    group_out_features = self.out_features // self.n_groups
    payload = self.weight.as_subclass(torch.Tensor)
    expected_payload_rows = self.n_groups * group_out_features
    if payload.ndim != 2 or payload.shape[0] != expected_payload_rows:
        raise RuntimeError(
            "DeepSeek fixed grouped MMQ expects a flattened grouped payload with "
            f"{expected_payload_rows} rows, got {tuple(payload.shape)}."
        )
    packed_groups = payload.reshape(self.n_groups, group_out_features, -1)
    output = fixed_grouped_mmq(compute_input, packed_groups)
    return output.to(original_input_dtype)


def configure_deepseek_v4_grouped_mmq(model: torch.nn.Module) -> dict[str, Any]:
    """Install the native fixed-grouped Q8_0 path on one model instance.

    The modules deliberately remain ordinary frozen ``GGUFGroupedLinear``
    instances: no PEFT wrapper or grouped ``o_a_proj`` adapter is created.
    This hardcoded DeepSeek integration fails closed instead of retaining the
    logical grouped fallback when the checkpoint contract does not match.
    """

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    paths: list[str] = []
    for name, module in base.named_modules():
        if not isinstance(module, GGUFGroupedLinear):
            continue
        if not isinstance(module.weight, GGUFQuantizedTensor):
            raise RuntimeError(
                f"DeepSeek grouped projection {name!r} is not GGUF-quantized."
            )
        supported = (
            int(module.weight.quant_type) == 8
            and module.n_groups == 8
            and module.in_features == 4096
            and module.out_features % 8 == 0
            and module.out_features // 8 == 1024
        )
        if not supported:
            raise RuntimeError(
                "DeepSeek grouped projection does not match the fixed eight-group "
                f"Q8_0 4096->1024 contract: {name!r}."
            )
        if not getattr(module, "_deepseek_v4_grouped_mmq_enabled", False):
            module.forward = MethodType(_deepseek_v4_fixed_grouped_mmq_forward, module)
            module._deepseek_v4_grouped_mmq_enabled = True
        paths.append(name)
    return {"enabled": len(paths), "paths": sorted(paths)}


def normalize_peft_path(path: str) -> str:
    """Return the pre-PEFT model path used by the adapter contract."""

    for prefix in ("base_model.model.", "model."):
        if path.startswith(prefix):
            path = path.removeprefix(prefix)
    if path.startswith("model.layers."):
        return path
    if path.startswith("layers."):
        return f"model.{path}"
    return path


def audit_deepseek_v4_injection(
    model: torch.nn.Module,
    *,
    expert_wrapper_type: type[torch.nn.Module],
    rank: int,
) -> dict[str, Any]:
    """Validate the complete intended adapter surface before training."""

    ordinary_types = (DeepseekV4LoraLinear, DeepseekV4GGUFLoraLinear)
    ordinary_paths = [
        normalize_peft_path(name)
        for name, module in model.named_modules()
        if isinstance(module, ordinary_types)
        and not isinstance(module, expert_wrapper_type)
    ]
    expert_paths = [
        normalize_peft_path(name)
        for name, module in model.named_modules()
        if isinstance(module, expert_wrapper_type)
    ]
    grouped_paths = [
        normalize_peft_path(name)
        for name, module in model.named_modules()
        if isinstance(module, GGUFGroupedLinear)
    ]
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    trainable_parameters = sum(parameter.numel() for _, parameter in trainable)
    expected_parameters = EXPECTED_RANK4_PARAMETERS * rank // 4

    errors: list[str] = []
    if len(ordinary_paths) != EXPECTED_ORDINARY_WRAPPERS:
        errors.append(
            f"expected {EXPECTED_ORDINARY_WRAPPERS} ordinary wrappers, found {len(ordinary_paths)}"
        )
    if len(expert_paths) != EXPECTED_EXPERT_WRAPPERS:
        errors.append(
            f"expected {EXPECTED_EXPERT_WRAPPERS} expert wrappers, found {len(expert_paths)}"
        )
    if len(grouped_paths) != EXPECTED_EXPERT_WRAPPERS:
        errors.append(
            f"expected 43 frozen grouped o_a_proj modules, found {len(grouped_paths)}"
        )
    if trainable_parameters != expected_parameters:
        errors.append(
            f"expected {expected_parameters} rank-{rank} adapter parameters, found {trainable_parameters}"
        )

    invalid_trainable = [
        name for name, _ in trainable if _ADAPTER_PARAMETER.search(name) is None
    ]
    forbidden_trainable = [
        name
        for name, _ in trainable
        if any(fragment in name for fragment in _FORBIDDEN_TRAINABLE_PATHS)
    ]
    non_bf16 = [
        name for name, parameter in trainable if parameter.dtype != torch.bfloat16
    ]
    non_cuda = [
        name for name, parameter in trainable if parameter.device.type != "cuda"
    ]
    grouped_trainable = [
        path
        for path, module in model.named_modules()
        if isinstance(module, GGUFGroupedLinear) and module.weight.requires_grad
    ]
    packed_trainable = [
        name
        for name, parameter in model.named_parameters()
        if isinstance(parameter, GGUFQuantizedTensor) and parameter.requires_grad
    ]
    if invalid_trainable:
        errors.append(f"non-adapter trainable tensors: {invalid_trainable[:8]}")
    if forbidden_trainable:
        errors.append(f"forbidden trainable paths: {forbidden_trainable[:8]}")
    if non_bf16:
        errors.append(f"non-BF16 adapter tensors: {non_bf16[:8]}")
    if non_cuda:
        errors.append(f"adapter tensors outside cuda:0: {non_cuda[:8]}")
    if grouped_trainable:
        errors.append(f"trainable grouped o_a_proj weights: {grouped_trainable[:8]}")
    if packed_trainable:
        errors.append(f"trainable packed payloads: {packed_trainable[:8]}")
    if errors:
        raise RuntimeError(
            "DeepSeek V4 adapter injection audit failed: " + "; ".join(errors)
        )

    return {
        "ordinary_wrappers": len(ordinary_paths),
        "expert_wrappers": len(expert_paths),
        "wrapped_modules": len(ordinary_paths) + len(expert_paths),
        "trainable_tensors": len(trainable),
        "trainable_parameters": trainable_parameters,
        "trainable_bytes": sum(
            parameter.numel() * parameter.element_size() for _, parameter in trainable
        ),
        "adapter_dtypes": sorted({str(parameter.dtype) for _, parameter in trainable}),
        "ordinary_target_paths": sorted(ordinary_paths),
        "expert_target_paths": sorted(expert_paths),
        "grouped_output_paths": sorted(grouped_paths),
        "packed_parameters": sum(
            isinstance(parameter, GGUFQuantizedTensor)
            for parameter in model.parameters()
        ),
    }
