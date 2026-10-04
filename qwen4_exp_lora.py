"""Qwen4-Exp adapter surface: the target pattern, its inventory, and its MMQ wiring.

The LoRA surface follows the Qwen3.5-MoE scripts (the same attention, recurrent, and shared-expert
families, rank 4, BF16 factors) with two Qwen4-Exp-specific decisions:
- The QSA indexer stays frozen. Its selection is a `topk` over block scores, so it receives no
  gradient from the causal-LM loss and an adapter there would be a dead parameter that never
  updates. The target pattern also keeps `index_qk_proj.q_proj` and `index_qk_proj.k_proj` out of
  the ordinary families, which they would otherwise match by their leaf names. The reference stack
  freezes indexers too and trains them with a separate objective.
- The hyper-connection and PLE projections stay frozen: they are the model's routing and injection
  machinery, not the attention or MLP transforms this repository adapts.

The routed experts use the shared grouped-MMQ base from `qwen4_exp_moe_lora.py`. This module also
owns the frozen projections that are executed but not adapted.
"""

import re
from typing import Any

import torch
from peft import LoraConfig
from transformers.integrations.gguf.gguf_quantized_parameter import (
    GgufQuantizedParameter,
)
from transformers.integrations.gguf.modules import GgufLinear

from fast_lora import packed_mmq_linear, register_fast_lora
from module_patching import (
    ModulePatchSpec,
    patch_module_forwards,
    require_complete_inventory,
)
from qwen4_exp_moe_lora import register_qwen4_exp_moe_lora

QWEN4_EXP_TARGET_MODULES_PATTERN = (
    r"^(?!.*\.self_attn\.indexer\.index_qk_proj\.)(?:.*\.)?"
    r"(?:q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj"
    r"|in_proj_qkv|in_proj_z|out_proj|experts)$"
)

EXPECTED_ORDINARY_WRAPPERS = 300
EXPECTED_EXPERT_WRAPPERS = 48
EXPECTED_ADAPTER_TENSORS = 792
EXPECTED_RANK4_PARAMETERS = 699_678_720
EXPECTED_NATIVE_MMQ_WRAPPERS = 300

# Frozen packed projections that are executed but not adapted. The LM head is not one of them:
# training reaches it through the chunked packed loss, and this forward only serves evaluation.
_FROZEN_MMQ_SUFFIXES = (".ple.key_proj",)
_FROZEN_MMQ_MARKER = "_patched_qwen4_exp_frozen_mmq"
EXPECTED_FROZEN_MMQ = 1

_TARGET_RE = re.compile(QWEN4_EXP_TARGET_MODULES_PATTERN)


def is_qwen4_exp_target(key: str) -> bool:
    """Whether a module key is one of the fixed LoRA targets."""

    return _TARGET_RE.fullmatch(key) is not None


def _matches_frozen_mmq(name: str, module: GgufLinear) -> bool:
    """Whether a frozen projection owns the native packed MMQ forward."""

    del module
    return f".{name}".endswith(_FROZEN_MMQ_SUFFIXES)


def _frozen_mmq_forward(self: GgufLinear, input: torch.Tensor) -> torch.Tensor:
    """Run one frozen packed projection natively.

    Its input is a constant of the graph (the n-gram embedding lookup and the frozen value
    projection are both adapter-free and untrainable), so the native path needs the forward only
    and the module's weight never receives a gradient.
    """

    return packed_mmq_linear(self, input)


def _validate_frozen_mmq(name: str, module: GgufLinear) -> None:
    if not isinstance(module.weight, GgufQuantizedParameter):
        raise TypeError(f"Frozen MMQ projection {name!r} is not GGUF-quantized.")
    if module.weight.requires_grad:
        raise RuntimeError(f"Frozen MMQ projection {name!r} is trainable.")
    if module.bias is not None:
        raise RuntimeError(f"Frozen MMQ projection {name!r} carries a bias.")
    if module.input_permutation is not None:
        raise RuntimeError(f"Frozen MMQ projection {name!r} gathers its input.")


_FROZEN_MMQ_SPECS = (
    ModulePatchSpec(
        module_type=GgufLinear,
        forward=_frozen_mmq_forward,
        handled_key="enabled",
        matches=_matches_frozen_mmq,
        validate=_validate_frozen_mmq,
        marker=_FROZEN_MMQ_MARKER,
        freeze_weight=False,
    ),
)


def configure_qwen4_exp_frozen_mmq(model: torch.nn.Module) -> dict[str, Any]:
    """Install the native packed MMQ forward on the frozen Qwen4-Exp projections.

    The modules stay ordinary frozen `GgufLinear` instances: no adapter is created for them, and
    the patch fails closed instead of silently keeping the dequantize-and-multiply forward.
    """

    get_base_model = getattr(model, "get_base_model", None)
    base = get_base_model() if callable(get_base_model) else model
    report = patch_module_forwards(base, _FROZEN_MMQ_SPECS)
    report["paths"] = sorted(report["handled_by_key"]["enabled"])
    return report


def require_complete_qwen4_exp_frozen_mmq(report: dict[str, Any]) -> None:
    """Fail closed unless every frozen packed projection runs natively."""

    require_complete_inventory(
        report,
        {"enabled": EXPECTED_FROZEN_MMQ},
        subject="Qwen4-Exp frozen packed MMQ",
    )


def register_qwen4_exp_adapters(
    lora_config: LoraConfig, model: torch.nn.Module
) -> LoraConfig:
    """Register ordinary and routed-expert wrappers for one Qwen4-Exp LoraConfig.

    The ordinary wrappers keep the fused LoRA-B plus residual `addmm` path and run every packed base
    projection through the native dense MMQ base forward.
    """

    if lora_config.target_modules != QWEN4_EXP_TARGET_MODULES_PATTERN:
        raise ValueError(
            "Qwen4-Exp target_modules must use the exact indexer-excluding pattern "
            f"{QWEN4_EXP_TARGET_MODULES_PATTERN!r}."
        )
    register_fast_lora(lora_config, model)
    register_qwen4_exp_moe_lora(lora_config, model)
    return lora_config
