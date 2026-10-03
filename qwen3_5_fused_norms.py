"""Instance-local Liger RMSNorm and FLA gated RMSNorm for Qwen3.5-MoE training.

Transformers wires both norms through the `kernels` hub, but the hub mapping is only registered
when a model is loaded with the opt-in `use_kernels=True`, and the `kernels-community/fla` layer
that owns the gated kernel is not part of this environment. TRL's `use_liger_kernel` would patch
the plain norms from `TrainingArguments`, which hides the decision in the trainer. This repository
installs both kernels here instead, on the module instances, so the exact configuration is visible
to the audit and to the trainer.

Semantics preserved:
- `Qwen3_5MoeRMSNorm` computes `x_norm * (1 + weight)` in FP32 and returns the input dtype.
  Liger's contract for that is `offset=1.0`, `casting_mode="gemma"`. `in_place=False` keeps the
  residual stream readable after the norm.
- `Qwen3_5MoeRMSNormGated` computes `weight * x_norm * silu(gate)` in FP32. FLA's contract for
  that is `FusedRMSNormGated`, i.e. `LayerNormGatedFunction` with `is_rms_norm=True`.

The patched module classes, parameter names, shapes, and dtypes are unchanged, so PEFT
serialization, the frozen-base contract, and the audit inventories stay valid. The shared
patching protocol lives in `module_patching.py`.
"""

from typing import Any

import torch
from fla.modules.fused_norm_gate import LayerNormGatedFunction
from liger_kernel.ops import LigerRMSNormFunction
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeRMSNorm,
    Qwen3_5MoeRMSNormGated,
)

from module_patching import (
    ModulePatchSpec,
    patch_module_forwards,
    require_complete_inventory,
    require_cuda_weight,
)

# Text-only Qwen3.5-MoE: 40 input + 40 post-attention + 1 final + 10x2 Q/K head norms,
# and one gated norm per GatedDeltaNet layer.
EXPECTED_RMSNORMS = 101
EXPECTED_GATED_RMSNORMS = 30
_SUBJECT = "Qwen3.5 fused norm"
_LIGER_OFFSET = 1.0
_LIGER_CASTING_MODE = "gemma"
_LIGER_IN_PLACE = False
_FLA_ACTIVATIONS = frozenset({"swish", "silu", "sigmoid"})


def _liger_rmsnorm_forward(
    self: Qwen3_5MoeRMSNorm, hidden_states: torch.Tensor
) -> torch.Tensor:
    return LigerRMSNormFunction.apply(
        hidden_states,
        self.weight,
        self.eps,
        _LIGER_OFFSET,
        _LIGER_CASTING_MODE,
        _LIGER_IN_PLACE,
        None,
    )


def _fla_gated_rmsnorm_forward(
    self: Qwen3_5MoeRMSNormGated,
    hidden_states: torch.Tensor,
    gate: torch.Tensor | None = None,
) -> torch.Tensor:
    if gate is None:
        raise RuntimeError("Qwen3.5 gated RMSNorm requires its gate tensor")
    return LayerNormGatedFunction.apply(
        hidden_states,
        gate,
        self.weight,
        None,
        self.activation,
        None,
        self.variance_epsilon,
        False,
        False,
        True,
    )


def _validate_rmsnorm(name: str, module: Qwen3_5MoeRMSNorm) -> None:
    require_cuda_weight(name, module, subject=_SUBJECT)


def _validate_gated_rmsnorm(name: str, module: Qwen3_5MoeRMSNormGated) -> None:
    require_cuda_weight(name, module, subject=_SUBJECT)
    if module.activation not in _FLA_ACTIVATIONS:
        raise RuntimeError(
            f"Qwen3.5 gated RMSNorm {name!r} uses unsupported activation "
            f"{module.activation!r}"
        )


_SPECS = (
    ModulePatchSpec(
        module_type=Qwen3_5MoeRMSNormGated,
        forward=_fla_gated_rmsnorm_forward,
        handled_key="gated_rmsnorms",
        validate=_validate_gated_rmsnorm,
    ),
    ModulePatchSpec(
        module_type=Qwen3_5MoeRMSNorm,
        forward=_liger_rmsnorm_forward,
        handled_key="rmsnorms",
        validate=_validate_rmsnorm,
    ),
)


def configure_qwen35_fused_norms(model: torch.nn.Module) -> dict[str, Any]:
    """Install the Liger and FLA kernels on one loaded Qwen3.5-MoE model."""

    report = patch_module_forwards(model, _SPECS)
    report["liger"] = {
        "offset": _LIGER_OFFSET,
        "casting_mode": _LIGER_CASTING_MODE,
        "in_place": _LIGER_IN_PLACE,
    }
    report["fla"] = {"activation": "silu", "is_rms_norm": True}
    return report


def require_complete_qwen35_fused_norms(
    report: dict[str, Any],
    *,
    expected_rmsnorms: int = EXPECTED_RMSNORMS,
    expected_gated_rmsnorms: int = EXPECTED_GATED_RMSNORMS,
) -> None:
    """Fail closed unless the fixed Qwen3.5-MoE norm inventory was handled."""

    require_complete_inventory(
        report,
        {
            "rmsnorms": expected_rmsnorms,
            "gated_rmsnorms": expected_gated_rmsnorms,
        },
        subject=_SUBJECT,
    )
