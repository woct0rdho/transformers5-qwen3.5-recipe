"""Fused grouped RMSNorm for the Qwen4 hyper-connections.

The checkpoint's norm runs as five kernels over an fp32 copy of the whole `[rows, hc_count * hidden]`
stream: the upcast, the square, the group mean, `rsqrt`, and the `(1 + weight)` multiply with the cast
back. This module replaces it with one kernel per direction, `grouped_rms_norm` from
`qwen4_exp_fused_norms.py`, which is where the rest of the norm family lives. The forward reads bf16,
accumulates each group's sum of squares in fp32 registers, applies `rsqrt` and `(1 + weight)` in fp32
and stores bf16, so the stream is read once and written once. The backward holds the whole group in
registers, so the group mean is a register reduction and `dX` is written in the same pass.

The norm weight is frozen on this checkpoint and carries no adapter, so the backward returns `dX`
only, which is what the grouped-norm item asks for. A trainable norm weight falls back to the
reference rather than silently dropping its gradient, as do inputs outside the shapes, the dtype and
the cache the kernels serve.

The rest of the hyper-connection stays the module's own: the two projections with their SiLU and
sigmoid, the gated mean and the injections are `nn.Linear` and elementwise work the library already
serves.
"""

from typing import Any

import torch
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextGatedResidual,
)

from module_patching import (
    ModulePatchSpec,
    patch_module_forwards,
    require_complete_inventory,
)
from qwen4_exp_fused_norms import grouped_rms_norm, grouped_rms_norm_serves

_HC = 4
_HIDDEN = 2560
_FLAT = _HC * _HIDDEN
_EXPECTED_CONNECTIONS = 97
_SUBJECT = "Qwen4-Exp hyper-connections"
_MARKER = "_patched_qwen4_hc_norm"
_REFERENCE_ATTRIBUTE = "_qwen4_hc_forward_reference"


def hc_norm_serves(
    module: Qwen4ExpTextGatedResidual, hyper_input: torch.Tensor
) -> bool:
    """Whether the fused norm serves this call as it stands.

    The audit requires this to be true on the loaded model, so a shape, dtype or frozen-weight
    mismatch shows up as a gate failure instead of a silent fall back to the reference.
    """

    norm = module.hc_norm
    if module.hc_count != _HC or module.hidden_size != _HIDDEN:
        return False
    if norm.group_size != module.hidden_size:
        return False
    return grouped_rms_norm_serves(hyper_input, norm.weight, norm.group_size)


def _qwen4_hc_forward(
    self: Qwen4ExpTextGatedResidual, hyper_input: torch.Tensor
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reference = getattr(self, _REFERENCE_ATTRIBUTE, None)
    if reference is None:
        raise RuntimeError(
            "Qwen4-Exp hyper-connection has no reference forward to fall back to"
        )
    if not hc_norm_serves(self, hyper_input):
        return reference(hyper_input)

    hc_count = self.hc_count
    hidden_size = self.hidden_size
    group = self.hc_norm.group_size
    if group is None:
        return reference(hyper_input)
    hyper_input_normed = grouped_rms_norm(
        hyper_input, self.hc_norm.weight, self.hc_norm.eps, group
    )
    input_mix_weight = torch.nn.functional.silu(
        self.input_mix_weight_down(hyper_input_normed) / hc_count
    )
    input_mix_weight = torch.sigmoid(self.input_mix_weight_up(input_mix_weight))
    input_mix_weight = input_mix_weight.unflatten(-1, (hc_count, hidden_size))
    mixed_input = (
        input_mix_weight * hyper_input_normed.unflatten(-1, (hc_count, hidden_size))
    ).mean(dim=-2)
    if self.block_inject_weight is None:
        return mixed_input
    injection_weights = 2 * torch.sigmoid(
        self.block_inject_weight(hyper_input_normed) / hc_count
    )
    return mixed_input, hyper_input, injection_weights


def reference_hc_output(
    module: Qwen4ExpTextGatedResidual, hyper_input: torch.Tensor
) -> Any:
    """Run the module's own forward, the one the patch replaces.

    The audit uses this to prove the fused norm agrees with the module's reference norm on real
    activations, forward and backward.
    """

    reference = getattr(module, _REFERENCE_ATTRIBUTE, None)
    if reference is None:
        raise RuntimeError(
            "Qwen4-Exp hyper-connection was not patched, so it has no reference forward"
        )
    return reference(hyper_input)


def _validate_qwen4_hc(name: str, module: Qwen4ExpTextGatedResidual) -> None:
    if module.hc_count != _HC or module.hidden_size != _HIDDEN:
        raise RuntimeError(
            f"Qwen4-Exp hyper-connection {name!r} has {module.hc_count} streams of "
            f"{module.hidden_size}, expected {_HC} of {_HIDDEN}."
        )
    if module.hc_norm.group_size != module.hidden_size:
        raise RuntimeError(
            f"Qwen4-Exp hyper-connection {name!r} norm group {module.hc_norm.group_size} "
            f"does not match hidden size {module.hidden_size}."
        )
    if module.hc_norm.weight.numel() != _FLAT:
        raise RuntimeError(
            f"Qwen4-Exp hyper-connection {name!r} norm has "
            f"{module.hc_norm.weight.numel()} weights, expected {_FLAT}."
        )


def _prepare_qwen4_hc(name: str, module: Qwen4ExpTextGatedResidual) -> None:
    """Keep the module's own forward, once, before it is replaced.

    This runs on every pass, including passes over already-patched modules, so it must not capture
    the patched forward as its own reference.
    """

    if not hasattr(module, _REFERENCE_ATTRIBUTE):
        setattr(module, _REFERENCE_ATTRIBUTE, module.forward)


_SPECS = (
    ModulePatchSpec(
        module_type=Qwen4ExpTextGatedResidual,
        forward=_qwen4_hc_forward,
        handled_key="hyper_connections",
        validate=_validate_qwen4_hc,
        prepare=_prepare_qwen4_hc,
        marker=_MARKER,
        freeze_weight=False,
    ),
)


def configure_qwen4_exp_hc_norm(model: torch.nn.Module) -> dict[str, Any]:
    """Install the fused grouped RMSNorm on one loaded Qwen4-Exp model instance."""

    report = patch_module_forwards(model, _SPECS)
    report["group_size"] = _HIDDEN
    report["flat_size"] = _FLAT
    return report


def require_complete_qwen4_exp_hc_norm(
    report: dict[str, Any],
    *,
    expected_connections: int = _EXPECTED_CONNECTIONS,
) -> None:
    """Fail closed unless every hyper-connection was configured."""

    require_complete_inventory(
        report,
        {
            "hyper_connections": expected_connections,
            "group_size": _HIDDEN,
            "flat_size": _FLAT,
        },
        subject=_SUBJECT,
    )
