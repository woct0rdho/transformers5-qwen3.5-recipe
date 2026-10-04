"""Instance-local Liger, FLA and project kernels for the Qwen4 norms.

Transformers can route both norm classes through the `kernels` hub, but that mapping is only
registered when a model is loaded with the opt-in `use_kernels=True`, and the `kernels-community/fla`
layer that owns the gated kernel is not part of this environment. TRL's `use_liger_kernel` would patch
the plain norms from `TrainingArguments`, which hides the decision in the trainer. This repository
installs the kernels on the module instances instead, so the exact configuration is visible to the
audit and to the trainer.

Semantics preserved:
- `Qwen4ExpTextRMSNorm` computes `x_norm * (1 + weight)` in FP32 and returns the input dtype. Liger's
  contract for that is `offset=1.0` and `casting_mode="gemma"`. `in_place=False` keeps the residual
  stream readable after the norm.
- The same class with a `group_size` normalizes each group of the last dimension on its own. Liger has
  no group dimension, so those sites run the grouped kernel below, which is `dX`-only and therefore
  requires the frozen weight the checkpoint has.
- `Qwen4ExpTextRMSNormGated` computes `weight * x_norm * activation(gate)` for the activation the
  config declares, which is sigmoid on this checkpoint. FLA's contract for that is
  `LayerNormGatedFunction` with `is_rms_norm=True`.

The 97 grouped norms inside the hyper-connections are served by that module's own forward, which owns
the whole module, so this module counts them as skipped. What it serves is the 84 plain and gated
norms (24 attention Q/K head, 24 indexer, 36 gated recurrent) and the three PLE grouped norms. The
patched module classes, parameter names, shapes and dtypes are unchanged, so PEFT serialization, the
frozen-base contract and the audit inventories stay valid.
"""

from typing import Any

import torch
import triton
import triton.language as tl
from fla.modules.fused_norm_gate import LayerNormGatedFunction
from liger_kernel.ops import LigerRMSNormFunction
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextRMSNorm,
    Qwen4ExpTextRMSNormGated,
)

from module_patching import (
    ModulePatchSpec,
    patch_module_forwards,
    require_complete_inventory,
    require_cuda_weight,
)

_EXPECTED_RMSNORMS = 48
_EXPECTED_GROUPED_RMSNORMS = 3
_EXPECTED_SKIPPED_HYPER_CONNECTION_RMSNORMS = 97
_EXPECTED_GATED_RMSNORMS = 36
_SUBJECT = "Qwen4-Exp fused norm"
# The checkpoint's grouped geometry, which is what the launch configuration below is tuned for.
_GROUP = 2560
_GROUPED_WIDTH = 10240
_SUPPORTED_ROWS = frozenset({2048, 8192, 32768})
# One program per (row, group) pair. A group is 2560 wide, which is not a power of two, so the tile is
# 4096 with a mask. One launch configuration serves every supported row count.
_NUM_WARPS = 2
_NUM_STAGES = 1
_LIGER_OFFSET = 1.0
_LIGER_CASTING_MODE = "gemma"
_LIGER_IN_PLACE = False
_FLA_ACTIVATIONS = frozenset({"swish", "silu", "sigmoid"})
_HYPER_CONNECTION_SUFFIX = ".hc_norm"
_REFERENCE_ATTRIBUTE = "_qwen4_rmsnorm_reference"


@triton.jit
def _grouped_rms_norm_forward_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    invr_ptr,
    eps,
    stride_row,
    groups,
    GROUP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    group = tl.program_id(1)
    offs = tl.arange(0, BLOCK)
    mask = offs < GROUP
    column = group * GROUP + offs
    x = tl.load(x_ptr + row * stride_row + column, mask=mask, other=0.0).to(tl.float32)
    invr = tl.rsqrt(tl.sum(x * x, axis=0) / GROUP + eps)
    tl.store(invr_ptr + row * groups + group, invr)
    weight = tl.load(weight_ptr + column, mask=mask, other=0.0).to(tl.float32)
    tl.store(
        out_ptr + row * stride_row + column,
        (x * invr * (1.0 + weight)).to(out_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _grouped_rms_norm_backward_kernel(
    dout_ptr,
    x_ptr,
    weight_ptr,
    invr_ptr,
    dx_ptr,
    stride_row,
    groups,
    GROUP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    group = tl.program_id(1)
    offs = tl.arange(0, BLOCK)
    mask = offs < GROUP
    column = group * GROUP + offs
    offset = row * stride_row + column
    invr = tl.load(invr_ptr + row * groups + group)
    x_norm = tl.load(x_ptr + offset, mask=mask, other=0.0).to(tl.float32) * invr
    dout = tl.load(dout_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + column, mask=mask, other=0.0).to(tl.float32)
    # The whole group is in this program, so the norm's mean is a register reduction and dX needs no
    # second pass over the stream. The scale vector belongs inside the reduction: it is free within a
    # group, so the unweighted `mean(dout * x_norm)` is the gradient of a different function.
    scaled_dout = (1.0 + weight) * dout
    coefficient = tl.sum(scaled_dout * x_norm, axis=0) / GROUP
    dx = invr * (scaled_dout - x_norm * coefficient)
    tl.store(dx_ptr + offset, dx.to(dx_ptr.dtype.element_ty), mask=mask)


def _rows_and_shapes(x: torch.Tensor, group: int) -> tuple[int, int, int]:
    rows = x.numel() // x.shape[-1]
    groups = x.shape[-1] // group
    return rows, groups, x.shape[-1]


def grouped_rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float, group: int
) -> torch.Tensor:
    """Grouped RMSNorm with the `(1 + weight)` contract the Qwen4 checkpoint uses."""

    return _GroupedRMSNormFunction.apply(x, weight, float(eps), int(group))


class _GroupedRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, x: torch.Tensor, weight: torch.Tensor, eps: float, group: int
    ) -> torch.Tensor:
        rows, groups, width = _rows_and_shapes(x, group)
        out = torch.empty_like(x)
        invr = torch.empty((rows, groups), dtype=torch.float32, device=x.device)
        _grouped_rms_norm_forward_kernel[(rows, groups)](
            x,
            weight,
            out,
            invr,
            eps,
            width,
            groups,
            GROUP=group,
            BLOCK=triton.next_power_of_2(group),
            num_warps=_NUM_WARPS,
            num_stages=_NUM_STAGES,
        )
        ctx.save_for_backward(x, weight, invr)
        ctx.group = group
        return out

    @staticmethod
    def backward(ctx: Any, dout: torch.Tensor):  # ty: ignore[invalid-method-override]
        x, weight, invr = ctx.saved_tensors
        group = ctx.group
        rows, groups, width = _rows_and_shapes(x, group)
        # Autograd hands the incoming cotangent in whatever layout the next op produced, often an
        # expanded or otherwise strided tensor, and the kernel indexes rows with a linear stride.
        dout = dout.contiguous()
        dx = torch.empty_like(x)
        _grouped_rms_norm_backward_kernel[(rows, groups)](
            dout,
            x,
            weight,
            invr,
            dx,
            width,
            groups,
            GROUP=group,
            BLOCK=triton.next_power_of_2(group),
            num_warps=_NUM_WARPS,
            num_stages=_NUM_STAGES,
        )
        return dx, None, None, None


def grouped_rms_norm_serves(
    x: torch.Tensor, weight: torch.Tensor, group_size: int | None
) -> bool:
    """Whether the grouped kernel serves this call as it stands.

    Anything outside this contract falls back to the module's own forward. The audit requires the
    predicate to be true on the loaded model, so a shape, dtype or frozen-weight mismatch shows up as
    a gate failure instead of a silent fall back to the reference.
    """

    if group_size != _GROUP:
        return False
    if not weight.is_floating_point() or weight.requires_grad:
        return False
    if weight.numel() != _GROUPED_WIDTH:
        return False
    if x.ndim == 0 or x.shape[-1] != _GROUPED_WIDTH:
        return False
    if x.dtype != torch.bfloat16 or not x.is_contiguous():
        return False
    return x.numel() // _GROUPED_WIDTH in _SUPPORTED_ROWS


def _liger_rmsnorm_forward(
    self: Qwen4ExpTextRMSNorm, hidden_states: torch.Tensor
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


def _grouped_rmsnorm_forward(
    self: Qwen4ExpTextRMSNorm, hidden_states: torch.Tensor
) -> torch.Tensor:
    group = self.group_size
    if group is None or not grouped_rms_norm_serves(hidden_states, self.weight, group):
        reference = getattr(self, _REFERENCE_ATTRIBUTE, None)
        if reference is None:
            raise RuntimeError(
                "Qwen4-Exp grouped RMSNorm has no reference forward to fall back to"
            )
        return reference(hidden_states)
    return grouped_rms_norm(hidden_states, self.weight, self.eps, group)


def _fla_gated_rmsnorm_forward(
    self: Qwen4ExpTextRMSNormGated,
    hidden_states: torch.Tensor,
    gate: torch.Tensor | None = None,
) -> torch.Tensor:
    if gate is None:
        raise RuntimeError("Qwen4-Exp gated RMSNorm requires its gate tensor")
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


def _validate_rmsnorm(name: str, module: Qwen4ExpTextRMSNorm) -> None:
    require_cuda_weight(name, module, subject=_SUBJECT)


def _validate_grouped_rmsnorm(name: str, module: Qwen4ExpTextRMSNorm) -> None:
    require_cuda_weight(name, module, subject=_SUBJECT)
    if module.group_size is None:
        raise RuntimeError(f"Qwen4-Exp grouped RMSNorm {name!r} has no group size")


def _validate_gated_rmsnorm(name: str, module: Qwen4ExpTextRMSNormGated) -> None:
    require_cuda_weight(name, module, subject=_SUBJECT)
    if module.activation not in _FLA_ACTIVATIONS:
        raise RuntimeError(
            f"Qwen4-Exp gated RMSNorm {name!r} uses unsupported activation "
            f"{module.activation!r}"
        )


def _is_grouped_rmsnorm(name: str, module: Qwen4ExpTextRMSNorm) -> bool:
    del name
    return module.group_size is not None


def _is_plain_rmsnorm(name: str, module: Qwen4ExpTextRMSNorm) -> bool:
    del name
    return module.group_size is None


def _is_served_grouped_rmsnorm(name: str, module: Qwen4ExpTextRMSNorm) -> bool:
    del module
    return not name.endswith(_HYPER_CONNECTION_SUFFIX)


def _keep_reference(name: str, module: Qwen4ExpTextRMSNorm) -> None:
    """Keep the module's own forward, once, for the calls the grouped kernel does not serve.

    This runs on every pass, including passes over already-patched modules, so it must not capture the
    patched forward as its own reference.
    """

    if not hasattr(module, _REFERENCE_ATTRIBUTE):
        setattr(module, _REFERENCE_ATTRIBUTE, module.forward)


_SPECS = (
    ModulePatchSpec(
        module_type=Qwen4ExpTextRMSNorm,
        forward=_grouped_rmsnorm_forward,
        handled_key="grouped_rmsnorms",
        matches=_is_grouped_rmsnorm,
        accept=_is_served_grouped_rmsnorm,
        skip_key="skipped_hyper_connection_rmsnorms",
        validate=_validate_grouped_rmsnorm,
        prepare=_keep_reference,
    ),
    ModulePatchSpec(
        module_type=Qwen4ExpTextRMSNorm,
        forward=_liger_rmsnorm_forward,
        handled_key="rmsnorms",
        matches=_is_plain_rmsnorm,
        validate=_validate_rmsnorm,
    ),
    ModulePatchSpec(
        module_type=Qwen4ExpTextRMSNormGated,
        forward=_fla_gated_rmsnorm_forward,
        handled_key="gated_rmsnorms",
        validate=_validate_gated_rmsnorm,
    ),
)


def configure_qwen4_exp_fused_norms(model: torch.nn.Module) -> dict[str, Any]:
    """Install the fused norm kernels on one loaded Qwen4-Exp model instance."""

    report = patch_module_forwards(model, _SPECS)
    report["liger"] = {
        "offset": _LIGER_OFFSET,
        "casting_mode": _LIGER_CASTING_MODE,
        "in_place": _LIGER_IN_PLACE,
    }
    report["fla"] = {"activation": "silu", "is_rms_norm": True}
    report["group"] = _GROUP
    report["grouped_width"] = _GROUPED_WIDTH
    report["rows"] = sorted(_SUPPORTED_ROWS)
    return report


def require_complete_qwen4_exp_fused_norms(
    report: dict[str, Any],
    *,
    expected_rmsnorms: int = _EXPECTED_RMSNORMS,
    expected_grouped_rmsnorms: int = _EXPECTED_GROUPED_RMSNORMS,
    expected_skipped_hyper_connection_rmsnorms: int = (
        _EXPECTED_SKIPPED_HYPER_CONNECTION_RMSNORMS
    ),
    expected_gated_rmsnorms: int = _EXPECTED_GATED_RMSNORMS,
) -> None:
    """Fail closed unless the fixed Qwen4-Exp norm inventory was handled."""

    require_complete_inventory(
        report,
        {
            "rmsnorms": expected_rmsnorms,
            "grouped_rmsnorms": expected_grouped_rmsnorms,
            "skipped_hyper_connection_rmsnorms": (
                expected_skipped_hyper_connection_rmsnorms
            ),
            "gated_rmsnorms": expected_gated_rmsnorms,
        },
        subject=_SUBJECT,
    )
