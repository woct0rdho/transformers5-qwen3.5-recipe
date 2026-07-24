"""Fast instance-local Liger RMSNorm integration for DeepSeek V4.

Weighted DeepSeek norms have frozen FP32 scale vectors. Stock Liger computes a
scale gradient whenever a weight tensor is supplied, regardless of
``requires_grad``. This module reuses Liger's weighted forward and supplies an
in-place backward that computes only the activation gradient. Q-B's scale-free
norm uses stock Liger with ``W=None``, which already has no weight-gradient
work.

Width-128 weighted norms use a narrow-geometry launch specialization. The
mHC input norms remain strict for the dedicated norm/projection/Sinkhorn/collapse
fusion.
"""

import math
from types import MethodType
from typing import Any

import torch
import triton
import triton.language as tl
from liger_kernel.ops import LigerRMSNormFunction
from liger_kernel.ops.rms_norm import rms_norm_forward
from liger_kernel.ops.utils import ensure_contiguous, torch_to_triton_dtype
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4RMSNorm,
    DeepseekV4UnweightedRMSNorm,
)

EXPECTED_WEIGHTED_RMSNORMS = 235
EXPECTED_SKIPPED_WEIGHTED_128_RMSNORMS = 0
EXPECTED_Q_B_RMSNORMS = 43
EXPECTED_SKIPPED_MHC_RMSNORMS = 87
_PATCH_MARKER = "_deepseek_v4_liger_rmsnorm"


@triton.jit
def _frozen_weight_rms_norm_backward_kernel(
    dY_ptr,
    dY_row_stride,
    X_ptr,
    X_row_stride,
    X_dtype: tl.constexpr,
    W_ptr,
    W_row_stride,
    RSTD_ptr,
    RSTD_row_stride,
    n_rows,
    n_cols,
    rows_per_program,
    BLOCK_SIZE: tl.constexpr,
):
    """Gemma-casting RMSNorm activation gradient without ``dW``."""

    row_block_id = tl.program_id(0).to(tl.int64)
    row_start = row_block_id * rows_per_program
    row_end = min((row_block_id + 1) * rows_per_program, n_rows)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    weight = tl.load(W_ptr + col_offsets * W_row_stride, mask=mask, other=0.0).to(
        tl.float32
    )

    for row_idx in range(row_start, row_end):
        dy_base = dY_ptr + row_idx * dY_row_stride
        x_base = X_ptr + row_idx * X_row_stride
        rstd_base = RSTD_ptr + row_idx * RSTD_row_stride

        dy = tl.load(dy_base + col_offsets, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_base + col_offsets, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.load(rstd_base).to(tl.float32)

        weighted_dy = dy * weight
        projection = tl.sum(weighted_dy * x, axis=0)
        dx = rstd * (weighted_dy - x * (projection * rstd * rstd / n_cols))
        # The incoming cotangent is dead at every accepted DeepSeek call site.
        tl.store(dy_base + col_offsets, dx.to(X_dtype), mask=mask)


def _frozen_weight_rms_norm_backward(
    dY: torch.Tensor,
    X: torch.Tensor,
    W: torch.Tensor,
    RSTD: torch.Tensor,
    block_size: int,
    num_warps: int,
) -> torch.Tensor:
    shape = dY.shape
    dim = shape[-1]
    dY = dY.view(-1, dim)
    n_rows, n_cols = dY.shape
    if n_cols > block_size:
        raise RuntimeError("Liger RMSNorm does not support feature dimensions >= 64K")
    if X.device.type != "cuda":
        raise RuntimeError("DeepSeek V4 Liger RMSNorm requires a CUDA/ROCm device")

    sm_count = torch.cuda.get_device_properties(X.device).multi_processor_count
    # Width 128 is latency/occupancy limited. The normal one-program-per-SM
    # geometry is tuned for wide norms and serializes too many narrow rows.
    # Extra independent programs are cheap here because this path has no dW
    # partials to merge. Keep this as a single launch-geometry specialization.
    if n_cols == 128:
        grid = min(n_rows, sm_count * 4)
        launch_num_warps = 1
    else:
        grid = sm_count
        launch_num_warps = num_warps
    rows_per_program = math.ceil(n_rows / grid)
    _frozen_weight_rms_norm_backward_kernel[(grid,)](
        dY,
        dY.stride(0),
        X,
        X.stride(0),
        torch_to_triton_dtype[X.dtype],
        W,
        W.stride(0),
        RSTD,
        RSTD.stride(0),
        n_rows,
        n_cols,
        rows_per_program,
        BLOCK_SIZE=block_size,
        num_warps=launch_num_warps,
    )
    return dY.view(*shape)


class _LigerFrozenWeightRMSNormFunction(torch.autograd.Function):
    """Liger weighted forward with an in-place frozen-weight backward."""

    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        X: torch.Tensor,
        W: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        if X.device.type != "cuda" or W.device.type != "cuda":
            raise RuntimeError("DeepSeek V4 Liger RMSNorm requires CUDA/ROCm tensors")
        Y, X, RSTD, block_size, num_warps, _ = rms_norm_forward(
            X,
            W,
            eps,
            0.0,
            "gemma",
            True,
        )
        ctx.block_size = block_size
        ctx.num_warps = num_warps
        ctx.save_for_backward(X, W, RSTD)
        return Y

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dY: torch.Tensor):
        X, W, RSTD = ctx.saved_tensors
        dX = _frozen_weight_rms_norm_backward(
            dY,
            X,
            W,
            RSTD,
            ctx.block_size,
            ctx.num_warps,
        )
        return dX, None, None


def _weighted_liger_forward(
    self: DeepseekV4RMSNorm, hidden_states: torch.Tensor
) -> torch.Tensor:
    return _LigerFrozenWeightRMSNormFunction.apply(
        hidden_states,
        self.weight,
        self.variance_epsilon,
    )


def _unweighted_liger_forward(
    self: DeepseekV4UnweightedRMSNorm, hidden_states: torch.Tensor
) -> torch.Tensor:
    return LigerRMSNormFunction.apply(
        hidden_states,
        None,
        self.eps,
        0.0,
        "llama",
        True,
        None,
    )


def configure_deepseek_v4_liger_rmsnorm(
    model: torch.nn.Module,
) -> dict[str, Any]:
    """Patch the accepted weighted and Q-B norms on one frozen model."""

    weighted = 0
    skipped_weighted_128 = 0
    q_b = 0
    skipped_mhc = 0
    already_patched = 0
    patched_names: list[str] = []

    for name, module in model.named_modules():
        if isinstance(module, DeepseekV4RMSNorm):
            # Norm scales are outside the fixed LoRA target contract. Freeze
            # them here so this base-model patch does not depend on PEFT order.
            module.weight.requires_grad_(False)
            if getattr(module, _PATCH_MARKER, False):
                already_patched += 1
                weighted += 1
                continue
            if module.weight.dtype != torch.float32:
                raise RuntimeError(
                    "DeepSeek weighted Liger RMSNorm requires an FP32 scale vector; "
                    f"{name!r} has {module.weight.dtype}"
                )
            module.forward = MethodType(_weighted_liger_forward, module)
            setattr(module, _PATCH_MARKER, True)
            weighted += 1
            patched_names.append(name)
        elif isinstance(module, DeepseekV4UnweightedRMSNorm):
            if name.endswith("q_b_norm"):
                if getattr(module, _PATCH_MARKER, False):
                    already_patched += 1
                    q_b += 1
                    continue
                module.forward = MethodType(_unweighted_liger_forward, module)
                setattr(module, _PATCH_MARKER, True)
                q_b += 1
                patched_names.append(name)
            else:
                skipped_mhc += 1

    return {
        "weighted": weighted,
        "skipped_weighted_128": skipped_weighted_128,
        "q_b_unweighted": q_b,
        "skipped_mhc_unweighted": skipped_mhc,
        "patched": len(patched_names),
        "already_patched": already_patched,
        "backward": "in_place_frozen_dx_only",
        "patched_names": patched_names,
    }


def require_complete_deepseek_v4_liger_rmsnorm(report: dict[str, Any]) -> None:
    """Fail closed unless the fixed DeepSeek V4 norm inventory was handled."""

    expected = {
        "weighted": EXPECTED_WEIGHTED_RMSNORMS,
        "skipped_weighted_128": EXPECTED_SKIPPED_WEIGHTED_128_RMSNORMS,
        "q_b_unweighted": EXPECTED_Q_B_RMSNORMS,
        "skipped_mhc_unweighted": EXPECTED_SKIPPED_MHC_RMSNORMS,
    }
    mismatches = {
        key: (expected_value, report.get(key))
        for key, expected_value in expected.items()
        if report.get(key) != expected_value
    }
    if mismatches:
        raise RuntimeError(
            f"incomplete DeepSeek V4 Liger RMSNorm configuration: {mismatches}"
        )
