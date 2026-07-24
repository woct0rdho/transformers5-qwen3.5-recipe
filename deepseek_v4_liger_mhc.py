"""Fixed-geometry Triton mHC training kernels for DeepSeek V4.

The implementation follows the Liger custom-autograd decomposition while using
one-program-per-token projection loops inspired by SGLang's fused HC head.  It
is deliberately specialized for the DeepSeek V4 Flash training geometry:
HC=4, hidden size 4096, BF16 activations, FP32 controls with a native-layout
F16 projection cache, 20 Sinkhorn iterations, and physical
sequence-2048 batches 1, 4, and 16.

``deepseek_v4_mhc_prepare`` returns the residual streams as an alias. Passing
that alias to ``deepseek_v4_mhc_merge`` routes the direct residual gradient back
through the prepare backward, where it is updated in place with collapse and
coefficient-mediated gradients. Merge backward reuses its dead incoming
cotangent for that path. This avoids a second full activation-gradient
allocation and a framework add.
"""

from types import MethodType
from typing import Any

import torch
import triton
import triton.language as tl
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4DecoderLayer,
    DeepseekV4HyperConnection,
    DeepseekV4HyperHead,
)

_HC = 4
_HIDDEN = 4096
_FLAT = _HC * _HIDDEN
_MIX = (_HC + 2) * _HC
_SINKHORN_ITERS = 20
_SINKHORN_STATE_STEPS = 1 + 2 * (_SINKHORN_ITERS - 1)
_SUPPORTED_ROWS = frozenset({2048, 8192, 32768})
EXPECTED_DEEPSEEK_V4_MHC_CONNECTIONS = 86
EXPECTED_DEEPSEEK_V4_MHC_DECODER_LAYERS = 43
EXPECTED_DEEPSEEK_V4_MHC_HEADS = 1
_MHC_PATCH_MARKER = "_deepseek_v4_liger_mhc"

# Values are (block sizes..., num_warps, num_stages). Each kernel is tuned
# independently because its register pressure and reduction geometry differ.
_PREPARE_FORWARD_CONFIGS = {
    2048: (256, 256, 2, 2),
    8192: (256, 512, 2, 2),
    32768: (256, 256, 2, 2),
}
_CONTROLS_BACKWARD_CONFIGS = {
    2048: (1024, 8, 2),
    8192: (1024, 8, 2),
    32768: (1024, 2, 2),
}
_PREPARE_DX_CONFIGS = {
    2048: (512, 8, 2),
    8192: (512, 4, 2),
    32768: (512, 4, 2),
}
_MERGE_FORWARD_CONFIGS = {rows: (1024, 8, 2) for rows in _SUPPORTED_ROWS}
_MERGE_BACKWARD_CONFIGS = {rows: (1024, 8, 2) for rows in _SUPPORTED_ROWS}
_HEAD_FORWARD_CONFIGS = {rows: (512, 1024, 4, 2) for rows in _SUPPORTED_ROWS}
_HEAD_BACKWARD_CONFIGS = {
    2048: (512, 1024, 4, 2),
    8192: (512, 1024, 4, 2),
    32768: (512, 1024, 8, 2),
}


@triton.jit
def _mhc_prepare_forward_kernel(
    x_ptr,
    fn_ptr,
    base_ptr,
    scale_ptr,
    mix_ptr,
    invr_ptr,
    coeff_ptr,
    sink_state_ptr,
    collapsed_ptr,
    RMS_EPS: tl.constexpr,
    HC_EPS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    _HC: tl.constexpr,
    _HIDDEN: tl.constexpr,
    _FLAT: tl.constexpr,
    _MIX: tl.constexpr,
    _SINKHORN_ITERS: tl.constexpr,
    _SINKHORN_STATE_STEPS: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int32)
    mix16_offsets = tl.arange(0, 16)
    mix8_offsets = 16 + tl.arange(0, 8)
    projected16 = tl.zeros((16,), tl.float32)
    projected8 = tl.zeros((8,), tl.float32)
    sumsq = tl.zeros((), tl.float32)
    x_row = x_ptr + token * _FLAT

    # Native fn layout is [24, 16384]. Keeping its K dimension contiguous makes
    # the one-CTA-per-token loads coalesced and removes a cached transpose.
    for k_start in tl.range(0, _FLAT, BLOCK_K, loop_unroll_factor=1):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        x = tl.load(x_row + k_offsets).to(tl.float32)
        fn16 = tl.load(fn_ptr + mix16_offsets[:, None] * _FLAT + k_offsets[None, :]).to(
            tl.float32
        )
        fn8 = tl.load(fn_ptr + mix8_offsets[:, None] * _FLAT + k_offsets[None, :]).to(
            tl.float32
        )
        projected16 += tl.sum(fn16 * x[None, :], axis=1)
        projected8 += tl.sum(fn8 * x[None, :], axis=1)
        sumsq += tl.sum(x * x, axis=0)

    invr = tl.rsqrt(sumsq / _FLAT + RMS_EPS)
    mix16 = projected16 * invr
    mix8 = projected8 * invr
    tl.store(mix_ptr + token * _MIX + mix16_offsets, mix16)
    tl.store(mix_ptr + token * _MIX + mix8_offsets, mix8)
    tl.store(invr_ptr + token, invr)

    # Triton tensors do not support contiguous slices. Reshape and split the
    # exact 16-value tile into the semantic 4 pre, 4 post, and 8 comb values.
    mix16_pairs = tl.permute(tl.reshape(mix16, (2, 8)), (1, 0))
    mix_first8, mix_comb0 = tl.split(mix16_pairs)
    mix_first8_pairs = tl.permute(tl.reshape(mix_first8, (2, 4)), (1, 0))
    mix_pre, mix_post = tl.split(mix_first8_pairs)

    h = tl.arange(0, _HC)
    base_pre = tl.load(base_ptr + h).to(tl.float32)
    base_post = tl.load(base_ptr + _HC + h).to(tl.float32)
    pre_scale = tl.load(scale_ptr).to(tl.float32)
    post_scale = tl.load(scale_ptr + 1).to(tl.float32)
    comb_scale = tl.load(scale_ptr + 2).to(tl.float32)
    pre = tl.sigmoid(mix_pre * pre_scale + base_pre) + HC_EPS
    post = 2.0 * tl.sigmoid(mix_post * post_scale + base_post)
    tl.store(coeff_ptr + token * _MIX + h, pre)
    tl.store(coeff_ptr + token * _MIX + _HC + h, post)

    rows = tl.arange(0, _HC)[:, None]
    cols = tl.arange(0, _HC)[None, :]
    flat = rows * _HC + cols
    comb_base = tl.load(base_ptr + 2 * _HC + flat).to(tl.float32)
    mix_comb_flat = tl.cat(mix_comb0, mix8)
    mix_comb = tl.reshape(mix_comb_flat, (_HC, _HC))
    logits = mix_comb * comb_scale + comb_base
    row_max = tl.max(logits, axis=1)
    exp_logits = tl.exp(logits - row_max[:, None])
    softmax = exp_logits / tl.sum(exp_logits, axis=1)[:, None]
    matrix = softmax + HC_EPS

    col_denom = tl.sum(matrix, axis=0) + HC_EPS
    matrix = matrix / col_denom[None, :]
    state_base = sink_state_ptr + token * _SINKHORN_STATE_STEPS * _HC
    tl.store(state_base + cols, col_denom[None, :])

    for step in tl.static_range(0, _SINKHORN_ITERS - 1):
        row_denom = tl.sum(matrix, axis=1) + HC_EPS
        matrix = matrix / row_denom[:, None]
        col_denom = tl.sum(matrix, axis=0) + HC_EPS
        matrix = matrix / col_denom[None, :]
        tl.store(state_base + (1 + 2 * step) * _HC + rows, row_denom[:, None])
        tl.store(state_base + (2 + 2 * step) * _HC + cols, col_denom[None, :])

    tl.store(coeff_ptr + token * _MIX + 2 * _HC + flat, matrix)

    # The collapse is the second and final activation pass in prepare forward.
    for d_start in tl.range(0, _HIDDEN, BLOCK_D, loop_unroll_factor=1):
        d_offsets = d_start + tl.arange(0, BLOCK_D)
        streams = tl.load(x_row + h[:, None] * _HIDDEN + d_offsets[None, :]).to(
            tl.float32
        )
        collapsed = tl.sum(pre[:, None] * streams, axis=0)
        tl.store(
            collapsed_ptr + token * _HIDDEN + d_offsets,
            collapsed.to(collapsed_ptr.dtype.element_ty),
        )


@triton.jit
def _mhc_controls_backward_kernel(
    x_ptr,
    mix_ptr,
    coeff_ptr,
    sink_state_ptr,
    scale_ptr,
    grad_coeff_ptr,
    grad_collapsed_ptr,
    grad_mix_ptr,
    HC_EPS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    _HC: tl.constexpr,
    _HIDDEN: tl.constexpr,
    _FLAT: tl.constexpr,
    _MIX: tl.constexpr,
    _SINKHORN_ITERS: tl.constexpr,
    _SINKHORN_STATE_STEPS: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int32)
    h = tl.arange(0, _HC)
    x_row = x_ptr + token * _FLAT
    grad_pre = tl.load(grad_coeff_ptr + token * _MIX + h).to(tl.float32)

    for d_start in tl.range(0, _HIDDEN, BLOCK_D, loop_unroll_factor=1):
        d_offsets = d_start + tl.arange(0, BLOCK_D)
        grad_collapsed = tl.load(grad_collapsed_ptr + token * _HIDDEN + d_offsets).to(
            tl.float32
        )
        streams = tl.load(x_row + h[:, None] * _HIDDEN + d_offsets[None, :]).to(
            tl.float32
        )
        grad_pre += tl.sum(streams * grad_collapsed[None, :], axis=1)

    pre = tl.load(coeff_ptr + token * _MIX + h).to(tl.float32) - HC_EPS
    post = tl.load(coeff_ptr + token * _MIX + _HC + h).to(tl.float32) * 0.5
    pre_scale = tl.load(scale_ptr).to(tl.float32)
    post_scale = tl.load(scale_ptr + 1).to(tl.float32)
    comb_scale = tl.load(scale_ptr + 2).to(tl.float32)
    grad_post = tl.load(grad_coeff_ptr + token * _MIX + _HC + h).to(tl.float32)
    grad_mix_pre = grad_pre * pre * (1.0 - pre) * pre_scale
    grad_mix_post = grad_post * (2.0 * post * (1.0 - post)) * post_scale
    tl.store(grad_mix_ptr + token * _MIX + h, grad_mix_pre)
    tl.store(grad_mix_ptr + token * _MIX + _HC + h, grad_mix_post)

    rows = tl.arange(0, _HC)[:, None]
    cols = tl.arange(0, _HC)[None, :]
    flat = rows * _HC + cols
    matrix = tl.load(coeff_ptr + token * _MIX + 2 * _HC + flat).to(tl.float32)
    grad = tl.load(grad_coeff_ptr + token * _MIX + 2 * _HC + flat).to(tl.float32)
    state_base = sink_state_ptr + token * _SINKHORN_STATE_STEPS * _HC

    # Reverse the 19 row/column normalization pairs. The saved denominators
    # reconstruct each input matrix, so no full 20x4x4 history is needed.
    for step in tl.static_range(_SINKHORN_ITERS - 2, -1, -1):
        col_denom = tl.load(state_base + (2 + 2 * step) * _HC + cols).to(tl.float32)
        grad_row = (grad - tl.sum(grad * matrix, axis=0)[None, :]) / col_denom
        row_matrix = matrix * col_denom
        row_denom = tl.load(state_base + (1 + 2 * step) * _HC + rows).to(tl.float32)
        grad = (grad_row - tl.sum(grad_row * row_matrix, axis=1)[:, None]) / row_denom
        matrix = row_matrix * row_denom

    initial_col_denom = tl.load(state_base + cols).to(tl.float32)
    grad_softmax = (grad - tl.sum(grad * matrix, axis=0)[None, :]) / initial_col_denom
    softmax = matrix * initial_col_denom - HC_EPS
    grad_logits = softmax * (
        grad_softmax - tl.sum(grad_softmax * softmax, axis=1)[:, None]
    )
    tl.store(
        grad_mix_ptr + token * _MIX + 2 * _HC + flat,
        grad_logits * comb_scale,
    )


@triton.jit
def _mhc_prepare_dx_kernel(
    x_ptr,
    fn_ptr,
    mix_ptr,
    invr_ptr,
    coeff_ptr,
    grad_mix_ptr,
    grad_collapsed_ptr,
    grad_residual_ptr,
    BLOCK_K: tl.constexpr,
    _HIDDEN: tl.constexpr,
    _FLAT: tl.constexpr,
    _MIX: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int32)
    mix16_offsets = tl.arange(0, 16)
    mix8_offsets = 16 + tl.arange(0, 8)
    grad_mix16 = tl.load(grad_mix_ptr + token * _MIX + mix16_offsets).to(tl.float32)
    grad_mix8 = tl.load(grad_mix_ptr + token * _MIX + mix8_offsets).to(tl.float32)
    mix16 = tl.load(mix_ptr + token * _MIX + mix16_offsets).to(tl.float32)
    mix8 = tl.load(mix_ptr + token * _MIX + mix8_offsets).to(tl.float32)
    norm_projection = tl.sum(grad_mix16 * mix16, axis=0) + tl.sum(
        grad_mix8 * mix8, axis=0
    )
    invr = tl.load(invr_ptr + token).to(tl.float32)
    norm_factor = invr * invr * norm_projection / _FLAT
    x_row = x_ptr + token * _FLAT
    grad_row = grad_residual_ptr + token * _FLAT

    for k_start in tl.range(0, _FLAT, BLOCK_K, loop_unroll_factor=1):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        x = tl.load(x_row + k_offsets).to(tl.float32)
        fn16 = tl.load(fn_ptr + mix16_offsets[:, None] * _FLAT + k_offsets[None, :]).to(
            tl.float32
        )
        fn8 = tl.load(fn_ptr + mix8_offsets[:, None] * _FLAT + k_offsets[None, :]).to(
            tl.float32
        )
        projected = tl.sum(grad_mix16[:, None] * fn16, axis=0) + tl.sum(
            grad_mix8[:, None] * fn8, axis=0
        )
        stream = k_offsets // _HIDDEN
        dim = k_offsets % _HIDDEN
        pre = tl.load(coeff_ptr + token * _MIX + stream).to(tl.float32)
        grad_collapsed = tl.load(grad_collapsed_ptr + token * _HIDDEN + dim).to(
            tl.float32
        )
        incoming = tl.load(grad_row + k_offsets).to(tl.float32)
        grad_x = incoming + pre * grad_collapsed + invr * projected - x * norm_factor
        tl.store(grad_row + k_offsets, grad_x.to(grad_residual_ptr.dtype.element_ty))


@triton.jit
def _mhc_merge_forward_kernel(
    residual_ptr,
    branch_ptr,
    coeff_ptr,
    output_ptr,
    BLOCK_D: tl.constexpr,
    _HC: tl.constexpr,
    _HIDDEN: tl.constexpr,
    _FLAT: tl.constexpr,
    _MIX: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int32)
    h = tl.arange(0, _HC)
    rows = tl.arange(0, _HC)[:, None]
    cols = tl.arange(0, _HC)[None, :]
    comb = tl.load(coeff_ptr + token * _MIX + 2 * _HC + rows * _HC + cols).to(
        tl.float32
    )
    post = tl.load(coeff_ptr + token * _MIX + _HC + h).to(tl.float32)
    residual_row = residual_ptr + token * _FLAT
    output_row = output_ptr + token * _FLAT

    for d_start in tl.range(0, _HIDDEN, BLOCK_D, loop_unroll_factor=1):
        d_offsets = d_start + tl.arange(0, BLOCK_D)
        streams = tl.load(residual_row + h[:, None] * _HIDDEN + d_offsets[None, :]).to(
            tl.float32
        )
        branch = tl.load(branch_ptr + token * _HIDDEN + d_offsets).to(tl.float32)
        output = post[:, None] * branch[None, :] + tl.sum(
            comb[:, :, None] * streams[:, None, :], axis=0
        )
        tl.store(
            output_row + h[:, None] * _HIDDEN + d_offsets[None, :],
            output.to(output_ptr.dtype.element_ty),
        )


@triton.jit
def _mhc_merge_backward_kernel(
    residual_ptr,
    branch_ptr,
    coeff_ptr,
    grad_output_ptr,
    grad_residual_ptr,
    grad_branch_ptr,
    grad_coeff_ptr,
    BLOCK_D: tl.constexpr,
    _HC: tl.constexpr,
    _HIDDEN: tl.constexpr,
    _FLAT: tl.constexpr,
    _MIX: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int32)
    h = tl.arange(0, _HC)
    rows = tl.arange(0, _HC)[:, None]
    cols = tl.arange(0, _HC)[None, :]
    comb = tl.load(coeff_ptr + token * _MIX + 2 * _HC + rows * _HC + cols).to(
        tl.float32
    )
    post = tl.load(coeff_ptr + token * _MIX + _HC + h).to(tl.float32)
    grad_post = tl.zeros((_HC,), tl.float32)
    grad_comb = tl.zeros((_HC, _HC), tl.float32)
    residual_row = residual_ptr + token * _FLAT
    grad_output_row = grad_output_ptr + token * _FLAT
    grad_residual_row = grad_residual_ptr + token * _FLAT

    for d_start in tl.range(0, _HIDDEN, BLOCK_D, loop_unroll_factor=1):
        d_offsets = d_start + tl.arange(0, BLOCK_D)
        residual = tl.load(residual_row + h[:, None] * _HIDDEN + d_offsets[None, :]).to(
            tl.float32
        )
        branch = tl.load(branch_ptr + token * _HIDDEN + d_offsets).to(tl.float32)
        grad_output = tl.load(
            grad_output_row + h[:, None] * _HIDDEN + d_offsets[None, :]
        ).to(tl.float32)

        grad_branch = tl.sum(post[:, None] * grad_output, axis=0)
        grad_residual = tl.sum(comb[:, :, None] * grad_output[None, :, :], axis=1)
        tl.store(
            grad_branch_ptr + token * _HIDDEN + d_offsets,
            grad_branch.to(grad_branch_ptr.dtype.element_ty),
        )
        tl.store(
            grad_residual_row + h[:, None] * _HIDDEN + d_offsets[None, :],
            grad_residual.to(grad_residual_ptr.dtype.element_ty),
        )
        grad_post += tl.sum(grad_output * branch[None, :], axis=1)
        grad_comb += tl.sum(residual[:, None, :] * grad_output[None, :, :], axis=2)

    tl.store(grad_coeff_ptr + token * _MIX + h, 0.0)
    tl.store(grad_coeff_ptr + token * _MIX + _HC + h, grad_post)
    tl.store(
        grad_coeff_ptr + token * _MIX + 2 * _HC + rows * _HC + cols,
        grad_comb,
    )


@triton.jit
def _mhc_head_forward_kernel(
    x_ptr,
    fn_ptr,
    base_ptr,
    scale_ptr,
    mix_ptr,
    invr_ptr,
    pre_ptr,
    output_ptr,
    RMS_EPS: tl.constexpr,
    HC_EPS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    _HC: tl.constexpr,
    _HIDDEN: tl.constexpr,
    _FLAT: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int32)
    h = tl.arange(0, _HC)
    projected = tl.zeros((_HC,), tl.float32)
    sumsq = tl.zeros((), tl.float32)
    x_row = x_ptr + token * _FLAT

    for k_start in tl.range(0, _FLAT, BLOCK_K, loop_unroll_factor=1):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        x = tl.load(x_row + k_offsets).to(tl.float32)
        fn = tl.load(fn_ptr + h[:, None] * _FLAT + k_offsets[None, :]).to(tl.float32)
        projected += tl.sum(fn * x[None, :], axis=1)
        sumsq += tl.sum(x * x, axis=0)

    invr = tl.rsqrt(sumsq / _FLAT + RMS_EPS)
    mix = projected * invr
    base = tl.load(base_ptr + h).to(tl.float32)
    scale = tl.load(scale_ptr).to(tl.float32)
    pre = tl.sigmoid(mix * scale + base) + HC_EPS
    tl.store(mix_ptr + token * _HC + h, mix)
    tl.store(invr_ptr + token, invr)
    tl.store(pre_ptr + token * _HC + h, pre)

    for d_start in tl.range(0, _HIDDEN, BLOCK_D, loop_unroll_factor=1):
        d_offsets = d_start + tl.arange(0, BLOCK_D)
        streams = tl.load(x_row + h[:, None] * _HIDDEN + d_offsets[None, :]).to(
            tl.float32
        )
        output = tl.sum(pre[:, None] * streams, axis=0)
        tl.store(
            output_ptr + token * _HIDDEN + d_offsets,
            output.to(output_ptr.dtype.element_ty),
        )


@triton.jit
def _mhc_head_backward_kernel(
    x_ptr,
    fn_ptr,
    scale_ptr,
    mix_ptr,
    invr_ptr,
    pre_ptr,
    grad_output_ptr,
    grad_x_ptr,
    HC_EPS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    _HC: tl.constexpr,
    _HIDDEN: tl.constexpr,
    _FLAT: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int32)
    h = tl.arange(0, _HC)
    x_row = x_ptr + token * _FLAT
    grad_pre = tl.zeros((_HC,), tl.float32)

    for d_start in tl.range(0, _HIDDEN, BLOCK_D, loop_unroll_factor=1):
        d_offsets = d_start + tl.arange(0, BLOCK_D)
        grad_output = tl.load(grad_output_ptr + token * _HIDDEN + d_offsets).to(
            tl.float32
        )
        streams = tl.load(x_row + h[:, None] * _HIDDEN + d_offsets[None, :]).to(
            tl.float32
        )
        grad_pre += tl.sum(streams * grad_output[None, :], axis=1)

    pre = tl.load(pre_ptr + token * _HC + h).to(tl.float32)
    sigmoid = pre - HC_EPS
    scale = tl.load(scale_ptr).to(tl.float32)
    grad_mix = grad_pre * sigmoid * (1.0 - sigmoid) * scale
    mix = tl.load(mix_ptr + token * _HC + h).to(tl.float32)
    invr = tl.load(invr_ptr + token).to(tl.float32)
    norm_factor = invr * invr * tl.sum(grad_mix * mix, axis=0) / _FLAT

    for k_start in tl.range(0, _FLAT, BLOCK_K, loop_unroll_factor=1):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        x = tl.load(x_row + k_offsets).to(tl.float32)
        fn = tl.load(fn_ptr + h[:, None] * _FLAT + k_offsets[None, :]).to(tl.float32)
        projected = tl.sum(grad_mix[:, None] * fn, axis=0)
        stream = k_offsets // _HIDDEN
        dim = k_offsets % _HIDDEN
        direct_pre = tl.load(pre_ptr + token * _HC + stream).to(tl.float32)
        grad_output = tl.load(grad_output_ptr + token * _HIDDEN + dim).to(tl.float32)
        grad_x = direct_pre * grad_output + invr * projected - x * norm_factor
        tl.store(
            grad_x_ptr + token * _FLAT + k_offsets,
            grad_x.to(grad_x_ptr.dtype.element_ty),
        )


def _f16_fn_cache(
    fn: torch.Tensor,
    shape: tuple[int, int],
    label: str,
) -> torch.Tensor:
    if fn.device.type != "cuda":
        raise RuntimeError(f"DeepSeek V4 {label} cache requires a CUDA/ROCm tensor")
    if tuple(fn.shape) != shape:
        raise ValueError(f"DeepSeek V4 {label} must have shape {shape}")
    if fn.requires_grad:
        raise RuntimeError(f"DeepSeek V4 {label} must be frozen")
    if fn.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise TypeError(f"DeepSeek V4 {label} has unsupported dtype {fn.dtype}")
    if fn.dtype == torch.float16 and fn.is_contiguous():
        return fn
    return fn.to(dtype=torch.float16, memory_format=torch.contiguous_format)


def deepseek_v4_mhc_fn_cache(fn: torch.Tensor) -> torch.Tensor:
    """Return the native-layout F16 layer projection consumed by the kernels."""

    return _f16_fn_cache(fn, (_MIX, _FLAT), "mHC fn")


def _validate_prepare_inputs(
    x: torch.Tensor,
    fn: torch.Tensor,
    base: torch.Tensor,
    scale: torch.Tensor,
) -> int:
    if x.device.type != "cuda":
        raise RuntimeError("DeepSeek V4 mHC requires CUDA/ROCm tensors")
    if x.dtype != torch.bfloat16:
        raise TypeError(f"DeepSeek V4 mHC requires BF16 activations, got {x.dtype}")
    if x.ndim < 3 or tuple(x.shape[-2:]) != (_HC, _HIDDEN):
        raise ValueError(
            f"DeepSeek V4 mHC requires [..., {_HC}, {_HIDDEN}], got {tuple(x.shape)}"
        )
    rows = x.numel() // _FLAT
    if rows not in _SUPPORTED_ROWS:
        raise ValueError(
            f"DeepSeek V4 mHC supports rows {sorted(_SUPPORTED_ROWS)}, got {rows}"
        )
    expected = (
        (fn, (_MIX, _FLAT), torch.float16, "fn"),
        (base, (_MIX,), torch.float32, "base"),
        (scale, (3,), torch.float32, "scale"),
    )
    for tensor, shape, dtype, name in expected:
        if tensor.device != x.device or tensor.dtype != dtype:
            raise TypeError(
                f"DeepSeek V4 mHC {name} has unsupported dtype/device "
                f"{tensor.dtype}/{tensor.device}"
            )
        if tuple(tensor.shape) != shape:
            raise ValueError(f"DeepSeek V4 mHC {name} must have shape {shape}")
        if tensor.requires_grad:
            raise RuntimeError(f"DeepSeek V4 mHC {name} must be frozen")
    if (
        not x.is_contiguous()
        or not fn.is_contiguous()
        or not base.is_contiguous()
        or not scale.is_contiguous()
    ):
        raise ValueError("DeepSeek V4 mHC inputs must be contiguous")
    return rows


def _validate_merge_inputs(
    residual: torch.Tensor,
    branch_output: torch.Tensor,
    coeff: torch.Tensor,
) -> int:
    if residual.device.type != "cuda" or residual.dtype != torch.bfloat16:
        raise TypeError("DeepSeek V4 mHC merge requires BF16 CUDA/ROCm residuals")
    if residual.ndim < 3 or tuple(residual.shape[-2:]) != (_HC, _HIDDEN):
        raise ValueError("DeepSeek V4 mHC merge received an unsupported residual shape")
    rows = residual.numel() // _FLAT
    if rows not in _SUPPORTED_ROWS:
        raise ValueError("DeepSeek V4 mHC merge received an unsupported row count")
    outer = residual.shape[:-2]
    if tuple(branch_output.shape) != (*outer, _HIDDEN):
        raise ValueError("DeepSeek V4 mHC branch output shape does not match residuals")
    if tuple(coeff.shape) != (*outer, _MIX):
        raise ValueError("DeepSeek V4 mHC coefficient shape does not match residuals")
    if branch_output.dtype != torch.bfloat16 or coeff.dtype != torch.float32:
        raise TypeError(
            "DeepSeek V4 mHC merge requires BF16 branch output and FP32 coefficients"
        )
    if branch_output.device != residual.device or coeff.device != residual.device:
        raise ValueError("DeepSeek V4 mHC merge tensors must share one device")
    if (
        not residual.is_contiguous()
        or not branch_output.is_contiguous()
        or not coeff.is_contiguous()
    ):
        raise ValueError("DeepSeek V4 mHC merge tensors must be contiguous")
    return rows


def _validate_head_inputs(
    x: torch.Tensor,
    fn: torch.Tensor,
    base: torch.Tensor,
    scale: torch.Tensor,
) -> int:
    if x.device.type != "cuda" or x.dtype != torch.bfloat16:
        raise TypeError("DeepSeek V4 mHC head requires BF16 CUDA/ROCm activations")
    if x.ndim < 3 or tuple(x.shape[-2:]) != (_HC, _HIDDEN):
        raise ValueError(
            "DeepSeek V4 mHC head received an unsupported activation shape"
        )
    rows = x.numel() // _FLAT
    if rows not in _SUPPORTED_ROWS:
        raise ValueError(
            f"DeepSeek V4 mHC head supports rows {sorted(_SUPPORTED_ROWS)}, got {rows}"
        )
    expected = (
        (fn, (_HC, _FLAT), torch.float16, "hc_fn"),
        (base, (_HC,), torch.float32, "hc_base"),
        (scale, (1,), torch.float32, "hc_scale"),
    )
    for tensor, shape, dtype, name in expected:
        if tensor.device != x.device or tensor.dtype != dtype:
            raise TypeError(
                f"DeepSeek V4 mHC head {name} has unsupported dtype/device "
                f"{tensor.dtype}/{tensor.device}"
            )
        if tuple(tensor.shape) != shape:
            raise ValueError(f"DeepSeek V4 mHC head {name} must have shape {shape}")
        if tensor.requires_grad:
            raise RuntimeError(f"DeepSeek V4 mHC head {name} must be frozen")
        if not tensor.is_contiguous():
            raise ValueError(f"DeepSeek V4 mHC head {name} must be contiguous")
    if not x.is_contiguous():
        raise ValueError("DeepSeek V4 mHC head activations must be contiguous")
    return rows


class _DeepseekV4MHCPrepareFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        fn: torch.Tensor,
        base: torch.Tensor,
        scale: torch.Tensor,
        rms_eps: float,
        hc_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows = _validate_prepare_inputs(x, fn, base, scale)
        outer = x.shape[:-2]
        mix = torch.empty((rows, _MIX), device=x.device, dtype=torch.float32)
        invr = torch.empty((rows,), device=x.device, dtype=torch.float32)
        coeff = torch.empty((rows, _MIX), device=x.device, dtype=torch.float32)
        sink_state = torch.empty(
            (rows, _SINKHORN_STATE_STEPS, _HC),
            device=x.device,
            dtype=torch.float16,
        )
        collapsed = torch.empty((rows, _HIDDEN), device=x.device, dtype=x.dtype)
        block_k, block_d, num_warps, num_stages = _PREPARE_FORWARD_CONFIGS[rows]
        _mhc_prepare_forward_kernel[(rows,)](
            x,
            fn,
            base,
            scale,
            mix,
            invr,
            coeff,
            sink_state,
            collapsed,
            RMS_EPS=float(rms_eps),
            HC_EPS=float(hc_eps),
            BLOCK_K=block_k,
            BLOCK_D=block_d,
            _HC=_HC,
            _HIDDEN=_HIDDEN,
            _FLAT=_FLAT,
            _MIX=_MIX,
            _SINKHORN_ITERS=_SINKHORN_ITERS,
            _SINKHORN_STATE_STEPS=_SINKHORN_STATE_STEPS,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        ctx.save_for_backward(x, fn, scale, mix, invr, coeff, sink_state)
        ctx.meta = (x.shape, float(hc_eps))
        # x is intentionally returned as an alias. Its gradient from merge is
        # the writable accumulation buffer used by prepare backward.
        return x, coeff.view(*outer, _MIX), collapsed.view(*outer, _HIDDEN)

    @staticmethod
    def backward(
        ctx: Any,
        grad_residual: torch.Tensor | None,
        grad_coeff: torch.Tensor | None,
        grad_collapsed: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, ...]:
        if torch.is_grad_enabled():
            raise RuntimeError(
                "DeepSeek V4 mHC does not support higher-order gradients"
            )
        x, fn, scale, mix, invr, coeff, sink_state = ctx.saved_tensors
        x_shape, hc_eps = ctx.meta
        rows = x.numel() // _FLAT
        if grad_residual is None:
            grad_residual = torch.zeros_like(x)
        elif not grad_residual.is_contiguous():
            grad_residual = grad_residual.contiguous()
        if grad_coeff is None:
            grad_coeff = torch.zeros_like(coeff)
        else:
            grad_coeff = grad_coeff.reshape(rows, _MIX)
            if not grad_coeff.is_contiguous():
                grad_coeff = grad_coeff.contiguous()
        if grad_collapsed is None:
            grad_collapsed = torch.zeros(
                (rows, _HIDDEN), device=x.device, dtype=x.dtype
            )
        else:
            grad_collapsed = grad_collapsed.reshape(rows, _HIDDEN)
            if not grad_collapsed.is_contiguous():
                grad_collapsed = grad_collapsed.contiguous()

        grad_mix = torch.empty_like(mix)
        controls_block_d, controls_warps, controls_stages = _CONTROLS_BACKWARD_CONFIGS[
            rows
        ]
        _mhc_controls_backward_kernel[(rows,)](
            x,
            mix,
            coeff,
            sink_state,
            scale,
            grad_coeff,
            grad_collapsed,
            grad_mix,
            HC_EPS=hc_eps,
            BLOCK_D=controls_block_d,
            _HC=_HC,
            _HIDDEN=_HIDDEN,
            _FLAT=_FLAT,
            _MIX=_MIX,
            _SINKHORN_ITERS=_SINKHORN_ITERS,
            _SINKHORN_STATE_STEPS=_SINKHORN_STATE_STEPS,
            num_warps=controls_warps,
            num_stages=controls_stages,
        )
        dx_block_k, dx_warps, dx_stages = _PREPARE_DX_CONFIGS[rows]
        _mhc_prepare_dx_kernel[(rows,)](
            x,
            fn,
            mix,
            invr,
            coeff,
            grad_mix,
            grad_collapsed,
            grad_residual,
            BLOCK_K=dx_block_k,
            _HIDDEN=_HIDDEN,
            _FLAT=_FLAT,
            _MIX=_MIX,
            num_warps=dx_warps,
            num_stages=dx_stages,
        )
        return grad_residual.view(x_shape), None, None, None, None, None


class _DeepseekV4MHCMergeFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        residual: torch.Tensor,
        branch_output: torch.Tensor,
        coeff: torch.Tensor,
    ) -> torch.Tensor:
        rows = _validate_merge_inputs(residual, branch_output, coeff)
        output = torch.empty_like(residual)
        block_d, num_warps, num_stages = _MERGE_FORWARD_CONFIGS[rows]
        _mhc_merge_forward_kernel[(rows,)](
            residual,
            branch_output,
            coeff,
            output,
            BLOCK_D=block_d,
            _HC=_HC,
            _HIDDEN=_HIDDEN,
            _FLAT=_FLAT,
            _MIX=_MIX,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        ctx.save_for_backward(residual, branch_output, coeff)
        return output

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if torch.is_grad_enabled():
            raise RuntimeError(
                "DeepSeek V4 mHC does not support higher-order gradients"
            )
        residual, branch_output, coeff = ctx.saved_tensors
        rows = residual.numel() // _FLAT
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        # The incoming cotangent is dead at the fixed DeepSeek layer boundary.
        # Reuse it for direct dResidual, then let prepare backward add the other
        # activation-gradient contributions into the same storage.
        grad_residual = grad_output
        grad_branch = torch.empty_like(branch_output)
        grad_coeff = torch.empty_like(coeff)
        block_d, num_warps, num_stages = _MERGE_BACKWARD_CONFIGS[rows]
        _mhc_merge_backward_kernel[(rows,)](
            residual,
            branch_output,
            coeff,
            grad_output,
            grad_residual,
            grad_branch,
            grad_coeff,
            BLOCK_D=block_d,
            _HC=_HC,
            _HIDDEN=_HIDDEN,
            _FLAT=_FLAT,
            _MIX=_MIX,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return grad_residual, grad_branch, grad_coeff


class _DeepseekV4MHCHeadFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        fn: torch.Tensor,
        base: torch.Tensor,
        scale: torch.Tensor,
        rms_eps: float,
        hc_eps: float,
    ) -> torch.Tensor:
        rows = _validate_head_inputs(x, fn, base, scale)
        outer = x.shape[:-2]
        mix = torch.empty((rows, _HC), device=x.device, dtype=torch.float32)
        invr = torch.empty((rows,), device=x.device, dtype=torch.float32)
        pre = torch.empty((rows, _HC), device=x.device, dtype=torch.float32)
        output = torch.empty((rows, _HIDDEN), device=x.device, dtype=x.dtype)
        block_k, block_d, num_warps, num_stages = _HEAD_FORWARD_CONFIGS[rows]
        _mhc_head_forward_kernel[(rows,)](
            x,
            fn,
            base,
            scale,
            mix,
            invr,
            pre,
            output,
            RMS_EPS=float(rms_eps),
            HC_EPS=float(hc_eps),
            BLOCK_K=block_k,
            BLOCK_D=block_d,
            _HC=_HC,
            _HIDDEN=_HIDDEN,
            _FLAT=_FLAT,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        ctx.save_for_backward(x, fn, scale, mix, invr, pre)
        ctx.meta = (x.shape, float(hc_eps))
        return output.view(*outer, _HIDDEN)

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, ...]:
        if torch.is_grad_enabled():
            raise RuntimeError(
                "DeepSeek V4 mHC head does not support higher-order gradients"
            )
        x, fn, scale, mix, invr, pre = ctx.saved_tensors
        x_shape, hc_eps = ctx.meta
        rows = x.numel() // _FLAT
        grad_output = grad_output.reshape(rows, _HIDDEN)
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        grad_x = torch.empty_like(x)
        block_k, block_d, num_warps, num_stages = _HEAD_BACKWARD_CONFIGS[rows]
        _mhc_head_backward_kernel[(rows,)](
            x,
            fn,
            scale,
            mix,
            invr,
            pre,
            grad_output,
            grad_x,
            HC_EPS=hc_eps,
            BLOCK_K=block_k,
            BLOCK_D=block_d,
            _HC=_HC,
            _HIDDEN=_HIDDEN,
            _FLAT=_FLAT,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return grad_x.view(x_shape), None, None, None, None, None


def deepseek_v4_mhc_prepare(
    hidden_streams: torch.Tensor,
    fn: torch.Tensor,
    base: torch.Tensor,
    scale: torch.Tensor,
    *,
    rms_eps: float = 1e-6,
    hc_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return residual alias, compact coefficients, and collapsed activations."""

    return _DeepseekV4MHCPrepareFunction.apply(
        hidden_streams,
        fn,
        base,
        scale,
        rms_eps,
        hc_eps,
    )


def deepseek_v4_mhc_merge(
    residual: torch.Tensor,
    branch_output: torch.Tensor,
    coefficients: torch.Tensor,
) -> torch.Tensor:
    """Apply the fixed four-stream post and correctly oriented residual merge."""

    return _DeepseekV4MHCMergeFunction.apply(
        residual,
        branch_output,
        coefficients,
    )


def deepseek_v4_mhc_head(
    hidden_streams: torch.Tensor,
    fn: torch.Tensor,
    base: torch.Tensor,
    scale: torch.Tensor,
    *,
    rms_eps: float = 1e-6,
    hc_eps: float = 1e-6,
) -> torch.Tensor:
    """Collapse the final four streams with frozen F16 mHC controls."""

    return _DeepseekV4MHCHeadFunction.apply(
        hidden_streams,
        fn,
        base,
        scale,
        rms_eps,
        hc_eps,
    )


def _patched_mhc_connection_forward(
    self: DeepseekV4HyperConnection,
    hidden_streams: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return deepseek_v4_mhc_prepare(
        hidden_streams,
        self.fn,
        self.base,
        self.scale,
        rms_eps=self.input_norm.eps,
        hc_eps=self.hc_eps,
    )


def _patched_mhc_decoder_forward(
    self: DeepseekV4DecoderLayer,
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    residual, coefficients, collapsed = self.attn_hc(hidden_states)
    attn_output, _ = self.self_attn(self.input_layernorm(collapsed), **kwargs)
    hidden_states = deepseek_v4_mhc_merge(
        residual,
        attn_output,
        coefficients,
    )

    residual, coefficients, collapsed = self.ffn_hc(hidden_states)
    mlp_output = self.mlp(
        self.post_attention_layernorm(collapsed),
        input_ids=input_ids,
    )
    return deepseek_v4_mhc_merge(
        residual,
        mlp_output,
        coefficients,
    )


def _patched_mhc_head_forward(
    self: DeepseekV4HyperHead,
    hidden_streams: torch.Tensor,
) -> torch.Tensor:
    return deepseek_v4_mhc_head(
        hidden_streams,
        self.hc_fn,
        self.hc_base,
        self.hc_scale,
        rms_eps=self.input_norm.eps,
        hc_eps=self.eps,
    )


def _install_f16_projection_parameter(
    module: torch.nn.Module,
    name: str,
    shape: tuple[int, int],
    label: str,
) -> bool:
    parameter = getattr(module, name)
    parameter.requires_grad_(False)
    cached = _f16_fn_cache(parameter, shape, label)
    if cached is parameter:
        return False
    setattr(module, name, torch.nn.Parameter(cached, requires_grad=False))
    return True


def _freeze_fp32_control(
    parameter: torch.nn.Parameter,
    shape: tuple[int, ...],
    label: str,
) -> None:
    parameter.requires_grad_(False)
    if parameter.dtype != torch.float32 or tuple(parameter.shape) != shape:
        raise TypeError(
            f"DeepSeek V4 {label} must be contiguous FP32 with shape {shape}, "
            f"got {parameter.dtype} {tuple(parameter.shape)}"
        )
    if parameter.device.type != "cuda" or not parameter.is_contiguous():
        raise ValueError(f"DeepSeek V4 {label} must be contiguous on CUDA/ROCm")


def configure_deepseek_v4_liger_mhc(
    model: torch.nn.Module,
) -> dict[str, Any]:
    """Install the fixed layer and final-head mHC boundaries on one model."""

    connections = 0
    decoder_layers = 0
    heads = 0
    f16_projections = 0
    converted_projections = 0
    frozen_controls = 0
    already_patched = 0
    patched_names: list[str] = []

    for name, module in model.named_modules():
        if isinstance(module, DeepseekV4HyperConnection):
            connections += 1
            if (
                module.hc_mult != _HC
                or module.hc_sinkhorn_iters != _SINKHORN_ITERS
                or module.fn.shape != (_MIX, _FLAT)
            ):
                raise RuntimeError(f"unsupported DeepSeek V4 mHC geometry at {name!r}")
            _freeze_fp32_control(module.base, (_MIX,), f"{name}.base")
            _freeze_fp32_control(module.scale, (3,), f"{name}.scale")
            frozen_controls += 3
            if _install_f16_projection_parameter(
                module, "fn", (_MIX, _FLAT), f"{name}.fn"
            ):
                converted_projections += 1
            f16_projections += int(module.fn.dtype == torch.float16)
            if getattr(module, _MHC_PATCH_MARKER, False):
                already_patched += 1
                continue
            module.forward = MethodType(_patched_mhc_connection_forward, module)
            setattr(module, _MHC_PATCH_MARKER, True)
            patched_names.append(name)
        elif isinstance(module, DeepseekV4DecoderLayer):
            decoder_layers += 1
            if getattr(module, _MHC_PATCH_MARKER, False):
                already_patched += 1
                continue
            module.forward = MethodType(_patched_mhc_decoder_forward, module)
            setattr(module, _MHC_PATCH_MARKER, True)
            patched_names.append(name)
        elif isinstance(module, DeepseekV4HyperHead):
            heads += 1
            if module.hc_mult != _HC or module.hc_fn.shape != (_HC, _FLAT):
                raise RuntimeError(
                    f"unsupported DeepSeek V4 final mHC geometry at {name!r}"
                )
            _freeze_fp32_control(module.hc_base, (_HC,), f"{name}.hc_base")
            _freeze_fp32_control(module.hc_scale, (1,), f"{name}.hc_scale")
            frozen_controls += 3
            if _install_f16_projection_parameter(
                module, "hc_fn", (_HC, _FLAT), f"{name}.hc_fn"
            ):
                converted_projections += 1
            f16_projections += int(module.hc_fn.dtype == torch.float16)
            if getattr(module, _MHC_PATCH_MARKER, False):
                already_patched += 1
                continue
            module.forward = MethodType(_patched_mhc_head_forward, module)
            setattr(module, _MHC_PATCH_MARKER, True)
            patched_names.append(name)

    return {
        "connections": connections,
        "decoder_layers": decoder_layers,
        "heads": heads,
        "f16_projection_parameters": f16_projections,
        "converted_projection_parameters": converted_projections,
        "frozen_control_parameters": frozen_controls,
        "patched": len(patched_names),
        "already_patched": already_patched,
        "patched_names": patched_names,
    }


def require_complete_deepseek_v4_liger_mhc(report: dict[str, Any]) -> None:
    """Fail closed unless every fixed DeepSeek V4 mHC site was installed."""

    expected = {
        "connections": EXPECTED_DEEPSEEK_V4_MHC_CONNECTIONS,
        "decoder_layers": EXPECTED_DEEPSEEK_V4_MHC_DECODER_LAYERS,
        "heads": EXPECTED_DEEPSEEK_V4_MHC_HEADS,
        "f16_projection_parameters": (
            EXPECTED_DEEPSEEK_V4_MHC_CONNECTIONS + EXPECTED_DEEPSEEK_V4_MHC_HEADS
        ),
        "frozen_control_parameters": 3
        * (EXPECTED_DEEPSEEK_V4_MHC_CONNECTIONS + EXPECTED_DEEPSEEK_V4_MHC_HEADS),
    }
    mismatches = {
        key: (value, report.get(key))
        for key, value in expected.items()
        if report.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"incomplete DeepSeek V4 mHC configuration: {mismatches}")
