"""One fused preparation kernel for the GatedDeltaNet forward, and its backward.

The reference path runs, per layer and per forward pass: `in_proj_qkv` writes a `[B, S, 10240]`
tensor, a transpose copies it to channel-last for the depthwise conv, `causal_conv1d_fn` convolves
and applies SiLU, a second transpose copies the result back, `torch.split` and four `reshape`s cut it
into query, key and value, `repeat` materializes the key-to-value head broadcast, `torch.sigmoid`
and the `softplus` gate chain write two more tensors, and FLA's `l2norm_fwd` normalizes query and key
over the expanded 48 heads. That is three passes over the 20 MB mixed tensor, four elementwise
kernels, two head-expanding copies and two normalization kernels, all of it memory-bound work that
one kernel can do in a single pass.

The kernel here takes the projection's output in the token-major layout the projection already
produces and writes exactly what the chunked GatedDeltaNet core consumes:
- `q`, `k`: `[B, S, HV, D]` contiguous, L2-normalized, expanded to the value-head count in the tiled
  order (`head i` carries key head `i % H`), which is the order `out_proj`'s packed rows want.
- `v`: `[B, S, HV, D]` contiguous, convolved and SiLU'd.
- `g`: `[B, S, HV]` fp32 log-space decay, `-exp(A_log) * softplus(a + dt_bias)`.
- `beta`: `[B, S, HV]` `sigmoid(b)`.

The grid is one program per (channel group, token block, batch), where a channel group is one query
head, one key head or one value head. The convolution taps are gathered with four shifted loads rather
than a shifted register tile, which keeps the reads in L1 and the register tile small.

The backward is split into four kernels so that each keeps a fixed ownership order and the sequence
transposed convolution never needs a cross-program halo: `bwd` recomputes the pre-activation
convolution, applies the SiLU and row-normalization gradients, writes `dz` and accumulates the
per-block convolution-weight partials. `gate` reverses the sigmoid and softplus chain and accumulates
the gate parameter partials. `dmixed` runs the transposed convolution over `dz`. `reduce` sums the
partials. Nothing is stored that the reference would not store, and no stage uses atomics.

`fused_prep` reproduces the reference chain to within fp32 summation order: the convolution
accumulates in fp32, the SiLU result is rounded to BF16 before the row sum of squares (exactly as the
reference does when it stores the conv output), and the row normalization is the same
`rsqrt(sum(x^2) + eps)` with `eps = 1e-6`.

The module takes plain tensors so it can be tested against the reference chain without a model, and
the wiring at the end of the file patches the layer forward through the shared module-patching
protocol: `configure_fused_preparation(model)` installs it and `require_fused_preparation(report, ...)`
is the inventory gate.
"""

import sys
from collections.abc import Mapping
from typing import Any, cast

import torch
import triton
import triton.language as tl
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeGatedDeltaNet
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextGatedDeltaNet
from triton.language.extra import libdevice

from gdn_tiled_value_heads import record_tiled_broadcast
from module_patching import (
    ModulePatchSpec,
    patch_module_forwards,
    require_complete_inventory,
)

# Token rows per program and rows per loop iteration. Chosen by a block-size sweep, which the plan
# records: the differences between neighbouring settings are small because both directions are
# bandwidth-bound.
BLOCK_T = 128
BLOCK_T_LOAD = 4
# Channels per program in the transposed-convolution kernel.
BLOCK_C = 256
# Channels per program in the partial reduction. Small enough to fill the device: the reduction
# reads 5 MB of partials, so it wants many programs rather than wide ones.
BLOCK_REDUCE = 64
# The dtype of the `dz` intermediate between the two backward kernels. It is the single largest buffer
# of the backward (the full `[B, S, C]` tensor), so its width sets the traffic of both kernels.
DZ_DTYPE = torch.bfloat16
# Convolution taps.
TAPS = 4
L2_EPS = 1e-6


@triton.jit
def _silu(x):
    return x * tl.sigmoid(x)


@triton.jit
def _row_mask(row, T, MASK_T: tl.constexpr):
    """Row validity for a token index: never before the start, and past the end only when the row
    count is not a multiple of the token block."""
    if MASK_T:
        return (row >= 0) & (row < T)
    return row >= 0


@triton.jit
def _softplus(x):
    # PyTorch's default: beta = 1, threshold = 20.
    return tl.where(x > 20.0, x, libdevice.log1p(tl.exp(x)))


@triton.jit
def _prep_fwd_kernel(
    mixed_ptr,
    weight_ptr,
    a_ptr,
    b_ptr,
    alog_ptr,
    dtb_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    T,
    C,
    D: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BTL: tl.constexpr,
    EPS: tl.constexpr,
    MASK_T: tl.constexpr,
):
    """Convolve, activate, normalize and lay out one token block of one channel group.

    Group `gid < H` is query head `gid`, `H <= gid < 2H` is key head `gid - H`, and `gid >= 2H` is
    value head `gid - 2H`. The query and key programs write all `G` broadcast copies. The query
    programs also write the gate and beta of the value heads they pair with.
    """
    gid = tl.program_id(0)
    pid_t = tl.program_id(1)
    batch = tl.program_id(2).to(tl.int64)

    if gid < H:
        kind = 0
        head = gid
        channel = head * D
    elif gid < 2 * H:
        kind = 1
        head = gid - H
        channel = (H + head) * D
    else:
        kind = 2
        head = gid - 2 * H
        channel = (2 * H + head) * D

    cols = tl.arange(0, D)
    mask_col = cols < D
    t0 = pid_t * BT
    mixed = mixed_ptr + batch * T * C

    # The depthwise convolution weights for this group's channels, one vector per tap.
    w0 = tl.load(weight_ptr + (channel + cols) * K + 0, mask=mask_col, other=0.0)
    w1 = tl.load(weight_ptr + (channel + cols) * K + 1, mask=mask_col, other=0.0)
    w2 = tl.load(weight_ptr + (channel + cols) * K + 2, mask=mask_col, other=0.0)
    w3 = tl.load(weight_ptr + (channel + cols) * K + 3, mask=mask_col, other=0.0)

    if kind == 0:
        out_row = q_ptr + batch * T * HV * D
    elif kind == 1:
        out_row = k_ptr + batch * T * HV * D
    else:
        out_row = v_ptr + batch * T * HV * D

    for it in range(BT // BTL):
        rows = t0 + it * BTL + tl.arange(0, BTL)
        mask_row = _row_mask(rows, T, MASK_T)
        z = tl.zeros([BTL, D], dtype=tl.float32)
        # Tap j reads token rows - 3 + j, so the four loads overlap by three rows.
        p0 = rows - 3
        z += (
            tl.load(
                mixed + p0[:, None] * C + (channel + cols)[None, :],
                mask=_row_mask(p0, T, MASK_T)[:, None] & mask_col[None, :],
                other=0.0,
            ).to(tl.float32)
            * w0[None, :]
        )
        p1 = rows - 2
        z += (
            tl.load(
                mixed + p1[:, None] * C + (channel + cols)[None, :],
                mask=_row_mask(p1, T, MASK_T)[:, None] & mask_col[None, :],
                other=0.0,
            ).to(tl.float32)
            * w1[None, :]
        )
        p2 = rows - 1
        z += (
            tl.load(
                mixed + p2[:, None] * C + (channel + cols)[None, :],
                mask=_row_mask(p2, T, MASK_T)[:, None] & mask_col[None, :],
                other=0.0,
            ).to(tl.float32)
            * w2[None, :]
        )
        z += (
            tl.load(
                mixed + rows[:, None] * C + (channel + cols)[None, :],
                mask=mask_row[:, None] & mask_col[None, :],
                other=0.0,
            ).to(tl.float32)
            * w3[None, :]
        )
        x = _silu(z)
        store_mask = mask_row[:, None] & mask_col[None, :]
        if kind == 2:
            tl.store(
                out_row + rows[:, None] * (HV * D) + head * D + cols[None, :],
                x.to(v_ptr.dtype.element_ty),
                mask=store_mask,
            )
        else:
            # The reference stores the conv output in BF16, so the sum of squares sees BF16 values.
            xb = x.to(tl.bfloat16).to(tl.float32)
            rstd = 1.0 / tl.sqrt(tl.sum(xb * xb, axis=1) + EPS)
            y = (xb * rstd[:, None]).to(tl.bfloat16)
            for copy in range(G):
                tl.store(
                    out_row
                    + rows[:, None] * (HV * D)
                    + (copy * H + head) * D
                    + cols[None, :],
                    y,
                    mask=store_mask,
                )
            if kind == 0:
                for copy in range(G):
                    ih = copy * H + head
                    p = batch * T * HV + rows * HV + ih
                    alog = tl.load(alog_ptr + ih).to(tl.float32)
                    dtb = tl.load(dtb_ptr + ih).to(tl.float32)
                    av = tl.load(a_ptr + p, mask=mask_row, other=0.0).to(tl.float32)
                    bv = tl.load(b_ptr + p, mask=mask_row, other=0.0).to(tl.float32)
                    tl.store(
                        g_ptr + p, -tl.exp(alog) * _softplus(av + dtb), mask=mask_row
                    )
                    tl.store(
                        beta_ptr + p,
                        tl.sigmoid(bv).to(beta_ptr.dtype.element_ty),
                        mask=mask_row,
                    )


@triton.jit
def _prep_bwd_kernel(
    mixed_ptr,
    weight_ptr,
    dq_ptr,
    dk_ptr,
    dv_ptr,
    dz_ptr,
    dw_partial_ptr,
    T,
    C,
    NT,
    D: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BTL: tl.constexpr,
    EPS: tl.constexpr,
    MASK_T: tl.constexpr,
):
    """Reverse everything up to the depthwise convolution, and accumulate its weight partials.

    The row-normalization backward is `dy * rstd - (dy . y) * y * rstd` with `y` the normalized row,
    which is FLA's own formula, applied once per key head over the sum of the `G` broadcast copies.
    """
    gid = tl.program_id(0)
    pid_t = tl.program_id(1)
    batch = tl.program_id(2).to(tl.int64)

    if gid < H:
        kind = 0
        head = gid
        channel = head * D
    elif gid < 2 * H:
        kind = 1
        head = gid - H
        channel = (H + head) * D
    else:
        kind = 2
        head = gid - 2 * H
        channel = (2 * H + head) * D

    cols = tl.arange(0, D)
    mask_col = cols < D
    t0 = pid_t * BT
    mixed = mixed_ptr + batch * T * C
    base = batch * T * HV * D

    w0 = tl.load(weight_ptr + (channel + cols) * K + 0, mask=mask_col, other=0.0)
    w1 = tl.load(weight_ptr + (channel + cols) * K + 1, mask=mask_col, other=0.0)
    w2 = tl.load(weight_ptr + (channel + cols) * K + 2, mask=mask_col, other=0.0)
    w3 = tl.load(weight_ptr + (channel + cols) * K + 3, mask=mask_col, other=0.0)

    if kind == 0:
        dsrc = dq_ptr + base
    elif kind == 1:
        dsrc = dk_ptr + base
    else:
        dsrc = dv_ptr + base

    dw0 = tl.zeros([D], dtype=tl.float32)
    dw1 = tl.zeros([D], dtype=tl.float32)
    dw2 = tl.zeros([D], dtype=tl.float32)
    dw3 = tl.zeros([D], dtype=tl.float32)
    dz_out = dz_ptr + batch * T * C

    for it in range(BT // BTL):
        rows = t0 + it * BTL + tl.arange(0, BTL)
        mask_row = _row_mask(rows, T, MASK_T)
        mask_tile = mask_row[:, None] & mask_col[None, :]
        # The pre-activation convolution, from the same four shifted loads the forward made.
        z = tl.zeros([BTL, D], dtype=tl.float32)
        p0 = rows - 3
        x0 = tl.load(
            mixed + p0[:, None] * C + (channel + cols)[None, :],
            mask=_row_mask(p0, T, MASK_T)[:, None] & mask_col[None, :],
            other=0.0,
        )
        z += x0.to(tl.float32) * w0[None, :]
        p1 = rows - 2
        x1 = tl.load(
            mixed + p1[:, None] * C + (channel + cols)[None, :],
            mask=_row_mask(p1, T, MASK_T)[:, None] & mask_col[None, :],
            other=0.0,
        )
        z += x1.to(tl.float32) * w1[None, :]
        p2 = rows - 1
        x2 = tl.load(
            mixed + p2[:, None] * C + (channel + cols)[None, :],
            mask=_row_mask(p2, T, MASK_T)[:, None] & mask_col[None, :],
            other=0.0,
        )
        z += x2.to(tl.float32) * w2[None, :]
        x3 = tl.load(
            mixed + rows[:, None] * C + (channel + cols)[None, :],
            mask=mask_tile,
            other=0.0,
        )
        z += x3.to(tl.float32) * w3[None, :]
        sig = tl.sigmoid(z)

        if kind == 2:
            dx = tl.load(
                dsrc + head * D + rows[:, None] * (HV * D) + cols[None, :],
                mask=mask_tile,
                other=0.0,
            ).to(tl.float32)
        else:
            # The row normalization was applied to one key head and then copied to the G value heads,
            # so the incoming gradient is the sum over the copies while `y` is a single row. `x` and
            # `rstd` are recomputed from the convolution the way the forward rounded them, which is
            # exactly the pair FLA's `l2norm` saves, so no normalization state has to be stored.
            dy = tl.zeros([BTL, D], dtype=tl.float32)
            for copy in range(G):
                off = rows[:, None] * (HV * D) + (copy * H + head) * D + cols[None, :]
                dy += tl.load(dsrc + off, mask=mask_tile, other=0.0).to(tl.float32)
            xb = _silu(z).to(tl.bfloat16).to(tl.float32)
            rstd = 1.0 / tl.sqrt(tl.sum(xb * xb, axis=1) + EPS)
            yy = (xb * rstd[:, None]).to(tl.bfloat16).to(tl.float32)
            dx = (
                dy * rstd[:, None]
                - tl.sum(dy * yy, axis=1)[:, None] * yy * rstd[:, None]
            )

        dz = dx * sig * (1.0 + z * (1.0 - sig))
        tl.store(
            dz_out + rows[:, None] * C + (channel + cols)[None, :],
            dz.to(dz_out.dtype.element_ty),
            mask=mask_tile,
        )
        dw0 += tl.sum(dz * x0.to(tl.float32), axis=0)
        dw1 += tl.sum(dz * x1.to(tl.float32), axis=0)
        dw2 += tl.sum(dz * x2.to(tl.float32), axis=0)
        dw3 += tl.sum(dz * x3.to(tl.float32), axis=0)

    block = pid_t + batch.to(tl.int32) * NT
    obase = block * C * K + (channel + cols) * K
    tl.store(dw_partial_ptr + obase, dw0, mask=mask_col)
    tl.store(dw_partial_ptr + obase + 1, dw1, mask=mask_col)
    tl.store(dw_partial_ptr + obase + 2, dw2, mask=mask_col)
    tl.store(dw_partial_ptr + obase + 3, dw3, mask=mask_col)


@triton.jit
def _prep_gate_bwd_kernel(
    a_ptr,
    dtb_ptr,
    alog_ptr,
    g_ptr,
    beta_ptr,
    dg_ptr,
    dbeta_ptr,
    da_ptr,
    db_ptr,
    dalog_partial_ptr,
    ddtb_partial_ptr,
    T,
    HV,
    HVP: tl.constexpr,
    BT: tl.constexpr,
    MASK_T: tl.constexpr,
):
    """Gradients of `g = -exp(A_log) * softplus(a + dt_bias)` and `beta = sigmoid(b)`.

    One program per (token block, batch) covers all value heads, so each head's parameter partial is
    reduced inside the program and the blocks are summed afterwards.
    """
    pid_t = tl.program_id(0)
    batch = tl.program_id(1).to(tl.int64)
    heads = tl.arange(0, HVP)
    mask_head = heads < HV
    rows = pid_t * BT + tl.arange(0, BT)
    mask_row = _row_mask(rows, T, MASK_T)
    mask = mask_row[:, None] & mask_head[None, :]
    offset = batch * T * HV + rows[:, None] * HV + heads[None, :]

    alog = tl.load(alog_ptr + heads, mask=mask_head, other=0.0).to(tl.float32)
    dtb = tl.load(dtb_ptr + heads, mask=mask_head, other=0.0).to(tl.float32)
    av = tl.load(a_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    bv = tl.load(beta_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    dgv = tl.load(dg_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    dbv = tl.load(dbeta_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    gv = tl.load(g_ptr + offset, mask=mask, other=0.0).to(tl.float32)

    # d(softplus)/d(x) = sigmoid(x), and d(g)/d(A_log) = g.
    d_sp = dgv * -tl.exp(alog[None, :])
    da = d_sp * tl.sigmoid(av + dtb[None, :])
    tl.store(da_ptr + offset, da.to(da_ptr.dtype.element_ty), mask=mask)
    tl.store(
        db_ptr + offset, (dbv * bv * (1.0 - bv)).to(db_ptr.dtype.element_ty), mask=mask
    )
    zero = tl.zeros([BT, HVP], dtype=tl.float32)
    slot = batch.to(tl.int32) * tl.num_programs(0) + pid_t
    tl.store(
        dalog_partial_ptr + slot * HV + heads,
        tl.sum(tl.where(mask, dgv * gv, zero), axis=0),
        mask=mask_head,
    )
    tl.store(
        ddtb_partial_ptr + slot * HV + heads,
        tl.sum(tl.where(mask, d_sp * tl.sigmoid(av + dtb[None, :]), zero), axis=0),
        mask=mask_head,
    )


@triton.jit
def _prep_dmixed_kernel(
    dz_ptr,
    weight_ptr,
    dmixed_ptr,
    T,
    C,
    K: tl.constexpr,
    BC: tl.constexpr,
    BT: tl.constexpr,
    BTL: tl.constexpr,
    MASK_T: tl.constexpr,
):
    """Transposed depthwise convolution over `dz`, in token-major layout.

    Forward token `t` reads input rows `t - 3 .. t`, so input row `s` receives `dz[s + 3 - j] * w[j]`.
    """
    pid_c = tl.program_id(0)
    pid_t = tl.program_id(1)
    batch = tl.program_id(2).to(tl.int64)
    cols = pid_c * BC + tl.arange(0, BC)
    mask_col = cols < C
    t0 = pid_t * BT
    dz = dz_ptr + batch * T * C
    dmixed = dmixed_ptr + batch * T * C
    for it in range(BT // BTL):
        rows = t0 + it * BTL + tl.arange(0, BTL)
        mask_row = _row_mask(rows, T, MASK_T)
        out = tl.zeros([BTL, BC], dtype=tl.float32)
        for j in range(K):
            source = rows + (K - 1) - j
            # This side reads ahead of the row, so it always has to be bounded by the sequence: the
            # `dz` buffer has exactly T rows and the last block reaches three rows past its own.
            mask_source = (source >= 0) & (source < T)
            weight = tl.load(weight_ptr + cols * K + j, mask=mask_col, other=0.0)
            x = tl.load(
                dz + source[:, None] * C + cols[None, :],
                mask=mask_source[:, None] & mask_col[None, :],
                other=0.0,
            )
            out += x.to(tl.float32) * weight[None, :]
        tl.store(
            dmixed + rows[:, None] * C + cols[None, :],
            out.to(dmixed.dtype.element_ty),
            mask=mask_row[:, None] & mask_col[None, :],
        )


@triton.jit
def _prep_reduce_kernel(
    dw_partial_ptr,
    dalog_partial_ptr,
    ddtb_partial_ptr,
    dw_ptr,
    dalog_ptr,
    ddtb_ptr,
    num_blocks,
    C,
    HV,
    K: tl.constexpr,
    BC: tl.constexpr,
    NB: tl.constexpr,
    HVP: tl.constexpr,
):
    """Sum the per-block partials of the convolution weight and the two gate parameters."""
    pid = tl.program_id(0)
    blocks = tl.arange(0, NB)
    mask_block = blocks < num_blocks
    if pid == 0:
        heads = tl.arange(0, HVP)
        mask_head = heads < HV
        dalog = tl.sum(
            tl.load(
                dalog_partial_ptr + blocks[:, None] * HV + heads[None, :],
                mask=mask_block[:, None] & mask_head[None, :],
                other=0.0,
            ),
            axis=0,
        )
        tl.store(dalog_ptr + heads, dalog, mask=mask_head)
        ddtb = tl.sum(
            tl.load(
                ddtb_partial_ptr + blocks[:, None] * HV + heads[None, :],
                mask=mask_block[:, None] & mask_head[None, :],
                other=0.0,
            ),
            axis=0,
        )
        tl.store(ddtb_ptr + heads, ddtb, mask=mask_head)
        return
    cols = (pid - 1) * BC + tl.arange(0, BC)
    mask_col = cols < C
    for j in range(K):
        partial = tl.load(
            dw_partial_ptr + blocks[:, None] * C * K + (cols * K + j)[None, :],
            mask=mask_block[:, None] & mask_col[None, :],
            other=0.0,
        )
        tl.store(dw_ptr + cols * K + j, tl.sum(partial, axis=0), mask=mask_col)


def _power_of_two(value: int) -> int:
    return max(1, 1 << (value - 1).bit_length()) if value > 0 else 1


def fused_prep(
    mixed: torch.Tensor,
    conv_weight: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    key_heads: int,
    value_heads: int,
    head_dim: int,
    eps: float = L2_EPS,
    block_t: int = BLOCK_T,
    block_t_load: int = BLOCK_T_LOAD,
) -> tuple[torch.Tensor, ...]:
    """Run the fused preparation. Returns `(q, k, v, g, beta)`.

    `q` and `k` come out L2-normalized and expanded to `value_heads` in tiled order, so the caller can
    disable `use_qk_l2norm_in_kernel` and hand them straight to the chunked core.
    """
    return _GdnPrepFunction.apply(
        mixed,
        conv_weight,
        a,
        b,
        a_log,
        dt_bias,
        key_heads,
        value_heads,
        head_dim,
        float(eps),
        int(block_t),
        int(block_t_load),
    )


class _GdnPrepFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        mixed: torch.Tensor,
        conv_weight: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        key_heads: int,
        value_heads: int,
        head_dim: int,
        eps: float,
        block_t: int,
        block_t_load: int,
    ):
        batch, sequence, channels = mixed.shape
        expected = key_heads * 2 * head_dim + value_heads * head_dim
        if channels != expected:
            raise ValueError(f"mixed has {channels} channels, expected {expected}")
        if not mixed.is_contiguous():
            raise ValueError("mixed must be contiguous")
        if conv_weight.shape != (channels, TAPS):
            raise ValueError(
                f"conv_weight must be [{channels}, {TAPS}], got {tuple(conv_weight.shape)}"
            )
        for name, tensor, size in (
            ("a", a, value_heads),
            ("b", b, value_heads),
            ("A_log", a_log, value_heads),
            ("dt_bias", dt_bias, value_heads),
        ):
            if tensor.numel() != batch * sequence * size and tensor.numel() != size:
                raise ValueError(
                    f"{name} has {tensor.numel()} entries, expected {size} or {batch * sequence * size}"
                )
        if value_heads % key_heads:
            raise ValueError(
                f"value heads ({value_heads}) must be a multiple of key heads ({key_heads})"
            )
        if block_t % block_t_load:
            raise ValueError(
                f"token block {block_t} must be a multiple of the load block {block_t_load}"
            )

        device = mixed.device
        conv_weight = conv_weight.contiguous()
        shape = (batch, sequence, value_heads, head_dim)
        q = torch.empty(shape, dtype=mixed.dtype, device=device)
        k = torch.empty(shape, dtype=mixed.dtype, device=device)
        v = torch.empty(shape, dtype=mixed.dtype, device=device)
        g = torch.empty(
            (batch, sequence, value_heads), dtype=torch.float32, device=device
        )
        beta = torch.empty(
            (batch, sequence, value_heads), dtype=mixed.dtype, device=device
        )
        groups = 2 * key_heads + value_heads
        num_blocks = triton.cdiv(sequence, block_t)
        _prep_fwd_kernel[(groups, num_blocks, batch)](
            mixed,
            conv_weight,
            a,
            b,
            a_log,
            dt_bias,
            q,
            k,
            v,
            g,
            beta,
            sequence,
            channels,
            D=head_dim,
            H=key_heads,
            HV=value_heads,
            G=value_heads // key_heads,
            K=TAPS,
            BT=block_t,
            BTL=block_t_load,
            EPS=eps,
            MASK_T=bool(sequence % block_t),
            num_warps=4,
        )
        # Only the inputs and `beta` are needed: the row normalization state is recomputed in the
        # backward from the same rounding the forward used, so `q`, `k` and `v` are not held alive.
        ctx.save_for_backward(mixed, conv_weight, a, b, a_log, dt_bias, beta)
        ctx.dims = (key_heads, value_heads, head_dim, eps, block_t, block_t_load)
        ctx.g = g
        return q, k, v, g, beta

    @staticmethod
    def backward(  # ty: ignore[invalid-method-override]  (autograd's base takes *grad_outputs)
        ctx,
        dq: torch.Tensor,
        dk: torch.Tensor,
        dv: torch.Tensor,
        dg: torch.Tensor,
        dbeta: torch.Tensor,
    ):
        mixed, conv_weight, a, b, a_log, dt_bias, beta = ctx.saved_tensors
        key_heads, value_heads, head_dim, eps, block_t, block_t_load = ctx.dims
        batch, sequence, channels = mixed.shape
        # The kernels index every gradient row-major, so an expanded or strided incoming gradient would
        # silently read the wrong elements. Autograd produces them for these outputs. Make it explicit
        # and cheap rather than trusting the layout.
        dq, dk, dv, dg, dbeta = (
            tensor if tensor.is_contiguous() else tensor.contiguous()
            for tensor in (dq, dk, dv, dg, dbeta)
        )
        device = mixed.device
        groups = 2 * key_heads + value_heads
        num_blocks = triton.cdiv(sequence, block_t)
        mask_t = bool(sequence % block_t)
        group = value_heads // key_heads

        dz = torch.empty(mixed.shape, dtype=DZ_DTYPE, device=device)
        d_mixed = torch.empty(mixed.shape, dtype=mixed.dtype, device=device)
        d_conv_weight = torch.empty_like(conv_weight)
        dw_partial = torch.empty(
            (num_blocks * batch, channels, TAPS), dtype=torch.float32, device=device
        )
        da = torch.empty_like(a)
        db = torch.empty_like(b)
        d_a_log = torch.empty(value_heads, dtype=torch.float32, device=device)
        d_dt_bias = torch.empty(value_heads, dtype=torch.float32, device=device)
        gate_blocks = batch * num_blocks
        dalog_partial = torch.empty(
            (gate_blocks, value_heads), dtype=torch.float32, device=device
        )
        ddtb_partial = torch.empty(
            (gate_blocks, value_heads), dtype=torch.float32, device=device
        )

        _prep_bwd_kernel[(groups, num_blocks, batch)](
            mixed,
            conv_weight,
            dq,
            dk,
            dv,
            dz,
            dw_partial,
            sequence,
            channels,
            num_blocks,
            D=head_dim,
            H=key_heads,
            HV=value_heads,
            G=group,
            K=TAPS,
            BT=block_t,
            BTL=block_t_load,
            EPS=eps,
            MASK_T=mask_t,
            num_warps=4,
        )
        _prep_gate_bwd_kernel[(triton.cdiv(sequence, block_t), batch)](
            a,
            dt_bias,
            a_log,
            _g_output(ctx),
            beta,
            dg,
            dbeta,
            da,
            db,
            dalog_partial,
            ddtb_partial,
            sequence,
            value_heads,
            HVP=_power_of_two(value_heads),
            BT=block_t,
            MASK_T=mask_t,
            num_warps=4,
        )
        _prep_dmixed_kernel[(triton.cdiv(channels, BLOCK_C), num_blocks, batch)](
            dz,
            conv_weight,
            d_mixed,
            sequence,
            channels,
            K=TAPS,
            BC=BLOCK_C,
            BT=block_t,
            BTL=block_t_load,
            MASK_T=mask_t,
            num_warps=4,
        )
        _prep_reduce_kernel[(1 + triton.cdiv(channels, BLOCK_REDUCE),)](
            dw_partial,
            dalog_partial,
            ddtb_partial,
            d_conv_weight,
            d_a_log,
            d_dt_bias,
            num_blocks * batch,
            channels,
            value_heads,
            K=TAPS,
            BC=BLOCK_REDUCE,
            NB=_power_of_two(num_blocks * batch),
            HVP=_power_of_two(value_heads),
            num_warps=4,
        )
        return (
            d_mixed,
            d_conv_weight.to(conv_weight.dtype),
            da,
            db,
            d_a_log.to(a_log.dtype),
            d_dt_bias.to(dt_bias.dtype),
            None,
            None,
            None,
            None,
            None,
            None,
        )


def _g_output(ctx) -> torch.Tensor:
    """The fp32 decay `g`, an output of the forward and an input of the gate backward."""
    return ctx.g


def report() -> dict[str, Any]:
    """Sizes the kernels are compiled for, for a gate or a log line to record."""
    return {
        "block_t": BLOCK_T,
        "block_t_load": BLOCK_T_LOAD,
        "block_c": BLOCK_C,
        "block_reduce": BLOCK_REDUCE,
        "dz_dtype": str(DZ_DTYPE),
        "taps": TAPS,
    }


UNPATCHED_FORWARD = "_gdn_fused_prep_unpatched_forward"
MARKER = "_patched_fused_prep"
SUBJECT = "GatedDeltaNet fused preparation"
CHUNK_RULE_NAME = "torch_chunk_gated_delta_rule"
SKIPPED_KEY = "skipped_geometry"
HANDLED_KEY = "gdn_layers"

# The same pair of layer types `gdn_tiled_value_heads` owns: the Qwen4-Exp recurrent layer and the
# Qwen3.5/3.6 MoE one, which share the forward this patch replaces.
_GDN_LAYER_TYPES: tuple[type[torch.nn.Module], ...] = (
    Qwen4ExpTextGatedDeltaNet,
    Qwen3_5MoeGatedDeltaNet,
)

# Call keywords that mean a path the fused preparation does not implement: variable lengths and
# per-token sequence indices, which the conv kernel and the chunked core take but the fused kernel
# does not model.
_VARIABLE_LENGTH_KEYS = ("cu_seqlens", "cu_seq_lens_q", "seq_idx")

# The eager forward's only use of the padding mask, which is cheap enough to keep: it zeroes the
# padded rows before the projection. Every other consumer of a mask in this architecture belongs to
# the packed-sequence path the guard above rejects.
MASK_HELPER_NAME = "apply_mask_to_padding_states"


def _site_supports(module: torch.nn.Module) -> bool:
    """Whether one GatedDeltaNet layer is a site the fused preparation can serve."""

    # The layer carries these as plain Python numbers. The cast is for the type checker, for which
    # `nn.Module.__getattr__` is untyped.
    layer = cast(Any, module)
    head_dim = int(layer.head_k_dim)
    if head_dim != int(layer.head_v_dim):
        # The preparation tiles one head dimension for query, key and value alike.
        return False
    if head_dim <= 0 or head_dim & (head_dim - 1):
        # The kernel indexes the head dimension with `tl.arange`, so it has to be a power of two.
        return False
    if int(layer.num_v_heads) % int(layer.num_k_heads):
        return False
    if int(layer.conv_kernel_size) != TAPS:
        return False
    conv = getattr(layer, "conv1d", None)
    if conv is None or conv.bias is not None:
        # The fused convolution has no bias path, and this architecture's conv is configured without
        # one. Anything else keeps the eager path.
        return False
    if str(getattr(layer, "activation", "")) not in ("silu", "swish"):
        return False
    channels = 2 * int(layer.num_k_heads) * head_dim + int(layer.num_v_heads) * head_dim
    return channels == int(conv.weight.shape[0])


def _training_path_applies(cache_params: Any, kwargs: Mapping[str, Any]) -> bool:
    """Whether this call is the training path, which is the only one the patch replaces."""

    if cache_params is not None:
        # Decode and cached prefill stay on the eager forward: they carry recurrent state and use the
        # single-token kernels.
        return False
    return not any(kwargs.get(key) is not None for key in _VARIABLE_LENGTH_KEYS)


def prepare_fused_preparation_site(name: str, module: torch.nn.Module) -> None:
    """Stash the forward this patch replaces, so a call it does not serve can still fall through."""

    del name
    if not hasattr(module, UNPATCHED_FORWARD):
        setattr(module, UNPATCHED_FORWARD, module.forward)


def fused_preparation_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    cache_params: Any = None,
    attention_mask: torch.Tensor | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """The GatedDeltaNet layer's forward, with its preparation fused into one kernel."""

    # The layer's attributes are untyped through `nn.Module.__getattr__`. The cast is for the type
    # checker only.
    layer = cast(Any, self)

    if not _training_path_applies(cache_params, kwargs):
        unpatched = getattr(self, UNPATCHED_FORWARD)
        return unpatched(hidden_states, cache_params, attention_mask, **kwargs)

    batch_size, seq_len, _ = hidden_states.shape
    head_dim = int(layer.head_v_dim)
    if attention_mask is not None:
        # The eager path zeroes the padded rows and then ignores the mask, so the patch does the same
        # rather than turning the whole fused path off for the padded batches the trainers feed it.
        mask_helper = getattr(
            sys.modules[type(self).__module__], MASK_HELPER_NAME, None
        )
        if mask_helper is None:
            unpatched = getattr(self, UNPATCHED_FORWARD)
            return unpatched(hidden_states, cache_params, attention_mask, **kwargs)
        hidden_states = mask_helper(hidden_states, attention_mask)

    mixed = layer.in_proj_qkv(hidden_states)
    z = layer.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, head_dim)
    b = layer.in_proj_b(hidden_states)
    a = layer.in_proj_a(hidden_states)

    query, key, value, g, beta = fused_prep(
        mixed,
        layer.conv1d.weight.squeeze(1),
        a,
        b,
        layer.A_log,
        layer.dt_bias,
        key_heads=int(layer.num_k_heads),
        value_heads=int(layer.num_v_heads),
        head_dim=int(layer.head_k_dim),
    )

    # `query` and `key` came out broadcast to the value heads in the tiled order, which is what the
    # tiled-convention gate counts. Nothing calls `repeat_interleave` on this path.
    record_tiled_broadcast(self)

    # The same dispatch the eager path uses, from the module that defines this layer's class, so the
    # patch works for every family that shares the forward.
    chunk_rule = getattr(sys.modules[type(self).__module__], CHUNK_RULE_NAME)
    core_attn_out, _ = chunk_rule(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=None,
        output_final_state=False,
        # `query` and `key` arrive normalized and already broadcast to the value heads.
        use_qk_l2norm_in_kernel=False,
        cu_seqlens=None,
        **kwargs,
    )

    core_attn_out = core_attn_out.reshape(-1, head_dim)
    z = z.reshape(-1, head_dim)
    core_attn_out = layer.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
    return layer.out_proj(core_attn_out)


def _spec(layer_type: type[torch.nn.Module]) -> ModulePatchSpec[Any]:
    """One spec per layer family, sharing the counters so the gate sees every GDN layer."""

    return ModulePatchSpec(
        module_type=layer_type,
        forward=fused_preparation_forward,
        handled_key=HANDLED_KEY,
        skip_key=SKIPPED_KEY,
        accept=lambda name, module: _site_supports(module),
        prepare=prepare_fused_preparation_site,
        marker=MARKER,
        # The layer's parameters are the training target. Nothing here is frozen.
        freeze_weight=False,
    )


_SPECS: tuple[ModulePatchSpec[Any], ...] = tuple(
    _spec(layer_type) for layer_type in _GDN_LAYER_TYPES
)


def configure_fused_preparation(model: torch.nn.Module) -> dict[str, Any]:
    """Install the fused preparation on every GatedDeltaNet layer whose geometry it serves."""

    return patch_module_forwards(model, _SPECS)


def require_fused_preparation(
    report: Mapping[str, Any],
    *,
    expected_gdn_layers: int,
    expected_skipped: int = 0,
) -> None:
    """Fail closed unless every expected layer was offered to the patch and the split is the one
    the caller declared."""

    require_complete_inventory(
        report,
        {HANDLED_KEY: expected_gdn_layers, SKIPPED_KEY: expected_skipped},
        subject=SUBJECT,
    )
