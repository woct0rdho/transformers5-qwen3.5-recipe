"""A faster `recompute_w_u_fwd`: the two per-chunk matmuls that build W and U.

FLA's chunked GatedDeltaNet forward spends `recompute_w_u_fwd` turning the solved inverse `A` into
the two tensors the state pass consumes:
u[t] = sum_j A[t, j] * (v[j] * beta[j])
w[t] = sum_j A[t, j] * (k[j] * beta[j] * exp2(g[t]))

Both are `[BT, BT] @ [BT, D]` matmuls per (chunk, value head), with `A` the same solved
`(I + A)^{-1}` the solve kernel already wrote, and `g` the cumulative decay in log2 space.

The kernel here keeps the same three inputs and the same two outputs as FLA's, so it is a drop-in for
`fla.ops.gated_delta_rule.wy_fast.recompute_w_u_fwd` inside FLA's own autograd function, which is where
the call sits: the function is not autograd-tracked, so a plain tensor-in, tensor-out function is the
whole interface.

The difference from FLA's kernel is the tile and the warp count. FLA fixes `BK = BV = 64` and the table
runs it with two warps, which spreads a `[64, 64]` FP32 accumulator over 64 lanes and puts the kernel
at the architectural register ceiling, where it spills. The plan records the measured cost of that.
Here the head is tiled 32 wide and the program covers whole chunks instead, with one warp per program
and the chunk loop unrolled so that more than one load-to-dot chain is in flight.
"""

from typing import Any

import torch
import triton
import triton.language as tl

# Head-tile width. The token block is fixed by the chunk length that `A` carries, so the only tiling
# choice the kernel has is how much of the head it covers per program.
BLOCK_D = 32
NUM_WARPS = 1
NUM_STAGES = 2
# Chunks per program. Unrolling two chunks gives the memory system a second independent load-to-dot
# chain to work on while the first one waits. Four is already past the point where that helps, and the
# plan records both measurements.
CHUNKS = 2

# The FLA function this module replaces, kept so `install` is idempotent and a report can say whether
# it has been installed.
_ORIGINAL = None


@triton.jit
def _row_mask(row, T, MASK_T: tl.constexpr):
    if MASK_T:
        return (row >= 0) & (row < T)
    return row >= 0


@triton.jit
def _wu_kernel(
    k_ptr,
    v_ptr,
    beta_ptr,
    a_ptr,
    g_ptr,
    w_ptr,
    u_ptr,
    T,
    H,
    HV,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    CHUNKS: tl.constexpr,
    MASK_T: tl.constexpr,
):
    """One chunk of one value head: load the solved inverse, then two matmuls.

    `A` is `[B, T, HV, BT]`, `v` and `u` are `[B, T, HV, V]`, `k` and `w` are `[B, T, H, K]`, and `g`
    is the cumulative log2 decay `[B, T, HV]`. When the key and value head counts differ, key head
    `i // (HV // H)` serves value head `i`.
    """
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // HV
    i_h = i_bh % HV
    i_kh = i_h // (HV // H)

    bos = i_b * T
    # Several chunks per program, unrolled, so their loads are independent of each other and the memory
    # system can work on the next chunk while the previous one's dots run. A single chunk per program
    # leaves one load-to-dot chain exposed, which is what every kernel in this family turns out to be
    # limited by.
    for c in tl.static_range(CHUNKS):
        _wu_chunk(
            k_ptr,
            v_ptr,
            beta_ptr,
            a_ptr,
            g_ptr,
            w_ptr,
            u_ptr,
            bos,
            (i_t * CHUNKS + c) * BT,
            i_h,
            i_kh,
            T,
            H,
            HV,
            K=K,
            V=V,
            BT=BT,
            BK=BK,
            BV=BV,
            USE_G=USE_G,
            MASK_T=MASK_T,
        )


@triton.jit
def _wu_chunk(
    k_ptr,
    v_ptr,
    beta_ptr,
    a_ptr,
    g_ptr,
    w_ptr,
    u_ptr,
    bos,
    t0,
    i_h,
    i_kh,
    T,
    H,
    HV,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    MASK_T: tl.constexpr,
):
    local = t0 + tl.arange(0, BT)
    # The mask is per batch, the pointers use the flat token index.
    mask_row = _row_mask(local, T, MASK_T)
    rows = bos + local
    cols = tl.arange(0, BT)
    mask_a = mask_row[:, None] & (cols[None, :] < BT)

    # The solved inverse for this chunk and head, one row per token.
    a = tl.load(
        a_ptr + (rows[:, None] * HV + i_h) * BT + cols[None, :],
        mask=mask_a,
        other=0.0,
    )
    beta = tl.load(beta_ptr + rows * HV + i_h, mask=mask_row, other=0.0).to(tl.float32)
    # `v`, `u`, `beta` and `g` carry the value axis. `k` and `w` carry the key axis, which is a
    # different head count and a different stride. Reading one with the other's offset walks off the
    # tensor, which is how the first version of this kernel faulted the device.
    off_token = rows * HV + i_h
    off_key = rows * H + i_kh

    # A definite tensor either way: with `USE_G` false the decay is one, so the multiply stays in the
    # same place in both variants and the checker can see the type.
    if USE_G:
        decay = tl.math.exp2(
            tl.load(g_ptr + off_token, mask=mask_row, other=0.0).to(tl.float32)
        )
    else:
        decay = tl.zeros([BT], dtype=tl.float32) + 1.0

    # Both matmuls per program. Splitting them across a third grid axis was measured and is worse: the
    # smaller live set does not pay for loading `A` twice and for a second exposed load-to-dot chain,
    # which is what this kernel is actually bound by. The plan records the numbers.
    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        mask_v = mask_row[:, None] & (o_v[None, :] < V)
        value = tl.load(
            v_ptr + off_token[:, None] * V + o_v[None, :], mask=mask_v, other=0.0
        )
        # The reference rounds `beta * v` to the compute dtype before the dot, and so must this.
        scaled = (value * beta[:, None]).to(value.dtype)
        out_u = tl.dot(a, scaled)
        tl.store(
            u_ptr + off_token[:, None] * V + o_v[None, :],
            out_u.to(u_ptr.dtype.element_ty),
            mask=mask_v,
        )

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        mask_k = mask_row[:, None] & (o_k[None, :] < K)
        key = tl.load(
            k_ptr + off_key[:, None] * K + o_k[None, :], mask=mask_k, other=0.0
        )
        weight = key * beta[:, None] * decay[:, None]
        # A plain product, so the rounding to the compute dtype happens once, as in the reference.
        out_w = tl.dot(a, weight.to(key.dtype))
        # `w` is laid out like the value axis, one copy per value head, because the state pass reads it
        # per value head. Only the `k` operand uses the key axis.
        tl.store(
            w_ptr + off_token[:, None] * K + o_k[None, :],
            out_w.to(w_ptr.dtype.element_ty),
            mask=mask_k,
        )


def recompute_w_u(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    g: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    *,
    block_d: int = BLOCK_D,
    num_warps: int = NUM_WARPS,
    num_stages: int = NUM_STAGES,
    chunks: int = CHUNKS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the `w` and `u` tensors from the solved inverse. Same contract as FLA's version.

    `cu_seqlens` and `chunk_indices` are FLA's variable-length arguments. This kernel serves the
    fixed-length training geometry and rejects them rather than reading the wrong chunk boundaries.


    `k` is `[B, T, H, K]`, `v` is `[B, T, HV, V]`, `beta` and (when given) `g` are `[B, T, HV]`, and
    `A` is `[B, T, HV, BT]` as the solve kernel leaves it. Returns `(w, u)` shaped like `k` and `v`.
    """

    if cu_seqlens is not None or chunk_indices is not None:
        raise NotImplementedError(
            "variable lengths are not part of this training geometry"
        )
    if k.dim() != 4 or v.dim() != 4 or A.dim() != 4:
        raise ValueError("k, v and A must be 4-D")
    batch, sequence, key_heads, key_dim = k.shape
    if v.shape[:2] != (batch, sequence):
        raise ValueError(
            f"v must share k's batch and sequence, got {tuple(v.shape)} and {tuple(k.shape)}"
        )
    value_heads, value_dim = v.shape[2], v.shape[3]
    if A.shape != (batch, sequence, value_heads, A.shape[-1]):
        raise ValueError(
            f"A must be [B, T, HV, BT] = {batch, sequence, value_heads, A.shape[-1]}, got {tuple(A.shape)}"
        )
    if beta.shape != (batch, sequence, value_heads):
        raise ValueError(
            f"beta must be [B, T, HV] = {batch, sequence, value_heads}, got {tuple(beta.shape)}"
        )
    if g is not None and g.shape != beta.shape:
        raise ValueError(
            f"g must be [B, T, HV] = {tuple(beta.shape)}, got {tuple(g.shape)}"
        )
    if value_heads % key_heads:
        raise ValueError(
            f"value heads ({value_heads}) must be a multiple of key heads ({key_heads})"
        )
    if key_dim % block_d or value_dim % block_d:
        # The kernel tiles the head width. A non-multiple would silently drop the tail.
        raise ValueError(
            f"head dims {key_dim} and {value_dim} must be multiples of {block_d}"
        )
    # The contraction of both matmuls is the chunk, and `A`'s last axis is the chunk length, so the
    # token block is not a free tiling parameter: a smaller block would sum only part of the chunk and a
    # larger one would read into the neighbouring chunk, both silently. It comes from `A`, not from an
    # argument, and only has to satisfy the dot's shortest axis.
    chunk = A.shape[-1]
    if chunk & (chunk - 1):
        raise ValueError(
            f"A's chunk length must be a power of two for the row tile, got {chunk}"
        )
    if chunk < 16:
        raise ValueError(
            f"A's chunk length is below the dot's shortest axis (16), got {chunk}"
        )
    for name, tensor in (("k", k), ("v", v), ("beta", beta), ("A", A)):
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    # `w` keeps the value-head count, one copy per value head, as FLA's version does.
    w = torch.empty(
        (batch, sequence, value_heads, key_dim), dtype=k.dtype, device=k.device
    )
    u = torch.empty_like(v)
    num_blocks = triton.cdiv(sequence, chunk)
    _wu_kernel[(triton.cdiv(num_blocks, chunks), batch * value_heads)](
        k,
        v,
        beta,
        A,
        g if g is not None else beta,
        w,
        u,
        sequence,
        key_heads,
        value_heads,
        K=key_dim,
        V=value_dim,
        BT=chunk,
        BK=block_d,
        BV=block_d,
        USE_G=g is not None,
        CHUNKS=chunks,
        # A chunk is in range when its first row is: with several chunks per program the last program
        # can overrun the sequence even when a single chunk divides it, so the mask follows the group
        # width. Without it the overrun reads and writes past the buffers.
        MASK_T=bool(sequence % (chunk * chunks)),
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return w, u


def install() -> dict[str, Any]:
    """Replace FLA's recomputation where the chunked forward and backward look it up.

    `fla.ops.gated_delta_rule.chunk` imports the name at module scope and calls it from there, once
    in the forward and once inside the backward's recomputation, so rebinding the module attribute is
    enough. The signature is FLA's, so nothing else at either call site changes.
    """

    global _ORIGINAL
    from fla.ops.gated_delta_rule import chunk

    if _ORIGINAL is None:
        _ORIGINAL = chunk.recompute_w_u_fwd
    chunk.recompute_w_u_fwd = recompute_w_u
    return report()


def report() -> dict[str, Any]:
    """The tiling the kernel is compiled for, for a gate or a log line to record."""
    return {
        "block_d": BLOCK_D,
        "num_warps": NUM_WARPS,
        "num_stages": NUM_STAGES,
        "chunks": CHUNKS,
        "installed": _ORIGINAL is not None,
    }


def require_matching_reference(
    candidate: tuple[torch.Tensor, torch.Tensor],
    reference: tuple[torch.Tensor, torch.Tensor],
    *,
    maximum_relative_rmse: float,
) -> dict[str, float]:
    """Compare a `(w, u)` pair against the reference pair and report the worst relative RMSE."""

    worst = 0.0
    result = {}
    for name, got, want in zip(("w", "u"), candidate, reference, strict=True):
        delta = (got.float() - want.float()).flatten()
        scale = want.float().flatten().square().mean().sqrt().clamp_min(1e-12)
        value = float(delta.square().mean().sqrt() / scale)
        result[name] = value
        worst = max(worst, value)
    result["worst"] = worst
    if worst > maximum_relative_rmse:
        raise AssertionError(
            f"w/u recomputation drifted: {result} against {maximum_relative_rmse}"
        )
    return result
