"""Drop-in replacement for FLA's backward state-gradient walk (`chunk_gated_delta_rule_bwd_dhu`).

FLA's kernel holds the decayed query operand in FP32. The line
b_q = b_q * b_g_exp[None, :]
b_dh1 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - ...
casts to the dtype the operand has after the multiply, which is FP32, so `tl.dot` is handed two
FP32 operands and lowers to FP32 FMA on the VALU instead of to the matrix cores. That one dot is why
this kernel carries the largest VALU count in the GatedDeltaNet family, why it reserves kilobytes of
private segment per thread, and why its L2 traffic runs to many times what its tensors account for:
the FP32 tiles are what spills, every iteration.

Casting the operand back to BF16 before the dot, which is what the cast was written for, keeps the
recurrence identical and moves that dot to the matrix cores. `docs/plan_qwen4_exp_gdn_backward.md`
records what that is worth and the screens that rejected the obvious alternatives - accumulator
multi-buffering, the barrier, unrolling, and the value-block width all have measurements there.

The kernel is specialised to the geometry this repository trains: BF16 activations, an FP32
recurrent state, chunk 64, `K = V = 128`, no initial or final state, no variable lengths and
the default `state_v_first=False` layout. Anything else is rejected rather than guessed at.
"""

from typing import Any

import torch
import triton
import triton.language as tl

# The training geometry, as constexpr blocks. `KEY_BLOCK` only splits the key dimension into two
# dots per chunk. The value block is the one dimension the recurrence leaves free - each value column
# of the state evolves on its own - so it sets the grid and it sets the register cost of the state
# tiles, and the plan records the screen that picked 32 over the 64 FLA's launcher uses here.
BLOCK_T = 64
KEY_BLOCK = 64
BLOCK_V = 32
NUM_WARPS = 8
NUM_STAGES = 2

_ORIGINAL = None


@triton.jit
def _dhu_kernel(
    q,
    k,
    w,
    g,
    do,
    dv,
    dv2,
    dh,
    T,
    scale,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    """One value block of the reverse walk over chunks."""

    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // HV, i_nh % HV
    NT = tl.cdiv(T, BT)
    bos = i_n * T

    # Token-major layout: q and k are [B, T, H, K], w is [B, T, HV, K], do, dv and dv2 are
    # [B, T, HV, V], and the state-gradient workspace is [B, NT, HV, K, V].
    q += (bos * H + i_h // (HV // H)) * K
    k += (bos * H + i_h // (HV // H)) * K
    w += (bos * HV + i_h) * K
    do += (bos * HV + i_h) * V
    dv += (bos * HV + i_h) * V
    dv2 += (bos * HV + i_h) * V
    dh += (i_n * NT * HV + i_h) * K * V

    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    o_k1 = tl.arange(0, BK)
    m_k1 = o_k1 < K
    o_k2 = BK + o_k1
    m_k2 = o_k2 < K

    # The state gradient, one FP32 tile per key block, live across the whole walk.
    b_dh1 = tl.zeros([BK, BV], dtype=tl.float32)
    b_dh2 = tl.zeros([BK, BV], dtype=tl.float32)

    # The chunk index stays an int32: the largest offset it forms is `NT * HV * K * V`, which at this
    # training geometry is tens of millions, and the alternative is a 64-bit index tensor per chunk.
    for i_t in range(NT - 1, -1, -1):
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T

        # The gradient of the state entering this chunk, which is what the caller consumes.
        p_dh1 = dh + i_t * HV * K * V + o_k1[:, None] * V + o_v[None, :]
        tl.store(p_dh1, b_dh1.to(tl.bfloat16), mask=m_k1[:, None] & m_v[None, :])
        p_dh2 = dh + i_t * HV * K * V + o_k2[:, None] * V + o_v[None, :]
        tl.store(p_dh2, b_dh2.to(tl.bfloat16), mask=m_k2[:, None] & m_v[None, :])

        last_idx = min((i_t + 1) * BT, T) - 1
        bg_last = tl.load(g + (bos + last_idx) * HV + i_h)
        b_g = tl.load(g + bos * HV + i_h + o_t * HV, mask=m_t, other=0.0)
        bg_last_exp = tl.exp2(bg_last)
        b_g_exp = tl.exp2(b_g)

        m_tv = m_t[:, None] & m_v[None, :]
        p_do = do + o_t[:, None] * (HV * V) + o_v[None, :]
        b_do = tl.load(p_do, mask=m_tv, other=0.0)

        # The value gradient: the gradient arriving through the state, decayed to each
        # token, plus the incoming `dv`.
        p_k1 = k + o_t[:, None] * (H * K) + o_k1[None, :]
        b_k1 = tl.load(p_k1, mask=m_t[:, None] & m_k1[None, :], other=0.0)
        b_dv = tl.dot(b_k1, b_dh1.to(b_k1.dtype))
        p_k2 = k + o_t[:, None] * (H * K) + o_k2[None, :]
        b_k2 = tl.load(p_k2, mask=m_t[:, None] & m_k2[None, :], other=0.0)
        b_dv += tl.dot(b_k2, b_dh2.to(b_k2.dtype))
        b_dv *= tl.where(m_t, tl.exp2(bg_last - b_g), 0.0)[:, None]
        p_dv = dv + o_t[:, None] * (HV * V) + o_v[None, :]
        b_dv += tl.load(p_dv, mask=m_tv, other=0.0)
        p_dv2 = dv2 + o_t[:, None] * (HV * V) + o_v[None, :]
        tl.store(p_dv2, b_dv.to(tl.bfloat16), mask=m_tv)

        # The state gradient: decay it, then add the two per-chunk terms. The query operand
        # is cast back to BF16 on purpose - see the module docstring.
        p_w1 = w + o_k1[:, None] + o_t[None, :] * (HV * K)
        p_q1 = q + o_k1[:, None] + o_t[None, :] * (H * K)
        b_w1 = tl.load(p_w1, mask=m_k1[:, None] & m_t[None, :], other=0.0)
        b_q1 = tl.load(p_q1, mask=m_k1[:, None] & m_t[None, :], other=0.0)
        b_dh1 *= bg_last_exp
        b_q1 = (b_q1 * b_g_exp[None, :]).to(tl.bfloat16)
        b_dh1 += tl.dot(b_q1, b_do) * scale - tl.dot(b_w1, b_dv.to(b_w1.dtype))

        p_w2 = w + o_k2[:, None] + o_t[None, :] * (HV * K)
        p_q2 = q + o_k2[:, None] + o_t[None, :] * (H * K)
        b_w2 = tl.load(p_w2, mask=m_k2[:, None] & m_t[None, :], other=0.0)
        b_q2 = tl.load(p_q2, mask=m_k2[:, None] & m_t[None, :], other=0.0)
        b_dh2 *= bg_last_exp
        b_q2 = (b_q2 * b_g_exp[None, :]).to(tl.bfloat16)
        b_dh2 += tl.dot(b_q2, b_do) * scale - tl.dot(b_w2, b_dv.to(b_w2.dtype))


def chunk_gated_delta_rule_bwd_dhu(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    h0: torch.Tensor | None = None,
    dht: torch.Tensor | None = None,
    scale: float | None = None,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """FLA's launcher signature and layout, with the walk above in place of its kernel."""

    if gk is not None or g is None:
        raise NotImplementedError(
            "this device's configuration always has `g` and never `gk`"
        )
    if h0 is not None or dht is not None:
        raise NotImplementedError(
            "initial and final states are not part of this training geometry"
        )
    if cu_seqlens is not None or chunk_indices is not None:
        raise NotImplementedError(
            "variable lengths are not part of this training geometry"
        )
    if state_v_first:
        raise NotImplementedError("the deployed layout is state_v_first=False")
    if chunk_size != BLOCK_T:
        raise NotImplementedError(
            f"chunk size {chunk_size} is not the trained {BLOCK_T}"
        )

    batch, sequence, heads, key_dim = q.shape
    value_dim, value_heads = do.shape[-1], do.shape[2]
    if key_dim != 128 or value_dim != 128:
        raise NotImplementedError(
            f"head dimensions {key_dim}/{value_dim} are not the trained 128"
        )
    if not q.is_contiguous() or not k.is_contiguous():
        raise NotImplementedError("q and k are contiguous in this path")
    if heads * (value_heads // heads) != value_heads or value_heads % heads:
        raise NotImplementedError(
            f"{heads} key heads do not divide {value_heads} value heads"
        )

    NT = triton.cdiv(sequence, BLOCK_T)
    dh = q.new_empty(batch, NT, value_heads, key_dim, value_dim)
    dv2 = torch.empty_like(dv)
    grid = lambda meta: (triton.cdiv(value_dim, meta["BV"]), batch * value_heads)
    _dhu_kernel[grid](
        q=q,
        k=k,
        w=w,
        g=g,
        do=do,
        dv=dv,
        dv2=dv2,
        dh=dh,
        T=sequence,
        scale=1.0 if scale is None else scale,
        H=heads,
        HV=value_heads,
        K=key_dim,
        V=value_dim,
        BT=BLOCK_T,
        BK=KEY_BLOCK,
        BV=BLOCK_V,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return dh, None, dv2


def install() -> dict[str, Any]:
    """Replace FLA's walk wherever the chunked backward looks it up.

    `fla.ops.gated_delta_rule.chunk` imports the name at module scope and its autograd
    Function calls it from there, so rebinding the module attribute is enough.
    """

    global _ORIGINAL
    from fla.ops.gated_delta_rule import chunk

    if _ORIGINAL is None:
        _ORIGINAL = chunk.chunk_gated_delta_rule_bwd_dhu
    chunk.chunk_gated_delta_rule_bwd_dhu = chunk_gated_delta_rule_bwd_dhu
    return report()


def report() -> dict[str, Any]:
    """What is installed, for a gate or a log line to record."""

    return {
        "block_t": BLOCK_T,
        "key_block": KEY_BLOCK,
        "block_v": BLOCK_V,
        "num_warps": NUM_WARPS,
        "num_stages": NUM_STAGES,
        "query_operand_dtype": "bfloat16",
        "installed": _ORIGINAL is not None,
    }


def require_matching_reference(
    candidate: tuple[torch.Tensor, None, torch.Tensor],
    reference: tuple[torch.Tensor, None, torch.Tensor],
    *,
    maximum_relative_rmse: float,
) -> dict[str, float]:
    """Compare `(dh, None, dv2)` against the reference and report the worst relative RMSE."""

    worst = 0.0
    result = {}
    for name, got, want in zip(
        ("dh", "dv2"),
        (candidate[0], candidate[2]),
        (reference[0], reference[2]),
        strict=True,
    ):
        delta = (got.float() - want.float()).flatten()
        scale = want.float().flatten().square().mean().sqrt().clamp_min(1e-12)
        value = float(delta.square().mean().sqrt() / scale)
        result[name] = value
        worst = max(worst, value)
    result["worst"] = worst
    if worst > maximum_relative_rmse:
        raise AssertionError(
            f"state-gradient walk drifted: {result} against {maximum_relative_rmse}"
        )
    return result
