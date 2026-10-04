"""Drop-in launcher for FLA's `chunk_bwd_dqkwg` with tiles the device's gates hide.

FLA sizes this kernel's key and value blocks from `CONST_TILING`, which is 32 here because gfx1151
reports a smaller shared-memory budget than every `check_shared_mem` gate tests for, so its key loop
runs four times per program and recomputes the `[BT, BT]` score tile each time. The kernel itself is
fine - it is FLA's, unchanged - and its tiles are plain constexpr arguments. Only the launcher, whose
constant is collapsed, prevents asking for wider ones.

A launcher of our own is the whole change, and it is also what keeps the patching local: the
module-wide alternative, forcing `check_shared_mem` inside `fla.ops.common.chunk_o`, would move
`dv_local` too and would need the widened cache keys in the tuning table.

`docs/plan_qwen4_exp_gdn_backward.md` carries the tile screen and what each point is worth. The shape
of the answer: `BK=128` removes a fourfold redundancy, so there is one score tile per chunk and head
where there were four and one read of `v`, `do` and `dv` where there were four, while `BV=32` keeps
the value loop at four iterations because one wide iteration measures worse even though it re-reads
the key side.

The accumulation order changes with the tile, which is why the gate is the layer's gradients and not
bitwise equality: FLA's own autotune would pick a different order for a different device.
"""

from typing import Any

import torch
import triton

# The winning screen point. `BK` is the whole key dimension, so the kernel runs with one key block.
BLOCK_K = 128
BLOCK_V = 32
NUM_WARPS = 8
NUM_STAGES = 2

_ORIGINAL = None


def chunk_bwd_dqkwg(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    h: torch.Tensor,
    dh: torch.Tensor,
    w: torch.Tensor | None = None,
    g: torch.Tensor | None = None,
    g_gamma: torch.Tensor | None = None,
    dv: torch.Tensor | None = None,
    scale: float | None = None,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """FLA's launcher, with the tiles this device's gates hide."""

    if g is None or dv is None or w is None:
        raise NotImplementedError("this training geometry always has `g`, `dv` and `w`")
    if g_gamma is not None:
        raise NotImplementedError(
            "the gated gamma variant is not part of this training geometry"
        )
    if cu_seqlens is not None or chunk_indices is not None:
        raise NotImplementedError(
            "variable lengths are not part of this training geometry"
        )
    if state_v_first:
        raise NotImplementedError("the deployed layout is state_v_first=False")
    if chunk_size != 64:
        raise NotImplementedError(f"chunk size {chunk_size} is not the trained 64")

    batch, sequence, heads, key_dim = k.shape
    value_dim, value_heads = v.shape[-1], v.shape[2]
    if key_dim != BLOCK_K or value_dim % BLOCK_V:
        raise NotImplementedError(
            f"head dimensions {key_dim}/{value_dim} are not the trained 128"
        )

    dq = q.new_empty(batch, sequence, value_heads, key_dim)
    dk = k.new_empty(batch, sequence, value_heads, key_dim)
    dw = torch.empty_like(w)
    # With one key block the kernel's partial-gradient axis is a single slice, so the workspace the
    # reference allocates, writes and then sums over can be the output itself.
    dg = torch.empty(batch, sequence, value_heads, dtype=torch.float32, device=g.device)

    from triton.runtime.jit import JITFunction

    kernel = _fla_kernel()
    grid = (1, triton.cdiv(sequence, 64), batch * value_heads)
    kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        g_gamma=None,
        h=h,
        do=do,
        dh=dh,
        dw=dw,
        dq=dq,
        dk=dk,
        dv=dv,
        dg=dg,
        cu_seqlens=None,
        chunk_indices=None,
        scale=scale,
        B=batch,
        T=sequence,
        H=heads,
        HV=value_heads,
        K=key_dim,
        V=value_dim,
        BT=64,
        BK=BLOCK_K,
        BV=BLOCK_V,
        STATE_V_FIRST=False,
        USE_G=True,
        USE_G_GAMMA=False,
        USE_DW=True,
        IS_VARLEN=False,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    assert isinstance(kernel, JITFunction)

    if heads != value_heads:
        dq = dq.view(batch, sequence, heads, value_heads // heads, key_dim).sum(3)
        dk = dk.view(batch, sequence, heads, value_heads // heads, key_dim).sum(3)
    return dq, dk, dw, dg


def _fla_kernel():
    """FLA's kernel, unwrapped past the autotuner and the heuristics that compute its flags."""

    from fla.ops.common import chunk_o
    from triton.runtime.jit import JITFunction

    kernel = chunk_o.chunk_bwd_kernel_dqkwg
    while not isinstance(kernel, JITFunction):
        kernel = kernel.fn
    return kernel


def install() -> dict[str, Any]:
    """Replace FLA's launcher where the chunked backward looks it up."""

    global _ORIGINAL
    from fla.ops.gated_delta_rule import chunk

    if _ORIGINAL is None:
        _ORIGINAL = chunk.chunk_bwd_dqkwg
    chunk.chunk_bwd_dqkwg = chunk_bwd_dqkwg
    return report()


def report() -> dict[str, Any]:
    """What is installed, for a gate or a log line to record."""

    return {
        "block_k": BLOCK_K,
        "block_v": BLOCK_V,
        "num_warps": NUM_WARPS,
        "num_stages": NUM_STAGES,
        "installed": _ORIGINAL is not None,
    }


def require_matching_reference(
    candidate: tuple[
        torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None
    ],
    reference: tuple[
        torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None
    ],
    *,
    maximum_relative_rmse: float,
) -> dict[str, float]:
    """Compare `(dq, dk, dw, dg)` against the reference and report the worst relative RMSE."""

    worst = 0.0
    result = {}
    for name, got, want in zip(
        ("dq", "dk", "dw", "dg"), candidate, reference, strict=True
    ):
        assert got is not None and want is not None
        delta = (got.float() - want.float()).flatten()
        scale = want.float().flatten().square().mean().sqrt().clamp_min(1e-12)
        value = float(delta.square().mean().sqrt() / scale)
        result[name] = value
        worst = max(worst, value)
    result["worst"] = worst
    if worst > maximum_relative_rmse:
        raise AssertionError(f"dqkwg drifted: {result} against {maximum_relative_rmse}")
    return result
