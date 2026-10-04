"""Gluon dK/dV owner for the QSA attention backward.

The Triton kernel of the same shape is DRAM-bound on re-reading Q and dO, because BLOCK_N=16 is all
that fits in 256 VGPRs once two FP32 [BLOCK_N, 256] accumulators, the resident key/value tiles and
Triton's layout-conversion copies all want registers. This kernel takes the two steps Triton cannot
express: key and value stay in shared memory and are read as wmma B operands straight from there, and
the operand layouts of the streamed tensors are chosen at the load instead of being converted
afterwards.

Its output is bit-identical to the Triton kernel, at full length and with right padding, and the
launcher takes the same partial-sums workspace, so the existing reduction kernel sums the split
partials for both implementations.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

# BLOCK_N is 16 for the same reason it is 16 in Triton: the two FP32 [BLOCK_N, 256] accumulators
# dominate the register file. Four warps fit here where two won in Triton, because the key and value
# tiles no longer occupy registers.
GLUON_DKDV_CONFIG = {"block_m": 32, "block_n": 16, "num_warps": 4}

_LAYOUTS = {}


def _layouts(num_warps, block_m, block_n, head_dim):
    key = (num_warps, block_m, block_n, head_dim)
    if key not in _LAYOUTS:
        index = max(0, num_warps.bit_length() - 1)
        mma_s = gl.amd.AMDWMMALayout(
            version=1,
            transposed=True,
            warp_bases=[[1 << i, 0] for i in range(index)],
            instr_shape=[16, 16, 16],
        )
        mma_g = gl.amd.AMDWMMALayout(
            version=1,
            transposed=True,
            warp_bases=[[0, 1 << i] for i in range(index)],
            instr_shape=[16, 16, 16],
        )
        _LAYOUTS[key] = {
            "mma_s": gl.constexpr(mma_s),
            "mma_g": gl.constexpr(mma_g),
            "q_layout": gl.constexpr(gl.DotOperandLayout(0, mma_s, 16)),
            "kt_layout": gl.constexpr(gl.DotOperandLayout(1, mma_s, 16)),
            "a_layout": gl.constexpr(gl.DotOperandLayout(0, mma_g, 16)),
            "b_layout": gl.constexpr(gl.DotOperandLayout(1, mma_g, 16)),
        }
    return _LAYOUTS[key]


@gluon.jit
def _qsa_dkdv_gluon_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    doutput_ptr,
    delta_ptr,
    lse_ptr,
    key_end_ptr,
    dkey_ptr,
    dvalue_ptr,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_km,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vm,
    stride_vd,
    stride_ob,
    stride_om,
    stride_db,
    stride_dm,
    stride_lb,
    stride_lm,
    SM_SCALE: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    GROUP_SIZE: gl.constexpr,
    SEQUENCE_LENGTH: gl.constexpr,
    SPLIT: gl.constexpr,
    PARTIAL_SPLIT_STRIDE: gl.constexpr,
    mma_s: gl.constexpr,
    mma_g: gl.constexpr,
    q_layout: gl.constexpr,
    kt_layout: gl.constexpr,
    a_layout: gl.constexpr,
    b_layout: gl.constexpr,
):
    log2e: gl.constexpr = 1.4426950408889634
    scale2: gl.constexpr = SM_SCALE * log2e
    # The score tile and the gradient accumulators have their own warp layouts. In Gluon the warps
    # must be told how to cover a tile, which is exactly the freedom the Triton version does not have.
    # Layouts arrive from the host as constexpr objects (Gluon's own examples do the same): the
    # frontend cannot hash the raw nested lists, but layout instances are hashable.

    key_tile = gl.program_id(0)
    head_split = gl.program_id(1)
    batch = gl.program_id(2)
    kv_head = head_split // SPLIT
    split_index = head_split % SPLIT

    start_n = key_tile * BLOCK_N
    key_end = gl.load(key_end_ptr + batch)
    # Right padding: padded key rows have no real scores so they must not be stored, and padded query
    # rows carry no incoming gradient, so their probabilities are zeroed by the mask below.
    key_in_range = (
        start_n + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, mma_g))
    ) < key_end
    # The key tile's row offset belongs in these offsets: without it every program loads and stores
    # the first key tile and races with the others on the same addresses.
    offs_n = start_n + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, mma_g))
    offs_dg = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, mma_g))
    key_offsets = (
        batch * stride_kb
        + kv_head * stride_kh
        + offs_n[:, None] * stride_km
        + offs_dg[None, :] * stride_kd
    )
    # Key and value are read once and used by every head and every query tile of this program, so
    # they live in shared memory and are read as wmma operands from there.
    key_shared = gl.allocate_shared_memory(
        gl.bfloat16, [BLOCK_N, HEAD_DIM], gl.SwizzledSharedLayout(8, 1, 8, [1, 0])
    )
    value_shared = gl.allocate_shared_memory(
        gl.bfloat16, [BLOCK_N, HEAD_DIM], gl.SwizzledSharedLayout(8, 1, 8, [1, 0])
    )
    key_shared.store(gl.load(key_ptr + key_offsets))
    value_shared.store(
        gl.load(
            value_ptr
            + batch * stride_vb
            + kv_head * stride_vh
            + offs_n[:, None] * stride_vm
            + offs_dg[None, :] * stride_vd
        )
    )
    # A shared-memory store followed by a load in a different layout crosses warps, so the workgroup
    # has to be synchronised between them. Without this every load can read another warp's stale data.
    gl.barrier()
    query_shared = gl.allocate_shared_memory(
        gl.bfloat16, [BLOCK_M, HEAD_DIM], gl.SwizzledSharedLayout(8, 1, 8, [1, 0])
    )
    output_shared = gl.allocate_shared_memory(
        gl.bfloat16, [BLOCK_M, HEAD_DIM], gl.SwizzledSharedLayout(8, 1, 8, [1, 0])
    )
    probability_shared = gl.allocate_shared_memory(
        gl.bfloat16, [BLOCK_M, BLOCK_N], gl.SwizzledSharedLayout(8, 1, 8, [1, 0])
    )
    score_shared = gl.allocate_shared_memory(
        gl.bfloat16, [BLOCK_M, BLOCK_N], gl.SwizzledSharedLayout(8, 1, 8, [1, 0])
    )
    key_t = key_shared.permute([1, 0]).load(layout=kt_layout)
    value_t = value_shared.permute([1, 0]).load(layout=kt_layout)

    dkey = gl.zeros([BLOCK_N, HEAD_DIM], gl.float32, layout=mma_g)
    dvalue = gl.zeros([BLOCK_N, HEAD_DIM], gl.float32, layout=mma_g)

    row_offsets = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, q_layout))
    # A second row vector in the score layout: lse, delta and the causal mask live on the score tile,
    # so they must be broadcast in that layout, not in the operand layout the loads use.
    row_scores = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, mma_s))
    offs_ds = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, q_layout))
    offs_ns = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, mma_s))
    first_tile = start_n // BLOCK_M
    last_tile = SEQUENCE_LENGTH // BLOCK_M
    split_chunk = (last_tile - first_tile + SPLIT - 1) // SPLIT
    tile_lo = first_tile + split_index * split_chunk
    tile_hi = min(tile_lo + split_chunk, last_tile)

    for head_slot in range(GROUP_SIZE):
        head = kv_head * GROUP_SIZE + head_slot
        for query_tile in range(tile_lo, tile_hi):
            offs_m = query_tile * BLOCK_M + row_offsets
            offs_m_scores = query_tile * BLOCK_M + row_scores
            # Both streamed tensors are loaded once in the layout the first dot wants, then staged in
            # shared memory so the accumulation dots can read them as the other operand without a
            # register layout conversion.
            query = gl.load(
                query_ptr
                + batch * stride_qb
                + head * stride_qh
                + offs_m[:, None] * stride_qm
                + offs_ds[None, :] * stride_qd
            )
            doutput = gl.load(
                doutput_ptr
                + batch * stride_ob
                + head * HEAD_DIM
                + offs_m[:, None] * stride_om
                + offs_ds[None, :]
            )
            query_shared.store(query)
            output_shared.store(doutput)
            lse = (
                gl.load(lse_ptr + batch * stride_lb + head * stride_lm + offs_m_scores)
                * log2e
            )
            delta = gl.load(
                delta_ptr + batch * stride_db + head * stride_dm + offs_m_scores
            )

            scores = gl.zeros([BLOCK_M, BLOCK_N], gl.float32, layout=mma_s)
            scores = gl.amd.rdna3.wmma(query, key_t, scores)
            scores = scores * scale2 - lse[:, None]
            probabilities = gl.exp2(scores)
            probabilities = gl.where(
                (offs_m_scores[:, None] >= (start_n + offs_ns)[None, :])
                & (offs_m_scores[:, None] < key_end)
                & ((start_n + offs_ns)[None, :] < key_end),
                probabilities,
                0.0,
            )
            dprobability = gl.zeros([BLOCK_M, BLOCK_N], gl.float32, layout=mma_s)
            dprobability = gl.amd.rdna3.wmma(doutput, value_t, dprobability)
            dscore = probabilities * (dprobability - delta[:, None]) * SM_SCALE

            # dV += p^T dO and dK += dS^T Q: the two computed tiles are the A operand of the
            # accumulation, so they are stored and read back transposed through shared memory.
            probability_shared.store(
                probabilities.to(gl.bfloat16, fp_downcast_rounding="rtne")
            )
            score_shared.store(dscore.to(gl.bfloat16, fp_downcast_rounding="rtne"))
            # One barrier covers all four staged tiles: nothing is read back before it.
            gl.barrier()
            query_b = query_shared.load(layout=b_layout)
            output_b = output_shared.load(layout=b_layout)
            # A second barrier before the accumulation reads. It is not redundant with the one above:
            # it also fixes the schedule the compiler picks for the shared-memory reads, so it has to
            # stay even though it looks removable.
            gl.barrier()
            dvalue = gl.amd.rdna3.wmma(
                probability_shared.permute([1, 0]).load(layout=a_layout),
                output_b,
                dvalue,
            )
            dkey = gl.amd.rdna3.wmma(
                score_shared.permute([1, 0]).load(layout=a_layout),
                query_b,
                dkey,
            )

    partial_offsets = key_offsets + split_index * PARTIAL_SPLIT_STRIDE
    gl.store(dkey_ptr + partial_offsets, dkey, mask=key_in_range[:, None])
    gl.store(dvalue_ptr + partial_offsets, dvalue, mask=key_in_range[:, None])


def qsa_dkdv_gluon(
    query,
    key,
    value,
    doutput,
    delta,
    lse,
    key_end,
    partial_key,
    partial_value,
    split,
) -> None:
    """dK and dV for every key tile, written as `split` partial sums into the workspace.

    Shapes follow the Triton kernel: `query` [batch, query_heads, sequence, head_dim], `key` and
    `value` [batch, kv_heads, sequence, head_dim] in bf16, `doutput` [batch, sequence, heads * dim],
    `lse` and `delta` [batch, heads, sequence] in fp32, `key_end` [batch] int32, and the workspace
    [2, split, batch, kv_heads, sequence, head_dim] in fp32.
    """
    batch, query_heads, sequence, head_dim = query.shape
    kv_heads = key.shape[1]
    config = GLUON_DKDV_CONFIG
    block_m = config["block_m"]
    block_n = config["block_n"]
    num_warps = config["num_warps"]
    layouts = _layouts(num_warps, block_m, block_n, head_dim)
    grid = (sequence // block_n, kv_heads * split, batch)
    _qsa_dkdv_gluon_kernel[grid](
        query,
        key,
        value,
        doutput,
        delta,
        lse,
        key_end,
        partial_key,
        partial_value,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        query.stride(3),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        key.stride(3),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        value.stride(3),
        doutput.stride(0),
        doutput.stride(1),
        delta.stride(0),
        delta.stride(1),
        lse.stride(0),
        lse.stride(1),
        SM_SCALE=1.0 / (head_dim**0.5),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HEAD_DIM=head_dim,
        GROUP_SIZE=query_heads // kv_heads,
        SEQUENCE_LENGTH=sequence,
        SPLIT=split,
        PARTIAL_SPLIT_STRIDE=batch * kv_heads * sequence * head_dim,
        **layouts,
        num_warps=num_warps,
        num_stages=1,
    )
