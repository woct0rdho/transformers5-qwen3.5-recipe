"""Fixed-shape DeepSeek V4 heavily compressed attention for gfx1151.

The family path owns non-overlapping rate-128 compression and one shared
online softmax over local KV, 16 compressed KV entries, and attention sinks.
"""

import math
from typing import Any

import torch
import triton
import triton.language as tl

_SUPPORTED_BATCHES = frozenset({1, 4, 16})
_SEQUENCE_LENGTH = 2048
_QUERY_HEADS = 64
_KV_HEADS = 1
_HEAD_DIM = 512
_ROPE_DIM = 64
_ROPE_PAIRS = _ROPE_DIM // 2
_WINDOW = 128
_WINDOW_LEFT = _WINDOW - 1
_COMPRESS_RATE = 128
_COMPRESSED_LENGTH = _SEQUENCE_LENGTH // _COMPRESS_RATE
_SOFTMAX_SCALE = 1.0 / math.sqrt(_HEAD_DIM)

_FORWARD_CONFIGS = {
    1: {
        "block_m": 8,
        "local_n": 16,
        "compressed_n": 16,
        "head_group": 8,
        "num_warps": 8,
        "waves_per_eu": 1,
    },
    4: {
        "block_m": 8,
        "local_n": 16,
        "compressed_n": 16,
        "head_group": 8,
        "num_warps": 8,
        "waves_per_eu": 1,
    },
}

_DQ_CONFIGS = {
    1: {
        "block_m": 4,
        "local_n": 16,
        "compressed_n": 16,
        "head_group": 16,
        "num_warps": 8,
        "waves_per_eu": 0,
    },
    4: {
        "block_m": 4,
        "local_n": 16,
        "compressed_n": 16,
        "head_group": 16,
        "num_warps": 8,
        "waves_per_eu": 0,
    },
}

_LOCAL_DKV_CONFIGS = {
    1: {
        "dv": {
            "block_m": 32,
            "block_n": 32,
            "num_warps": 8,
            "waves_per_eu": 1,
            "head_unroll": 1,
        },
        "dk": {
            "block_m": 32,
            "block_n": 32,
            "num_warps": 4,
            "waves_per_eu": 2,
            "head_unroll": 1,
        },
    },
    4: {
        "dv": {
            "block_m": 32,
            "block_n": 32,
            "num_warps": 8,
            "waves_per_eu": 1,
            "head_unroll": 2,
        },
        "dk": {
            "block_m": 32,
            "block_n": 32,
            "num_warps": 4,
            "waves_per_eu": 2,
            "head_unroll": 1,
        },
    },
    16: {
        "dv": {
            "block_m": 32,
            "block_n": 32,
            "num_warps": 8,
            "waves_per_eu": 1,
            "head_unroll": 2,
        },
        "dk": {
            "block_m": 32,
            "block_n": 32,
            "num_warps": 8,
            "waves_per_eu": 0,
            "head_unroll": 2,
        },
    },
}

_COMPRESSED_DKV_CONFIGS = {
    1: {
        "dv": {
            "block_m": 32,
            "block_n": 16,
            "head_group": 4,
            "num_warps": 4,
            "waves_per_eu": 2,
        },
        "dk": {
            "block_m": 32,
            "block_n": 16,
            "head_group": 4,
            "num_warps": 4,
            "waves_per_eu": 2,
        },
    },
    4: {
        "dv": {
            "block_m": 32,
            "block_n": 16,
            "head_group": 16,
            "num_warps": 8,
            "waves_per_eu": 2,
        },
        "dk": {
            "block_m": 32,
            "block_n": 16,
            "head_group": 16,
            "num_warps": 8,
            "waves_per_eu": 2,
        },
    },
    16: {
        "dv": {
            "block_m": 16,
            "block_n": 16,
            "head_group": 16,
            "num_warps": 8,
            "waves_per_eu": 2,
        },
        "dk": {
            "block_m": 16,
            "block_n": 16,
            "head_group": 16,
            "num_warps": 8,
            "waves_per_eu": 2,
        },
    },
}

_PRODUCER_CONFIGS = {
    1: {"num_warps": 4, "waves_per_eu": 0},
    4: {"num_warps": 8, "waves_per_eu": 1},
    16: {"num_warps": 8, "waves_per_eu": 2},
}


@triton.jit
def _hca_forward_kernel(
    query_ptr,
    local_kv_ptr,
    compressed_kv_ptr,
    sink_ptr,
    output_ptr,
    lse_ptr,
    local_score_ptr,
    compressed_score_ptr,
    stride_qb,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_lkb,
    stride_lkm,
    stride_lkd,
    stride_ckb,
    stride_ckm,
    stride_ckd,
    stride_ob,
    stride_om,
    stride_oh,
    stride_od,
    stride_lseb,
    stride_lseh,
    stride_lsem,
    stride_lscb,
    stride_lsch,
    stride_lscm,
    stride_lscw,
    stride_cscb,
    stride_csch,
    stride_cscm,
    stride_cscw,
    SM_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    LOCAL_N: tl.constexpr,
    COMPRESSED_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SEQUENCE_LENGTH: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    HEAD_GROUP: tl.constexpr,
):
    query_tile = tl.program_id(0).to(tl.int32)
    head_group = tl.program_id(1).to(tl.int32)
    batch = tl.program_id(2).to(tl.int32)
    start_m = query_tile * BLOCK_M
    rows = tl.arange(0, BLOCK_M * HEAD_GROUP)
    offs_m = start_m + rows % BLOCK_M
    heads = head_group * HEAD_GROUP + rows // BLOCK_M
    offs_d = tl.arange(0, HEAD_DIM)

    query_offsets = (
        batch * stride_qb
        + offs_m[:, None] * stride_qm
        + heads[:, None] * stride_qh
        + offs_d[None, :] * stride_qd
    )
    query = tl.load(query_ptr + query_offsets)
    sink = tl.load(sink_ptr + heads).to(tl.float32)
    log2e: tl.constexpr = 1.4426950408889634
    ln2: tl.constexpr = 0.6931471805599453
    max_score = sink * log2e
    denominator = tl.full((BLOCK_M * HEAD_GROUP,), 1.0, tl.float32)
    accumulator = tl.zeros((BLOCK_M * HEAD_GROUP, HEAD_DIM), tl.float32)

    first_key = max(start_m - WINDOW_LEFT, 0)
    first_key = (first_key // LOCAL_N) * LOCAL_N
    key_end = min(start_m + BLOCK_M, SEQUENCE_LENGTH)
    for start_n in tl.range(first_key, key_end, LOCAL_N, loop_unroll_factor=1):
        offs_n = start_n + tl.arange(0, LOCAL_N)
        kv_offsets = (
            batch * stride_lkb
            + offs_n[:, None] * stride_lkm
            + offs_d[None, :] * stride_lkd
        )
        kv = tl.load(local_kv_ptr + kv_offsets)
        scores = tl.dot(query, tl.trans(kv)) * (SM_SCALE * log2e)
        visible = (offs_n[None, :] <= offs_m[:, None]) & (
            offs_n[None, :] >= offs_m[:, None] - WINDOW_LEFT
        )
        band = offs_n[None, :] - offs_m[:, None] + WINDOW_LEFT
        score_offsets = (
            batch * stride_lscb
            + heads[:, None] * stride_lsch
            + offs_m[:, None] * stride_lscm
            + band * stride_lscw
        )
        tl.store(local_score_ptr + score_offsets, scores, mask=visible)
        scores = tl.where(visible, scores, float("-inf"))
        next_max = tl.maximum(max_score, tl.max(scores, axis=1))
        probabilities = tl.math.exp2(scores - next_max[:, None])
        probabilities = tl.where(visible, probabilities, 0.0)
        correction = tl.math.exp2(max_score - next_max)
        accumulator *= correction[:, None]
        denominator = denominator * correction + tl.sum(probabilities, axis=1)
        accumulator = tl.dot(probabilities.to(tl.bfloat16), kv, acc=accumulator)
        max_score = next_max

    compressed_end = (start_m + BLOCK_M) // 128
    for start_n in tl.range(0, compressed_end, COMPRESSED_N, loop_unroll_factor=1):
        offs_n = start_n + tl.arange(0, COMPRESSED_N)
        kv_offsets = (
            batch * stride_ckb
            + offs_n[:, None] * stride_ckm
            + offs_d[None, :] * stride_ckd
        )
        kv = tl.load(compressed_kv_ptr + kv_offsets)
        scores = tl.dot(query, tl.trans(kv)) * (SM_SCALE * log2e)
        visible = offs_n[None, :] < (offs_m[:, None] + 1) // 128
        score_offsets = (
            batch * stride_cscb
            + heads[:, None] * stride_csch
            + offs_m[:, None] * stride_cscm
            + offs_n[None, :] * stride_cscw
        )
        tl.store(compressed_score_ptr + score_offsets, scores, mask=visible)
        scores = tl.where(visible, scores, float("-inf"))
        next_max = tl.maximum(max_score, tl.max(scores, axis=1))
        probabilities = tl.math.exp2(scores - next_max[:, None])
        probabilities = tl.where(visible, probabilities, 0.0)
        correction = tl.math.exp2(max_score - next_max)
        accumulator *= correction[:, None]
        denominator = denominator * correction + tl.sum(probabilities, axis=1)
        accumulator = tl.dot(probabilities.to(tl.bfloat16), kv, acc=accumulator)
        max_score = next_max

    accumulator *= (1.0 / denominator)[:, None]
    output_offsets = (
        batch * stride_ob
        + offs_m[:, None] * stride_om
        + heads[:, None] * stride_oh
        + offs_d[None, :] * stride_od
    )
    tl.store(output_ptr + output_offsets, accumulator)
    lse_offsets = batch * stride_lseb + heads * stride_lseh + offs_m * stride_lsem
    tl.store(lse_ptr + lse_offsets, (max_score + tl.math.log2(denominator)) * ln2)


@triton.jit
def _hca_score_lse_kernel(
    query_ptr,
    local_kv_ptr,
    compressed_kv_ptr,
    sink_ptr,
    lse_ptr,
    local_score_ptr,
    compressed_score_ptr,
    stride_qb,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_lkb,
    stride_lkm,
    stride_lkd,
    stride_ckb,
    stride_ckm,
    stride_ckd,
    stride_lseb,
    stride_lseh,
    stride_lsem,
    stride_lscb,
    stride_lsch,
    stride_lscm,
    stride_lscw,
    stride_cscb,
    stride_csch,
    stride_cscm,
    stride_cscw,
    SM_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    LOCAL_N: tl.constexpr,
    COMPRESSED_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SEQUENCE_LENGTH: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    HEAD_GROUP: tl.constexpr,
):
    query_tile = tl.program_id(0).to(tl.int32)
    head_group = tl.program_id(1).to(tl.int32)
    batch = tl.program_id(2).to(tl.int32)
    start_m = query_tile * BLOCK_M
    rows = tl.arange(0, BLOCK_M * HEAD_GROUP)
    offs_m = start_m + rows % BLOCK_M
    heads = head_group * HEAD_GROUP + rows // BLOCK_M
    offs_d = tl.arange(0, HEAD_DIM)
    query = tl.load(
        query_ptr
        + batch * stride_qb
        + offs_m[:, None] * stride_qm
        + heads[:, None] * stride_qh
        + offs_d[None, :] * stride_qd
    )
    log2e: tl.constexpr = 1.4426950408889634
    ln2: tl.constexpr = 0.6931471805599453
    max_score = tl.load(sink_ptr + heads).to(tl.float32) * log2e
    denominator = tl.full((BLOCK_M * HEAD_GROUP,), 1.0, tl.float32)

    first_key = max(start_m - WINDOW_LEFT, 0)
    first_key = (first_key // LOCAL_N) * LOCAL_N
    key_end = min(start_m + BLOCK_M, SEQUENCE_LENGTH)
    for start_n in tl.range(first_key, key_end, LOCAL_N, loop_unroll_factor=1):
        offs_n = start_n + tl.arange(0, LOCAL_N)
        kv = tl.load(
            local_kv_ptr
            + batch * stride_lkb
            + offs_n[:, None] * stride_lkm
            + offs_d[None, :] * stride_lkd
        )
        scores = tl.dot(query, tl.trans(kv)) * (SM_SCALE * log2e)
        visible = (offs_n[None, :] <= offs_m[:, None]) & (
            offs_n[None, :] >= offs_m[:, None] - WINDOW_LEFT
        )
        band = offs_n[None, :] - offs_m[:, None] + WINDOW_LEFT
        score_offsets = (
            batch * stride_lscb
            + heads[:, None] * stride_lsch
            + offs_m[:, None] * stride_lscm
            + band * stride_lscw
        )
        tl.store(local_score_ptr + score_offsets, scores, mask=visible)
        scores = tl.where(visible, scores, float("-inf"))
        next_max = tl.maximum(max_score, tl.max(scores, axis=1))
        probabilities = tl.math.exp2(scores - next_max[:, None])
        probabilities = tl.where(visible, probabilities, 0.0)
        correction = tl.math.exp2(max_score - next_max)
        denominator = denominator * correction + tl.sum(probabilities, axis=1)
        max_score = next_max

    compressed_end = (start_m + BLOCK_M) // 128
    for start_n in tl.range(0, compressed_end, COMPRESSED_N, loop_unroll_factor=1):
        offs_n = start_n + tl.arange(0, COMPRESSED_N)
        kv = tl.load(
            compressed_kv_ptr
            + batch * stride_ckb
            + offs_n[:, None] * stride_ckm
            + offs_d[None, :] * stride_ckd
        )
        scores = tl.dot(query, tl.trans(kv)) * (SM_SCALE * log2e)
        visible = offs_n[None, :] < (offs_m[:, None] + 1) // 128
        score_offsets = (
            batch * stride_cscb
            + heads[:, None] * stride_csch
            + offs_m[:, None] * stride_cscm
            + offs_n[None, :] * stride_cscw
        )
        tl.store(compressed_score_ptr + score_offsets, scores, mask=visible)
        scores = tl.where(visible, scores, float("-inf"))
        next_max = tl.maximum(max_score, tl.max(scores, axis=1))
        probabilities = tl.math.exp2(scores - next_max[:, None])
        probabilities = tl.where(visible, probabilities, 0.0)
        correction = tl.math.exp2(max_score - next_max)
        denominator = denominator * correction + tl.sum(probabilities, axis=1)
        max_score = next_max

    lse_offsets = batch * stride_lseb + heads * stride_lseh + offs_m * stride_lsem
    tl.store(lse_ptr + lse_offsets, (max_score + tl.math.log2(denominator)) * ln2)


@triton.jit
def _hca_output_from_scores_kernel(
    local_kv_ptr,
    compressed_kv_ptr,
    lse_ptr,
    output_ptr,
    local_score_ptr,
    compressed_score_ptr,
    stride_lkb,
    stride_lkm,
    stride_lkd,
    stride_ckb,
    stride_ckm,
    stride_ckd,
    stride_lseb,
    stride_lseh,
    stride_lsem,
    stride_ob,
    stride_om,
    stride_oh,
    stride_od,
    stride_lscb,
    stride_lsch,
    stride_lscm,
    stride_lscw,
    stride_cscb,
    stride_csch,
    stride_cscm,
    stride_cscw,
    BLOCK_M: tl.constexpr,
    LOCAL_N: tl.constexpr,
    COMPRESSED_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SEQUENCE_LENGTH: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    HEAD_GROUP: tl.constexpr,
):
    query_tile = tl.program_id(0).to(tl.int32)
    combined_group = tl.program_id(1).to(tl.int32)
    batch = tl.program_id(2).to(tl.int32)
    d_slices: tl.constexpr = HEAD_DIM // BLOCK_D
    d_slice = combined_group % d_slices
    head_group = combined_group // d_slices
    start_m = query_tile * BLOCK_M
    rows = tl.arange(0, BLOCK_M * HEAD_GROUP)
    offs_m = start_m + rows % BLOCK_M
    heads = head_group * HEAD_GROUP + rows // BLOCK_M
    offs_d = d_slice * BLOCK_D + tl.arange(0, BLOCK_D)
    lse = tl.load(
        lse_ptr + batch * stride_lseb + heads * stride_lseh + offs_m * stride_lsem
    )
    log2e: tl.constexpr = 1.4426950408889634
    accumulator = tl.zeros((BLOCK_M * HEAD_GROUP, BLOCK_D), tl.float32)

    first_key = max(start_m - WINDOW_LEFT, 0)
    first_key = (first_key // LOCAL_N) * LOCAL_N
    key_end = min(start_m + BLOCK_M, SEQUENCE_LENGTH)
    for start_n in tl.range(first_key, key_end, LOCAL_N, loop_unroll_factor=1):
        offs_n = start_n + tl.arange(0, LOCAL_N)
        kv = tl.load(
            local_kv_ptr
            + batch * stride_lkb
            + offs_n[:, None] * stride_lkm
            + offs_d[None, :] * stride_lkd
        )
        visible = (offs_n[None, :] <= offs_m[:, None]) & (
            offs_n[None, :] >= offs_m[:, None] - WINDOW_LEFT
        )
        band = offs_n[None, :] - offs_m[:, None] + WINDOW_LEFT
        score_offsets = (
            batch * stride_lscb
            + heads[:, None] * stride_lsch
            + offs_m[:, None] * stride_lscm
            + band * stride_lscw
        )
        scores = tl.load(local_score_ptr + score_offsets, mask=visible, other=0.0).to(
            tl.float32
        )
        probabilities = tl.math.exp2(scores - lse[:, None] * log2e)
        probabilities = tl.where(visible, probabilities, 0.0)
        accumulator = tl.dot(probabilities.to(tl.bfloat16), kv, acc=accumulator)

    compressed_end = (start_m + BLOCK_M) // 128
    for start_n in tl.range(0, compressed_end, COMPRESSED_N, loop_unroll_factor=1):
        offs_n = start_n + tl.arange(0, COMPRESSED_N)
        kv = tl.load(
            compressed_kv_ptr
            + batch * stride_ckb
            + offs_n[:, None] * stride_ckm
            + offs_d[None, :] * stride_ckd
        )
        visible = offs_n[None, :] < (offs_m[:, None] + 1) // 128
        score_offsets = (
            batch * stride_cscb
            + heads[:, None] * stride_csch
            + offs_m[:, None] * stride_cscm
            + offs_n[None, :] * stride_cscw
        )
        scores = tl.load(
            compressed_score_ptr + score_offsets, mask=visible, other=0.0
        ).to(tl.float32)
        probabilities = tl.math.exp2(scores - lse[:, None] * log2e)
        probabilities = tl.where(visible, probabilities, 0.0)
        accumulator = tl.dot(probabilities.to(tl.bfloat16), kv, acc=accumulator)

    output_offsets = (
        batch * stride_ob
        + offs_m[:, None] * stride_om
        + heads[:, None] * stride_oh
        + offs_d[None, :] * stride_od
    )
    tl.store(output_ptr + output_offsets, accumulator)


@triton.jit
def _hca_dq_kernel(
    local_kv_ptr,
    compressed_kv_ptr,
    output_ptr,
    grad_output_ptr,
    lse_ptr,
    sink_ptr,
    grad_query_ptr,
    sink_partial_ptr,
    local_score_ptr,
    compressed_score_ptr,
    stride_lkb,
    stride_lkm,
    stride_lkd,
    stride_ckb,
    stride_ckm,
    stride_ckd,
    stride_ob,
    stride_om,
    stride_oh,
    stride_od,
    stride_dob,
    stride_dom,
    stride_doh,
    stride_dod,
    stride_dqb,
    stride_dqm,
    stride_dqh,
    stride_dqd,
    stride_lseb,
    stride_lseh,
    stride_lsem,
    stride_spb,
    stride_sph,
    stride_spt,
    stride_lscb,
    stride_lsch,
    stride_lscm,
    stride_lscw,
    stride_cscb,
    stride_csch,
    stride_cscm,
    stride_cscw,
    SM_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    LOCAL_N: tl.constexpr,
    COMPRESSED_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SEQUENCE_LENGTH: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    HEAD_GROUP: tl.constexpr,
):
    query_tile = tl.program_id(0).to(tl.int32)
    head_group = tl.program_id(1).to(tl.int32)
    batch = tl.program_id(2).to(tl.int32)
    start_m = query_tile * BLOCK_M
    rows = tl.arange(0, BLOCK_M * HEAD_GROUP)
    offs_m = start_m + rows % BLOCK_M
    heads = head_group * HEAD_GROUP + rows // BLOCK_M
    offs_d = tl.arange(0, HEAD_DIM)

    output_offsets = (
        batch * stride_ob
        + heads[:, None] * stride_oh
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_od
    )
    grad_output_offsets = (
        batch * stride_dob
        + heads[:, None] * stride_doh
        + offs_m[:, None] * stride_dom
        + offs_d[None, :] * stride_dod
    )
    output = tl.load(output_ptr + output_offsets)
    grad_output = tl.load(grad_output_ptr + grad_output_offsets)
    delta = tl.sum(output.to(tl.float32) * grad_output.to(tl.float32), axis=1)
    lse_offsets = batch * stride_lseb + heads * stride_lseh + offs_m * stride_lsem
    lse = tl.load(lse_ptr + lse_offsets)
    grad_query = tl.zeros((BLOCK_M * HEAD_GROUP, HEAD_DIM), tl.float32)
    sink_delta = tl.zeros((BLOCK_M * HEAD_GROUP,), tl.float32)
    log2e: tl.constexpr = 1.4426950408889634

    first_key = max(start_m - WINDOW_LEFT, 0)
    first_key = (first_key // LOCAL_N) * LOCAL_N
    key_end = min(start_m + BLOCK_M, SEQUENCE_LENGTH)
    for start_n in tl.range(first_key, key_end, LOCAL_N, loop_unroll_factor=1):
        offs_n = start_n + tl.arange(0, LOCAL_N)
        kv_offsets = (
            batch * stride_lkb
            + offs_n[:, None] * stride_lkm
            + offs_d[None, :] * stride_lkd
        )
        kv = tl.load(local_kv_ptr + kv_offsets)
        visible = (offs_n[None, :] <= offs_m[:, None]) & (
            offs_n[None, :] >= offs_m[:, None] - WINDOW_LEFT
        )
        band = offs_n[None, :] - offs_m[:, None] + WINDOW_LEFT
        score_offsets = (
            batch * stride_lscb
            + heads[:, None] * stride_lsch
            + offs_m[:, None] * stride_lscm
            + band * stride_lscw
        )
        scores = tl.load(local_score_ptr + score_offsets, mask=visible, other=0.0).to(
            tl.float32
        )
        probabilities = tl.math.exp2(scores - lse[:, None] * log2e)
        probabilities = tl.where(visible, probabilities, 0.0)
        grad_probabilities = tl.dot(grad_output, tl.trans(kv))
        grad_scores = probabilities * (grad_probabilities - delta[:, None])
        tl.store(local_score_ptr + score_offsets, grad_scores, mask=visible)
        grad_query = tl.dot(grad_scores.to(tl.bfloat16), kv, acc=grad_query)
        sink_delta += tl.sum(probabilities * grad_probabilities, axis=1)

    compressed_end = (start_m + BLOCK_M) // 128
    for start_n in tl.range(0, compressed_end, COMPRESSED_N, loop_unroll_factor=1):
        offs_n = start_n + tl.arange(0, COMPRESSED_N)
        kv_offsets = (
            batch * stride_ckb
            + offs_n[:, None] * stride_ckm
            + offs_d[None, :] * stride_ckd
        )
        kv = tl.load(compressed_kv_ptr + kv_offsets)
        visible = offs_n[None, :] < (offs_m[:, None] + 1) // 128
        score_offsets = (
            batch * stride_cscb
            + heads[:, None] * stride_csch
            + offs_m[:, None] * stride_cscm
            + offs_n[None, :] * stride_cscw
        )
        scores = tl.load(
            compressed_score_ptr + score_offsets, mask=visible, other=0.0
        ).to(tl.float32)
        probabilities = tl.math.exp2(scores - lse[:, None] * log2e)
        probabilities = tl.where(visible, probabilities, 0.0)
        grad_probabilities = tl.dot(grad_output, tl.trans(kv))
        grad_scores = probabilities * (grad_probabilities - delta[:, None])
        tl.store(compressed_score_ptr + score_offsets, grad_scores, mask=visible)
        grad_query = tl.dot(grad_scores.to(tl.bfloat16), kv, acc=grad_query)
        sink_delta += tl.sum(probabilities * grad_probabilities, axis=1)

    grad_query *= SM_SCALE
    grad_query_offsets = (
        batch * stride_dqb
        + heads[:, None] * stride_dqh
        + offs_m[:, None] * stride_dqm
        + offs_d[None, :] * stride_dqd
    )
    tl.store(grad_query_ptr + grad_query_offsets, grad_query)
    sink = tl.load(sink_ptr + heads).to(tl.float32)
    sink_probability = tl.math.exp2(sink * log2e - lse * log2e)
    sink_rows = -sink_probability * sink_delta
    sink_partials = tl.sum(sink_rows.reshape((HEAD_GROUP, BLOCK_M)), axis=1)
    group_heads = head_group * HEAD_GROUP + tl.arange(0, HEAD_GROUP)
    tl.store(
        sink_partial_ptr
        + batch * stride_spb
        + group_heads * stride_sph
        + query_tile * stride_spt,
        sink_partials,
    )


@triton.jit
def _hca_ds_kernel(
    local_kv_ptr,
    compressed_kv_ptr,
    output_ptr,
    grad_output_ptr,
    lse_ptr,
    sink_ptr,
    sink_partial_ptr,
    local_score_ptr,
    compressed_score_ptr,
    stride_lkb,
    stride_lkm,
    stride_lkd,
    stride_ckb,
    stride_ckm,
    stride_ckd,
    stride_ob,
    stride_om,
    stride_oh,
    stride_od,
    stride_dob,
    stride_dom,
    stride_doh,
    stride_dod,
    stride_lseb,
    stride_lseh,
    stride_lsem,
    stride_spb,
    stride_sph,
    stride_spt,
    stride_lscb,
    stride_lsch,
    stride_lscm,
    stride_lscw,
    stride_cscb,
    stride_csch,
    stride_cscm,
    stride_cscw,
    BLOCK_M: tl.constexpr,
    LOCAL_N: tl.constexpr,
    COMPRESSED_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SEQUENCE_LENGTH: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    HEAD_GROUP: tl.constexpr,
):
    query_tile = tl.program_id(0).to(tl.int32)
    head_group = tl.program_id(1).to(tl.int32)
    batch = tl.program_id(2).to(tl.int32)
    start_m = query_tile * BLOCK_M
    rows = tl.arange(0, BLOCK_M * HEAD_GROUP)
    offs_m = start_m + rows % BLOCK_M
    heads = head_group * HEAD_GROUP + rows // BLOCK_M
    offs_d = tl.arange(0, HEAD_DIM)
    output = tl.load(
        output_ptr
        + batch * stride_ob
        + heads[:, None] * stride_oh
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_od
    )
    grad_output = tl.load(
        grad_output_ptr
        + batch * stride_dob
        + heads[:, None] * stride_doh
        + offs_m[:, None] * stride_dom
        + offs_d[None, :] * stride_dod
    )
    delta = tl.sum(output.to(tl.float32) * grad_output.to(tl.float32), axis=1)
    lse = tl.load(
        lse_ptr + batch * stride_lseb + heads * stride_lseh + offs_m * stride_lsem
    )
    sink_delta = tl.zeros((BLOCK_M * HEAD_GROUP,), tl.float32)
    log2e: tl.constexpr = 1.4426950408889634

    first_key = max(start_m - WINDOW_LEFT, 0)
    first_key = (first_key // LOCAL_N) * LOCAL_N
    key_end = min(start_m + BLOCK_M, SEQUENCE_LENGTH)
    for start_n in tl.range(first_key, key_end, LOCAL_N, loop_unroll_factor=1):
        offs_n = start_n + tl.arange(0, LOCAL_N)
        kv = tl.load(
            local_kv_ptr
            + batch * stride_lkb
            + offs_n[:, None] * stride_lkm
            + offs_d[None, :] * stride_lkd
        )
        visible = (offs_n[None, :] <= offs_m[:, None]) & (
            offs_n[None, :] >= offs_m[:, None] - WINDOW_LEFT
        )
        band = offs_n[None, :] - offs_m[:, None] + WINDOW_LEFT
        score_offsets = (
            batch * stride_lscb
            + heads[:, None] * stride_lsch
            + offs_m[:, None] * stride_lscm
            + band * stride_lscw
        )
        scores = tl.load(local_score_ptr + score_offsets, mask=visible, other=0.0).to(
            tl.float32
        )
        probabilities = tl.math.exp2(scores - lse[:, None] * log2e)
        probabilities = tl.where(visible, probabilities, 0.0)
        grad_probabilities = tl.dot(grad_output, tl.trans(kv))
        grad_scores = probabilities * (grad_probabilities - delta[:, None])
        tl.store(local_score_ptr + score_offsets, grad_scores, mask=visible)
        sink_delta += tl.sum(probabilities * grad_probabilities, axis=1)

    compressed_end = (start_m + BLOCK_M) // 128
    for start_n in tl.range(0, compressed_end, COMPRESSED_N, loop_unroll_factor=1):
        offs_n = start_n + tl.arange(0, COMPRESSED_N)
        kv = tl.load(
            compressed_kv_ptr
            + batch * stride_ckb
            + offs_n[:, None] * stride_ckm
            + offs_d[None, :] * stride_ckd
        )
        visible = offs_n[None, :] < (offs_m[:, None] + 1) // 128
        score_offsets = (
            batch * stride_cscb
            + heads[:, None] * stride_csch
            + offs_m[:, None] * stride_cscm
            + offs_n[None, :] * stride_cscw
        )
        scores = tl.load(
            compressed_score_ptr + score_offsets, mask=visible, other=0.0
        ).to(tl.float32)
        probabilities = tl.math.exp2(scores - lse[:, None] * log2e)
        probabilities = tl.where(visible, probabilities, 0.0)
        grad_probabilities = tl.dot(grad_output, tl.trans(kv))
        grad_scores = probabilities * (grad_probabilities - delta[:, None])
        tl.store(compressed_score_ptr + score_offsets, grad_scores, mask=visible)
        sink_delta += tl.sum(probabilities * grad_probabilities, axis=1)

    sink = tl.load(sink_ptr + heads).to(tl.float32)
    sink_probability = tl.math.exp2(sink * log2e - lse * log2e)
    sink_rows = -sink_probability * sink_delta
    sink_partials = tl.sum(sink_rows.reshape((HEAD_GROUP, BLOCK_M)), axis=1)
    group_heads = head_group * HEAD_GROUP + tl.arange(0, HEAD_GROUP)
    tl.store(
        sink_partial_ptr
        + batch * stride_spb
        + group_heads * stride_sph
        + query_tile * stride_spt,
        sink_partials,
    )


@triton.jit
def _hca_dq_from_ds_kernel(
    local_kv_ptr,
    compressed_kv_ptr,
    grad_query_ptr,
    local_score_ptr,
    compressed_score_ptr,
    stride_lkb,
    stride_lkm,
    stride_lkd,
    stride_ckb,
    stride_ckm,
    stride_ckd,
    stride_dqb,
    stride_dqm,
    stride_dqh,
    stride_dqd,
    stride_lscb,
    stride_lsch,
    stride_lscm,
    stride_lscw,
    stride_cscb,
    stride_csch,
    stride_cscm,
    stride_cscw,
    SM_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    LOCAL_N: tl.constexpr,
    COMPRESSED_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SEQUENCE_LENGTH: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    HEAD_GROUP: tl.constexpr,
):
    query_tile = tl.program_id(0).to(tl.int32)
    combined_group = tl.program_id(1).to(tl.int32)
    batch = tl.program_id(2).to(tl.int32)
    d_slices: tl.constexpr = HEAD_DIM // BLOCK_D
    d_slice = combined_group % d_slices
    head_group = combined_group // d_slices
    start_m = query_tile * BLOCK_M
    rows = tl.arange(0, BLOCK_M * HEAD_GROUP)
    offs_m = start_m + rows % BLOCK_M
    heads = head_group * HEAD_GROUP + rows // BLOCK_M
    offs_d = d_slice * BLOCK_D + tl.arange(0, BLOCK_D)
    grad_query = tl.zeros((BLOCK_M * HEAD_GROUP, BLOCK_D), tl.float32)

    first_key = max(start_m - WINDOW_LEFT, 0)
    first_key = (first_key // LOCAL_N) * LOCAL_N
    key_end = min(start_m + BLOCK_M, SEQUENCE_LENGTH)
    for start_n in tl.range(first_key, key_end, LOCAL_N, loop_unroll_factor=1):
        offs_n = start_n + tl.arange(0, LOCAL_N)
        kv = tl.load(
            local_kv_ptr
            + batch * stride_lkb
            + offs_n[:, None] * stride_lkm
            + offs_d[None, :] * stride_lkd
        )
        visible = (offs_n[None, :] <= offs_m[:, None]) & (
            offs_n[None, :] >= offs_m[:, None] - WINDOW_LEFT
        )
        band = offs_n[None, :] - offs_m[:, None] + WINDOW_LEFT
        grad_scores = tl.load(
            local_score_ptr
            + batch * stride_lscb
            + heads[:, None] * stride_lsch
            + offs_m[:, None] * stride_lscm
            + band * stride_lscw,
            mask=visible,
            other=0.0,
        ).to(tl.float32)
        grad_query = tl.dot(grad_scores.to(tl.bfloat16), kv, acc=grad_query)

    compressed_end = (start_m + BLOCK_M) // 128
    for start_n in tl.range(0, compressed_end, COMPRESSED_N, loop_unroll_factor=1):
        offs_n = start_n + tl.arange(0, COMPRESSED_N)
        kv = tl.load(
            compressed_kv_ptr
            + batch * stride_ckb
            + offs_n[:, None] * stride_ckm
            + offs_d[None, :] * stride_ckd
        )
        visible = offs_n[None, :] < (offs_m[:, None] + 1) // 128
        grad_scores = tl.load(
            compressed_score_ptr
            + batch * stride_cscb
            + heads[:, None] * stride_csch
            + offs_m[:, None] * stride_cscm
            + offs_n[None, :] * stride_cscw,
            mask=visible,
            other=0.0,
        ).to(tl.float32)
        grad_query = tl.dot(grad_scores.to(tl.bfloat16), kv, acc=grad_query)

    grad_query *= SM_SCALE
    tl.store(
        grad_query_ptr
        + batch * stride_dqb
        + heads[:, None] * stride_dqh
        + offs_m[:, None] * stride_dqm
        + offs_d[None, :] * stride_dqd,
        grad_query,
    )


@triton.jit
def _local_dv_kernel(
    grad_output_ptr,
    lse_ptr,
    grad_kv_ptr,
    score_ptr,
    stride_dob,
    stride_dom,
    stride_doh,
    stride_dod,
    stride_lseb,
    stride_lseh,
    stride_lsem,
    stride_gb,
    stride_gm,
    stride_gd,
    stride_scb,
    stride_sch,
    stride_scm,
    stride_scw,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    QUERY_HEADS: tl.constexpr,
    SEQUENCE_LENGTH: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    HEAD_UNROLL: tl.constexpr,
):
    key_tile = tl.program_id(0).to(tl.int32)
    batch = tl.program_id(1).to(tl.int32)
    start_n = key_tile * BLOCK_N
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    grad_value = tl.zeros((BLOCK_N, HEAD_DIM), tl.float32)
    log2e: tl.constexpr = 1.4426950408889634
    first_query = (start_n // BLOCK_M) * BLOCK_M
    query_end = min(start_n + BLOCK_N + WINDOW_LEFT, SEQUENCE_LENGTH)
    for head in tl.range(0, QUERY_HEADS, 1, loop_unroll_factor=HEAD_UNROLL):
        for start_m in tl.range(first_query, query_end, BLOCK_M, loop_unroll_factor=1):
            offs_m = start_m + tl.arange(0, BLOCK_M)
            do_offsets = (
                batch * stride_dob
                + head * stride_doh
                + offs_m[:, None] * stride_dom
                + offs_d[None, :] * stride_dod
            )
            grad_output = tl.load(grad_output_ptr + do_offsets)
            lse = tl.load(
                lse_ptr
                + batch * stride_lseb
                + head * stride_lseh
                + offs_m * stride_lsem
            )
            visible = (offs_n[None, :] <= offs_m[:, None]) & (
                offs_n[None, :] >= offs_m[:, None] - WINDOW_LEFT
            )
            band = offs_n[None, :] - offs_m[:, None] + WINDOW_LEFT
            score_offsets = (
                batch * stride_scb
                + head * stride_sch
                + offs_m[:, None] * stride_scm
                + band * stride_scw
            )
            scores = tl.load(score_ptr + score_offsets, mask=visible, other=0.0).to(
                tl.float32
            )
            probabilities = tl.math.exp2(scores - lse[:, None] * log2e)
            probabilities = tl.where(visible, probabilities, 0.0)
            grad_value = tl.dot(
                tl.trans(probabilities.to(tl.bfloat16)), grad_output, acc=grad_value
            )
    grad_offsets = (
        batch * stride_gb + offs_n[:, None] * stride_gm + offs_d[None, :] * stride_gd
    )
    tl.store(grad_kv_ptr + grad_offsets, grad_value)


@triton.jit
def _local_dk_kernel(
    query_ptr,
    grad_kv_ptr,
    grad_score_ptr,
    stride_qb,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_gb,
    stride_gm,
    stride_gd,
    stride_scb,
    stride_sch,
    stride_scm,
    stride_scw,
    SM_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    QUERY_HEADS: tl.constexpr,
    SEQUENCE_LENGTH: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    HEAD_UNROLL: tl.constexpr,
):
    key_tile = tl.program_id(0).to(tl.int32)
    batch = tl.program_id(1).to(tl.int32)
    start_n = key_tile * BLOCK_N
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    grad_key = tl.zeros((BLOCK_N, HEAD_DIM), tl.float32)
    first_query = (start_n // BLOCK_M) * BLOCK_M
    query_end = min(start_n + BLOCK_N + WINDOW_LEFT, SEQUENCE_LENGTH)
    for head in tl.range(0, QUERY_HEADS, 1, loop_unroll_factor=HEAD_UNROLL):
        for start_m in tl.range(first_query, query_end, BLOCK_M, loop_unroll_factor=1):
            offs_m = start_m + tl.arange(0, BLOCK_M)
            query = tl.load(
                query_ptr
                + batch * stride_qb
                + head * stride_qh
                + offs_m[:, None] * stride_qm
                + offs_d[None, :] * stride_qd
            )
            visible = (offs_n[None, :] <= offs_m[:, None]) & (
                offs_n[None, :] >= offs_m[:, None] - WINDOW_LEFT
            )
            band = offs_n[None, :] - offs_m[:, None] + WINDOW_LEFT
            score_offsets = (
                batch * stride_scb
                + head * stride_sch
                + offs_m[:, None] * stride_scm
                + band * stride_scw
            )
            grad_scores = tl.load(
                grad_score_ptr + score_offsets, mask=visible, other=0.0
            ).to(tl.float32)
            grad_key = tl.dot(
                tl.trans(grad_scores.to(tl.bfloat16)), query, acc=grad_key
            )
    grad_offsets = (
        batch * stride_gb + offs_n[:, None] * stride_gm + offs_d[None, :] * stride_gd
    )
    combined = tl.load(grad_kv_ptr + grad_offsets).to(tl.float32)
    combined += grad_key * SM_SCALE
    tl.store(grad_kv_ptr + grad_offsets, combined)


@triton.jit
def _compressed_dv_kernel(
    grad_output_ptr,
    lse_ptr,
    partial_ptr,
    score_ptr,
    stride_dob,
    stride_dom,
    stride_doh,
    stride_dod,
    stride_lseb,
    stride_lseh,
    stride_lsem,
    stride_pb,
    stride_pg,
    stride_pm,
    stride_pd,
    stride_scb,
    stride_sch,
    stride_scm,
    stride_scw,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SEQUENCE_LENGTH: tl.constexpr,
    HEAD_GROUP: tl.constexpr,
):
    key_tile = tl.program_id(0).to(tl.int32)
    head_group = tl.program_id(1).to(tl.int32)
    batch = tl.program_id(2).to(tl.int32)
    start_n = key_tile * BLOCK_N
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    grad_value = tl.zeros((BLOCK_N, HEAD_DIM), tl.float32)
    log2e: tl.constexpr = 1.4426950408889634
    first_query = 128 * start_n + 127
    first_query = (first_query // BLOCK_M) * BLOCK_M
    for head_offset in tl.static_range(0, HEAD_GROUP):
        head = head_group * HEAD_GROUP + head_offset
        for start_m in tl.range(
            first_query, SEQUENCE_LENGTH, BLOCK_M, loop_unroll_factor=1
        ):
            offs_m = start_m + tl.arange(0, BLOCK_M)
            grad_output = tl.load(
                grad_output_ptr
                + batch * stride_dob
                + head * stride_doh
                + offs_m[:, None] * stride_dom
                + offs_d[None, :] * stride_dod
            )
            lse = tl.load(
                lse_ptr
                + batch * stride_lseb
                + head * stride_lseh
                + offs_m * stride_lsem
            )
            visible = offs_n[None, :] < (offs_m[:, None] + 1) // 128
            score_offsets = (
                batch * stride_scb
                + head * stride_sch
                + offs_m[:, None] * stride_scm
                + offs_n[None, :] * stride_scw
            )
            scores = tl.load(score_ptr + score_offsets, mask=visible, other=0.0).to(
                tl.float32
            )
            probabilities = tl.math.exp2(scores - lse[:, None] * log2e)
            probabilities = tl.where(visible, probabilities, 0.0)
            grad_value = tl.dot(
                tl.trans(probabilities.to(tl.bfloat16)), grad_output, acc=grad_value
            )
    partial_offsets = (
        batch * stride_pb
        + head_group * stride_pg
        + offs_n[:, None] * stride_pm
        + offs_d[None, :] * stride_pd
    )
    tl.store(partial_ptr + partial_offsets, grad_value)


@triton.jit
def _compressed_dk_kernel(
    query_ptr,
    partial_ptr,
    grad_score_ptr,
    stride_qb,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_pb,
    stride_pg,
    stride_pm,
    stride_pd,
    stride_scb,
    stride_sch,
    stride_scm,
    stride_scw,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SEQUENCE_LENGTH: tl.constexpr,
    HEAD_GROUP: tl.constexpr,
):
    key_tile = tl.program_id(0).to(tl.int32)
    head_group = tl.program_id(1).to(tl.int32)
    batch = tl.program_id(2).to(tl.int32)
    start_n = key_tile * BLOCK_N
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    grad_key = tl.zeros((BLOCK_N, HEAD_DIM), tl.float32)
    first_query = 128 * start_n + 127
    first_query = (first_query // BLOCK_M) * BLOCK_M
    for head_offset in tl.static_range(0, HEAD_GROUP):
        head = head_group * HEAD_GROUP + head_offset
        for start_m in tl.range(
            first_query, SEQUENCE_LENGTH, BLOCK_M, loop_unroll_factor=1
        ):
            offs_m = start_m + tl.arange(0, BLOCK_M)
            query = tl.load(
                query_ptr
                + batch * stride_qb
                + head * stride_qh
                + offs_m[:, None] * stride_qm
                + offs_d[None, :] * stride_qd
            )
            visible = offs_n[None, :] < (offs_m[:, None] + 1) // 128
            score_offsets = (
                batch * stride_scb
                + head * stride_sch
                + offs_m[:, None] * stride_scm
                + offs_n[None, :] * stride_scw
            )
            grad_scores = tl.load(
                grad_score_ptr + score_offsets, mask=visible, other=0.0
            ).to(tl.float32)
            grad_key = tl.dot(
                tl.trans(grad_scores.to(tl.bfloat16)), query, acc=grad_key
            )
    partial_offsets = (
        batch * stride_pb
        + head_group * stride_pg
        + offs_n[:, None] * stride_pm
        + offs_d[None, :] * stride_pd
    )
    tl.store(partial_ptr + partial_offsets, grad_key)


@triton.jit
def _compressed_reduce_kernel(
    partial_ptr,
    grad_kv_ptr,
    stride_pb,
    stride_pg,
    stride_pm,
    stride_pd,
    stride_gb,
    stride_gm,
    stride_gd,
    HEAD_GROUPS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ADD_KEY: tl.constexpr,
    SM_SCALE: tl.constexpr,
):
    key = tl.program_id(0).to(tl.int32)
    batch = tl.program_id(1).to(tl.int32)
    offs_d = tl.arange(0, HEAD_DIM)
    total = tl.zeros((HEAD_DIM,), tl.float32)
    for group in tl.static_range(0, HEAD_GROUPS):
        total += tl.load(
            partial_ptr
            + batch * stride_pb
            + group * stride_pg
            + key * stride_pm
            + offs_d * stride_pd
        ).to(tl.float32)
    grad_offsets = batch * stride_gb + key * stride_gm + offs_d * stride_gd
    if ADD_KEY:
        total = tl.load(grad_kv_ptr + grad_offsets).to(tl.float32) + total * SM_SCALE
    tl.store(grad_kv_ptr + grad_offsets, total)


@triton.jit
def _sink_reduce_kernel(
    partial_ptr,
    grad_sink_ptr,
    stride_pb,
    stride_ph,
    stride_pt,
    BATCH: tl.constexpr,
    QUERY_TILES: tl.constexpr,
    REDUCE_SIZE: tl.constexpr,
):
    head = tl.program_id(0).to(tl.int32)
    offsets = tl.arange(0, REDUCE_SIZE)
    batch = offsets // QUERY_TILES
    tile = offsets - batch * QUERY_TILES
    values = tl.load(
        partial_ptr + batch * stride_pb + head * stride_ph + tile * stride_pt
    )
    tl.store(grad_sink_ptr + head, tl.sum(values, axis=0))


@triton.jit
def _producer_forward_kernel(
    kv_ptr,
    gate_ptr,
    bias_ptr,
    weight_ptr,
    cos_ptr,
    sin_ptr,
    output_ptr,
    stride_kb,
    stride_ks,
    stride_kd,
    stride_gb,
    stride_gs,
    stride_gd,
    stride_br,
    stride_bd,
    stride_cb,
    stride_cs,
    stride_cp,
    stride_ob,
    stride_oc,
    stride_od,
    RMS_EPS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    entry = tl.program_id(0).to(tl.int32)
    batch = tl.program_id(1).to(tl.int32)
    d = tl.arange(0, HEAD_DIM)
    maximum = tl.full((HEAD_DIM,), float("-inf"), tl.float32)
    denominator = tl.zeros((HEAD_DIM,), tl.float32)
    pooled = tl.zeros((HEAD_DIM,), tl.float32)
    source_start = entry * 128

    for slot in tl.range(0, 128, 1, loop_unroll_factor=1):
        source = source_start + slot
        kv = tl.load(
            kv_ptr + batch * stride_kb + source * stride_ks + d * stride_kd
        ).to(tl.float32)
        score = tl.load(
            gate_ptr + batch * stride_gb + source * stride_gs + d * stride_gd
        ).to(tl.float32)
        score += tl.load(bias_ptr + slot * stride_br + d * stride_bd).to(tl.float32)
        next_max = tl.maximum(maximum, score)
        old_scale = tl.where(maximum == float("-inf"), 0.0, tl.exp(maximum - next_max))
        probability = tl.exp(score - next_max)
        pooled = pooled * old_scale + probability * kv
        denominator = denominator * old_scale + probability
        maximum = next_max

    pooled /= denominator
    variance = tl.sum(pooled * pooled, axis=0) / HEAD_DIM
    rms_inv = tl.rsqrt(variance + RMS_EPS)
    weight = tl.load(weight_ptr + d).to(tl.float32)
    normed = (pooled * rms_inv * weight).to(tl.bfloat16).to(tl.float32)
    pairs = tl.arange(0, HEAD_DIM // 2)
    even, odd = tl.split(tl.reshape(normed, (HEAD_DIM // 2, 2)))
    rope_start_pair: tl.constexpr = (HEAD_DIM - 64) // 2
    rope_pair = pairs - rope_start_pair
    is_rope = pairs >= rope_start_pair
    position = entry * 128
    cos = tl.load(
        cos_ptr + batch * stride_cb + position * stride_cs + rope_pair * stride_cp,
        mask=is_rope,
        other=1.0,
    ).to(tl.float32)
    sin = tl.load(
        sin_ptr + batch * stride_cb + position * stride_cs + rope_pair * stride_cp,
        mask=is_rope,
        other=0.0,
    ).to(tl.float32)
    rotated_even = tl.where(is_rope, even * cos - odd * sin, even)
    rotated_odd = tl.where(is_rope, odd * cos + even * sin, odd)
    output = tl.reshape(tl.join(rotated_even, rotated_odd), (HEAD_DIM,))
    tl.store(
        output_ptr + batch * stride_ob + entry * stride_oc + d * stride_od,
        output,
    )


@triton.jit
def _producer_backward_kernel(
    grad_output_ptr,
    kv_ptr,
    gate_ptr,
    bias_ptr,
    weight_ptr,
    cos_ptr,
    sin_ptr,
    grad_kv_ptr,
    grad_gate_ptr,
    stride_dob,
    stride_doc,
    stride_dod,
    stride_kb,
    stride_ks,
    stride_kd,
    stride_gb,
    stride_gs,
    stride_gd,
    stride_br,
    stride_bd,
    stride_cb,
    stride_cs,
    stride_cp,
    stride_dkb,
    stride_dks,
    stride_dkd,
    stride_dgb,
    stride_dgs,
    stride_dgd,
    RMS_EPS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    entry = tl.program_id(0).to(tl.int32)
    batch = tl.program_id(1).to(tl.int32)
    d = tl.arange(0, HEAD_DIM)
    maximum = tl.full((HEAD_DIM,), float("-inf"), tl.float32)
    denominator = tl.zeros((HEAD_DIM,), tl.float32)
    pooled = tl.zeros((HEAD_DIM,), tl.float32)
    source_start = entry * 128

    for slot in tl.range(0, 128, 1, loop_unroll_factor=1):
        source = source_start + slot
        kv = tl.load(
            kv_ptr + batch * stride_kb + source * stride_ks + d * stride_kd
        ).to(tl.float32)
        score = tl.load(
            gate_ptr + batch * stride_gb + source * stride_gs + d * stride_gd
        ).to(tl.float32)
        score += tl.load(bias_ptr + slot * stride_br + d * stride_bd).to(tl.float32)
        next_max = tl.maximum(maximum, score)
        old_scale = tl.where(maximum == float("-inf"), 0.0, tl.exp(maximum - next_max))
        probability = tl.exp(score - next_max)
        pooled = pooled * old_scale + probability * kv
        denominator = denominator * old_scale + probability
        maximum = next_max
    pooled /= denominator

    grad = tl.load(
        grad_output_ptr + batch * stride_dob + entry * stride_doc + d * stride_dod
    ).to(tl.float32)
    rope_start: tl.constexpr = HEAD_DIM - 64
    is_rope = d >= rope_start
    pair = (d - rope_start) // 2
    partner = tl.where((d & 1) == 0, d + 1, d - 1)
    partner_grad = tl.load(
        grad_output_ptr
        + batch * stride_dob
        + entry * stride_doc
        + partner * stride_dod,
        mask=is_rope,
        other=0.0,
    ).to(tl.float32)
    position = entry * 128
    cos = tl.load(
        cos_ptr + batch * stride_cb + position * stride_cs + pair * stride_cp,
        mask=is_rope,
        other=1.0,
    ).to(tl.float32)
    sin = tl.load(
        sin_ptr + batch * stride_cb + position * stride_cs + pair * stride_cp,
        mask=is_rope,
        other=0.0,
    ).to(tl.float32)
    inverse = tl.where(
        (d & 1) == 0,
        grad * cos + partner_grad * sin,
        grad * cos - partner_grad * sin,
    )
    grad = tl.where(is_rope, inverse, grad)

    variance = tl.sum(pooled * pooled, axis=0) / HEAD_DIM
    rms_inv = tl.rsqrt(variance + RMS_EPS)
    normalized = pooled * rms_inv
    grad_normalized = grad * tl.load(weight_ptr + d).to(tl.float32)
    correction = tl.sum(grad_normalized * normalized, axis=0) / HEAD_DIM
    grad_pooled = rms_inv * (grad_normalized - normalized * correction)

    for slot in tl.range(0, 128, 1, loop_unroll_factor=1):
        source = source_start + slot
        kv = tl.load(
            kv_ptr + batch * stride_kb + source * stride_ks + d * stride_kd
        ).to(tl.float32)
        score = tl.load(
            gate_ptr + batch * stride_gb + source * stride_gs + d * stride_gd
        ).to(tl.float32)
        score += tl.load(bias_ptr + slot * stride_br + d * stride_bd).to(tl.float32)
        probability = tl.exp(score - maximum) / denominator
        grad_kv = probability * grad_pooled
        grad_gate = probability * grad_pooled * (kv - pooled)
        tl.store(
            grad_kv_ptr + batch * stride_dkb + source * stride_dks + d * stride_dkd,
            grad_kv,
        )
        tl.store(
            grad_gate_ptr + batch * stride_dgb + source * stride_dgs + d * stride_dgd,
            grad_gate,
        )


def _require_int32_offsets(name: str, tensor: torch.Tensor) -> None:
    maximum_offset = tensor.storage_offset() + sum(
        (size - 1) * abs(stride) for size, stride in zip(tensor.shape, tensor.stride())
    )
    if maximum_offset > torch.iinfo(torch.int32).max:
        raise ValueError(
            f"{name} element offsets must fit signed int32, got {maximum_offset}"
        )


def _validate_attention_inputs(query, local_kv, compressed_kv, sink) -> None:
    if query.ndim != 4 or local_kv.ndim != 4 or compressed_kv.ndim != 4:
        raise ValueError("HCA query and KV tensors must be rank-4 BSHD")
    batch = query.shape[0]
    if batch not in _SUPPORTED_BATCHES:
        raise ValueError(f"unsupported DeepSeek V4 HCA batch {batch}")
    expected = {
        "query": (batch, _SEQUENCE_LENGTH, _QUERY_HEADS, _HEAD_DIM),
        "local_kv": (batch, _SEQUENCE_LENGTH, _KV_HEADS, _HEAD_DIM),
        "compressed_kv": (batch, _COMPRESSED_LENGTH, _KV_HEADS, _HEAD_DIM),
    }
    for name, tensor in (
        ("query", query),
        ("local_kv", local_kv),
        ("compressed_kv", compressed_kv),
    ):
        if tuple(tensor.shape) != expected[name]:
            raise ValueError(
                f"{name} shape must be {expected[name]}, got {tuple(tensor.shape)}"
            )
        if tensor.dtype != torch.bfloat16:
            raise TypeError(f"{name} must use torch.bfloat16")
        if tensor.stride(-1) != 1:
            raise ValueError(f"{name} requires a contiguous last dimension")
        _require_int32_offsets(name, tensor)
    if (
        tuple(sink.shape) != (_QUERY_HEADS,)
        or sink.dtype != torch.float32
        or sink.stride(0) != 1
    ):
        raise ValueError("sink must be contiguous FP32 with shape [64]")
    if any(tensor.device != query.device for tensor in (local_kv, compressed_kv, sink)):
        raise ValueError("all HCA tensors must share one device")


def _launch_b16_forward_phase(
    query,
    local_kv,
    compressed_kv,
    sink,
    output,
    lse,
    local_scores,
    compressed_scores,
) -> None:
    block_m = 16
    head_group = 4
    _hca_score_lse_kernel[
        (
            triton.cdiv(_SEQUENCE_LENGTH, block_m),
            triton.cdiv(_QUERY_HEADS, head_group),
            query.shape[0],
        )
    ](
        query,
        local_kv,
        compressed_kv,
        sink,
        lse,
        local_scores,
        compressed_scores,
        *query.stride(),
        local_kv.stride(0),
        local_kv.stride(1),
        local_kv.stride(3),
        compressed_kv.stride(0),
        compressed_kv.stride(1),
        compressed_kv.stride(3),
        *lse.stride(),
        *local_scores.stride(),
        *compressed_scores.stride(),
        SM_SCALE=_SOFTMAX_SCALE,
        BLOCK_M=block_m,
        LOCAL_N=16,
        COMPRESSED_N=16,
        HEAD_DIM=_HEAD_DIM,
        SEQUENCE_LENGTH=_SEQUENCE_LENGTH,
        WINDOW_LEFT=_WINDOW_LEFT,
        HEAD_GROUP=head_group,
        num_warps=4,
        num_stages=1,
        waves_per_eu=2,
    )
    block_d = 256
    d_slices = _HEAD_DIM // block_d
    _hca_output_from_scores_kernel[
        (
            triton.cdiv(_SEQUENCE_LENGTH, block_m),
            triton.cdiv(_QUERY_HEADS, head_group) * d_slices,
            query.shape[0],
        )
    ](
        local_kv,
        compressed_kv,
        lse,
        output,
        local_scores,
        compressed_scores,
        local_kv.stride(0),
        local_kv.stride(1),
        local_kv.stride(3),
        compressed_kv.stride(0),
        compressed_kv.stride(1),
        compressed_kv.stride(3),
        *lse.stride(),
        *output.stride(),
        *local_scores.stride(),
        *compressed_scores.stride(),
        BLOCK_M=block_m,
        LOCAL_N=16,
        COMPRESSED_N=16,
        BLOCK_D=block_d,
        HEAD_DIM=_HEAD_DIM,
        SEQUENCE_LENGTH=_SEQUENCE_LENGTH,
        WINDOW_LEFT=_WINDOW_LEFT,
        HEAD_GROUP=head_group,
        num_warps=4,
        num_stages=1,
        waves_per_eu=1,
    )


def _attention_forward(query, local_kv, compressed_kv, sink):
    batch = query.shape[0]
    output = torch.empty_like(query, memory_format=torch.contiguous_format)
    lse = torch.empty(
        (batch, _QUERY_HEADS, _SEQUENCE_LENGTH),
        device=query.device,
        dtype=torch.float32,
    )
    local_scores = torch.empty(
        (batch, _QUERY_HEADS, _SEQUENCE_LENGTH, _WINDOW),
        device=query.device,
        dtype=torch.float16,
    )
    compressed_scores = torch.empty(
        (batch, _QUERY_HEADS, _SEQUENCE_LENGTH, _COMPRESSED_LENGTH),
        device=query.device,
        dtype=torch.float16,
    )
    if batch == 16:
        _launch_b16_forward_phase(
            query,
            local_kv,
            compressed_kv,
            sink,
            output,
            lse,
            local_scores,
            compressed_scores,
        )
        return output, lse, local_scores, compressed_scores

    config = _FORWARD_CONFIGS[batch]
    _hca_forward_kernel[
        (
            triton.cdiv(_SEQUENCE_LENGTH, config["block_m"]),
            triton.cdiv(_QUERY_HEADS, config["head_group"]),
            batch,
        )
    ](
        query,
        local_kv,
        compressed_kv,
        sink,
        output,
        lse,
        local_scores,
        compressed_scores,
        *query.stride(),
        local_kv.stride(0),
        local_kv.stride(1),
        local_kv.stride(3),
        compressed_kv.stride(0),
        compressed_kv.stride(1),
        compressed_kv.stride(3),
        *output.stride(),
        *lse.stride(),
        *local_scores.stride(),
        *compressed_scores.stride(),
        SM_SCALE=_SOFTMAX_SCALE,
        BLOCK_M=config["block_m"],
        LOCAL_N=config["local_n"],
        COMPRESSED_N=config["compressed_n"],
        HEAD_DIM=_HEAD_DIM,
        SEQUENCE_LENGTH=_SEQUENCE_LENGTH,
        WINDOW_LEFT=_WINDOW_LEFT,
        HEAD_GROUP=config["head_group"],
        num_warps=config["num_warps"],
        num_stages=1,
        waves_per_eu=config["waves_per_eu"],
    )
    return output, lse, local_scores, compressed_scores


def _launch_local_owner(
    kernel, query, grad_output, lse, grad_kv, scores, config, *, is_key: bool
):
    block_n = config["block_n"]
    grid = (triton.cdiv(_SEQUENCE_LENGTH, block_n), query.shape[0])
    common = dict(
        BLOCK_M=config["block_m"],
        BLOCK_N=block_n,
        HEAD_DIM=_HEAD_DIM,
        QUERY_HEADS=_QUERY_HEADS,
        SEQUENCE_LENGTH=_SEQUENCE_LENGTH,
        WINDOW_LEFT=_WINDOW_LEFT,
        HEAD_UNROLL=config["head_unroll"],
        num_warps=config["num_warps"],
        num_stages=1,
        waves_per_eu=config["waves_per_eu"],
    )
    if is_key:
        kernel[grid](
            query,
            grad_kv,
            scores,
            *query.stride(),
            grad_kv.stride(0),
            grad_kv.stride(1),
            grad_kv.stride(3),
            *scores.stride(),
            SM_SCALE=_SOFTMAX_SCALE,
            **common,
        )
    else:
        kernel[grid](
            grad_output,
            lse,
            grad_kv,
            scores,
            *grad_output.stride(),
            *lse.stride(),
            grad_kv.stride(0),
            grad_kv.stride(1),
            grad_kv.stride(3),
            *scores.stride(),
            **common,
        )


def _launch_compressed_owner(
    kernel, query, grad_output, lse, partial, scores, config, *, is_key: bool
):
    block_n = config["block_n"]
    head_group = config["head_group"]
    head_groups = triton.cdiv(_QUERY_HEADS, head_group)
    grid = (triton.cdiv(_COMPRESSED_LENGTH, block_n), head_groups, query.shape[0])
    common = dict(
        BLOCK_M=config["block_m"],
        BLOCK_N=block_n,
        HEAD_DIM=_HEAD_DIM,
        SEQUENCE_LENGTH=_SEQUENCE_LENGTH,
        HEAD_GROUP=head_group,
        num_warps=config["num_warps"],
        num_stages=1,
        waves_per_eu=config["waves_per_eu"],
    )
    if is_key:
        kernel[grid](
            query,
            partial,
            scores,
            *query.stride(),
            *partial.stride(),
            *scores.stride(),
            **common,
        )
    else:
        kernel[grid](
            grad_output,
            lse,
            partial,
            scores,
            *grad_output.stride(),
            *lse.stride(),
            *partial.stride(),
            *scores.stride(),
            **common,
        )


def _reduce_compressed_partial(partial, grad_compressed, *, add_key: bool) -> None:
    _compressed_reduce_kernel[(_COMPRESSED_LENGTH, partial.shape[0])](
        partial,
        grad_compressed,
        *partial.stride(),
        grad_compressed.stride(0),
        grad_compressed.stride(1),
        grad_compressed.stride(3),
        HEAD_GROUPS=partial.shape[1],
        HEAD_DIM=_HEAD_DIM,
        ADD_KEY=add_key,
        SM_SCALE=_SOFTMAX_SCALE,
        num_warps=4,
        num_stages=1,
    )


def _launch_b16_query_backward(
    grad_output,
    local_kv,
    compressed_kv,
    output,
    lse,
    sink,
    grad_query,
    sink_partial,
    local_scores,
    compressed_scores,
) -> None:
    block_m = 16
    head_group = 4
    query_tiles = triton.cdiv(_SEQUENCE_LENGTH, block_m)
    _hca_ds_kernel[(query_tiles, triton.cdiv(_QUERY_HEADS, head_group), 16)](
        local_kv,
        compressed_kv,
        output,
        grad_output,
        lse,
        sink,
        sink_partial,
        local_scores,
        compressed_scores,
        local_kv.stride(0),
        local_kv.stride(1),
        local_kv.stride(3),
        compressed_kv.stride(0),
        compressed_kv.stride(1),
        compressed_kv.stride(3),
        *output.stride(),
        *grad_output.stride(),
        *lse.stride(),
        *sink_partial.stride(),
        *local_scores.stride(),
        *compressed_scores.stride(),
        BLOCK_M=block_m,
        LOCAL_N=16,
        COMPRESSED_N=16,
        HEAD_DIM=_HEAD_DIM,
        SEQUENCE_LENGTH=_SEQUENCE_LENGTH,
        WINDOW_LEFT=_WINDOW_LEFT,
        HEAD_GROUP=head_group,
        num_warps=4,
        num_stages=1,
        waves_per_eu=0,
    )
    block_d = 256
    d_slices = _HEAD_DIM // block_d
    _hca_dq_from_ds_kernel[
        (
            query_tiles,
            triton.cdiv(_QUERY_HEADS, head_group) * d_slices,
            16,
        )
    ](
        local_kv,
        compressed_kv,
        grad_query,
        local_scores,
        compressed_scores,
        local_kv.stride(0),
        local_kv.stride(1),
        local_kv.stride(3),
        compressed_kv.stride(0),
        compressed_kv.stride(1),
        compressed_kv.stride(3),
        *grad_query.stride(),
        *local_scores.stride(),
        *compressed_scores.stride(),
        SM_SCALE=_SOFTMAX_SCALE,
        BLOCK_M=block_m,
        LOCAL_N=16,
        COMPRESSED_N=16,
        BLOCK_D=block_d,
        HEAD_DIM=_HEAD_DIM,
        SEQUENCE_LENGTH=_SEQUENCE_LENGTH,
        WINDOW_LEFT=_WINDOW_LEFT,
        HEAD_GROUP=head_group,
        num_warps=4,
        num_stages=1,
        waves_per_eu=1,
    )


def _attention_backward(
    grad_output,
    query,
    local_kv,
    compressed_kv,
    output,
    lse,
    sink,
    local_scores,
    compressed_scores,
):
    batch = query.shape[0]
    query_block_m = 16 if batch == 16 else _DQ_CONFIGS[batch]["block_m"]
    query_tiles = triton.cdiv(_SEQUENCE_LENGTH, query_block_m)
    grad_query = torch.empty_like(query)
    grad_local = torch.empty_like(local_kv)
    grad_compressed = torch.empty_like(compressed_kv)
    compressed_dv_config = _COMPRESSED_DKV_CONFIGS[batch]["dv"]
    compressed_dk_config = _COMPRESSED_DKV_CONFIGS[batch]["dk"]
    if compressed_dv_config["head_group"] != compressed_dk_config["head_group"]:
        raise RuntimeError("HCA compressed dV/dK head groups must match")
    compressed_head_groups = _QUERY_HEADS // compressed_dv_config["head_group"]
    compressed_partial = torch.empty(
        (batch, compressed_head_groups, _COMPRESSED_LENGTH, _HEAD_DIM),
        device=query.device,
        dtype=torch.float32,
    )
    sink_partial = torch.empty(
        (batch, _QUERY_HEADS, query_tiles), device=query.device, dtype=torch.float32
    )
    grad_sink = torch.empty_like(sink)

    _launch_local_owner(
        _local_dv_kernel,
        query,
        grad_output,
        lse,
        grad_local,
        local_scores,
        _LOCAL_DKV_CONFIGS[batch]["dv"],
        is_key=False,
    )
    _launch_compressed_owner(
        _compressed_dv_kernel,
        query,
        grad_output,
        lse,
        compressed_partial,
        compressed_scores,
        compressed_dv_config,
        is_key=False,
    )
    _reduce_compressed_partial(compressed_partial, grad_compressed, add_key=False)

    if batch == 16:
        _launch_b16_query_backward(
            grad_output,
            local_kv,
            compressed_kv,
            output,
            lse,
            sink,
            grad_query,
            sink_partial,
            local_scores,
            compressed_scores,
        )
    else:
        dq_config = _DQ_CONFIGS[batch]
        head_group = dq_config["head_group"]
        _hca_dq_kernel[(query_tiles, triton.cdiv(_QUERY_HEADS, head_group), batch)](
            local_kv,
            compressed_kv,
            output,
            grad_output,
            lse,
            sink,
            grad_query,
            sink_partial,
            local_scores,
            compressed_scores,
            local_kv.stride(0),
            local_kv.stride(1),
            local_kv.stride(3),
            compressed_kv.stride(0),
            compressed_kv.stride(1),
            compressed_kv.stride(3),
            *output.stride(),
            *grad_output.stride(),
            *grad_query.stride(),
            *lse.stride(),
            *sink_partial.stride(),
            *local_scores.stride(),
            *compressed_scores.stride(),
            SM_SCALE=_SOFTMAX_SCALE,
            BLOCK_M=dq_config["block_m"],
            LOCAL_N=dq_config["local_n"],
            COMPRESSED_N=dq_config["compressed_n"],
            HEAD_DIM=_HEAD_DIM,
            SEQUENCE_LENGTH=_SEQUENCE_LENGTH,
            WINDOW_LEFT=_WINDOW_LEFT,
            HEAD_GROUP=head_group,
            num_warps=dq_config["num_warps"],
            num_stages=1,
            waves_per_eu=dq_config["waves_per_eu"],
        )

    _launch_local_owner(
        _local_dk_kernel,
        query,
        grad_output,
        lse,
        grad_local,
        local_scores,
        _LOCAL_DKV_CONFIGS[batch]["dk"],
        is_key=True,
    )
    _launch_compressed_owner(
        _compressed_dk_kernel,
        query,
        grad_output,
        lse,
        compressed_partial,
        compressed_scores,
        compressed_dk_config,
        is_key=True,
    )
    _reduce_compressed_partial(compressed_partial, grad_compressed, add_key=True)

    _sink_reduce_kernel[(_QUERY_HEADS,)](
        sink_partial,
        grad_sink,
        *sink_partial.stride(),
        BATCH=batch,
        QUERY_TILES=query_tiles,
        REDUCE_SIZE=batch * query_tiles,
        num_warps=8,
        num_stages=1,
    )
    return grad_query, grad_local, grad_compressed, grad_sink


class _HCAAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, query, local_kv, compressed_kv, sink):
        output, lse, local_scores, compressed_scores = _attention_forward(
            query, local_kv, compressed_kv, sink
        )
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(
            query,
            local_kv,
            compressed_kv,
            output,
            lse,
            sink,
            local_scores,
            compressed_scores,
        )
        return output

    @staticmethod
    def backward(ctx: Any, grad_output):
        if grad_output is None:
            return None, None, None, None
        (
            query,
            local_kv,
            compressed_kv,
            output,
            lse,
            sink,
            local_scores,
            compressed_scores,
        ) = ctx.saved_tensors
        if grad_output.dtype != torch.bfloat16 or grad_output.stride(-1) != 1:
            raise ValueError(
                "HCA output gradient must be BF16 with a contiguous last dimension"
            )
        _require_int32_offsets("grad_output", grad_output)
        return _attention_backward(
            grad_output,
            query,
            local_kv,
            compressed_kv,
            output,
            lse,
            sink,
            local_scores,
            compressed_scores,
        )


def deepseek_v4_hca_attention(query, local_kv, compressed_kv, sink):
    """Consume model-native BHSD tensors and return contiguous BSHD output."""
    if query.ndim != 4 or local_kv.ndim != 4 or compressed_kv.ndim != 4:
        raise ValueError("HCA query and KV tensors must be rank-4 BHSD")
    query_bshd = query.transpose(1, 2)
    local_bshd = local_kv.transpose(1, 2)
    compressed_bshd = compressed_kv.transpose(1, 2)
    _validate_attention_inputs(query_bshd, local_bshd, compressed_bshd, sink)
    return _HCAAttentionFunction.apply(query_bshd, local_bshd, compressed_bshd, sink)


def _validate_producer_inputs(kv, gate, bias, weight, cos, sin) -> None:
    if kv.ndim != 3 or gate.ndim != 3:
        raise ValueError("HCA producer projections must be rank-3")
    batch = kv.shape[0]
    if batch not in _SUPPORTED_BATCHES:
        raise ValueError(f"unsupported DeepSeek V4 HCA producer batch {batch}")
    expected = (batch, _SEQUENCE_LENGTH, _HEAD_DIM)
    if tuple(kv.shape) != expected or tuple(gate.shape) != expected:
        raise ValueError(f"HCA producer projections must have shape {expected}")
    if any(tensor.dtype != torch.bfloat16 for tensor in (kv, gate, cos, sin)):
        raise TypeError("HCA producer activations and RoPE tables must use BF16")
    if any(
        tensor.dtype not in (torch.bfloat16, torch.float32) for tensor in (bias, weight)
    ):
        raise TypeError("HCA producer bias and RMS weight must use BF16 or FP32")
    if any(tensor.requires_grad for tensor in (bias, weight, cos, sin)):
        raise ValueError(
            "HCA producer bias, RMS weight, and RoPE tables must be frozen"
        )
    if tuple(bias.shape) != (_COMPRESS_RATE, _HEAD_DIM):
        raise ValueError("HCA position bias must have shape [128,512]")
    if tuple(weight.shape) != (_HEAD_DIM,):
        raise ValueError("HCA RMS weight must have shape [512]")
    if (
        cos.ndim != 3
        or sin.shape != cos.shape
        or sin.stride() != cos.stride()
        or cos.shape[0] not in (1, batch)
        or tuple(cos.shape[1:]) != (_SEQUENCE_LENGTH, _ROPE_PAIRS)
    ):
        raise ValueError("HCA cos/sin must have matching [1|B,2048,32] layouts")
    tensors = (kv, gate, bias, weight, cos, sin)
    if any(tensor.device != kv.device for tensor in tensors):
        raise ValueError("all HCA producer tensors must share one device")
    if kv.stride(-1) != 1 or gate.stride(-1) != 1 or weight.stride(0) != 1:
        raise ValueError(
            "HCA producer activations and RMS weight require contiguous last dimensions"
        )
    for name, tensor in zip(("kv", "gate", "bias", "weight", "cos", "sin"), tensors):
        _require_int32_offsets(name, tensor)


def _producer_forward(kv, gate, bias, weight, cos, sin, rms_eps: float):
    batch = kv.shape[0]
    output = torch.empty(
        (batch, _COMPRESSED_LENGTH, _HEAD_DIM), device=kv.device, dtype=torch.bfloat16
    )
    cos_stride_b = 0 if cos.shape[0] == 1 else cos.stride(0)
    config = _PRODUCER_CONFIGS[batch]
    _producer_forward_kernel[(_COMPRESSED_LENGTH, batch)](
        kv,
        gate,
        bias,
        weight,
        cos,
        sin,
        output,
        *kv.stride(),
        *gate.stride(),
        *bias.stride(),
        cos_stride_b,
        cos.stride(1),
        cos.stride(2),
        *output.stride(),
        RMS_EPS=float(rms_eps),
        HEAD_DIM=_HEAD_DIM,
        num_warps=config["num_warps"],
        num_stages=1,
        waves_per_eu=config["waves_per_eu"],
    )
    return output


def _producer_backward(grad_output, kv, gate, bias, weight, cos, sin, rms_eps: float):
    batch = kv.shape[0]
    grad_kv = torch.empty_like(kv)
    grad_gate = torch.empty_like(gate)
    cos_stride_b = 0 if cos.shape[0] == 1 else cos.stride(0)
    config = _PRODUCER_CONFIGS[batch]
    _producer_backward_kernel[(_COMPRESSED_LENGTH, batch)](
        grad_output,
        kv,
        gate,
        bias,
        weight,
        cos,
        sin,
        grad_kv,
        grad_gate,
        *grad_output.stride(),
        *kv.stride(),
        *gate.stride(),
        *bias.stride(),
        cos_stride_b,
        cos.stride(1),
        cos.stride(2),
        *grad_kv.stride(),
        *grad_gate.stride(),
        RMS_EPS=float(rms_eps),
        HEAD_DIM=_HEAD_DIM,
        num_warps=config["num_warps"],
        num_stages=1,
        waves_per_eu=config["waves_per_eu"],
    )
    return grad_kv, grad_gate


class _HCAProducerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, kv, gate, bias, weight, cos, sin, rms_eps: float):
        output = _producer_forward(kv, gate, bias, weight, cos, sin, rms_eps)
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(kv, gate, bias, weight, cos, sin)
        ctx.rms_eps = float(rms_eps)
        return output

    @staticmethod
    def backward(ctx: Any, grad_output):
        if grad_output is None:
            return (None,) * 7
        kv, gate, bias, weight, cos, sin = ctx.saved_tensors
        if grad_output.dtype != torch.bfloat16 or grad_output.stride(-1) != 1:
            raise ValueError(
                "HCA producer gradient must be BF16 with a contiguous last dimension"
            )
        _require_int32_offsets("grad_output", grad_output)
        grad_kv, grad_gate = _producer_backward(
            grad_output, kv, gate, bias, weight, cos, sin, ctx.rms_eps
        )
        return grad_kv, grad_gate, None, None, None, None, None


def deepseek_v4_hca_compress(kv, gate, bias, weight, cos, sin, rms_eps: float):
    """Produce non-overlapping rate-128 shared KV as contiguous BHSD."""
    _validate_producer_inputs(kv, gate, bias, weight, cos, sin)
    return _HCAProducerFunction.apply(
        kv, gate, bias, weight, cos, sin, float(rms_eps)
    ).unsqueeze(1)
