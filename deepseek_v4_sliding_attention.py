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
_TRANSFORMERS_WINDOW = 128
_WINDOW_LEFT = _TRANSFORMERS_WINDOW - 1
_SOFTMAX_SCALE = 1.0 / math.sqrt(_HEAD_DIM)

# Exact-shape launch table. Eight query heads share each compact-KV load while
# eight rows per head keep the logical MFMA/output tile at 64 rows.
_FORWARD_CONFIGS = {
    1: {
        "block_m": 4,
        "block_n": 16,
        "head_group": 16,
        "num_warps": 8,
        "num_stages": 1,
        "waves_per_eu": 2,
    },
    4: {
        "block_m": 8,
        "block_n": 16,
        "head_group": 8,
        "num_warps": 8,
        "num_stages": 1,
        "waves_per_eu": 2,
    },
    16: {
        "block_m": 4,
        "block_n": 16,
        "head_group": 16,
        "num_warps": 8,
        "num_stages": 1,
        "waves_per_eu": 2,
    },
}

# FP16 score-state split geometry. dV consumes scores first, dQ overwrites
# them with dS, and the one-dot dK owner consumes the same compact band.
_DKV_CONFIGS = {
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

_DQ_CONFIGS = {
    1: {
        "block_m": 8,
        "block_n": 16,
        "num_warps": 8,
        "waves_per_eu": 0,
        "head_group": 8,
    },
    4: {
        "block_m": 8,
        "block_n": 16,
        "num_warps": 8,
        "waves_per_eu": 0,
        "head_group": 8,
    },
    16: {
        "block_m": 16,
        "block_n": 16,
        "num_warps": 8,
        "waves_per_eu": 1,
        "head_group": 4,
    },
}


@triton.jit
def _sliding_grouped_forward_kernel(
    query_ptr,
    kv_ptr,
    sink_ptr,
    output_ptr,
    lse_ptr,
    score_ptr,
    stride_qb,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_kvb,
    stride_kvm,
    stride_kvd,
    stride_ob,
    stride_om,
    stride_oh,
    stride_od,
    stride_lseb,
    stride_lseh,
    stride_lsem,
    stride_scb,
    stride_sch,
    stride_scm,
    stride_scw,
    SM_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
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
    first_key = (first_key // BLOCK_N) * BLOCK_N
    key_end = min(start_m + BLOCK_M, SEQUENCE_LENGTH)
    for start_n in tl.range(first_key, key_end, BLOCK_N, loop_unroll_factor=1):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        shared_kv_offsets = (
            batch * stride_kvb
            + offs_n[:, None] * stride_kvm
            + offs_d[None, :] * stride_kvd
        )
        shared_kv = tl.load(kv_ptr + shared_kv_offsets)
        scores = tl.dot(query, tl.trans(shared_kv)) * (SM_SCALE * log2e)
        visible = (offs_n[None, :] <= offs_m[:, None]) & (
            offs_n[None, :] >= offs_m[:, None] - WINDOW_LEFT
        )
        band = offs_n[None, :] - offs_m[:, None] + WINDOW_LEFT
        score_offsets = (
            batch * stride_scb
            + heads[:, None] * stride_sch
            + offs_m[:, None] * stride_scm
            + band * stride_scw
        )
        tl.store(score_ptr + score_offsets, scores, mask=visible)
        scores = tl.where(visible, scores, float("-inf"))
        next_max = tl.maximum(max_score, tl.max(scores, axis=1))
        probabilities = tl.math.exp2(scores - next_max[:, None])
        probabilities = tl.where(visible, probabilities, 0.0)
        correction = tl.math.exp2(max_score - next_max)
        accumulator *= correction[:, None]
        denominator = denominator * correction + tl.sum(probabilities, axis=1)
        accumulator = tl.dot(probabilities.to(tl.bfloat16), shared_kv, acc=accumulator)
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


def _sliding_forward(
    query: torch.Tensor,
    shared_kv: torch.Tensor,
    sink: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = query.shape[0]
    config = _FORWARD_CONFIGS[batch]
    block_m = config["block_m"]
    head_group = config["head_group"]
    output = torch.empty_like(query, memory_format=torch.contiguous_format)
    softmax_lse = torch.empty(
        (batch, _QUERY_HEADS, _SEQUENCE_LENGTH),
        device=query.device,
        dtype=torch.float32,
    )
    scores = torch.empty(
        (batch, _QUERY_HEADS, _SEQUENCE_LENGTH, _TRANSFORMERS_WINDOW),
        device=query.device,
        dtype=torch.float16,
    )
    _sliding_grouped_forward_kernel[
        (
            triton.cdiv(_SEQUENCE_LENGTH, block_m),
            triton.cdiv(_QUERY_HEADS, head_group),
            batch,
        )
    ](
        query,
        shared_kv,
        sink,
        output,
        softmax_lse,
        scores,
        *query.stride(),
        shared_kv.stride(0),
        shared_kv.stride(1),
        shared_kv.stride(3),
        *output.stride(),
        *softmax_lse.stride(),
        *scores.stride(),
        SM_SCALE=_SOFTMAX_SCALE,
        BLOCK_M=block_m,
        BLOCK_N=config["block_n"],
        HEAD_DIM=_HEAD_DIM,
        SEQUENCE_LENGTH=_SEQUENCE_LENGTH,
        WINDOW_LEFT=_WINDOW_LEFT,
        HEAD_GROUP=head_group,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
        waves_per_eu=config["waves_per_eu"],
    )
    return output, softmax_lse, scores


@triton.jit
def _sliding_dq_delta_kernel(
    kv_ptr,
    output_ptr,
    grad_output_ptr,
    lse_ptr,
    sink_ptr,
    grad_query_ptr,
    sink_partial_ptr,
    score_ptr,
    stride_kvb,
    stride_kvm,
    stride_kvd,
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
    stride_scb,
    stride_sch,
    stride_scm,
    stride_scw,
    SM_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
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
    mask_m = offs_m < SEQUENCE_LENGTH
    matrix_mask = mask_m[:, None]

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
    output = tl.load(output_ptr + output_offsets, mask=matrix_mask, other=0.0)
    grad_output = tl.load(
        grad_output_ptr + grad_output_offsets, mask=matrix_mask, other=0.0
    )
    delta = tl.sum(output.to(tl.float32) * grad_output.to(tl.float32), axis=1)

    lse_offsets = batch * stride_lseb + heads * stride_lseh + offs_m * stride_lsem
    lse = tl.load(lse_ptr + lse_offsets, mask=mask_m, other=0.0)
    grad_query = tl.zeros((BLOCK_M * HEAD_GROUP, HEAD_DIM), dtype=tl.float32)
    sink_delta = tl.zeros((BLOCK_M * HEAD_GROUP,), dtype=tl.float32)
    log2e: tl.constexpr = 1.4426950408889634

    first_key = max(start_m - WINDOW_LEFT, 0)
    first_key = (first_key // BLOCK_N) * BLOCK_N
    key_end = min(start_m + BLOCK_M, SEQUENCE_LENGTH)
    for start_n in tl.range(first_key, key_end, BLOCK_N, loop_unroll_factor=1):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < SEQUENCE_LENGTH
        kv_offsets = (
            batch * stride_kvb
            + offs_n[:, None] * stride_kvm
            + offs_d[None, :] * stride_kvd
        )
        shared_kv = tl.load(kv_ptr + kv_offsets, mask=mask_n[:, None], other=0.0)
        visible = (
            mask_m[:, None]
            & mask_n[None, :]
            & (offs_n[None, :] <= offs_m[:, None])
            & (offs_n[None, :] >= offs_m[:, None] - WINDOW_LEFT)
        )
        band = offs_n[None, :] - offs_m[:, None] + WINDOW_LEFT
        score_offsets = (
            batch * stride_scb
            + heads[:, None] * stride_sch
            + offs_m[:, None] * stride_scm
            + band * stride_scw
        )
        scores = tl.load(score_ptr + score_offsets, mask=visible, other=0.0).to(
            tl.float32
        )
        probabilities = tl.math.exp2(scores - lse[:, None] * log2e)
        probabilities = tl.where(visible, probabilities, 0.0)
        grad_probabilities = tl.dot(grad_output, tl.trans(shared_kv))
        grad_scores = probabilities * (grad_probabilities - delta[:, None])
        tl.store(score_ptr + score_offsets, grad_scores, mask=visible)
        grad_query = tl.dot(grad_scores.to(tl.bfloat16), shared_kv, acc=grad_query)
        sink_delta += tl.sum(probabilities * grad_probabilities, axis=1)

    grad_query *= SM_SCALE
    grad_query_offsets = (
        batch * stride_dqb
        + heads[:, None] * stride_dqh
        + offs_m[:, None] * stride_dqm
        + offs_d[None, :] * stride_dqd
    )
    tl.store(
        grad_query_ptr + grad_query_offsets,
        grad_query,
        mask=matrix_mask,
    )
    sink = tl.load(sink_ptr + heads).to(tl.float32)
    sink_probability = tl.math.exp2(sink * log2e - lse * log2e)
    sink_rows = tl.where(mask_m, -sink_probability * sink_delta, 0.0)
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
def _sliding_dv_kernel(
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
    stride_gkvb,
    stride_gkvm,
    stride_gkvd,
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
    mask_n = offs_n < SEQUENCE_LENGTH
    grad_value = tl.zeros((BLOCK_N, HEAD_DIM), dtype=tl.float32)
    log2e: tl.constexpr = 1.4426950408889634

    first_query = (start_n // BLOCK_M) * BLOCK_M
    query_end = min(start_n + BLOCK_N + WINDOW_LEFT, SEQUENCE_LENGTH)
    for head in tl.range(0, QUERY_HEADS, 1, loop_unroll_factor=HEAD_UNROLL):
        for start_m in tl.range(first_query, query_end, BLOCK_M, loop_unroll_factor=1):
            offs_m = start_m + tl.arange(0, BLOCK_M)
            mask_m = offs_m < SEQUENCE_LENGTH
            grad_output_offsets = (
                batch * stride_dob
                + head * stride_doh
                + offs_m[:, None] * stride_dom
                + offs_d[None, :] * stride_dod
            )
            grad_output = tl.load(
                grad_output_ptr + grad_output_offsets,
                mask=mask_m[:, None],
                other=0.0,
            )
            lse_offsets = (
                batch * stride_lseb + head * stride_lseh + offs_m * stride_lsem
            )
            lse = tl.load(lse_ptr + lse_offsets, mask=mask_m, other=0.0)
            visible = (
                mask_m[:, None]
                & mask_n[None, :]
                & (offs_n[None, :] <= offs_m[:, None])
                & (offs_n[None, :] >= offs_m[:, None] - WINDOW_LEFT)
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
                tl.trans(probabilities.to(tl.bfloat16)),
                grad_output,
                acc=grad_value,
            )

    grad_kv_offsets = (
        batch * stride_gkvb
        + offs_n[:, None] * stride_gkvm
        + offs_d[None, :] * stride_gkvd
    )
    tl.store(grad_kv_ptr + grad_kv_offsets, grad_value, mask=mask_n[:, None])


@triton.jit
def _sliding_dk_kernel(
    query_ptr,
    grad_kv_ptr,
    grad_score_ptr,
    stride_qb,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_gkvb,
    stride_gkvm,
    stride_gkvd,
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
    mask_n = offs_n < SEQUENCE_LENGTH
    grad_key = tl.zeros((BLOCK_N, HEAD_DIM), dtype=tl.float32)

    first_query = (start_n // BLOCK_M) * BLOCK_M
    query_end = min(start_n + BLOCK_N + WINDOW_LEFT, SEQUENCE_LENGTH)
    for head in tl.range(0, QUERY_HEADS, 1, loop_unroll_factor=HEAD_UNROLL):
        for start_m in tl.range(first_query, query_end, BLOCK_M, loop_unroll_factor=1):
            offs_m = start_m + tl.arange(0, BLOCK_M)
            mask_m = offs_m < SEQUENCE_LENGTH
            query_offsets = (
                batch * stride_qb
                + head * stride_qh
                + offs_m[:, None] * stride_qm
                + offs_d[None, :] * stride_qd
            )
            query = tl.load(query_ptr + query_offsets, mask=mask_m[:, None], other=0.0)
            visible = (
                mask_m[:, None]
                & mask_n[None, :]
                & (offs_n[None, :] <= offs_m[:, None])
                & (offs_n[None, :] >= offs_m[:, None] - WINDOW_LEFT)
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

    grad_kv_offsets = (
        batch * stride_gkvb
        + offs_n[:, None] * stride_gkvm
        + offs_d[None, :] * stride_gkvd
    )
    combined = tl.load(
        grad_kv_ptr + grad_kv_offsets, mask=mask_n[:, None], other=0.0
    ).to(tl.float32)
    combined += grad_key * SM_SCALE
    tl.store(grad_kv_ptr + grad_kv_offsets, combined, mask=mask_n[:, None])


@triton.jit
def _sliding_sink_reduce_kernel(
    sink_partial_ptr,
    grad_sink_ptr,
    stride_spb,
    stride_sph,
    stride_spt,
    BATCH: tl.constexpr,
    QUERY_TILES: tl.constexpr,
    BLOCK_REDUCE: tl.constexpr,
):
    head = tl.program_id(0).to(tl.int32)
    offsets = tl.arange(0, BLOCK_REDUCE)
    mask = offsets < BATCH * QUERY_TILES
    batch = offsets // QUERY_TILES
    tile = offsets - batch * QUERY_TILES
    partials = tl.load(
        sink_partial_ptr + batch * stride_spb + head * stride_sph + tile * stride_spt,
        mask=mask,
        other=0.0,
    )
    tl.store(grad_sink_ptr + head, tl.sum(partials, axis=0))


def _sliding_backward(
    grad_output: torch.Tensor,
    query: torch.Tensor,
    shared_kv: torch.Tensor,
    output: torch.Tensor,
    softmax_lse: torch.Tensor,
    sink: torch.Tensor,
    scores: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = query.shape[0]
    dq_config = _DQ_CONFIGS[batch]
    dkv_config = _DKV_CONFIGS[batch]
    block_m = dq_config["block_m"]
    block_n = dq_config["block_n"]
    query_tiles = triton.cdiv(_SEQUENCE_LENGTH, block_m)
    grad_query = torch.empty_like(query)
    grad_kv = torch.empty_like(shared_kv)
    sink_partial = torch.empty(
        (batch, _QUERY_HEADS, query_tiles),
        device=query.device,
        dtype=torch.float32,
    )
    grad_sink = torch.empty_like(sink, dtype=torch.float32)

    dv_config = dkv_config["dv"]
    dv_block_n = dv_config["block_n"]
    _sliding_dv_kernel[(triton.cdiv(_SEQUENCE_LENGTH, dv_block_n), batch)](
        grad_output,
        softmax_lse,
        grad_kv,
        scores,
        *grad_output.stride(),
        *softmax_lse.stride(),
        grad_kv.stride(0),
        grad_kv.stride(1),
        grad_kv.stride(3),
        *scores.stride(),
        BLOCK_M=dv_config["block_m"],
        BLOCK_N=dv_block_n,
        HEAD_DIM=_HEAD_DIM,
        QUERY_HEADS=_QUERY_HEADS,
        SEQUENCE_LENGTH=_SEQUENCE_LENGTH,
        WINDOW_LEFT=_WINDOW_LEFT,
        HEAD_UNROLL=dv_config["head_unroll"],
        num_warps=dv_config["num_warps"],
        num_stages=1,
        waves_per_eu=dv_config["waves_per_eu"],
    )

    head_group = dq_config["head_group"]
    _sliding_dq_delta_kernel[
        (query_tiles, triton.cdiv(_QUERY_HEADS, head_group), batch)
    ](
        shared_kv,
        output,
        grad_output,
        softmax_lse,
        sink,
        grad_query,
        sink_partial,
        scores,
        shared_kv.stride(0),
        shared_kv.stride(1),
        shared_kv.stride(3),
        *output.stride(),
        *grad_output.stride(),
        *grad_query.stride(),
        *softmax_lse.stride(),
        *sink_partial.stride(),
        *scores.stride(),
        SM_SCALE=_SOFTMAX_SCALE,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HEAD_DIM=_HEAD_DIM,
        SEQUENCE_LENGTH=_SEQUENCE_LENGTH,
        WINDOW_LEFT=_WINDOW_LEFT,
        HEAD_GROUP=head_group,
        num_warps=dq_config["num_warps"],
        num_stages=1,
        waves_per_eu=dq_config["waves_per_eu"],
    )

    dk_config = dkv_config["dk"]
    dk_block_n = dk_config["block_n"]
    _sliding_dk_kernel[(triton.cdiv(_SEQUENCE_LENGTH, dk_block_n), batch)](
        query,
        grad_kv,
        scores,
        *query.stride(),
        grad_kv.stride(0),
        grad_kv.stride(1),
        grad_kv.stride(3),
        *scores.stride(),
        SM_SCALE=_SOFTMAX_SCALE,
        BLOCK_M=dk_config["block_m"],
        BLOCK_N=dk_block_n,
        HEAD_DIM=_HEAD_DIM,
        QUERY_HEADS=_QUERY_HEADS,
        SEQUENCE_LENGTH=_SEQUENCE_LENGTH,
        WINDOW_LEFT=_WINDOW_LEFT,
        HEAD_UNROLL=dk_config["head_unroll"],
        num_warps=dk_config["num_warps"],
        num_stages=1,
        waves_per_eu=dk_config["waves_per_eu"],
    )

    reduce_block = triton.next_power_of_2(batch * query_tiles)
    _sliding_sink_reduce_kernel[(_QUERY_HEADS,)](
        sink_partial,
        grad_sink,
        *sink_partial.stride(),
        BATCH=batch,
        QUERY_TILES=query_tiles,
        BLOCK_REDUCE=reduce_block,
        num_warps=min(8, max(1, reduce_block // 32)),
        num_stages=1,
    )
    return grad_query, grad_kv, grad_sink


def _require_int32_offsets(name: str, tensor: torch.Tensor) -> None:
    maximum_offset = tensor.storage_offset() + sum(
        (size - 1) * abs(stride) for size, stride in zip(tensor.shape, tensor.stride())
    )
    if maximum_offset > torch.iinfo(torch.int32).max:
        raise ValueError(
            f"{name} element offsets must fit signed int32, got {maximum_offset}"
        )


def _validate_bshd_inputs(
    query: torch.Tensor,
    shared_kv: torch.Tensor,
    sink: torch.Tensor,
) -> None:
    if query.ndim != 4 or shared_kv.ndim != 4:
        raise ValueError("query and shared_kv must be rank-4 BSHD tensors")
    batch = query.shape[0]
    expected_query = (batch, _SEQUENCE_LENGTH, _QUERY_HEADS, _HEAD_DIM)
    expected_kv = (batch, _SEQUENCE_LENGTH, _KV_HEADS, _HEAD_DIM)
    if batch not in _SUPPORTED_BATCHES:
        raise ValueError(
            f"unsupported DeepSeek V4 sliding-attention batch {batch}; "
            f"expected one of {sorted(_SUPPORTED_BATCHES)}"
        )
    if tuple(query.shape) != expected_query:
        raise ValueError(
            f"query shape must be {expected_query}, got {tuple(query.shape)}"
        )
    if tuple(shared_kv.shape) != expected_kv:
        raise ValueError(
            f"shared_kv shape must be {expected_kv}, got {tuple(shared_kv.shape)}"
        )
    if tuple(sink.shape) != (_QUERY_HEADS,):
        raise ValueError(
            f"sink shape must be ({_QUERY_HEADS},), got {tuple(sink.shape)}"
        )
    if query.dtype != torch.bfloat16 or shared_kv.dtype != torch.bfloat16:
        raise TypeError("query and shared_kv must use torch.bfloat16")
    if sink.dtype != torch.float32:
        raise TypeError("sink must use torch.float32")
    if query.device != shared_kv.device or query.device != sink.device:
        raise ValueError("query, shared_kv, and sink must be on one device")
    if query.stride(-1) != 1 or shared_kv.stride(-1) != 1 or sink.stride(0) != 1:
        raise ValueError(
            "query, shared_kv, and sink require contiguous last dimensions"
        )
    _require_int32_offsets("query", query)
    _require_int32_offsets("shared_kv", shared_kv)


class _DeepseekV4SlidingAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        query: torch.Tensor,
        shared_kv: torch.Tensor,
        sink: torch.Tensor,
    ) -> torch.Tensor:
        output, softmax_lse, scores = _sliding_forward(query, shared_kv, sink)
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(query, shared_kv, output, softmax_lse, sink, scores)
        return output

    @staticmethod
    def backward(  # ty: ignore[invalid-method-override]
        ctx: Any,
        grad_output: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if grad_output is None:
            return None, None, None
        query, shared_kv, output, softmax_lse, sink, scores = ctx.saved_tensors
        if grad_output.dtype != torch.bfloat16:
            raise TypeError(
                "DeepSeek V4 sliding-attention output gradient must be BF16"
            )
        if grad_output.stride(-1) != 1:
            raise ValueError(
                "DeepSeek V4 sliding-attention output gradient requires a "
                "contiguous last dimension"
            )
        _require_int32_offsets("grad_output", grad_output)

        return _sliding_backward(
            grad_output,
            query,
            shared_kv,
            output,
            softmax_lse,
            sink,
            scores,
        )


def deepseek_v4_sliding_attention_bshd(
    query: torch.Tensor,
    shared_kv: torch.Tensor,
    sink: torch.Tensor,
) -> torch.Tensor:
    """Run exact-shape DeepSeek sliding MQA and return contiguous BSHD output."""

    _validate_bshd_inputs(query, shared_kv, sink)
    return _DeepseekV4SlidingAttentionFunction.apply(query, shared_kv, sink)


def deepseek_v4_sliding_attention(
    query: torch.Tensor,
    shared_kv: torch.Tensor,
    sink: torch.Tensor,
) -> torch.Tensor:
    """Consume model-native BHSD views and return the attention BSHD contract.

    Both transposes are metadata-only. The kernels consume their supplied
    strides directly and allocate output in BSHD order.
    """

    if query.ndim != 4 or shared_kv.ndim != 4:
        raise ValueError("query and shared_kv must be rank-4 BHSD tensors")
    return deepseek_v4_sliding_attention_bshd(
        query.transpose(1, 2),
        shared_kv.transpose(1, 2),
        sink,
    )
