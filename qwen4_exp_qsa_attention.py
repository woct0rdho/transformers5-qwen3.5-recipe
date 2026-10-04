"""Dense QSA forward for the Qwen4-Exp training shapes.

At S=2048 the indexer's selection is exhaustive, so the twelve indexed-attention layers are ordinary
causal grouped-query attention: 24 query heads over 2 KV heads, head_dim 256, partial RoPE applied
before the call, and a per-head sigmoid gate applied to the attention output. This module implements
exactly that contract for the batches the training uses, and nothing else.

A program owns a tile of query positions across a group of query heads that share one KV head, so a
K/V tile is loaded once per group, and the whole 256-wide head dimension stays inside the program
instead of being split across warps. Causal masking is the loop's upper bound per tile, right
padding is a per-sample key bound rather than an additive mask, and the epilogue applies the gate
and stores straight into the layout `o_proj` consumes, so no transpose copy is needed.
"""

import math
from typing import Any

import torch
import triton
import triton.language as tl

# The dK/dV owner is the Gluon kernel. The Triton one below is kept for reference.
from qwen4_exp_qsa_gluon import GLUON_DKDV_CONFIG, qsa_dkdv_gluon

_SUPPORTED_BATCHES = frozenset({1, 4, 16})
_SEQUENCE_LENGTH = 2048
_QUERY_HEADS = 24
_KV_HEADS = 2
_HEAD_DIM = 256
_GROUP_SIZE = _QUERY_HEADS // _KV_HEADS
_QUERY_FEATURES = _QUERY_HEADS * _HEAD_DIM
_SOFTMAX_SCALE = 1.0 / math.sqrt(_HEAD_DIM)

# Exact-shape launch table. HEAD_GROUP query heads share one K/V tile load, and BLOCK_M * HEAD_GROUP
# is the number of logical rows, which stays at or below 64 so the FP32 accumulator fits the
# register file at HEAD_DIM=256.
_FORWARD_CONFIGS: dict[int, dict[str, Any]] = {
    1: {
        "block_m": 16,
        "block_n": 16,
        "head_group": 4,
        "num_warps": 4,
        "num_stages": 1,
        "waves_per_eu": 1,
    },
    4: {
        "block_m": 16,
        "block_n": 16,
        "head_group": 4,
        "num_warps": 4,
        "num_stages": 1,
        "waves_per_eu": 1,
    },
    16: {
        "block_m": 16,
        "block_n": 16,
        "head_group": 4,
        "num_warps": 4,
        "num_stages": 1,
        "waves_per_eu": 1,
    },
}


@triton.jit
def _online_softmax_update(scores, value, max_score, denominator, accumulator):
    """One online-softmax step: rescale, accumulate, and update the running max."""
    next_max = tl.maximum(max_score, tl.max(scores, axis=1))
    probabilities = tl.math.exp2(scores - next_max[:, None])
    correction = tl.math.exp2(max_score - next_max)
    accumulator = accumulator * correction[:, None]
    denominator = denominator * correction + tl.sum(probabilities, axis=1)
    accumulator = tl.dot(probabilities.to(tl.bfloat16), value, acc=accumulator)
    return next_max, denominator, accumulator


@triton.jit
def _qsa_forward_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    gate_ptr,
    key_end_ptr,
    output_ptr,
    lse_ptr,
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
    stride_gb,
    stride_gm,
    stride_ob,
    stride_om,
    stride_lb,
    stride_lm,
    SM_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_GROUP: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    APPLY_GATE: tl.constexpr,
    ROWS: tl.constexpr,
):
    log2e: tl.constexpr = 1.4426950408889634
    ln2: tl.constexpr = 0.6931471805599453

    query_tile = tl.program_id(0).to(tl.int32)
    head_group = tl.program_id(1).to(tl.int32)
    batch = tl.program_id(2).to(tl.int32)

    start_m = query_tile * BLOCK_M
    rows = tl.arange(0, ROWS)
    row_valid = rows < BLOCK_M * HEAD_GROUP
    offs_m = start_m + rows % BLOCK_M
    heads = head_group * HEAD_GROUP + rows // BLOCK_M
    offs_d = tl.arange(0, HEAD_DIM)
    kv_head = (head_group * HEAD_GROUP) // GROUP_SIZE

    query_offsets = (
        batch * stride_qb
        + heads[:, None] * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qd
    )
    query = tl.load(query_ptr + query_offsets, mask=row_valid[:, None], other=0.0)

    key_end = tl.load(key_end_ptr + batch)

    max_score = tl.full((ROWS,), float("-inf"), tl.float32)
    denominator = tl.zeros((ROWS,), tl.float32)
    accumulator = tl.zeros((ROWS, HEAD_DIM), tl.float32)

    # Keys below the tile's first row are visible to every row, so those iterations need neither a
    # load mask nor a causal test: every key in them is below the diagonal and below the padding
    # bound. Only the diagonal tiles and the padded tail take the masked path.
    diagonal_start = (start_m // BLOCK_N) * BLOCK_N
    key_cap = tl.minimum(start_m + BLOCK_M, key_end)
    # Round down: a tile in the unmasked range must end at or below its upper bound, otherwise its
    # overhang would include keys after the query rows or past the padding bound.
    unmasked_end = (tl.minimum(diagonal_start, key_cap) // BLOCK_N) * BLOCK_N
    for start_n in tl.range(0, unmasked_end, BLOCK_N, loop_unroll_factor=1):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        key_offsets = (
            batch * stride_kb
            + kv_head * stride_kh
            + offs_n[:, None] * stride_km
            + offs_d[None, :] * stride_kd
        )
        key = tl.load(key_ptr + key_offsets)
        scores = tl.dot(query, tl.trans(key)) * (SM_SCALE * log2e)
        value_offsets = (
            batch * stride_vb
            + kv_head * stride_vh
            + offs_n[:, None] * stride_vm
            + offs_d[None, :] * stride_vd
        )
        value = tl.load(value_ptr + value_offsets)
        max_score, denominator, accumulator = _online_softmax_update(
            scores, value, max_score, denominator, accumulator
        )

    for start_n in tl.range(unmasked_end, key_cap, BLOCK_N, loop_unroll_factor=1):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        key_in_range = offs_n < key_end
        key_offsets = (
            batch * stride_kb
            + kv_head * stride_kh
            + offs_n[:, None] * stride_km
            + offs_d[None, :] * stride_kd
        )
        key = tl.load(key_ptr + key_offsets, mask=key_in_range[:, None], other=0.0)
        scores = tl.dot(query, tl.trans(key)) * (SM_SCALE * log2e)
        value_offsets = (
            batch * stride_vb
            + kv_head * stride_vh
            + offs_n[:, None] * stride_vm
            + offs_d[None, :] * stride_vd
        )
        value = tl.load(
            value_ptr + value_offsets, mask=key_in_range[:, None], other=0.0
        )
        visible = (
            (offs_n[None, :] <= offs_m[:, None])
            & key_in_range[None, :]
            & row_valid[:, None]
        )
        scores = tl.where(visible, scores, float("-inf"))
        max_score, denominator, accumulator = _online_softmax_update(
            scores, value, max_score, denominator, accumulator
        )

    accumulator *= (1.0 / denominator)[:, None]

    if APPLY_GATE:
        gate_offsets = (
            batch * stride_gb
            + offs_m[:, None] * stride_gm
            + heads[:, None] * HEAD_DIM
            + offs_d[None, :]
        )
        gate = tl.load(gate_ptr + gate_offsets, mask=row_valid[:, None], other=0.0).to(
            tl.float32
        )
        accumulator *= tl.sigmoid(gate)

    output_offsets = (
        batch * stride_ob
        + offs_m[:, None] * stride_om
        + heads[:, None] * HEAD_DIM
        + offs_d[None, :]
    )
    tl.store(output_ptr + output_offsets, accumulator, mask=row_valid[:, None])

    lse_offsets = batch * stride_lb + heads * stride_lm + offs_m
    tl.store(
        lse_ptr + lse_offsets,
        (max_score + tl.math.log2(denominator)) * ln2,
        mask=row_valid,
    )


def _resolve_config(
    batch: int, override: dict[str, Any] | None = None
) -> dict[str, Any]:
    config = dict(_FORWARD_CONFIGS[batch])
    if override:
        config.update(override)
    if _GROUP_SIZE % config["head_group"] != 0:
        raise ValueError(
            f"head_group must divide the {_GROUP_SIZE}-head GQA group, got {config['head_group']}"
        )
    if config["block_m"] * config["head_group"] > 64:
        raise ValueError(
            "block_m * head_group must not exceed 64 logical rows at head_dim 256, got "
            f"{config['block_m'] * config['head_group']}"
        )
    return config


def _qsa_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor | None,
    key_end: torch.Tensor,
    override: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = query.shape[0]
    config = _resolve_config(batch, override)
    head_group = config["head_group"]
    output = torch.empty(
        (batch, _SEQUENCE_LENGTH, _QUERY_FEATURES),
        device=query.device,
        dtype=query.dtype,
    )
    softmax_lse = torch.empty(
        (batch, _QUERY_HEADS, _SEQUENCE_LENGTH),
        device=query.device,
        dtype=torch.float32,
    )
    grid = (
        triton.cdiv(_SEQUENCE_LENGTH, config["block_m"]),
        _QUERY_HEADS // head_group,
        batch,
    )
    _qsa_forward_kernel[grid](
        query,
        key,
        value,
        gate if gate is not None else query,
        key_end,
        output,
        softmax_lse,
        *query.stride(),
        *key.stride(),
        *value.stride(),
        gate.stride(0) if gate is not None else query.stride(0),
        gate.stride(1) if gate is not None else query.stride(1),
        output.stride(0),
        output.stride(1),
        softmax_lse.stride(0),
        softmax_lse.stride(1),
        SM_SCALE=_SOFTMAX_SCALE,
        BLOCK_M=config["block_m"],
        BLOCK_N=config["block_n"],
        HEAD_DIM=_HEAD_DIM,
        HEAD_GROUP=head_group,
        GROUP_SIZE=_GROUP_SIZE,
        APPLY_GATE=gate is not None,
        ROWS=triton.next_power_of_2(config["block_m"] * head_group),
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
        waves_per_eu=config["waves_per_eu"],
    )
    return output, softmax_lse


def _require_int32_offsets(name: str, tensor: torch.Tensor) -> None:
    if tensor.numel() > 2**31 - 1:
        raise ValueError(
            f"{name} needs offsets beyond signed int32 ({tensor.numel()} elements)"
        )


def _validate_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor | None,
    key_end: torch.Tensor | None,
) -> torch.Tensor:
    for name, tensor in (("query", query), ("key", key), ("value", value)):
        if tensor.dim() != 4:
            raise ValueError(f"{name} must be a rank-4 BHSD tensor")
    batch = query.shape[0]
    if batch not in _SUPPORTED_BATCHES:
        raise ValueError(f"batch {batch} is not one of {sorted(_SUPPORTED_BATCHES)}")
    if tuple(query.shape) != (batch, _QUERY_HEADS, _SEQUENCE_LENGTH, _HEAD_DIM):
        raise ValueError(
            f"query must be [{batch}, {_QUERY_HEADS}, {_SEQUENCE_LENGTH}, {_HEAD_DIM}], got {tuple(query.shape)}"
        )
    if tuple(key.shape) != (batch, _KV_HEADS, _SEQUENCE_LENGTH, _HEAD_DIM):
        raise ValueError(
            f"key must be [{batch}, {_KV_HEADS}, {_SEQUENCE_LENGTH}, {_HEAD_DIM}]"
        )
    if tuple(value.shape) != tuple(key.shape):
        raise ValueError("key and value must share a shape")
    if (
        query.dtype != torch.bfloat16
        or key.dtype != torch.bfloat16
        or value.dtype != torch.bfloat16
    ):
        raise ValueError("query, key and value must be bfloat16")
    # BHSD only, and contiguous. The kernels take every stride from the caller, so a transposed
    # layout would run, but a tile row is then `heads * head_dim` from the next instead of `head_dim`,
    # which measures slower end to end at every batch.
    if not (query.is_contiguous() and key.is_contiguous() and value.is_contiguous()):
        raise ValueError("query, key and value must be contiguous BHSD")
    if not (query.device == key.device == value.device):
        raise ValueError("query, key and value must share a device")
    if gate is not None:
        if tuple(gate.shape) != (batch, _SEQUENCE_LENGTH, _QUERY_FEATURES):
            raise ValueError(
                f"gate must be [{batch}, {_SEQUENCE_LENGTH}, {_QUERY_FEATURES}], got {tuple(gate.shape)}"
            )
        if gate.dtype != torch.bfloat16 or not gate.is_contiguous():
            raise ValueError("gate must be contiguous bfloat16")
        if gate.device != query.device:
            raise ValueError("gate must share the query device")
    for name, tensor in (
        ("query", query),
        ("key", key),
        ("value", value),
        ("gate", gate),
    ):
        if tensor is not None:
            _require_int32_offsets(name, tensor)
    if key_end is None:
        key_end = torch.full(
            (batch,), _SEQUENCE_LENGTH, device=query.device, dtype=torch.int32
        )
    else:
        if tuple(key_end.shape) != (batch,) or key_end.dtype != torch.int32:
            raise ValueError(f"key_end must be int32 of shape [{batch}]")
        if key_end.device != query.device:
            raise ValueError("key_end must share the query device")
        if bool((key_end < 1).any()) or bool((key_end > _SEQUENCE_LENGTH).any()):
            raise ValueError(f"key_end values must be in [1, {_SEQUENCE_LENGTH}]")
    return key_end


def qwen4_exp_qsa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor | None = None,
    key_end: torch.Tensor | None = None,
    override: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense causal QSA forward for the training shapes.

    Returns the attention output in the `o_proj` input layout `[B, S, 6144]` and the softmax LSE in
    `[B, 24, S]` FP32 for the backward. `gate` is `[B, S, 6144]` and is applied as
    `out * sigmoid(gate)` when given. `key_end` is `[B]` int32 and caps the visible keys per sample
    for right padding. The default is the full sequence.
    """
    key_end = _validate_inputs(query, key, value, gate, key_end)
    return _qsa_forward(query, key, value, gate, key_end, override)


def qwen4_exp_qsa_attention_configuration() -> dict[str, Any]:
    """The contract this module claims, for inventory and completeness checks."""
    return {
        "batches": sorted(_SUPPORTED_BATCHES),
        "sequence_length": _SEQUENCE_LENGTH,
        "query_heads": _QUERY_HEADS,
        "kv_heads": _KV_HEADS,
        "head_dim": _HEAD_DIM,
        "group_size": _GROUP_SIZE,
        "scale": _SOFTMAX_SCALE,
        "dtype": "bfloat16",
        "configs": {batch: dict(config) for batch, config in _FORWARD_CONFIGS.items()},
    }


def _reference_qsa_attention_fp32(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor | None = None,
    key_end: torch.Tensor | None = None,
    row_indices: torch.Tensor | None = None,
    row_chunk: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Blockwise FP32 oracle: exact causal softmax per head, in float32, never the kernel's math."""
    batch, heads, sequence, head_dim = query.shape
    if key_end is None:
        key_end = torch.full((batch,), sequence, device=query.device, dtype=torch.int32)
    positions = torch.arange(sequence, device=query.device)
    if row_indices is None:
        row_indices = torch.arange(sequence, device=query.device)
    output = torch.empty(
        (batch, sequence, heads * head_dim), device=query.device, dtype=torch.float32
    )
    lse = torch.empty(
        (batch, heads, sequence), device=query.device, dtype=torch.float32
    )
    key_f32 = key.float().view(batch, _KV_HEADS, 1, sequence, head_dim)
    value_f32 = value.float().view(batch, _KV_HEADS, 1, sequence, head_dim)
    for start in range(0, row_indices.numel(), row_chunk):
        rows = row_indices[start : start + row_chunk]
        query_f32 = (
            query[:, :, rows, :]
            .float()
            .view(batch, _KV_HEADS, _GROUP_SIZE, -1, head_dim)
        )
        scores = torch.matmul(query_f32, key_f32.transpose(-1, -2)) * _SOFTMAX_SCALE
        scores = scores.reshape(batch, _QUERY_HEADS, -1, sequence)
        visible = positions[None, :] <= positions[rows][:, None]
        visible = visible[None, None, :, :] & (
            positions[None, None, None, :] < key_end[:, None, None, None]
        )
        scores = scores.masked_fill(~visible, float("-inf"))
        lse[:, :, rows] = torch.logsumexp(scores, dim=-1)
        probabilities = torch.softmax(scores, dim=-1).view(
            batch, _KV_HEADS, _GROUP_SIZE, -1, sequence
        )
        chunk = torch.matmul(probabilities, value_f32).reshape(
            batch, _QUERY_HEADS, -1, head_dim
        )
        if gate is not None:
            gate_chunk = (
                gate[:, rows, :]
                .float()
                .reshape(batch, -1, _QUERY_HEADS, head_dim)
                .transpose(1, 2)
            )
            chunk = chunk * torch.sigmoid(gate_chunk)
        output[:, rows, :] = chunk.transpose(1, 2).reshape(batch, -1, _QUERY_FEATURES)
    return output, lse


# Backward launch seeds, per batch. The query owner reuses the forward's row layout. The KV owner
# keeps two FP32 [block_n, HEAD_DIM] accumulators plus the key and value tiles in registers, which
# bounds block_n.
_BACKWARD_CONFIGS: dict[int, dict[str, dict[str, Any]]] = {
    1: {
        "dq": {
            "head_group": 4,
            "block_m": 16,
            "block_n": 16,
            "num_warps": 4,
            "num_stages": 2,
        },
        "dkdv": {
            "block_n": 16,
            "block_m": 32,
            "num_warps": 2,
            "num_stages": 1,
            "split": 4,
        },
    },
    4: {
        "dq": {
            "head_group": 4,
            "block_m": 16,
            "block_n": 16,
            "num_warps": 4,
            "num_stages": 2,
        },
        "dkdv": {
            "block_n": 16,
            "block_m": 32,
            "num_warps": 2,
            "num_stages": 1,
            "split": 4,
        },
    },
    16: {
        "dq": {
            "head_group": 4,
            "block_m": 16,
            "block_n": 16,
            "num_warps": 4,
            "num_stages": 1,
        },
        "dkdv": {
            "block_n": 16,
            "block_m": 32,
            "num_warps": 2,
            "num_stages": 1,
            "split": 4,
        },
    },
}
_DELTA_BLOCK_M = 128


@triton.jit
def _qsa_delta_kernel(
    output_ptr,
    doutput_ptr,
    delta_ptr,
    stride_ob,
    stride_om,
    stride_db,
    stride_dm,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Delta[b, h, m] = rowsum(dO * O), the row reduction the two owner kernels need."""
    tile = tl.program_id(0).to(tl.int32)
    head = tl.program_id(1).to(tl.int32)
    batch = tl.program_id(2).to(tl.int32)
    offs_m = tile * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    offsets = (
        batch * stride_ob
        + offs_m[:, None] * stride_om
        + head * HEAD_DIM
        + offs_d[None, :]
    )
    output = tl.load(output_ptr + offsets).to(tl.float32)
    doutput = tl.load(doutput_ptr + offsets).to(tl.float32)
    tl.store(
        delta_ptr + batch * stride_db + head * stride_dm + offs_m,
        tl.sum(output * doutput, axis=1),
    )


@triton.jit
def _qsa_dq_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    doutput_ptr,
    delta_ptr,
    lse_ptr,
    key_end_ptr,
    dquery_ptr,
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
    SM_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_GROUP: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    ROWS: tl.constexpr,
    SEQUENCE_LENGTH: tl.constexpr,
):
    """dQ for one tile of query rows.

    Tiles run in reverse order: the last tiles of the sequence loop over the whole key range while
    the first ones loop over almost nothing, so starting with the heavy tiles keeps the tail of the
    grid short instead of ending on it.
    """
    query_tile = (SEQUENCE_LENGTH // BLOCK_M - 1) - tl.program_id(0).to(tl.int32)
    head_group = tl.program_id(1).to(tl.int32)
    batch = tl.program_id(2).to(tl.int32)
    log2e: tl.constexpr = 1.4426950408889634
    scale2: tl.constexpr = SM_SCALE * log2e

    start_m = query_tile * BLOCK_M
    rows = tl.arange(0, ROWS)
    row_valid = rows < BLOCK_M * HEAD_GROUP
    offs_m = start_m + rows % BLOCK_M
    heads = head_group * HEAD_GROUP + rows // BLOCK_M
    offs_d = tl.arange(0, HEAD_DIM)
    kv_head = (head_group * HEAD_GROUP) // GROUP_SIZE

    query_offsets = (
        batch * stride_qb
        + heads[:, None] * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qd
    )
    query = tl.load(query_ptr + query_offsets, mask=row_valid[:, None], other=0.0)
    output_offsets = (
        batch * stride_ob
        + offs_m[:, None] * stride_om
        + heads[:, None] * HEAD_DIM
        + offs_d[None, :]
    )
    doutput = tl.load(doutput_ptr + output_offsets, mask=row_valid[:, None], other=0.0)
    delta = tl.load(
        delta_ptr + batch * stride_db + heads * stride_dm + offs_m,
        mask=row_valid,
        other=0.0,
    )
    # The stored LSE is natural log. Reconstruction is base 2, like the forward's own path.
    lse = (
        tl.load(
            lse_ptr + batch * stride_lb + heads * stride_lm + offs_m,
            mask=row_valid,
            other=0.0,
        )
        * log2e
    )

    dquery = tl.zeros((ROWS, HEAD_DIM), tl.float32)
    key_end = tl.load(key_end_ptr + batch)
    key_limit = tl.minimum(start_m + BLOCK_M, key_end)
    for start_n in tl.range(0, key_limit, BLOCK_N, loop_unroll_factor=1):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        in_range = offs_n < key_limit
        key_offsets = (
            batch * stride_kb
            + kv_head * stride_kh
            + offs_n[:, None] * stride_km
            + offs_d[None, :] * stride_kd
        )
        key = tl.load(key_ptr + key_offsets, mask=in_range[:, None], other=0.0)
        value_offsets = (
            batch * stride_vb
            + kv_head * stride_vh
            + offs_n[:, None] * stride_vm
            + offs_d[None, :] * stride_vd
        )
        value = tl.load(value_ptr + value_offsets, mask=in_range[:, None], other=0.0)
        scores = tl.dot(query, tl.trans(key)) * scale2
        probabilities = tl.math.exp2(scores - lse[:, None])
        # A key tile that ends at or before the first row of this tile is visible to every row, so
        # the causal comparison per element is only needed on the diagonal band.
        if start_n + BLOCK_N <= start_m:
            probabilities = tl.where(
                in_range[None, :] & row_valid[:, None], probabilities, 0.0
            )
        else:
            probabilities = tl.where(
                (offs_n[None, :] <= offs_m[:, None])
                & in_range[None, :]
                & row_valid[:, None],
                probabilities,
                0.0,
            )
        dprobability = tl.dot(doutput, tl.trans(value))
        # dS is the gradient with respect to the raw logit, so it takes the plain softmax scale.
        # scale2 belongs to the base-2 reconstruction only.
        dscore = probabilities * (dprobability - delta[:, None]) * SM_SCALE
        dquery = tl.dot(dscore.to(tl.bfloat16), key, acc=dquery)
    tl.store(dquery_ptr + query_offsets, dquery, mask=row_valid[:, None])


@triton.jit
def _qsa_dkdv_kernel(
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
    SM_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    SEQUENCE_LENGTH: tl.constexpr,
    SPLIT: tl.constexpr,
    PARTIAL_SPLIT_STRIDE: tl.constexpr,
):
    """dK and dV for one key tile, owned by one program across the whole GQA group.

    Both accumulators live in one pass so the score dot and dP are computed once. Splitting dK and
    dV into two launches would allow a larger BLOCK_N and less Q re-reading, but it repeats the score
    and dP dots, which costs more than the traffic it saves.
    """
    key_tile = tl.program_id(0).to(tl.int32)
    head_split = tl.program_id(1).to(tl.int32)
    kv_head = head_split // SPLIT
    split_index = head_split % SPLIT
    batch = tl.program_id(2).to(tl.int32)
    log2e: tl.constexpr = 1.4426950408889634
    scale2: tl.constexpr = SM_SCALE * log2e

    start_n = key_tile * BLOCK_N
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    key_end = tl.load(key_end_ptr + batch)
    key_offsets = (
        batch * stride_kb
        + kv_head * stride_kh
        + offs_n[:, None] * stride_km
        + offs_d[None, :] * stride_kd
    )
    value_offsets = (
        batch * stride_vb
        + kv_head * stride_vh
        + offs_n[:, None] * stride_vm
        + offs_d[None, :] * stride_vd
    )
    key_in_range = offs_n < key_end
    key = tl.load(key_ptr + key_offsets, mask=key_in_range[:, None], other=0.0)
    value = tl.load(value_ptr + value_offsets, mask=key_in_range[:, None], other=0.0)

    dkey = tl.zeros((BLOCK_N, HEAD_DIM), tl.float32)
    dvalue = tl.zeros((BLOCK_N, HEAD_DIM), tl.float32)
    row_offsets = tl.arange(0, BLOCK_M)
    first_tile = start_n // BLOCK_M
    last_tile = SEQUENCE_LENGTH // BLOCK_M
    # Each split owns a contiguous slice of the query tiles. The key tiles with the smallest index
    # own the largest causal prefix, so without the split the grid's makespan is set by the first
    # key tile alone. With it every program does about 1/SPLIT of that work and the partials are
    # summed deterministically by a reduction kernel.
    split_chunk = (last_tile - first_tile + SPLIT - 1) // SPLIT
    tile_lo = first_tile + split_index * split_chunk
    tile_hi = tl.minimum(tile_lo + split_chunk, last_tile)
    for head_slot in tl.range(0, GROUP_SIZE, loop_unroll_factor=1):
        head = kv_head * GROUP_SIZE + head_slot
        for query_tile in tl.range(tile_lo, tile_hi, 1, loop_unroll_factor=1):
            offs_m = query_tile * BLOCK_M + row_offsets
            row_valid = offs_m < key_end
            query_offsets = (
                batch * stride_qb
                + head * stride_qh
                + offs_m[:, None] * stride_qm
                + offs_d[None, :] * stride_qd
            )
            query = tl.load(
                query_ptr + query_offsets, mask=row_valid[:, None], other=0.0
            )
            output_offsets = (
                batch * stride_ob
                + offs_m[:, None] * stride_om
                + head * HEAD_DIM
                + offs_d[None, :]
            )
            doutput = tl.load(
                doutput_ptr + output_offsets, mask=row_valid[:, None], other=0.0
            )
            # lse and delta are defined for every row the forward produced, padded rows included, so
            # they are loaded unmasked. Masking them to zero is what made exp2 overflow.
            delta = tl.load(delta_ptr + batch * stride_db + head * stride_dm + offs_m)
            lse = (
                tl.load(lse_ptr + batch * stride_lb + head * stride_lm + offs_m) * log2e
            )
            scores = tl.dot(query, tl.trans(key)) * scale2
            probabilities = tl.math.exp2(scores - lse[:, None])
            probabilities = tl.where(
                (offs_m[:, None] >= offs_n[None, :])
                & key_in_range[None, :]
                & row_valid[:, None],
                probabilities,
                0.0,
            )
            dprobability = tl.dot(doutput, tl.trans(value))
            dscore = probabilities * (dprobability - delta[:, None]) * SM_SCALE
            dvalue = tl.dot(
                tl.trans(probabilities).to(tl.bfloat16), doutput, acc=dvalue
            )
            dkey = tl.dot(tl.trans(dscore).to(tl.bfloat16), query, acc=dkey)

    if SPLIT == 1:
        tl.store(dkey_ptr + key_offsets, dkey, mask=key_in_range[:, None])
        tl.store(dvalue_ptr + value_offsets, dvalue, mask=key_in_range[:, None])
    else:
        # Partial sums go to a workspace the reduction kernel owns. dkey_ptr is its base.
        partial_offsets = key_offsets + split_index * PARTIAL_SPLIT_STRIDE
        tl.store(dkey_ptr + partial_offsets, dkey, mask=key_in_range[:, None])
        tl.store(dvalue_ptr + partial_offsets, dvalue, mask=key_in_range[:, None])


@triton.jit
def _qsa_dkdv_reduce_kernel(
    partial_key_ptr,
    partial_value_ptr,
    dkey_ptr,
    dvalue_ptr,
    stride_kb,
    stride_kh,
    stride_km,
    stride_kd,
    stride_sb,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SPLIT: tl.constexpr,
    SPLIT_STRIDE: tl.constexpr,
):
    """Sum the SPLIT partial dK/dV tiles in a fixed order, then store them as BF16."""
    key_tile = tl.program_id(0).to(tl.int32)
    kv_head = tl.program_id(1).to(tl.int32)
    batch = tl.program_id(2).to(tl.int32)
    offs_n = key_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    base = (
        batch * stride_kb
        + kv_head * stride_kh
        + offs_n[:, None] * stride_km
        + offs_d[None, :] * stride_kd
    )
    dkey = tl.zeros((BLOCK_N, HEAD_DIM), tl.float32)
    dvalue = tl.zeros((BLOCK_N, HEAD_DIM), tl.float32)
    for split in tl.range(0, SPLIT, loop_unroll_factor=1):
        offset = base + split * SPLIT_STRIDE
        dkey += tl.load(partial_key_ptr + offset)
        dvalue += tl.load(partial_value_ptr + offset)
    tl.store(dkey_ptr + base, dkey)
    tl.store(dvalue_ptr + base, dvalue)


def _backward_config(
    batch: int, kind: str, override: dict[str, Any] | None = None
) -> dict[str, Any]:
    config = dict(_BACKWARD_CONFIGS[batch][kind])
    if override:
        config.update(override)
    if kind == "dq":
        if _GROUP_SIZE % config["head_group"] != 0:
            raise ValueError(f"head_group must divide the {_GROUP_SIZE}-head group")
        if config["block_m"] * config["head_group"] > 64:
            raise ValueError("block_m * head_group must not exceed 64 logical rows")
    return config


def _qsa_backward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    doutput: torch.Tensor,
    softmax_lse: torch.Tensor,
    key_end: torch.Tensor,
    dq_override: dict[str, Any] | None = None,
    dkdv_override: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = query.shape[0]
    device = query.device
    delta = torch.empty(
        (batch, _QUERY_HEADS, _SEQUENCE_LENGTH), device=device, dtype=torch.float32
    )
    dquery = torch.empty_like(query)
    dkey = torch.empty_like(key)
    dvalue = torch.empty_like(value)
    _qsa_delta_kernel[
        (triton.cdiv(_SEQUENCE_LENGTH, _DELTA_BLOCK_M), _QUERY_HEADS, batch)
    ](
        output,
        doutput,
        delta,
        output.stride(0),
        output.stride(1),
        delta.stride(0),
        delta.stride(1),
        HEAD_DIM=_HEAD_DIM,
        BLOCK_M=_DELTA_BLOCK_M,
        num_warps=4,
    )

    dq_config = _backward_config(batch, "dq", dq_override)
    head_group = dq_config["head_group"]
    _qsa_dq_kernel[
        (_SEQUENCE_LENGTH // dq_config["block_m"], _QUERY_HEADS // head_group, batch)
    ](
        query,
        key,
        value,
        doutput,
        delta,
        softmax_lse,
        key_end,
        dquery,
        *query.stride(),
        *key.stride(),
        *value.stride(),
        output.stride(0),
        output.stride(1),
        delta.stride(0),
        delta.stride(1),
        softmax_lse.stride(0),
        softmax_lse.stride(1),
        SM_SCALE=_SOFTMAX_SCALE,
        BLOCK_M=dq_config["block_m"],
        BLOCK_N=dq_config["block_n"],
        HEAD_DIM=_HEAD_DIM,
        HEAD_GROUP=head_group,
        GROUP_SIZE=_GROUP_SIZE,
        ROWS=triton.next_power_of_2(dq_config["block_m"] * head_group),
        SEQUENCE_LENGTH=_SEQUENCE_LENGTH,
        num_warps=dq_config["num_warps"],
        num_stages=dq_config["num_stages"],
    )

    dkdv_config = _backward_config(batch, "dkdv", dkdv_override)
    split = int(dkdv_config.get("split", 1))
    if split > 1:
        partials = torch.empty(
            (2, split, batch, _KV_HEADS, _SEQUENCE_LENGTH, _HEAD_DIM),
            device=device,
            dtype=torch.float32,
        )
        partial_key, partial_value = partials[0], partials[1]
        split_stride = partial_key.stride(0)
    else:
        partial_key, partial_value = dkey, dvalue
        split_stride = 0
    if dkdv_override is None:
        qsa_dkdv_gluon(
            query,
            key,
            value,
            doutput,
            delta,
            softmax_lse,
            key_end,
            partial_key,
            partial_value,
            split,
        )
        owner_block_n = GLUON_DKDV_CONFIG["block_n"]
    else:
        _qsa_dkdv_kernel[
            (
                _SEQUENCE_LENGTH // dkdv_config["block_n"],
                _KV_HEADS * split,
                batch,
            )
        ](
            query,
            key,
            value,
            doutput,
            delta,
            softmax_lse,
            key_end,
            partial_key,
            partial_value,
            *query.stride(),
            *key.stride(),
            *value.stride(),
            output.stride(0),
            output.stride(1),
            delta.stride(0),
            delta.stride(1),
            softmax_lse.stride(0),
            softmax_lse.stride(1),
            SM_SCALE=_SOFTMAX_SCALE,
            BLOCK_M=dkdv_config["block_m"],
            BLOCK_N=dkdv_config["block_n"],
            HEAD_DIM=_HEAD_DIM,
            GROUP_SIZE=_GROUP_SIZE,
            SEQUENCE_LENGTH=_SEQUENCE_LENGTH,
            SPLIT=split,
            PARTIAL_SPLIT_STRIDE=split_stride,
            num_warps=dkdv_config["num_warps"],
            num_stages=dkdv_config["num_stages"],
        )
        owner_block_n = dkdv_config["block_n"]
    if split > 1:
        _qsa_dkdv_reduce_kernel[(_SEQUENCE_LENGTH // owner_block_n, _KV_HEADS, batch)](
            partial_key,
            partial_value,
            dkey,
            dvalue,
            dkey.stride(0),
            dkey.stride(1),
            dkey.stride(2),
            dkey.stride(3),
            partial_key.stride(0),
            BLOCK_N=owner_block_n,
            HEAD_DIM=_HEAD_DIM,
            SPLIT=split,
            SPLIT_STRIDE=split_stride,
            num_warps=4,
        )
    return dquery, dkey, dvalue, delta


class _QsaAttentionFunction(torch.autograd.Function):
    """Differentiable wrapper: the forward kernel, then the Delta, dQ and dK/dV kernels."""

    @staticmethod
    def forward(ctx, query, key, value, key_end):
        output, softmax_lse = _qsa_forward(query, key, value, None, key_end)
        ctx.save_for_backward(query, key, value, output, softmax_lse, key_end)
        return output

    @staticmethod
    def backward(ctx, doutput):  # ty: ignore[invalid-method-override]
        query, key, value, output, softmax_lse, key_end = ctx.saved_tensors
        dquery, dkey, dvalue, _ = _qsa_backward(
            query, key, value, output, doutput.contiguous(), softmax_lse, key_end
        )
        return dquery, dkey, dvalue, None


def qwen4_exp_qsa_attention_autograd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_end: torch.Tensor | None = None,
) -> torch.Tensor:
    """Differentiable QSA attention: the forward kernel, then the Delta, dQ and dK/dV kernels.

    `key_end` is `[B]` int32 and caps the visible keys per sample. The default is the full sequence.
    Pad rows are expected to carry no loss, which is what the collator's masked labels give them, so
    their incoming gradient is zero and the backward skips their contribution.
    """
    key_end = _validate_inputs(query, key, value, None, key_end)
    return _QsaAttentionFunction.apply(query, key, value, key_end)
