"""Triton routing-gate selection for the fixed Qwen and DeepSeek MoE shapes.

The optimized training workloads are sequence length 2,048 at physical batches
1, 4, and 16:

* Qwen3.5/3.6: ``[2048|8192|32768, 2048]`` hidden states, 256 experts,
  top-8 selection, and 16,384/65,536/262,144 routed rows.
* DeepSeek V4 learned routers: ``[2048|8192|32768, 4096]`` hidden states,
  256 experts, top-6 sqrt-softplus-plus-correction-bias selection, and
  12,288/49,152/196,608 routed rows.
* DeepSeek V4 hash routers have the same hidden/expert geometry and top-6
  weights, but their expert IDs come from the fixed token lookup table.

The linear projection remains a framework GEMM. The Triton kernel replaces
full-width softmax/sqrt-softplus plus ``torch.topk`` with one 256-expert streaming
selection. Normalization is then evaluated only for the selected 8 or 6
experts, preserving ordinary autograd for router-score gradients.
"""

from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4HashRouter,
    DeepseekV4TopKRouter,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeTopKRouter,
)

_NUM_EXPERTS = 256
_SEQUENCE_LENGTH = 2048
_MAX_TOKENS = 16 * _SEQUENCE_LENGTH
_QWEN_HIDDEN_SIZE = 2048
_QWEN_TOP_K = 8
_DEEPSEEK_HIDDEN_SIZE = 4096
_DEEPSEEK_TOP_K = 6
_TRITON_NUM_EXPERTS = tl.constexpr(256)
_TRITON_TOP_K_PAD = tl.constexpr(8)


def _is_supported_router_geometry(
    num_tokens: int,
    num_experts: int,
    top_k: int,
) -> bool:
    return (
        0 < num_tokens <= _MAX_TOKENS
        and num_experts == _NUM_EXPERTS
        and top_k in {_QWEN_TOP_K, _DEEPSEEK_TOP_K}
    )


def _router_topk_launch(num_tokens: int) -> tuple[int, int, int]:
    """Return ``(BLOCK_M, BLOCK_N, num_warps)`` for the fixed token buckets.

    The expert axis always has 256 entries. A 64-expert streaming tile won for
    both statically specialized top-8 identity scoring and top-6
    sqrt-softplus-plus-bias scoring. Batch 1 uses four rows/four warps. Batches 4
    and 16 use eight rows/eight warps. Larger row tiles increased register
    pressure and regressed both router shapes.
    """

    if num_tokens <= _SEQUENCE_LENGTH:
        return 4, 64, 4
    return 8, 64, 8


@triton.jit
def _float_key(value):
    """Map IEEE floating-point bits to unsigned keys ordered by value."""

    nbits: tl.constexpr = value.dtype.primitive_bitwidth
    unsigned: tl.constexpr = tl.dtype(f"uint{nbits}")
    bits = value.to(unsigned, bitcast=True)
    top = 1 << (nbits - 1)
    full = (1 << nbits) - 1
    return bits ^ tl.where((bits & top) != 0, full, top)


@triton.jit
def _load_router_scores(
    logits,
    correction_bias,
    row_offsets,
    expert_offsets,
    row_mask,
    APPLY_SQRT_SOFTPLUS: tl.constexpr,
    HAS_CORRECTION_BIAS: tl.constexpr,
):
    pointers = (
        logits + row_offsets[:, None] * _TRITON_NUM_EXPERTS + expert_offsets[None, :]
    )
    scores = tl.load(pointers, mask=row_mask, other=float("-inf"))
    if APPLY_SQRT_SOFTPLUS:
        scores = scores.to(tl.float32)
        softplus = tl.maximum(scores, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(scores)))
        scores = tl.sqrt(softplus)
    if HAS_CORRECTION_BIAS:
        bias = tl.load(correction_bias + expert_offsets)[None, :]
        scores += bias
    return scores


@triton.jit
def _streaming_router_topk(
    logits,
    correction_bias,
    row_offsets,
    row_mask,
    BLOCK_N: tl.constexpr,
    APPLY_SQRT_SOFTPLUS: tl.constexpr,
    HAS_CORRECTION_BIAS: tl.constexpr,
):
    """Return eight sorted packed ``(score, inverse-index)`` keys per row."""

    score_dtype: tl.constexpr = (
        tl.float32 if APPLY_SQRT_SOFTPLUS else logits.dtype.element_ty
    )
    score_bits: tl.constexpr = score_dtype.primitive_bitwidth
    key_dtype: tl.constexpr = tl.dtype(f"uint{score_bits * 2}")
    iterations: tl.constexpr = _TRITON_NUM_EXPERTS // BLOCK_N

    expert_offsets = (iterations - 1) * BLOCK_N + tl.arange(0, BLOCK_N)
    scores = _load_router_scores(
        logits,
        correction_bias,
        row_offsets,
        expert_offsets,
        row_mask,
        APPLY_SQRT_SOFTPLUS,
        HAS_CORRECTION_BIAS,
    ).to(score_dtype)
    score_keys = _float_key(scores)
    index_keys = (_TRITON_NUM_EXPERTS - expert_offsets)[None, :]
    packed = (score_keys.to(key_dtype) << 16) | index_keys
    selected = tl.topk(packed, _TRITON_TOP_K_PAD, dim=1)

    for _ in tl.static_range(0, iterations - 1):
        selected = tl.bitonic_merge(selected)
        expert_offsets -= BLOCK_N
        scores = _load_router_scores(
            logits,
            correction_bias,
            row_offsets,
            expert_offsets,
            row_mask,
            APPLY_SQRT_SOFTPLUS,
            HAS_CORRECTION_BIAS,
        ).to(score_dtype)
        score_keys = _float_key(scores)
        index_keys = (_TRITON_NUM_EXPERTS - expert_offsets)[None, :]
        packed = (score_keys.to(key_dtype) << 16) | index_keys
        selected = tl.maximum(
            selected,
            tl.topk(packed, _TRITON_TOP_K_PAD, dim=1),
        )

    return tl.sort(selected, dim=1, descending=True)


@triton.jit
def _router_topk_kernel(
    logits,
    correction_bias,
    indices,
    num_rows,
    TOP_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    APPLY_SQRT_SOFTPLUS: tl.constexpr,
    HAS_CORRECTION_BIAS: tl.constexpr,
):
    row_offsets = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = row_offsets[:, None] < num_rows
    selected = _streaming_router_topk(
        logits,
        correction_bias,
        row_offsets,
        row_mask,
        BLOCK_N,
        APPLY_SQRT_SOFTPLUS,
        HAS_CORRECTION_BIAS,
    )
    rank_offsets = tl.arange(0, _TRITON_TOP_K_PAD)
    inverse_indices = (selected & 0xFFFF).to(tl.int64)
    selected_indices = _TRITON_NUM_EXPERTS - inverse_indices
    tl.store(
        indices + row_offsets[:, None] * TOP_K + rank_offsets[None, :],
        selected_indices,
        mask=row_mask & (rank_offsets[None, :] < TOP_K),
    )


def router_topk_indices(
    logits: torch.Tensor,
    top_k: int,
    *,
    correction_bias: torch.Tensor | None = None,
    score_function: str = "identity",
) -> torch.Tensor:
    """Select experts for the fixed Qwen/DeepSeek router geometries.

    Ties are resolved deterministically in favor of the smaller expert index.
    The model contract does not depend on that choice: correctness comparisons
    must accept any unique expert whose score is equal to the kth threshold.
    """

    if logits.ndim != 2:
        raise ValueError(f"Router logits must be [tokens,experts], got {logits.shape}.")
    if score_function not in {"identity", "sqrtsoftplus"}:
        raise ValueError(f"Unsupported router score function: {score_function!r}.")

    num_tokens, num_experts = logits.shape
    apply_sqrt_softplus = score_function == "sqrtsoftplus"
    if not _is_supported_router_geometry(num_tokens, num_experts, top_k):
        raise RuntimeError(
            "Optimized router selection supports only 256-expert top-8/top-6 "
            "workloads with at most 32,768 tokens."
        )
    if logits.device.type != "cuda" or logits.dtype not in {
        torch.bfloat16,
        torch.float32,
    }:
        raise RuntimeError(
            "Optimized router selection requires CUDA/ROCm BF16 or FP32 logits."
        )

    has_correction_bias = correction_bias is not None
    if correction_bias is not None:
        if not apply_sqrt_softplus:
            raise ValueError(
                "Router correction bias is supported only with sqrt-softplus scoring."
            )
        if correction_bias.shape != (num_experts,):
            raise ValueError(
                f"Router correction bias must have shape {(num_experts,)}, got {correction_bias.shape}."
            )
        if correction_bias.device != logits.device:
            raise ValueError("Router correction bias must be on the logits device.")
        correction_bias = correction_bias.contiguous()
    elif apply_sqrt_softplus:
        # Triton still requires a pointer argument. It is not loaded when the
        # compile-time HAS_CORRECTION_BIAS flag is false.
        correction_bias = logits

    logits = logits.contiguous()
    indices = torch.empty(
        (num_tokens, top_k),
        dtype=torch.long,
        device=logits.device,
    )
    block_m, block_n, num_warps = _router_topk_launch(num_tokens)
    _router_topk_kernel[(triton.cdiv(num_tokens, block_m),)](
        logits,
        correction_bias if correction_bias is not None else logits,
        indices,
        num_tokens,
        TOP_K=top_k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        APPLY_SQRT_SOFTPLUS=apply_sqrt_softplus,
        HAS_CORRECTION_BIAS=has_correction_bias,
        num_warps=num_warps,
    )
    return indices


def _qwen_router_forward(
    self: Qwen3_5MoeTopKRouter,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat = hidden_states.reshape(-1, self.hidden_dim)
    router_logits = F.linear(flat, self.weight)
    router_indices = router_topk_indices(router_logits, self.top_k)
    selected_logits = router_logits.gather(1, router_indices)
    router_scores = torch.softmax(selected_logits, dtype=torch.float32, dim=-1).to(
        router_logits.dtype
    )
    return router_logits, router_scores, router_indices


def _deepseek_topk_router_forward(
    self: DeepseekV4TopKRouter,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat = hidden_states.reshape(-1, self.hidden_dim)
    logits = F.linear(flat.float(), self.weight.float())
    indices = router_topk_indices(
        logits,
        self.top_k,
        correction_bias=self.e_score_correction_bias,
        score_function="sqrtsoftplus",
    )
    scores = self.score_fn(logits.gather(1, indices))
    weights = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)
    return logits, weights * self.routed_scaling_factor, indices


def _deepseek_hash_router_forward(
    self: DeepseekV4HashRouter,
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat = hidden_states.reshape(-1, self.hidden_dim)
    logits = F.linear(flat.float(), self.weight.float())
    indices = self.tid2eid[input_ids.reshape(-1)].long()
    scores = self.score_fn(logits.gather(1, indices))
    weights = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)
    return logits, weights * self.routed_scaling_factor, indices


def configure_fast_moe_ranking(model: torch.nn.Module) -> dict[str, Any]:
    """Install the fixed-shape Qwen or DeepSeek routing-gate implementation."""

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    model_type = getattr(getattr(base, "config", None), "model_type", None)
    scoring_func = getattr(getattr(base, "config", None), "scoring_func", None)
    paths: dict[str, list[str]] = {"qwen": [], "deepseek_topk": [], "deepseek_hash": []}

    for name, module in base.named_modules():
        if isinstance(module, Qwen3_5MoeTopKRouter):
            if (
                module.hidden_dim != _QWEN_HIDDEN_SIZE
                or module.num_experts != _NUM_EXPERTS
                or module.top_k != _QWEN_TOP_K
            ):
                raise RuntimeError(
                    f"Qwen router {name!r} does not match 2048/256/top-8."
                )
            module.forward = MethodType(_qwen_router_forward, module)
            paths["qwen"].append(name)
        elif isinstance(module, DeepseekV4TopKRouter):
            if (
                module.hidden_dim != _DEEPSEEK_HIDDEN_SIZE
                or module.num_experts != _NUM_EXPERTS
                or module.top_k != _DEEPSEEK_TOP_K
                or scoring_func != "sqrtsoftplus"
            ):
                raise RuntimeError(
                    f"DeepSeek router {name!r} does not match 4096/256/top-6 sqrtsoftplus."
                )
            module.forward = MethodType(_deepseek_topk_router_forward, module)
            paths["deepseek_topk"].append(name)
        elif isinstance(module, DeepseekV4HashRouter):
            if (
                module.hidden_dim != _DEEPSEEK_HIDDEN_SIZE
                or module.num_experts != _NUM_EXPERTS
                or module.top_k != _DEEPSEEK_TOP_K
                or scoring_func != "sqrtsoftplus"
            ):
                raise RuntimeError(
                    f"DeepSeek hash router {name!r} does not match 4096/256/top-6 sqrtsoftplus."
                )
            module.forward = MethodType(_deepseek_hash_router_forward, module)
            paths["deepseek_hash"].append(name)

    if model_type in {"qwen3_5_moe", "qwen3_5_moe_text"} and len(paths["qwen"]) != 40:
        raise RuntimeError(f"expected 40 Qwen routers, found {len(paths['qwen'])}")
    if model_type == "deepseek_v4" and (
        len(paths["deepseek_topk"]) != 40 or len(paths["deepseek_hash"]) != 3
    ):
        raise RuntimeError(
            "expected 40 learned and 3 hash DeepSeek routers, found "
            f"{len(paths['deepseek_topk'])} and {len(paths['deepseek_hash'])}"
        )

    return {
        "qwen": len(paths["qwen"]),
        "deepseek_topk": len(paths["deepseek_topk"]),
        "deepseek_hash": len(paths["deepseek_hash"]),
        "paths": {key: sorted(value) for key, value in paths.items()},
    }
