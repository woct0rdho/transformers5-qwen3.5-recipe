"""Triton routing-gate selection for the fixed Qwen and DeepSeek MoE shapes.

The optimized training workloads are sequence length 2,048 at physical batches
1, 4, and 16:
- Qwen3.5/3.6: `[2048|8192|32768, 2048]` hidden states, 256 experts,
  top-8 selection, and 16,384/65,536/262,144 routed rows.
- Qwen4-Exp: `[2048|8192|32768, 2560]` hidden states, 512 experts, top-10
  selection, and 98,304/393,216/1,572,864 rows over its 48 layers.
- DeepSeek V4 learned routers: `[2048|8192|32768, 4096]` hidden states,
  256 experts, top-6 sqrt-softplus-plus-correction-bias selection, and
  12,288/49,152/196,608 routed rows.
- DeepSeek V4 hash routers have the same hidden/expert geometry and top-6
  weights, but their expert IDs come from the fixed token lookup table. Because
  the lookup fixes the six experts before any projection, their logits are
  projected directly from the selected gate rows instead of a 256-expert
  projection. No full-width score tensor is materialized.

The learned-router projection is a normal BF16 `F.linear` (FP32 internal
accumulation) whose BF16 result is upcast to FP32 so the scoring path stays in
FP32. The Triton kernel replaces full-width softmax/sqrt-softplus plus
`torch.topk` with one streaming selection over the expert axis. Normalization is
then evaluated only for the selected experts, preserving ordinary autograd for
router-score gradients.

The selection here is deterministic by construction - a tie resolves to the
lower expert index - because a checkpointed layer's replay has to save as many
tensors as its forward did. It is not required to reproduce `torch.topk`'s choice
among tied experts, and no check should compare expert identities: correctness
compares the selected weights, with a tolerance, and accepts any expert at the
kth score threshold.
"""

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
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextTopKRouter

from module_patching import (
    ModulePatchSpec,
    patch_module_forwards,
    require_complete_inventory,
)

_NUM_EXPERTS = 256
_SEQUENCE_LENGTH = 2048
_MAX_TOKENS = 16 * _SEQUENCE_LENGTH
_QWEN_HIDDEN_SIZE = 2048
_QWEN_TOP_K = 8
_QWEN4_HIDDEN_SIZE = 2560
_QWEN4_TOP_K = 10
_DEEPSEEK_HIDDEN_SIZE = 4096
_DEEPSEEK_TOP_K = 6
_SUPPORTED_ROUTER_GEOMETRIES = frozenset(
    {
        (_NUM_EXPERTS, _QWEN_TOP_K),
        (_NUM_EXPERTS, _DEEPSEEK_TOP_K),
        (512, _QWEN4_TOP_K),
    }
)


def _is_supported_router_geometry(
    num_tokens: int,
    num_experts: int,
    top_k: int,
) -> bool:
    return (
        0 < num_tokens <= _MAX_TOKENS
        and (num_experts, top_k) in _SUPPORTED_ROUTER_GEOMETRIES
    )


def _router_topk_launch(num_tokens: int, num_experts: int) -> tuple[int, int, int]:
    """Return `(BLOCK_M, BLOCK_N, num_warps)` for the fixed token and expert buckets.

    Every geometry streams a 64-expert tile. The 256-expert shapes use four rows
    and four warps up to 2,048 tokens and eight rows and eight warps above it,
    because larger row tiles raised register pressure. The 512-expert top-10
    shape keeps four rows and two warps at every row count: it streams twice as
    many tiles per row, and more warps regress it.
    """

    if num_experts == 512:
        return 4, 64, 2
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
    NUM_EXPERTS: tl.constexpr,
    APPLY_SQRT_SOFTPLUS: tl.constexpr,
    HAS_CORRECTION_BIAS: tl.constexpr,
):
    pointers = logits + row_offsets[:, None] * NUM_EXPERTS + expert_offsets[None, :]
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
    NUM_EXPERTS: tl.constexpr,
    TOP_K_PAD: tl.constexpr,
    BLOCK_N: tl.constexpr,
    APPLY_SQRT_SOFTPLUS: tl.constexpr,
    HAS_CORRECTION_BIAS: tl.constexpr,
):
    """Return `TOP_K_PAD` sorted packed `(score, inverse-index)` keys per row."""

    score_dtype: tl.constexpr = (
        tl.float32 if APPLY_SQRT_SOFTPLUS else logits.dtype.element_ty
    )
    score_bits: tl.constexpr = score_dtype.primitive_bitwidth
    key_dtype: tl.constexpr = tl.dtype(f"uint{score_bits * 2}")
    iterations: tl.constexpr = NUM_EXPERTS // BLOCK_N

    expert_offsets = (iterations - 1) * BLOCK_N + tl.arange(0, BLOCK_N)
    scores = _load_router_scores(
        logits,
        correction_bias,
        row_offsets,
        expert_offsets,
        row_mask,
        NUM_EXPERTS,
        APPLY_SQRT_SOFTPLUS,
        HAS_CORRECTION_BIAS,
    ).to(score_dtype)
    score_keys = _float_key(scores)
    index_keys = (NUM_EXPERTS - expert_offsets)[None, :]
    packed = (score_keys.to(key_dtype) << 16) | index_keys
    selected = tl.topk(packed, TOP_K_PAD, dim=1)

    for _ in tl.static_range(0, iterations - 1):
        selected = tl.bitonic_merge(selected)
        expert_offsets -= BLOCK_N
        scores = _load_router_scores(
            logits,
            correction_bias,
            row_offsets,
            expert_offsets,
            row_mask,
            NUM_EXPERTS,
            APPLY_SQRT_SOFTPLUS,
            HAS_CORRECTION_BIAS,
        ).to(score_dtype)
        score_keys = _float_key(scores)
        index_keys = (NUM_EXPERTS - expert_offsets)[None, :]
        packed = (score_keys.to(key_dtype) << 16) | index_keys
        selected = tl.maximum(
            selected,
            tl.topk(packed, TOP_K_PAD, dim=1),
        )

    return tl.sort(selected, dim=1, descending=True)


@triton.jit
def _router_topk_kernel(
    logits,
    correction_bias,
    indices,
    num_rows,
    TOP_K: tl.constexpr,
    TOP_K_PAD: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
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
        NUM_EXPERTS,
        TOP_K_PAD,
        BLOCK_N,
        APPLY_SQRT_SOFTPLUS,
        HAS_CORRECTION_BIAS,
    )
    rank_offsets = tl.arange(0, TOP_K_PAD)
    inverse_indices = (selected & 0xFFFF).to(tl.int64)
    selected_indices = NUM_EXPERTS - inverse_indices
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
            "Optimized router selection supports only the 256-expert top-8/top-6 "
            "and 512-expert top-10 workloads with at most 32,768 tokens."
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
        dtype=torch.int64,
        device=logits.device,
    )
    block_m, block_n, num_warps = _router_topk_launch(num_tokens, num_experts)
    _router_topk_kernel[(triton.cdiv(num_tokens, block_m),)](
        logits,
        correction_bias if correction_bias is not None else logits,
        indices,
        num_tokens,
        TOP_K=top_k,
        TOP_K_PAD=triton.next_power_of_2(top_k),
        NUM_EXPERTS=num_experts,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        APPLY_SQRT_SOFTPLUS=apply_sqrt_softplus,
        HAS_CORRECTION_BIAS=has_correction_bias,
        num_warps=num_warps,
    )
    return indices


# Fixed DeepSeek V4 hash-router geometry. The token lookup fixes the six
# experts before any projection, so one program per token streams the token row
# once and reads the six selected rows of the L2-resident gate table. This
# replaces a 256-expert projection whose other 250 columns were discarded.
_HASH_BLOCK_K = 1024
_HASH_BLOCK_D = 2048
_HASH_FORWARD_WARPS = 4
_HASH_BACKWARD_WARPS = 2
_HASH_TOP_K_PAD = tl.constexpr(8)


@triton.jit
def _hash_router_logits_forward_kernel(
    hidden_ptr,
    weight_ptr,
    indices_ptr,
    logits_ptr,
    HIDDEN: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    token = tl.program_id(0)
    rank_offsets = tl.arange(0, _HASH_TOP_K_PAD)
    rank_mask = rank_offsets < TOP_K
    expert_ids = tl.load(
        indices_ptr + token * TOP_K + rank_offsets,
        mask=rank_mask,
        other=0,
    ).to(tl.int64)
    projected = tl.zeros((_HASH_TOP_K_PAD,), tl.float32)
    hidden_base = hidden_ptr + token * HIDDEN
    for k_start in tl.range(0, HIDDEN, BLOCK_K, loop_unroll_factor=1):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < HIDDEN
        x = tl.load(hidden_base + k_offsets, mask=k_mask, other=0.0).to(tl.float32)
        weight = tl.load(
            weight_ptr + expert_ids[:, None] * HIDDEN + k_offsets[None, :],
            mask=rank_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        projected += tl.sum(weight * x[None, :], axis=1)
    tl.store(logits_ptr + token * TOP_K + rank_offsets, projected, mask=rank_mask)


@triton.jit
def _hash_router_input_grad_kernel(
    grad_logits_ptr,
    weight_ptr,
    indices_ptr,
    grad_hidden_ptr,
    HIDDEN: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token = tl.program_id(0)
    rank_offsets = tl.arange(0, _HASH_TOP_K_PAD)
    rank_mask = rank_offsets < TOP_K
    expert_ids = tl.load(
        indices_ptr + token * TOP_K + rank_offsets,
        mask=rank_mask,
        other=0,
    ).to(tl.int64)
    grads = tl.load(
        grad_logits_ptr + token * TOP_K + rank_offsets,
        mask=rank_mask,
        other=0.0,
    ).to(tl.float32)
    grad_base = grad_hidden_ptr + token * HIDDEN
    for d_start in tl.range(0, HIDDEN, BLOCK_D, loop_unroll_factor=1):
        d_offsets = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_offsets < HIDDEN
        weight = tl.load(
            weight_ptr + expert_ids[:, None] * HIDDEN + d_offsets[None, :],
            mask=rank_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        accumulated = tl.sum(weight * grads[:, None], axis=0)
        tl.store(grad_base + d_offsets, accumulated, mask=d_mask)


class _DeepseekHashRouterLogits(torch.autograd.Function):
    """Six hash-selected expert logits for the fixed DeepSeek V4 geometry.

    The token lookup fixes the selected experts, so only their dot products are
    evaluated. Backward accumulates `grad_logits[k] * weight[expert_k]` in
    FP32 and rounds once to the BF16 activation boundary. The frozen gate table
    needs no gradient.
    """

    @staticmethod
    def forward(
        ctx: Any,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        logits = torch.empty(
            (num_tokens, _DEEPSEEK_TOP_K),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        _hash_router_logits_forward_kernel[(num_tokens,)](
            hidden_states,
            weight,
            indices,
            logits,
            HIDDEN=hidden_dim,
            TOP_K=_DEEPSEEK_TOP_K,
            BLOCK_K=_HASH_BLOCK_K,
            num_warps=_HASH_FORWARD_WARPS,
        )
        ctx.save_for_backward(weight, indices)
        ctx.hidden_meta = (hidden_states.shape, hidden_states.dtype)
        return logits

    @staticmethod
    def backward(  # ty: ignore[invalid-method-override]
        ctx: Any,
        grad_logits: torch.Tensor,
    ) -> tuple[torch.Tensor | None, None, None]:
        weight, indices = ctx.saved_tensors
        grad_hidden = None
        if ctx.needs_input_grad[0]:
            hidden_shape, hidden_dtype = ctx.hidden_meta
            grad_logits = grad_logits.contiguous()
            grad_hidden = torch.empty(
                hidden_shape,
                dtype=hidden_dtype,
                device=grad_logits.device,
            )
            _hash_router_input_grad_kernel[(hidden_shape[0],)](
                grad_logits,
                weight,
                indices,
                grad_hidden,
                HIDDEN=hidden_shape[1],
                TOP_K=_DEEPSEEK_TOP_K,
                BLOCK_D=_HASH_BLOCK_D,
                num_warps=_HASH_BACKWARD_WARPS,
            )
        return grad_hidden, None, None


def hash_router_selected_logits(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Return the six hash-selected expert logits as FP32 `[tokens,6]`."""

    if hidden_states.ndim != 2 or weight.ndim != 2 or indices.ndim != 2:
        raise ValueError(
            "Hash-router logits require two-dimensional hidden states, weight, and indices."
        )
    num_tokens, hidden_dim = hidden_states.shape
    if hidden_dim != _DEEPSEEK_HIDDEN_SIZE:
        raise ValueError(
            f"Hash-router logits require hidden size {_DEEPSEEK_HIDDEN_SIZE}, got {hidden_dim}."
        )
    if tuple(weight.shape) != (_NUM_EXPERTS, _DEEPSEEK_HIDDEN_SIZE):
        raise ValueError(
            f"Hash-router gate weight must have shape {(_NUM_EXPERTS, _DEEPSEEK_HIDDEN_SIZE)}, "
            f"got {tuple(weight.shape)}."
        )
    if tuple(indices.shape) != (num_tokens, _DEEPSEEK_TOP_K):
        raise ValueError(
            f"Hash-router indices must have shape {(num_tokens, _DEEPSEEK_TOP_K)}, "
            f"got {tuple(indices.shape)}."
        )
    if hidden_states.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError(
            "Hash-router logits require BF16 hidden states and gate weights."
        )
    if indices.dtype not in (torch.int32, torch.int64):
        raise TypeError(
            f"Hash-router indices must use int32 or int64, got {indices.dtype}."
        )
    if weight.requires_grad:
        raise RuntimeError("Hash-router gate weights must remain frozen.")
    if hidden_states.device.type != "cuda":
        raise RuntimeError("Hash-router logits require CUDA/ROCm tensors.")
    if any(tensor.device != hidden_states.device for tensor in (weight, indices)):
        raise ValueError("Hash-router logits require all tensors on one device.")
    if (
        not hidden_states.is_contiguous()
        or not weight.is_contiguous()
        or not indices.is_contiguous()
    ):
        raise ValueError("Hash-router logits require contiguous inputs.")
    if num_tokens == 0:
        return torch.empty(
            (0, _DEEPSEEK_TOP_K),
            dtype=torch.float32,
            device=hidden_states.device,
        )
    return _DeepseekHashRouterLogits.apply(hidden_states, weight, indices)


def _qwen_router_forward(
    self: Qwen3_5MoeTopKRouter,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat = hidden_states.reshape(-1, self.hidden_dim)
    # BF16 GEMM (FP32 accumulation) rounded to BF16 at the projection, then
    # upcast so the top-k selection and softmax below stay in FP32.
    router_logits = F.linear(flat, self.weight).to(torch.float32)
    router_indices = router_topk_indices(router_logits, self.top_k)
    selected_logits = router_logits.gather(1, router_indices)
    router_scores = torch.softmax(selected_logits, dtype=torch.float32, dim=-1)
    return router_logits, router_scores, router_indices


def _qwen4_router_forward(
    self: Qwen4ExpTextTopKRouter,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat = hidden_states.reshape(-1, self.hidden_dim)
    # The projection rounds to BF16 at the module boundary and the scoring path then runs in FP32, as
    # the model's own forward does.
    projection = F.linear(flat, self.weight)
    logits = projection.to(torch.float32)
    indices = router_topk_indices(logits, self.top_k)
    # A softmax over the selected scores is the model's full-width softmax renormalized over its own
    # top-k: `softmax(x)_i / sum_{j in S} softmax(x)_j` over a selected set depends only on that set.
    weights = torch.softmax(logits.gather(1, indices), dim=-1)
    return logits, weights.to(projection.dtype), indices


def _deepseek_topk_router_forward(
    self: DeepseekV4TopKRouter,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat = hidden_states.reshape(-1, self.hidden_dim)
    logits = F.linear(flat, self.weight).to(torch.float32)
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
    indices = self.tid2eid[input_ids.reshape(-1)].long()
    # The token lookup fixes the experts, so only the six selected logits are
    # projected. Returning the selected logits (not a 256-wide tensor) keeps the
    # scoring path in FP32 without materializing the other 250 expert columns.
    selected_logits = hash_router_selected_logits(flat, self.weight, indices)
    scores = self.score_fn(selected_logits)
    weights = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)
    return selected_logits, weights * self.routed_scaling_factor, indices


def _bind_router_expert_prior(
    base: torch.nn.Module,
    router_name: str,
    expert_prior: str,
) -> bool:
    block_name, separator, suffix = router_name.rpartition(".gate")
    if not separator or suffix:
        return False
    block = base.get_submodule(block_name)
    experts = getattr(block, "experts", None)
    if experts is None:
        return False
    experts.__dict__["_aiter_expert_prior"] = expert_prior
    return True


_EXPECTED_QWEN_ROUTERS = 40
_EXPECTED_QWEN4_ROUTERS = 48
_EXPECTED_DEEPSEEK_TOPK_ROUTERS = 40
_EXPECTED_DEEPSEEK_HASH_ROUTERS = 3
_QWEN_ROUTER_MARKER = "_patched_qwen_router"
_QWEN4_ROUTER_MARKER = "_patched_qwen4_router"
_DEEPSEEK_TOPK_ROUTER_MARKER = "_patched_deepseek_topk_router"
_DEEPSEEK_HASH_ROUTER_MARKER = "_patched_deepseek_hash_router"


def _validate_qwen_router(name: str, module: Qwen3_5MoeTopKRouter) -> None:
    if (
        module.hidden_dim != _QWEN_HIDDEN_SIZE
        or module.num_experts != _NUM_EXPERTS
        or module.top_k != _QWEN_TOP_K
    ):
        raise RuntimeError(f"Qwen router {name!r} does not match 2048/256/top-8.")


def _validate_qwen4_router(name: str, module: Qwen4ExpTextTopKRouter) -> None:
    if (
        module.hidden_dim != _QWEN4_HIDDEN_SIZE
        or module.num_experts != 512
        or module.top_k != _QWEN4_TOP_K
    ):
        raise RuntimeError(f"Qwen4 router {name!r} does not match 2560/512/top-10.")
    if module.weight.device.type != "cuda":
        raise RuntimeError(f"Qwen4 router {name!r} requires a CUDA/ROCm weight")
    if not module.norm_topk_prob:
        # Without the renormalization the selected weight is the full-width softmax value, which the
        # selected scores alone cannot reproduce.
        raise RuntimeError(
            f"Qwen4 router {name!r} does not normalize the top-k probabilities"
        )


def _validate_deepseek_router(
    name: str,
    module: DeepseekV4TopKRouter | DeepseekV4HashRouter,
    scoring_func: Any,
    *,
    label: str,
) -> None:
    if (
        module.hidden_dim != _DEEPSEEK_HIDDEN_SIZE
        or module.num_experts != _NUM_EXPERTS
        or module.top_k != _DEEPSEEK_TOP_K
        or scoring_func != "sqrtsoftplus"
    ):
        raise RuntimeError(
            f"{label} {name!r} does not match 4096/256/top-6 sqrtsoftplus."
        )


def _router_specs(
    base: torch.nn.Module, scoring_func: Any
) -> tuple[ModulePatchSpec[Any], ...]:
    """Build the router specs, binding the owning experts prior on every pass."""

    def bind(prior: str):
        def prepare(name: str, module: torch.nn.Module) -> None:
            del module
            _bind_router_expert_prior(base, name, prior)

        return prepare

    def validate_deepseek_topk(name: str, module: DeepseekV4TopKRouter) -> None:
        _validate_deepseek_router(name, module, scoring_func, label="DeepSeek router")

    def validate_deepseek_hash(name: str, module: DeepseekV4HashRouter) -> None:
        _validate_deepseek_router(
            name, module, scoring_func, label="DeepSeek hash router"
        )

    return (
        ModulePatchSpec(
            module_type=Qwen3_5MoeTopKRouter,
            forward=_qwen_router_forward,
            handled_key="qwen",
            validate=_validate_qwen_router,
            prepare=bind("qwen-learned"),
            marker=_QWEN_ROUTER_MARKER,
            freeze_weight=False,
        ),
        ModulePatchSpec(
            module_type=Qwen4ExpTextTopKRouter,
            forward=_qwen4_router_forward,
            handled_key="qwen4",
            validate=_validate_qwen4_router,
            marker=_QWEN4_ROUTER_MARKER,
            freeze_weight=False,
        ),
        ModulePatchSpec(
            module_type=DeepseekV4TopKRouter,
            forward=_deepseek_topk_router_forward,
            handled_key="deepseek_topk",
            validate=validate_deepseek_topk,
            prepare=bind("deepseek-learned"),
            marker=_DEEPSEEK_TOPK_ROUTER_MARKER,
            freeze_weight=False,
        ),
        ModulePatchSpec(
            module_type=DeepseekV4HashRouter,
            forward=_deepseek_hash_router_forward,
            handled_key="deepseek_hash",
            validate=validate_deepseek_hash,
            prepare=bind("deepseek-hash"),
            marker=_DEEPSEEK_HASH_ROUTER_MARKER,
            freeze_weight=False,
        ),
    )


def configure_fast_moe_ranking(model: torch.nn.Module) -> dict[str, Any]:
    """Install the fixed-shape Qwen or DeepSeek routing-gate implementation."""

    get_base_model = getattr(model, "get_base_model", None)
    base = get_base_model() if callable(get_base_model) else model
    config = getattr(base, "config", None)
    model_type = getattr(config, "model_type", None)
    scoring_func = getattr(config, "scoring_func", None)

    report = patch_module_forwards(base, _router_specs(base, scoring_func))
    report["paths"] = {
        key: sorted(names) for key, names in report["handled_by_key"].items()
    }
    require_complete_fast_moe_ranking(report, model_type)
    return report


def require_complete_fast_moe_ranking(
    report: dict[str, Any], model_type: str | None = None
) -> None:
    """Fail closed unless every router of the model family was handled."""

    if model_type in {"qwen3_5_moe", "qwen3_5_moe_text"}:
        require_complete_inventory(
            report, {"qwen": _EXPECTED_QWEN_ROUTERS}, subject="Qwen router"
        )
    elif model_type == "qwen4_exp_text":
        require_complete_inventory(
            report, {"qwen4": _EXPECTED_QWEN4_ROUTERS}, subject="Qwen4-Exp router"
        )
    elif model_type == "deepseek_v4":
        require_complete_inventory(
            report,
            {
                "deepseek_topk": _EXPECTED_DEEPSEEK_TOPK_ROUTERS,
                "deepseek_hash": _EXPECTED_DEEPSEEK_HASH_ROUTERS,
            },
            subject="DeepSeek V4 router",
        )
