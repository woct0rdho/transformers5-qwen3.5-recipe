from dataclasses import dataclass

import torch
import triton
import triton.language as tl

# Fixed sequence-2048 training geometries:
# - Qwen batches 1/4/16: [2048|8192|32768, 2048], top-8, with
#   16,384/65,536/262,144 routed rows.
# - DeepSeek batches 1/4/16: [2048|8192|32768, 4096], top-6, with
#   12,288/49,152/196,608 routed rows.
# The two smaller entries keep the synthetic expert-wrapper regression tests
# on the same optimized implementation instead of retaining a Torch fallback.
_SUPPORTED_ROUTING_GEOMETRIES = frozenset({(8, 2048), (6, 4096), (4, 2048), (1, 256)})
_MAX_TOKENS = 16 * 2048


def _is_supported_routing_geometry(
    num_tokens: int,
    num_top_k: int,
    hidden_dim: int,
) -> bool:
    return (
        0 < num_tokens <= _MAX_TOKENS
        and (num_top_k, hidden_dim) in _SUPPORTED_ROUTING_GEOMETRIES
    )


# Wide tiles won the gfx1151 sweep for both hidden sizes: gather uses 2,048
# columns per program, while combine uses the full hidden width. Backward keeps
# one full-width program per token or sorted route and varies only warp count.
def _routing_gather_forward_launch(
    num_tokens: int,
    top_k: int,
    hidden_dim: int,
) -> tuple[int, int]:
    """Return ``(BLOCK_H, num_warps)`` for fused route gathering."""

    del num_tokens, top_k
    if hidden_dim >= 2048:
        return 2048, 8
    return hidden_dim, 4


def _routing_combine_forward_launch(
    num_tokens: int,
    top_k: int,
    hidden_dim: int,
) -> tuple[int, int]:
    """Return ``(BLOCK_H, num_warps)`` for fused weighted route combining.

    Production workloads are Qwen ``[T, 2048]`` top-8 and DeepSeek
    ``[T, 4096]`` top-6 for ``T = 2048, 8192, 32768``.
    """

    del top_k
    if hidden_dim >= 4096:
        return 4096, 8
    return hidden_dim, 8 if num_tokens <= 2048 else 4


def _routing_launch_warps(
    num_tokens: int,
    num_top_k: int = 8,
    hidden_dim: int = 2048,
) -> tuple[int, int]:
    if (num_top_k, hidden_dim) == (6, 4096):
        return 16, 16 if num_tokens <= 4 * 2048 else 8
    if num_tokens <= 2048:
        return 4, 8
    if num_tokens <= 4 * 2048:
        return 8, 16
    return 16, 16


@dataclass(frozen=True)
class ExpertRoutingPlan:
    selected_hidden_states: torch.Tensor
    expert_indices: torch.Tensor
    routing_weights: torch.Tensor
    permutation: torch.Tensor
    inverse_permutation: torch.Tensor
    num_tokens: int
    num_top_k: int
    hidden_dim: int


@triton.jit
def _inverse_permutation_kernel(
    permutation_ptr,
    inverse_permutation_ptr,
    NUM_ROUTES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < NUM_ROUTES
    original_routes = tl.load(permutation_ptr + offsets, mask=mask)
    tl.store(inverse_permutation_ptr + original_routes, offsets, mask=mask)


@triton.jit
def _route_gather_forward_kernel(
    hidden_states_ptr,
    permutation_ptr,
    selected_hidden_states_ptr,
    HIDDEN: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    sorted_route = tl.program_id(0)
    hidden_block = tl.program_id(1)
    hidden_offsets = hidden_block * BLOCK_H + tl.arange(0, BLOCK_H)
    original_route = tl.load(permutation_ptr + sorted_route)
    token = original_route // TOP_K
    values = tl.load(
        hidden_states_ptr + token * HIDDEN + hidden_offsets,
        mask=hidden_offsets < HIDDEN,
    )
    tl.store(
        selected_hidden_states_ptr + sorted_route * HIDDEN + hidden_offsets,
        values,
        mask=hidden_offsets < HIDDEN,
    )


@triton.jit
def _route_gather_backward_kernel(
    grad_selected,
    inverse_permutation,
    grad_hidden,
    HIDDEN_SIZE: tl.constexpr,
    TOP_K: tl.constexpr,
):
    token = tl.program_id(0)
    hidden_offsets = tl.arange(0, HIDDEN_SIZE)
    route_ranks = tl.arange(0, 8)
    route_offsets = token * TOP_K + route_ranks
    remaining_positions = tl.load(
        inverse_permutation + route_offsets,
        mask=route_ranks < TOP_K,
        other=0x7FFFFFFF,
    )
    reduced = tl.zeros((HIDDEN_SIZE,), dtype=tl.float32)

    # PyTorch's sorted index backward rounds the destination to BF16 after
    # every duplicate. Preserve that order and rounding exactly.
    for _ in tl.static_range(0, TOP_K):
        position = tl.min(remaining_positions, axis=0)
        grad = tl.load(grad_selected + position * HIDDEN_SIZE + hidden_offsets).to(
            tl.float32
        )
        reduced = (reduced + grad).to(tl.bfloat16).to(tl.float32)
        remaining_positions = tl.where(
            remaining_positions == position,
            0x7FFFFFFF,
            remaining_positions,
        )

    tl.store(grad_hidden + token * HIDDEN_SIZE + hidden_offsets, reduced)


@triton.jit
def _route_combine_forward_kernel(
    output_ptr,
    routing_weights_ptr,
    inverse_permutation_ptr,
    result_ptr,
    HIDDEN: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token = tl.program_id(0)
    hidden_block = tl.program_id(1)
    hidden_offsets = hidden_block * BLOCK_H + tl.arange(0, BLOCK_H)
    hidden_mask = hidden_offsets < HIDDEN
    route_base = token * TOP_K
    accumulated = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for route_rank in tl.static_range(0, TOP_K):
        original_route = route_base + route_rank
        sorted_route = tl.load(inverse_permutation_ptr + original_route)
        values = tl.load(
            output_ptr + sorted_route * HIDDEN + hidden_offsets,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        weight = (
            tl.load(routing_weights_ptr + original_route).to(tl.bfloat16).to(tl.float32)
        )
        accumulated += values * weight
    tl.store(
        result_ptr + token * HIDDEN + hidden_offsets,
        accumulated,
        mask=hidden_mask,
    )


@triton.jit
def _route_combine_backward_kernel(
    expert_output,
    routing_weights,
    permutation,
    grad_final,
    grad_expert_output,
    grad_routing_weights,
    HIDDEN_SIZE: tl.constexpr,
    TOP_K: tl.constexpr,
):
    sorted_route = tl.program_id(0)
    original_route = tl.load(permutation + sorted_route)
    token = original_route // TOP_K
    hidden_offsets = tl.arange(0, HIDDEN_SIZE)

    grad = tl.load(grad_final + token * HIDDEN_SIZE + hidden_offsets).to(tl.float32)
    output = tl.load(expert_output + sorted_route * HIDDEN_SIZE + hidden_offsets).to(
        tl.float32
    )
    weight = tl.load(routing_weights + original_route).to(tl.bfloat16).to(tl.float32)

    output_grad = (grad * weight).to(tl.bfloat16)
    weight_grad_product = grad * output
    tl.store(
        grad_expert_output + sorted_route * HIDDEN_SIZE + hidden_offsets,
        output_grad,
    )
    weight_grad = tl.sum(weight_grad_product, axis=0)
    tl.store(grad_routing_weights + original_route, weight_grad)


class _RouteGather(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        permutation: torch.Tensor,
        inverse_permutation: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        num_top_k = permutation.numel() // num_tokens
        gather_warps, _ = _routing_launch_warps(num_tokens, num_top_k, hidden_dim)
        block_h, forward_warps = _routing_gather_forward_launch(
            num_tokens, num_top_k, hidden_dim
        )
        selected_hidden_states = torch.empty(
            (permutation.numel(), hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        _route_gather_forward_kernel[
            (permutation.numel(), triton.cdiv(hidden_dim, block_h))
        ](
            hidden_states,
            permutation,
            selected_hidden_states,
            HIDDEN=hidden_dim,
            TOP_K=num_top_k,
            BLOCK_H=block_h,
            num_warps=forward_warps,
        )
        ctx.save_for_backward(inverse_permutation)
        ctx.routing_geometry = (num_tokens, num_top_k, hidden_dim)
        ctx.gather_warps = gather_warps
        return selected_hidden_states

    @staticmethod
    def backward(ctx, grad_selected: torch.Tensor):
        if not ctx.needs_input_grad[0]:
            return None, None, None

        (inverse_permutation,) = ctx.saved_tensors
        num_tokens, num_top_k, hidden_dim = ctx.routing_geometry
        grad_selected = grad_selected.contiguous()
        grad_hidden = torch.empty(
            (num_tokens, hidden_dim),
            dtype=grad_selected.dtype,
            device=grad_selected.device,
        )
        _route_gather_backward_kernel[(num_tokens,)](
            grad_selected,
            inverse_permutation,
            grad_hidden,
            HIDDEN_SIZE=hidden_dim,
            TOP_K=num_top_k,
            num_warps=ctx.gather_warps,
        )
        return grad_hidden, None, None


class _RouteCombine(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        expert_output: torch.Tensor,
        routing_weights: torch.Tensor,
        permutation: torch.Tensor,
        inverse_permutation: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens, num_top_k = routing_weights.shape
        hidden_dim = expert_output.shape[1]
        _, combine_warps = _routing_launch_warps(num_tokens, num_top_k, hidden_dim)
        block_h, forward_warps = _routing_combine_forward_launch(
            num_tokens, num_top_k, hidden_dim
        )
        result = torch.empty(
            (num_tokens, hidden_dim),
            dtype=expert_output.dtype,
            device=expert_output.device,
        )
        _route_combine_forward_kernel[(num_tokens, triton.cdiv(hidden_dim, block_h))](
            expert_output,
            routing_weights,
            inverse_permutation,
            result,
            HIDDEN=hidden_dim,
            TOP_K=num_top_k,
            BLOCK_H=block_h,
            num_warps=forward_warps,
        )
        ctx.save_for_backward(expert_output, routing_weights, permutation)
        ctx.routing_geometry = (num_tokens, num_top_k, hidden_dim)
        ctx.combine_warps = combine_warps
        return result

    @staticmethod
    def backward(ctx, grad_final: torch.Tensor):
        expert_output, routing_weights, permutation = ctx.saved_tensors
        _, num_top_k, hidden_dim = ctx.routing_geometry
        num_routes = expert_output.shape[0]
        grad_final = grad_final.contiguous()
        grad_expert_output = torch.empty_like(expert_output)
        grad_routing_weights = torch.empty_like(routing_weights)
        _route_combine_backward_kernel[(num_routes,)](
            expert_output,
            routing_weights,
            permutation,
            grad_final,
            grad_expert_output,
            grad_routing_weights,
            HIDDEN_SIZE=hidden_dim,
            TOP_K=num_top_k,
            num_warps=ctx.combine_warps,
        )
        return (
            grad_expert_output if ctx.needs_input_grad[0] else None,
            grad_routing_weights if ctx.needs_input_grad[1] else None,
            None,
            None,
        )


def _validate_optimized_routing(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> None:
    num_tokens, hidden_dim = hidden_states.shape
    num_top_k = top_k_index.shape[1]
    if not _is_supported_routing_geometry(num_tokens, num_top_k, hidden_dim):
        raise RuntimeError(
            "Unsupported optimized expert-routing geometry. Production requires "
            "Qwen top-8/hidden-2048 or DeepSeek top-6/hidden-4096 with at most "
            "32,768 tokens."
        )
    if hidden_states.device.type != "cuda":
        raise RuntimeError("Optimized expert routing requires CUDA/ROCm tensors.")
    if hidden_states.dtype != torch.bfloat16 or not hidden_states.is_contiguous():
        raise RuntimeError(
            "Optimized expert routing requires contiguous BF16 hidden states."
        )
    if top_k_index.device != hidden_states.device:
        raise RuntimeError("Expert indices must be on the hidden-state device.")
    if top_k_weights.device != hidden_states.device:
        raise RuntimeError("Routing weights must be on the hidden-state device.")
    if top_k_index.dtype not in (torch.int32, torch.int64):
        raise RuntimeError("Expert indices must use int32 or int64.")
    if top_k_weights.dtype not in (torch.bfloat16, torch.float32):
        raise RuntimeError("Routing weights must use BF16 or FP32.")


def prepare_expert_routing(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> ExpertRoutingPlan:
    if hidden_states.ndim != 2:
        raise ValueError(
            f"Expert routing expects hidden states [tokens, hidden], got {hidden_states.shape}."
        )
    if top_k_index.ndim != 2 or top_k_weights.shape != top_k_index.shape:
        raise ValueError(
            "Expert routing expects matching top-k index and weight matrices."
        )
    if top_k_index.shape[0] != hidden_states.shape[0]:
        raise ValueError(
            "Expert routing token count does not match the hidden-state token count."
        )

    num_tokens = hidden_states.shape[0]
    num_top_k = top_k_index.shape[1]
    hidden_dim = hidden_states.shape[1]
    expert_indices, permutation = torch.sort(top_k_index.reshape(-1))
    permutation = permutation.contiguous()
    inverse_permutation = torch.empty_like(permutation)
    inverse_block = 256
    _inverse_permutation_kernel[(triton.cdiv(permutation.numel(), inverse_block),)](
        permutation,
        inverse_permutation,
        NUM_ROUTES=permutation.numel(),
        BLOCK=inverse_block,
        num_warps=4,
    )
    routing_weights = top_k_weights.contiguous()
    _validate_optimized_routing(hidden_states, top_k_index, routing_weights)
    selected_hidden_states = _RouteGather.apply(
        hidden_states,
        permutation,
        inverse_permutation,
    )

    return ExpertRoutingPlan(
        selected_hidden_states=selected_hidden_states,
        expert_indices=expert_indices,
        routing_weights=routing_weights,
        permutation=permutation,
        inverse_permutation=inverse_permutation,
        num_tokens=num_tokens,
        num_top_k=num_top_k,
        hidden_dim=hidden_dim,
    )


def finalize_expert_routing(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    routing_plan: ExpertRoutingPlan,
    output_mask: torch.Tensor | None,
) -> torch.Tensor:
    if output_mask is not None:
        raise RuntimeError("Optimized expert routing does not support output masks.")
    expected_shape = (
        routing_plan.num_tokens * routing_plan.num_top_k,
        routing_plan.hidden_dim,
    )
    if (
        output.device.type != "cuda"
        or output.dtype != torch.bfloat16
        or not output.is_contiguous()
        or tuple(output.shape) != expected_shape
    ):
        raise RuntimeError(
            "Optimized expert combine requires contiguous CUDA BF16 output with "
            f"shape {expected_shape}, got {tuple(output.shape)}."
        )
    result = _RouteCombine.apply(
        output,
        routing_plan.routing_weights,
        routing_plan.permutation,
        routing_plan.inverse_permutation,
    )
    return result.to(hidden_states.dtype)
