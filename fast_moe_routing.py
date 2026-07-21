from dataclasses import dataclass

import torch
import triton
import triton.language as tl

_TOP_K = 8
_HIDDEN_SIZE = 2048
_MAX_TOKENS = 16 * 2048


def _is_supported_routing_geometry(
    num_tokens: int,
    num_top_k: int,
    hidden_dim: int,
) -> bool:
    return (
        0 < num_tokens <= _MAX_TOKENS
        and num_top_k == _TOP_K
        and hidden_dim == _HIDDEN_SIZE
    )


# Full-width kernels won the gfx1151 sweep; only the warp count scales with T.
def _routing_launch_warps(num_tokens: int) -> tuple[int, int]:
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
    use_triton: bool


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
    route_offsets = token * TOP_K + tl.arange(0, TOP_K)
    remaining_positions = tl.load(inverse_permutation + route_offsets)
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
    weight_grad_product = (grad * output).to(tl.bfloat16).to(tl.float32)
    tl.store(
        grad_expert_output + sorted_route * HIDDEN_SIZE + hidden_offsets,
        output_grad,
    )
    weight_grad = tl.sum(weight_grad_product, axis=0).to(tl.bfloat16).to(tl.float32)
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
        gather_warps, _ = _routing_launch_warps(num_tokens)
        ctx.save_for_backward(inverse_permutation)
        ctx.routing_geometry = (num_tokens, num_top_k, hidden_dim)
        ctx.gather_warps = gather_warps
        return hidden_states[permutation // num_top_k]

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
        _, combine_warps = _routing_launch_warps(num_tokens)
        ctx.save_for_backward(expert_output, routing_weights, permutation)
        ctx.routing_geometry = (num_tokens, num_top_k, hidden_dim)
        ctx.combine_warps = combine_warps
        sorted_weights = routing_weights.reshape(-1)[permutation].to(
            expert_output.dtype
        )
        weighted_output = expert_output * sorted_weights.unsqueeze(-1)
        token_order = weighted_output[inverse_permutation]
        return token_order.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)

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


def _can_use_triton_routing(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> bool:
    num_tokens, hidden_dim = hidden_states.shape
    num_top_k = top_k_index.shape[1]
    return (
        _is_supported_routing_geometry(num_tokens, num_top_k, hidden_dim)
        and hidden_states.device.type == "cuda"
        and hidden_states.dtype == torch.bfloat16
        and hidden_states.is_contiguous()
        and top_k_index.device == hidden_states.device
        and top_k_weights.device == hidden_states.device
        and top_k_weights.shape == top_k_index.shape
        and top_k_index.dtype in (torch.int32, torch.int64)
        and top_k_weights.dtype in (torch.bfloat16, torch.float32)
    )


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
    inverse_permutation[permutation] = torch.arange(
        permutation.numel(),
        device=permutation.device,
        dtype=permutation.dtype,
    )
    routing_weights = top_k_weights.contiguous()
    use_triton = _can_use_triton_routing(hidden_states, top_k_index, routing_weights)
    if use_triton:
        selected_hidden_states = _RouteGather.apply(
            hidden_states,
            permutation,
            inverse_permutation,
        )
    else:
        selected_hidden_states = hidden_states[permutation // num_top_k]

    return ExpertRoutingPlan(
        selected_hidden_states=selected_hidden_states,
        expert_indices=expert_indices,
        routing_weights=routing_weights,
        permutation=permutation,
        inverse_permutation=inverse_permutation,
        num_tokens=num_tokens,
        num_top_k=num_top_k,
        hidden_dim=hidden_dim,
        use_triton=use_triton,
    )


def finalize_expert_routing(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    routing_plan: ExpertRoutingPlan,
    output_mask: torch.Tensor | None,
) -> torch.Tensor:
    use_triton = (
        routing_plan.use_triton
        and output_mask is None
        and output.device.type == "cuda"
        and output.dtype == torch.bfloat16
        and output.is_contiguous()
        and tuple(output.shape)
        == (
            routing_plan.num_tokens * routing_plan.num_top_k,
            routing_plan.hidden_dim,
        )
    )
    if use_triton:
        result = _RouteCombine.apply(
            output,
            routing_plan.routing_weights,
            routing_plan.permutation,
            routing_plan.inverse_permutation,
        )
    else:
        routing_weights = routing_plan.routing_weights.reshape(-1)[
            routing_plan.permutation
        ].to(output.dtype)
        result = output * routing_weights.unsqueeze(-1)
        if output_mask is not None:
            result.masked_fill_(output_mask, 0.0)
        result = result[routing_plan.inverse_permutation]
        result = result.view(
            routing_plan.num_tokens,
            routing_plan.num_top_k,
            routing_plan.hidden_dim,
        ).sum(dim=1)
    return result.to(hidden_states.dtype)


__all__ = [
    "ExpertRoutingPlan",
    "finalize_expert_routing",
    "prepare_expert_routing",
]
