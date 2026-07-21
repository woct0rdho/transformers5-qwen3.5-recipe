import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from fast_moe_routing import (
    ExpertRoutingPlan,
    _is_supported_routing_geometry,
    _routing_launch_warps,
    finalize_expert_routing,
    prepare_expert_routing,
)

_TOKENS = 2048
_TOP_K = 8
_HIDDEN = 2048
_ROUTES = _TOKENS * _TOP_K


class _RecordOps(TorchDispatchMode):
    def __init__(self, dispatched_ops: list[str]) -> None:
        super().__init__()
        self.dispatched_ops = dispatched_ops

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.dispatched_ops.append(str(func))
        return func(*args, **(kwargs or {}))


def _routing_indices(
    num_tokens: int = _TOKENS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(19260817)
    top_k_index = torch.randint(
        0,
        256,
        (num_tokens, _TOP_K),
        generator=generator,
        device="cuda",
        dtype=torch.int64,
    )
    expert_indices, permutation = torch.sort(top_k_index.reshape(-1))
    inverse_permutation = torch.empty_like(permutation)
    inverse_permutation[permutation] = torch.arange(num_tokens * _TOP_K, device="cuda")
    return top_k_index, expert_indices, inverse_permutation


@pytest.mark.parametrize(
    ("batch_size", "expected_warps"),
    [(1, (4, 8)), (4, (8, 16)), (16, (16, 16))],
)
def test_supported_batch_geometry_and_launch_heuristics(
    batch_size: int,
    expected_warps: tuple[int, int],
) -> None:
    num_tokens = batch_size * 2048
    assert _is_supported_routing_geometry(num_tokens, _TOP_K, _HIDDEN)
    assert _routing_launch_warps(num_tokens) == expected_warps


@pytest.mark.parametrize(
    ("num_tokens", "num_top_k", "hidden_dim"),
    [(0, 8, 2048), (32769, 8, 2048), (2048, 4, 2048), (2048, 8, 1024)],
)
def test_unsupported_routing_geometry(
    num_tokens: int,
    num_top_k: int,
    hidden_dim: int,
) -> None:
    assert not _is_supported_routing_geometry(num_tokens, num_top_k, hidden_dim)


def test_route_gather_backward_is_exact_and_has_no_index_put() -> None:
    top_k_index, expected_experts, expected_inverse = _routing_indices()
    generator = torch.Generator(device="cuda").manual_seed(2468)
    hidden = torch.randn(
        (_TOKENS, _HIDDEN),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    reference_hidden = hidden.detach().clone().requires_grad_(True)
    routing_weights = torch.rand(
        (_TOKENS, _TOP_K),
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )

    plan = prepare_expert_routing(hidden, top_k_index, routing_weights)
    _, permutation = torch.sort(top_k_index.reshape(-1))
    reference_selected = reference_hidden[permutation // _TOP_K]
    torch.testing.assert_close(
        plan.selected_hidden_states, reference_selected, rtol=0, atol=0
    )
    torch.testing.assert_close(plan.expert_indices, expected_experts, rtol=0, atol=0)
    torch.testing.assert_close(
        plan.inverse_permutation, expected_inverse, rtol=0, atol=0
    )
    assert plan.use_triton

    grad_selected = torch.randn(
        (_ROUTES, _HIDDEN),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    dispatched_ops: list[str] = []
    with _RecordOps(dispatched_ops):
        plan.selected_hidden_states.backward(grad_selected)
    reference_selected.backward(grad_selected)

    torch.testing.assert_close(hidden.grad, reference_hidden.grad, rtol=0, atol=0)
    assert not any("_index_put_impl_" in operation for operation in dispatched_ops)


@pytest.mark.parametrize("routing_dtype", [torch.bfloat16, torch.float32])
def test_route_combine_forward_and_backward_are_exact(
    routing_dtype: torch.dtype,
) -> None:
    top_k_index, expert_indices, inverse_permutation = _routing_indices()
    _, permutation = torch.sort(top_k_index.reshape(-1))
    generator = torch.Generator(device="cuda").manual_seed(97531)
    output = torch.randn(
        (_ROUTES, _HIDDEN),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    reference_output = output.detach().clone().requires_grad_(True)
    routing_weights = torch.rand(
        (_TOKENS, _TOP_K),
        generator=generator,
        device="cuda",
        dtype=routing_dtype,
        requires_grad=True,
    )
    reference_weights = routing_weights.detach().clone().requires_grad_(True)
    hidden_states = torch.empty((_TOKENS, _HIDDEN), device="cuda", dtype=torch.bfloat16)
    plan = ExpertRoutingPlan(
        selected_hidden_states=torch.empty(0, device="cuda"),
        expert_indices=expert_indices,
        routing_weights=routing_weights,
        permutation=permutation,
        inverse_permutation=inverse_permutation,
        num_tokens=_TOKENS,
        num_top_k=_TOP_K,
        hidden_dim=_HIDDEN,
        use_triton=True,
    )

    actual = finalize_expert_routing(output, hidden_states, plan, None)
    sorted_weights = reference_weights.reshape(-1)[permutation].to(output.dtype)
    expected = reference_output * sorted_weights.unsqueeze(-1)
    expected = expected[inverse_permutation]
    expected = expected.view(_TOKENS, _TOP_K, _HIDDEN).sum(dim=1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    grad_final = torch.randn(
        (_TOKENS, _HIDDEN),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    dispatched_ops: list[str] = []
    with _RecordOps(dispatched_ops):
        actual.backward(grad_final)
    expected.backward(grad_final)

    torch.testing.assert_close(output.grad, reference_output.grad, rtol=0, atol=0)
    torch.testing.assert_close(
        routing_weights.grad, reference_weights.grad, rtol=0, atol=0
    )
    assert not any("_index_put_impl_" in operation for operation in dispatched_ops)


def test_noncanonical_token_count_is_exact() -> None:
    num_tokens = 257
    num_routes = num_tokens * _TOP_K
    top_k_index, _, inverse_permutation = _routing_indices(num_tokens)
    _, permutation = torch.sort(top_k_index.reshape(-1))
    generator = torch.Generator(device="cuda").manual_seed(86420)
    hidden = torch.randn(
        (num_tokens, _HIDDEN),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    reference_hidden = hidden.detach().clone().requires_grad_(True)
    routing_weights = torch.rand(
        (num_tokens, _TOP_K),
        generator=generator,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    reference_weights = routing_weights.detach().clone().requires_grad_(True)
    output = torch.randn(
        (num_routes, _HIDDEN),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    reference_output = output.detach().clone().requires_grad_(True)

    plan = prepare_expert_routing(hidden, top_k_index, routing_weights)
    actual = finalize_expert_routing(output, hidden, plan, None)
    reference_selected = reference_hidden[permutation // _TOP_K]
    sorted_weights = reference_weights.reshape(-1)[permutation].to(output.dtype)
    expected = reference_output * sorted_weights.unsqueeze(-1)
    expected = expected[inverse_permutation]
    expected = expected.view(num_tokens, _TOP_K, _HIDDEN).sum(dim=1)

    torch.testing.assert_close(
        plan.selected_hidden_states, reference_selected, rtol=0, atol=0
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert plan.use_triton

    grad_selected = torch.randn(
        (num_routes, _HIDDEN),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    grad_final = torch.randn(
        (num_tokens, _HIDDEN),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    torch.autograd.backward(
        (plan.selected_hidden_states, actual),
        (grad_selected, grad_final),
    )
    torch.autograd.backward(
        (reference_selected, expected),
        (grad_selected, grad_final),
    )

    torch.testing.assert_close(hidden.grad, reference_hidden.grad, rtol=0, atol=0)
    torch.testing.assert_close(output.grad, reference_output.grad, rtol=0, atol=0)
    torch.testing.assert_close(
        routing_weights.grad, reference_weights.grad, rtol=0, atol=0
    )


def test_unknown_routing_shape_uses_torch_fallback() -> None:
    generator = torch.Generator(device="cuda").manual_seed(13579)
    hidden = torch.randn(
        (8, _HIDDEN),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    top_k_index = torch.randint(
        0,
        8,
        (8, 4),
        generator=generator,
        device="cuda",
        dtype=torch.int64,
    )
    routing_weights = torch.rand(
        (8, 4),
        generator=generator,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    plan = prepare_expert_routing(hidden, top_k_index, routing_weights)
    assert not plan.use_triton

    output = torch.randn(
        (32, _HIDDEN),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    actual = finalize_expert_routing(output, hidden, plan, None)
    assert actual.shape == hidden.shape
    loss = actual.sum() + plan.selected_hidden_states.sum()
    loss.backward()
    assert hidden.grad is not None
    assert routing_weights.grad is not None
    assert output.grad is not None
