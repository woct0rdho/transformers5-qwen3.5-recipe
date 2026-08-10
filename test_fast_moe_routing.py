import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from fast_moe_routing import (
    ExpertRoutingPlan,
    _is_supported_routing_geometry,
    _routing_combine_forward_launch,
    _routing_gather_forward_launch,
    _routing_launch_warps,
    finalize_expert_routing,
    prepare_expert_routing,
)

_TOKENS = 2048
_TOP_K = 8
_HIDDEN = 2048
_ROUTES = _TOKENS * _TOP_K


def _require_grad(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.grad is None:
        raise AssertionError("expected a tensor gradient")
    return tensor.grad


class _RecordOps(TorchDispatchMode):
    def __init__(self, dispatched_ops: list[str]) -> None:
        super().__init__()
        self.dispatched_ops = dispatched_ops

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.dispatched_ops.append(str(func))
        return func(*args, **(kwargs or {}))


def _assert_reasonable_routing_gradient(
    actual: torch.Tensor,
    reference: torch.Tensor,
) -> None:
    actual_float = actual.float().reshape(-1)
    reference_float = reference.float().reshape(-1)
    delta_rmse = (actual_float - reference_float).square().mean().sqrt()
    reference_rms = reference_float.square().mean().sqrt()
    cosine = torch.nn.functional.cosine_similarity(actual_float, reference_float, dim=0)
    assert torch.isfinite(actual_float).all()
    assert float(delta_rmse / reference_rms) < 0.005
    assert float(cosine) > 0.99999


def _routing_indices(
    num_tokens: int = _TOKENS,
    top_k: int = _TOP_K,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(19260817)
    top_k_index = torch.randint(
        0,
        256,
        (num_tokens, top_k),
        generator=generator,
        device="cuda",
        dtype=torch.int64,
    )
    expert_indices, permutation = torch.sort(top_k_index.reshape(-1))
    inverse_permutation = torch.empty_like(permutation)
    inverse_permutation[permutation] = torch.arange(num_tokens * top_k, device="cuda")
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
    assert _routing_gather_forward_launch(num_tokens, _TOP_K, _HIDDEN) == (
        2048,
        8,
    )
    assert _routing_combine_forward_launch(num_tokens, _TOP_K, _HIDDEN) == (
        2048,
        8 if batch_size == 1 else 4,
    )


@pytest.mark.parametrize(
    ("batch_size", "expected_warps"),
    [(1, (16, 16)), (4, (16, 16)), (16, (16, 8))],
)
def test_supported_deepseek_batch_geometry_and_launch_heuristics(
    batch_size: int,
    expected_warps: tuple[int, int],
) -> None:
    num_tokens = batch_size * 2048
    assert _is_supported_routing_geometry(num_tokens, 6, 4096)
    assert _routing_launch_warps(num_tokens, 6, 4096) == expected_warps
    assert _routing_gather_forward_launch(num_tokens, 6, 4096) == (2048, 8)
    assert _routing_combine_forward_launch(num_tokens, 6, 4096) == (4096, 8)


@pytest.mark.parametrize(
    ("num_tokens", "num_top_k", "hidden_dim"),
    [
        (0, 8, 2048),
        (32769, 8, 2048),
        (2048, 3, 1024),
        (2048, 8, 1024),
        (2048, 6, 2048),
        (2048, 8, 4096),
    ],
)
def test_unsupported_routing_geometry(
    num_tokens: int,
    num_top_k: int,
    hidden_dim: int,
) -> None:
    assert not _is_supported_routing_geometry(num_tokens, num_top_k, hidden_dim)


def test_route_gather_is_exact_and_has_no_index_put() -> None:
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

    dispatched_ops: list[str] = []
    with _RecordOps(dispatched_ops):
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

    grad_selected = torch.randn(
        (_ROUTES, _HIDDEN),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    with _RecordOps(dispatched_ops):
        plan.selected_hidden_states.backward(grad_selected)
    reference_selected.backward(grad_selected)

    torch.testing.assert_close(hidden.grad, reference_hidden.grad, rtol=0, atol=0)
    assert not any("_index_put_impl_" in operation for operation in dispatched_ops)


@pytest.mark.parametrize("routing_dtype", [torch.bfloat16, torch.float32])
def test_route_combine_forward_and_backward_match_reference(
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
    raw_weights = torch.rand(
        (_TOKENS, _TOP_K),
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    routing_weights = (
        (raw_weights / raw_weights.sum(dim=-1, keepdim=True))
        .to(routing_dtype)
        .detach()
        .requires_grad_(True)
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
    )

    actual = finalize_expert_routing(output, hidden_states, plan, None)
    sorted_weights = reference_weights.reshape(-1)[permutation].to(output.dtype)
    expected = reference_output * sorted_weights.unsqueeze(-1)
    expected = expected[inverse_permutation]
    expected = expected.view(_TOKENS, _TOP_K, _HIDDEN).sum(dim=1)
    # The fused kernel accumulates weighted BF16 expert outputs in FP32 before
    # the final BF16 store. This is more accurate but can differ by one BF16 step.
    torch.testing.assert_close(actual, expected, rtol=0, atol=0.015625)

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

    torch.testing.assert_close(
        _require_grad(output), _require_grad(reference_output), rtol=0, atol=0
    )
    _assert_reasonable_routing_gradient(
        _require_grad(routing_weights), _require_grad(reference_weights)
    )
    assert not any("_index_put_impl_" in operation for operation in dispatched_ops)


@pytest.mark.parametrize(
    ("top_k", "hidden_dim"),
    [(8, 2048), (6, 4096)],
)
def test_noncanonical_token_count_matches_reference(
    top_k: int,
    hidden_dim: int,
) -> None:
    num_tokens = 257
    num_routes = num_tokens * top_k
    top_k_index, _, inverse_permutation = _routing_indices(num_tokens, top_k)
    _, permutation = torch.sort(top_k_index.reshape(-1))
    generator = torch.Generator(device="cuda").manual_seed(86420)
    hidden = torch.randn(
        (num_tokens, hidden_dim),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    reference_hidden = hidden.detach().clone().requires_grad_(True)
    raw_weights = torch.rand(
        (num_tokens, top_k),
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    routing_weights = (
        (raw_weights / raw_weights.sum(dim=-1, keepdim=True))
        .detach()
        .requires_grad_(True)
    )
    reference_weights = routing_weights.detach().clone().requires_grad_(True)
    output = torch.randn(
        (num_routes, hidden_dim),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    reference_output = output.detach().clone().requires_grad_(True)

    plan = prepare_expert_routing(hidden, top_k_index, routing_weights)
    actual = finalize_expert_routing(output, hidden, plan, None)
    reference_selected = reference_hidden[permutation // top_k]
    sorted_weights = reference_weights.reshape(-1)[permutation].to(output.dtype)
    expected = reference_output * sorted_weights.unsqueeze(-1)
    expected = expected[inverse_permutation]
    expected = expected.view(num_tokens, top_k, hidden_dim).sum(dim=1)

    torch.testing.assert_close(
        plan.selected_hidden_states, reference_selected, rtol=0, atol=0
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0.015625)

    grad_selected = torch.randn(
        (num_routes, hidden_dim),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    grad_final = torch.randn(
        (num_tokens, hidden_dim),
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
    torch.testing.assert_close(
        _require_grad(output), _require_grad(reference_output), rtol=0, atol=0
    )
    _assert_reasonable_routing_gradient(
        _require_grad(routing_weights), _require_grad(reference_weights)
    )


def test_unknown_routing_shape_is_rejected() -> None:
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
        (8, 3),
        generator=generator,
        device="cuda",
        dtype=torch.int64,
    )
    routing_weights = torch.rand(
        (8, 3),
        generator=generator,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    with pytest.raises(RuntimeError, match="Unsupported optimized"):
        prepare_expert_routing(hidden, top_k_index, routing_weights)
