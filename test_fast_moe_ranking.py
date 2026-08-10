from copy import deepcopy
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4HashRouter,
    DeepseekV4TopKRouter,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeTopKRouter,
)

from fast_moe_ranking import (
    _is_supported_router_geometry,
    _router_topk_launch,
    configure_fast_moe_ranking,
    router_topk_indices,
)


def _qwen_config() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=2048,
        num_experts=256,
        num_experts_per_tok=8,
    )


def _deepseek_config() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=4096,
        num_local_experts=256,
        num_experts_per_tok=6,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=2.5,
        vocab_size=512,
    )


class _RouterModel(torch.nn.Module):
    def __init__(self, router: torch.nn.Module, *, scoring_func: str | None = None):
        super().__init__()
        self.config = SimpleNamespace(model_type="test", scoring_func=scoring_func)
        self.router = router


def _sort_routes(
    weights: torch.Tensor, indices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    sorted_indices, order = torch.sort(indices, dim=-1)
    return weights.gather(1, order), sorted_indices


def _require_grad(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.grad is None:
        raise AssertionError("expected a tensor gradient")
    return tensor.grad


@pytest.mark.parametrize(
    ("batch_size", "qwen_launch", "deepseek_launch"),
    [
        (1, (4, 64, 4), (4, 64, 4)),
        (4, (8, 64, 8), (8, 64, 8)),
        (16, (8, 64, 8), (8, 64, 8)),
    ],
)
def test_router_geometry_and_launch_heuristics(
    batch_size: int,
    qwen_launch: tuple[int, int, int],
    deepseek_launch: tuple[int, int, int],
) -> None:
    tokens = batch_size * 2048
    assert _is_supported_router_geometry(tokens, 256, 8)
    assert _is_supported_router_geometry(tokens, 256, 6)
    assert _router_topk_launch(tokens) == qwen_launch == deepseek_launch


@pytest.mark.parametrize(
    ("tokens", "experts", "top_k"),
    [(0, 256, 8), (32769, 256, 8), (2048, 128, 8), (2048, 256, 4)],
)
def test_unknown_router_geometry_is_not_specialized(
    tokens: int, experts: int, top_k: int
) -> None:
    assert not _is_supported_router_geometry(tokens, experts, top_k)


def test_unknown_router_geometry_is_rejected() -> None:
    logits = torch.randn(8, 128, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="supports only 256-expert"):
        router_topk_indices(logits, 4)


def test_qwen_router_matches_reference_forward_and_gradient() -> None:
    torch.manual_seed(1234)
    reference = Qwen3_5MoeTopKRouter(_qwen_config()).to(
        device="cuda", dtype=torch.bfloat16
    )
    reference.weight.data.normal_(std=0.02)
    optimized = deepcopy(reference)
    configure_fast_moe_ranking(_RouterModel(optimized))

    hidden_reference = torch.randn(
        257, 2048, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    hidden_optimized = hidden_reference.detach().clone().requires_grad_(True)
    logits_reference, weights_reference, indices_reference = reference(hidden_reference)
    logits_optimized, weights_optimized, indices_optimized = optimized(hidden_optimized)

    reference_sorted, reference_indices = _sort_routes(
        weights_reference, indices_reference
    )
    optimized_sorted, optimized_indices = _sort_routes(
        weights_optimized, indices_optimized
    )
    torch.testing.assert_close(logits_optimized, logits_reference, rtol=0, atol=0)
    torch.testing.assert_close(optimized_indices, reference_indices, rtol=0, atol=0)
    torch.testing.assert_close(optimized_sorted, reference_sorted, rtol=1e-3, atol=1e-3)

    expert_values = torch.randn(256, device="cuda", dtype=torch.float32)
    loss_reference = (
        weights_reference.float() * expert_values[indices_reference]
    ).sum()
    loss_optimized = (
        weights_optimized.float() * expert_values[indices_optimized]
    ).sum()
    loss_reference.backward()
    loss_optimized.backward()
    torch.testing.assert_close(
        _require_grad(hidden_optimized),
        _require_grad(hidden_reference),
        rtol=2e-3,
        atol=2e-3,
    )


def test_deepseek_router_matches_reference_forward_and_gradient() -> None:
    torch.manual_seed(5678)
    reference = cast(
        Any,
        DeepseekV4TopKRouter(cast(Any, _deepseek_config())).to(
            device="cuda", dtype=torch.bfloat16
        ),
    )
    reference.weight.data.normal_(std=0.02)
    reference.e_score_correction_bias.data.normal_(std=0.01)
    optimized = deepcopy(reference)
    configure_fast_moe_ranking(_RouterModel(optimized, scoring_func="sqrtsoftplus"))

    hidden_reference = torch.randn(
        257, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    hidden_optimized = hidden_reference.detach().clone().requires_grad_(True)
    logits_reference, weights_reference, indices_reference = reference(hidden_reference)
    logits_optimized, weights_optimized, indices_optimized = optimized(hidden_optimized)

    reference_sorted, reference_indices = _sort_routes(
        weights_reference, indices_reference
    )
    optimized_sorted, optimized_indices = _sort_routes(
        weights_optimized, indices_optimized
    )
    torch.testing.assert_close(logits_optimized, logits_reference, rtol=0, atol=0)
    torch.testing.assert_close(optimized_indices, reference_indices, rtol=0, atol=0)
    torch.testing.assert_close(optimized_sorted, reference_sorted, rtol=1e-6, atol=1e-6)

    expert_values = torch.randn(256, device="cuda", dtype=torch.float32)
    loss_reference = (weights_reference * expert_values[indices_reference]).sum()
    loss_optimized = (weights_optimized * expert_values[indices_optimized]).sum()
    loss_reference.backward()
    loss_optimized.backward()
    torch.testing.assert_close(
        _require_grad(hidden_optimized),
        _require_grad(hidden_reference),
        rtol=1e-5,
        atol=1e-5,
    )


def test_router_ties_accept_any_expert_at_the_kth_threshold() -> None:
    logits = torch.zeros(19, 256, device="cuda", dtype=torch.float32)
    qwen_indices = router_topk_indices(logits, 8)
    deepseek_indices = router_topk_indices(
        logits,
        6,
        correction_bias=torch.zeros(256, device="cuda"),
        score_function="sqrtsoftplus",
    )

    for indices, top_k in ((qwen_indices, 8), (deepseek_indices, 6)):
        assert indices.shape == (19, top_k)
        assert bool(torch.all((indices >= 0) & (indices < 256)))
        assert bool(torch.all(torch.sort(indices, dim=-1).values.diff(dim=-1) > 0))
        # Every expert is tied at the kth threshold, so expert identity is not
        # compared with torch.topk's implementation-defined tie choice.
        selected_scores = logits.gather(1, indices)
        kth_threshold = torch.topk(logits, top_k, dim=-1).values[:, -1:]
        assert bool(torch.all(selected_scores >= kth_threshold))


def test_deepseek_hash_router_avoids_full_width_score_materialization() -> None:
    torch.manual_seed(9012)
    reference = cast(
        Any,
        DeepseekV4HashRouter(cast(Any, _deepseek_config())).to(
            device="cuda", dtype=torch.bfloat16
        ),
    )
    reference.weight.data.normal_(std=0.02)
    reference.tid2eid.copy_(
        torch.randint(0, 256, reference.tid2eid.shape, device="cuda")
    )
    optimized = deepcopy(reference)
    configure_fast_moe_ranking(_RouterModel(optimized, scoring_func="sqrtsoftplus"))

    hidden = torch.randn(2, 17, 4096, device="cuda", dtype=torch.bfloat16)
    input_ids = torch.randint(0, 512, (2, 17), device="cuda")
    expected = reference(hidden, input_ids)
    actual = optimized(hidden, input_ids)
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)
