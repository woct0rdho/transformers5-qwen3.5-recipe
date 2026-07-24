import pytest
import torch

from deepseek_v4_routing import (
    compare_deepseek_v4_route_weights,
    summarize_deepseek_v4_routes,
    validate_deepseek_v4_routes,
)


def test_top6_hidden4096_route_contract_and_summary() -> None:
    hidden = torch.randn(5, 4096, dtype=torch.bfloat16)
    indices = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [0, 1, 2, 3, 4, 6],
            [0, 1, 2, 3, 7, 8],
            [0, 1, 2, 9, 10, 11],
            [0, 1, 12, 13, 14, 15],
        ],
        dtype=torch.long,
    )
    weights = torch.softmax(torch.randn(5, 6), dim=-1)
    validate_deepseek_v4_routes(hidden, indices, weights)
    summary = summarize_deepseek_v4_routes(
        indices, weights, layer=3, router_kind="hash"
    )
    assert summary["layer"] == 3
    assert summary["router_kind"] == "hash"
    assert summary["tokens"] == 5
    assert summary["routes"] == 30
    assert summary["active_experts"] == 16
    assert summary["rows_max"] == 5
    assert sum(summary["rows_per_expert"]) == 30
    assert len(summary["sorted_weight_sample"]) == 5
    assert summary["weight_min"] >= 0.0


def test_route_weight_comparison_ignores_expert_identity_and_weight_order() -> None:
    torch.manual_seed(7)
    reference_indices = torch.arange(6).repeat(17, 1)
    candidate_indices = torch.arange(100, 106).repeat(17, 1)
    reference_weights = torch.softmax(torch.randn(17, 6), dim=-1)
    candidate_weights = reference_weights.flip(-1)
    reference = [
        summarize_deepseek_v4_routes(
            reference_indices,
            reference_weights,
            layer=9,
            router_kind="learned",
        )
    ]
    candidate = [
        summarize_deepseek_v4_routes(
            candidate_indices,
            candidate_weights,
            layer=9,
            router_kind="learned",
        )
    ]

    comparison = compare_deepseek_v4_route_weights(reference, candidate)
    assert comparison["sampled_weights"] == 17 * 6
    assert comparison["cosine"] == pytest.approx(1.0)
    assert comparison["relative_rmse"] == pytest.approx(0.0)
    assert comparison["max_abs"] == pytest.approx(0.0)


def test_route_contract_rejects_qwen_geometry() -> None:
    hidden = torch.randn(4, 2048, dtype=torch.bfloat16)
    indices = torch.zeros(4, 8, dtype=torch.long)
    weights = torch.ones(4, 8)
    with pytest.raises(ValueError, match="4096"):
        validate_deepseek_v4_routes(hidden, indices, weights)
