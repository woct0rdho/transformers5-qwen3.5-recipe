from copy import deepcopy
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4HashRouter,
    DeepseekV4TopKRouter,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeTopKRouter,
)
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextTopKRouter

from fast_moe_ranking import (
    _is_supported_router_geometry,
    _router_topk_launch,
    configure_fast_moe_ranking,
    require_complete_fast_moe_ranking,
    router_topk_indices,
)
from test_support import assert_relative_rmse, require_grad


def _qwen_config() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=2048,
        num_experts=256,
        num_experts_per_tok=8,
    )


def _qwen4_config() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=2560,
        num_experts=512,
        num_experts_per_tok=10,
        norm_topk_prob=True,
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
    def __init__(
        self,
        router: torch.nn.Module,
        *,
        scoring_func: str | None = None,
        model_type: str = "test",
    ):
        super().__init__()
        self.config = SimpleNamespace(model_type=model_type, scoring_func=scoring_func)
        self.router = router


class _MoeLayer(torch.nn.Module):
    def __init__(self, router: torch.nn.Module) -> None:
        super().__init__()
        self.gate = router
        self.experts = torch.nn.Module()


class _MoeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(model_type="test", scoring_func="sqrtsoftplus")
        self.layers = torch.nn.ModuleList(
            [
                _MoeLayer(DeepseekV4HashRouter(cast(Any, _deepseek_config()))),
                _MoeLayer(DeepseekV4TopKRouter(cast(Any, _deepseek_config()))),
            ]
        )


def _assert_valid_route_indices(
    indices: torch.Tensor, top_k: int, num_experts: int = 256
) -> None:
    assert indices.shape[-1] == top_k
    assert bool(torch.all((indices >= 0) & (indices < num_experts)))
    sorted_indices = torch.sort(indices, dim=-1).values
    assert bool(torch.all(sorted_indices.diff(dim=-1) > 0))


def _assert_selected_weights_close(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    *,
    minimum_cosine: float = 0.999,
    maximum_relative_rmse: float = 3e-2,
    maximum_absolute_error: float = 2e-2,
    maximum_normalized_row_sum_error: float = 1e-2,
) -> None:
    """Compare selected weights without making expert IDs part of the gate."""

    candidate = candidate.detach().float().sort(dim=-1).values
    reference = reference.detach().float().sort(dim=-1).values
    delta = candidate - reference
    candidate_flat = candidate.flatten()
    reference_flat = reference.flatten()
    cosine = torch.nn.functional.cosine_similarity(
        candidate_flat, reference_flat, dim=0
    )
    relative_rmse = delta.square().mean().sqrt() / (
        reference_flat.square().mean().sqrt() + 1e-12
    )
    assert float(cosine) >= minimum_cosine
    assert float(relative_rmse) <= maximum_relative_rmse
    assert float(delta.abs().max()) <= maximum_absolute_error

    reference_row_sum = reference.sum(dim=-1)
    normalized_row_sum_error = (candidate.sum(dim=-1) - reference_row_sum).abs() / (
        reference_row_sum.abs() + 1e-12
    )
    assert float(normalized_row_sum_error.max()) <= maximum_normalized_row_sum_error


def _symmetric_weight_loss(weights: torch.Tensor) -> torch.Tensor:
    """Give the selected-weight backward test no expert-ID or order preference."""

    return weights.float().square().sum()


def _assert_finite_nonzero_gradient(tensor: torch.Tensor) -> None:
    assert bool(torch.isfinite(tensor).all())
    assert bool(torch.any(tensor != 0))


class _RecordOps(TorchDispatchMode):
    def __init__(self, dispatched_ops: list[str]) -> None:
        super().__init__()
        self.dispatched_ops = dispatched_ops

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.dispatched_ops.append(str(func))
        return func(*args, **(kwargs or {}))


def _full_fp32_linear(
    hidden_states: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    return torch.mm(flat.float(), weight.float().transpose(0, 1))


def _qwen_fp32_reference(
    router: Qwen3_5MoeTopKRouter, hidden_states: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = _full_fp32_linear(hidden_states, router.weight)
    probabilities = torch.softmax(logits, dim=-1)
    values, indices = torch.topk(probabilities, router.top_k, dim=-1)
    weights = values / values.sum(dim=-1, keepdim=True)
    return logits, weights, indices


def _qwen4_fp32_reference(
    router: Qwen4ExpTextTopKRouter, hidden_states: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = _full_fp32_linear(hidden_states, router.weight)
    probabilities = torch.softmax(logits, dim=-1)
    values, indices = torch.topk(probabilities, router.top_k, dim=-1)
    weights = values / values.sum(dim=-1, keepdim=True)
    return logits, weights, indices


def _deepseek_fp32_reference(
    router: Any, hidden_states: torch.Tensor, input_ids: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = _full_fp32_linear(hidden_states, router.weight)
    scores = router.score_fn(logits)
    if input_ids is None:
        indices = torch.topk(
            scores + router.e_score_correction_bias.float(),
            router.top_k,
            dim=-1,
            sorted=False,
        ).indices
    else:
        indices = router.tid2eid[input_ids.reshape(-1)].long()
    weights = scores.gather(1, indices)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return logits, weights * router.routed_scaling_factor, indices


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
    assert _is_supported_router_geometry(tokens, 512, 10)
    assert _router_topk_launch(tokens, 256) == qwen_launch == deepseek_launch
    assert _router_topk_launch(tokens, 512) == (4, 64, 2)


@pytest.mark.parametrize(
    ("tokens", "experts", "top_k"),
    [
        (0, 256, 8),
        (32769, 256, 8),
        (2048, 128, 8),
        (2048, 256, 4),
        (2048, 512, 8),
        (2048, 512, 12),
    ],
)
def test_unknown_router_geometry_is_not_specialized(
    tokens: int, experts: int, top_k: int
) -> None:
    assert not _is_supported_router_geometry(tokens, experts, top_k)


def test_unknown_router_geometry_is_rejected() -> None:
    logits = torch.randn(8, 128, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="supports only the 256-expert"):
        router_topk_indices(logits, 4)


def test_router_configuration_binds_prior_to_owning_experts() -> None:
    model = _MoeModel()
    result = configure_fast_moe_ranking(model)

    assert result["deepseek_hash"] == 1
    assert result["deepseek_topk"] == 1
    first_experts = model.layers[0].experts
    second_experts = model.layers[1].experts
    assert first_experts.__dict__["_aiter_expert_prior"] == "deepseek-hash"
    assert second_experts.__dict__["_aiter_expert_prior"] == "deepseek-learned"


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
    logits_reference, weights_reference, indices_reference = _qwen_fp32_reference(
        reference, hidden_reference
    )
    logits_optimized, weights_optimized, indices_optimized = optimized(hidden_optimized)

    _assert_valid_route_indices(indices_reference, 8)
    _assert_valid_route_indices(indices_optimized, 8)
    assert logits_optimized.dtype == torch.float32
    assert weights_optimized.dtype == torch.float32
    # The projection is intentionally a BF16 mm upcast to FP32, so the logits
    # match the full-FP32 oracle only to the single BF16 output rounding
    # (measured 1.7e-3 relative RMSE). The selected weights below are still
    # gated against the full-FP32 reference.
    assert_relative_rmse(logits_optimized, logits_reference, 1e-2)
    _assert_selected_weights_close(weights_optimized, weights_reference)

    loss_reference = _symmetric_weight_loss(weights_reference)
    loss_optimized = _symmetric_weight_loss(weights_optimized)
    loss_reference.backward()
    loss_optimized.backward()
    _assert_finite_nonzero_gradient(require_grad(hidden_reference))
    _assert_finite_nonzero_gradient(require_grad(hidden_optimized))


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
    logits_reference, weights_reference, indices_reference = _deepseek_fp32_reference(
        reference, hidden_reference
    )
    logits_optimized, weights_optimized, indices_optimized = optimized(hidden_optimized)

    _assert_valid_route_indices(indices_reference, 6)
    _assert_valid_route_indices(indices_optimized, 6)
    assert logits_optimized.dtype == torch.float32
    assert weights_optimized.dtype == torch.float32
    # Same intentional BF16 projection rounding as the Qwen router test.
    assert_relative_rmse(logits_optimized, logits_reference, 1e-2)
    _assert_selected_weights_close(weights_optimized, weights_reference)

    loss_reference = _symmetric_weight_loss(weights_reference)
    loss_optimized = _symmetric_weight_loss(weights_optimized)
    loss_reference.backward()
    loss_optimized.backward()
    _assert_finite_nonzero_gradient(require_grad(hidden_reference))
    _assert_finite_nonzero_gradient(require_grad(hidden_optimized))


def test_qwen4_router_matches_reference_forward_and_gradient() -> None:
    torch.manual_seed(4242)
    reference = Qwen4ExpTextTopKRouter(_qwen4_config()).to(
        device="cuda", dtype=torch.bfloat16
    )
    reference.weight.data.normal_(std=0.02)
    optimized = deepcopy(reference)
    configure_fast_moe_ranking(_RouterModel(optimized))

    hidden_reference = torch.randn(
        257, 2560, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    hidden_optimized = hidden_reference.detach().clone().requires_grad_(True)
    logits_reference, weights_reference, indices_reference = _qwen4_fp32_reference(
        reference, hidden_reference
    )
    logits_optimized, weights_optimized, indices_optimized = optimized(hidden_optimized)

    _assert_valid_route_indices(indices_reference, 10, num_experts=512)
    _assert_valid_route_indices(indices_optimized, 10, num_experts=512)
    assert logits_optimized.dtype == torch.float32
    # Same intentional BF16 projection rounding as the other router tests.
    assert_relative_rmse(logits_optimized, logits_reference, 1e-2)
    _assert_selected_weights_close(
        weights_optimized.float(),
        weights_reference.to(weights_optimized.dtype).float(),
    )

    loss_reference = _symmetric_weight_loss(weights_reference)
    loss_optimized = _symmetric_weight_loss(weights_optimized)
    loss_reference.backward()
    loss_optimized.backward()
    _assert_finite_nonzero_gradient(require_grad(hidden_reference))
    _assert_finite_nonzero_gradient(require_grad(hidden_optimized))


def test_qwen4_tied_scores_resolve_to_the_lower_expert_index() -> None:
    reference = Qwen4ExpTextTopKRouter(_qwen4_config()).to(
        device="cuda", dtype=torch.bfloat16
    )
    reference.weight.data.zero_()
    optimized = deepcopy(reference)
    configure_fast_moe_ranking(_RouterModel(optimized))

    hidden = torch.randn(32, 2560, device="cuda", dtype=torch.bfloat16)
    _, first_values, first_indices = optimized(hidden)
    _, second_values, second_indices = optimized(hidden)

    # Every expert ties, so the selection is the lowest ten ids, and it repeats across calls. The
    # reference leaves both the choice and the repeat to `torch.topk`.
    expected = torch.arange(10, device="cuda").expand(32, 10)
    assert torch.equal(first_indices, expected)
    assert torch.equal(second_indices, expected)
    assert torch.equal(first_values, second_values)


def test_qwen4_router_inventory_is_required() -> None:
    routers = [
        Qwen4ExpTextTopKRouter(_qwen4_config()).cuda().to(torch.bfloat16)
        for _ in range(2)
    ]
    layers = torch.nn.ModuleList([_MoeLayer(router) for router in routers])
    model = _RouterModel(torch.nn.Module())
    model.layers = layers
    report = configure_fast_moe_ranking(model)
    assert report["qwen4"] == 2
    with pytest.raises(RuntimeError, match="incomplete"):
        require_complete_fast_moe_ranking(report, "qwen4_exp_text")


def test_router_ties_accept_any_expert_at_the_kth_threshold() -> None:
    qwen_logits = torch.zeros(19, 256, device="cuda", dtype=torch.float32)
    qwen4_logits = torch.zeros(19, 512, device="cuda", dtype=torch.float32)
    qwen_indices = router_topk_indices(qwen_logits, 8)
    deepseek_indices = router_topk_indices(
        qwen_logits,
        6,
        correction_bias=torch.zeros(256, device="cuda"),
        score_function="sqrtsoftplus",
    )
    qwen4_indices = router_topk_indices(qwen4_logits, 10)

    for logits, indices, top_k in (
        (qwen_logits, qwen_indices, 8),
        (qwen_logits, deepseek_indices, 6),
        (qwen4_logits, qwen4_indices, 10),
    ):
        assert indices.shape == (19, top_k)
        _assert_valid_route_indices(indices, top_k, num_experts=logits.shape[1])
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
    reference.weight.requires_grad_(False)
    reference.tid2eid.copy_(
        torch.randint(0, 256, reference.tid2eid.shape, device="cuda")
    )
    optimized = deepcopy(reference)
    configure_fast_moe_ranking(_RouterModel(optimized, scoring_func="sqrtsoftplus"))

    hidden = torch.randn(2, 17, 4096, device="cuda", dtype=torch.bfloat16)
    input_ids = torch.randint(0, 512, (2, 17), device="cuda")
    expected_logits, expected_weights, expected_indices = _deepseek_fp32_reference(
        reference, hidden, input_ids
    )
    dispatches: list[str] = []
    with _RecordOps(dispatches):
        actual_logits, actual_weights, actual_indices = optimized(hidden, input_ids)

    torch.testing.assert_close(actual_indices, expected_indices, rtol=0, atol=0)
    assert actual_logits.dtype == torch.float32
    assert actual_logits.shape == (2 * 17, 6)
    assert actual_weights.dtype == torch.float32
    # The six selected logits are the only ones projected: no full-width expert
    # GEMM or [tokens, 256] score tensor may be materialized.
    assert not any(
        op in dispatches
        for op in (
            "torch.ops.aten.mm.default",
            "torch.ops.aten.addmm.default",
            "torch.ops.aten.linear.default",
            "torch.ops.aten.bmm.default",
        )
    )
    # Same intentional BF16 projection rounding as the other router tests.
    assert_relative_rmse(
        actual_logits,
        expected_logits.gather(1, expected_indices),
        1e-2,
    )
    _assert_selected_weights_close(actual_weights, expected_weights)


def test_deepseek_hash_router_selected_logits_gradient_matches_reference() -> None:
    torch.manual_seed(3141)
    reference = cast(
        Any,
        DeepseekV4HashRouter(cast(Any, _deepseek_config())).to(
            device="cuda", dtype=torch.bfloat16
        ),
    )
    reference.weight.data.normal_(std=0.02)
    reference.weight.requires_grad_(False)
    reference.tid2eid.copy_(
        torch.randint(0, 256, reference.tid2eid.shape, device="cuda")
    )
    optimized = deepcopy(reference)
    configure_fast_moe_ranking(_RouterModel(optimized, scoring_func="sqrtsoftplus"))

    input_ids = torch.randint(0, 512, (2, 17), device="cuda")
    hidden_reference = torch.randn(
        2, 17, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    hidden_optimized = hidden_reference.detach().clone().requires_grad_(True)

    _, weights_reference, _ = _deepseek_fp32_reference(
        reference, hidden_reference, input_ids
    )
    _, weights_optimized, _ = optimized(hidden_optimized, input_ids)
    _symmetric_weight_loss(weights_reference).backward()
    _symmetric_weight_loss(weights_optimized).backward()

    _assert_finite_nonzero_gradient(require_grad(hidden_reference))
    _assert_finite_nonzero_gradient(require_grad(hidden_optimized))
    assert_relative_rmse(
        require_grad(hidden_optimized),
        require_grad(hidden_reference),
        1e-2,
    )


def test_deepseek_hash_router_rejects_trainable_gate_weights() -> None:
    router = cast(Any, DeepseekV4HashRouter(cast(Any, _deepseek_config()))).to(
        device="cuda", dtype=torch.bfloat16
    )
    router.tid2eid.copy_(torch.randint(0, 256, router.tid2eid.shape, device="cuda"))
    configure_fast_moe_ranking(_RouterModel(router, scoring_func="sqrtsoftplus"))

    hidden = torch.randn(2, 17, 4096, device="cuda", dtype=torch.bfloat16)
    input_ids = torch.randint(0, 512, (2, 17), device="cuda")
    with pytest.raises(RuntimeError, match="frozen"):
        router(hidden, input_ids)
