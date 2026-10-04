import os
import warnings
from pathlib import Path
from types import SimpleNamespace

import gguf
import numpy as np
import pytest
import torch
from peft import LoraConfig
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.checkpoint import checkpoint
from transformers.integrations.gguf.gguf_quantized_parameter import (
    GgufQuantizedParameter,
)
from transformers.integrations.gguf.moe import ALL_GGUF_EXPERTS_FUNCTIONS, GgufExperts

import fast_moe_lora
from fast_moe_lora import (
    QWEN3_5_MOE_EXPERTS_IMPLEMENTATION,
    FastGgufMoeLora,
    _aiter_input_grad,
    _base_grouped_linear,
    _base_grouped_pair,
    _prepare_packed_expert_execution,
    aiter_grouped_mm,
    gguf_mmq_aiter_lora_forward,
)
from gguf_support import dequantize_gguf_tensor
from test_support import require_grad


class _RecordOps(TorchDispatchMode):
    def __init__(self, dispatched_ops: list[str]) -> None:
        super().__init__()
        self.dispatched_ops = dispatched_ops

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.dispatched_ops.append(str(func))
        return func(*args, **(kwargs or {}))


_MODEL = Path(
    os.environ.get(
        "GGUF_MMQ_TEST_MODEL",
        os.path.expanduser("~/models/qwen3.6/Qwen3.6-35B-A3B-APEX-I-Mini.gguf"),
    )
)


@pytest.fixture(autouse=True)
def synthetic_aiter_configs(monkeypatch) -> None:
    config = {
        "BLOCK_SIZE_M": 32,
        "BLOCK_SIZE_K": 32,
        "BLOCK_SIZE_N": 32,
        "GROUP_SIZE": 1,
        "GRID_DIM": 40,
        "num_warps": 4,
        "num_stages": 1,
    }
    monkeypatch.setattr(fast_moe_lora, "_gmm_config", lambda *_, **__: dict(config))
    monkeypatch.setattr(fast_moe_lora, "_ptgmm_config", lambda *_, **__: dict(config))


@pytest.fixture(scope="module")
def reader() -> gguf.GGUFReader:
    if not _MODEL.is_file():
        pytest.skip("GGUF model is unavailable")
    return gguf.GGUFReader(_MODEL)


def _packed_projection(
    reader: gguf.GGUFReader,
    projection: str,
    *,
    num_experts: int,
    out_features: int,
    layer: int = 10,
) -> GgufQuantizedParameter:
    tensor = next(
        item
        for item in reader.tensors
        if item.name == f"blk.{layer}.ffn_{projection}_exps.weight"
    )
    payload = torch.from_numpy(
        np.array(
            tensor.data[:num_experts, :out_features],
            dtype=np.uint8,
            copy=True,
            order="C",
        )
    ).to("cuda")
    return GgufQuantizedParameter(
        payload,
        quant_type=tensor.tensor_type,
        logical_shape=(num_experts, out_features, int(tensor.shape[0])),
    )


def _logical_pair_input_gradient(
    first_grad_output: torch.Tensor,
    second_grad_output: torch.Tensor,
    first_weight: torch.Tensor,
    second_weight: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    grad_input = torch.empty(
        first_grad_output.shape[0],
        first_weight.shape[-1],
        device=first_grad_output.device,
        dtype=first_grad_output.dtype,
    )
    row_begin = 0
    for group, row_end in enumerate(offsets.cpu().tolist()):
        combined = (
            first_grad_output[row_begin:row_end].float() @ first_weight[group].float()
        )
        combined.addmm_(
            second_grad_output[row_begin:row_end].float(),
            second_weight[group].float(),
        )
        grad_input[row_begin:row_end] = combined.to(first_grad_output.dtype)
        row_begin = row_end
    return grad_input


def test_packed_expert_projection_backward_is_exact_logical_jacobian(
    reader: gguf.GGUFReader,
) -> None:
    experts = torch.tensor([0, 2, 5], device="cuda", dtype=torch.int64)
    offsets = torch.tensor([4096, 12288, 16384], device="cuda", dtype=torch.int32)
    group_sizes = torch.tensor([4096, 8192, 4096], device="cuda", dtype=torch.int32)
    generator = torch.Generator(device="cuda").manual_seed(2468)

    gate = _packed_projection(reader, "gate", num_experts=256, out_features=512)
    up = _packed_projection(reader, "up", num_experts=256, out_features=512)
    hidden = torch.randn(
        16384,
        2048,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    gate_grad = torch.randn(
        16384,
        512,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    up_grad = torch.randn(
        16384,
        512,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    pair_ops: list[str] = []
    with _RecordOps(pair_ops):
        gate_output, up_output = _base_grouped_pair(
            hidden,
            gate,
            up,
            experts,
            offsets,
            group_sizes,
            torch.bfloat16,
        )
        torch.autograd.backward((gate_output, up_output), (gate_grad, up_grad))
    assert "torch_ggml_ops._grouped_mmq_pair_launch.default" in pair_ops
    assert "torch_ggml_ops._grouped_mmq_pair_grad_input_launch.default" in pair_ops

    logical_gate = dequantize_gguf_tensor(
        gate.as_subclass(torch.Tensor).index_select(0, experts),
        gate.quant_type,
        dtype=torch.bfloat16,
        device="cuda",
    )
    logical_up = dequantize_gguf_tensor(
        up.as_subclass(torch.Tensor).index_select(0, experts),
        up.quant_type,
        dtype=torch.bfloat16,
        device="cuda",
    )
    expected_hidden_grad = _logical_pair_input_gradient(
        gate_grad,
        up_grad,
        logical_gate,
        logical_up,
        offsets,
    )
    # Paired packed backward combines both terms in one FP32 accumulator and
    # rounds once to BF16, so Torch GEMM may differ only by reduction order.
    torch.testing.assert_close(
        require_grad(hidden), expected_hidden_grad, rtol=0, atol=2e-2
    )
    assert gate.grad is None
    assert up.grad is None

    down = _packed_projection(reader, "down", num_experts=256, out_features=2048)
    intermediate = torch.randn(
        16384,
        512,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    down_grad = torch.randn(
        16384,
        2048,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    down_ops: list[str] = []
    with _RecordOps(down_ops):
        down_output = _base_grouped_linear(
            intermediate,
            down,
            experts,
            offsets,
            group_sizes,
            torch.bfloat16,
        )
        down_output.backward(down_grad)
    assert "torch_ggml_ops._grouped_mmq_launch.default" in down_ops
    assert "torch_ggml_ops._grouped_mmq_grad_input_launch.default" in down_ops
    logical_down = dequantize_gguf_tensor(
        down.as_subclass(torch.Tensor).index_select(0, experts),
        down.quant_type,
        dtype=torch.bfloat16,
        device="cuda",
    )
    expected_intermediate_grad = _aiter_input_grad(
        down_grad,
        logical_down.transpose(1, 2),
        group_sizes,
        expert_prior="qwen-learned",
    )
    torch.testing.assert_close(
        require_grad(intermediate), expected_intermediate_grad, rtol=0, atol=0
    )
    assert down.grad is None


@pytest.fixture
def sync_debug_mode():
    """Have the CUDA runtime report the synchronizing operations a call performs."""

    if not hasattr(torch.cuda, "set_sync_debug_mode"):
        pytest.skip("torch.cuda.set_sync_debug_mode is unavailable")
    torch.cuda.set_sync_debug_mode("warn")
    yield
    torch.cuda.set_sync_debug_mode("default")


def _count_synchronizing_calls(call) -> int:
    """Count the synchronizing CUDA operations a call performs, under the `sync_debug_mode` fixture."""

    call()
    torch.cuda.synchronize()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        call()
    return sum("synchronizing CUDA operation" in str(entry.message) for entry in caught)


def test_packed_expert_execution_returns_fixed_length_group_metadata(
    sync_debug_mode,
) -> None:
    """Every expert is a group, so no routed row count reaches the host."""

    num_tokens, top_k, num_experts = 2048, 8, 256
    generator = torch.Generator(device="cuda").manual_seed(2468)
    top_k_index = torch.randint(
        0,
        num_experts,
        (num_tokens, top_k),
        generator=generator,
        device="cuda",
        dtype=torch.int64,
    )
    expert_indices, _ = torch.sort(top_k_index.reshape(-1))
    plan = SimpleNamespace(expert_indices=expert_indices)

    expert_ids, offsets, group_sizes = _prepare_packed_expert_execution(
        plan, num_experts
    )

    # Reference counts from the compacted form, scattered back onto every expert.
    unique_ids, unique_counts = torch.unique_consecutive(
        expert_indices, return_counts=True
    )
    expected_sizes = torch.zeros(
        num_experts, device="cuda", dtype=torch.int32
    ).index_copy(0, unique_ids, unique_counts.to(torch.int32))

    torch.testing.assert_close(
        expert_ids,
        torch.arange(num_experts, device="cuda", dtype=torch.int64),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(group_sizes, expected_sizes, rtol=0, atol=0)
    # Offsets are the prefix sums of the sizes they are returned with, not a
    # second derivation of them.
    torch.testing.assert_close(
        offsets, expected_sizes.cumsum(0, dtype=torch.int32), rtol=0, atol=0
    )
    assert offsets.dtype == torch.int32
    assert group_sizes.dtype == torch.int32
    assert expert_ids.is_contiguous()
    assert int(offsets[-1]) == expert_indices.numel()
    assert int(group_sizes.sum()) == expert_indices.numel()
    # `unique_consecutive` used to run here, sizing its output on the host and
    # stalling the pipeline once per MoE layer.
    assert (
        _count_synchronizing_calls(
            lambda: _prepare_packed_expert_execution(plan, num_experts)
        )
        == 0
    )


@pytest.mark.parametrize("projection", ["pair", "single"])
def test_fixed_group_layout_is_inert_for_unselected_experts(
    reader: gguf.GGUFReader,
    projection: str,
) -> None:
    """Empty groups must not change kernel results or gradients."""

    routes, num_experts = 16384, 256
    generator = torch.Generator(device="cuda").manual_seed(1357)
    # Three selected experts leave 253 empty groups in the fixed layout.
    expert_indices = (
        torch.randint(0, 3, (routes,), generator=generator, device="cuda").sort().values
    )
    plan = SimpleNamespace(expert_indices=expert_indices)

    fixed_ids, fixed_offsets, fixed_sizes = _prepare_packed_expert_execution(
        plan, num_experts
    )
    compact_ids, compact_counts = torch.unique_consecutive(
        expert_indices, return_counts=True
    )
    compact_sizes = compact_counts.to(torch.int32)
    compact_offsets = compact_sizes.cumsum(0, dtype=torch.int32)
    assert int(fixed_sizes.numel()) == num_experts
    assert int(compact_sizes.numel()) == 3

    results = {}
    if projection == "pair":
        gate = _packed_projection(reader, "gate", num_experts=256, out_features=512)
        up = _packed_projection(reader, "up", num_experts=256, out_features=512)
        hidden = torch.randn(
            routes, 2048, generator=generator, device="cuda", dtype=torch.bfloat16
        )
        cotangents = tuple(
            torch.randn(
                routes, 512, generator=generator, device="cuda", dtype=torch.bfloat16
            )
            for _ in range(2)
        )
    else:
        gate = _packed_projection(reader, "down", num_experts=256, out_features=2048)
        up = None
        hidden = torch.randn(
            routes, 512, generator=generator, device="cuda", dtype=torch.bfloat16
        )
        cotangents = (
            torch.randn(
                routes, 2048, generator=generator, device="cuda", dtype=torch.bfloat16
            ),
        )

    for name, (ids, offsets, sizes) in (
        ("fixed", (fixed_ids, fixed_offsets, fixed_sizes)),
        ("compacted", (compact_ids, compact_offsets, compact_sizes)),
    ):
        leaf = hidden.detach().clone().requires_grad_(True)
        if projection == "pair":
            assert up is not None
            first, second = _base_grouped_pair(
                leaf, gate, up, ids, offsets, sizes, torch.bfloat16
            )
            gradients = torch.autograd.grad((first, second), leaf, cotangents)
            results[name] = (first, second, *gradients)
        else:
            output = _base_grouped_linear(
                leaf, gate, ids, offsets, sizes, torch.bfloat16
            )
            (gradient,) = torch.autograd.grad(output, leaf, cotangents)
            results[name] = (output, gradient)

    assert all(
        torch.equal(fixed, compacted)
        for fixed, compacted in zip(results["fixed"], results["compacted"], strict=True)
    )


def test_full_group_lora_eliminates_selection_and_zeros_inactive_gradients() -> None:
    generator = torch.Generator(device="cuda").manual_seed(8642)
    active_experts = torch.tensor([1, 4, 7], device="cuda", dtype=torch.int64)
    active_sizes = torch.tensor([2, 3, 1], device="cuda", dtype=torch.int32)
    # Sorted route-level expert ids: expert 1 twice, 4 three times, 7 once. These
    # are the group sizes the production path hands to every grouped MM.
    routes = torch.tensor([1, 1, 4, 4, 4, 7], device="cuda", dtype=torch.int64)
    _, _, full_sizes = _prepare_packed_expert_execution(
        SimpleNamespace(expert_indices=routes), 8
    )
    torch.testing.assert_close(
        full_sizes,
        torch.tensor([0, 2, 0, 0, 3, 0, 0, 1], device="cuda", dtype=torch.int32),
        rtol=0,
        atol=0,
    )
    lhs = torch.randn(
        6,
        16,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    factor = torch.randn(
        8,
        4,
        16,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    reference_lhs = lhs.detach().clone().requires_grad_()
    reference_factor = factor.detach().clone().requires_grad_()
    grad_output = torch.randn(
        6,
        4,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )

    dispatched_ops: list[str] = []
    with _RecordOps(dispatched_ops):
        output = aiter_grouped_mm(
            lhs,
            factor.transpose(1, 2),
            full_sizes,
            expert_prior="qwen-learned",
        )
    reference = aiter_grouped_mm(
        reference_lhs,
        reference_factor.index_select(0, active_experts).transpose(1, 2),
        active_sizes,
        expert_prior="qwen-learned",
    )
    gradients = torch.autograd.grad(output, (lhs, factor), grad_output)
    reference_gradients = torch.autograd.grad(
        reference, (reference_lhs, reference_factor), grad_output
    )

    assert not any("index_select" in operation for operation in dispatched_ops)
    torch.testing.assert_close(output, reference, rtol=0, atol=0)
    for gradient, reference_gradient in zip(gradients, reference_gradients):
        torch.testing.assert_close(gradient, reference_gradient, rtol=0, atol=0)
    inactive = torch.ones(8, device="cuda", dtype=torch.bool)
    inactive[active_experts] = False
    assert torch.count_nonzero(gradients[1][inactive]) == 0

    checkpoint_lhs = lhs.detach().clone().requires_grad_()
    checkpoint_factor = factor.detach().clone().requires_grad_()
    checkpoint_output = checkpoint(
        lambda left, right: aiter_grouped_mm(
            left,
            right.transpose(1, 2),
            full_sizes,
            expert_prior="qwen-learned",
        ),
        checkpoint_lhs,
        checkpoint_factor,
        use_reentrant=False,
    )
    checkpoint_gradients = torch.autograd.grad(
        checkpoint_output, (checkpoint_lhs, checkpoint_factor), grad_output
    )
    assert torch.equal(checkpoint_output, output)
    for checkpoint_gradient, direct_gradient in zip(checkpoint_gradients, gradients):
        assert torch.equal(checkpoint_gradient, direct_gradient)


def test_aiter_config_dispatch_uses_rows_and_factor_layout(monkeypatch) -> None:
    gmm_keys = []
    ptgmm_keys = []

    def record_gmm_config(m, k, n, transposed_rhs, expert_prior):
        gmm_keys.append((m, k, n, transposed_rhs, expert_prior))
        return {}

    def record_ptgmm_config(m, k, n, expert_prior):
        ptgmm_keys.append((m, k, n, expert_prior))
        return {}

    def fake_gmm(lhs, rhs, group_sizes, **kwargs):
        output_n = rhs.shape[-1]
        return lhs.new_empty((lhs.shape[0], output_n))

    def fake_ptgmm(lhs, rhs, group_sizes, **kwargs):
        return rhs.new_empty((group_sizes.numel(), lhs.shape[-1], rhs.shape[-1]))

    monkeypatch.setattr(fast_moe_lora, "_gmm_config", record_gmm_config)
    monkeypatch.setattr(fast_moe_lora, "_ptgmm_config", record_ptgmm_config)
    monkeypatch.setattr(fast_moe_lora, "gmm", fake_gmm)
    monkeypatch.setattr(fast_moe_lora, "ptgmm", fake_ptgmm)

    lhs = torch.empty(6, 8)
    factor = torch.empty(3, 3, 8).transpose(1, 2)
    grad_output = torch.empty(6, 3)
    group_sizes = torch.tensor([2, 2, 2], dtype=torch.int32)
    fast_moe_lora._aiter_forward(lhs, factor, group_sizes, expert_prior="qwen-learned")
    fast_moe_lora._aiter_input_grad(
        grad_output, factor, group_sizes, expert_prior="qwen-learned"
    )
    fast_moe_lora._aiter_weight_grad(
        lhs, grad_output, group_sizes, expert_prior="qwen-learned"
    )

    assert gmm_keys == [
        (6, 8, 3, True, "qwen-learned"),
        (6, 3, 8, False, "qwen-learned"),
    ]
    assert ptgmm_keys == [(6, 8, 3, "qwen-learned")]


def test_aiter_grouped_mm_rejects_layout_repairs() -> None:
    lhs = torch.randn(16, 6, device="cuda", dtype=torch.bfloat16).T
    factor = torch.randn(3, 4, 16, device="cuda", dtype=torch.bfloat16)
    group_sizes = torch.tensor([2, 2, 2], device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="lhs must be row-major"):
        aiter_grouped_mm(
            lhs,
            factor.transpose(1, 2),
            group_sizes,
            expert_prior="qwen-learned",
        )

    valid_lhs = torch.randn(
        6, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    valid_factor = factor.detach().requires_grad_()
    output = aiter_grouped_mm(
        valid_lhs,
        valid_factor.transpose(1, 2),
        group_sizes,
        expert_prior="qwen-learned",
    )
    strided_gradient = torch.randn(
        output.shape[1], output.shape[0], device="cuda", dtype=torch.bfloat16
    ).T
    with pytest.raises(ValueError, match="output gradient must be row-major"):
        torch.autograd.grad(output, (valid_lhs, valid_factor), strided_gradient)


def test_aiter_grouped_mm_rebuilds_routed_rows_instead_of_retaining_them() -> None:
    """The routed rows are replayed from the routing index, not held."""

    generator = torch.Generator(device="cuda").manual_seed(13579)
    tokens, top_k, experts, hidden, rank = 2, 3, 8, 16, 4
    routes = tokens * top_k
    top_k_index = torch.randint(
        0,
        experts,
        (tokens, top_k),
        generator=generator,
        device="cuda",
        dtype=torch.int64,
    )
    expert_ids, permutation = torch.sort(top_k_index.reshape(-1))
    group_sizes = torch.bincount(expert_ids, minlength=experts).to(torch.int32)
    source = torch.randn(
        (tokens, hidden), generator=generator, device="cuda", dtype=torch.bfloat16
    )
    # The reconstruction contract `_lora_grouped_linear` relies on.
    routed_rows = source[permutation // top_k]

    def build_lhs() -> torch.Tensor:
        return routed_rows.detach().clone().requires_grad_(True)

    factor_values = torch.randn(
        (experts, rank, hidden),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )

    grad_output = torch.randn(
        (routes, rank), generator=generator, device="cuda", dtype=torch.bfloat16
    )

    retained: list[torch.Tensor] = []

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        retained.append(tensor)
        return tensor

    rebuild_lhs = build_lhs()
    rebuild_source = source.detach().clone().requires_grad_(True)
    rebuild_factor = factor_values.detach().clone().requires_grad_(True)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        rebuilt_output = aiter_grouped_mm(
            rebuild_lhs,
            rebuild_factor.transpose(1, 2),
            group_sizes,
            expert_prior="qwen-learned",
            lhs_source=rebuild_source,
            lhs_permutation=permutation,
            lhs_top_k=top_k,
        )
    assert not any(tensor is rebuild_lhs for tensor in retained)
    assert any(tensor is rebuild_source for tensor in retained)
    assert any(tensor is permutation for tensor in retained)

    direct_lhs = build_lhs()
    direct_factor = factor_values.detach().clone().requires_grad_(True)
    direct_output = aiter_grouped_mm(
        direct_lhs,
        direct_factor.transpose(1, 2),
        group_sizes,
        expert_prior="qwen-learned",
    )

    assert torch.equal(rebuilt_output, direct_output)
    rebuilt_gradients = torch.autograd.grad(
        rebuilt_output, (rebuild_lhs, rebuild_factor), grad_output
    )
    direct_gradients = torch.autograd.grad(
        direct_output, (direct_lhs, direct_factor), grad_output
    )
    for rebuilt_gradient, direct_gradient in zip(rebuilt_gradients, direct_gradients):
        assert torch.equal(rebuilt_gradient, direct_gradient)


def test_aiter_grouped_mm_rejects_incomplete_row_rebuilds() -> None:
    lhs = torch.randn(6, 16, device="cuda", dtype=torch.bfloat16)
    factor = torch.randn(8, 4, 16, device="cuda", dtype=torch.bfloat16)
    group_sizes = torch.tensor(
        [0, 2, 0, 0, 3, 0, 0, 1], device="cuda", dtype=torch.int32
    )
    source = lhs.detach().clone()
    permutation = torch.zeros(6, device="cuda", dtype=torch.int64)

    with pytest.raises(ValueError, match="requires the routing permutation"):
        aiter_grouped_mm(
            lhs,
            factor.transpose(1, 2),
            group_sizes,
            expert_prior="qwen-learned",
            lhs_source=source,
        )
    with pytest.raises(ValueError, match="requires top_k >= 1"):
        aiter_grouped_mm(
            lhs,
            factor.transpose(1, 2),
            group_sizes,
            expert_prior="qwen-learned",
            lhs_source=source,
            lhs_permutation=permutation,
            lhs_top_k=0,
        )
    with pytest.raises(ValueError, match="row-major source"):
        aiter_grouped_mm(
            lhs,
            factor.transpose(1, 2),
            group_sizes,
            expert_prior="qwen-learned",
            lhs_source=source.T,
            lhs_permutation=permutation,
            lhs_top_k=1,
        )
    with pytest.raises(ValueError, match="one row index per lhs row"):
        aiter_grouped_mm(
            lhs,
            factor.transpose(1, 2),
            group_sizes,
            expert_prior="qwen-learned",
            lhs_source=source,
            lhs_permutation=permutation[:5],
            lhs_top_k=1,
        )


def test_one_expert_layer_has_finite_lora_gradients_and_no_packed_gradients(
    reader: gguf.GGUFReader,
    monkeypatch,
) -> None:
    config = SimpleNamespace(
        num_experts=256,
        hidden_size=2048,
        moe_intermediate_size=512,
        hidden_act="silu",
        _experts_implementation=QWEN3_5_MOE_EXPERTS_IMPLEMENTATION,
    )
    experts = GgufExperts(config, device="meta", compute_dtype=torch.bfloat16)
    experts.config = config
    experts.gate_proj = _packed_projection(
        reader, "gate", num_experts=256, out_features=512
    )
    experts.up_proj = _packed_projection(
        reader, "up", num_experts=256, out_features=512
    )
    experts.down_proj = _packed_projection(
        reader, "down", num_experts=256, out_features=2048
    )
    ALL_GGUF_EXPERTS_FUNCTIONS[QWEN3_5_MOE_EXPERTS_IMPLEMENTATION] = (
        gguf_mmq_aiter_lora_forward
    )

    lora_config = LoraConfig(
        target_modules=["experts"],
        r=4,
        lora_alpha=4,
        lora_dropout=0.0,
        bias="none",
    )
    experts.__dict__["_aiter_expert_prior"] = "qwen-learned"
    layer = FastGgufMoeLora(
        experts,
        "default",
        config=lora_config,
        r=4,
        lora_alpha=4,
    )
    generator = torch.Generator(device="cuda").manual_seed(97531)
    with torch.no_grad():
        for name, parameter in layer.named_parameters():
            if "lora_B" in name:
                parameter.normal_(generator=generator, std=0.01)

    hidden = torch.randn(
        2048,
        2048,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    top_k_index = (
        torch.randn(
            2048,
            256,
            generator=generator,
            device="cuda",
        )
        .topk(8, dim=-1)
        .indices
    )
    top_k_weights = torch.softmax(
        torch.randn(
            2048,
            8,
            generator=generator,
            device="cuda",
            dtype=torch.float32,
        ),
        dim=-1,
    ).requires_grad_(True)
    grad_output = torch.randn(
        2048,
        2048,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )

    # Each LoRA delta has to be folded into the base projection output it belongs
    # to instead of allocating a full routed-size replacement, so the activations
    # the gate consumes must be the very objects the base projections produced.
    base_outputs = []
    gate_inputs = []
    real_base_pair = fast_moe_lora._base_grouped_pair
    real_split_gate = GgufExperts._apply_split_gate

    def recording_base_pair(*args, **kwargs):
        outputs = real_base_pair(*args, **kwargs)
        base_outputs.append(outputs)
        return outputs

    def recording_split_gate(gate, up):
        gate_inputs.append((gate, up))
        # nn.Module's dynamic attribute typing hides the real bound method.
        return real_split_gate(experts, gate, up)  # ty: ignore[call-non-callable]

    monkeypatch.setattr(fast_moe_lora, "_base_grouped_pair", recording_base_pair)
    experts.__dict__["_apply_split_gate"] = recording_split_gate

    dispatched_ops: list[str] = []
    with _RecordOps(dispatched_ops):
        output = layer(hidden, top_k_index, top_k_weights)
        output.backward(grad_output)

    trainable_gradients = []
    for parameter in layer.parameters():
        if parameter.requires_grad:
            if parameter.grad is None:
                raise AssertionError("expected a trainable parameter gradient")
            trainable_gradients.append(parameter.grad)
    assert output.shape == hidden.shape
    assert torch.isfinite(output).all()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert top_k_weights.grad is not None and torch.isfinite(top_k_weights.grad).all()
    assert torch.count_nonzero(top_k_weights.grad) > 0
    assert len(trainable_gradients) == 4
    assert all(torch.isfinite(gradient).all() for gradient in trainable_gradients)
    assert all(torch.count_nonzero(gradient) > 0 for gradient in trainable_gradients)
    active_experts = torch.unique(top_k_index)
    inactive = torch.ones(config.num_experts, device="cuda", dtype=torch.bool)
    inactive[active_experts] = False
    assert all(
        torch.count_nonzero(gradient[inactive]) == 0 for gradient in trainable_gradients
    )
    assert not any("index_select" in operation for operation in dispatched_ops)
    assert all(parameter.grad is None for parameter in experts.parameters())

    assert len(base_outputs) == 1
    assert len(gate_inputs) == 1
    assert gate_inputs[0][0] is base_outputs[0][0]
    assert gate_inputs[0][1] is base_outputs[0][1]
