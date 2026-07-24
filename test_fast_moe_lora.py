import os
from pathlib import Path
from types import SimpleNamespace

import gguf
import numpy as np
import pytest
import torch
from peft import LoraConfig
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.checkpoint import checkpoint
from transformers.integrations.gguf import ALL_GGUF_EXPERTS_FUNCTIONS, GGUFExperts
from transformers.integrations.gguf_dequant import (
    GGUFQuantizedTensor,
    dequantize_gguf_tensor,
)

import fast_moe_lora
from fast_moe_lora import (
    EXPERTS_IMPLEMENTATION,
    FastGGUFMoeLora,
    _aiter_input_grad,
    _base_grouped_linear,
    _base_grouped_pair,
    _complete_group_sizes,
    _group_sizes_from_offsets,
    aiter_grouped_mm,
    qwen3_5_moe_gguf_mmq_aiter_lora_forward,
)


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
    monkeypatch.setattr(fast_moe_lora, "_gmm_config", lambda *_: dict(config))
    monkeypatch.setattr(fast_moe_lora, "_ptgmm_config", lambda *_: dict(config))


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
) -> GGUFQuantizedTensor:
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
    return GGUFQuantizedTensor(
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
    offsets = torch.tensor([2, 5, 6], device="cuda", dtype=torch.int32)
    group_sizes = _group_sizes_from_offsets(offsets)
    generator = torch.Generator(device="cuda").manual_seed(2468)

    gate = _packed_projection(reader, "gate", num_experts=8, out_features=64)
    up = _packed_projection(reader, "up", num_experts=8, out_features=64)
    hidden = torch.randn(
        6,
        2048,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    gate_grad = torch.randn(
        6,
        64,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    up_grad = torch.randn(
        6,
        64,
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
    assert "torch_ggml_ops.grouped_mmq_pair.default" in pair_ops
    assert "torch_ggml_ops.grouped_mmq_pair_grad_input.default" in pair_ops

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
    torch.testing.assert_close(hidden.grad, expected_hidden_grad, rtol=0, atol=1e-4)
    assert gate.grad is None
    assert up.grad is None

    q3_gate = _packed_projection(
        reader, "gate", num_experts=8, out_features=64, layer=0
    )
    mixed_gate, mixed_up = _base_grouped_pair(
        hidden.detach(),
        q3_gate,
        up,
        experts,
        offsets,
        group_sizes,
        torch.bfloat16,
    )
    separate_gate = _base_grouped_linear(
        hidden.detach(),
        q3_gate,
        experts,
        offsets,
        group_sizes,
        torch.bfloat16,
    )
    separate_up = _base_grouped_linear(
        hidden.detach(),
        up,
        experts,
        offsets,
        group_sizes,
        torch.bfloat16,
    )
    torch.testing.assert_close(mixed_gate, separate_gate, rtol=0, atol=0)
    torch.testing.assert_close(mixed_up, separate_up, rtol=0, atol=0)

    down = _packed_projection(reader, "down", num_experts=8, out_features=64)
    intermediate = torch.randn(
        6,
        512,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    down_grad = torch.randn(
        6,
        64,
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
    assert "torch_ggml_ops.grouped_mmq.default" in down_ops
    assert "torch_ggml_ops.grouped_mmq_grad_input.default" in down_ops
    logical_down = dequantize_gguf_tensor(
        down.as_subclass(torch.Tensor).index_select(0, experts),
        down.quant_type,
        dtype=torch.bfloat16,
        device="cuda",
    )
    expected_intermediate_grad = _aiter_input_grad(
        down_grad, logical_down.transpose(1, 2), group_sizes
    )
    torch.testing.assert_close(
        intermediate.grad, expected_intermediate_grad, rtol=0, atol=0
    )
    assert down.grad is None


def test_full_group_lora_eliminates_selection_and_zeros_inactive_gradients() -> None:
    generator = torch.Generator(device="cuda").manual_seed(8642)
    active_experts = torch.tensor([1, 4, 7], device="cuda", dtype=torch.int64)
    active_sizes = torch.tensor([2, 3, 1], device="cuda", dtype=torch.int32)
    full_sizes = _complete_group_sizes(active_sizes, active_experts, 8)
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
        output = aiter_grouped_mm(lhs, factor.transpose(1, 2), full_sizes)
    reference = aiter_grouped_mm(
        reference_lhs,
        reference_factor.index_select(0, active_experts).transpose(1, 2),
        active_sizes,
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
        lambda left, right: aiter_grouped_mm(left, right.transpose(1, 2), full_sizes),
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

    def record_gmm_config(m, k, n, transposed_rhs):
        gmm_keys.append((m, k, n, transposed_rhs))
        return {}

    def record_ptgmm_config(m, k, n):
        ptgmm_keys.append((m, k, n))
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
    fast_moe_lora._aiter_forward(lhs, factor, group_sizes)
    fast_moe_lora._aiter_input_grad(grad_output, factor, group_sizes)
    fast_moe_lora._aiter_weight_grad(lhs, grad_output, group_sizes)

    assert gmm_keys == [(6, 8, 3, True), (6, 3, 8, False)]
    assert ptgmm_keys == [(6, 8, 3)]


def test_aiter_grouped_mm_rejects_layout_repairs() -> None:
    lhs = torch.randn(16, 6, device="cuda", dtype=torch.bfloat16).T
    factor = torch.randn(3, 4, 16, device="cuda", dtype=torch.bfloat16)
    group_sizes = torch.tensor([2, 2, 2], device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="lhs must be row-major"):
        aiter_grouped_mm(lhs, factor.transpose(1, 2), group_sizes)

    valid_lhs = torch.randn(
        6, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    valid_factor = factor.detach().requires_grad_()
    output = aiter_grouped_mm(valid_lhs, valid_factor.transpose(1, 2), group_sizes)
    strided_gradient = torch.randn(
        output.shape[1], output.shape[0], device="cuda", dtype=torch.bfloat16
    ).T
    with pytest.raises(ValueError, match="output gradient must be row-major"):
        torch.autograd.grad(output, (valid_lhs, valid_factor), strided_gradient)


def test_one_expert_layer_has_finite_lora_gradients_and_no_packed_gradients(
    reader: gguf.GGUFReader,
) -> None:
    config = SimpleNamespace(
        num_experts=8,
        hidden_size=2048,
        moe_intermediate_size=512,
        hidden_act="silu",
        _experts_implementation=EXPERTS_IMPLEMENTATION,
    )
    experts = GGUFExperts(config, device="meta", compute_dtype=torch.bfloat16)
    experts.config = config
    experts.gate_proj = _packed_projection(
        reader, "gate", num_experts=8, out_features=512
    )
    experts.up_proj = _packed_projection(reader, "up", num_experts=8, out_features=512)
    experts.down_proj = _packed_projection(
        reader, "down", num_experts=8, out_features=2048
    )
    ALL_GGUF_EXPERTS_FUNCTIONS[EXPERTS_IMPLEMENTATION] = (
        qwen3_5_moe_gguf_mmq_aiter_lora_forward
    )

    lora_config = LoraConfig(
        target_modules=["experts"],
        r=4,
        lora_alpha=4,
        lora_dropout=0.0,
        bias="none",
    )
    layer = FastGGUFMoeLora(
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
        8,
        2048,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    top_k_index = (
        torch.randn(
            8,
            8,
            generator=generator,
            device="cuda",
        )
        .topk(4, dim=-1)
        .indices
    )
    top_k_weights = torch.softmax(
        torch.randn(
            8,
            4,
            generator=generator,
            device="cuda",
            dtype=torch.float32,
        ),
        dim=-1,
    ).requires_grad_(True)
    grad_output = torch.randn(
        8,
        2048,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )

    dispatched_ops: list[str] = []
    with _RecordOps(dispatched_ops):
        output = layer(hidden, top_k_index, top_k_weights)
        output.backward(grad_output)

    trainable_gradients = [
        parameter.grad for parameter in layer.parameters() if parameter.requires_grad
    ]
    assert output.shape == hidden.shape
    assert torch.isfinite(output).all()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert top_k_weights.grad is not None and torch.isfinite(top_k_weights.grad).all()
    assert torch.count_nonzero(top_k_weights.grad) > 0
    assert len(trainable_gradients) == 4
    assert all(gradient is not None for gradient in trainable_gradients)
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
