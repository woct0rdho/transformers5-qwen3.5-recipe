from types import SimpleNamespace

import gguf
import pytest
import torch
from peft import LoraConfig
from torch.utils._python_dispatch import TorchDispatchMode
from transformers.integrations.gguf import (
    ALL_GGUF_EXPERTS_FUNCTIONS,
    DeepseekV4GGUFExperts,
)
from transformers.integrations.gguf_dequant import (
    GGUFQuantizedTensor,
    dequantize_gguf_tensor,
)

import fast_moe_lora
from deepseek_v4_moe_lora import (
    EXPERTS_IMPLEMENTATION,
    DeepseekV4GGUFMoeLora,
    deepseek_v4_gguf_dequant_aiter_lora_forward,
)
from fast_moe_lora import (
    _aiter_forward,
    _base_grouped_linear,
    _group_sizes_from_offsets,
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


class _RecordOps(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.operations: list[str] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.operations.append(str(func))
        return func(*args, **(kwargs or {}))


def _packed_experts(
    qtype: gguf.GGMLQuantizationType, value: int
) -> GGUFQuantizedTensor:
    type_size = gguf.GGML_QUANT_SIZES[qtype][1]
    payload = torch.full((2, 256, type_size), value, device="cuda", dtype=torch.uint8)
    return GGUFQuantizedTensor(
        payload,
        quant_type=qtype,
        logical_shape=(2, 256, 256),
    )


def test_iq2_xxs_and_q2_k_use_native_grouped_mmq_backward() -> None:
    experts = torch.tensor([0, 1], device="cuda", dtype=torch.long)
    offsets = torch.tensor([3, 7], device="cuda", dtype=torch.int32)
    group_sizes = _group_sizes_from_offsets(offsets)
    for qtype, value in (
        (gguf.GGMLQuantizationType.IQ2_XXS, 32),
        (gguf.GGMLQuantizationType.Q2_K, 32),
    ):
        weight = _packed_experts(qtype, value)
        hidden = torch.randn(
            7, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        grad_output = torch.randn(7, 256, device="cuda", dtype=torch.bfloat16)
        recorder = _RecordOps()
        with recorder:
            output = _base_grouped_linear(
                hidden,
                weight,
                experts,
                offsets,
                group_sizes,
                torch.bfloat16,
            )
            output.backward(grad_output)
        assert any(
            "torch_ggml_ops.grouped_mmq.default" in operation
            for operation in recorder.operations
        )
        assert any(
            "torch_ggml_ops.grouped_mmq_grad_input.default" in operation
            for operation in recorder.operations
        )

        logical = dequantize_gguf_tensor(
            weight.as_subclass(torch.Tensor).index_select(0, experts),
            qtype,
            dtype=torch.bfloat16,
            device="cuda",
        )
        expected_output = _aiter_forward(
            hidden.detach(), logical.transpose(1, 2), group_sizes
        )
        expected_grad = torch.empty_like(hidden)
        row_begin = 0
        for group, row_end in enumerate(offsets.cpu().tolist()):
            expected_grad[row_begin:row_end] = (
                grad_output[row_begin:row_end].float() @ logical[group].float()
            ).to(torch.bfloat16)
            row_begin = row_end
        torch.testing.assert_close(output, expected_output, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(hidden.grad, expected_grad, rtol=1e-2, atol=1e-2)
        assert weight.grad is None


def test_complete_deepseek_expert_lora_preserves_clamp_and_has_finite_gradients() -> (
    None
):
    config = SimpleNamespace(
        num_experts=2,
        hidden_size=256,
        moe_intermediate_size=256,
        hidden_act="silu",
        swiglu_limit=0.05,
        _experts_implementation=EXPERTS_IMPLEMENTATION,
    )
    experts = DeepseekV4GGUFExperts(
        config,
        device="meta",
        compute_dtype=torch.bfloat16,
    )
    experts.config = config
    experts.gate_proj = _packed_experts(gguf.GGMLQuantizationType.IQ2_XXS, 32)
    experts.up_proj = _packed_experts(gguf.GGMLQuantizationType.IQ2_XXS, 32)
    experts.down_proj = _packed_experts(gguf.GGMLQuantizationType.Q2_K, 32)
    ALL_GGUF_EXPERTS_FUNCTIONS[EXPERTS_IMPLEMENTATION] = (
        deepseek_v4_gguf_dequant_aiter_lora_forward
    )

    lora_config = LoraConfig(
        target_modules=["experts"],
        r=4,
        lora_alpha=4,
        lora_dropout=0.0,
        bias="none",
    )
    layer = DeepseekV4GGUFMoeLora(
        experts,
        "default",
        config=lora_config,
        r=4,
        lora_alpha=4,
    )
    with torch.no_grad():
        for name, parameter in layer.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=0.01)

    clamp_inputs: list[tuple[torch.Tensor, torch.Tensor]] = []
    original_apply = experts._apply_split_gate

    def tracked_apply(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        clamp_inputs.append((gate.detach(), up.detach()))
        return original_apply(gate, up)

    experts.__dict__["_apply_split_gate"] = tracked_apply
    hidden = torch.randn(
        6, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    top_k_index = torch.tensor([[0], [1], [0], [1], [0], [1]], device="cuda")
    top_k_weights = torch.ones(
        6, 1, device="cuda", dtype=torch.float32, requires_grad=True
    )
    output = layer(hidden, top_k_index, top_k_weights)
    output.float().square().mean().backward()

    assert clamp_inputs
    gate, up = clamp_inputs[0]
    probe_gate = torch.tensor(
        [[config.swiglu_limit * 4, -config.swiglu_limit * 4]],
        device="cuda",
        dtype=torch.bfloat16,
    )
    probe_up = torch.tensor(
        [[config.swiglu_limit * 4, -config.swiglu_limit * 4]],
        device="cuda",
        dtype=torch.bfloat16,
    )
    expected_probe = experts.act_fn(
        probe_gate.clamp(max=config.swiglu_limit)
    ) * probe_up.clamp(min=-config.swiglu_limit, max=config.swiglu_limit)
    torch.testing.assert_close(
        original_apply(probe_gate, probe_up), expected_probe, rtol=0, atol=0
    )
    assert gate.shape == up.shape
    assert output.shape == hidden.shape and torch.isfinite(output).all()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert top_k_weights.grad is not None and torch.isfinite(top_k_weights.grad).all()
    trainable_gradients = []
    for parameter in layer.parameters():
        if parameter.requires_grad:
            if parameter.grad is None:
                raise AssertionError("expected a trainable parameter gradient")
            trainable_gradients.append(parameter.grad)
    assert len(trainable_gradients) == 4
    assert all(torch.isfinite(gradient).all() for gradient in trainable_gradients)
    assert all(torch.count_nonzero(gradient) > 0 for gradient in trainable_gradients)
    assert all(parameter.grad is None for parameter in experts.parameters())
