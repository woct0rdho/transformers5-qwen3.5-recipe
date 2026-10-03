import os
from pathlib import Path
from types import SimpleNamespace

import gguf
import numpy as np
import pytest
import torch
from peft import LoraConfig
from torch.utils._python_dispatch import TorchDispatchMode
from transformers.integrations.gguf.gguf_quantized_parameter import (
    GgufQuantizedParameter,
)
from transformers.integrations.gguf.moe import (
    ALL_GGUF_EXPERTS_FUNCTIONS,
    DeepseekV4GgufExperts,
)

import fast_moe_lora
from deepseek_v4_moe_lora import (
    DEEPSEEK_V4_EXPERTS_IMPLEMENTATION,
    DeepseekV4GgufMoeLora,
    _bind_deepseek_expert_priors,
    deepseek_v4_gguf_mmq_aiter_lora_forward,
)
from fast_moe_lora import (
    _base_grouped_linear,
    _base_grouped_pair,
)
from gguf_support import dequantize_gguf_tensor


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


class _RecordOps(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.operations: list[str] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.operations.append(str(func))
        return func(*args, **(kwargs or {}))


_MODEL = Path(
    os.environ.get(
        "GGUF_DEEPSEEK_MMQ_TEST_MODEL",
        os.path.expanduser("~/models/ds4/DeepSeek-V4-Flash-IQ2XXS.gguf"),
    )
)


class _BindingToyBlock(torch.nn.Module):
    def __init__(self, is_hash: bool) -> None:
        super().__init__()
        self.is_hash = is_hash
        config = SimpleNamespace(
            num_experts=2,
            hidden_size=2,
            moe_intermediate_size=2,
            hidden_act="silu",
            swiglu_limit=1.0,
            _experts_implementation="eager",
        )
        self.experts = DeepseekV4GgufExperts(
            config,
            device="cpu",
            compute_dtype=torch.bfloat16,
        )
        self.experts.config = config


class _BindingToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [_BindingToyBlock(is_hash=True), _BindingToyBlock(is_hash=False)]
        )


def test_deepseek_prior_binding_follows_each_moe_router() -> None:
    model = _BindingToyModel()
    counts = _bind_deepseek_expert_priors(model, "deepseek-learned")

    assert counts == {"deepseek-learned": 1, "deepseek-hash": 1}
    first_experts = model.layers[0].experts
    second_experts = model.layers[1].experts
    assert first_experts.__dict__["_aiter_expert_prior"] == "deepseek-hash"
    assert second_experts.__dict__["_aiter_expert_prior"] == "deepseek-learned"


@pytest.fixture(scope="module")
def reader() -> gguf.GGUFReader:
    if not _MODEL.is_file():
        pytest.skip("DeepSeek GGUF model is unavailable")
    return gguf.GGUFReader(_MODEL)


def _packed_experts(reader: gguf.GGUFReader, projection: str) -> GgufQuantizedParameter:
    tensor = next(
        item
        for item in reader.tensors
        if item.name == f"blk.0.ffn_{projection}_exps.weight"
    )
    payload = torch.from_numpy(
        np.array(tensor.data, dtype=np.uint8, copy=True, order="C")
    ).to("cuda")
    return GgufQuantizedParameter(
        payload,
        quant_type=tensor.tensor_type,
        logical_shape=tuple(int(size) for size in reversed(tensor.shape)),
    )


def test_iq2_xxs_and_q2_k_use_native_grouped_mmq_backward(
    reader: gguf.GGUFReader,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(2468)
    experts = torch.tensor([0, 1], device="cuda", dtype=torch.int64)
    offsets = torch.tensor([4096, 12288], device="cuda", dtype=torch.int32)
    group_sizes = torch.tensor([4096, 8192], device="cuda", dtype=torch.int32)
    probe_rows = (0, 4096)

    gate = _packed_experts(reader, "gate")
    up = _packed_experts(reader, "up")
    hidden = torch.zeros(
        12288, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    gate_grad = torch.zeros(12288, 2048, device="cuda", dtype=torch.bfloat16)
    up_grad = torch.zeros_like(gate_grad)
    with torch.no_grad():
        for row in probe_rows:
            hidden[row].normal_(generator=generator)
            gate_grad[row].normal_(generator=generator)
            up_grad[row].normal_(generator=generator)

    recorder = _RecordOps()
    with recorder:
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
    assert "torch_ggml_ops._grouped_mmq_pair_launch.default" in recorder.operations
    assert (
        "torch_ggml_ops._grouped_mmq_pair_grad_input_launch.default"
        in recorder.operations
    )

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
    expected_grad = torch.zeros_like(hidden)
    for group, row in enumerate(probe_rows):
        torch.testing.assert_close(
            gate_output[row],
            torch.nn.functional.linear(hidden.detach()[row], logical_gate[group]),
            rtol=2e-2,
            atol=4e-2,
        )
        torch.testing.assert_close(
            up_output[row],
            torch.nn.functional.linear(hidden.detach()[row], logical_up[group]),
            rtol=2e-2,
            atol=4e-2,
        )
        expected_grad[row] = (
            gate_grad[row].float() @ logical_gate[group].float()
            + up_grad[row].float() @ logical_up[group].float()
        ).to(torch.bfloat16)
    torch.testing.assert_close(hidden.grad, expected_grad, rtol=1e-2, atol=2e-2)
    assert gate.grad is None and up.grad is None

    down = _packed_experts(reader, "down")
    intermediate = torch.zeros(
        12288, 2048, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    down_grad = torch.zeros(12288, 4096, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        for row in probe_rows:
            intermediate[row].normal_(generator=generator)
            down_grad[row].normal_(generator=generator)
    recorder = _RecordOps()
    with recorder:
        down_output = _base_grouped_linear(
            intermediate,
            down,
            experts,
            offsets,
            group_sizes,
            torch.bfloat16,
        )
        down_output.backward(down_grad)
    assert "torch_ggml_ops._grouped_mmq_launch.default" in recorder.operations
    assert (
        "torch_ggml_ops._grouped_mmq_grad_input_launch.default" in recorder.operations
    )

    logical_down = dequantize_gguf_tensor(
        down.as_subclass(torch.Tensor).index_select(0, experts),
        down.quant_type,
        dtype=torch.bfloat16,
        device="cuda",
    )
    expected_grad = torch.zeros_like(intermediate)
    for group, row in enumerate(probe_rows):
        torch.testing.assert_close(
            down_output[row],
            torch.nn.functional.linear(intermediate.detach()[row], logical_down[group]),
            rtol=2e-2,
            atol=5e-2,
        )
        expected_grad[row] = (down_grad[row].float() @ logical_down[group].float()).to(
            torch.bfloat16
        )
    torch.testing.assert_close(intermediate.grad, expected_grad, rtol=1e-2, atol=2e-2)
    assert down.grad is None


def test_complete_deepseek_expert_lora_preserves_clamp_and_has_finite_gradients(
    reader: gguf.GGUFReader,
) -> None:
    config = SimpleNamespace(
        num_experts=256,
        hidden_size=4096,
        moe_intermediate_size=2048,
        hidden_act="silu",
        swiglu_limit=0.05,
        _experts_implementation=DEEPSEEK_V4_EXPERTS_IMPLEMENTATION,
    )
    experts = DeepseekV4GgufExperts(
        config,
        device="meta",
        compute_dtype=torch.bfloat16,
    )
    experts.config = config
    experts.gate_proj = _packed_experts(reader, "gate")
    experts.up_proj = _packed_experts(reader, "up")
    experts.down_proj = _packed_experts(reader, "down")
    ALL_GGUF_EXPERTS_FUNCTIONS[DEEPSEEK_V4_EXPERTS_IMPLEMENTATION] = (
        deepseek_v4_gguf_mmq_aiter_lora_forward
    )

    lora_config = LoraConfig(
        target_modules=["experts"],
        r=4,
        lora_alpha=4,
        lora_dropout=0.0,
        bias="none",
    )
    experts.__dict__["_aiter_expert_prior"] = "deepseek-learned"
    layer = DeepseekV4GgufMoeLora(
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
        2048, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    top_k_index = torch.randn(2048, 256, device="cuda").topk(6, dim=-1).indices
    top_k_weights = torch.softmax(
        torch.randn(2048, 6, device="cuda", dtype=torch.float32), dim=-1
    ).requires_grad_(True)
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
