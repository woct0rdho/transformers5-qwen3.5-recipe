import os
from pathlib import Path

import gguf
import numpy as np
import pytest
import torch
import torch_ggml_ops
from peft import LoraConfig, get_peft_model
from torch.utils._python_dispatch import TorchDispatchMode
from transformers.integrations.gguf.gguf_quantized_parameter import (
    GgufQuantizedParameter,
)
from transformers.integrations.gguf.modules import GgufLinear

from fast_lora import FastGgufLoraLinear, register_fast_lora
from gguf_support import dequantize_gguf_tensor
from test_support import require_grad

_MODEL = Path(
    os.environ.get(
        "GGUF_MMQ_TEST_MODEL",
        os.path.expanduser("~/models/qwen3.6/Qwen3.6-35B-A3B-APEX-I-Mini.gguf"),
    )
)


def test_fast_lora_keeps_original_bf16_input_and_exact_base_jacobian() -> None:
    if not _MODEL.is_file():
        pytest.skip("GGUF model is unavailable")

    reader = gguf.GGUFReader(_MODEL)
    tensor = next(t for t in reader.tensors if t.name == "blk.0.attn_gate.weight")
    out_features = 512
    payload = torch.from_numpy(
        np.array(tensor.data[:out_features], dtype=np.uint8, copy=True, order="C")
    ).to("cuda")
    packed = GgufQuantizedParameter(
        payload,
        quant_type=tensor.tensor_type,
        logical_shape=(out_features, 2048),
    )

    class Toy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = GgufLinear(
                2048,
                out_features,
                bias=False,
                device="cuda",
                dtype=torch.bfloat16,
                compute_dtype=torch.bfloat16,
            )
            self.proj.weight = packed

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.proj(input)

    config = LoraConfig(
        target_modules=["proj"],
        r=4,
        lora_alpha=4,
        lora_dropout=0.0,
        bias="none",
    )
    toy = Toy()
    register_fast_lora(config, toy)
    model = get_peft_model(toy, config, autocast_adapter_dtype=False)
    wrapped = model.base_model.model
    assert isinstance(wrapped, Toy)
    layer = wrapped.proj
    assert isinstance(layer, FastGgufLoraLinear)
    lora_a = layer.lora_A["default"]
    lora_b = layer.lora_B["default"]
    assert isinstance(lora_a, torch.nn.Linear)
    assert isinstance(lora_b, torch.nn.Linear)
    assert lora_a.weight.dtype == torch.bfloat16
    assert lora_b.weight.dtype == torch.bfloat16

    generator = torch.Generator(device="cuda").manual_seed(2468)
    with torch.no_grad():
        lora_b.weight.normal_(generator=generator, std=0.02)
    input = torch.randn(
        1,
        2048,
        2048,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    grad_output = torch.randn(
        1,
        2048,
        out_features,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )

    dispatched_ops: list[str] = []

    class _RecordOps(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            dispatched_ops.append(str(func))
            return func(*args, **(kwargs or {}))

    with _RecordOps():
        actual = model(input)
        actual.backward(grad_output)
    assert "torch_ggml_ops._mmq_launch.default" in dispatched_ops
    assert "torch_ggml_ops._mmq_grad_input_launch.default" in dispatched_ops
    actual_input_grad = require_grad(input).detach().clone()
    actual_a_grad = require_grad(lora_a.weight).detach().clone()
    actual_b_grad = require_grad(lora_b.weight).detach().clone()

    logical_weight = dequantize_gguf_tensor(
        payload,
        tensor.tensor_type,
        dtype=torch.bfloat16,
        device="cuda",
    ).reshape(out_features, 2048)
    input_ref = input.detach().clone().requires_grad_(True)
    a_ref = lora_a.weight.detach().clone().requires_grad_(True)
    b_ref = lora_b.weight.detach().clone().requires_grad_(True)
    base_ref = torch.nn.functional.linear(input_ref, logical_weight)
    hidden_ref = torch.matmul(input_ref, a_ref.transpose(0, 1))
    output_ref = torch.addmm(
        base_ref.reshape(-1, out_features),
        hidden_ref.reshape(-1, 4),
        b_ref.transpose(0, 1),
        beta=1,
        alpha=layer.scaling["default"],
    ).reshape_as(actual)
    output_ref.backward(grad_output)

    # GGTensile accumulates the packed Jacobian directly and may differ from
    # dequantize-then-GEMM by one BF16 rounding step.
    torch.testing.assert_close(
        actual_input_grad, require_grad(input_ref), rtol=0, atol=8e-3
    )
    torch.testing.assert_close(actual_a_grad, require_grad(a_ref), rtol=0, atol=0)
    torch.testing.assert_close(actual_b_grad, require_grad(b_ref), rtol=0, atol=0)
    assert layer.base_layer.weight.grad is None

    mmq_base = torch_ggml_ops.mmq(
        input.detach(), payload, int(tensor.tensor_type), out_features
    )
    expected_actual = torch.addmm(
        mmq_base.reshape(-1, out_features),
        torch.matmul(input.detach(), a_ref.detach().transpose(0, 1)).reshape(-1, 4),
        b_ref.detach().transpose(0, 1),
        beta=1,
        alpha=layer.scaling["default"],
    ).reshape_as(actual)
    torch.testing.assert_close(actual, expected_actual, rtol=0, atol=0)


def test_fast_lora_gdn_projection_uses_generic_dequant_forward() -> None:
    if not _MODEL.is_file():
        pytest.skip("GGUF model is unavailable")

    reader = gguf.GGUFReader(_MODEL)
    tensor = next(t for t in reader.tensors if t.name == "blk.0.attn_gate.weight")
    out_features = 512
    payload = torch.from_numpy(
        np.array(tensor.data[:out_features], dtype=np.uint8, copy=True, order="C")
    ).to("cuda")
    packed = GgufQuantizedParameter(
        payload,
        quant_type=tensor.tensor_type,
        logical_shape=(out_features, 2048),
    )

    class LinearAttention(torch.nn.Module):
        # Named like the GatedDeltaNet projections that must stay on the generic packed
        # base forward, whose packed row reorder leaves no runtime permutation.
        def __init__(self) -> None:
            super().__init__()
            self.in_proj_z = GgufLinear(
                2048,
                out_features,
                bias=False,
                device="cuda",
                dtype=torch.bfloat16,
                compute_dtype=torch.bfloat16,
            )
            self.in_proj_z.weight = packed

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.in_proj_z(input)

    class Toy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear_attn = LinearAttention()

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.linear_attn(input)

    config = LoraConfig(
        target_modules=["in_proj_z"],
        r=4,
        lora_alpha=4,
        lora_dropout=0.0,
        bias="none",
    )
    toy = Toy()
    register_fast_lora(config, toy)
    model = get_peft_model(toy, config, autocast_adapter_dtype=False)
    wrapped = model.base_model.model
    assert isinstance(wrapped, Toy)
    layer = wrapped.linear_attn.in_proj_z
    assert isinstance(layer, FastGgufLoraLinear)
    assert layer.uses_packed_mmq() is False
    lora_a = layer.lora_A["default"]
    lora_b = layer.lora_B["default"]
    assert isinstance(lora_a, torch.nn.Linear)
    assert isinstance(lora_b, torch.nn.Linear)

    generator = torch.Generator(device="cuda").manual_seed(9753)
    with torch.no_grad():
        lora_b.weight.normal_(generator=generator, std=0.02)
    input = torch.randn(
        1,
        2048,
        2048,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    grad_output = torch.randn(
        1,
        2048,
        out_features,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )

    dispatched_ops: list[str] = []

    class _RecordOps(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            dispatched_ops.append(str(func))
            return func(*args, **(kwargs or {}))

    with _RecordOps():
        actual = model(input)
        actual.backward(grad_output)
    assert "torch_ggml_ops._mmq_launch.default" not in dispatched_ops
    assert "torch_ggml_ops._mmq_grad_input_launch.default" not in dispatched_ops

    logical_weight = dequantize_gguf_tensor(
        payload,
        tensor.tensor_type,
        dtype=torch.bfloat16,
        device="cuda",
    ).reshape(out_features, 2048)
    input_ref = input.detach().clone().requires_grad_(True)
    a_ref = lora_a.weight.detach().clone().requires_grad_(True)
    b_ref = lora_b.weight.detach().clone().requires_grad_(True)
    base_ref = torch.nn.functional.linear(input_ref, logical_weight)
    hidden_ref = torch.matmul(input_ref, a_ref.transpose(0, 1))
    output_ref = torch.addmm(
        base_ref.reshape(-1, out_features),
        hidden_ref.reshape(-1, 4),
        b_ref.transpose(0, 1),
        beta=1,
        alpha=layer.scaling["default"],
    ).reshape_as(actual)
    output_ref.backward(grad_output)

    torch.testing.assert_close(actual, output_ref, rtol=0, atol=0)
    torch.testing.assert_close(
        require_grad(input), require_grad(input_ref), rtol=0, atol=8e-3
    )
    torch.testing.assert_close(
        require_grad(lora_a.weight),
        require_grad(a_ref),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        require_grad(lora_b.weight),
        require_grad(b_ref),
        rtol=0,
        atol=0,
    )
    assert layer.base_layer.weight.grad is None
