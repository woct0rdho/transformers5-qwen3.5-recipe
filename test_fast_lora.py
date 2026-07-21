import os
from pathlib import Path

import gguf
import numpy as np
import pytest
import torch
import torch_ggml_ops
from peft import LoraConfig, get_peft_model
from torch.utils._python_dispatch import TorchDispatchMode
from transformers.integrations.gguf import GGUFLinear
from transformers.integrations.gguf_dequant import (
    GGUFQuantizedTensor,
    dequantize_gguf_tensor,
)

from fast_lora import FastGGUFLoraLinear, register_fast_lora

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
    payload = torch.from_numpy(
        np.array(tensor.data[:37], dtype=np.uint8, copy=True, order="C")
    ).to("cuda")
    packed = GGUFQuantizedTensor(
        payload,
        quant_type=tensor.tensor_type,
        logical_shape=(37, 2048),
    )

    class Toy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = GGUFLinear(
                2048,
                37,
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
    register_fast_lora(config)
    model = get_peft_model(Toy(), config, autocast_adapter_dtype=False)
    layer = model.base_model.model.proj
    assert isinstance(layer, FastGGUFLoraLinear)
    assert layer.lora_A["default"].weight.dtype == torch.bfloat16
    assert layer.lora_B["default"].weight.dtype == torch.bfloat16

    generator = torch.Generator(device="cuda").manual_seed(2468)
    with torch.no_grad():
        layer.lora_B["default"].weight.normal_(generator=generator, std=0.02)
    input = torch.randn(
        3,
        11,
        2048,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    grad_output = torch.randn(
        3, 11, 37, generator=generator, device="cuda", dtype=torch.bfloat16
    )

    dispatched_ops: list[str] = []

    class _RecordOps(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            dispatched_ops.append(str(func))
            return func(*args, **(kwargs or {}))

    with _RecordOps():
        actual = model(input)
        actual.backward(grad_output)
    assert "torch_ggml_ops.mmq.default" in dispatched_ops
    assert "torch_ggml_ops.mmq_grad_input.default" in dispatched_ops
    actual_input_grad = input.grad.detach().clone()
    actual_a_grad = layer.lora_A["default"].weight.grad.detach().clone()
    actual_b_grad = layer.lora_B["default"].weight.grad.detach().clone()

    logical_weight = dequantize_gguf_tensor(
        payload,
        tensor.tensor_type,
        dtype=torch.bfloat16,
        device="cuda",
    ).reshape(37, 2048)
    input_ref = input.detach().clone().requires_grad_(True)
    a_ref = layer.lora_A["default"].weight.detach().clone().requires_grad_(True)
    b_ref = layer.lora_B["default"].weight.detach().clone().requires_grad_(True)
    base_ref = torch.nn.functional.linear(input_ref, logical_weight)
    hidden_ref = torch.matmul(input_ref, a_ref.transpose(0, 1))
    output_ref = torch.addmm(
        base_ref.reshape(-1, 37),
        hidden_ref.reshape(-1, 4),
        b_ref.transpose(0, 1),
        beta=1,
        alpha=layer.scaling["default"],
    ).reshape_as(actual)
    output_ref.backward(grad_output)

    torch.testing.assert_close(actual_input_grad, input_ref.grad, rtol=0, atol=0)
    torch.testing.assert_close(actual_a_grad, a_ref.grad, rtol=0, atol=0)
    torch.testing.assert_close(actual_b_grad, b_ref.grad, rtol=0, atol=0)
    assert layer.base_layer.weight.grad is None

    mmq_base = torch_ggml_ops.mmq(input.detach(), payload, int(tensor.tensor_type), 37)
    expected_actual = torch.addmm(
        mmq_base.reshape(-1, 37),
        torch.matmul(input.detach(), a_ref.detach().transpose(0, 1)).reshape(-1, 4),
        b_ref.detach().transpose(0, 1),
        beta=1,
        alpha=layer.scaling["default"],
    ).reshape_as(actual)
    torch.testing.assert_close(actual, expected_actual, rtol=0, atol=0)
