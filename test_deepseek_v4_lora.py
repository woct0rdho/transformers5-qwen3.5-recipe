import gguf
import numpy as np
import pytest
import torch
import torch_ggml_ops
from peft import LoraConfig, get_peft_model
from torch.utils._python_dispatch import TorchDispatchMode
from transformers.integrations.gguf import GGUFGroupedLinear, GGUFLinear
from transformers.integrations.gguf_dequant import GGUFQuantizedTensor

from deepseek_v4_lora import (
    DEEPSEEK_V4_TARGET_MODULES_PATTERN,
    DeepseekV4GGUFLoraLinear,
    _RejectedDeepseekV4GroupedLora,
    configure_deepseek_v4_grouped_mmq,
    register_deepseek_v4_lora,
)


class _RecordOps(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.operations: list[str] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.operations.append(str(func))
        return func(*args, **(kwargs or {}))


def _q8_linear(weight: np.ndarray) -> GGUFLinear:
    packed = torch.from_numpy(
        gguf.quantize(weight.astype(np.float32), gguf.GGMLQuantizationType.Q8_0).copy()
    ).to("cuda")
    module = GGUFLinear(
        weight.shape[1],
        weight.shape[0],
        bias=False,
        device="cuda",
        dtype=torch.bfloat16,
        compute_dtype=torch.bfloat16,
    )
    module.weight = GGUFQuantizedTensor(
        packed,
        quant_type=gguf.GGMLQuantizationType.Q8_0,
        logical_shape=weight.shape,
    )
    return module


def test_q8_0_ordinary_lora_uses_native_base_and_fused_residual() -> None:
    generator = np.random.default_rng(1234)

    class Toy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_a_proj = _q8_linear(generator.standard_normal((48, 256)))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.q_a_proj(x)

    config = LoraConfig(
        target_modules=DEEPSEEK_V4_TARGET_MODULES_PATTERN,
        r=4,
        lora_alpha=4,
        lora_dropout=0.0,
        bias="none",
    )
    register_deepseek_v4_lora(config)
    model = get_peft_model(Toy(), config, autocast_adapter_dtype=False)
    layer = model.base_model.model.q_a_proj
    assert isinstance(layer, DeepseekV4GGUFLoraLinear)

    with torch.no_grad():
        layer.lora_B["default"].weight.normal_(std=0.02)
    x = torch.randn(7, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    recorder = _RecordOps()
    with recorder:
        output = model(x)
        output.square().float().mean().backward()

    assert any(
        "torch_ggml_ops.mmq.default" in operation for operation in recorder.operations
    )
    assert any(
        "torch_ggml_ops.mmq_grad_input.default" in operation
        for operation in recorder.operations
    )
    assert output.shape == (7, 48)
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert layer.lora_A["default"].weight.grad is not None
    assert layer.lora_B["default"].weight.grad is not None
    assert layer.base_layer.weight.grad is None


def test_fixed_grouped_q8_0_mmq_matches_dense_packed_reference() -> None:
    generator = np.random.default_rng(42)
    logical_weight = generator.standard_normal((8192, 4096), dtype=np.float32)
    packed = torch.from_numpy(
        gguf.quantize(logical_weight, gguf.GGMLQuantizationType.Q8_0).copy()
    ).to("cuda")
    grouped = GGUFGroupedLinear(
        4096,
        8192,
        8,
        device="cuda",
        dtype=torch.bfloat16,
        compute_dtype=torch.bfloat16,
    )
    grouped.weight = GGUFQuantizedTensor(
        packed,
        quant_type=gguf.GGMLQuantizationType.Q8_0,
        logical_shape=logical_weight.shape,
    )
    report = configure_deepseek_v4_grouped_mmq(grouped)
    assert report["enabled"] == 1

    hidden = torch.randn(
        2, 8, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    grad_output = torch.randn(2, 8, 1024, device="cuda", dtype=torch.bfloat16)
    actual = grouped(hidden)
    actual.backward(grad_output)
    actual_grad = hidden.grad.detach().clone()

    hidden_reference = hidden.detach().clone().requires_grad_(True)
    packed_groups = packed.reshape(8, 1024, -1)
    reference = torch.stack(
        [
            torch_ggml_ops.mmq(
                hidden_reference[:, group, :].contiguous(),
                packed_groups[group].clone(),
                int(gguf.GGMLQuantizationType.Q8_0),
                1024,
            )
            for group in range(8)
        ],
        dim=1,
    )
    reference.backward(grad_output)

    # Both paths use the same packed Q8_0 dense MMQ arithmetic.  Exact equality
    # is a stronger check than an independent dequantized BF16 tolerance here.
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    torch.testing.assert_close(actual_grad, hidden_reference.grad, rtol=0, atol=0)


def test_grouped_output_lora_is_explicitly_rejected() -> None:
    grouped = GGUFGroupedLinear(
        32,
        24,
        4,
        device="cuda",
        dtype=torch.bfloat16,
        compute_dtype=torch.bfloat16,
        floating_weight=True,
    )
    with pytest.raises(RuntimeError, match="grouped o_a_proj LoRA is unsupported"):
        _RejectedDeepseekV4GroupedLora(
            grouped,
            "default",
            r=4,
            lora_alpha=4,
            lora_dropout=0.0,
            init_lora_weights=True,
            use_rslora=False,
            use_dora=False,
            lora_bias=False,
            ephemeral_gpu_offload=False,
        )
