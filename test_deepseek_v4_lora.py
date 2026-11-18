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
from transformers.integrations.gguf.modules import GgufGroupedLinear, GgufLinear

from deepseek_v4_lora import (
    DEEPSEEK_V4_TARGET_MODULES_PATTERN,
    DeepseekV4GgufLoraLinear,
    _RejectedDeepseekV4GroupedLora,
    configure_deepseek_v4_grouped_mmq,
    register_deepseek_v4_lora,
    require_complete_deepseek_v4_grouped_mmq,
)
from test_support import require_grad

_MODEL = Path(
    os.environ.get(
        "GGUF_DEEPSEEK_MMQ_TEST_MODEL",
        os.path.expanduser("~/models/ds4/DeepSeek-V4-Flash-IQ2XXS.gguf"),
    )
)


class _RecordOps(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.operations: list[str] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.operations.append(str(func))
        return func(*args, **(kwargs or {}))


def _q8_linear(weight: np.ndarray) -> GgufLinear:
    packed = torch.from_numpy(
        gguf.quantize(weight.astype(np.float32), gguf.GGMLQuantizationType.Q8_0).copy()
    ).to("cuda")
    module = GgufLinear(
        weight.shape[1],
        weight.shape[0],
        bias=False,
        device="cuda",
        dtype=torch.bfloat16,
        compute_dtype=torch.bfloat16,
    )
    module.weight = GgufQuantizedParameter(
        packed,
        quant_type=gguf.GGMLQuantizationType.Q8_0,
        logical_shape=weight.shape,
    )
    return module


def test_q8_0_ordinary_lora_uses_native_base_and_fused_residual() -> None:
    if not _MODEL.is_file():
        pytest.skip("DeepSeek GGUF model is unavailable")
    reader = gguf.GGUFReader(_MODEL)
    tensor = next(
        item for item in reader.tensors if item.name == "blk.0.attn_q_a.weight"
    )
    packed = torch.from_numpy(
        np.array(tensor.data, dtype=np.uint8, copy=True, order="C")
    ).to("cuda")

    class Toy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_a_proj = GgufLinear(
                4096,
                1024,
                bias=False,
                device="cuda",
                dtype=torch.bfloat16,
                compute_dtype=torch.bfloat16,
            )
            self.q_a_proj.weight = GgufQuantizedParameter(
                packed,
                quant_type=tensor.tensor_type,
                logical_shape=(1024, 4096),
            )

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
    wrapped = model.base_model.model
    assert isinstance(wrapped, Toy)
    layer = wrapped.q_a_proj
    assert isinstance(layer, DeepseekV4GgufLoraLinear)
    lora_b = layer.lora_B["default"]
    assert isinstance(lora_b, torch.nn.Linear)

    with torch.no_grad():
        lora_b.weight.normal_(std=0.02)
    x = torch.randn(2048, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    recorder = _RecordOps()
    with recorder:
        output = model(x)
        output.square().float().mean().backward()

    assert any(
        "torch_ggml_ops._mmq_launch.default" in operation
        for operation in recorder.operations
    )
    assert any(
        "torch_ggml_ops._mmq_grad_input_launch.default" in operation
        for operation in recorder.operations
    )
    assert output.shape == (2048, 1024)
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
    grouped = GgufGroupedLinear(
        4096,
        8192,
        8,
        device="cuda",
        dtype=torch.bfloat16,
        compute_dtype=torch.bfloat16,
    )
    grouped.weight = GgufQuantizedParameter(
        packed,
        quant_type=gguf.GGMLQuantizationType.Q8_0,
        logical_shape=logical_weight.shape,
    )
    report = configure_deepseek_v4_grouped_mmq(grouped)
    assert report["enabled"] == 1
    assert report["patched"] == 1
    again = configure_deepseek_v4_grouped_mmq(grouped)
    assert again["enabled"] == 1
    assert again["patched"] == 0
    assert again["already_patched"] == 1

    hidden = torch.randn(
        2048, 8, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    grad_output_groups = [
        torch.randn(2048, 1024, device="cuda", dtype=torch.bfloat16) for _ in range(8)
    ]
    grad_output = torch.stack(grad_output_groups, dim=1)
    actual = grouped(hidden)
    actual.backward(grad_output)
    actual_grad = require_grad(hidden).detach().clone()

    hidden_reference = hidden.detach().clone().requires_grad_(True)
    packed_groups = packed.reshape(8, 1024, -1)
    reference_outputs = [
        torch_ggml_ops.mmq(
            hidden_reference[:, group, :].contiguous(),
            packed_groups[group].clone(),
            int(gguf.GGMLQuantizationType.Q8_0),
            1024,
        )
        for group in range(8)
    ]
    reference = torch.stack(reference_outputs, dim=1)
    # Each stacked branch receives a strided slice of the public gradient.
    # Supply a contiguous cotangent at the native MMQ boundary instead of
    # making torch-ggml-ops hide a copy in its autograd wrapper.
    reference_grad = torch.stack(
        [
            torch.autograd.grad(
                reference_outputs[group],
                hidden_reference,
                grad_output_groups[group],
                retain_graph=group < 7,
            )[0][:, group]
            for group in range(8)
        ],
        dim=1,
    )

    # Both paths use the same packed Q8_0 dense MMQ arithmetic. Exact equality
    # is a stronger check than an independent dequantized BF16 tolerance here.
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    torch.testing.assert_close(actual_grad, reference_grad, rtol=0, atol=0)


def test_grouped_output_lora_is_explicitly_rejected() -> None:
    grouped = GgufGroupedLinear(
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


def test_grouped_mmq_gate_rejects_incomplete_inventory() -> None:
    with pytest.raises(RuntimeError, match="incomplete DeepSeek V4 grouped MMQ"):
        require_complete_deepseek_v4_grouped_mmq({"enabled": 42})
