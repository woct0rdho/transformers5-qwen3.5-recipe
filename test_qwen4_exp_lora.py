"""Qwen4-Exp MMQ wiring: the frozen projection patch and the ordinary capability gate.

`gguf-py` ships quantizers only for Q8_0, Q5_0, and Q4_0, and `torch_ggml_ops` deploys exact keys, so
these tests build their payloads at the shared-expert down geometry [2560, 640] with Q8_0, which is
deployed for 2048 rows in both directions. The patch and the capability gate are geometry
independent, so the site names stay the Qwen4-Exp ones.
"""

import gguf
import numpy as np
import pytest
import torch
from peft import LoraConfig
from torch_ggml_ops import mmq
from transformers.integrations.gguf.gguf_quantized_parameter import (
    GgufQuantizedParameter,
)
from transformers.integrations.gguf.modules import GgufLinear

from fast_lora import FastGgufLoraLinear, packed_mmq_linear
from gguf_support import dequantize_gguf_tensor
from qwen4_exp_lora import (
    EXPECTED_FROZEN_MMQ,
    EXPECTED_NATIVE_MMQ_WRAPPERS,
    configure_qwen4_exp_frozen_mmq,
    require_complete_qwen4_exp_frozen_mmq,
)
from test_support import require_grad

_OUT_FEATURES = 2560
_IN_FEATURES = 640
_QUANT_TYPE = 8
_ROWS = 2048


def _packed_linear(weight: np.ndarray, **kwargs) -> GgufLinear:
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
        **kwargs,
    )
    module.weight = GgufQuantizedParameter(
        packed,
        quant_type=gguf.GGMLQuantizationType.Q8_0,
        logical_shape=weight.shape,
    )
    return module


def _real_weight() -> np.ndarray:
    generator = np.random.default_rng(20261007)
    return (
        generator.standard_normal((_OUT_FEATURES, _IN_FEATURES), dtype=np.float32)
        * 0.02
    )


class _PleSite(torch.nn.Module):
    def __init__(self, key_proj: GgufLinear) -> None:
        super().__init__()
        self.key_proj = key_proj


class _PleToy(torch.nn.Module):
    """One frozen site under the name the patch matches."""

    def __init__(self, key_proj: GgufLinear) -> None:
        super().__init__()
        self.ple = _PleSite(key_proj)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.ple.key_proj(input)


def test_packed_mmq_linear_matches_the_dequantized_reference() -> None:
    module = _packed_linear(_real_weight())
    hidden = torch.randn(_ROWS, _IN_FEATURES, device="cuda", dtype=torch.bfloat16)

    actual = packed_mmq_linear(module, hidden)
    expected = torch.nn.functional.linear(
        hidden, dequantize_gguf_tensor(module.weight, _QUANT_TYPE, dtype=torch.bfloat16)
    )
    relative_l2 = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert float(relative_l2) < 2e-2


def test_packed_mmq_linear_rejects_a_floating_weight() -> None:
    module = GgufLinear(
        256, 320, bias=False, device="cuda", dtype=torch.bfloat16, floating_weight=True
    )
    hidden = torch.randn(8, 256, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(TypeError, match="GGUF-quantized"):
        packed_mmq_linear(module, hidden)


def test_native_mmq_requires_bf16_compute_dtype() -> None:
    module = _packed_linear(_real_weight())
    module.set_compute_dtype(torch.float32)
    hidden = torch.randn(8, _IN_FEATURES, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="BF16"):
        packed_mmq_linear(module, hidden)


def test_frozen_ple_projection_takes_the_native_forward() -> None:
    module = _packed_linear(_real_weight())
    model = _PleToy(module)

    report = configure_qwen4_exp_frozen_mmq(model)
    assert report["enabled"] == 1
    assert report["patched"] == 1
    assert report["paths"] == ["ple.key_proj"]
    require_complete_qwen4_exp_frozen_mmq(report)

    again = configure_qwen4_exp_frozen_mmq(model)
    assert again["enabled"] == 1
    assert again["already_patched"] == 1
    assert again["patched"] == 0

    hidden = torch.randn(
        _ROWS, _IN_FEATURES, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    grad_output = torch.randn(_ROWS, _OUT_FEATURES, device="cuda", dtype=torch.bfloat16)
    actual = model(hidden)
    actual.backward(grad_output)
    actual_grad = require_grad(hidden).detach().clone()

    reference_input = hidden.detach().clone().requires_grad_(True)
    reference = mmq(
        reference_input,
        module.weight.as_subclass(torch.Tensor),
        _QUANT_TYPE,
        _OUT_FEATURES,
    )
    reference.backward(grad_output)
    assert torch.equal(actual, reference)
    assert torch.equal(actual_grad, require_grad(reference_input))
    assert isinstance(module.weight, GgufQuantizedParameter)
    assert not module.weight.requires_grad


def test_frozen_mmq_gate_rejects_an_incomplete_inventory() -> None:
    report = configure_qwen4_exp_frozen_mmq(torch.nn.Module())
    with pytest.raises(RuntimeError, match="incomplete"):
        require_complete_qwen4_exp_frozen_mmq(report)


def test_an_input_permutation_keeps_the_generic_base_forward() -> None:
    """A module that gathers its input keeps the kind of forward that applies the gather."""

    base = GgufLinear(
        4,
        2,
        bias=False,
        device="cuda",
        dtype=torch.bfloat16,
        input_permutation=torch.tensor([1, 0, 2]),
    )
    config = LoraConfig(target_modules=["toy"], r=4, lora_alpha=4, lora_dropout=0.0)
    wrapper = FastGgufLoraLinear(
        base_layer=base, adapter_name="toy", config=config, r=4, lora_alpha=4
    )
    assert base.input_permutation is not None
    assert wrapper.packed_mmq_weight() is None
    assert wrapper.uses_packed_mmq() is False


def test_the_wired_inventory_constants_are_explicit() -> None:
    assert EXPECTED_NATIVE_MMQ_WRAPPERS == 300
    assert EXPECTED_FROZEN_MMQ == 1
