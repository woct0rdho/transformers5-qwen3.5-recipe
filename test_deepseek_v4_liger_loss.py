import os
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import gguf
import numpy as np
import pytest
import torch
from liger_kernel.transformers.model.loss_utils import LigerForCausalLMLoss
from liger_kernel.transformers.model.output_classes import (
    LigerMoeCausalLMOutputWithPast,
)
from torch.utils._python_dispatch import TorchDispatchMode
from transformers.integrations.gguf.gguf_quantized_parameter import (
    GgufQuantizedParameter,
)
from transformers.integrations.gguf.modules import GgufLinear

from deepseek_v4_liger_loss import (
    _deepseek_v4_liger_forward,
    deepseek_v4_liger_causal_lm_loss,
    deepseek_v4_packed_liger_causal_lm_loss,
)
from packed_liger_loss import PackedLossResult
from test_support import require_grad

_MODEL = Path(
    os.environ.get(
        "GGUF_DEEPSEEK_MMQ_TEST_MODEL",
        os.path.expanduser("~/models/ds4/DeepSeek-V4-Flash-IQ2XXS.gguf"),
    )
)


def _q8_lm_head(weight: np.ndarray) -> GgufLinear:
    packed = torch.from_numpy(
        gguf.quantize(weight.astype(np.float32), gguf.GGMLQuantizationType.Q8_0).copy()
    ).to("cuda")
    head = GgufLinear(
        weight.shape[1],
        weight.shape[0],
        bias=False,
        device="cuda",
        dtype=torch.bfloat16,
        compute_dtype=torch.bfloat16,
    )
    head.weight = GgufQuantizedParameter(
        packed,
        quant_type=gguf.GGMLQuantizationType.Q8_0,
        logical_shape=weight.shape,
    )
    return head


class _StubInnerModel(torch.nn.Module):
    def __init__(self, hidden_states: torch.Tensor) -> None:
        super().__init__()
        self.hidden_states = hidden_states

    def forward(self, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(last_hidden_state=self.hidden_states)


class _StubForwardModel(torch.nn.Module):
    """Minimal DeepSeek surface the scoped packed forward reads."""

    def __init__(self, hidden_states: torch.Tensor) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=hidden_states.shape[-1],
            output_router_logits=False,
            return_dict=True,
        )
        self.vocab_size = 16
        self.lm_head = GgufLinear(
            hidden_states.shape[-1],
            16,
            bias=False,
            device="cpu",
            dtype=torch.bfloat16,
            floating_weight=True,
        )
        self.num_experts = 2
        self.num_experts_per_tok = 1
        self.router_aux_loss_coef = 0.0
        self.model = _StubInnerModel(hidden_states)

    def loss_function(self, *args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("packed forward must not materialize logits")


def _run_stub_forward(
    monkeypatch: pytest.MonkeyPatch,
    packed_loss: Callable[..., PackedLossResult],
    **forward_kwargs: object,
) -> LigerMoeCausalLMOutputWithPast:
    hidden = torch.zeros(2, 3, 8, dtype=torch.bfloat16)
    model = _StubForwardModel(hidden)
    labels = torch.zeros(2, 3, dtype=torch.long)
    monkeypatch.setattr(
        "deepseek_v4_liger_loss.deepseek_v4_packed_liger_causal_lm_loss",
        packed_loss,
    )
    return _deepseek_v4_liger_forward(model, labels=labels, **forward_kwargs)


def test_deepseek_v4_forward_surfaces_optional_loss_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accuracy = torch.tensor(0.5)
    predicted = torch.zeros(6, dtype=torch.long)

    def packed_loss(**kwargs: object) -> PackedLossResult:
        assert kwargs["return_token_accuracy"] is True
        assert kwargs["return_predicted_tokens"] is True
        return torch.tensor(2.0), None, accuracy, predicted

    output = _run_stub_forward(
        monkeypatch,
        packed_loss,
        return_token_accuracy=True,
        return_predicted_tokens=True,
    )

    assert isinstance(output, LigerMoeCausalLMOutputWithPast)
    assert output.loss is not None and float(output.loss) == 2.0
    assert output.logits is None
    assert output.token_accuracy is accuracy
    assert output.predicted_tokens is predicted


def test_deepseek_v4_forward_defaults_optional_loss_outputs_to_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def packed_loss(**kwargs: object) -> PackedLossResult:
        return torch.tensor(2.0)

    output = _run_stub_forward(monkeypatch, packed_loss)

    assert isinstance(output, LigerMoeCausalLMOutputWithPast)
    assert output.loss is not None and float(output.loss) == 2.0
    assert output.logits is None
    assert output.token_accuracy is None
    assert output.predicted_tokens is None


def test_scoped_q8_0_liger_loss_matches_logical_reference() -> None:
    generator = np.random.default_rng(2468)
    head = _q8_lm_head(generator.standard_normal((37, 256)))
    hidden_reference = torch.randn(
        2, 17, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    hidden_scoped = hidden_reference.detach().clone().requires_grad_(True)
    labels = torch.randint(0, 37, (2, 17), device="cuda")
    labels[0, 5] = -100

    logical_weight = head.materialize_logical_weight(
        dtype=torch.bfloat16, device="cuda"
    )
    reference = LigerForCausalLMLoss(
        hidden_states=hidden_reference,
        lm_head_weight=logical_weight,
        labels=labels,
        hidden_size=256,
    )
    reference.backward()
    del logical_weight

    materializations = 0
    original_materialize = head.materialize_logical_weight

    def counted_materialize(**kwargs):
        nonlocal materializations
        materializations += 1
        return original_materialize(**kwargs)

    head.__dict__["materialize_logical_weight"] = counted_materialize
    scoped = deepseek_v4_liger_causal_lm_loss(
        hidden_scoped,
        head,
        labels,
        hidden_size=256,
    )
    scoped.backward()

    torch.testing.assert_close(scoped, reference, rtol=0, atol=0)
    torch.testing.assert_close(
        require_grad(hidden_scoped), require_grad(hidden_reference), rtol=0, atol=0
    )
    assert materializations == 1
    assert head.weight.grad is None
    assert torch.isfinite(scoped)
    assert torch.isfinite(require_grad(hidden_scoped)).all()


def test_packed_q8_0_liger_loss_uses_native_mmq_without_materializing_head() -> None:
    if not _MODEL.is_file():
        pytest.skip("DeepSeek GGUF model is unavailable")
    torch.manual_seed(9753)
    reader = gguf.GGUFReader(_MODEL)
    tensor = next(item for item in reader.tensors if item.name == "output.weight")
    packed_weight = torch.from_numpy(
        np.array(tensor.data, dtype=np.uint8, copy=True, order="C")
    ).to("cuda")
    head = GgufLinear(
        4096,
        129280,
        bias=False,
        device="cuda",
        dtype=torch.bfloat16,
        compute_dtype=torch.bfloat16,
    )
    head.weight = GgufQuantizedParameter(
        packed_weight,
        quant_type=tensor.tensor_type,
        logical_shape=(129280, 4096),
    )
    hidden_reference = torch.randn(
        1, 32, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    hidden_packed = hidden_reference.detach().clone().requires_grad_(True)
    labels = torch.randint(0, 129280, (1, 32), device="cuda")
    labels[0, 4] = -100

    reference = deepseek_v4_liger_causal_lm_loss(
        hidden_reference,
        head,
        labels,
        hidden_size=4096,
    )
    reference.backward()

    materializations = 0
    original_materialize = head.materialize_logical_weight

    def counted_materialize(**kwargs):
        nonlocal materializations
        materializations += 1
        return original_materialize(**kwargs)

    head.__dict__["materialize_logical_weight"] = counted_materialize
    operations: list[str] = []

    class _RecordOps(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            operations.append(str(func))
            return func(*args, **(kwargs or {}))

    with _RecordOps():
        packed = deepseek_v4_packed_liger_causal_lm_loss(
            hidden_packed,
            head,
            labels,
            hidden_size=4096,
        )
        assert isinstance(packed, torch.Tensor)
        packed.backward()

    torch.testing.assert_close(packed, reference, rtol=1e-3, atol=1e-3)
    assert materializations == 0
    assert "torch_ggml_ops._mmq_launch.default" in operations
    assert "torch_ggml_ops._mmq_grad_input_launch.default" in operations
    assert torch.isfinite(require_grad(hidden_packed)).all()
