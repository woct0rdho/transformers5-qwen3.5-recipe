import math
import os
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import gguf
import numpy as np
import pytest
import torch
from liger_kernel.transformers.model.loss_utils import (
    LigerForCausalLMLoss,
    unpack_cross_entropy_result,
)
from liger_kernel.transformers.model.output_classes import (
    LigerMoeCausalLMOutputWithPast,
)
from torch.utils._python_dispatch import TorchDispatchMode
from transformers.integrations.gguf.gguf_quantized_parameter import (
    GgufQuantizedParameter,
)
from transformers.integrations.gguf.modules import GgufLinear

from gguf_liger_loss import (
    _PACKED_LM_HEAD_CHUNK_SIZE,
    _packed_q8_liger_for_causal_lm_loss,
    gguf_liger_lce_forward,
)
from packed_liger_loss import PackedLossResult

_MODEL = Path(
    os.environ.get(
        "GGUF_MMQ_TEST_MODEL",
        os.path.expanduser("~/models/qwen3.6/Qwen3.6-35B-A3B-APEX-I-Mini.gguf"),
    )
)


def _require_tensor(value: torch.Tensor | None) -> torch.Tensor:
    if value is None:
        raise AssertionError("expected a tensor")
    return value


class _MMQCounter(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.counts = defaultdict(int)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = str(func)
        if name.startswith("torch_ggml_ops._mmq"):
            self.counts[name] += 1
        return func(*args, **(kwargs or {}))


@pytest.fixture(scope="module")
def q6_lm_head() -> GgufLinear:
    if not _MODEL.is_file():
        pytest.skip("GGUF model is unavailable")
    reader = gguf.GGUFReader(_MODEL)
    tensor = next(tensor for tensor in reader.tensors if tensor.name == "output.weight")
    out_features = int(tensor.data.shape[0])
    packed_host = np.array(tensor.data, dtype=np.uint8, copy=True, order="C")
    packed = torch.from_numpy(packed_host).to("cuda")
    module = GgufLinear(
        2048,
        out_features,
        bias=False,
        device="cuda",
        dtype=torch.bfloat16,
        compute_dtype=torch.bfloat16,
    )
    module.weight = GgufQuantizedParameter(
        packed,
        quant_type=tensor.tensor_type,
        logical_shape=(out_features, 2048),
    )
    return module


def test_packed_q8_liger_loss_matches_logical_reference_and_uses_native_ops(
    q6_lm_head: GgufLinear,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(12345)
    rows = 2 * _PACKED_LM_HEAD_CHUNK_SIZE
    hidden_reference = torch.randn(
        1,
        rows,
        2048,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    hidden_packed = hidden_reference.detach().clone().requires_grad_(True)
    labels = torch.randint(
        0,
        q6_lm_head.out_features,
        (1, rows),
        generator=generator,
        device="cuda",
    )
    labels[0, 11] = -100

    logical_weight = q6_lm_head.materialize_logical_weight(
        dtype=torch.bfloat16, device="cuda"
    )
    reference_result = LigerForCausalLMLoss(
        hidden_states=hidden_reference,
        lm_head_weight=logical_weight,
        labels=labels,
        hidden_size=2048,
        return_token_accuracy=True,
        return_predicted_tokens=True,
    )
    reference_loss, _, reference_accuracy, reference_predictions = (
        unpack_cross_entropy_result(reference_result)
    )
    reference_loss.backward()

    monkeypatch.setattr(
        q6_lm_head,
        "materialize_logical_weight",
        lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError("logical LM-head materialization is forbidden")
        ),
    )
    counter = _MMQCounter()
    with counter:
        packed_result = _packed_q8_liger_for_causal_lm_loss(
            hidden_states=hidden_packed,
            lm_head=q6_lm_head,
            labels=labels,
            hidden_size=2048,
            return_token_accuracy=True,
            return_predicted_tokens=True,
        )
        packed_loss, _, packed_accuracy, packed_predictions = (
            unpack_cross_entropy_result(packed_result)
        )
        packed_loss.backward()

    loss_relative_error = float(
        ((packed_loss - reference_loss).abs() / reference_loss.abs()).detach()
    )
    reference_gradient = _require_tensor(hidden_reference.grad).float()
    packed_gradient = _require_tensor(hidden_packed.grad).float()
    gradient_cosine = float(
        torch.nn.functional.cosine_similarity(
            reference_gradient.flatten(), packed_gradient.flatten(), dim=0
        )
    )
    gradient_relative_l2 = float(
        torch.linalg.vector_norm(packed_gradient - reference_gradient)
        / torch.linalg.vector_norm(reference_gradient)
    )

    assert loss_relative_error < 5e-3
    assert gradient_cosine > 0.999
    assert gradient_relative_l2 < 0.03
    packed_accuracy = _require_tensor(packed_accuracy)
    reference_accuracy = _require_tensor(reference_accuracy)
    packed_predictions = _require_tensor(packed_predictions)
    reference_predictions = _require_tensor(reference_predictions)
    assert packed_accuracy.shape == reference_accuracy.shape == torch.Size([])
    assert packed_predictions.shape == reference_predictions.shape == (rows,)
    assert torch.isfinite(packed_loss)
    assert torch.isfinite(_require_tensor(hidden_packed.grad)).all()
    assert q6_lm_head.weight.grad is None
    assert _PACKED_LM_HEAD_CHUNK_SIZE == 256
    expected_calls = math.ceil(rows / _PACKED_LM_HEAD_CHUNK_SIZE)
    assert counter.counts["torch_ggml_ops._mmq_launch.default"] == expected_calls
    assert (
        counter.counts["torch_ggml_ops._mmq_grad_input_launch.default"]
        == expected_calls
    )


@pytest.mark.parametrize(
    ("loss_kwargs", "message"),
    (
        ({"ce_weight": torch.ones(37)}, "class weights"),
        ({"label_smoothing": 0.1}, "label smoothing"),
        ({"use_token_scaling": True}, "token scaling"),
        ({"final_logit_softcapping": 30.0}, "logit softcapping"),
    ),
)
def test_packed_q8_liger_loss_rejects_unsupported_objectives(
    q6_lm_head: GgufLinear,
    loss_kwargs: dict,
    message: str,
) -> None:
    hidden = torch.randn(1, 2, 2048, device="cuda", dtype=torch.bfloat16)
    labels = torch.tensor([[3, 5]], device="cuda")

    with pytest.raises(RuntimeError, match=message):
        _packed_q8_liger_for_causal_lm_loss(
            hidden_states=hidden,
            lm_head=q6_lm_head,
            labels=labels,
            hidden_size=2048,
            **loss_kwargs,
        )


def test_packed_q8_liger_loss_rejects_higher_order_gradients(
    q6_lm_head: GgufLinear,
) -> None:
    hidden = torch.randn(
        1, 64, 2048, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    labels = torch.randint(0, q6_lm_head.out_features, (1, 64), device="cuda")
    loss = _packed_q8_liger_for_causal_lm_loss(
        hidden_states=hidden,
        lm_head=q6_lm_head,
        labels=labels,
        hidden_size=2048,
    )
    assert isinstance(loss, torch.Tensor)

    with pytest.raises(
        RuntimeError,
        match="Packed Q8_1 GGUF LM-head loss does not support higher-order gradients",
    ):
        torch.autograd.grad(loss, hidden, create_graph=True)


class _StubInnerModel(torch.nn.Module):
    def __init__(self, hidden_states: torch.Tensor) -> None:
        super().__init__()
        self.hidden_states = hidden_states

    def forward(self, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(last_hidden_state=self.hidden_states)


class _StubForwardModel(torch.nn.Module):
    """Minimal Qwen surface the scoped packed forward reads."""

    def __init__(self, hidden_states: torch.Tensor) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=hidden_states.shape[-1],
            output_attentions=False,
            output_hidden_states=False,
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


def test_qwen_forward_surfaces_optional_loss_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accuracy = torch.tensor(0.5)
    predicted = torch.zeros(6, dtype=torch.long)

    def packed_loss(**kwargs: object) -> PackedLossResult:
        assert kwargs["return_token_accuracy"] is True
        assert kwargs["return_predicted_tokens"] is True
        return torch.tensor(2.0), None, accuracy, predicted

    hidden = torch.zeros(2, 3, 8, dtype=torch.bfloat16)
    model = _StubForwardModel(hidden)
    labels = torch.zeros(2, 3, dtype=torch.long)
    monkeypatch.setattr(
        "gguf_liger_loss._packed_q8_liger_for_causal_lm_loss", packed_loss
    )

    output = gguf_liger_lce_forward(
        model,
        labels=labels,
        return_token_accuracy=True,
        return_predicted_tokens=True,
    )

    assert isinstance(output, LigerMoeCausalLMOutputWithPast)
    assert output.loss is not None and float(output.loss) == 2.0
    assert output.logits is None
    assert output.token_accuracy is accuracy
    assert output.predicted_tokens is predicted
