import math
import os
from collections import defaultdict
from pathlib import Path

import gguf
import numpy as np
import pytest
import torch
from liger_kernel.transformers.model.loss_utils import (
    LigerForCausalLMLoss,
    unpack_cross_entropy_result,
)
from torch.utils._python_dispatch import TorchDispatchMode
from transformers.integrations.gguf import GGUFLinear
from transformers.integrations.gguf_dequant import GGUFQuantizedTensor

from gguf_liger_loss import _packed_q8_liger_for_causal_lm_loss

_MODEL = Path(
    os.environ.get(
        "GGUF_MMQ_TEST_MODEL",
        os.path.expanduser("~/models/qwen3.6/Qwen3.6-35B-A3B-APEX-I-Mini.gguf"),
    )
)


class _MMQCounter(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.counts = defaultdict(int)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = str(func)
        if name.startswith("torch_ggml_ops.mmq"):
            self.counts[name] += 1
        return func(*args, **(kwargs or {}))


@pytest.fixture(scope="module")
def q6_lm_head() -> GGUFLinear:
    if not _MODEL.is_file():
        pytest.skip("GGUF model is unavailable")
    reader = gguf.GGUFReader(_MODEL)
    tensor = next(tensor for tensor in reader.tensors if tensor.name == "output.weight")
    out_features = 37
    packed_host = np.array(
        tensor.data[:out_features], dtype=np.uint8, copy=True, order="C"
    )
    packed = torch.from_numpy(packed_host).to("cuda")
    module = GGUFLinear(
        2048,
        out_features,
        bias=False,
        device="cuda",
        dtype=torch.bfloat16,
        compute_dtype=torch.bfloat16,
    )
    module.weight = GGUFQuantizedTensor(
        packed,
        quant_type=tensor.tensor_type,
        logical_shape=(out_features, 2048),
    )
    return module


def test_packed_q8_liger_loss_matches_logical_reference_and_uses_native_ops(
    q6_lm_head: GGUFLinear,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(12345)
    hidden_reference = torch.randn(
        1,
        65,
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
        (1, 65),
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
    reference_gradient = hidden_reference.grad.float()
    packed_gradient = hidden_packed.grad.float()
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
    assert packed_accuracy.shape == reference_accuracy.shape == torch.Size([])
    assert packed_predictions.shape == reference_predictions.shape == (65,)
    assert torch.isfinite(packed_loss)
    assert torch.isfinite(hidden_packed.grad).all()
    assert q6_lm_head.weight.grad is None
    assert counter.counts["torch_ggml_ops.mmq.default"] == math.ceil(65 / 64)
    assert counter.counts["torch_ggml_ops.mmq_grad_input.default"] == math.ceil(65 / 64)


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
    q6_lm_head: GGUFLinear,
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
    q6_lm_head: GGUFLinear,
) -> None:
    hidden = torch.randn(
        1, 2, 2048, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    labels = torch.tensor([[3, 5]], device="cuda")
    loss = _packed_q8_liger_for_causal_lm_loss(
        hidden_states=hidden,
        lm_head=q6_lm_head,
        labels=labels,
        hidden_size=2048,
    )

    with pytest.raises(
        RuntimeError,
        match="Packed Q8_1 GGUF LM-head loss does not support higher-order gradients",
    ):
        torch.autograd.grad(loss, hidden, create_graph=True)
