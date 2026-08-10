import gguf
import numpy as np
import torch
from liger_kernel.transformers.model.loss_utils import LigerForCausalLMLoss
from torch.utils._python_dispatch import TorchDispatchMode
from transformers.integrations.gguf import GGUFLinear
from transformers.integrations.gguf_dequant import GGUFQuantizedTensor

from deepseek_v4_liger_loss import (
    deepseek_v4_liger_causal_lm_loss,
    deepseek_v4_packed_liger_causal_lm_loss,
)


def _require_grad(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.grad is None:
        raise AssertionError("expected a tensor gradient")
    return tensor.grad


def _q8_lm_head(weight: np.ndarray) -> GGUFLinear:
    packed = torch.from_numpy(
        gguf.quantize(weight.astype(np.float32), gguf.GGMLQuantizationType.Q8_0).copy()
    ).to("cuda")
    head = GGUFLinear(
        weight.shape[1],
        weight.shape[0],
        bias=False,
        device="cuda",
        dtype=torch.bfloat16,
        compute_dtype=torch.bfloat16,
    )
    head.weight = GGUFQuantizedTensor(
        packed,
        quant_type=gguf.GGMLQuantizationType.Q8_0,
        logical_shape=weight.shape,
    )
    return head


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
        _require_grad(hidden_scoped), _require_grad(hidden_reference), rtol=0, atol=0
    )
    assert materializations == 1
    assert head.weight.grad is None
    assert torch.isfinite(scoped)
    assert torch.isfinite(_require_grad(hidden_scoped)).all()


def test_packed_q8_0_liger_loss_uses_native_mmq_without_materializing_head() -> None:
    torch.manual_seed(9753)
    generator = np.random.default_rng(9753)
    head = _q8_lm_head(generator.standard_normal((37, 256)))
    hidden_reference = torch.randn(
        2, 17, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    hidden_packed = hidden_reference.detach().clone().requires_grad_(True)
    labels = torch.randint(0, 37, (2, 17), device="cuda")
    labels[1, 4] = -100

    reference = deepseek_v4_liger_causal_lm_loss(
        hidden_reference,
        head,
        labels,
        hidden_size=256,
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
            hidden_size=256,
        )
        packed.backward()

    torch.testing.assert_close(packed, reference, rtol=1e-3, atol=1e-3)
    assert materializations == 0
    assert "torch_ggml_ops.mmq.default" in operations
    assert "torch_ggml_ops.mmq_grad_input.default" in operations
    assert torch.isfinite(_require_grad(hidden_packed)).all()
