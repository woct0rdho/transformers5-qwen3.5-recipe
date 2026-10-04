"""Tests for the fused grouped RMSNorm of the Qwen4-Exp hyper-connections."""

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextGatedResidual,
)

from qwen4_exp_liger_hc import (
    _EXPECTED_CONNECTIONS,
    _MARKER,
    _REFERENCE_ATTRIBUTE,
    configure_qwen4_exp_hc_norm,
    grouped_rms_norm,
    hc_norm_serves,
    require_complete_qwen4_exp_hc_norm,
)

SEQ = 2048
HIDDEN = 2560
HC = 4
FLAT = HC * HIDDEN
LOWRANK = 320


def _connection(weight_scale: float = 1.0) -> Qwen4ExpTextGatedResidual:
    config = cast(
        "Any",
        SimpleNamespace(
            hc_count=HC, hidden_size=HIDDEN, hc_lowrank=LOWRANK, rms_norm_eps=1e-6
        ),
    )
    module = Qwen4ExpTextGatedResidual(config).to(device="cuda", dtype=torch.bfloat16)
    generator = torch.Generator(device="cpu").manual_seed(3)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(
                (
                    torch.randn(
                        *parameter.shape, generator=generator, dtype=torch.float32
                    )
                    * weight_scale
                ).to(dtype=torch.bfloat16)
            )
    module.hc_norm.weight.requires_grad_(False)
    return module


def _inputs(rows: int = SEQ):
    generator = torch.Generator(device="cpu").manual_seed(5)
    hyper_input = torch.randn(rows, FLAT, generator=generator, dtype=torch.float32).to(
        device="cuda", dtype=torch.bfloat16
    )
    cotangent = torch.randn(rows, FLAT, generator=generator, dtype=torch.float32).to(
        device="cuda", dtype=torch.bfloat16
    )
    return hyper_input, cotangent


def test_fused_norm_matches_the_module_on_the_forward() -> None:
    module = _connection()
    hyper_input, _ = _inputs()
    reference = module.hc_norm(hyper_input)
    fused = grouped_rms_norm(hyper_input, module.hc_norm.weight, 1e-6, HIDDEN)
    # Relative, because a bf16 ulp at the larger outputs is well above any sane absolute tolerance.
    difference = fused.float() - reference.float()
    rmse = float(
        difference.square().mean().sqrt()
        / (reference.float().square().mean().sqrt() + 1e-12)
    )
    assert rmse <= 5e-3, rmse


@pytest.mark.parametrize("weight_scale", [0.1, 1.0, 3.0])
def test_fused_backward_is_within_the_bf16_floor_of_fp64(weight_scale: float) -> None:
    # The target is fp64 rather than the reference, because both the kernel and the module sit on the
    # bf16 store floor and the reference cannot tell a kernel error from its own arithmetic.
    module = _connection(weight_scale)
    hyper_input, cotangent = _inputs()
    leaf = hyper_input.detach().requires_grad_(True)
    grouped_rms_norm(leaf, module.hc_norm.weight, 1e-6, HIDDEN).backward(cotangent)

    # Autograd on the module's own expression, in fp64, so the truth does not repeat the kernel's
    # algebra: the scale vector is free within a group, and a transcription of the backward that
    # leaves it out of the mean agrees with itself.
    x64 = hyper_input.double().view(SEQ, -1, HIDDEN).requires_grad_(True)
    w64 = 1.0 + module.hc_norm.weight.double().view(-1, HIDDEN)
    y64 = x64 * torch.rsqrt(x64.pow(2).mean(-1, keepdim=True) + 1e-6) * w64
    y64.backward(cotangent.double().view(SEQ, -1, HIDDEN))
    truth = x64.grad.view(SEQ, FLAT)

    difference = leaf.grad.float().flatten() - truth.float().flatten()
    rmse = float(
        difference.square().mean().sqrt()
        / (truth.float().square().mean().sqrt() + 1e-12)
    )
    assert rmse <= 5e-3, rmse


def test_configure_reports_the_connection_and_installs_the_patch() -> None:
    module = _connection()

    class Container(torch.nn.Module):
        def __init__(self, connection: Qwen4ExpTextGatedResidual) -> None:
            super().__init__()
            self.connection = connection

    container = Container(module)
    report = configure_qwen4_exp_hc_norm(container)
    require_complete_qwen4_exp_hc_norm(report, expected_connections=1)
    assert report["hyper_connections"] == 1
    assert report["group_size"] == HIDDEN
    assert getattr(module, _MARKER, False)
    assert hasattr(module, _REFERENCE_ATTRIBUTE)
    assert module.forward.__name__ == "_qwen4_hc_forward"
    # The kept reference is the module's own forward, not the patched one.
    assert getattr(module, _REFERENCE_ATTRIBUTE).__name__ == "forward"


def test_require_fails_closed_on_the_wrong_count() -> None:
    module = _connection()

    class Container(torch.nn.Module):
        def __init__(self, connection: Qwen4ExpTextGatedResidual) -> None:
            super().__init__()
            self.connection = connection

    report = configure_qwen4_exp_hc_norm(Container(module))
    with pytest.raises(RuntimeError):
        require_complete_qwen4_exp_hc_norm(report)
    assert _EXPECTED_CONNECTIONS == 97


def test_predicate_rejects_shapes_dtypes_and_trainable_weights() -> None:
    module = _connection()
    hyper_input, _ = _inputs()
    assert hc_norm_serves(module, hyper_input)
    assert not hc_norm_serves(module, hyper_input.float())
    assert not hc_norm_serves(module, hyper_input[:, :HIDDEN])
    assert not hc_norm_serves(
        module, hyper_input.view(1, SEQ, FLAT)[:, :, :].expand(2, SEQ, FLAT)
    )
    module.hc_norm.weight.requires_grad_(True)
    assert not hc_norm_serves(module, hyper_input)


def test_fallback_runs_the_reference_when_the_predicate_rejects() -> None:
    module = _connection()
    hyper_input, _ = _inputs()
    reference = module(hyper_input)
    assert isinstance(reference, tuple)
    module.hc_norm.weight.requires_grad_(True)
    with torch.no_grad():
        fallback = module(hyper_input)
    assert isinstance(fallback, tuple)
    assert torch.equal(cast("Any", fallback)[0], reference[0])
