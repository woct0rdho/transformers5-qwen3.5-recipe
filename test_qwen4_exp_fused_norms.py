"""Tests for the fused Qwen4-Exp norm kernels and their inventory."""

import types
from typing import Any

import pytest
import torch
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextRMSNorm,
    Qwen4ExpTextRMSNormGated,
)

from module_patching import PATCH_MARKER
from qwen4_exp_fused_norms import (
    _EXPECTED_GATED_RMSNORMS,
    _EXPECTED_GROUPED_RMSNORMS,
    _EXPECTED_RMSNORMS,
    _EXPECTED_SKIPPED_HYPER_CONNECTION_RMSNORMS,
    _GROUP,
    _GROUPED_WIDTH,
    _fla_gated_rmsnorm_forward,
    _grouped_rmsnorm_forward,
    _liger_rmsnorm_forward,
    configure_qwen4_exp_fused_norms,
    grouped_rms_norm_serves,
    require_complete_qwen4_exp_fused_norms,
)
from test_support import assert_close_mixed_precision, require_grad

SEQ = 2048
GROUPED_ROWS = 2048
_MIN_COSINE = 0.999
_MAX_RELATIVE_RMSE = 0.005


def _randomize(weight: torch.Tensor, generator: torch.Generator) -> None:
    with torch.no_grad():
        weight.normal_(mean=0.0, std=0.2, generator=generator)


def _patched_forward(module: torch.nn.Module) -> Any:
    forward = module.__dict__.get("forward")
    if not isinstance(forward, types.MethodType):
        raise TypeError("expected an instance-local patched forward")
    return forward.__func__


class _PlainNormToy(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = Qwen4ExpTextRMSNorm(width, eps=1e-6)
        self.other = torch.nn.LayerNorm(width)


class _GroupedNormToy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ple = _PleHolder()


class _GatedNormHolder(torch.nn.Module):
    def __init__(self, head_dim: int) -> None:
        super().__init__()
        self.norm = Qwen4ExpTextRMSNormGated(head_dim, eps=1e-6)


class _GatedNormToy(torch.nn.Module):
    def __init__(self, head_dim: int) -> None:
        super().__init__()
        self.linear_attn = _GatedNormHolder(head_dim)


class _InventoryToy(torch.nn.Module):
    """One holder per hyper-connection so the skipped sites carry the real name."""

    def __init__(self, rmsnorms: int, gated: int, hyper_connections: int) -> None:
        super().__init__()
        self.plain = torch.nn.ModuleList(
            [Qwen4ExpTextRMSNorm(256, eps=1e-6) for _ in range(rmsnorms)]
        )
        self.gated_norms = torch.nn.ModuleList(
            [Qwen4ExpTextRMSNormGated(128, eps=1e-6) for _ in range(gated)]
        )
        self.layers = torch.nn.ModuleList(
            [_HyperConnectionHolder() for _ in range(hyper_connections)]
        )
        self.ple = _PleHolder()
        self.untouched = torch.nn.LayerNorm(256)


class _HyperConnectionHolder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hc_norm = Qwen4ExpTextRMSNorm(10240, group_size=2560, eps=1e-6)


class _PleHolder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm_key = Qwen4ExpTextRMSNorm(_GROUPED_WIDTH, group_size=_GROUP, eps=1e-6)
        self.norm_query = Qwen4ExpTextRMSNorm(
            _GROUPED_WIDTH, group_size=_GROUP, eps=1e-6
        )
        self.norm_conv = Qwen4ExpTextRMSNorm(
            _GROUPED_WIDTH, group_size=_GROUP, eps=1e-6
        )


@pytest.mark.parametrize("width", [128, 256])
def test_liger_rmsnorm_matches_eager_reference(width: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(31415)
    reference = _PlainNormToy(width).cuda().to(torch.bfloat16)
    candidate = _PlainNormToy(width).cuda().to(torch.bfloat16)
    _randomize(reference.norm.weight, generator)
    candidate.load_state_dict(reference.state_dict())
    reference.requires_grad_(False)
    candidate.requires_grad_(False)

    report = configure_qwen4_exp_fused_norms(candidate)
    assert report["rmsnorms"] == 1
    assert report["gated_rmsnorms"] == 0
    assert _patched_forward(candidate.norm) is _liger_rmsnorm_forward

    reference_input = torch.randn(
        2, 16, width, generator=generator, device="cuda", dtype=torch.bfloat16
    ).requires_grad_(True)
    candidate_input = reference_input.detach().clone().requires_grad_(True)
    grad_output = torch.randn(
        2, 16, width, generator=generator, device="cuda", dtype=torch.bfloat16
    )

    reference_output = reference.norm(reference_input)
    candidate_output = candidate.norm(candidate_input)
    reference_output.backward(grad_output)
    candidate_output.backward(grad_output)

    assert_close_mixed_precision(
        candidate_output,
        reference_output,
        minimum_cosine=_MIN_COSINE,
        maximum_relative_rmse=_MAX_RELATIVE_RMSE,
    )
    assert_close_mixed_precision(
        require_grad(candidate_input),
        require_grad(reference_input),
        minimum_cosine=_MIN_COSINE,
        maximum_relative_rmse=_MAX_RELATIVE_RMSE,
    )
    assert candidate.norm.weight.grad is None
    assert reference.norm.weight.grad is None


def test_fla_gated_rmsnorm_matches_eager_reference() -> None:
    head_dim = 128
    generator = torch.Generator(device="cuda").manual_seed(2718)
    reference = _GatedNormToy(head_dim).cuda().to(torch.bfloat16)
    candidate = _GatedNormToy(head_dim).cuda().to(torch.bfloat16)
    _randomize(reference.linear_attn.norm.weight, generator)
    candidate.load_state_dict(reference.state_dict())
    reference.requires_grad_(False)
    candidate.requires_grad_(False)

    report = configure_qwen4_exp_fused_norms(candidate)
    assert report["gated_rmsnorms"] == 1
    assert report["rmsnorms"] == 0
    norm = candidate.linear_attn.norm
    assert _patched_forward(norm) is _fla_gated_rmsnorm_forward

    reference_input = torch.randn(
        512, head_dim, generator=generator, device="cuda", dtype=torch.bfloat16
    ).requires_grad_(True)
    candidate_input = reference_input.detach().clone().requires_grad_(True)
    reference_gate = torch.randn(
        512, head_dim, generator=generator, device="cuda", dtype=torch.bfloat16
    ).requires_grad_(True)
    candidate_gate = reference_gate.detach().clone().requires_grad_(True)
    grad_output = torch.randn(
        512, head_dim, generator=generator, device="cuda", dtype=torch.bfloat16
    )

    reference_output = reference.linear_attn.norm(reference_input, reference_gate)
    candidate_output = candidate.linear_attn.norm(candidate_input, candidate_gate)
    reference_output.backward(grad_output)
    candidate_output.backward(grad_output)

    assert_close_mixed_precision(
        candidate_output,
        reference_output,
        minimum_cosine=_MIN_COSINE,
        maximum_relative_rmse=_MAX_RELATIVE_RMSE,
    )
    assert_close_mixed_precision(
        require_grad(candidate_input),
        require_grad(reference_input),
        minimum_cosine=_MIN_COSINE,
        maximum_relative_rmse=_MAX_RELATIVE_RMSE,
    )
    assert_close_mixed_precision(
        require_grad(candidate_gate),
        require_grad(reference_gate),
        minimum_cosine=_MIN_COSINE,
        maximum_relative_rmse=_MAX_RELATIVE_RMSE,
    )
    assert norm.weight.grad is None
    assert reference.linear_attn.norm.weight.grad is None


def test_grouped_rmsnorm_matches_eager_reference_and_falls_back() -> None:
    generator = torch.Generator(device="cuda").manual_seed(1618)
    reference = _GroupedNormToy().cuda().to(torch.bfloat16)
    candidate = _GroupedNormToy().cuda().to(torch.bfloat16)
    _randomize(reference.ple.norm_key.weight, generator)
    candidate.load_state_dict(reference.state_dict())
    reference.requires_grad_(False)
    candidate.requires_grad_(False)

    report = configure_qwen4_exp_fused_norms(candidate)
    assert report["grouped_rmsnorms"] == _EXPECTED_GROUPED_RMSNORMS
    assert report["skipped_hyper_connection_rmsnorms"] == 0
    assert _patched_forward(candidate.ple.norm_key) is _grouped_rmsnorm_forward

    hidden = torch.randn(
        GROUPED_ROWS,
        _GROUPED_WIDTH,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    assert grouped_rms_norm_serves(
        hidden, candidate.ple.norm_key.weight, candidate.ple.norm_key.group_size
    )
    reference_input = hidden.detach().requires_grad_(True)
    candidate_input = hidden.detach().requires_grad_(True)
    grad_output = torch.randn(
        GROUPED_ROWS,
        _GROUPED_WIDTH,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    reference_output = reference.ple.norm_key(reference_input)
    candidate_output = candidate.ple.norm_key(candidate_input)
    reference_output.backward(grad_output)
    candidate_output.backward(grad_output)
    assert_close_mixed_precision(
        candidate_output,
        reference_output,
        minimum_cosine=_MIN_COSINE,
        maximum_relative_rmse=_MAX_RELATIVE_RMSE,
    )
    assert_close_mixed_precision(
        require_grad(candidate_input),
        require_grad(reference_input),
        minimum_cosine=_MIN_COSINE,
        maximum_relative_rmse=_MAX_RELATIVE_RMSE,
    )

    # Outside the kernel's contract the site falls back to the module's own forward.
    assert not grouped_rms_norm_serves(
        hidden.float(), candidate.ple.norm_key.weight, candidate.ple.norm_key.group_size
    )
    assert not grouped_rms_norm_serves(hidden, candidate.ple.norm_key.weight, None)
    assert not grouped_rms_norm_serves(
        hidden, candidate.ple.norm_key.weight.detach().requires_grad_(True), _GROUP
    )
    assert not grouped_rms_norm_serves(
        hidden[:16], candidate.ple.norm_key.weight, _GROUP
    )
    float_input = hidden.float().detach().requires_grad_(True)
    assert_close_mixed_precision(
        candidate.ple.norm_key(float_input),
        reference.ple.norm_key(float_input.detach()),
        minimum_cosine=_MIN_COSINE,
        maximum_relative_rmse=_MAX_RELATIVE_RMSE,
    )


def test_patched_full_inventory_is_required_and_idempotent() -> None:
    model = (
        _InventoryToy(
            _EXPECTED_RMSNORMS,
            _EXPECTED_GATED_RMSNORMS,
            _EXPECTED_SKIPPED_HYPER_CONNECTION_RMSNORMS,
        )
        .cuda()
        .to(torch.bfloat16)
    )
    report = configure_qwen4_exp_fused_norms(model)
    require_complete_qwen4_exp_fused_norms(report)
    assert report["patched"] == (
        _EXPECTED_RMSNORMS + _EXPECTED_GATED_RMSNORMS + _EXPECTED_GROUPED_RMSNORMS
    )
    assert report["already_patched"] == 0
    assert not getattr(model.untouched, PATCH_MARKER, False)
    assert not getattr(model.layers[0].hc_norm, PATCH_MARKER, False)

    again = configure_qwen4_exp_fused_norms(model)
    require_complete_qwen4_exp_fused_norms(again)
    assert again["patched"] == 0
    assert again["already_patched"] == report["patched"]


def test_incomplete_inventory_is_rejected() -> None:
    model = (
        _InventoryToy(
            _EXPECTED_RMSNORMS - 1,
            _EXPECTED_GATED_RMSNORMS,
            _EXPECTED_SKIPPED_HYPER_CONNECTION_RMSNORMS,
        )
        .cuda()
        .to(torch.bfloat16)
    )
    report = configure_qwen4_exp_fused_norms(model)
    with pytest.raises(RuntimeError, match="incomplete Qwen4-Exp fused norm"):
        require_complete_qwen4_exp_fused_norms(report)


def test_unsupported_activation_is_rejected() -> None:
    model = _GatedNormToy(128).cuda().to(torch.bfloat16)
    model.linear_attn.norm.activation = "gelu"
    with pytest.raises(RuntimeError, match="unsupported activation"):
        configure_qwen4_exp_fused_norms(model)


def test_cpu_norm_is_rejected() -> None:
    model = _PlainNormToy(64)
    with pytest.raises(RuntimeError, match="requires a CUDA/ROCm weight"):
        configure_qwen4_exp_fused_norms(model)


def test_gated_norm_requires_its_gate() -> None:
    model = _GatedNormToy(128).cuda().to(torch.bfloat16)
    configure_qwen4_exp_fused_norms(model)
    hidden = torch.randn(8, 128, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="requires its gate tensor"):
        model.linear_attn.norm(hidden)
