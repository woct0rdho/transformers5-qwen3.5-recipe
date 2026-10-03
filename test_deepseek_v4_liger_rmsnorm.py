from typing import Any, cast

import pytest
import torch
from peft import LoraConfig, get_peft_model
from peft.tuners.lora.layer import LoraLayer
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4RMSNorm,
    DeepseekV4UnweightedRMSNorm,
)

from deepseek_v4_liger_rmsnorm import (
    _LigerFrozenWeightRMSNormFunction,
    configure_deepseek_v4_liger_rmsnorm,
)
from module_patching import PATCH_MARKER
from test_support import assert_close_mixed_precision, require_grad

_MIN_RMSNORM_COSINE = 0.999
_MAX_RMSNORM_RELATIVE_RMSE = 0.005


class _NormToy(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = DeepseekV4RMSNorm(width, eps=1e-6)
        self.q_b_norm = DeepseekV4UnweightedRMSNorm(eps=1e-6)
        self.attn_hc = torch.nn.Module()
        self.attn_hc.input_norm = DeepseekV4UnweightedRMSNorm(eps=1e-6)


def _frozen_pair(width: int) -> tuple[_NormToy, _NormToy]:
    reference = _NormToy(width).cuda()
    candidate = _NormToy(width).cuda()
    with torch.no_grad():
        reference.norm.weight.normal_(mean=1.0, std=0.2)
    candidate.load_state_dict(reference.state_dict())
    reference.requires_grad_(False)
    candidate.requires_grad_(False)
    return reference, candidate


@pytest.mark.parametrize("width", [128, 512, 1024, 4096])
def test_liger_frozen_weight_rmsnorm_meets_accuracy_gate(
    width: int,
) -> None:
    torch.manual_seed(1000 + width)
    reference, candidate = _frozen_pair(width)
    report = configure_deepseek_v4_liger_rmsnorm(candidate)
    assert report["weighted"] == 1
    assert report["q_b_unweighted"] == 1
    assert report["skipped_weighted_128"] == 0
    assert report["skipped_mhc_unweighted"] == 1
    assert report["backward"] == "in_place_frozen_dx_only"

    x = torch.randn(19, width, device="cuda", dtype=torch.bfloat16)
    reference_x = x.clone().requires_grad_()
    candidate_x = x.clone().requires_grad_()
    output_gradient = torch.randn_like(x)

    reference_output = reference.norm(reference_x)
    candidate_output = candidate.norm(candidate_x)
    reference_output.backward(output_gradient.clone())
    candidate_output.backward(output_gradient.clone())

    assert candidate_output.dtype == torch.bfloat16
    assert_close_mixed_precision(
        candidate_output,
        reference_output,
        minimum_cosine=_MIN_RMSNORM_COSINE,
        maximum_relative_rmse=_MAX_RMSNORM_RELATIVE_RMSE,
    )
    assert_close_mixed_precision(
        require_grad(candidate_x),
        require_grad(reference_x),
        minimum_cosine=_MIN_RMSNORM_COSINE,
        maximum_relative_rmse=_MAX_RMSNORM_RELATIVE_RMSE,
    )
    assert candidate.norm.weight.grad is None


def test_width_128_weighted_norm_uses_tuned_launch_geometry() -> None:
    _, candidate = _frozen_pair(128)
    report = configure_deepseek_v4_liger_rmsnorm(candidate)

    assert report["weighted"] == 1
    assert report["skipped_weighted_128"] == 0
    assert not candidate.norm.weight.requires_grad
    assert getattr(candidate.norm, PATCH_MARKER)


def test_in_place_backward_is_stable_on_branched_norm_output() -> None:
    torch.manual_seed(2026)
    width = 512
    reference, candidate = _frozen_pair(width)
    configure_deepseek_v4_liger_rmsnorm(candidate)
    projection = torch.randn(37, width, device="cuda", dtype=torch.bfloat16)
    scale = torch.randn(width, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(23, width, device="cuda", dtype=torch.bfloat16)
    reference_x = x.clone().requires_grad_()
    candidate_x = x.clone().requires_grad_()

    def branched_loss(module: _NormToy, inputs: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            normalized = module.norm(inputs)
            projected = normalized @ projection.transpose(0, 1)
            residual_branch = normalized * scale
        return (
            projected.float().square().mean() + residual_branch.float().square().mean()
        )

    reference_loss = branched_loss(reference, reference_x)
    candidate_loss = branched_loss(candidate, candidate_x)
    reference_loss.backward()
    candidate_loss.backward()

    assert_close_mixed_precision(
        candidate_loss,
        reference_loss,
        minimum_cosine=_MIN_RMSNORM_COSINE,
        maximum_relative_rmse=_MAX_RMSNORM_RELATIVE_RMSE,
    )
    assert_close_mixed_precision(
        require_grad(candidate_x),
        require_grad(reference_x),
        minimum_cosine=_MIN_RMSNORM_COSINE,
        maximum_relative_rmse=_MAX_RMSNORM_RELATIVE_RMSE,
    )


def test_q_b_scale_free_liger_norm_matches_strict_gradient() -> None:
    torch.manual_seed(77)
    reference, candidate = _frozen_pair(512)
    configure_deepseek_v4_liger_rmsnorm(candidate)
    x = torch.randn(2, 8, 64, 512, device="cuda", dtype=torch.bfloat16)
    reference_x = x.clone().requires_grad_()
    candidate_x = x.clone().requires_grad_()
    output_gradient = torch.randn_like(x)

    reference_output = reference.q_b_norm(reference_x)
    candidate_output = candidate.q_b_norm(candidate_x)
    reference_output.backward(output_gradient.clone())
    candidate_output.backward(output_gradient.clone())

    assert_close_mixed_precision(
        candidate_output,
        reference_output,
        minimum_cosine=0.99999,
        maximum_relative_rmse=0.005,
    )
    assert_close_mixed_precision(
        require_grad(candidate_x),
        require_grad(reference_x),
        minimum_cosine=0.99999,
        maximum_relative_rmse=0.005,
    )


def test_mhc_unweighted_norm_remains_authoritative() -> None:
    _, candidate = _frozen_pair(512)
    input_norm = cast(Any, candidate.attn_hc.input_norm)
    mhc_forward = input_norm.forward
    configure_deepseek_v4_liger_rmsnorm(candidate)
    assert input_norm.forward.__func__ is mhc_forward.__func__


def test_frozen_weight_function_returns_no_weight_gradient() -> None:
    torch.manual_seed(88)
    x = torch.randn(13, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(256, device="cuda", dtype=torch.float32, requires_grad=True)
    output = _LigerFrozenWeightRMSNormFunction.apply(x, weight, 1e-6)
    output.float().square().mean().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert weight.grad is None


def test_base_model_patch_survives_lora_injection() -> None:
    class Toy(torch.nn.Module):
        # `q_a_proj` is a LoRA layer once PEFT has injected the adapter, so the test only needs the
        # module surface here. `LoraLayer` below narrows it back to the adapter.
        q_a_proj: torch.nn.Module

        def __init__(self) -> None:
            super().__init__()
            self.q_a_proj = torch.nn.Linear(512, 512, bias=False)
            self.norm = DeepseekV4RMSNorm(512, eps=1e-6)
            self.q_b_norm = DeepseekV4UnweightedRMSNorm(eps=1e-6)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.norm(self.q_a_proj(x))

    base = Toy().cuda().to(torch.bfloat16)
    base.norm.float()
    report = configure_deepseek_v4_liger_rmsnorm(base)
    assert report["patched"] == 2
    model = get_peft_model(
        base,
        LoraConfig(target_modules=["q_a_proj"], r=4, lora_alpha=4),
        autocast_adapter_dtype=False,
    )
    x = torch.randn(7, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    model(x).float().square().mean().backward()

    patched_base = model.base_model.model
    assert isinstance(patched_base, Toy)
    assert getattr(patched_base.norm, PATCH_MARKER)
    assert patched_base.norm.weight.grad is None
    q_a_proj = patched_base.q_a_proj
    assert isinstance(q_a_proj, LoraLayer)
    lora_b = q_a_proj.lora_B["default"]
    assert isinstance(lora_b, torch.nn.Linear)
    assert lora_b.weight.grad is not None


def test_configuration_freezes_norm_weights_and_is_idempotent() -> None:
    model = _NormToy(512).cuda()
    assert model.norm.weight.requires_grad

    first = configure_deepseek_v4_liger_rmsnorm(model)
    second = configure_deepseek_v4_liger_rmsnorm(model)
    assert not model.norm.weight.requires_grad
    assert first["patched"] == 2
    assert second["patched"] == 0
    assert second["already_patched"] == 2
