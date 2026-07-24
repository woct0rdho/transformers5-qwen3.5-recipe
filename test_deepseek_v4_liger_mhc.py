from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.checkpoint import checkpoint
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4DecoderLayer,
    DeepseekV4HyperConnection,
    DeepseekV4HyperHead,
)

from deepseek_v4_liger_mhc import (
    configure_deepseek_v4_liger_mhc,
    deepseek_v4_mhc_fn_cache,
    deepseek_v4_mhc_head,
    deepseek_v4_mhc_merge,
    deepseek_v4_mhc_prepare,
    require_complete_deepseek_v4_liger_mhc,
)

_HC = 4
_HIDDEN = 4096
_FLAT = _HC * _HIDDEN
_MIX = 24
_EPS = 1e-6


def _controls(seed: int = 2026):
    torch.manual_seed(seed)
    # The checkpoint stores layer mHC fn tensors as F16. Construct the FP32
    # reference from that source so the optimized cache introduces no rounding.
    fn_source = (
        torch.randn(_MIX, _FLAT, device="cuda", dtype=torch.float32) * 1e-4
    ).to(torch.float16)
    fn_reference = fn_source.float()
    base = (torch.randn(_MIX, device="cuda") * 0.1).float().contiguous()
    scale = (torch.randn(3, device="cuda") * 0.1).float().contiguous()
    return fn_source, fn_reference, base, scale


def _candidate(
    x: torch.Tensor,
    branch: torch.Tensor,
    fn: torch.Tensor,
    base: torch.Tensor,
    scale: torch.Tensor,
):
    residual, coefficients, collapsed = deepseek_v4_mhc_prepare(x, fn, base, scale)
    return collapsed, deepseek_v4_mhc_merge(residual, branch, coefficients)


def _head_reference(
    x: torch.Tensor,
    fn: torch.Tensor,
    base: torch.Tensor,
    scale: torch.Tensor,
):
    rows = x.numel() // _FLAT
    flat = x.reshape(rows, _FLAT).float()
    invr = torch.rsqrt(flat.square().mean(dim=-1, keepdim=True) + _EPS)
    mix = F.linear(flat, fn.float()) * invr
    pre = torch.sigmoid(mix * scale.float() + base.float()) + _EPS
    return (
        (pre.unsqueeze(-1) * x.reshape(rows, _HC, _HIDDEN).float())
        .sum(dim=1)
        .to(x.dtype)
    )


def _reference(
    x: torch.Tensor,
    branch: torch.Tensor,
    fn: torch.Tensor,
    base: torch.Tensor,
    scale: torch.Tensor,
):
    rows = x.numel() // _FLAT
    flat = x.reshape(rows, _FLAT).float()
    invr = torch.rsqrt(flat.square().mean(dim=-1, keepdim=True) + _EPS)
    mix = F.linear(flat, fn) * invr
    pre = torch.sigmoid(mix[:, :_HC] * scale[0] + base[:_HC]) + _EPS
    post = 2.0 * torch.sigmoid(mix[:, _HC : 2 * _HC] * scale[1] + base[_HC : 2 * _HC])
    comb = (
        torch.softmax(
            (mix[:, 2 * _HC :] * scale[2] + base[2 * _HC :]).view(rows, _HC, _HC),
            dim=-1,
        )
        + _EPS
    )
    comb = comb / (comb.sum(dim=-2, keepdim=True) + _EPS)
    for _ in range(19):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + _EPS)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + _EPS)
    collapsed = (pre.unsqueeze(-1) * x.reshape(rows, _HC, _HIDDEN).float()).sum(dim=1)
    merged = post.unsqueeze(-1) * branch.reshape(rows, _HIDDEN).float().unsqueeze(
        1
    ) + torch.matmul(comb.transpose(-1, -2), x.reshape(rows, _HC, _HIDDEN).float())
    return collapsed.to(x.dtype).view(*x.shape[:-2], _HIDDEN), merged.to(
        x.dtype
    ).view_as(x)


def _metrics(candidate: torch.Tensor, reference: torch.Tensor):
    candidate = candidate.detach().float().flatten()
    reference = reference.detach().float().flatten()
    delta = candidate - reference
    return (
        float(F.cosine_similarity(candidate, reference, dim=0)),
        float(
            delta.square().mean().sqrt() / (reference.square().mean().sqrt() + 1e-12)
        ),
    )


def _assert_mixed_close(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    *,
    minimum_cosine: float,
    maximum_relative_rmse: float,
):
    cosine, relative_rmse = _metrics(candidate, reference)
    assert cosine >= minimum_cosine
    assert relative_rmse <= maximum_relative_rmse


def test_fused_mhc_matches_transformers_forward_and_activation_gradients() -> None:
    torch.manual_seed(17)
    rows = 2048
    fn, fn_reference, base, scale = _controls()
    x_data = torch.randn(rows, _HC, _HIDDEN, device="cuda", dtype=torch.bfloat16)
    branch_data = torch.randn(rows, _HIDDEN, device="cuda", dtype=torch.bfloat16)
    grad_collapsed = torch.randn_like(branch_data)
    grad_merged = torch.randn_like(x_data)

    reference_x = x_data.clone().requires_grad_()
    reference_branch = branch_data.clone().requires_grad_()
    reference_collapsed, reference_merged = _reference(
        reference_x, reference_branch, fn_reference, base, scale
    )
    torch.autograd.backward(
        (reference_collapsed, reference_merged),
        (grad_collapsed, grad_merged),
    )

    candidate_x = x_data.clone().requires_grad_()
    candidate_branch = branch_data.clone().requires_grad_()
    candidate_collapsed, candidate_merged = _candidate(
        candidate_x, candidate_branch, fn, base, scale
    )
    candidate_merged_cotangent = grad_merged.clone()
    torch.autograd.backward(
        (candidate_collapsed, candidate_merged),
        (grad_collapsed, candidate_merged_cotangent),
    )

    _assert_mixed_close(
        candidate_collapsed,
        reference_collapsed,
        minimum_cosine=0.999999,
        maximum_relative_rmse=1e-4,
    )
    _assert_mixed_close(
        candidate_merged,
        reference_merged,
        minimum_cosine=0.999999,
        maximum_relative_rmse=1e-4,
    )
    _assert_mixed_close(
        candidate_x.grad,
        reference_x.grad,
        minimum_cosine=0.9999,
        maximum_relative_rmse=0.01,
    )
    _assert_mixed_close(
        candidate_branch.grad,
        reference_branch.grad,
        minimum_cosine=0.999999,
        maximum_relative_rmse=1e-4,
    )
    assert (
        candidate_x.grad.untyped_storage().data_ptr()
        == candidate_merged_cotangent.untyped_storage().data_ptr()
    )
    assert fn.grad is None
    assert base.grad is None
    assert scale.grad is None


def test_fused_mhc_runs_exact_batch_1_4_16_geometries() -> None:
    fn, _, base, scale = _controls(44)
    for rows in (2048, 8192, 32768):
        x = torch.randn(
            rows, _HC, _HIDDEN, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        branch = torch.randn(
            rows, _HIDDEN, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        collapsed, merged = _candidate(x, branch, fn, base, scale)
        torch.autograd.backward(
            (collapsed, merged),
            (torch.ones_like(collapsed), torch.ones_like(merged)),
        )
        assert collapsed.shape == (rows, _HIDDEN)
        assert merged.shape == (rows, _HC, _HIDDEN)
        assert torch.isfinite(x.grad).all()
        assert torch.isfinite(branch.grad).all()
        del x, branch, collapsed, merged


def test_prepare_returns_residual_alias_and_compact_saved_state() -> None:
    rows = 2048
    fn, _, base, scale = _controls(55)
    x = torch.randn(
        1, rows, _HC, _HIDDEN, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    saved = []

    def pack(tensor):
        saved.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        residual, coefficients, collapsed = deepseek_v4_mhc_prepare(x, fn, base, scale)

    assert residual.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
    assert coefficients.shape == (1, rows, _MIX)
    assert collapsed.shape == (1, rows, _HIDDEN)
    assert any(
        tensor.shape == (rows, 39, _HC) and tensor.dtype == torch.float16
        for tensor in saved
    )
    assert all(tensor.shape != (rows, 20, _HC, _HC) for tensor in saved)
    unique = {}
    for tensor in saved:
        storage = tensor.untyped_storage()
        unique[storage.data_ptr()] = max(
            unique.get(storage.data_ptr(), 0), storage.nbytes()
        )
    assert sum(unique.values()) < 83 * 1024 * 1024


def test_fn_cache_preserves_native_f16_without_copy_or_transpose() -> None:
    fn, fn_reference, _, _ = _controls(66)
    cached = deepseek_v4_mhc_fn_cache(fn)
    converted = deepseek_v4_mhc_fn_cache(fn_reference)

    assert cached is fn
    assert converted.dtype == torch.float16
    assert converted.shape == (_MIX, _FLAT)
    assert converted.is_contiguous()
    torch.testing.assert_close(converted, fn, rtol=0, atol=0)


def test_mhc_fails_closed_for_unsupported_rows_and_trainable_controls() -> None:
    fn, _, base, scale = _controls(77)
    unsupported = torch.randn(17, _HC, _HIDDEN, device="cuda", dtype=torch.bfloat16)
    branch = torch.randn(17, _HIDDEN, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="supports rows"):
        _candidate(unsupported, branch, fn, base, scale)

    trainable_fn = fn.detach().requires_grad_()
    supported = torch.randn(2048, _HC, _HIDDEN, device="cuda", dtype=torch.bfloat16)
    supported_branch = torch.randn(2048, _HIDDEN, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="must be frozen"):
        _candidate(supported, supported_branch, trainable_fn, base, scale)

    with pytest.raises(TypeError, match="fn has unsupported dtype"):
        deepseek_v4_mhc_prepare(supported, fn.bfloat16(), base, scale)


def test_mhc_head_matches_transformers_forward_and_activation_gradient() -> None:
    torch.manual_seed(818)
    rows = 2048
    fn = (torch.randn(_HC, _FLAT, device="cuda") * 1e-4).half()
    base = (torch.randn(_HC, device="cuda") * 0.1).float()
    scale = (torch.randn(1, device="cuda") * 0.1).float()
    x_data = torch.randn(rows, _HC, _HIDDEN, device="cuda", dtype=torch.bfloat16)
    grad_output = torch.randn(rows, _HIDDEN, device="cuda", dtype=torch.bfloat16)

    reference_x = x_data.clone().requires_grad_()
    reference_output = _head_reference(reference_x, fn, base, scale)
    reference_output.backward(grad_output)
    candidate_x = x_data.clone().requires_grad_()
    candidate_output = deepseek_v4_mhc_head(candidate_x, fn, base, scale)
    candidate_output.backward(grad_output)

    _assert_mixed_close(
        candidate_output,
        reference_output,
        minimum_cosine=0.999999,
        maximum_relative_rmse=1e-4,
    )
    _assert_mixed_close(
        candidate_x.grad,
        reference_x.grad,
        minimum_cosine=0.9999,
        maximum_relative_rmse=0.01,
    )


def test_mhc_head_runs_exact_batch_1_4_16_geometries() -> None:
    torch.manual_seed(919)
    fn = (torch.randn(_HC, _FLAT, device="cuda") * 1e-4).half()
    base = torch.randn(_HC, device="cuda").float()
    scale = torch.randn(1, device="cuda").float()
    for rows in (2048, 8192, 32768):
        x = torch.randn(
            rows, _HC, _HIDDEN, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        output = deepseek_v4_mhc_head(x, fn, base, scale)
        output.backward(torch.ones_like(output))
        assert output.shape == (rows, _HIDDEN)
        assert torch.isfinite(x.grad).all()
        del x, output


def test_mhc_boundaries_survive_non_reentrant_checkpoint() -> None:
    fn, _, base, scale = _controls(1020)
    head_fn = (torch.randn(_HC, _FLAT, device="cuda") * 1e-4).half()
    head_base = torch.randn(_HC, device="cuda").float()
    head_scale = torch.randn(1, device="cuda").float()
    x = torch.randn(
        2048, _HC, _HIDDEN, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    branch = torch.randn(
        2048, _HIDDEN, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )

    def boundary(streams: torch.Tensor, branch_output: torch.Tensor) -> torch.Tensor:
        collapsed, merged = _candidate(streams, branch_output, fn, base, scale)
        return collapsed + deepseek_v4_mhc_head(merged, head_fn, head_base, head_scale)

    output = checkpoint(boundary, x, branch, use_reentrant=False)
    output.float().square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert branch.grad is not None and torch.isfinite(branch.grad).all()


class _ToyAttention(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, **_kwargs):
        return hidden_states * 0.25, None


class _ToyMLP(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del input_ids
        return hidden_states * 0.5


class _MHCIntegrationToy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        config = SimpleNamespace(
            hc_mult=_HC,
            hc_sinkhorn_iters=20,
            hc_eps=_EPS,
            rms_norm_eps=_EPS,
            hidden_size=_HIDDEN,
        )
        layer = DeepseekV4DecoderLayer.__new__(DeepseekV4DecoderLayer)
        torch.nn.Module.__init__(layer)
        layer.layer_idx = 0
        layer.self_attn = _ToyAttention()
        layer.mlp = _ToyMLP()
        layer.input_layernorm = torch.nn.Identity()
        layer.post_attention_layernorm = torch.nn.Identity()
        layer.attn_hc = DeepseekV4HyperConnection(config)
        layer.ffn_hc = DeepseekV4HyperConnection(config)
        self.layer = layer
        self.hc_head = DeepseekV4HyperHead(config)
        self.q_a_proj = torch.nn.Linear(4, 4, bias=False)


def test_model_configurator_is_idempotent_and_survives_peft() -> None:
    model = _MHCIntegrationToy().cuda()
    state_keys = set(model.state_dict())
    first = configure_deepseek_v4_liger_mhc(model)
    second = configure_deepseek_v4_liger_mhc(model)

    assert first["connections"] == 2
    assert first["decoder_layers"] == 1
    assert first["heads"] == 1
    assert first["f16_projection_parameters"] == 3
    assert first["converted_projection_parameters"] == 3
    assert first["frozen_control_parameters"] == 9
    assert first["patched"] == 4
    assert second["patched"] == 0
    assert second["already_patched"] == 4
    assert set(model.state_dict()) == state_keys
    assert model.layer.attn_hc.fn.dtype == torch.float16
    assert model.layer.ffn_hc.fn.dtype == torch.float16
    assert model.hc_head.hc_fn.dtype == torch.float16
    with pytest.raises(RuntimeError, match="incomplete DeepSeek V4 mHC"):
        require_complete_deepseek_v4_liger_mhc(first)

    wrapped = get_peft_model(
        model,
        LoraConfig(target_modules={"q_a_proj"}, r=2, lora_alpha=2),
        autocast_adapter_dtype=False,
    )
    patched = wrapped.base_model.model
    assert getattr(patched.layer, "_deepseek_v4_liger_mhc")
    assert getattr(patched.layer.attn_hc, "_deepseek_v4_liger_mhc")
    assert getattr(patched.hc_head, "_deepseek_v4_liger_mhc")
