"""Correctness tests for the dense QSA forward kernel.

The oracle is the blockwise FP32 reference in the module, and the thresholds follow the DeepSeek
attention plans: relative RMSE and cosine on the output, plus the FP32 LSE that the backward will
consume.
"""

import math

import pytest
import torch

from qwen4_exp_qsa_attention import (
    _BACKWARD_CONFIGS,
    _FORWARD_CONFIGS,
    _GROUP_SIZE,
    _HEAD_DIM,
    _KV_HEADS,
    _QUERY_FEATURES,
    _QUERY_HEADS,
    _SEQUENCE_LENGTH,
    _qsa_backward,
    _qsa_forward,
    _reference_qsa_attention_fp32,
    qwen4_exp_qsa_attention,
    qwen4_exp_qsa_attention_autograd,
    qwen4_exp_qsa_attention_configuration,
)

_OUTPUT_RMSE_LIMIT = 0.0025
_COSINE_FLOOR = 0.99999
_LSE_RMSE_LIMIT = 1e-4


def _inputs(batch: int, seed: int = 7, key_end: torch.Tensor | None = None):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    query = torch.randn(
        batch,
        _QUERY_HEADS,
        _SEQUENCE_LENGTH,
        _HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    key = torch.randn(
        batch,
        _KV_HEADS,
        _SEQUENCE_LENGTH,
        _HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    value = torch.randn(
        batch,
        _KV_HEADS,
        _SEQUENCE_LENGTH,
        _HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    gate = torch.randn(
        batch,
        _SEQUENCE_LENGTH,
        _QUERY_FEATURES,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    if key_end is None:
        key_end = torch.full(
            (batch,), _SEQUENCE_LENGTH, device="cuda", dtype=torch.int32
        )
    return query, key, value, gate, key_end


def _relative_metrics(
    candidate: torch.Tensor, reference: torch.Tensor
) -> tuple[float, float]:
    left = candidate.detach().float().flatten()
    right = reference.detach().float().flatten()
    difference = left - right
    rmse = float(
        difference.square().mean().sqrt() / (right.square().mean().sqrt() + 1e-12)
    )
    cosine = float(torch.nn.functional.cosine_similarity(left, right, dim=0))
    return rmse, cosine


def _check(
    batch: int, key_end: torch.Tensor, apply_gate: bool = False, row_indices=None
) -> None:
    query, key, value, full_gate, _ = _inputs(batch, key_end=key_end)
    gate = full_gate if apply_gate else None
    output, lse = qwen4_exp_qsa_attention(query, key, value, gate=gate, key_end=key_end)
    reference, reference_lse = _reference_qsa_attention_fp32(
        query, key, value, gate=gate, key_end=key_end, row_indices=row_indices
    )
    rows = row_indices if row_indices is not None else slice(None)
    rmse, cosine = _relative_metrics(output[:, rows], reference[:, rows])
    lse_rmse, _ = _relative_metrics(lse[..., rows], reference_lse[..., rows])
    assert rmse <= _OUTPUT_RMSE_LIMIT, f"output RMSE {rmse}"
    assert cosine >= _COSINE_FLOOR, f"output cosine {cosine}"
    assert lse_rmse <= _LSE_RMSE_LIMIT, f"LSE RMSE {lse_rmse}"
    assert not torch.isnan(output).any() and not torch.isnan(lse).any()


def test_qsa_forward_matches_reference_at_batch_one() -> None:
    key_end = torch.full((1,), _SEQUENCE_LENGTH, device="cuda", dtype=torch.int32)
    _check(1, key_end=key_end)


def test_qsa_forward_matches_reference_with_gate() -> None:
    key_end = torch.full((1,), _SEQUENCE_LENGTH, device="cuda", dtype=torch.int32)
    rows = torch.cat(
        [
            torch.arange(0, 256, device="cuda"),
            torch.arange(_SEQUENCE_LENGTH - 256, _SEQUENCE_LENGTH, device="cuda"),
        ]
    )
    _check(1, key_end=key_end, apply_gate=True, row_indices=rows)


@pytest.mark.parametrize("key_end", [7, 1234, 2000])
def test_qsa_forward_matches_reference_with_padding(key_end: int) -> None:
    rows = torch.cat(
        [
            torch.arange(0, 256, device="cuda"),
            torch.arange(_SEQUENCE_LENGTH - 256, _SEQUENCE_LENGTH, device="cuda"),
        ]
    )
    ending = torch.full((1,), key_end, device="cuda", dtype=torch.int32)
    _check(1, key_end=ending, row_indices=rows)


def test_qsa_forward_covers_every_configured_batch() -> None:
    rows = torch.cat(
        [
            torch.arange(0, 128, device="cuda"),
            torch.arange(_SEQUENCE_LENGTH - 128, _SEQUENCE_LENGTH, device="cuda"),
        ]
    )
    for batch in sorted(_FORWARD_CONFIGS):
        key_end = torch.full(
            (batch,), _SEQUENCE_LENGTH, device="cuda", dtype=torch.int32
        )
        _check(batch, key_end=key_end, row_indices=rows)


def test_qsa_forward_rejects_unsupported_inputs() -> None:
    key_end = torch.full((1,), _SEQUENCE_LENGTH, device="cuda", dtype=torch.int32)
    query, key, value, gate, _ = _inputs(1, key_end=key_end)

    with pytest.raises(ValueError, match="batch 2"):
        qwen4_exp_qsa_attention(
            query.expand(2, -1, -1, -1).contiguous(),
            key.expand(2, -1, -1, -1).contiguous(),
            value.expand(2, -1, -1, -1).contiguous(),
            key_end=key_end.expand(2).contiguous().int(),
        )
    with pytest.raises(ValueError, match="bfloat16"):
        qwen4_exp_qsa_attention(query.float(), key, value, key_end=key_end)
    wider = torch.randn(
        1,
        _QUERY_HEADS,
        _SEQUENCE_LENGTH,
        _HEAD_DIM * 2,
        dtype=torch.bfloat16,
        device="cuda",
    )
    with pytest.raises(ValueError, match="contiguous"):
        qwen4_exp_qsa_attention(wider[..., :_HEAD_DIM], key, value, key_end=key_end)
    with pytest.raises(ValueError, match="key_end"):
        qwen4_exp_qsa_attention(
            query,
            key,
            value,
            key_end=torch.full(
                (1,), _SEQUENCE_LENGTH + 1, device="cuda", dtype=torch.int32
            ),
        )
    with pytest.raises(ValueError, match="gate"):
        qwen4_exp_qsa_attention(
            query, key, value, gate=gate[:, : _SEQUENCE_LENGTH // 2], key_end=key_end
        )
    with pytest.raises(ValueError, match="query must be"):
        qwen4_exp_qsa_attention(
            query[:, :, : _SEQUENCE_LENGTH // 2].contiguous(),
            key,
            value,
            key_end=key_end,
        )


def test_qsa_attention_configuration_is_complete() -> None:
    configuration = qwen4_exp_qsa_attention_configuration()
    assert configuration["sequence_length"] == _SEQUENCE_LENGTH
    assert configuration["query_heads"] == _QUERY_HEADS
    assert configuration["kv_heads"] == _KV_HEADS
    assert configuration["head_dim"] == _HEAD_DIM
    assert configuration["group_size"] == _GROUP_SIZE
    assert math.isclose(configuration["scale"], 1.0 / math.sqrt(_HEAD_DIM))
    assert sorted(configuration["configs"]) == [1, 4, 16]
    for batch, config in configuration["configs"].items():
        assert _GROUP_SIZE % config["head_group"] == 0, batch
        assert config["block_m"] * config["head_group"] <= 64, batch
        assert _SEQUENCE_LENGTH % config["block_m"] == 0, batch


def _eager_attention(query, key, value, key_end):
    """Differentiable eager reference, the pattern the model uses outside the QSA layer."""
    batch, heads, sequence, head_dim = query.shape
    repeated_key = key[:, :, None].expand(
        batch, _KV_HEADS, _GROUP_SIZE, sequence, head_dim
    )
    repeated_value = value[:, :, None].expand(
        batch, _KV_HEADS, _GROUP_SIZE, sequence, head_dim
    )
    repeated_key = repeated_key.reshape(batch, heads, sequence, head_dim)
    repeated_value = repeated_value.reshape(batch, heads, sequence, head_dim)
    scores = torch.matmul(query.float(), repeated_key.float().transpose(-1, -2)) * (
        1.0 / math.sqrt(head_dim)
    )
    positions = torch.arange(sequence, device=query.device)
    visible = (positions[None, :] <= positions[:, None])[None, None, :, :]
    visible = visible & (positions[None, :] < key_end[:, None, None])[:, None, :, :]
    scores = scores.masked_fill(~visible, float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    return (
        torch.matmul(probabilities, repeated_value.float())
        .transpose(1, 2)
        .reshape(batch, sequence, heads * head_dim)
    )


def _gradients(module, query, key, value, key_end, grad_output):
    dquery, dkey, dvalue = torch.autograd.grad(
        module, (query, key, value), grad_output, retain_graph=True
    )
    return dquery, dkey, dvalue


def _check_gradients(batch: int, key_end_value: int = _SEQUENCE_LENGTH) -> None:
    key_end = torch.full((batch,), key_end_value, device="cuda", dtype=torch.int32)
    query, key, value, _, _ = _inputs(batch, seed=11, key_end=key_end)
    grad_output = torch.randn(
        batch, _SEQUENCE_LENGTH, _QUERY_FEATURES, dtype=torch.bfloat16, device="cuda"
    ) / math.sqrt(_HEAD_DIM)
    # The collator masks the padded rows' labels, so their incoming gradient is exactly zero, which
    # is the contract the backward's pad handling assumes.
    grad_output[:, key_end_value:] = 0
    kernel_inputs = (
        query.clone().requires_grad_(True),
        key.clone().requires_grad_(True),
        value.clone().requires_grad_(True),
    )
    reference_inputs = (
        query.clone().requires_grad_(True),
        key.clone().requires_grad_(True),
        value.clone().requires_grad_(True),
    )
    kernel_grads = _gradients(
        qwen4_exp_qsa_attention_autograd(*kernel_inputs),
        *kernel_inputs,
        key_end,
        grad_output,
    )
    reference_grads = _gradients(
        _eager_attention(*reference_inputs, key_end),
        *reference_inputs,
        key_end,
        grad_output.float(),
    )
    for name, kernel_grad, reference_grad in zip(
        ("dQ", "dK", "dV"), kernel_grads, reference_grads
    ):
        # Padding rows must contribute nothing: compare every row, then assert the padding is zero.
        rmse, cosine = _relative_metrics(kernel_grad, reference_grad)
        stats = (
            f"{name}: kernel nan {int(torch.isnan(kernel_grad).sum())} inf {int(torch.isinf(kernel_grad).sum())} "
            f"absmax {float(kernel_grad.abs().max()):.4f} reference nan {int(torch.isnan(reference_grad).sum())} "
            f"inf {int(torch.isinf(reference_grad).sum())} absmax {float(reference_grad.abs().max()):.4f}"
        )
        assert rmse <= 0.006, f"{name} RMSE {rmse} | {stats}"
        assert cosine >= 0.9999, f"{name} cosine {cosine} | {stats}"


def _gluon_and_triton_owners(batch: int, key_end_value: int):
    """Run the backward twice: once with the Gluon dK/dV owner, once with the Triton one.

    Passing a `dkdv_override` selects the Triton owner, which is otherwise kept only for reference.
    """
    key_end = torch.full((batch,), key_end_value, dtype=torch.int32, device="cuda")
    query, key, value, _, _ = _inputs(batch, seed=11, key_end=key_end)
    output, softmax_lse = _qsa_forward(query, key, value, None, key_end)
    grad_output = torch.randn(
        batch, _SEQUENCE_LENGTH, _QUERY_FEATURES, dtype=torch.bfloat16, device="cuda"
    ) / math.sqrt(_HEAD_DIM)
    grad_output[:, key_end_value:] = 0
    gluon = _qsa_backward(query, key, value, output, grad_output, softmax_lse, key_end)
    triton = _qsa_backward(
        query,
        key,
        value,
        output,
        grad_output,
        softmax_lse,
        key_end,
        dkdv_override=dict(_BACKWARD_CONFIGS[batch]["dkdv"]),
    )
    return key_end_value, gluon, triton


def test_qsa_backward_gluon_owner_matches_triton() -> None:
    """The Gluon dK/dV owner replaces the Triton one, so the two must produce the same gradients."""
    _, gluon, triton = _gluon_and_triton_owners(4, _SEQUENCE_LENGTH)
    for name, got, want in zip(("dQ", "dK", "dV"), gluon[:3], triton[:3]):
        difference = (got.float() - want.float()).abs().max()
        assert difference <= 1e-4, f"{name} differs by {float(difference)}"


def test_qsa_backward_gluon_owner_handles_padding() -> None:
    """With right padding the two owners must still agree, padded rows included."""
    _key_end_value, gluon, triton = _gluon_and_triton_owners(1, 1234)
    for name, got, want in zip(("dQ", "dK", "dV"), gluon[:3], triton[:3]):
        difference = (got.float() - want.float()).abs().max()
        assert difference <= 1e-4, f"{name} differs by {float(difference)}"


@pytest.mark.parametrize("triton_owner", [False, True])
def test_qsa_backward_padding_rows_are_zero(triton_owner: bool) -> None:
    """Padded positions leave the backward as zeros, not as recycled allocation.

    The dK/dV owners hold exactly zero for keys past `key_end`, and the outputs arrive uninitialized,
    so a store masked to the visible keys would leave whatever the allocator last held in the
    gradient at padded positions. A full-length backward runs first on purpose: it fills the
    same-size workspaces with non-zero values, which the padded run then reuses, so a fresh process's
    zeroed pages cannot hide the difference.
    """
    key_end_value = 1234
    key_end = torch.full((4,), key_end_value, dtype=torch.int32, device="cuda")
    full_key_end = torch.full((4,), _SEQUENCE_LENGTH, dtype=torch.int32, device="cuda")
    query, key, value, _, _ = _inputs(4, seed=11, key_end=key_end)
    output, softmax_lse = _qsa_forward(query, key, value, None, key_end)
    full_output, full_lse = _qsa_forward(query, key, value, None, full_key_end)
    upstream = torch.randn(
        4, _SEQUENCE_LENGTH, _QUERY_FEATURES, dtype=torch.bfloat16, device="cuda"
    )
    override = dict(_BACKWARD_CONFIGS[4]["dkdv"]) if triton_owner else None
    _qsa_backward(
        query,
        key,
        value,
        full_output,
        upstream,
        full_lse,
        full_key_end,
        dkdv_override=override,
    )
    upstream = upstream.clone()
    upstream[:, key_end_value:] = 0
    dquery, dkey, dvalue, _ = _qsa_backward(
        query,
        key,
        value,
        output,
        upstream,
        softmax_lse,
        key_end,
        dkdv_override=override,
    )
    padding = (slice(None), slice(None), slice(key_end_value, None), slice(None))
    for name, gradient in (("dQ", dquery), ("dK", dkey), ("dV", dvalue)):
        magnitude = float(gradient.float()[padding].abs().max())
        assert magnitude == 0.0, f"{name} has {magnitude} on padded rows"


def test_qsa_backward_matches_reference_gradients() -> None:
    _check_gradients(1)


def test_qsa_backward_covers_every_configured_batch() -> None:
    # The backward kernels take the batch from the launch table, so every batch needs its gradients.
    _check_gradients(4)


@pytest.mark.parametrize("key_end_value", [1234, 2000])
def test_qsa_backward_handles_padding(key_end_value: int) -> None:
    _check_gradients(1, key_end_value)
