from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from benchmark_deepseek_v4_hca import (
    make_inputs,
    metrics,
    reference_hca_attention,
    reference_hca_compress,
)
from deepseek_v4_attention import (
    _HCA_CONFIG_MARKER,
    _deepseek_v4_hca_module_forward,
)
from deepseek_v4_hca import deepseek_v4_hca_attention, deepseek_v4_hca_compress

_SEQUENCE_LENGTH = 2048
_QUERY_HEADS = 64
_HEAD_DIM = 512
_COMPRESSED_LENGTH = 16


def _assert_metrics(
    label: str,
    candidate: torch.Tensor,
    reference: torch.Tensor,
    maximum_rmse: float,
    minimum_cosine: float,
) -> None:
    assert torch.isfinite(candidate).all(), f"nonfinite {label}"
    result = metrics(candidate, reference)
    assert result["relative_rmse"] <= maximum_rmse, (label, result)
    assert result["cosine"] >= minimum_cosine, (label, result)


def test_hca_attention_exact_shape_matches_blockwise_reference() -> None:
    values = make_inputs(1, 2701)
    output_gradient = torch.randn_like(values["query"]).transpose(1, 2) * 0.03
    candidate_inputs = tuple(
        values[name].clone().requires_grad_()
        for name in ("query", "local_kv", "compressed_kv", "sink")
    )
    candidate = deepseek_v4_hca_attention(*candidate_inputs)
    candidate_gradients = torch.autograd.grad(
        candidate, candidate_inputs, output_gradient
    )

    reference_inputs = tuple(
        tensor.detach().clone().requires_grad_() for tensor in candidate_inputs
    )
    reference = reference_hca_attention(
        reference_inputs[0],
        reference_inputs[1],
        reference_inputs[2],
        reference_inputs[3],
    )
    reference_gradients = torch.autograd.grad(
        reference, reference_inputs, output_gradient
    )
    torch.cuda.synchronize()

    gates = {
        "output": (0.0036, 0.99999),
        "query gradient": (0.0034, 0.99999),
        "local KV gradient": (0.0057, 0.99998),
        "compressed KV gradient": (0.0125, 0.99992),
        "sink gradient": (0.0025, 0.999997),
    }
    _assert_metrics("output", candidate, reference, *gates["output"])
    for label, got, expected in zip(
        (
            "query gradient",
            "local KV gradient",
            "compressed KV gradient",
            "sink gradient",
        ),
        candidate_gradients,
        reference_gradients,
    ):
        _assert_metrics(label, got, expected, *gates[label])

    # Cover the empty prefix and both ends of the deterministic C128 threshold.
    for row in (0, 126, 127, 128, 2047):
        _assert_metrics(
            f"output row {row}",
            candidate[:, row],
            reference[:, row],
            0.0045,
            0.99998,
        )


def test_hca_producer_exact_shape_matches_reference() -> None:
    values = make_inputs(1, 2801)
    output_gradient = (
        torch.randn(
            1,
            1,
            _COMPRESSED_LENGTH,
            _HEAD_DIM,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.03
    )
    candidate_inputs = tuple(
        values[name].clone().requires_grad_()
        for name in ("compressor_kv", "compressor_gate")
    )
    candidate = deepseek_v4_hca_compress(
        candidate_inputs[0],
        candidate_inputs[1],
        values["position_bias"],
        values["weight"],
        values["cos"],
        values["sin"],
        1e-6,
    )
    candidate_gradients = torch.autograd.grad(
        candidate, candidate_inputs, output_gradient
    )

    reference_inputs = tuple(
        tensor.detach().clone().requires_grad_() for tensor in candidate_inputs
    )
    reference = reference_hca_compress(
        reference_inputs[0],
        reference_inputs[1],
        values["position_bias"],
        values["weight"],
        values["cos"],
        values["sin"],
        1e-6,
    )
    reference_gradients = torch.autograd.grad(
        reference, reference_inputs, output_gradient.squeeze(1)
    )
    torch.cuda.synchronize()

    _assert_metrics(
        "producer output", candidate.squeeze(1), reference, 0.0026, 0.999995
    )
    _assert_metrics(
        "producer KV gradient",
        candidate_gradients[0],
        reference_gradients[0],
        0.0026,
        0.999995,
    )
    _assert_metrics(
        "producer gate gradient",
        candidate_gradients[1],
        reference_gradients[1],
        0.0026,
        0.999995,
    )


def test_hca_non_reentrant_checkpoint_recomputation_is_deterministic() -> None:
    values = make_inputs(1, 2901)
    output_gradient = torch.randn_like(values["query"]).transpose(1, 2)
    direct_inputs = tuple(
        values[name].clone().requires_grad_()
        for name in ("query", "local_kv", "compressed_kv", "sink")
    )
    checkpoint_inputs = tuple(
        tensor.detach().clone().requires_grad_() for tensor in direct_inputs
    )

    direct_output = deepseek_v4_hca_attention(*direct_inputs)
    direct_gradients = torch.autograd.grad(
        direct_output, direct_inputs, output_gradient
    )
    checkpoint_output = checkpoint(
        deepseek_v4_hca_attention,
        *checkpoint_inputs,
        use_reentrant=False,
    )
    checkpoint_gradients = torch.autograd.grad(
        checkpoint_output, checkpoint_inputs, output_gradient
    )
    torch.cuda.synchronize()

    assert torch.equal(checkpoint_output, direct_output)
    for checkpoint_gradient, direct_gradient in zip(
        checkpoint_gradients, direct_gradients
    ):
        assert torch.equal(checkpoint_gradient, direct_gradient)


def test_hca_rejects_output_gradient_copy() -> None:
    query = torch.empty(
        1,
        _QUERY_HEADS,
        _SEQUENCE_LENGTH,
        _HEAD_DIM,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    local_kv = torch.empty(
        1,
        1,
        _SEQUENCE_LENGTH,
        _HEAD_DIM,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    compressed_kv = torch.empty(
        1,
        1,
        _COMPRESSED_LENGTH,
        _HEAD_DIM,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    sink = torch.empty(
        _QUERY_HEADS, device="cuda", dtype=torch.float32, requires_grad=True
    )
    output_gradient = torch.empty(
        1,
        _SEQUENCE_LENGTH,
        _HEAD_DIM,
        _QUERY_HEADS,
        device="cuda",
        dtype=torch.bfloat16,
    ).transpose(2, 3)
    output = deepseek_v4_hca_attention(query, local_kv, compressed_kv, sink)
    assert output_gradient.shape == output.shape
    assert output_gradient.stride(-1) != 1
    with pytest.raises(ValueError, match="contiguous last dimension"):
        torch.autograd.grad(
            output,
            (query, local_kv, compressed_kv, sink),
            output_gradient,
        )


@pytest.mark.parametrize("batch", [2, 8])
def test_hca_rejects_unsupported_batches(batch: int) -> None:
    query = torch.empty(
        batch, _QUERY_HEADS, 1, _HEAD_DIM, device="cuda", dtype=torch.bfloat16
    )
    local_kv = torch.empty(batch, 1, 1, _HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    compressed_kv = torch.empty_like(local_kv)
    sink = torch.empty(_QUERY_HEADS, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="unsupported.*batch"):
        deepseek_v4_hca_attention(query, local_kv, compressed_kv, sink)


def test_hca_producer_accepts_strided_frozen_metadata() -> None:
    values = make_inputs(1, 2951)
    bias_storage = torch.empty(128, 2 * _HEAD_DIM, device="cuda", dtype=torch.float32)
    position_bias = bias_storage[:, ::2]
    position_bias.copy_(values["position_bias"])
    cos_storage = torch.empty(
        1, _SEQUENCE_LENGTH, 64, device="cuda", dtype=torch.bfloat16
    )
    sin_storage = torch.empty_like(cos_storage)
    cos = cos_storage[..., ::2]
    sin = sin_storage[..., ::2]
    cos.copy_(values["cos"])
    sin.copy_(values["sin"])
    weight = values["weight"].float()
    assert position_bias.stride(-1) == 2
    assert cos.stride(-1) == 2

    candidate = deepseek_v4_hca_compress(
        values["compressor_kv"],
        values["compressor_gate"],
        position_bias,
        weight,
        cos,
        sin,
        1e-6,
    )
    reference = reference_hca_compress(
        values["compressor_kv"],
        values["compressor_gate"],
        position_bias,
        weight,
        cos,
        sin,
        1e-6,
    )
    _assert_metrics(
        "strided producer output",
        candidate.squeeze(1),
        reference,
        0.0026,
        0.999995,
    )


def test_hca_producer_rejects_trainable_frozen_state() -> None:
    values = make_inputs(1, 3001)
    with pytest.raises(ValueError, match="must be frozen"):
        deepseek_v4_hca_compress(
            values["compressor_kv"],
            values["compressor_gate"],
            values["position_bias"].requires_grad_(),
            values["weight"],
            values["cos"],
            values["sin"],
            1e-6,
        )


def test_hca_module_dispatch_rejects_wrong_rate() -> None:
    module = SimpleNamespace(
        config=SimpleNamespace(hidden_size=1),
        training=True,
        attention_dropout=0.0,
        compressor=SimpleNamespace(compress_rate=64),
    )
    setattr(module, _HCA_CONFIG_MARKER, True)
    hidden_states = torch.empty(
        1, _SEQUENCE_LENGTH, 1, device="cuda", dtype=torch.bfloat16
    )
    position_ids = torch.arange(_SEQUENCE_LENGTH, device="cuda").unsqueeze(0)
    position_embeddings = {
        "compress": (
            torch.empty(1, _SEQUENCE_LENGTH, 32, device="cuda", dtype=torch.bfloat16),
            torch.empty(1, _SEQUENCE_LENGTH, 32, device="cuda", dtype=torch.bfloat16),
        )
    }
    canonical_mask = torch.empty(
        1,
        1,
        _SEQUENCE_LENGTH,
        _SEQUENCE_LENGTH,
        device="cuda",
        dtype=torch.bfloat16,
    )
    with pytest.raises(ValueError, match="rate 128"):
        _deepseek_v4_hca_module_forward(
            cast(Any, module),
            hidden_states,
            position_embeddings,
            position_ids,
            canonical_mask,
        )
