import math
from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint
from transformers.masking_utils import sliding_window_causal_mask_function
from transformers.models.deepseek_v4.configuration_deepseek_v4 import (
    DeepseekV4Config,
)
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4Attention,
    DeepseekV4Model,
)

from deepseek_v4_attention import (
    _CONFIG_MARKER,
    _canonical_training_mask,
    _deepseek_v4_attention_forward,
    _validate_model_inputs,
    configure_deepseek_v4_attention,
    require_complete_deepseek_v4_attention,
)
from deepseek_v4_sliding_attention import (
    deepseek_v4_sliding_attention,
    deepseek_v4_sliding_attention_bshd,
)

_SEQUENCE_LENGTH = 2048
_QUERY_HEADS = 64
_HEAD_DIM = 512
_WINDOW = 128


def _reference_sliding_attention_bshd(
    query: torch.Tensor,
    shared_kv: torch.Tensor,
    sink: torch.Tensor,
) -> torch.Tensor:
    query_f32 = query.float().transpose(1, 2)
    kv_f32 = shared_kv.float().transpose(1, 2)
    outputs = []
    scale = 1.0 / math.sqrt(query.shape[-1])

    for query_start in range(0, _SEQUENCE_LENGTH, _WINDOW):
        query_end = query_start + _WINDOW
        key_start = max(0, query_start - (_WINDOW - 1))
        key_end = query_end
        query_block = query_f32[:, :, query_start:query_end]
        key_block = kv_f32[:, :, key_start:key_end]
        scores = torch.matmul(query_block, key_block.transpose(-1, -2)) * scale

        query_positions = torch.arange(query_start, query_end, device=query.device)[
            :, None
        ]
        key_positions = torch.arange(key_start, key_end, device=query.device)[None, :]
        visible = (key_positions <= query_positions) & (
            key_positions > query_positions - _WINDOW
        )
        scores = scores.masked_fill(~visible[None, None], float("-inf"))
        sink_logits = sink[None, :, None, None].expand(query.shape[0], -1, _WINDOW, -1)
        probabilities = torch.softmax(torch.cat((scores, sink_logits), dim=-1), dim=-1)[
            ..., :-1
        ]
        outputs.append(torch.matmul(probabilities, key_block))

    return torch.cat(outputs, dim=2).transpose(1, 2)


def _relative_metrics(
    candidate: torch.Tensor, reference: torch.Tensor
) -> tuple[float, float, float]:
    candidate_f32 = candidate.detach().float().flatten()
    reference_f32 = reference.detach().float().flatten()
    difference = candidate_f32 - reference_f32
    relative_rmse = difference.square().mean().sqrt() / (
        reference_f32.square().mean().sqrt() + 1e-12
    )
    cosine = torch.nn.functional.cosine_similarity(candidate_f32, reference_f32, dim=0)
    return (
        float(relative_rmse),
        float(cosine),
        float(difference.abs().max()),
    )


def test_sliding_attention_exact_shape_matches_blockwise_reference() -> None:
    torch.manual_seed(1701)
    batch = 1
    query_storage = torch.randn(
        batch,
        _QUERY_HEADS,
        _SEQUENCE_LENGTH,
        _HEAD_DIM,
        device="cuda",
        dtype=torch.bfloat16,
    )
    kv_storage = torch.randn(
        batch,
        1,
        _SEQUENCE_LENGTH,
        _HEAD_DIM,
        device="cuda",
        dtype=torch.bfloat16,
    )
    sink_storage = torch.randn(_QUERY_HEADS, device="cuda", dtype=torch.float32)
    output_gradient = torch.randn_like(query_storage).transpose(1, 2)
    assert not output_gradient.is_contiguous()
    assert output_gradient.stride(-1) == 1

    candidate_query = query_storage.clone().requires_grad_()
    candidate_kv = kv_storage.clone().requires_grad_()
    candidate_sink = sink_storage.clone().requires_grad_()
    candidate = deepseek_v4_sliding_attention(
        candidate_query,
        candidate_kv,
        candidate_sink,
    )
    candidate_gradients = torch.autograd.grad(
        candidate,
        (candidate_query, candidate_kv, candidate_sink),
        output_gradient,
    )

    reference_query = query_storage.clone().requires_grad_()
    reference_kv = kv_storage.clone().requires_grad_()
    reference_sink = sink_storage.clone().requires_grad_()
    reference = _reference_sliding_attention_bshd(
        reference_query.transpose(1, 2),
        reference_kv.transpose(1, 2),
        reference_sink,
    )
    reference_gradients = torch.autograd.grad(
        reference,
        (reference_query, reference_kv, reference_sink),
        output_gradient.float(),
    )
    torch.cuda.synchronize()

    thresholds = {
        "output": (0.0022, 0.99999),
        "query gradient": (0.003, 0.99999),
        "shared-KV gradient": (0.0035, 0.99999),
        "sink gradient": (0.0005, 0.999999),
    }
    values = (("output", candidate, reference),) + tuple(
        (label, got, expected)
        for label, got, expected in zip(
            ("query gradient", "shared-KV gradient", "sink gradient"),
            candidate_gradients,
            reference_gradients,
        )
    )
    for label, got, expected in values:
        assert torch.isfinite(got).all(), f"nonfinite {label}"
        relative_rmse, cosine, _ = _relative_metrics(got, expected)
        maximum_rmse, minimum_cosine = thresholds[label]
        assert relative_rmse <= maximum_rmse, (label, relative_rmse)
        assert cosine >= minimum_cosine, (label, cosine)


def test_sliding_attention_accepts_model_native_transpose_views() -> None:
    query = torch.empty(
        1,
        _QUERY_HEADS,
        _SEQUENCE_LENGTH,
        _HEAD_DIM,
        device="cuda",
        dtype=torch.bfloat16,
    )
    shared_kv = torch.empty(
        1,
        1,
        _SEQUENCE_LENGTH,
        _HEAD_DIM,
        device="cuda",
        dtype=torch.bfloat16,
    )
    sink = torch.empty(_QUERY_HEADS, device="cuda", dtype=torch.float32)

    assert query.is_contiguous()
    assert not query.transpose(1, 2).is_contiguous()
    with torch.no_grad():
        output = deepseek_v4_sliding_attention(query, shared_kv, sink)
    assert output.shape == (1, _SEQUENCE_LENGTH, _QUERY_HEADS, _HEAD_DIM)
    assert output.is_contiguous()


@pytest.mark.parametrize("batch", [2, 8])
def test_sliding_attention_rejects_unsupported_batches(batch: int) -> None:
    query = torch.empty(
        batch, 1, _QUERY_HEADS, _HEAD_DIM, device="cuda", dtype=torch.bfloat16
    )
    shared_kv = torch.empty(batch, 1, 1, _HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    sink = torch.empty(_QUERY_HEADS, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="unsupported.*batch"):
        deepseek_v4_sliding_attention_bshd(query, shared_kv, sink)


def test_sliding_attention_rejects_output_gradient_copy() -> None:
    query = torch.empty(
        1,
        _QUERY_HEADS,
        _SEQUENCE_LENGTH,
        _HEAD_DIM,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    shared_kv = torch.empty(
        1,
        1,
        _SEQUENCE_LENGTH,
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

    output = deepseek_v4_sliding_attention(query, shared_kv, sink)
    assert output_gradient.shape == output.shape
    assert output_gradient.stride(-1) != 1
    with pytest.raises(ValueError, match="contiguous last dimension"):
        torch.autograd.grad(output, (query, shared_kv, sink), output_gradient)


def test_sliding_attention_rejects_wrong_dtype() -> None:
    query = torch.empty(
        1,
        _SEQUENCE_LENGTH,
        _QUERY_HEADS,
        _HEAD_DIM,
        device="cuda",
        dtype=torch.float16,
    )
    shared_kv = torch.empty(
        1,
        _SEQUENCE_LENGTH,
        1,
        _HEAD_DIM,
        device="cuda",
        dtype=torch.bfloat16,
    )
    sink = torch.empty(_QUERY_HEADS, device="cuda", dtype=torch.float32)
    with pytest.raises(TypeError, match="bfloat16"):
        deepseek_v4_sliding_attention_bshd(query, shared_kv, sink)


def test_attention_dispatch_uses_sliding_kernel_without_layout_copies() -> None:
    query_storage = torch.randn(
        1,
        _QUERY_HEADS,
        _SEQUENCE_LENGTH,
        _HEAD_DIM,
        device="cuda",
        dtype=torch.bfloat16,
    )
    kv_storage = torch.randn(
        1,
        1,
        _SEQUENCE_LENGTH,
        _HEAD_DIM,
        device="cuda",
        dtype=torch.bfloat16,
    )
    sink = torch.randn(_QUERY_HEADS, device="cuda", dtype=torch.float32)
    query = query_storage
    shared_kv = kv_storage
    module = SimpleNamespace(
        layer_type="sliding_attention",
        attention_dropout=0.0,
        sliding_window=_WINDOW,
        sinks=sink,
    )
    setattr(module, _CONFIG_MARKER, True)
    canonical_mask = torch.empty(
        1,
        1,
        _SEQUENCE_LENGTH,
        _SEQUENCE_LENGTH,
        device="cuda",
        dtype=torch.bfloat16,
    )

    with torch.no_grad():
        dispatched, weights = _deepseek_v4_attention_forward(
            module,
            query,
            shared_kv,
            shared_kv,
            canonical_mask,
            scaling=1.0 / math.sqrt(_HEAD_DIM),
            dropout=0.0,
            s_aux=sink,
        )
        direct = deepseek_v4_sliding_attention(query, shared_kv, sink)
    assert weights is None
    assert dispatched.is_contiguous()
    assert torch.equal(dispatched, direct)


def test_canonical_mask_has_one_physical_batch() -> None:
    mask = _canonical_training_mask(
        batch_size=4,
        q_length=_SEQUENCE_LENGTH,
        kv_length=_SEQUENCE_LENGTH,
        mask_function=sliding_window_causal_mask_function(_WINDOW),
        attention_mask=torch.ones(4, _SEQUENCE_LENGTH, device="cuda", dtype=torch.bool),
        dtype=torch.bfloat16,
        device="cuda",
    )
    assert mask.shape == (4, 1, _SEQUENCE_LENGTH, _SEQUENCE_LENGTH)
    assert mask.stride(0) == 0
    assert mask.untyped_storage().nbytes() == 2 * _SEQUENCE_LENGTH**2


def test_model_input_guard_rejects_noncanonical_shape() -> None:
    with pytest.raises(ValueError, match=r"B in \{1,4,16\} and S=2048"):
        _validate_model_inputs(
            None,
            (),
            {"input_ids": torch.zeros(1, 1024, device="cuda", dtype=torch.long)},
        )


def test_non_reentrant_checkpoint_recomputation_preserves_gradients() -> None:
    torch.manual_seed(2048)
    query = torch.randn(
        1,
        _QUERY_HEADS,
        _SEQUENCE_LENGTH,
        _HEAD_DIM,
        device="cuda",
        dtype=torch.bfloat16,
    )
    shared_kv = torch.randn(
        1,
        1,
        _SEQUENCE_LENGTH,
        _HEAD_DIM,
        device="cuda",
        dtype=torch.bfloat16,
    )
    sink = torch.randn(_QUERY_HEADS, device="cuda", dtype=torch.float32)
    output_gradient = torch.randn_like(query).transpose(1, 2)

    direct_inputs = (
        query.clone().requires_grad_(),
        shared_kv.clone().requires_grad_(),
        sink.clone().requires_grad_(),
    )
    checkpoint_inputs = (
        query.clone().requires_grad_(),
        shared_kv.clone().requires_grad_(),
        sink.clone().requires_grad_(),
    )

    def run(query_bhsd, kv_bhsd, sink_values):
        return deepseek_v4_sliding_attention(query_bhsd, kv_bhsd, sink_values)

    direct_output = run(*direct_inputs)
    direct_gradients = torch.autograd.grad(
        direct_output, direct_inputs, output_gradient
    )
    checkpoint_output = checkpoint(run, *checkpoint_inputs, use_reentrant=False)
    checkpoint_gradients = torch.autograd.grad(
        checkpoint_output, checkpoint_inputs, output_gradient
    )
    torch.cuda.synchronize()

    assert torch.equal(checkpoint_output, direct_output)
    assert torch.equal(checkpoint_gradients[0], direct_gradients[0])
    assert torch.equal(checkpoint_gradients[1], direct_gradients[1])
    sink_relative_rmse, sink_cosine, sink_max_abs = _relative_metrics(
        checkpoint_gradients[2], direct_gradients[2]
    )
    assert sink_relative_rmse <= 1e-6
    assert sink_cosine >= 0.999999
    assert sink_max_abs <= 1e-4


def test_attention_configuration_is_complete_and_idempotent() -> None:
    class AttentionShell(DeepseekV4Attention):
        def __init__(self, layer_type: str) -> None:
            torch.nn.Module.__init__(self)
            self.layer_type = layer_type

    class ModelShell(DeepseekV4Model):
        def __init__(self, config: DeepseekV4Config) -> None:
            torch.nn.Module.__init__(self)
            self.config = config
            layer_types = (
                ["sliding_attention"] * 2
                + ["compressed_sparse_attention"] * 21
                + ["heavily_compressed_attention"] * 20
            )
            self.attentions = torch.nn.ModuleList(
                AttentionShell(layer_type) for layer_type in layer_types
            )

    config = DeepseekV4Config(
        num_hidden_layers=43,
        layer_types=(
            ["sliding_attention"] * 2
            + ["compressed_sparse_attention"] * 21
            + ["heavily_compressed_attention"] * 20
        ),
        use_cache=False,
        attention_dropout=0.0,
    )
    model = ModelShell(config)
    first = configure_deepseek_v4_attention(model)
    require_complete_deepseek_v4_attention(first)
    second = configure_deepseek_v4_attention(model)
    require_complete_deepseek_v4_attention(second)

    assert first["configured_sliding"] == 2
    assert first["already_configured"] == 0
    assert first["configured_csa"] == 21
    assert first["already_configured_csa"] == 0
    assert second["configured_sliding"] == 0
    assert second["already_configured"] == 2
    assert second["configured_csa"] == 0
    assert second["already_configured_csa"] == 21
    assert config._attn_implementation == "deepseek_v4_project"
