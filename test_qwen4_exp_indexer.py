"""Tests for the Qwen4-Exp QSA indexer short circuit."""

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from transformers import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextQSAIndexer

from qwen4_exp_indexer import (
    _MARKER,
    _REFERENCE_ATTRIBUTE,
    _identity_mask,
    _qwen4_indexer_forward,
    configure_qwen4_exp_indexer_fast_path,
    is_exhaustive_at,
    require_complete_qwen4_exp_indexer,
)

SEQ = 2048


def test_is_exhaustive_boundary() -> None:
    # The audited length and the selection buffer's capacity are both inside the guard.
    assert is_exhaustive_at(512, 4, SEQ)
    assert is_exhaustive_at(512, 4, 2051)
    assert not is_exhaustive_at(512, 4, 2052)
    # A small budget only covers what fits its own buffer: block_topk * ratio + ratio - 1.
    assert is_exhaustive_at(4, 4, 19)
    assert not is_exhaustive_at(4, 4, 20)


def test_identity_mask_bool_is_the_input() -> None:
    mask = torch.zeros(2, 1, 8, 8, dtype=torch.bool).tril()
    identity = _identity_mask(mask)
    assert identity is mask
    assert torch.equal(mask & identity, mask)


def test_identity_mask_float_is_the_additive_identity() -> None:
    mask = torch.full((2, 1, 8, 8), torch.finfo(torch.float32).min)
    mask = mask.tril()
    identity = _identity_mask(mask)
    assert identity.dtype == mask.dtype
    assert torch.equal(mask + identity, mask)


def _fake_indexer(block_topk: int, compress_ratio: int, reference) -> SimpleNamespace:
    module = SimpleNamespace(block_topk=block_topk, compress_ratio=compress_ratio)
    setattr(module, _REFERENCE_ATTRIBUTE, reference)
    return module


def test_wrapper_skips_when_exhaustive_and_delegates_otherwise() -> None:
    calls: list[tuple] = []

    def reference(*args):
        calls.append(args)
        return torch.zeros(1)

    module = _fake_indexer(512, 4, reference)
    mask = torch.zeros(1, 1, SEQ, SEQ, dtype=torch.bool).tril()
    hidden = torch.zeros(1, SEQ, 8)

    assert (
        _qwen4_indexer_forward(
            cast(Any, module), hidden, (torch.zeros(1), torch.zeros(1)), mask, None
        )
        is mask
    )
    assert calls == []

    # A sequence past the buffer bound keeps the model's own selection.
    longer = torch.zeros(1, 2052, 8)
    _qwen4_indexer_forward(
        cast(Any, module), longer, (torch.zeros(1), torch.zeros(1)), mask, None
    )
    assert len(calls) == 1

    # A cache always delegates, since the indexer key state is then real work.
    _qwen4_indexer_forward(
        cast(Any, module), hidden, (torch.zeros(1), torch.zeros(1)), mask, object()
    )
    assert len(calls) == 2


def test_wrapper_requires_the_reference() -> None:
    module = SimpleNamespace(block_topk=512, compress_ratio=4)
    with pytest.raises(RuntimeError, match="no reference forward"):
        _qwen4_indexer_forward(
            cast(Any, module),
            torch.zeros(1, SEQ, 8),
            (torch.zeros(1), torch.zeros(1)),
            torch.zeros(1),
            None,
        )


def test_configure_and_gate_on_a_real_indexer() -> None:
    config = Qwen4ExpTextConfig(
        num_hidden_layers=48,
        hidden_size=2560,
        num_attention_heads=24,
        num_key_value_heads=2,
        head_dim=256,
        rope_parameters={"rope_theta": 10_000_000.0, "partial_rotary_factor": 0.25},
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=128,
        indexer_budget=2048,
        indexer_compress_ratio=4,
    )
    indexer = Qwen4ExpTextQSAIndexer(config, layer_idx=3)
    parent = torch.nn.Module()
    parent.indexer = indexer  # type: ignore[assignment]

    report = configure_qwen4_exp_indexer_fast_path(parent)
    assert report["indexers"] == 1
    assert report["selection"] == "skipped-when-exhaustive"
    assert report["exhaustive_upto"] == 2051
    require_complete_qwen4_exp_indexer(report, expected_layers=1, sequence_length=SEQ)
    assert getattr(indexer, _MARKER) is True

    mask = torch.zeros(1, 1, SEQ, SEQ, dtype=torch.bool).tril()
    hidden = torch.zeros(1, SEQ, config.hidden_size)
    embeddings = (torch.zeros(1, SEQ, 64), torch.zeros(1, SEQ, 64))
    assert indexer(hidden, embeddings, mask, None) is mask

    # A second pass keeps the first reference, so the fallback cannot point at the patch itself.
    reference = getattr(indexer, _REFERENCE_ATTRIBUTE)
    configure_qwen4_exp_indexer_fast_path(parent)
    assert getattr(indexer, _REFERENCE_ATTRIBUTE) == reference

    with pytest.raises(RuntimeError, match="not exhaustive"):
        require_complete_qwen4_exp_indexer(
            report, expected_layers=1, sequence_length=2052
        )
