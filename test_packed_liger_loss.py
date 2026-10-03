"""Contract tests for the shared packed-GGUF scoped loss driver.

These tests are CPU-only and stub the packed calculation. They pin the
fail-closed rules that both model forwards rely on: training with labels uses
the packed loss without retained logits, a packed step rejects
`logits_to_keep`, and evaluation keeps the ordinary materialized-logits loss.
"""

from collections.abc import Callable
from types import SimpleNamespace

import pytest
import torch
from liger_kernel.transformers.model.output_classes import (
    LigerMoeCausalLMOutputWithPast,
)
from transformers.integrations.gguf.modules import GgufLinear

from packed_liger_loss import (
    PackedLossResult,
    ScopedLossResult,
    assembled_scoped_output,
    scoped_packed_causal_lm_loss,
    validate_packed_lm_head,
)


class _Outputs:
    def __init__(self, hidden_states: torch.Tensor | None) -> None:
        self.last_hidden_state = hidden_states


class _Model(torch.nn.Module):
    """Minimal surface the scoped driver reads from a model."""

    def __init__(self, *, training: bool, lm_head: GgufLinear) -> None:
        super().__init__()
        self.training = training
        self.vocab_size = 16
        self.lm_head = lm_head
        self.config = SimpleNamespace(hidden_size=8)
        self.materialized_labels: list[torch.Tensor | None] = []

    def loss_function(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor | None,
        vocab_size: int,
        shift_labels: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        self.materialized_labels.append(labels)
        return torch.zeros(())


def _head(*, bias: bool = False) -> GgufLinear:
    return GgufLinear(
        8,
        16,
        bias=bias,
        device="cpu",
        dtype=torch.bfloat16,
        floating_weight=True,
    )


def _hidden() -> torch.Tensor:
    return torch.zeros(2, 3, 8, dtype=torch.bfloat16)


def _run(
    model: _Model,
    *,
    labels: torch.Tensor | None,
    shift_labels: torch.Tensor | None = None,
    logits_to_keep: int | torch.Tensor = 0,
    skip_logits: bool | None = None,
    packed_loss: Callable[..., PackedLossResult],
    outputs: _Outputs | None = None,
) -> ScopedLossResult:
    return scoped_packed_causal_lm_loss(
        model,
        outputs if outputs is not None else _Outputs(_hidden()),
        labels=labels,
        shift_labels=shift_labels,
        logits_to_keep=logits_to_keep,
        skip_logits=skip_logits,
        packed_loss=packed_loss,
        loss_kwargs={"num_items_in_batch": 6},
    )


def test_training_with_labels_uses_packed_loss_without_logits() -> None:
    model = _Model(training=True, lm_head=_head())
    labels = torch.zeros(2, 3, dtype=torch.long)
    shift_labels = labels.roll(-1, dims=-1)
    calls: list[dict[str, object]] = []

    def packed_loss(**kwargs: object) -> PackedLossResult:
        calls.append(kwargs)
        return torch.tensor(2.0)

    loss, logits, accuracy, predicted = _run(
        model, labels=labels, shift_labels=shift_labels, packed_loss=packed_loss
    )

    assert loss is not None and logits is None
    assert accuracy is None and predicted is None
    assert model.materialized_labels == []
    assert len(calls) == 1
    assert calls[0]["hidden_size"] == 8
    assert calls[0]["num_items_in_batch"] == 6
    assert calls[0]["shift_labels"] is shift_labels


def test_packed_training_rejects_retained_logits() -> None:
    model = _Model(training=True, lm_head=_head())
    with pytest.raises(RuntimeError, match="logits_to_keep=0"):
        _run(
            model,
            labels=torch.zeros(2, 3, dtype=torch.long),
            logits_to_keep=4,
            packed_loss=lambda **kwargs: torch.tensor(2.0),
        )


def test_packed_training_cannot_be_disabled_with_labels() -> None:
    model = _Model(training=True, lm_head=_head())
    with pytest.raises(RuntimeError, match="no-full-logits"):
        _run(
            model,
            labels=torch.zeros(2, 3, dtype=torch.long),
            skip_logits=False,
            packed_loss=lambda **kwargs: torch.tensor(2.0),
        )


def test_packed_step_requires_labels() -> None:
    model = _Model(training=True, lm_head=_head())
    with pytest.raises(RuntimeError, match="requires labels"):
        _run(
            model,
            labels=None,
            skip_logits=True,
            packed_loss=lambda **kwargs: torch.tensor(2.0),
        )


def test_evaluation_with_labels_keeps_materialized_logits() -> None:
    model = _Model(training=False, lm_head=_head())
    labels = torch.zeros(2, 3, dtype=torch.long)

    def packed_loss(**kwargs: object) -> PackedLossResult:
        raise AssertionError("evaluation with labels must not use the packed loss")

    loss, logits, accuracy, predicted = _run(
        model, labels=labels, packed_loss=packed_loss
    )

    assert loss is not None and logits is not None
    assert accuracy is None and predicted is None
    assert model.materialized_labels == [labels]


def test_training_without_labels_keeps_materialized_logits() -> None:
    model = _Model(training=True, lm_head=_head())

    def packed_loss(**kwargs: object) -> PackedLossResult:
        raise AssertionError("training without labels must not use the packed loss")

    loss, logits, _, _ = _run(model, labels=None, packed_loss=packed_loss)

    assert loss is None and logits is not None
    assert model.materialized_labels == []


def test_missing_hidden_states_fail_closed() -> None:
    model = _Model(training=True, lm_head=_head())
    with pytest.raises(RuntimeError, match="did not return hidden states"):
        _run(
            model,
            labels=torch.zeros(2, 3, dtype=torch.long),
            packed_loss=lambda **kwargs: torch.tensor(2.0),
            outputs=_Outputs(None),
        )


def test_validate_packed_lm_head_contract() -> None:
    validate_packed_lm_head(_head())
    with pytest.raises(TypeError, match="GgufLinear"):
        validate_packed_lm_head(torch.nn.Linear(8, 16, bias=False))
    with pytest.raises(RuntimeError, match="bias-free"):
        validate_packed_lm_head(_head(bias=True))


def test_packed_result_optional_outputs_survive_the_driver() -> None:
    model = _Model(training=True, lm_head=_head())
    labels = torch.zeros(2, 3, dtype=torch.long)
    accuracy = torch.tensor(0.5)
    predicted = torch.zeros(6, dtype=torch.long)

    def packed_loss(**kwargs: object) -> PackedLossResult:
        return torch.tensor(2.0), None, accuracy, predicted

    loss, logits, token_accuracy, predicted_tokens = _run(
        model, labels=labels, packed_loss=packed_loss
    )

    assert loss is not None and logits is None
    assert token_accuracy is accuracy
    assert predicted_tokens is predicted


def test_assembled_output_surfaces_requested_optional_fields() -> None:
    accuracy = torch.zeros(())
    predicted = torch.zeros(4, dtype=torch.long)

    assembled = assembled_scoped_output(
        SimpleNamespace(
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            router_logits=None,
        ),
        loss=torch.zeros(()),
        aux_loss=None,
        logits=None,
        token_accuracy=accuracy,
        predicted_tokens=predicted,
    )

    assert isinstance(assembled, LigerMoeCausalLMOutputWithPast)
    assert assembled.token_accuracy is accuracy
    assert assembled.predicted_tokens is predicted
    assert assembled.logits is None


def test_assembled_output_defaults_optional_fields_to_none() -> None:
    assembled = assembled_scoped_output(
        SimpleNamespace(),
        loss=torch.zeros(()),
        aux_loss=None,
        logits=torch.zeros(()),
    )

    assert assembled.token_accuracy is None
    assert assembled.predicted_tokens is None
    assert assembled.logits is not None
