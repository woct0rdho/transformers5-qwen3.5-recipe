"""Run the Qwen4-Exp QSA attention on the project's QSA kernels.

At `S = 2048` the twelve indexed-attention layers are dense causal grouped-query attention: 24 query
heads over 2 KV heads, `head_dim` 256, a partial RoPE rotation of 64 applied before the call, and a
sigmoid gate on the attention output. That is exactly the contract `qwen4_exp_qsa_attention_autograd`
implements. The model's own path reaches the same arithmetic through masked SDPA, which the kernels
replace.

The patch replaces the forward of each `Qwen4ExpTextAttention`. It skips the indexer call, whose
selection is exhaustive at this length and which `qwen4_exp_indexer` already short circuits, takes the
per-sample key bound from the model's own causal mask, and calls the kernels for the attention itself.
Everything else in the module forward is unchanged: the same projections, the same q/k RMSNorm, the
same partial RoPE, the same gate and the same `o_proj`.

The model's forward is kept on the module as `_qwen4_qsa_forward_reference` and is what runs above the
guard, so a batch, sequence length, dtype or cache the fixed-shape kernels do not serve keeps the
model's behaviour rather than a fast path that no longer holds. Right padding is served natively: the
audit's collator masks the padded rows' labels, so their incoming gradient is exactly zero, which is
the contract the backward's pad handling assumes.
"""

from typing import Any

import torch
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextAttention,
    apply_rotary_pos_emb,
)

from module_patching import (
    ModulePatchSpec,
    patch_module_forwards,
    require_complete_inventory,
)
from qwen4_exp_indexer import EXPECTED_QSA_LAYERS
from qwen4_exp_qsa_attention import qwen4_exp_qsa_attention_autograd

EXPECTED_SEQUENCE_LENGTH = 2048
EXPECTED_QUERY_HEADS = 24
EXPECTED_KV_HEADS = 2
EXPECTED_HEAD_DIM = 256
EXPECTED_QUERY_FEATURES = EXPECTED_QUERY_HEADS * EXPECTED_HEAD_DIM
SUPPORTED_BATCHES = frozenset({1, 4, 16})

_SUBJECT = "Qwen4-Exp QSA attention"
_MARKER = "_patched_qwen4_qsa_attention"
_REFERENCE_ATTRIBUTE = "_qwen4_qsa_forward_reference"


def _key_end_from_mask(attention_mask: torch.Tensor | None) -> torch.Tensor | None:
    """The per-sample key bound, read from the last row of the model's causal mask.

    The model's mask is `[batch, 1, query, key]`. None of its rows is masked on the query side, so the
    last row is an ordinary causal row over the whole sequence and the number of keys it leaves
    visible is the right-padding bound the kernels take as `key_end`. Returns None for a mask shape or
    dtype the wiring does not serve, which sends the call to the reference forward instead.
    """

    if not torch.is_tensor(attention_mask) or attention_mask.dim() != 4:
        return None
    if attention_mask.shape[1] != 1:
        return None
    if attention_mask.shape[-2] != EXPECTED_SEQUENCE_LENGTH:
        return None
    row = attention_mask[:, 0, -1, :]
    if row.dtype == torch.bool:
        visible = row
    elif row.is_floating_point():
        visible = row > torch.finfo(row.dtype).min / 2
    else:
        return None
    if visible.shape[-1] != EXPECTED_SEQUENCE_LENGTH:
        return None
    return visible.sum(dim=-1, dtype=torch.int32)


def _qwen4_qsa_forward_serves(
    module: Qwen4ExpTextAttention,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    past_key_values: Any,
) -> bool:
    """Whether the fixed-shape kernels serve this call as it stands."""

    if past_key_values is not None:
        return False
    if hidden_states.dim() != 3:
        return False
    batch, sequence_length, _ = hidden_states.shape
    if batch not in SUPPORTED_BATCHES or sequence_length != EXPECTED_SEQUENCE_LENGTH:
        return False
    if hidden_states.dtype != torch.bfloat16:
        return False
    if module.head_dim != EXPECTED_HEAD_DIM:
        return False
    if module.num_key_value_groups != EXPECTED_QUERY_HEADS // EXPECTED_KV_HEADS:
        return False
    if float(module.attention_dropout) != 0.0:
        return False
    return _key_end_from_mask(attention_mask) is not None


def _qwen4_qsa_attention_forward(
    self: Qwen4ExpTextAttention,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Any = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    reference = getattr(self, _REFERENCE_ATTRIBUTE, None)
    if reference is None:
        raise RuntimeError(
            "Qwen4-Exp QSA attention has no reference forward to fall back to"
        )
    if not _qwen4_qsa_forward_serves(
        self, hidden_states, attention_mask, past_key_values
    ):
        return reference(
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values,
            **kwargs,
        )
    key_end = _key_end_from_mask(attention_mask)
    if key_end is None:
        return reference(
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values,
            **kwargs,
        )

    # The model passes the full positions because the indexer needs them. The attention itself wants
    # the current window. The kernel takes the gate-free output in the `o_proj` input layout, so the
    # gate is applied here and nothing else changes relative to the module's own forward.
    current_length = hidden_states.shape[1]
    cos, sin = (value[:, -current_length:, :] for value in position_embeddings)
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states, gate = torch.chunk(
        self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
    )
    gate = gate.reshape(*input_shape, -1)

    query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(
        1, 2
    )
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # The kernels take contiguous BHSD. The module's own path works on the transposed projection
    # views, so the three tensors are copied once here rather than converted per operand.
    query_states = query_states.contiguous()
    key_states = key_states.contiguous()
    value_states = value_states.contiguous()

    attn_output = qwen4_exp_qsa_attention_autograd(
        query_states, key_states, value_states, key_end
    )
    attn_output = attn_output * torch.sigmoid(gate)
    return self.o_proj(attn_output), None


def reference_attention_output(
    module: Qwen4ExpTextAttention,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Run the module's own forward, the one the patch replaces.

    The audit uses this to prove that the patched forward agrees with the model's SDPA path on the
    same inputs, with a right-padded case included.
    """

    reference = getattr(module, _REFERENCE_ATTRIBUTE, None)
    if reference is None:
        raise RuntimeError(
            "Qwen4-Exp QSA attention was not patched, so it has no reference forward"
        )
    return reference(hidden_states, position_embeddings, attention_mask, None)[0]


def _validate_qwen4_attention(name: str, module: Qwen4ExpTextAttention) -> None:
    if module.head_dim != EXPECTED_HEAD_DIM:
        raise RuntimeError(
            f"Qwen4-Exp QSA attention {name!r} has head dim {module.head_dim}, "
            f"expected {EXPECTED_HEAD_DIM}."
        )
    if module.num_key_value_groups != EXPECTED_QUERY_HEADS // EXPECTED_KV_HEADS:
        raise RuntimeError(
            f"Qwen4-Exp QSA attention {name!r} has {module.num_key_value_groups} query heads per "
            f"KV head, expected {EXPECTED_QUERY_HEADS // EXPECTED_KV_HEADS}."
        )
    if float(module.attention_dropout) != 0.0:
        raise RuntimeError(
            f"Qwen4-Exp QSA attention {name!r} has dropout {module.attention_dropout}, expected 0."
        )


def _prepare_qwen4_attention(name: str, module: Qwen4ExpTextAttention) -> None:
    """Keep the module's own forward, once, before it is replaced.

    This runs on every pass, including passes over already-patched modules, so it must not capture
    the patched forward as its own reference.
    """

    if not hasattr(module, _REFERENCE_ATTRIBUTE):
        setattr(module, _REFERENCE_ATTRIBUTE, module.forward)


_SPECS = (
    ModulePatchSpec(
        module_type=Qwen4ExpTextAttention,
        forward=_qwen4_qsa_attention_forward,
        handled_key="attention_layers",
        validate=_validate_qwen4_attention,
        prepare=_prepare_qwen4_attention,
        marker=_MARKER,
        freeze_weight=False,
    ),
)


def configure_qwen4_exp_qsa_attention(
    model: torch.nn.Module, *, sequence_length: int = EXPECTED_SEQUENCE_LENGTH
) -> dict[str, Any]:
    """Enable the QSA kernels on one already-loaded Qwen4-Exp model instance.

    The report carries the lengths the kernels serve, so an audit can fail closed when the run is
    outside them instead of quietly paying the reference path.
    """

    report = patch_module_forwards(model, _SPECS)
    report["sequence_length"] = EXPECTED_SEQUENCE_LENGTH
    report["batches"] = sorted(SUPPORTED_BATCHES)
    report["applied"] = sequence_length == EXPECTED_SEQUENCE_LENGTH
    return report


def require_complete_qwen4_exp_qsa_attention(
    report: dict[str, Any],
    *,
    expected_layers: int = EXPECTED_QSA_LAYERS,
    sequence_length: int | None = None,
) -> None:
    """Fail closed unless every Qwen4-Exp indexed attention layer was configured."""

    expected: dict[str, Any] = {
        "attention_layers": expected_layers,
        "sequence_length": EXPECTED_SEQUENCE_LENGTH,
    }
    if sequence_length is not None:
        expected["applied"] = sequence_length == EXPECTED_SEQUENCE_LENGTH
    require_complete_inventory(report, expected, subject=_SUBJECT)
