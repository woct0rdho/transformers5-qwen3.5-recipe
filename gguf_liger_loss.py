"""Packed GGUF-aware Liger-style cross-entropy for Qwen3.5-MoE.

The frozen LM head remains in its authoritative GGUF representation. The shared
`packed_liger_loss` module owns the chunked Q8_1 MMQ, the in-place Liger
cross-entropy, the packed logical input Jacobian, and the scoped-forward
contract. This module owns the Qwen entry points and its validated constants
(hidden size 2048, Q6_K head, 256-row chunks).
"""

from types import MethodType
from typing import cast

import torch
from liger_kernel.transformers.model.output_classes import (
    LigerMoeCausalLMOutputWithPast,
)
from transformers.cache_utils import Cache
from transformers.integrations.gguf.modules import GgufLinear
from transformers.modeling_outputs import MoeModelOutputWithPast
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForCausalLM,
    load_balancing_loss_func,
)
from transformers.utils import can_return_tuple

from packed_liger_loss import (
    PackedLossResult,
    assembled_scoped_output,
    packed_q8_liger_for_causal_lm_loss,
    scoped_packed_causal_lm_loss,
)

_PACKED_LM_HEAD_CHUNK_SIZE = 256
_PACKED_LM_HEAD_QUANT_TYPE = 14
_PACKED_LM_HEAD_QUANT_NAME = "Q6_K"
_PACKED_LM_HEAD_LOSS_NAME = "Packed GGUF LM-head loss"
_PACKED_LM_HEAD_HIDDEN_SIZE = 2048


def _packed_q8_liger_for_causal_lm_loss(
    hidden_states: torch.Tensor,
    lm_head: GgufLinear,
    labels: torch.Tensor,
    hidden_size: int,
    num_items_in_batch: int | torch.Tensor | None = None,
    ignore_index: int = -100,
    shift_labels: torch.Tensor | None = None,
    final_logit_softcapping: float | None = None,
    return_token_accuracy: bool = False,
    return_predicted_tokens: bool = False,
    **kwargs: object,
) -> PackedLossResult:
    return packed_q8_liger_for_causal_lm_loss(
        hidden_states,
        lm_head,
        labels,
        hidden_size=hidden_size,
        chunk_size=_PACKED_LM_HEAD_CHUNK_SIZE,
        expected_quant_type=_PACKED_LM_HEAD_QUANT_TYPE,
        quant_name=_PACKED_LM_HEAD_QUANT_NAME,
        loss_name=_PACKED_LM_HEAD_LOSS_NAME,
        expected_hidden_size=_PACKED_LM_HEAD_HIDDEN_SIZE,
        num_items_in_batch=num_items_in_batch,
        ignore_index=ignore_index,
        shift_labels=shift_labels,
        final_logit_softcapping=final_logit_softcapping,
        return_token_accuracy=return_token_accuracy,
        return_predicted_tokens=return_predicted_tokens,
        **kwargs,
    )


@can_return_tuple
def gguf_liger_lce_forward(
    self: Qwen3_5MoeForCausalLM,
    input_ids: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    past_key_values: Cache | None = None,
    inputs_embeds: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    use_cache: bool | None = None,
    output_attentions: bool | None = None,
    output_hidden_states: bool | None = None,
    output_router_logits: bool | None = None,
    mm_token_type_ids: torch.Tensor | None = None,
    cache_position: torch.Tensor | None = None,
    logits_to_keep: int | torch.Tensor = 0,
    skip_logits: bool | None = None,
    shift_labels: torch.Tensor | None = None,
    **kwargs: object,
) -> LigerMoeCausalLMOutputWithPast:
    """Qwen3.5-MoE forward using the chunked packed GGUF LM-head loss."""

    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_router_logits = (
        output_router_logits
        if output_router_logits is not None
        else self.config.output_router_logits
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )

    outputs: MoeModelOutputWithPast = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        output_router_logits=output_router_logits,
        mm_token_type_ids=mm_token_type_ids,
        cache_position=cache_position,
        **kwargs,
    )

    loss, logits, token_accuracy, predicted_tokens = scoped_packed_causal_lm_loss(
        self,
        outputs,
        labels=labels,
        shift_labels=shift_labels,
        logits_to_keep=logits_to_keep,
        skip_logits=skip_logits,
        packed_loss=_packed_q8_liger_for_causal_lm_loss,
        loss_kwargs=kwargs,
    )

    aux_loss: torch.Tensor | None = None
    if output_router_logits:
        aux_loss = cast(
            torch.Tensor,
            load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            ),
        )
        if labels is not None:
            if loss is None:
                raise RuntimeError("router auxiliary loss requires a primary loss")
            loss = loss + self.router_aux_loss_coef * aux_loss.to(loss.device)

    return assembled_scoped_output(
        outputs,
        loss=loss,
        aux_loss=aux_loss,
        logits=logits,
        token_accuracy=token_accuracy,
        predicted_tokens=predicted_tokens,
    )


def apply_gguf_liger_fused_linear_cross_entropy(
    model: torch.nn.Module,
) -> Qwen3_5MoeForCausalLM:
    """Patch one loaded text model with the GGUF-aware Liger loss forward."""

    get_base_model = getattr(model, "get_base_model", None)
    target = get_base_model() if callable(get_base_model) else model
    if not isinstance(target, Qwen3_5MoeForCausalLM):
        raise TypeError(
            "GGUF-aware Liger loss requires Qwen3_5MoeForCausalLM after unwrapping PEFT, "
            f"got {type(target).__name__}."
        )
    if not isinstance(target.lm_head, GgufLinear):
        raise TypeError(
            f"GGUF-aware Liger loss requires GgufLinear, got {type(target.lm_head).__name__}."
        )
    target.forward = MethodType(gguf_liger_lce_forward, target)
    return target
