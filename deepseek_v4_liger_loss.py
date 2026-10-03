"""DeepSeek V4 Liger losses, including the packed Q8_0 LM-head path.

The reference helper intentionally remains available for correctness audits.
The model-instance patch uses the shared packed helper: the shared
`packed_liger_loss` module owns the chunked Q8_1 MMQ, the in-place Liger
cross-entropy, the packed logical input Jacobian, and the scoped-forward
contract. This module owns the DeepSeek entry points and its validated
constants (Q8_0 head, 512-row chunks). No logical vocabulary matrix or
full-sequence logits tensor is created during training.
"""

from types import MethodType
from typing import Any, cast

import torch
from liger_kernel.transformers.model.loss_utils import LigerForCausalLMLoss
from liger_kernel.transformers.model.output_classes import (
    LigerMoeCausalLMOutputWithPast,
)
from transformers.cache_utils import Cache
from transformers.integrations.gguf.modules import GgufLinear
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4ForCausalLM,
    load_balancing_loss_func,
)
from transformers.utils import can_return_tuple

from packed_liger_loss import (
    PackedLossResult,
    assembled_scoped_output,
    packed_q8_liger_for_causal_lm_loss,
    scoped_packed_causal_lm_loss,
)

_PACKED_LM_HEAD_CHUNK_SIZE = 512
_PACKED_LM_HEAD_QUANT_TYPE = 8
_PACKED_LM_HEAD_QUANT_NAME = "Q8_0"
_PACKED_LM_HEAD_LOSS_NAME = "Packed DeepSeek Q8_0 LM-head loss"
_PATCH_MARKER = "_patched_liger_loss"


def deepseek_v4_liger_causal_lm_loss(
    hidden_states: torch.Tensor,
    lm_head: GgufLinear,
    labels: torch.Tensor,
    *,
    hidden_size: int,
    loss_kwargs: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Reference loss that materializes the frozen head for comparison tests."""

    if lm_head.weight.requires_grad:
        raise RuntimeError("DeepSeek V4 reference loss requires a frozen LM head.")
    loss_kwargs = {} if loss_kwargs is None else loss_kwargs
    logical_weight = lm_head.materialize_logical_weight(
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    loss = LigerForCausalLMLoss(
        hidden_states=hidden_states,
        lm_head_weight=logical_weight,
        labels=labels,
        hidden_size=hidden_size,
        **loss_kwargs,
    )
    if not isinstance(loss, torch.Tensor):
        raise TypeError(
            f"Liger causal-LM loss returned {type(loss).__name__}, expected Tensor."
        )
    return loss


def deepseek_v4_packed_liger_causal_lm_loss(
    hidden_states: torch.Tensor,
    lm_head: GgufLinear,
    labels: torch.Tensor,
    *,
    hidden_size: int,
    num_items_in_batch: int | torch.Tensor | None = None,
    ignore_index: int = -100,
    shift_labels: torch.Tensor | None = None,
    final_logit_softcapping: float | None = None,
    return_token_accuracy: bool = False,
    return_predicted_tokens: bool = False,
    **kwargs: object,
) -> PackedLossResult:
    """Public scoped packed loss used by the optimized model forward."""

    return packed_q8_liger_for_causal_lm_loss(
        hidden_states,
        lm_head,
        labels,
        hidden_size=hidden_size,
        chunk_size=_PACKED_LM_HEAD_CHUNK_SIZE,
        expected_quant_type=_PACKED_LM_HEAD_QUANT_TYPE,
        quant_name=_PACKED_LM_HEAD_QUANT_NAME,
        loss_name=_PACKED_LM_HEAD_LOSS_NAME,
        expected_hidden_size=None,
        num_items_in_batch=num_items_in_batch,
        ignore_index=ignore_index,
        shift_labels=shift_labels,
        final_logit_softcapping=final_logit_softcapping,
        return_token_accuracy=return_token_accuracy,
        return_predicted_tokens=return_predicted_tokens,
        **kwargs,
    )


@can_return_tuple
def _deepseek_v4_liger_forward(
    self: DeepseekV4ForCausalLM,
    input_ids: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    past_key_values: Cache | None = None,
    inputs_embeds: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    use_cache: bool | None = None,
    output_router_logits: bool | None = None,
    logits_to_keep: int | torch.Tensor = 0,
    skip_logits: bool | None = None,
    shift_labels: torch.Tensor | None = None,
    **kwargs: object,
) -> LigerMoeCausalLMOutputWithPast:
    """DeepSeek V4 forward using the scoped packed Q8_0 LM-head loss."""

    output_router_logits = (
        output_router_logits
        if output_router_logits is not None
        else self.config.output_router_logits
    )

    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_router_logits=output_router_logits,
        **kwargs,
    )

    loss, logits, token_accuracy, predicted_tokens = scoped_packed_causal_lm_loss(
        self,
        outputs,
        labels=labels,
        shift_labels=shift_labels,
        logits_to_keep=logits_to_keep,
        skip_logits=skip_logits,
        packed_loss=deepseek_v4_packed_liger_causal_lm_loss,
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


def apply_deepseek_v4_liger_loss(
    model: torch.nn.Module,
) -> DeepseekV4ForCausalLM:
    """Patch one model instance with the packed Q8_0 scoped loss."""

    get_base_model = getattr(model, "get_base_model", None)
    base = get_base_model() if callable(get_base_model) else model
    if not isinstance(base, DeepseekV4ForCausalLM):
        raise TypeError(
            "DeepSeek V4 Liger loss requires DeepseekV4ForCausalLM, got "
            f"{type(base).__name__}."
        )
    if not isinstance(base.lm_head, GgufLinear):
        raise TypeError(
            "DeepSeek V4 GGUF loss requires GgufLinear lm_head, got "
            f"{type(base.lm_head).__name__}."
        )
    if getattr(base, _PATCH_MARKER, False):
        return base
    base.forward = MethodType(_deepseek_v4_liger_forward, base)
    base.__dict__[_PATCH_MARKER] = True
    return base
