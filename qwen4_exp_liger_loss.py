"""Packed GGUF-aware Liger-style cross-entropy for Qwen4-Exp.

The frozen LM head remains in its authoritative GGUF representation. The shared
`packed_liger_loss` module owns the chunked Q8_1 MMQ, the in-place Liger
cross-entropy, the packed logical input Jacobian, and the scoped-forward
contract. This module owns the Qwen4-Exp entry points and its validated
constants (hidden size 2560, Q5_K head, 256-row chunks, which `torch_ggml_ops`
deploys for the forward and for the split-contraction input gradient).

Training therefore never materializes the `[rows, 248320]` logits tensor, the
head's logical BF16 matrix, or an FP32 cross-entropy upcast. `logits` only
appears when a forward runs without labels, and that path keeps the head's own
forward.
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
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpForCausalLM,
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
_PACKED_LM_HEAD_QUANT_TYPE = 13
_PACKED_LM_HEAD_QUANT_NAME = "Q5_K"
_PACKED_LM_HEAD_LOSS_NAME = "Packed Qwen4-Exp Q5_K LM-head loss"
_PACKED_LM_HEAD_HIDDEN_SIZE = 2560
_PATCH_MARKER = "_patched_qwen4_exp_liger_loss"


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
def qwen4_exp_liger_lce_forward(
    self: Qwen4ExpForCausalLM,
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
    """Qwen4-Exp forward using the chunked packed GGUF LM-head loss."""

    output_router_logits = (
        output_router_logits
        if output_router_logits is not None
        else self.config.output_router_logits
    )

    outputs: MoeModelOutputWithPast = self.model(
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


def apply_qwen4_exp_liger_fused_linear_cross_entropy(
    model: torch.nn.Module,
) -> Qwen4ExpForCausalLM:
    """Patch one loaded text model with the GGUF-aware packed Q5_K loss forward."""

    get_base_model = getattr(model, "get_base_model", None)
    target = get_base_model() if callable(get_base_model) else model
    if not isinstance(target, Qwen4ExpForCausalLM):
        raise TypeError(
            "Packed Qwen4-Exp LM-head loss requires Qwen4ExpForCausalLM after unwrapping PEFT, "
            f"got {type(target).__name__}."
        )
    if not isinstance(target.lm_head, GgufLinear):
        raise TypeError(
            f"Packed Qwen4-Exp LM-head loss requires GgufLinear, got {type(target.lm_head).__name__}."
        )
    if getattr(target, _PATCH_MARKER, False):
        return target
    target.forward = MethodType(qwen4_exp_liger_lce_forward, target)
    target.__dict__[_PATCH_MARKER] = True
    return target
