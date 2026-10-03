"""Shared chunked packed-GGUF LM-head loss calculation and forward contract.

Qwen3.5-MoE and DeepSeek V4 both keep the LM head in its authoritative GGUF
representation and train through the same calculation:

1. quantize bounded BF16 hidden-state chunks to Q8_1,
2. multiply them by the packed head with native MMQ,
3. run Liger's cross-entropy kernel in place so its BF16 logits become
   cotangents,
4. decode the frozen packed input Jacobian directly into the hidden gradient.

No logical LM-head matrix or full-sequence logits tensor is materialized. The
model-specific modules own their public entry points, validation constants
(quantization type, validated hidden size, chunk size), and reference oracles,
while this module owns the calculation, the scoped-forward contract that decides
between the packed loss and the ordinary materialized-logits loss, and the
public output assembly that surfaces the optional token-accuracy and
predicted-token results.
"""

from collections.abc import Callable
from typing import Protocol, cast

import torch
import triton
from liger_kernel.ops.cross_entropy import liger_cross_entropy_kernel
from liger_kernel.ops.fused_linear_cross_entropy import MAX_FUSED_SIZE
from liger_kernel.ops.utils import (
    amp_custom_bwd,
    amp_custom_fwd,
    element_mul_kernel,
    is_hip,
)
from liger_kernel.transformers.model.loss_utils import unpack_cross_entropy_result
from liger_kernel.transformers.model.output_classes import (
    LigerMoeCausalLMOutputWithPast,
)
from torch import nn
from torch_ggml_ops import mmq_grad_input_inplace, mmq_inplace
from transformers.integrations.gguf.gguf_quantized_parameter import (
    GgufQuantizedParameter,
)
from transformers.integrations.gguf.modules import GgufLinear

PackedLossResult = (
    torch.Tensor
    | tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]
)
ScopedLossResult = tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]


class _ScopedPackedConfig(Protocol):
    hidden_size: int


class _ScopedPackedModel(Protocol):
    """Model surface the scoped packed loss needs, checked once at entry."""

    training: bool
    vocab_size: int
    lm_head: GgufLinear
    config: _ScopedPackedConfig

    def loss_function(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor | None,
        vocab_size: int,
        shift_labels: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor: ...


class _ModelOutputWithHiddenStates(Protocol):
    @property
    def last_hidden_state(self) -> torch.Tensor | None: ...


def scale_input_gradient_in_place(
    grad_input: torch.Tensor, grad_output: torch.Tensor
) -> None:
    """Multiply a retained hidden-state gradient by the scalar loss gradient.

    Liger's `fused_linear_cross_entropy_backward` opens with
    `torch.equal(grad_output, torch.tensor(1.0, ...))`, which synchronizes the
    device on every step. That test only decides whether this multiply can be
    skipped, so running it unconditionally is cheaper than the stall.
    """

    rows, hidden = grad_input.shape
    element_mul_kernel[(rows,)](
        grad_input,
        grad_input.stride(-2),
        grad_output,
        hidden,
        BLOCK_SIZE=min(MAX_FUSED_SIZE, triton.next_power_of_2(hidden)),
        num_warps=32 if not is_hip() else 16,
    )


def _packed_q8_linear_cross_entropy_forward(
    input: torch.Tensor,
    packed_weight: torch.Tensor,
    target: torch.Tensor,
    quant_type: int,
    out_features: int,
    chunk_size: int,
    ignore_index: int,
    lse_square_scale: float,
    label_smoothing: float,
    softcap: float | None,
    return_token_accuracy: bool,
    return_predicted_tokens: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor,
]:
    """Run chunked Q8_1 MMQ, in-place Liger CE, and packed dHidden.

    The returned loss is the un-normalized sum over rows. The caller applies the
    mean reduction on device, so the non-ignored count never reaches the host.
    """

    if input.ndim != 2:
        raise ValueError(
            f"Packed LM-head loss expects two-dimensional hidden states, got {input.shape}."
        )
    if input.dtype != torch.bfloat16:
        raise TypeError(
            f"Packed LM-head loss requires BF16 hidden states, got {input.dtype}."
        )
    if target.ndim != 1 or target.shape[0] != input.shape[0]:
        raise ValueError(
            "Packed LM-head targets must be one-dimensional and match the hidden-state row count."
        )
    if target.dtype != torch.int64:
        raise TypeError(
            f"Packed LM-head targets require torch.int64, got {target.dtype}."
        )
    if chunk_size <= 0:
        raise ValueError(
            f"Packed LM-head chunk_size must be positive, got {chunk_size}."
        )

    rows, hidden_size = input.shape
    # Kept on device: dividing the accuracy by a tensor stays off the host, and
    # the kernel's reduction no longer reads this count.
    total_n_non_ignore = (target != ignore_index).sum()
    block_size = min(MAX_FUSED_SIZE, triton.next_power_of_2(out_features))

    grad_input = torch.empty_like(input)
    buffer_rows = min(rows, chunk_size)
    logits_buffer = torch.empty(
        (buffer_rows, out_features), dtype=input.dtype, device=input.device
    )
    workspace_buffer = torch.empty(
        buffer_rows * hidden_size // 128 * 144,
        dtype=torch.uint8,
        device=input.device,
    )
    loss_1d = torch.zeros(rows, dtype=torch.float32, device=input.device)
    token_accuracy_1d = (
        torch.zeros(rows, dtype=torch.float32, device=input.device)
        if return_token_accuracy
        else None
    )
    predicted_tokens_1d = (
        torch.full((rows,), -1, dtype=torch.int64, device=input.device)
        if return_predicted_tokens
        else None
    )

    for start in range(0, rows, chunk_size):
        end = min(start + chunk_size, rows)
        input_chunk = input[start:end]
        chunk_rows = end - start
        logits_chunk = logits_buffer[:chunk_rows]
        workspace_chunk = workspace_buffer[: chunk_rows * hidden_size // 128 * 144]
        mmq_inplace(
            input_chunk,
            packed_weight,
            quant_type,
            out_features,
            logits_chunk,
            workspace_chunk,
        )
        target_chunk = target[start:end].contiguous()
        loss_1d_slice = loss_1d[start:end]
        if return_token_accuracy:
            if token_accuracy_1d is None:
                raise RuntimeError("token accuracy storage was not allocated")
            token_accuracy_1d_slice = token_accuracy_1d[start:end]
        else:
            token_accuracy_1d_slice = None
        if return_predicted_tokens:
            if predicted_tokens_1d is None:
                raise RuntimeError("predicted-token storage was not allocated")
            predicted_tokens_1d_slice = predicted_tokens_1d[start:end]
        else:
            predicted_tokens_1d_slice = None
        token_accuracy_stride = (
            token_accuracy_1d_slice.stride(-1)
            if token_accuracy_1d_slice is not None
            else 0
        )
        predicted_tokens_stride = (
            predicted_tokens_1d_slice.stride(-1)
            if predicted_tokens_1d_slice is not None
            else 0
        )

        liger_cross_entropy_kernel[(end - start,)](
            X_ptr=logits_chunk,
            X_stride=logits_chunk.stride(-2),
            Y_ptr=target_chunk,
            Y_stride=target_chunk.stride(-1),
            weight_ptr=None,
            loss_ptr=loss_1d_slice,
            z_loss_ptr=None,
            loss_stride=loss_1d_slice.stride(-1),
            token_accuracy_ptr=token_accuracy_1d_slice,
            token_accuracy_stride=token_accuracy_stride,
            predicted_tokens_ptr=predicted_tokens_1d_slice,
            predicted_tokens_stride=predicted_tokens_stride,
            n_cols=out_features,
            n_non_ignore=1,
            sum_non_ignore_weight=1,
            weight_sum=0.0,
            ignore_index=ignore_index,
            lse_square_scale=lse_square_scale,
            label_smoothing=label_smoothing,
            reduction="sum",
            softcap=softcap,
            RETURN_Z_LOSS=False,
            RETURN_TOKEN_ACCURACY=return_token_accuracy,
            RETURN_PREDICTED_TOKENS=return_predicted_tokens,
            HAS_WEIGHT=False,
            HAS_SOFTCAPPING=softcap is not None,
            HAS_GRADIENTS=True,
            BLOCK_SIZE=block_size,
            num_warps=16 if is_hip() else 32,
        )

        mmq_grad_input_inplace(
            logits_chunk,
            packed_weight,
            quant_type,
            hidden_size,
            grad_input[start:end],
        )

    loss = torch.sum(loss_1d)
    if return_token_accuracy:
        if token_accuracy_1d is None:
            raise RuntimeError("token accuracy storage was not allocated")
        token_accuracy = torch.sum(token_accuracy_1d) / total_n_non_ignore
    else:
        token_accuracy = None
    return loss, token_accuracy, predicted_tokens_1d, grad_input


class _PackedQ8LigerLinearCrossEntropyFunction(torch.autograd.Function):
    """Liger-style fused loss over one frozen packed GGUF LM head."""

    @staticmethod
    @amp_custom_fwd
    def forward(
        ctx,
        input: torch.Tensor,
        packed_weight: torch.Tensor,
        target: torch.Tensor,
        quant_type: int,
        out_features: int,
        chunk_size: int,
        ignore_index: int,
        lse_square_scale: float,
        label_smoothing: float,
        softcap: float | None,
        return_token_accuracy: bool,
        return_predicted_tokens: bool,
    ):
        loss, token_accuracy, predicted_tokens, grad_input = (
            _packed_q8_linear_cross_entropy_forward(
                input,
                packed_weight,
                target,
                quant_type,
                out_features,
                chunk_size,
                ignore_index,
                lse_square_scale,
                label_smoothing,
                softcap,
                return_token_accuracy,
                return_predicted_tokens,
            )
        )
        ctx.save_for_backward(grad_input.detach())
        non_differentiable = tuple(
            output
            for output in (token_accuracy, predicted_tokens)
            if output is not None
        )
        if non_differentiable:
            ctx.mark_non_differentiable(*non_differentiable)
        return loss, token_accuracy, predicted_tokens

    @staticmethod
    @amp_custom_bwd
    def backward(ctx, grad_output, grad_token_accuracy, grad_predicted_tokens):
        del grad_token_accuracy, grad_predicted_tokens
        if torch.is_grad_enabled():
            raise RuntimeError(
                "Packed Q8_1 GGUF LM-head loss does not support higher-order gradients"
            )
        (grad_input,) = ctx.saved_tensors
        scale_input_gradient_in_place(grad_input, grad_output)
        return (grad_input,) + (None,) * 11


def packed_q8_liger_for_causal_lm_loss(
    hidden_states: torch.Tensor,
    lm_head: GgufLinear,
    labels: torch.Tensor,
    *,
    hidden_size: int,
    chunk_size: int,
    expected_quant_type: int,
    quant_name: str,
    loss_name: str,
    expected_hidden_size: int | None = None,
    num_items_in_batch: int | torch.Tensor | None = None,
    ignore_index: int = -100,
    shift_labels: torch.Tensor | None = None,
    final_logit_softcapping: float | None = None,
    return_token_accuracy: bool = False,
    return_predicted_tokens: bool = False,
    **kwargs: object,
) -> PackedLossResult:
    """Run the packed head under one model's validated contract.

    `chunk_size`, `expected_quant_type`/`quant_name`,
    `expected_hidden_size`, and the `loss_name` used in errors are the only
    model-owned parameters. The calculation is identical for every caller.
    """

    label_smoothing = kwargs.get("label_smoothing", 0.0)
    lse_square_scale = kwargs.get("lse_square_scale", 0.0)
    unsupported = (
        ("z-loss output", bool(kwargs.get("return_z_loss", False))),
        ("token scaling", bool(kwargs.get("use_token_scaling", False))),
        ("class weights", kwargs.get("ce_weight") is not None),
        ("a fused bias", kwargs.get("bias") is not None),
        ("label smoothing", label_smoothing != 0.0),
        ("LSE square scaling", lse_square_scale != 0.0),
        ("logit softcapping", final_logit_softcapping is not None),
    )
    for name, enabled in unsupported:
        if enabled:
            raise RuntimeError(f"{loss_name} does not support {name}.")
    accum_dtype = kwargs.get("accum_dtype")
    if accum_dtype not in {None, torch.float32}:
        raise RuntimeError(f"{loss_name} supports only FP32 internal accumulation.")
    if not isinstance(lm_head.weight, GgufQuantizedParameter):
        raise TypeError(f"{loss_name} requires a GgufQuantizedParameter weight.")
    if lm_head.compute_dtype != torch.bfloat16:
        raise RuntimeError(f"{loss_name} requires BF16 compute_dtype.")
    if lm_head.in_features != hidden_size or hidden_states.shape[-1] != hidden_size:
        raise RuntimeError(f"{loss_name} hidden size does not match the LM head.")
    if expected_hidden_size is not None and hidden_size != expected_hidden_size:
        raise RuntimeError(
            f"{loss_name} is validated only for hidden size {expected_hidden_size}."
        )
    quant_type = int(lm_head.weight.quant_type)
    if quant_type != expected_quant_type:
        raise RuntimeError(f"{loss_name} is validated only for {quant_name} weights.")
    if lm_head.weight.requires_grad:
        raise RuntimeError(f"{loss_name} weights must remain frozen.")

    if shift_labels is None:
        labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()

    hidden_states = hidden_states.reshape(-1, hidden_size)
    shift_labels = shift_labels.reshape(-1).to(hidden_states.device)
    payload = lm_head.weight.as_subclass(torch.Tensor)
    loss, token_accuracy, predicted_tokens = (
        _PackedQ8LigerLinearCrossEntropyFunction.apply(
            hidden_states,
            payload,
            shift_labels,
            quant_type,
            lm_head.out_features,
            chunk_size,
            ignore_index,
            0.0,
            0.0,
            None,
            return_token_accuracy,
            return_predicted_tokens,
        )
    )
    # The kernel produces an un-normalized sum, and the reduction is applied here
    # on device so the non-ignored count never round-trips through the host. As a
    # plain tensor division it also scales the backward through autograd.
    denominator = (
        num_items_in_batch
        if num_items_in_batch is not None
        else (shift_labels != ignore_index).sum()
    )
    loss = loss / denominator
    if return_token_accuracy or return_predicted_tokens:
        return loss, None, token_accuracy, predicted_tokens
    return loss


def validate_packed_lm_head(lm_head: torch.nn.Module) -> None:
    """Fail before the packed path when the head cannot own that boundary."""

    if not isinstance(lm_head, GgufLinear):
        raise TypeError(
            "Packed GGUF LM-head loss requires a GgufLinear LM head, got "
            f"{type(lm_head).__name__}."
        )
    if lm_head.bias is not None:
        raise RuntimeError("Packed GGUF LM-head loss requires a bias-free LM head.")


def scoped_packed_causal_lm_loss(
    model: torch.nn.Module,
    outputs: _ModelOutputWithHiddenStates,
    *,
    labels: torch.Tensor | None,
    shift_labels: torch.Tensor | None,
    logits_to_keep: int | torch.Tensor,
    skip_logits: bool | None,
    packed_loss: Callable[..., PackedLossResult],
    loss_kwargs: dict[str, object],
) -> ScopedLossResult:
    """Return `(loss, logits, token_accuracy, predicted_tokens)`.

    Both model forwards share this decision so the packed boundary fails closed
    identically: training with labels must use the packed loss, a packed step
    must not request retained logits, and evaluation keeps the ordinary
    materialized-logits loss. Unsupported inputs raise instead of silently
    changing the loss or dropping logits.
    """

    scope = cast(_ScopedPackedModel, model)
    hidden_states = outputs.last_hidden_state
    if hidden_states is None:
        raise RuntimeError(f"{type(model).__name__} did not return hidden states")
    slice_indices = (
        slice(-logits_to_keep, None)
        if isinstance(logits_to_keep, int)
        else logits_to_keep
    )

    if skip_logits is None:
        skip_logits = scope.training and (
            labels is not None or shift_labels is not None
        )
    if skip_logits and not (isinstance(logits_to_keep, int) and logits_to_keep == 0):
        raise RuntimeError(
            "Packed GGUF LM-head loss requires logits_to_keep=0 during training."
        )
    if skip_logits and labels is None and shift_labels is None:
        raise RuntimeError("Packed GGUF LM-head loss requires labels or shift_labels.")
    if (
        scope.training
        and not skip_logits
        and (labels is not None or shift_labels is not None)
    ):
        raise RuntimeError(
            "Packed GGUF training with labels requires the no-full-logits fused loss."
        )

    if skip_logits:
        validate_packed_lm_head(scope.lm_head)
        result = packed_loss(
            hidden_states=hidden_states[:, slice_indices, :],
            lm_head=scope.lm_head,
            labels=labels,
            shift_labels=shift_labels,
            hidden_size=scope.config.hidden_size,
            **loss_kwargs,
        )
        loss, _, token_accuracy, predicted_tokens = unpack_cross_entropy_result(result)
        return loss, None, token_accuracy, predicted_tokens

    logits = scope.lm_head(hidden_states[:, slice_indices, :])
    loss = None
    if labels is not None or shift_labels is not None:
        loss = scope.loss_function(
            logits=logits,
            labels=labels,
            shift_labels=shift_labels,
            vocab_size=scope.vocab_size,
            **loss_kwargs,
        )
    return loss, logits, None, None


def assembled_scoped_output(
    outputs: object,
    *,
    loss: torch.Tensor | None,
    aux_loss: torch.Tensor | None,
    logits: torch.Tensor | None,
    token_accuracy: torch.Tensor | None = None,
    predicted_tokens: torch.Tensor | None = None,
) -> LigerMoeCausalLMOutputWithPast:
    """Assemble the shared public output so the optional fields cannot drift.

    Both model forwards return `LigerMoeCausalLMOutputWithPast` through this
    boundary, so `token_accuracy` and `predicted_tokens` are surfaced
    whenever the caller requested them instead of being computed and dropped by
    one model.
    """

    return LigerMoeCausalLMOutputWithPast(
        loss=cast(torch.FloatTensor | None, loss),
        aux_loss=cast(torch.FloatTensor | None, aux_loss),
        logits=cast(torch.FloatTensor | None, logits),
        past_key_values=getattr(outputs, "past_key_values", None),
        hidden_states=getattr(outputs, "hidden_states", None),
        attentions=getattr(outputs, "attentions", None),
        router_logits=getattr(outputs, "router_logits", None),
        token_accuracy=cast(torch.FloatTensor | None, token_accuracy),
        predicted_tokens=cast(torch.LongTensor | None, predicted_tokens),
    )
