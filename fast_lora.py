"""PEFT-native fast LoRA wrappers for ordinary linear layers.

The optimized path fuses the second LoRA matrix multiplication with the
addition to the frozen base-layer result via ``torch.addmm``. This avoids
materializing a full-size ``lora_B(lora_A(x))`` output before adding it to the
base result. Unsupported PEFT modes fall back to PEFT's original forwards.

Registration uses PEFT's experimental custom-module API instead of changing
installed PEFT classes globally. Persistent GGUF modules remain frozen and
cannot merge adapter deltas into their packed physical weights. MoE adapters
are handled separately by ``fast_moe_lora.py``.
"""

from typing import Any

import torch
from peft import LoraConfig
from peft.tuners.lora.layer import VARIANT_KWARG_KEYS
from peft.tuners.lora.layer import Linear as PeftLinear
from torch_ggml_ops import mmq
from transformers.integrations.gguf import GGUFLinear
from transformers.integrations.gguf_dequant import GGUFQuantizedTensor

# GGML quantization type IDs supported by the installed torch-ggml-ops wheel.
# The original Qwen integration covered IQ2_S/Q3_K/Q4_K/Q5_K/Q6_K. The
# DeepSeek-V4 wheel adds Q8_0, Q2_K, and IQ2_XXS.
_NATIVE_MMQ_QUANT_TYPES = frozenset({8, 10, 11, 12, 13, 14, 16, 22})


def supports_native_mmq(weight: GGUFQuantizedTensor) -> bool:
    return int(weight.quant_type) in _NATIVE_MMQ_QUANT_TYPES


def _fused_lora_add(
    result: torch.Tensor,
    x: torch.Tensor,
    lora_A: torch.nn.Linear,
    lora_B: torch.nn.Linear,
    dropout: torch.nn.Module,
    scaling: float,
) -> torch.Tensor:
    """Return ``result + scaling * lora_B(lora_A(dropout(x)))``.

    ``torch.addmm`` uses the base result as its matrix input, so the final LoRA
    projection and residual addition produce one full-size output rather than
    separate LoRA-output and summed-output tensors.
    """

    target_dtype = result.dtype
    x_A = torch.matmul(
        dropout(x).to(target_dtype),
        lora_A.weight.to(target_dtype).transpose(0, 1),
    )
    output_shape = result.shape
    output = torch.addmm(
        result.reshape(-1, output_shape[-1]),
        x_A.reshape(-1, x_A.shape[-1]),
        lora_B.weight.to(target_dtype).transpose(0, 1),
        beta=1,
        alpha=scaling,
    ).reshape(output_shape)

    if lora_B.bias is not None:
        output = torch.add(output, lora_B.bias.to(target_dtype), alpha=scaling)
    return output


class _FastLoraForwardMixin:
    """Use the fused path for ordinary single-adapter LoRA, else use PEFT."""

    _cast_input_only_without_autocast = False

    def _base_layer_forward(
        self, x: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        return self.base_layer(x, *args, **kwargs)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        # Keep all less-common PEFT behavior on the stock implementation:
        # disabled/merged adapters, mixed-adapter batches, multiple adapters,
        # DoRA/aLoRA/other variants, and modules not targeted by the adapter.
        adapter_names = kwargs.get("adapter_names")
        active_adapters = self.active_adapters
        if (
            self.disable_adapters
            or self.merged
            or adapter_names is not None
            or len(active_adapters) != 1
        ):
            return super().forward(x, *args, **kwargs)

        active_adapter = active_adapters[0]
        if active_adapter not in self.lora_A or active_adapter in self.lora_variant:
            return super().forward(x, *args, **kwargs)

        self._check_forward_args(x, *args, **kwargs)
        base_kwargs = kwargs.copy()
        base_kwargs.pop("adapter_names", None)
        for key in VARIANT_KWARG_KEYS:
            base_kwargs.pop(key, None)

        result = self._base_layer_forward(x, *args, **base_kwargs)
        lora_A = self.lora_A[active_adapter]
        lora_B = self.lora_B[active_adapter]
        dropout = self.lora_dropout[active_adapter]
        scaling = self.scaling[active_adapter]

        # A no-op dropout can consume x directly in the base-result dtype. In
        # particular, avoid BF16 -> FP32 adapter dtype -> BF16 before the fused
        # operation when PEFT adapters have not yet been converted by TRL.
        dropout_is_noop = isinstance(dropout, torch.nn.Identity) or (
            isinstance(dropout, torch.nn.Dropout) and dropout.p == 0.0
        )
        if not dropout_is_noop and (
            not self._cast_input_only_without_autocast
            or not torch.is_autocast_enabled()
        ):
            x = self._cast_input_dtype(x, lora_A.weight.dtype)

        return _fused_lora_add(result, x, lora_A, lora_B, dropout, scaling)


class FastLoraLinear(_FastLoraForwardMixin, PeftLinear):
    """Drop-in PEFT LoRA wrapper for ordinary floating linear modules."""


class FastGGUFLoraLinear(FastLoraLinear):
    """Fast LoRA wrapper for frozen packed ``GGUFLinear`` modules.

    Supported quantization types use dense MMQ in both directions. Other
    packed types stay on ``GGUFLinear.forward`` and its generic input Jacobian.
    """

    def _base_layer_forward(
        self, x: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        base = self.base_layer
        if (
            not isinstance(base.weight, GGUFQuantizedTensor)
            or not supports_native_mmq(base.weight)
            or base.input_permutation is not None
            or base.output_permutation is not None
        ):
            return base(x, *args, **kwargs)
        if args or kwargs:
            raise TypeError(
                "The packed GGUF MMQ linear path accepts only its input tensor."
            )
        if base.compute_dtype != torch.bfloat16:
            raise RuntimeError(
                "The packed GGUF MMQ linear path requires BF16 compute_dtype."
            )

        payload = base.weight.as_subclass(torch.Tensor)
        result = mmq(
            x,
            payload,
            int(base.weight.quant_type),
            base.out_features,
        )
        if base.bias is not None:
            result = result + base.bias.to(device=result.device, dtype=result.dtype)
        return result

    def merge(
        self, safe_merge: bool = False, adapter_names: list[str] | None = None
    ) -> None:
        raise RuntimeError(
            "GGUF LoRA adapters cannot be merged into packed base weights."
        )

    def unmerge(self) -> None:
        raise RuntimeError(
            "GGUF LoRA adapters cannot be unmerged because merging is unsupported."
        )


def register_fast_lora(lora_config: LoraConfig) -> LoraConfig:
    """Register fast ordinary-linear wrappers on one ``LoraConfig``.

    PEFT currently exposes custom LoRA modules through the experimental private
    ``LoraConfig._register_custom_module`` API. Registration is config-local:
    no PEFT or Transformers class is monkey-patched process-wide.
    """

    register = getattr(lora_config, "_register_custom_module", None)
    if register is None:
        raise RuntimeError(
            "This PEFT version has no LoraConfig._register_custom_module API. "
            "Cannot install fast LoRA without a global monkey patch."
        )

    # GGUFLinear subclasses nn.Linear, so its merge-safe wrapper must be checked first.
    register(
        {
            GGUFLinear: FastGGUFLoraLinear,
            torch.nn.Linear: FastLoraLinear,
        }
    )
    return lora_config
