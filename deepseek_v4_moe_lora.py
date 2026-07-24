"""DeepSeek V4 complete-expert GGUF LoRA registration."""

from typing import Any

import torch
from peft import LoraConfig
from transformers.integrations.gguf import (
    ALL_GGUF_EXPERTS_FUNCTIONS,
    DeepseekV4GGUFExperts,
    GGUFExperts,
)

from deepseek_v4_lora import DEEPSEEK_V4_TARGET_MODULES_PATTERN
from fast_moe_lora import (
    FastGGUFMoeLora,
    _ExpertLoraWeights,
    qwen3_5_moe_gguf_mmq_aiter_lora_forward,
)

EXPERTS_IMPLEMENTATION = "deepseek_v4_gguf_dequant_aiter_lora"
_LORA_WEIGHTS_KWARG = "_deepseek_v4_gguf_lora_weights"


class DeepseekV4GGUFMoeLora(FastGGUFMoeLora):
    """PEFT wrapper for all gate, up, and down transforms of one MoE layer."""

    def forward(
        self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        adapter_names = kwargs.pop("adapter_names", None)
        lora_weights = self._active_lora_weights(adapter_names)
        experts = self.get_base_layer()
        if not isinstance(experts, DeepseekV4GGUFExperts):
            raise TypeError(
                "DeepSeek V4 expert LoRA requires DeepseekV4GGUFExperts, got "
                f"{type(experts).__name__}."
            )
        if experts.config._experts_implementation != EXPERTS_IMPLEMENTATION:
            raise RuntimeError(
                f"DeepSeek V4 expert LoRA requires experts_implementation={EXPERTS_IMPLEMENTATION!r}, "
                f"got {experts.config._experts_implementation!r}."
            )
        kwargs[_LORA_WEIGHTS_KWARG] = lora_weights
        return self.base_layer(hidden_states, *args, **kwargs)


def deepseek_v4_gguf_dequant_aiter_lora_forward(
    self: GGUFExperts,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    _deepseek_v4_gguf_lora_weights: _ExpertLoraWeights | None = None,
) -> torch.Tensor:
    """Run split packed experts with generic dequantization and AITER grouped MM."""

    if not isinstance(self, DeepseekV4GGUFExperts):
        raise TypeError(
            f"{EXPERTS_IMPLEMENTATION} requires DeepseekV4GGUFExperts, got {type(self).__name__}."
        )
    return qwen3_5_moe_gguf_mmq_aiter_lora_forward(
        self,
        hidden_states,
        top_k_index,
        top_k_weights,
        _qwen3_5_moe_gguf_lora_weights=_deepseek_v4_gguf_lora_weights,
    )


def register_deepseek_v4_moe_lora(
    lora_config: LoraConfig, model: torch.nn.Module
) -> LoraConfig:
    """Register the DeepSeek expert backend and its config-local PEFT wrapper."""

    register = getattr(lora_config, "_register_custom_module", None)
    if register is None:
        raise RuntimeError(
            "This PEFT version has no LoraConfig._register_custom_module API."
        )
    if lora_config.target_parameters:
        raise ValueError(
            "DeepSeek GGUF experts target complete modules, not parameters."
        )
    if isinstance(lora_config.target_modules, str):
        if lora_config.target_modules != DEEPSEEK_V4_TARGET_MODULES_PATTERN:
            raise ValueError(
                "DeepSeek V4 LoRA uses one exact indexer-excluding target pattern."
            )
    else:
        target_modules = set(lora_config.target_modules or ())
        target_modules.add("experts")
        lora_config.target_modules = target_modules
    ALL_GGUF_EXPERTS_FUNCTIONS[EXPERTS_IMPLEMENTATION] = (
        deepseek_v4_gguf_dequant_aiter_lora_forward
    )
    model.set_experts_implementation(EXPERTS_IMPLEMENTATION)
    register({DeepseekV4GGUFExperts: DeepseekV4GGUFMoeLora})
    return lora_config
