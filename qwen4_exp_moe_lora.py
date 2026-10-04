"""Qwen4-Exp routed-expert LoRA on grouped MMQ with AITER factors.

`fast_moe_lora.py` owns the shared routed-expert adapter: routing plan, route gather, one
full-cover `group_sizes` vector, gate/up and down factor families through AITER `gmm`, factor
gradients through AITER `ptgmm`, and the route combine. The frozen base projections are its
default grouped-MMQ entry points, and this checkpoint is served by them: all 144 routed-expert
tensors are `Q2_0`, which `torch_ggml_ops` deploys for the gate/up pair, the down projection, and
both input gradients at the physical route counts 20480, 81920, and 327680.

This module therefore only declares what is Qwen4-Exp-specific:
- the experts implementation name, registered on the model so `GgufExperts` dispatches here,
- the `qwen3.8-learned` prior, which selects the measured AITER entries of `moe_gmm_configs.py`
  for the fitted 512-expert top-10 route law of this checkpoint.
- the regex-excluding target pattern this architecture is registered with, which the shared
  registration does not accept as a string.
"""

from typing import Any, cast

import torch
from peft import LoraConfig
from transformers.integrations.gguf.moe import ALL_GGUF_EXPERTS_FUNCTIONS, GgufExperts

from fast_moe_lora import (
    _LORA_WEIGHTS_KWARG,
    FastGgufMoeLora,
    _ExpertLoraWeights,
    gguf_mmq_aiter_lora_forward,
)

QWEN4_EXP_EXPERTS_IMPLEMENTATION = "qwen4_exp_gguf_mmq_aiter_lora"
QWEN4_EXP_EXPERT_PRIOR = "qwen3.8-learned"
_GGUF_EXPERTS_TYPE = cast(type[Any], GgufExperts)


class Qwen4ExpGgufMoeLora(FastGgufMoeLora):
    """PEFT wrapper for all gate, up, and down adapters of one Qwen4-Exp MoE layer."""

    def forward(
        self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        adapter_names = kwargs.pop("adapter_names", None)
        lora_weights = self._active_lora_weights(adapter_names)
        experts = self.get_base_layer()
        if not isinstance(experts, _GGUF_EXPERTS_TYPE):
            raise TypeError(
                "Qwen4-Exp expert LoRA requires GgufExperts, got "
                f"{type(experts).__name__}."
            )
        if experts.config._experts_implementation != QWEN4_EXP_EXPERTS_IMPLEMENTATION:
            raise RuntimeError(
                "Qwen4-Exp expert LoRA requires experts_implementation="
                f"{QWEN4_EXP_EXPERTS_IMPLEMENTATION!r}, "
                f"got {experts.config._experts_implementation!r}."
            )
        kwargs[_LORA_WEIGHTS_KWARG] = lora_weights
        return self.base_layer(hidden_states, *args, **kwargs)


def qwen4_exp_gguf_mmq_aiter_lora_forward(
    self: Any,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    _gguf_moe_lora_weights: _ExpertLoraWeights | None = None,
) -> torch.Tensor:
    """Run packed grouped base MMQ and AITER LoRA grouped MM."""

    if not isinstance(self, _GGUF_EXPERTS_TYPE):
        raise TypeError(
            f"{QWEN4_EXP_EXPERTS_IMPLEMENTATION} requires GgufExperts, "
            f"got {type(self).__name__}."
        )
    return gguf_mmq_aiter_lora_forward(
        self,
        hidden_states,
        top_k_index,
        top_k_weights,
        _gguf_moe_lora_weights,
    )


def register_qwen4_exp_moe_lora(
    lora_config: LoraConfig,
    model: torch.nn.Module,
) -> LoraConfig:
    """Register the grouped-MMQ expert backend and bind the Qwen3.8 route prior."""

    register = getattr(lora_config, "_register_custom_module", None)
    if register is None:
        raise RuntimeError(
            "This PEFT version has no LoraConfig._register_custom_module API. "
            "Cannot install the Qwen4-Exp expert LoRA without a global monkey patch."
        )
    if lora_config.target_parameters:
        raise ValueError(
            "Persistent GGUF expert LoRA targets the complete 'experts' module, not target_parameters."
        )
    if not isinstance(lora_config.target_modules, str):
        raise TypeError(
            "Qwen4-Exp LoRA uses one explicit pattern that selects the complete 'experts' module."
        )
    if "experts" not in lora_config.target_modules:
        raise TypeError(
            "The explicit target pattern must select the complete 'experts' module."
        )
    ALL_GGUF_EXPERTS_FUNCTIONS[QWEN4_EXP_EXPERTS_IMPLEMENTATION] = (
        qwen4_exp_gguf_mmq_aiter_lora_forward
    )
    cast(Any, model).set_experts_implementation(QWEN4_EXP_EXPERTS_IMPLEMENTATION)
    # The prior selects the exact-target GMM and PTGMM entries of moe_gmm_configs.py.
    for module in model.modules():
        if isinstance(module, _GGUF_EXPERTS_TYPE):
            module.__dict__["_aiter_expert_prior"] = QWEN4_EXP_EXPERT_PRIOR
    register({GgufExperts: Qwen4ExpGgufMoeLora})
    return lora_config
