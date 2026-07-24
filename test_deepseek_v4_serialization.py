from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig, get_peft_model
from peft.utils.save_and_load import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from transformers.integrations.gguf import DeepseekV4GGUFExperts

from deepseek_v4_lora import (
    DEEPSEEK_V4_TARGET_MODULES_PATTERN,
    register_deepseek_v4_lora,
)
from deepseek_v4_moe_lora import (
    DeepseekV4GGUFMoeLora,
    register_deepseek_v4_moe_lora,
)


class _ToyDeepseek(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_a_proj = torch.nn.Linear(
            32, 16, bias=False, device="cuda", dtype=torch.bfloat16
        )
        config = SimpleNamespace(
            num_experts=2,
            hidden_size=32,
            moe_intermediate_size=16,
            hidden_act="silu",
            swiglu_limit=7.0,
            _experts_implementation="eager",
        )
        self.experts = DeepseekV4GGUFExperts(
            config,
            device="cuda",
            compute_dtype=torch.bfloat16,
        )
        self.experts.config = config
        self.experts.gate_proj = torch.nn.Parameter(
            torch.randn(2, 16, 32, device="cuda", dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.experts.up_proj = torch.nn.Parameter(
            torch.randn(2, 16, 32, device="cuda", dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.experts.down_proj = torch.nn.Parameter(
            torch.randn(2, 32, 16, device="cuda", dtype=torch.bfloat16),
            requires_grad=False,
        )

    def set_experts_implementation(self, implementation: str) -> None:
        self.experts.config._experts_implementation = implementation


def _wrapped_toy() -> torch.nn.Module:
    model = _ToyDeepseek()
    config = LoraConfig(
        target_modules=DEEPSEEK_V4_TARGET_MODULES_PATTERN,
        r=4,
        lora_alpha=4,
        lora_dropout=0.0,
        bias="none",
    )
    register_deepseek_v4_lora(config)
    register_deepseek_v4_moe_lora(config, model)
    return get_peft_model(model, config, autocast_adapter_dtype=False)


def test_adapter_state_round_trip_contains_only_all_six_lora_factors() -> None:
    source = _wrapped_toy()
    with torch.no_grad():
        for parameter in source.parameters():
            if parameter.requires_grad:
                parameter.normal_(std=0.03)
    state = get_peft_model_state_dict(source)

    assert len(state) == 6
    assert all("lora_" in name for name in state)
    assert any("lora_A_down" in name for name in state)
    assert any("lora_B_down" in name for name in state)
    assert not any("base_layer" in name or "gate_proj" in name for name in state)

    target = _wrapped_toy()
    result = set_peft_model_state_dict(target, state)
    assert result.missing_keys == [
        "base_model.model.q_a_proj.base_layer.weight",
        "base_model.model.experts.base_layer.gate_proj",
        "base_model.model.experts.base_layer.up_proj",
        "base_model.model.experts.base_layer.down_proj",
    ]
    assert not result.unexpected_keys
    target_state = get_peft_model_state_dict(target)
    assert state.keys() == target_state.keys()
    for name in state:
        torch.testing.assert_close(target_state[name], state[name], rtol=0, atol=0)

    expert_wrapper = next(
        module
        for module in target.modules()
        if isinstance(module, DeepseekV4GGUFMoeLora)
    )
    with pytest.raises(RuntimeError, match="cannot be merged"):
        expert_wrapper.merge()
