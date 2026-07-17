from collections.abc import Mapping
from typing import Any

import triton
from aiter.ops.triton._triton_kernels.flash_attn_triton_amd import (
    bwd,
    fwd_prefill,
)


def _single_config(values: Mapping[str, Any]):
    config_values = dict(values)
    num_stages = config_values.pop("num_stages", 1)
    num_warps = config_values.pop("num_warps", 4)
    num_ctas = config_values.pop("num_ctas", 1)
    return triton.Config(
        config_values,
        num_stages=num_stages,
        num_warps=num_warps,
        num_ctas=num_ctas,
    )


def _replace_autotune_configs(autotuner, values: Mapping[str, Any]) -> None:
    autotuner.configs = [_single_config(values)]
    autotuner.cache.clear()


def configure_qwen35_flash_attention_2() -> bool:
    """Install one manually tuned config for the fixed Qwen3.5 training shape.

    This targets BF16 causal BSHD attention with Hq=16, Hkv=2, D=256, and a
    maximum sequence length of 2048 on gfx1151. Rank/layout work stays inside
    AITER; no process-wide AITER autotune environment variable is changed.
    """
    _replace_autotune_configs(
        fwd_prefill.attn_fwd,
        {
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "PRE_LOAD_V": False,
            "waves_per_eu": 2,
            "num_stages": 1,
            "num_warps": 4,
        },
    )
    _replace_autotune_configs(
        bwd._bwd_preprocess,
        {
            "PRE_BLOCK": 2048,
            "num_stages": 1,
            "num_warps": 4,
        },
    )
    _replace_autotune_configs(
        bwd.bwd_kernel_fused_causal,
        {
            "BLOCK_M1": 16,
            "BLOCK_N1": 64,
            "BLOCK_M2": 64,
            "BLOCK_N2": 16,
            "BLK_SLICE_FACTOR": 1,
            "waves_per_eu": 1,
            "num_stages": 1,
            "num_warps": 4,
        },
    )
    return True
