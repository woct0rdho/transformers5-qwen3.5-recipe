import importlib
import warnings
from typing import Any

import fla
import torch
import triton
from triton.runtime.autotuner import Autotuner

_REQUIRED_FLA_VERSION = "0.5.1"
_REQUIRED_TRITON_VERSION = "3.8.0"
_REQUIRED_ARCH = "gfx1151"

# Entries are (module, outer decorated kernel, exact Triton cache key, config).
# They were selected on Radeon 8060S / gfx1151 for Qwen3.5-35B-A3B training:
# B=4, T=2048, H=HV=32, K=V=128, BF16 q/k/v/beta, FP32 gate,
# no recurrent state, no varlen metadata, and fused Q/K L2 normalization.
_KNOWN_CONFIGS: tuple[tuple[str, str, tuple[Any, ...], dict[str, Any]], ...] = (
    (
        "fla.modules.fused_norm_gate",
        "layer_norm_gated_fwd_kernel",
        (
            128,
            4,
            True,
            False,
            False,
            True,
            "torch.bfloat16",
            "torch.bfloat16",
            "torch.bfloat16",
            "torch.bfloat16",
            "torch.float32",
        ),
        {"kwargs": {"BT": 16}, "num_warps": 16, "num_stages": 3},
    ),
    (
        "fla.modules.fused_norm_gate",
        "layer_norm_gated_bwd_kernel",
        (
            128,
            4,
            True,
            False,
            True,
            "torch.bfloat16",
            "torch.bfloat16",
            "torch.bfloat16",
            "torch.bfloat16",
            "torch.bfloat16",
            "torch.bfloat16",
            "torch.float32",
            "torch.float32",
        ),
        {"kwargs": {"BT": 32}, "num_warps": 4, "num_stages": 3},
    ),
    (
        "fla.modules.l2norm",
        "l2norm_fwd_kernel",
        (128, 4, "torch.bfloat16", "torch.bfloat16", "torch.float32"),
        {"kwargs": {"BT": 16}, "num_warps": 16, "num_stages": 3},
    ),
    (
        "fla.modules.l2norm",
        "l2norm_bwd_kernel",
        (
            128,
            4,
            "torch.bfloat16",
            "torch.float32",
            "torch.bfloat16",
            "torch.bfloat16",
        ),
        {"kwargs": {"BT": 8}, "num_warps": 8, "num_stages": 3},
    ),
    (
        "fla.ops.common.chunk_delta_h",
        "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
        (
            32,
            32,
            128,
            128,
            64,
            False,
            "torch.bfloat16",
            "torch.bfloat16",
            "torch.bfloat16",
            "torch.bfloat16",
            "torch.float32",
            "torch.bfloat16",
        ),
        {"kwargs": {"BV": 32}, "num_warps": 4, "num_stages": 1},
    ),
    (
        "fla.ops.common.chunk_delta_h",
        "chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64",
        (
            32,
            32,
            128,
            128,
            64,
            True,
            False,
            "torch.bfloat16",
            "torch.bfloat16",
            "torch.bfloat16",
            "torch.float32",
            "torch.bfloat16",
            "torch.bfloat16",
            "torch.bfloat16",
            "torch.bfloat16",
        ),
        {"kwargs": {"BV": 32}, "num_warps": 4, "num_stages": 1},
    ),
    (
        "fla.ops.common.chunk_o",
        "chunk_fwd_kernel_o",
        (
            32,
            32,
            128,
            128,
            64,
            False,
            "torch.bfloat16",
            "torch.bfloat16",
            "torch.bfloat16",
            "torch.bfloat16",
            "torch.float32",
            "torch.bfloat16",
        ),
        {"kwargs": {"BK": 32, "BV": 32}, "num_warps": 2, "num_stages": 3},
    ),
    (
        "fla.ops.common.chunk_o",
        "chunk_bwd_kernel_dqkwg",
        (
            32,
            32,
            128,
            128,
            64,
            32,
            32,
            True,
            False,
            True,
            False,
            "torch.bfloat16",
            "torch.bfloat16",
            "torch.bfloat16",
            "torch.float32",
            "torch.bfloat16",
            "torch.bfloat16",
            "torch.bfloat16",
            "torch.bfloat16",
            "torch.bfloat16",
            "torch.bfloat16",
            "torch.bfloat16",
            "torch.float32",
        ),
        {"kwargs": {}, "num_warps": 8, "num_stages": 2},
    ),
    (
        "fla.ops.common.chunk_o",
        "chunk_bwd_kernel_dv_local",
        (
            32,
            32,
            128,
            128,
            64,
            32,
            32,
            True,
            "torch.bfloat16",
            "torch.bfloat16",
            "torch.float32",
            "torch.bfloat16",
            "torch.bfloat16",
        ),
        {"kwargs": {}, "num_warps": 8, "num_stages": 2},
    ),
    (
        "fla.ops.gated_delta_rule.chunk_fwd",
        "chunk_gated_delta_rule_fwd_kkt_solve_kernel",
        (
            32,
            32,
            128,
            16,
            "torch.bfloat16",
            "torch.float32",
            "torch.bfloat16",
            "torch.bfloat16",
        ),
        {"kwargs": {"BK": 32}, "num_warps": 4, "num_stages": 3},
    ),
    (
        "fla.ops.gated_delta_rule.wy_fast",
        "recompute_w_u_fwd_kernel",
        (
            32,
            32,
            128,
            128,
            64,
            64,
            64,
            False,
            "torch.bfloat16",
            "torch.bfloat16",
            "torch.bfloat16",
            "torch.bfloat16",
            "torch.bfloat16",
            "torch.bfloat16",
            "torch.float32",
        ),
        {"kwargs": {}, "num_warps": 8, "num_stages": 4},
    ),
    (
        "fla.ops.gated_delta_rule.wy_fast",
        "prepare_wy_repr_bwd_kernel",
        (
            32,
            32,
            128,
            128,
            64,
            32,
            32,
            False,
            "torch.bfloat16",
            "torch.bfloat16",
            "torch.bfloat16",
            "torch.float32",
            "torch.bfloat16",
            "torch.bfloat16",
            "torch.bfloat16",
            "torch.bfloat16",
            "torch.bfloat16",
            "torch.bfloat16",
            "torch.float32",
        ),
        {"kwargs": {}, "num_warps": 4, "num_stages": 2},
    ),
    (
        "fla.ops.utils.cumsum",
        "chunk_local_cumsum_scalar_kernel",
        (4, 32, 64, False, False, "torch.float32", "torch.float32"),
        {"kwargs": {}, "num_warps": 1, "num_stages": 3},
    ),
    (
        "fla.ops.utils.cumsum",
        "chunk_local_cumsum_scalar_kernel",
        (4, 32, 64, False, True, "torch.float32", "torch.float32"),
        {"kwargs": {}, "num_warps": 2, "num_stages": 3},
    ),
)


def _unwrap_autotuner(value: Any) -> Autotuner:
    for _ in range(8):
        if isinstance(value, Autotuner):
            return value
        value = getattr(value, "fn", None)
        if value is None:
            break
    raise TypeError("decorated kernel does not contain a Triton Autotuner")


def _triton_config(values: dict[str, Any]) -> triton.Config:
    return triton.Config(
        dict(values["kwargs"]),
        num_warps=values["num_warps"],
        num_stages=values["num_stages"],
        num_ctas=1,
    )


def configure_qwen35_fla() -> int:
    """Preload exact gfx1151 FLA autotune winners for Qwen3.5 training.

    Only exact Triton cache keys are populated. Different batch geometry,
    dimensions, dtypes, recurrent-state modes, or variable-length modes retain
    FLA's normal autotuning behavior.
    """
    if not torch.cuda.is_available():
        warnings.warn(
            "Qwen3.5 FLA tuning skipped because no GPU is available", stacklevel=2
        )
        return 0

    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None)
    if arch != _REQUIRED_ARCH:
        warnings.warn(
            f"Qwen3.5 FLA tuning targets {_REQUIRED_ARCH}, not {arch}; using stock FLA",
            stacklevel=2,
        )
        return 0

    fla_version = getattr(fla, "__version__", None)
    if (
        fla_version != _REQUIRED_FLA_VERSION
        or triton.__version__ != _REQUIRED_TRITON_VERSION
    ):
        warnings.warn(
            "Qwen3.5 FLA configs require "
            f"FLA {_REQUIRED_FLA_VERSION} and Triton {_REQUIRED_TRITON_VERSION}; "
            f"found FLA {fla_version} and Triton {triton.__version__}. Using stock FLA.",
            stacklevel=2,
        )
        return 0

    configured = 0
    autotuners: dict[tuple[str, str], Autotuner] = {}
    for module_name, attribute, cache_key, config_values in _KNOWN_CONFIGS:
        identity = (module_name, attribute)
        autotuner = autotuners.get(identity)
        if autotuner is None:
            module = importlib.import_module(module_name)
            autotuner = _unwrap_autotuner(getattr(module, attribute))
            autotuners[identity] = autotuner
        autotuner.cache[cache_key] = _triton_config(config_values)
        configured += 1
    return configured
