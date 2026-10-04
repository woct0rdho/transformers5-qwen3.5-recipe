"""Pretuned gfx1151 FLA autotune winners for the GatedDeltaNet training paths.

Both architectures share one mechanism: a table of `(kernel name, exact Triton cache key, config)`
entries, injected from `Autotuner.run` before Triton benchmarks anything. Keying on the kernel name
and Triton's own cache key keeps the tables independent of how a kernel was imported, which matters
because the hub `kernels` package loads its own copies of these FLA kernels lazily inside a loader
closure, so no importable module exposes those `Autotuner` objects. Only exact keys are populated, so
different batch geometry, dimensions, dtypes, recurrent-state modes, or variable-length modes retain
FLA's normal autotuning behavior.
"""

from typing import Any

import triton
from triton.runtime.autotuner import Autotuner

_QWEN35_CONFIGS: tuple[tuple[str, tuple[Any, ...], dict[str, Any]], ...] = (
    (
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
        "layer_norm_gated_bwd_kernel",
        (
            128,
            1,
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
        "l2norm_fwd_kernel",
        (128, 4, "torch.bfloat16", "torch.bfloat16", "torch.float32"),
        {"kwargs": {"BT": 16}, "num_warps": 16, "num_stages": 3},
    ),
    (
        "l2norm_bwd_kernel",
        (128, 1, "torch.bfloat16", "torch.float32", "torch.bfloat16", "torch.bfloat16"),
        {"kwargs": {"BT": 16}, "num_warps": 16, "num_stages": 3},
    ),
    (
        "l2norm_bwd_kernel",
        (128, 4, "torch.bfloat16", "torch.float32", "torch.bfloat16", "torch.bfloat16"),
        {"kwargs": {"BT": 8}, "num_warps": 8, "num_stages": 3},
    ),
    (
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
        "chunk_local_cumsum_scalar_kernel",
        (4, 32, 64, False, False, "torch.float32", "torch.float32"),
        {"kwargs": {}, "num_warps": 1, "num_stages": 3},
    ),
    (
        "chunk_local_cumsum_scalar_kernel",
        (1, 32, 64, False, True, "torch.float32", "torch.float32"),
        {"kwargs": {}, "num_warps": 1, "num_stages": 3},
    ),
    (
        "chunk_local_cumsum_scalar_kernel",
        (4, 32, 64, False, True, "torch.float32", "torch.float32"),
        {"kwargs": {}, "num_warps": 2, "num_stages": 3},
    ),
)

_QWEN4_EXP_CONFIGS: tuple[tuple[str, tuple[Any, ...], dict[str, Any]], ...] = (
    (
        "l2norm_fwd_kernel",
        (128, 2, "torch.bfloat16", "torch.bfloat16", "torch.float32"),
        {"kwargs": {"BT": 8}, "num_warps": 8, "num_stages": 3, "num_ctas": 1},
    ),
    (
        "l2norm_fwd_kernel",
        (128, 6, "torch.bfloat16", "torch.bfloat16", "torch.float32"),
        {"kwargs": {"BT": 16}, "num_warps": 16, "num_stages": 3, "num_ctas": 1},
    ),
    (
        "l2norm_fwd_kernel",
        (128, 24, "torch.bfloat16", "torch.bfloat16", "torch.float32"),
        {"kwargs": {"BT": 16}, "num_warps": 16, "num_stages": 3, "num_ctas": 1},
    ),
    (
        "l2norm_bwd_kernel",
        (128, 2, "torch.bfloat16", "torch.float32", "torch.bfloat16", "torch.bfloat16"),
        {"kwargs": {"BT": 16}, "num_warps": 16, "num_stages": 3, "num_ctas": 1},
    ),
    (
        "l2norm_bwd_kernel",
        (128, 6, "torch.bfloat16", "torch.float32", "torch.bfloat16", "torch.bfloat16"),
        {"kwargs": {"BT": 16}, "num_warps": 16, "num_stages": 3, "num_ctas": 1},
    ),
    (
        "l2norm_bwd_kernel",
        (
            128,
            24,
            "torch.bfloat16",
            "torch.float32",
            "torch.bfloat16",
            "torch.bfloat16",
        ),
        {"kwargs": {"BT": 16}, "num_warps": 16, "num_stages": 3, "num_ctas": 1},
    ),
    (
        "chunk_local_cumsum_scalar_kernel",
        (1, 48, 64, False, False, "torch.float32", "torch.float32"),
        {"kwargs": {}, "num_warps": 1, "num_stages": 3, "num_ctas": 1},
    ),
    (
        "chunk_local_cumsum_scalar_kernel",
        (1, 48, 64, False, True, "torch.float32", "torch.float32"),
        {"kwargs": {}, "num_warps": 1, "num_stages": 3, "num_ctas": 1},
    ),
    (
        "chunk_local_cumsum_scalar_kernel",
        (4, 48, 64, False, False, "torch.float32", "torch.float32"),
        {"kwargs": {}, "num_warps": 1, "num_stages": 3, "num_ctas": 1},
    ),
    (
        "chunk_local_cumsum_scalar_kernel",
        (4, 48, 64, False, True, "torch.float32", "torch.float32"),
        {"kwargs": {}, "num_warps": 1, "num_stages": 3, "num_ctas": 1},
    ),
    (
        "chunk_local_cumsum_scalar_kernel",
        (16, 48, 64, False, False, "torch.float32", "torch.float32"),
        {"kwargs": {}, "num_warps": 1, "num_stages": 3, "num_ctas": 1},
    ),
    (
        "chunk_local_cumsum_scalar_kernel",
        (16, 48, 64, False, True, "torch.float32", "torch.float32"),
        {"kwargs": {}, "num_warps": 1, "num_stages": 3, "num_ctas": 1},
    ),
    (
        "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
        (
            48,
            48,
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
        {"kwargs": {"BV": 32}, "num_warps": 4, "num_stages": 1, "num_ctas": 1},
    ),
    (
        "chunk_gated_delta_rule_fwd_kkt_solve_kernel",
        (
            48,
            48,
            128,
            16,
            "torch.bfloat16",
            "torch.float32",
            "torch.bfloat16",
            "torch.bfloat16",
        ),
        {"kwargs": {"BK": 64}, "num_warps": 1, "num_stages": 3, "num_ctas": 1},
    ),
    (
        "recompute_w_u_fwd_kernel",
        (
            48,
            48,
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
        {"kwargs": {}, "num_warps": 2, "num_stages": 2, "num_ctas": 1},
    ),
    (
        "chunk_fwd_kernel_o",
        (
            48,
            48,
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
        {
            "kwargs": {"BK": 16, "BV": 128},
            "num_warps": 4,
            "num_stages": 2,
            "num_ctas": 1,
        },
    ),
    (
        "chunk_bwd_kernel_dv_local",
        (
            48,
            48,
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
        {"kwargs": {}, "num_warps": 2, "num_stages": 2, "num_ctas": 1},
    ),
    (
        "chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64",
        (
            48,
            48,
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
        {"kwargs": {"BV": 64}, "num_warps": 8, "num_stages": 2, "num_ctas": 1},
    ),
    (
        "chunk_bwd_kernel_dqkwg",
        (
            48,
            48,
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
        {"kwargs": {}, "num_warps": 4, "num_stages": 2, "num_ctas": 1},
    ),
    (
        "prepare_wy_repr_bwd_kernel",
        (
            48,
            48,
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
        {"kwargs": {}, "num_warps": 2, "num_stages": 2, "num_ctas": 1},
    ),
)

_FLA_TABLE: dict[tuple[str, tuple], triton.Config] = {}
_FLA_APPLIED: set[tuple[str, tuple]] = set()
_FLA_HOOK_INSTALLED = False


def _triton_config(values: dict[str, Any]) -> triton.Config:
    return triton.Config(
        dict(values["kwargs"]),
        num_warps=values["num_warps"],
        num_stages=values["num_stages"],
        num_ctas=1,
    )


def _fla_triton_key(autotuner: Autotuner, args: tuple, kwargs: dict) -> tuple:
    """Triton's own cache key computation for one autotuner call."""
    all_args = {**dict(zip(autotuner.arg_names, args)), **kwargs}
    named = {
        name: value for name, value in all_args.items() if name in autotuner.arg_names
    }
    key = [named[name] for name in autotuner.keys if name in named]
    for value in named.values():
        if hasattr(value, "dtype"):
            key.append(str(value.dtype))
    return tuple(key)


def _install_fla_table(entries: tuple[tuple[Any, ...], ...]) -> int:
    """Add entries to the shared table and install the injection hook once."""
    for kernel_name, cache_key, config_values in entries:
        _FLA_TABLE[(kernel_name, cache_key)] = _triton_config(config_values)
    global _FLA_HOOK_INSTALLED

    if _FLA_HOOK_INSTALLED:
        return len(entries)
    original_run = Autotuner.run

    def run_with_pretuned_configs(self: Autotuner, *args, **kwargs):
        kernel_name = getattr(self.fn, "__name__", "")
        if kernel_name:
            cache_key = _fla_triton_key(self, args, kwargs)
            entry = _FLA_TABLE.get((kernel_name, cache_key))
            if entry is not None and cache_key not in self.cache:
                self.cache[cache_key] = entry
                _FLA_APPLIED.add((kernel_name, cache_key))
        return original_run(self, *args, **kwargs)

    Autotuner.run = run_with_pretuned_configs
    _FLA_HOOK_INSTALLED = True
    return len(entries)


def configure_qwen35_fla() -> int:
    """Preload exact gfx1151 FLA autotune winners for Qwen3.5 training."""
    return _install_fla_table(_QWEN35_CONFIGS)


def configure_qwen4_exp_fla() -> int:
    """Preload the tuned Qwen4-Exp GatedDeltaNet winners before the first forward pass."""
    return _install_fla_table(_QWEN4_EXP_CONFIGS)


def require_complete_qwen4_exp_fla(configured: int) -> dict[str, Any]:
    """Report which tuned entries the step used, failing when none of them were applied.

    A single run reaches the keys of its own batch and of the entry points it exercises, so partial
    coverage of the table is expected. Zero coverage means the table was not wired in.
    """
    if not _FLA_HOOK_INSTALLED:
        raise RuntimeError(
            "configure_qwen4_exp_fla() was not called before the forward pass"
        )
    if not _FLA_APPLIED:
        raise RuntimeError(
            "no tuned GatedDeltaNet config was applied. The table did not match any call"
        )
    return {
        "table_entries": configured,
        "entries_used": len(_FLA_APPLIED),
        "kernels_used": sorted({name for name, _ in _FLA_APPLIED}),
    }
