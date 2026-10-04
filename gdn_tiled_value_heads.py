"""Carry the GatedDeltaNet value axis in llama.cpp's order, so nothing is permuted at run time.

The file stores every value-indexed tensor in llama.cpp's tiled head order - value heads outer, key
heads inner - because that is the order llama.cpp's graph consumes: its gated-delta-net output is
reshaped to `head_v_dim * num_v_heads` and handed straight to `ssm_out`, and its key-to-value broadcast
(`ggml_repeat_4d`) uses the same tiling. Transformers' convention is the grouped order, and its GGUF
loader converts the file into it. For the tensors that produce the value axis the conversion is a row
reorder of the packed payload, which is free, but the one tensor that consumes it would have to
reorder its input on every call: the gather is a per-call cost that buys nothing, because the packed
columns the projection multiplies are already in the order it wants to read them.

This module keeps the file's order end to end instead:
- `get_gguf_conversion_mapping` is wrapped so the load skips the value-head reorders. Everything else
  the mapping does stays: the `(1 + weight)` norm offsets, `-exp(A_log)`, the `conv1d` unsqueeze, and the
  renames.
- The GatedDeltaNet broadcast between key and value heads is switched from `repeat_interleave` (grouped
  pairing) to `repeat` (tiled pairing) while that forward runs. The two conventions compute the same
  function under a relabeling of the value heads, and every other operation on the value axis - the
  recurrent core, the gated norm, the depthwise conv, `A_log`, `dt_bias`, `in_proj_a`/`in_proj_b` - is
  per head, so nothing else changes.

`in_proj_a` and `in_proj_b` keep the compiled-dequant base: they are small, and their reorder was never
a layout obstacle.

The module patches classes and one loader function, so `configure_tiled_value_heads` has to run before
the model is loaded.
"""

import contextlib
from collections.abc import Iterator
from typing import Any

import torch
from transformers.integrations import gguf as gguf_package
from transformers.integrations.gguf import utils as gguf_utils
from transformers.integrations.gguf.gguf_conversion_mapping import (
    TiledToGroupedInputs,
    TiledToGroupedRows,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeGatedDeltaNet
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextGatedDeltaNet
from transformers.quantizers import quantizer_gguf

from module_patching import require_complete_inventory

# The architectures whose GGUF files carry llama.cpp's tiled value heads.
_TILED_ARCHITECTURES = frozenset({"qwen35", "qwen35moe", "qwen4exp"})
_VALUE_REORDER_TYPES = (TiledToGroupedRows, TiledToGroupedInputs)
# Four per-head scalars (`A_log`, `dt_bias`, `in_proj_a`, `in_proj_b`) plus four value-axis tensors
# (`in_proj_qkv`, `in_proj_z`, `conv1d`, `out_proj`). A load whose mapping drops a different number has
# moved on, and the layout decision has to be revisited.
_EXPECTED_VALUE_REORDERS = 8
_GDN_TYPES = (Qwen4ExpTextGatedDeltaNet, Qwen3_5MoeGatedDeltaNet)
_SUBJECT = "GatedDeltaNet value-head layout"

_ORIGINAL_REPEAT_INTERLEAVE = torch.Tensor.repeat_interleave
_STATE: dict[str, Any] = {
    "module": None,
    "rewrites": 0,
    "layers": set(),
    "enabled": True,
}
_CONFIGURED = False
_DROPPED = 0


def _strip_value_reorders(mapping: list[Any]) -> int:
    """Drop every value-head reorder from one load's transforms, and return how many were dropped."""

    dropped = 0
    for entry in mapping:
        operations = getattr(entry, "operations", None)
        if not operations:
            continue
        kept = [
            operation
            for operation in operations
            if not isinstance(operation, _VALUE_REORDER_TYPES)
        ]
        dropped += len(operations) - len(kept)
        if len(kept) != len(operations):
            entry.operations = kept
    return dropped


def _conversion_mapping(original):
    def mapping(gguf_arch: str, config) -> list[Any]:
        transforms = original(gguf_arch, config)
        if gguf_arch not in _TILED_ARCHITECTURES:
            return transforms
        global _DROPPED
        dropped = _strip_value_reorders(transforms)
        if dropped != _EXPECTED_VALUE_REORDERS:
            raise RuntimeError(
                f"{_SUBJECT}: expected {_EXPECTED_VALUE_REORDERS} value-head reorders for "
                f"{gguf_arch!r}, dropped {dropped}. The loader's mapping changed."
            )
        _DROPPED = dropped
        return transforms

    return mapping


def _tiled_repeat_interleave(self, repeats, dim=None, *, output_size=None):
    """`repeat_interleave`, tiled inside a GatedDeltaNet forward.

    The file's value heads are `[k0..k15, k0..k15, ...]`, which is `repeat` on the head axis, where
    the grouped convention is `[k0, k0, k1, k1, ...]`. Only that forward is rewritten, and only for the
    4-D key and query tensors it expands.
    """

    if (
        _STATE["enabled"]
        and _STATE["module"] is not None
        and dim == 2
        and self.dim() == 4
        and int(repeats) > 1
    ):
        _STATE["rewrites"] += 1
        _STATE["layers"].add(_STATE["module"])
        return self.repeat(1, 1, int(repeats), 1)
    return _ORIGINAL_REPEAT_INTERLEAVE(self, repeats, dim, output_size=output_size)


@contextlib.contextmanager
def _tiled_scope(module: torch.nn.Module) -> Iterator[None]:
    previous = _STATE["module"]
    _STATE["module"] = module
    try:
        yield
    finally:
        _STATE["module"] = previous


@contextlib.contextmanager
def grouped_broadcast() -> Iterator[None]:
    """Run one forward with the loader's grouped pairing, to compare against the tiled convention."""

    previous = _STATE["enabled"]
    _STATE["enabled"] = False
    try:
        yield
    finally:
        _STATE["enabled"] = previous


def _gdn_forward(original):
    def forward(self, *args, **kwargs):
        with _tiled_scope(self):
            return original(self, *args, **kwargs)

    return forward


def configure_tiled_value_heads() -> dict[str, Any]:
    """Install the loader and model patches for the tiled value-head convention.

    Must run before the model is loaded: the loader patch is consumed while weights are built.
    """

    global _CONFIGURED
    if _CONFIGURED:
        return report()

    wrapped = _conversion_mapping(gguf_utils.get_gguf_conversion_mapping)
    for module in (gguf_utils, gguf_package, quantizer_gguf):
        module.get_gguf_conversion_mapping = wrapped
    torch.Tensor.repeat_interleave = _tiled_repeat_interleave
    for gdn_type in _GDN_TYPES:
        gdn_type.forward = _gdn_forward(gdn_type.forward)
    _CONFIGURED = True
    return report()


def record_tiled_broadcast(module: torch.nn.Module) -> None:
    """Record a tiled key-head broadcast produced outside `repeat_interleave`.

    The fused preparation builds the broadcast itself, in the same tiled order the patched
    `repeat_interleave` produces, so the inventory gate that counts which layers took the tiled
    convention has to be told about it.
    """

    _STATE["layers"].add(module)
    _STATE["rewrites"] += 1


def report() -> dict[str, Any]:
    """What the patches have done, for a gate to inspect after a forward."""

    return {
        "configured": _CONFIGURED,
        "value_reorders_dropped": _DROPPED,
        "tiled_rewrites": _STATE["rewrites"],
        "layers": len(_STATE["layers"]),
    }


def require_tiled_value_heads(
    model: torch.nn.Module, *, expected_gdn_layers: int
) -> dict[str, Any]:
    """Fail closed unless the model runs the tiled convention and permutes nothing.

    The structural half of the check: the patches are installed, every GatedDeltaNet layer is
    patched, and no loaded module wants an input permutation. `report` carries the runtime half - how
    many layers actually took the tiled broadcast - which only a forward can fill in.
    """

    from transformers.integrations.gguf.modules import GgufLinear

    gdn_layers = [
        module for module in model.modules() if isinstance(module, _GDN_TYPES)
    ]
    permuted = [
        name
        for name, module in model.named_modules()
        if isinstance(module, GgufLinear)
        and getattr(module, "input_permutation", None) is not None
    ]
    result = {
        "gdn_layers": len(gdn_layers),
        "permuted_modules": len(permuted),
        **report(),
    }
    require_complete_inventory(
        result,
        {
            "configured": True,
            "value_reorders_dropped": _EXPECTED_VALUE_REORDERS,
            "gdn_layers": expected_gdn_layers,
            "permuted_modules": 0,
        },
        subject=_SUBJECT,
    )
    return result
