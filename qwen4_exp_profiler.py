"""Architecture categories for Qwen4-Exp full-step profiling.

The shared categorizer in `training_profiler.py` owns the LoRA wrapper vocabulary. This module
only supplies the Qwen4-Exp class names. The QSA indexer, the PLE layer, the hyper-connections,
and the routed expert base module are large enough to deserve their own categories, because they
are the sites the next round of kernels has to remove.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from fast_lora import FastGgufLoraLinear
from training_profiler import lora_module_category
from training_profiler import profile_warmed_training_update as _profile

_PACKED_LORA_CLASSES = frozenset({"FastGgufLoraLinear", "FastLoraLinear"})
_GENERIC_LORA_CLASSES: frozenset[str] = frozenset()
_EXPERT_LORA_CLASSES = frozenset({"Qwen4ExpGgufMoeLora"})
_LORA_ROLE_FRAGMENTS = (
    (".shared_expert.", "shared_expert"),
    (".shared_experts.", "shared_expert"),
)

_ARCHITECTURE_CLASSES = {
    "Qwen4ExpTextDecoderLayer": "decoder_layer",
    "Qwen4ExpTextQSAIndexer": "qsa_indexer",
    "Qwen4ExpTextAttention": "qsa_attention",
    "Qwen4ExpTextGatedDeltaNet": "gated_delta_net",
    "Qwen4ExpTextSparseMoeBlock": "moe_block",
    "Qwen4ExpTextTopKRouter": "moe_router",
    "Qwen4ExpTextPLELayer": "ple_layer",
    "Qwen4ExpTextNGramEmbedding": "ple_ngram_embedding",
    "Qwen4ExpTextGatedResidual": "hyper_connection",
    "GgufExperts": "routed_expert_base",
    "GgufEmbedding": "gguf_embedding",
    "GgufLinear": "gguf_linear",
}


def _module_category(name: str, module: torch.nn.Module) -> str | None:
    class_name = type(module).__name__
    if class_name in _ARCHITECTURE_CLASSES:
        return _ARCHITECTURE_CLASSES[class_name]
    if "RMSNorm" in class_name:
        return "rmsnorm"
    # One wrapper class serves both bases, so the capability decides. Every deployed projection
    # passes, so this reports a wrapper the wrapper itself refused, which is what a model loaded
    # without the tiled value-head convention leaves behind.
    if isinstance(module, FastGgufLoraLinear) and not module.uses_packed_mmq():
        return "ordinary_lora"
    return lora_module_category(
        name,
        class_name,
        packed_lora_classes=_PACKED_LORA_CLASSES,
        generic_lora_classes=_GENERIC_LORA_CLASSES,
        expert_lora_classes=_EXPERT_LORA_CLASSES,
        role_fragments=_LORA_ROLE_FRAGMENTS,
    )


def profile_warmed_training_update(
    model: torch.nn.Module,
    update: Callable[[str], dict[str, Any]],
    *,
    output_path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _profile(
        model,
        update,
        output_path=output_path,
        categorize=_module_category,
        metadata=metadata,
    )
