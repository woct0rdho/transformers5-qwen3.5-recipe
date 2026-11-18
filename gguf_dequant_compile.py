from collections.abc import Callable
from typing import Any, cast

import torch
from transformers.integrations.gguf import dequant as gguf_dequant
from transformers.integrations.gguf import gguf_quantized_parameter
from transformers.integrations.gguf import kernels as gguf_kernels

_PATCH_MARKER = "_patched_compiled_dequantize"
_RECOMPILE_LIMIT = 64


def configure_compiled_gguf_dequantize() -> bool:
    """Compile the shared GGUF dequantizer for the fixed training workload.

    The generic dispatcher specializes by quantization type, output dtype, and
    packed input shape. This training process uses a bounded set of each, so
    permit enough Dynamo variants for the complete model and retain full-graph
    failures instead of silently falling back to eager execution.
    """
    torch._dynamo.config.__dict__["recompile_limit"] = _RECOMPILE_LIMIT

    if getattr(gguf_dequant, _PATCH_MARKER, False):
        return False

    eager_dequantize = gguf_dequant.dequantize
    compile_fn = cast(Callable[..., Any], torch.compile)
    compiled_dequantize = compile_fn(
        eager_dequantize,
        fullgraph=True,
        mode="max-autotune-no-cudagraphs",
        recompile_limit=_RECOMPILE_LIMIT,
    )
    compiled_dequantize._eager_dequantize = eager_dequantize
    # Both consumers import the function into their module namespace. Replacing
    # only gguf.dequant.dequantize would leave persistent parameters and the
    # kernel fallback bound to the eager function imported at module load time.
    gguf_dequant.dequantize = compiled_dequantize
    gguf_quantized_parameter.dequantize = compiled_dequantize
    gguf_kernels.dequantize = compiled_dequantize
    gguf_dequant.__dict__[_PATCH_MARKER] = True
    return True
