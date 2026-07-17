import torch
from transformers.integrations import gguf_dequant_kernels

_PATCH_MARKER = "_no_unsloth_torch_compile_patch"
_RECOMPILE_LIMIT = 64


def configure_compiled_gguf_dequantize() -> bool:
    """Compile the shared GGUF dequantizer for the fixed training workload.

    The generic dispatcher specializes by quantization type, output dtype, and
    packed input shape. This training process uses a bounded set of each, so
    permit enough Dynamo variants for the complete model and retain full-graph
    failures instead of silently falling back to eager execution.
    """
    torch._dynamo.config.recompile_limit = _RECOMPILE_LIMIT

    if getattr(gguf_dequant_kernels, _PATCH_MARKER, False):
        return False

    eager_dequantize = gguf_dequant_kernels.dequantize
    compiled_dequantize = torch.compile(
        eager_dequantize,
        fullgraph=True,
        mode="max-autotune-no-cudagraphs",
        recompile_limit=_RECOMPILE_LIMIT,
    )
    compiled_dequantize._no_unsloth_eager_dequantize = eager_dequantize
    gguf_dequant_kernels.dequantize = compiled_dequantize
    setattr(gguf_dequant_kernels, _PATCH_MARKER, True)
    return True
