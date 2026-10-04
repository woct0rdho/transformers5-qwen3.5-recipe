"""Test-only raw GGUF dequantization oracle."""

import torch
from transformers.integrations.gguf.dequant import GGML_BLOCK, dequantize


def dequantize_gguf_tensor(
    data: torch.Tensor,
    quant_type: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Materialize raw canonical GGUF blocks while preserving leading dimensions.

    Transformers exposes raw block dequantization as a flat operation. Tests
    that select only a subset of expert payload rows need this small boundary
    so their logical leading dimensions are retained.
    """

    payload = data.to(device=device, dtype=torch.uint8) if device is not None else data
    block_elements, block_bytes = GGML_BLOCK[int(quant_type)]
    if payload.shape[-1] % block_bytes:
        raise ValueError(
            f"GGUF payload width {payload.shape[-1]} is not divisible by block size {block_bytes}."
        )
    logical_last_dim = payload.shape[-1] // block_bytes * block_elements
    return dequantize(payload, int(quant_type), dtype=dtype).reshape(
        *payload.shape[:-1], logical_last_dim
    )
