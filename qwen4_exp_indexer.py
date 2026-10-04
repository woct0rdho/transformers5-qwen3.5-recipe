"""Skip the QSA indexer at the training shapes, where its selection is exhaustive.

At `S = 2048` with compress ratio 4 and budget 2048, the indexer's `block_topk` (512) is at least
the number of complete blocks any query can see (`floor(S / 4) = 512`), so `topk` selects every
complete block, the tail holds the remaining visible tokens, and the mask the indexer returns is
exactly the causal-plus-padding mask it was handed. The attention is dense causal grouped-query
attention at this length, and the selection does not depend on the scores.

Running it anyway is what the twelve layers spend most of a step on, for 1.07 GFLOP of useful
arithmetic per layer, because each call runs the selection as a Python loop over the whole sequence.
The patch replaces each of the twelve `self_attn.indexer` forwards with a short circuit that returns
the mask without touching `index_qk_proj`, either RMSNorm or RoPE. The parameters stay frozen and
carry no adapter, so no gradient changes: the indexer's outputs reach the loss only through `topk`
indices.

The original forward is kept on the module as `_qwen4_indexer_forward_reference`, and the audit
uses it to prove the equivalence on real inputs and on a padded mask. Above the guard the wrapper
calls that reference, so a longer sequence keeps the model's own behaviour rather than a fast path
that no longer holds. The guard is `block_topk * compress_ratio + compress_ratio - 1 >= S`, the
same 2051 that the selection buffer holds, so the exhaustive case is also the case with room for
every visible token.
"""

from typing import Any

import torch
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextQSAIndexer

from module_patching import (
    ModulePatchSpec,
    patch_module_forwards,
    require_complete_inventory,
)

EXPECTED_QSA_LAYERS = 12
EXPECTED_INDEX_HEADS = 4
EXPECTED_INDEX_KV_HEADS = 1
EXPECTED_INDEX_HEAD_DIM = 128
EXPECTED_COMPRESS_RATIO = 4
EXPECTED_BLOCK_TOPK = 512
EXPECTED_SEQUENCE_LENGTH = 2048

_SELECTION = "skipped-when-exhaustive"
_SUBJECT = "Qwen4-Exp QSA indexer"
_MARKER = "_patched_qwen4_indexer"
_REFERENCE_ATTRIBUTE = "_qwen4_indexer_forward_reference"


def is_exhaustive_at(
    block_topk: int, compress_ratio: int, sequence_length: int
) -> bool:
    """Whether every complete block is selected, and the tail fills the remaining visible tokens.

    `max_complete_blocks` is `floor(visible / compress_ratio)`, which right-padding can only
    shrink, so testing the whole sequence is the conservative test. The buffer bound is the same
    inequality: `block_topk * compress_ratio + compress_ratio - 1` slots against at most
    `sequence_length` visible tokens.
    """

    return sequence_length <= block_topk * compress_ratio + compress_ratio - 1


def _guard_values(module: Qwen4ExpTextQSAIndexer) -> tuple[int, int]:
    """The block top-k and compress ratio the guard needs, with the optional fields narrowed."""

    block_topk, compress_ratio = module.block_topk, module.compress_ratio
    if block_topk is None or compress_ratio is None:
        raise RuntimeError("Qwen4-Exp QSA indexer has no budget or compress ratio")
    return block_topk, compress_ratio


def _identity_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """The mask that leaves the attention's combination of the indexer mask unchanged.

    The attention ANDs a bool mask or adds a float one. Returning the incoming mask makes the bool
    combination idempotent, and a scalar zero is the additive identity for the eager path without
    allocating a second `[B, 1, S, S]` tensor.
    """

    if attention_mask.dtype == torch.bool:
        return attention_mask
    return attention_mask.new_zeros(())


def _qwen4_indexer_forward(
    self: Qwen4ExpTextQSAIndexer,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor,
    past_key_values: Any,
) -> torch.Tensor:
    reference = getattr(self, _REFERENCE_ATTRIBUTE, None)
    if reference is None:
        raise RuntimeError(
            "Qwen4-Exp QSA indexer has no reference forward to fall back to"
        )
    if past_key_values is not None or not torch.is_tensor(attention_mask):
        return reference(
            hidden_states, position_embeddings, attention_mask, past_key_values
        )
    if not is_exhaustive_at(*_guard_values(self), hidden_states.shape[1]):
        return reference(
            hidden_states, position_embeddings, attention_mask, past_key_values
        )
    return _identity_mask(attention_mask)


def reference_indexer_mask(
    module: Qwen4ExpTextQSAIndexer,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Run the module's own indexer forward, the one the short circuit replaces.

    The audit uses this to prove that the mask the short circuit returns is the mask the reference
    would have produced, on the same shapes and with a right-padded case included.
    """

    reference = getattr(module, _REFERENCE_ATTRIBUTE, None)
    if reference is None:
        raise RuntimeError(
            "Qwen4-Exp QSA indexer was not patched, so it has no reference forward"
        )
    return reference(hidden_states, position_embeddings, attention_mask, None)


def _validate_qwen4_indexer(name: str, module: Qwen4ExpTextQSAIndexer) -> None:
    if (
        module.index_n_heads != EXPECTED_INDEX_HEADS
        or module.index_kv_heads != EXPECTED_INDEX_KV_HEADS
        or module.index_head_dim != EXPECTED_INDEX_HEAD_DIM
        or module.compress_ratio != EXPECTED_COMPRESS_RATIO
        or module.block_topk != EXPECTED_BLOCK_TOPK
    ):
        raise RuntimeError(
            f"Qwen4-Exp QSA indexer {name!r} does not match "
            f"{EXPECTED_INDEX_HEADS}/{EXPECTED_INDEX_KV_HEADS}/{EXPECTED_INDEX_HEAD_DIM} heads and "
            f"ratio {EXPECTED_COMPRESS_RATIO} with block top-k {EXPECTED_BLOCK_TOPK}."
        )


def _prepare_qwen4_indexer(name: str, module: Qwen4ExpTextQSAIndexer) -> None:
    """Keep the module's own forward, once, before it is replaced.

    This runs on every pass, including passes over already-patched modules, so it must not capture
    the patched forward as its own reference.
    """

    if not hasattr(module, _REFERENCE_ATTRIBUTE):
        setattr(module, _REFERENCE_ATTRIBUTE, module.forward)


_SPECS = (
    ModulePatchSpec(
        module_type=Qwen4ExpTextQSAIndexer,
        forward=_qwen4_indexer_forward,
        handled_key="indexers",
        validate=_validate_qwen4_indexer,
        prepare=_prepare_qwen4_indexer,
        marker=_MARKER,
        freeze_weight=False,
    ),
)


def configure_qwen4_exp_indexer_fast_path(model: torch.nn.Module) -> dict[str, Any]:
    """Install the exhaustive-selection short circuit on one loaded Qwen4-Exp model.

    The patch replaces the forward only and leaves gradient flags as it found them, since the
    indexer is frozen and adapter-free on this checkpoint. The report carries the arithmetic the
    gate needs, so an audit can fail closed when the run's sequence length leaves the range where
    the selection is exhaustive.
    """

    report = patch_module_forwards(model, _SPECS)
    report["selection"] = _SELECTION
    report["block_topk"] = EXPECTED_BLOCK_TOPK
    report["compress_ratio"] = EXPECTED_COMPRESS_RATIO
    report["exhaustive_upto"] = (
        EXPECTED_BLOCK_TOPK * EXPECTED_COMPRESS_RATIO + EXPECTED_COMPRESS_RATIO - 1
    )
    return report


def require_complete_qwen4_exp_indexer(
    report: dict[str, Any],
    *,
    expected_layers: int = EXPECTED_QSA_LAYERS,
    sequence_length: int | None = None,
) -> None:
    """Fail closed unless every Qwen4-Exp indexer was handled and the selection is exhaustive."""

    require_complete_inventory(
        report,
        {"indexers": expected_layers, "selection": _SELECTION},
        subject=_SUBJECT,
    )
    if sequence_length is not None:
        block_topk = int(report["block_topk"])
        compress_ratio = int(report["compress_ratio"])
        if not is_exhaustive_at(block_topk, compress_ratio, sequence_length):
            raise RuntimeError(
                f"Qwen4-Exp QSA indexer selection is not exhaustive at sequence length "
                f"{sequence_length}, so the short circuit does not apply."
            )
