"""Write stock-layout adapter checkpoints while training in llama.cpp's value-head order.

`gdn_tiled_value_heads` makes the training model carry the GGUF file's tiled GDN value-head
order end to end, which is what lets the packed base weights run without a per-call input
gather. The price is that every adapter factor on that axis (`in_proj_qkv` value rows,
`in_proj_z` rows, `out_proj` columns) is stored tiled, and `convert_lora_to_gguf.py` - which
assumes transformers' grouped order - would permute it a second time.

This module owns the checkpoint boundary instead of the training step. Saved checkpoints always
carry transformers' grouped order, whatever the model does in memory:
- `get_peft_model_state_dict` is wrapped so saved factors are permuted back to grouped order.
- `set_peft_model_state_dict` permutes them into the tiled layout again, which is the path
  `resume_from_checkpoint` and `PeftModel.from_pretrained` take.
- `saving_optimizer_state` / `loading_optimizer_state` do the same for `optimizer.pt`, whose
  8-bit states are relabelled exactly when the permutation moves whole quantization blocks and
  decoded/re-encoded when it does not (the `out_proj` columns, whose 128-element heads straddle
  the 256-element blocks).

A resume therefore assumes the adapter on disk is in the stock layout, like every adapter
transformers or peft produces.
"""

import logging
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import torch

from gdn_value_axis import GROUPED, TILED, ValueAxisGeometry, rule_for

logger = logging.getLogger(__name__)

# bitsandbytes 8-bit optimizers quantize blocks of 256 elements (`Optimizer2State.init_state`).
_BNB_BLOCK = 256
_BNB_QUANT_KEY = "__bnb_optimizer_quant_state__"
_ADAM_STATE_KEYS = ("exp_avg", "exp_avg_sq", "max_exp_avg_sq")

_CONFIGURED = False


def geometry_for(model: Any) -> ValueAxisGeometry:
    """The GDN head geometry of the model a checkpoint belongs to."""
    config = getattr(model, "config", None)
    if config is None:
        get_base_model = getattr(model, "get_base_model", None)
        if callable(get_base_model):
            config = getattr(get_base_model(), "config", None)
    if config is None:
        raise TypeError(
            f"{type(model).__name__} exposes no config to read the GDN head geometry from"
        )
    return ValueAxisGeometry.from_config(config)


def stock_layout_state_dict(
    model: Any, state_dict: dict[str, Any], *, to_layout: str
) -> dict[str, Any]:
    """Return `state_dict` with every value-axis adapter factor moved into `to_layout`."""
    geometry = geometry_for(model)
    converted: dict[str, Any] = {}
    for name, value in state_dict.items():
        if not torch.is_tensor(value):
            converted[name] = value
            continue
        rule = rule_for(name, geometry)
        if rule is None:
            converted[name] = value
            continue
        converted[name] = rule.permute(value.detach(), geometry, to_layout)
    return converted


def _flattened_parameters(optimizer: torch.optim.Optimizer) -> list[torch.Tensor]:
    """Parameters in the order `Optimizer.state_dict()` numbers its state entries."""
    return [param for group in optimizer.param_groups for param in group["params"]]


def _parameter_names(model: Any) -> dict[int, str]:
    return {id(param): name for name, param in model.named_parameters()}


def _block_aligned(moves: tuple[tuple[int, int, int], ...]) -> bool:
    return all(
        src % _BNB_BLOCK == 0 and dst % _BNB_BLOCK == 0 and length % _BNB_BLOCK == 0
        for src, dst, length in moves
    )


def _move_blocks(
    tensor: torch.Tensor, moves: tuple[tuple[int, int, int], ...]
) -> torch.Tensor:
    """Relabel whole blocks of a flat buffer. Exact: no block is split."""
    flat = tensor.reshape(-1)
    moved = flat.clone()
    for src, dst, length in moves:
        moved[dst : dst + length] = flat[src : src + length]
    return moved.reshape(tensor.shape)


def _move_blocks_as_blocks(
    tensor: torch.Tensor, moves: tuple[tuple[int, int, int], ...]
) -> torch.Tensor:
    blocks = tuple(
        (src // _BNB_BLOCK, dst // _BNB_BLOCK, length // _BNB_BLOCK)
        for src, dst, length in moves
    )
    return _move_blocks(tensor, blocks)


def _permute_quantized_entry(
    entry: dict[str, Any],
    rule: Any,
    geometry: ValueAxisGeometry,
    shape: tuple[int, ...],
    to_layout: str,
) -> bool:
    """Relabel one bitsandbytes state entry, 8-bit packed or its 32-bit fallback.

    Returns True when the permutation had to re-encode the state, which only happens when it
    splits quantization blocks.
    """
    state1 = entry.get("state1")
    if not torch.is_tensor(state1):
        return False
    if state1.dtype != torch.uint8:
        for key in ("state1", "state2"):
            value = entry.get(key)
            if torch.is_tensor(value) and tuple(value.shape) == shape:
                entry[key] = rule.permute(value, geometry, to_layout)
        return False

    moves = rule.chunks(shape, geometry, to_layout)
    if _block_aligned(moves):
        for key, absmax_key in (("state1", "absmax1"), ("state2", "absmax2")):
            value = entry.get(key)
            if torch.is_tensor(value):
                entry[key] = _move_blocks(value, moves)
            absmax = entry.get(absmax_key)
            if torch.is_tensor(absmax):
                entry[absmax_key] = _move_blocks_as_blocks(absmax, moves)
        return False

    # The permutation splits quantization blocks: decode, relabel, re-encode. The kernel
    # requantizes from the dense moments on every step anyway, so this is the same rounding
    # the next optimizer step would have applied.
    import bitsandbytes.functional as bnb_functional

    for key, absmax_key, qmap_key in (
        ("state1", "absmax1", "qmap1"),
        ("state2", "absmax2", "qmap2"),
    ):
        codes = entry.get(key)
        absmax = entry.get(absmax_key)
        qmap = entry.get(qmap_key)
        if not (
            torch.is_tensor(codes) and torch.is_tensor(absmax) and torch.is_tensor(qmap)
        ):
            continue
        if codes.dtype != torch.uint8:
            continue
        if codes.numel() % _BNB_BLOCK:
            raise ValueError(
                f"{codes.numel()} elements do not fill 8-bit optimizer blocks of {_BNB_BLOCK}"
            )
        dense = bnb_functional.dequantize_blockwise(
            codes, absmax=absmax, code=qmap, blocksize=_BNB_BLOCK
        )
        moved = rule.permute(dense, geometry, to_layout)
        recoded, quant_state = bnb_functional.quantize_blockwise(
            moved, code=qmap, blocksize=_BNB_BLOCK
        )
        entry[key] = recoded.to(device=codes.device)
        entry[absmax_key] = quant_state.absmax.to(
            device=absmax.device, dtype=absmax.dtype
        )
    return True


def permute_state_entry(
    entry: dict[str, Any],
    name: str,
    shape: tuple[int, ...],
    geometry: ValueAxisGeometry,
    to_layout: str,
) -> tuple[bool, bool]:
    """Move one optimizer state entry into `to_layout` in place.

    Returns `(converted, requantized)`: whether the entry has a value axis at all, and whether
    its 8-bit state had to be re-encoded.
    """
    rule = rule_for(name, geometry)
    if rule is None:
        return False, False
    requantized = False
    quant_state = entry.get(_BNB_QUANT_KEY)
    if isinstance(quant_state, dict):
        requantized |= _permute_quantized_entry(
            quant_state, rule, geometry, shape, to_layout
        )
    if "state1" in entry or "state2" in entry:
        requantized |= _permute_quantized_entry(entry, rule, geometry, shape, to_layout)
    for key in _ADAM_STATE_KEYS:
        value = entry.get(key)
        if (
            torch.is_tensor(value)
            and tuple(value.shape) == shape
            and value.dtype.is_floating_point
        ):
            entry[key] = rule.permute(value, geometry, to_layout)
    return True, requantized


def convert_optimizer_state(
    optimizer: torch.optim.Optimizer,
    model: Any,
    state_dict: dict[str, Any],
    *,
    to_layout: str,
) -> tuple[dict[str, Any], int, int]:
    """Convert the value-axis entries of an `Optimizer.state_dict()` payload.

    Returns the payload with `(entries converted, entries re-encoded)`.
    """
    geometry = geometry_for(model)
    parameters = _flattened_parameters(optimizer)
    names = _parameter_names(model)
    state = state_dict.get("state", {})
    converted = 0
    requantized = 0
    for index, entry in state.items():
        if not isinstance(entry, dict):
            continue
        if not isinstance(index, int) or not 0 <= index < len(parameters):
            raise ValueError(f"optimizer state entry {index!r} has no parameter")
        parameter = parameters[index]
        name = names.get(id(parameter))
        if name is None:
            raise ValueError(
                f"optimizer parameter {index} is not a parameter of the model"
            )
        touched, re_encoded = permute_state_entry(
            entry, name, tuple(parameter.shape), geometry, to_layout
        )
        converted += touched
        requantized += re_encoded
    return state_dict, converted, requantized


@contextmanager
def saving_optimizer_state(
    optimizer: torch.optim.Optimizer | None, model: Any | None
) -> Generator[None]:
    """Convert `optimizer.state_dict()` to grouped order for the duration of the block."""
    if not _CONFIGURED or optimizer is None or model is None:
        yield
        return
    original = optimizer.state_dict

    def state_dict(*args: Any, **kwargs: Any) -> dict[str, Any]:
        payload = original(*args, **kwargs)
        converted, entries, requantized = convert_optimizer_state(
            optimizer, model, payload, to_layout=GROUPED
        )
        logger.info(
            "optimizer state: %d value-axis entries written in grouped order (%d re-encoded)",
            entries,
            requantized,
        )
        return converted

    optimizer.state_dict = state_dict
    try:
        yield
    finally:
        del optimizer.state_dict


@contextmanager
def loading_optimizer_state(
    optimizer: torch.optim.Optimizer | None, model: Any | None
) -> Generator[None]:
    """Convert a loaded `optimizer.pt` from grouped order into the in-memory tiled layout."""
    if not _CONFIGURED or optimizer is None or model is None:
        yield
        return
    original = optimizer.load_state_dict

    def load_state_dict(state_dict: dict[str, Any], *args: Any, **kwargs: Any):
        converted, entries, requantized = convert_optimizer_state(
            optimizer, model, state_dict, to_layout=TILED
        )
        logger.info(
            "optimizer state: %d value-axis entries loaded into the tiled layout (%d re-encoded)",
            entries,
            requantized,
        )
        return original(converted, *args, **kwargs)

    optimizer.load_state_dict = load_state_dict
    try:
        yield
    finally:
        del optimizer.load_state_dict


def _wrap(target: Any, attribute: str, factory: Any) -> None:
    current = getattr(target, attribute)
    if getattr(current, "_gdn_stock_layout", False):
        return
    wrapped = factory(current)
    wrapped._gdn_stock_layout = True
    setattr(target, attribute, wrapped)


def configure_stock_checkpoint_layout() -> None:
    """Install the adapter save/load hooks. Idempotent."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    import peft
    from peft import peft_model
    from peft.utils import save_and_load

    def get_peft_model_state_dict_factory(original):
        def get_peft_model_state_dict(model, *args, **kwargs):
            return stock_layout_state_dict(
                model, original(model, *args, **kwargs), to_layout=GROUPED
            )

        return get_peft_model_state_dict

    def set_peft_model_state_dict_factory(original):
        def set_peft_model_state_dict(model, peft_model_state_dict, *args, **kwargs):
            return original(
                model,
                stock_layout_state_dict(model, peft_model_state_dict, to_layout=TILED),
                *args,
                **kwargs,
            )

        return set_peft_model_state_dict

    # `transformers` and `peft` reach these through different module attributes.
    for module in (save_and_load, peft_model, peft):
        if hasattr(module, "get_peft_model_state_dict"):
            _wrap(
                module, "get_peft_model_state_dict", get_peft_model_state_dict_factory
            )
        if hasattr(module, "set_peft_model_state_dict"):
            _wrap(
                module, "set_peft_model_state_dict", set_peft_model_state_dict_factory
            )

    _CONFIGURED = True
