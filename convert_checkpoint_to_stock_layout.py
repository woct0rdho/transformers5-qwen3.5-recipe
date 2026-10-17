#!/usr/bin/env python3
"""Rewrite one tiled-layout training checkpoint into transformers' (stock) value-head layout.

`gdn_stock_checkpoints` writes checkpoints in this layout at training time, from the live model.
This script does the same for checkpoints whose run predates it: the adapter factors on the GDN
value axis (`in_proj_qkv` value rows, `in_proj_z` rows, `out_proj` columns) and the optimizer
state of the same parameters are permuted from llama.cpp's tiled order into the grouped order
that transformers, `convert_lora_to_gguf.py` and `peft` expect.

python convert_checkpoint_to_stock_layout.py out_qwen38/checkpoint-14800 \
    --dst out_qwen38/checkpoint-14800-stock

The source directory is never modified. `--verify` (default) reloads the result, converts it
back in memory and compares it with the source.

The optimizer state is numbered by parameter order, so the mapping is rebuilt here from the
adapter's own tensor shapes: the adapter has exactly one state entry per saved factor, the
entries are contiguous per layer, and every value-axis factor has a shape that is unique within
its layer. Nothing is guessed about module ordering.

A checkpoint that is already in the stock layout - anything written since
`gdn_stock_checkpoints` was added, or one trained before the tiled loader existed - must not be
passed here: it would be permuted a second time.
"""

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from gdn_stock_checkpoints import permute_state_entry
from gdn_value_axis import GROUPED, TILED, ValueAxisGeometry, rule_for

ADAPTER_WEIGHTS = "adapter_model.safetensors"
OPTIMIZER_NAME = "optimizer.pt"
_ADAPTER_PREFIX = "base_model.model."
_INTERMEDIATE = 640  # moe_intermediate_size of Qwen3.8-Flash-Next, used by the verifier


def load_adapter(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    with safe_open(str(path), "pt") as handle:
        metadata = handle.metadata() or {}
        tensors = {
            name: handle.get_tensor(name)
            for name in handle.keys()  # noqa: SIM118  # safe_open is not iterable
        }
    return tensors, metadata


def strip_prefix(name: str) -> str:
    return name.removeprefix(_ADAPTER_PREFIX)


def layer_index(parameter: str) -> int:
    parts = parameter.split(".")
    for position, part in enumerate(parts[:-1]):
        if part == "layers" and parts[position + 1].isdigit():
            return int(parts[position + 1])
    raise ValueError(f"adapter factor {parameter!r} is not inside a decoder layer")


def parameter_names(tensors: dict[str, torch.Tensor]) -> list[str]:
    """The adapter factors, with the per-layer block boundaries the optimizer state follows.

    `named_parameters()` walks layers ascending and target modules in registration order. This
    script only needs the sizes of the per-layer blocks, which follow from the factors.
    """
    per_layer: dict[int, list[str]] = defaultdict(list)
    for name in tensors:
        parameter = strip_prefix(name)
        per_layer[layer_index(parameter)].append(parameter)
    layers = sorted(per_layer)
    if layers != list(range(len(layers))):
        raise ValueError(f"adapter layers are not contiguous: {layers[:5]}...")
    order: list[str] = []
    for layer in layers:
        order.extend(sorted(per_layer[layer]))
    return order


def assign_shapes(
    shapes: list[tuple[int, ...]],
    names: list[str],
    tensors: dict[str, torch.Tensor],
) -> list[str]:
    """Name the optimizer state entries of one layer by shape.

    Every value-axis factor has a shape that appears once in its layer, and shapes that repeat
    are factors whose other side is the hidden axis, so any consistent assignment of those is
    equally correct.
    """
    pool: dict[tuple[int, ...], list[str]] = defaultdict(list)
    for name in sorted(names):
        key = name if name in tensors else _ADAPTER_PREFIX + name
        pool[tuple(tensors[key].shape)].append(name)
    assigned: list[str] = []
    for shape in shapes:
        candidates = pool.get(shape)
        if not candidates:
            raise ValueError(
                f"optimizer state of shape {shape} has no adapter factor left to match"
            )
        assigned.append(candidates.pop(0))
    leftover = [name for names in pool.values() for name in names]
    if leftover:
        raise ValueError(
            f"adapter factors without an optimizer state entry: {leftover[:4]}"
        )
    return assigned


def verify_unique_value_axis(
    names: list[str], tensors: dict[str, torch.Tensor], geometry: ValueAxisGeometry
) -> None:
    shapes: dict[tuple[int, ...], list[str]] = defaultdict(list)
    for name in names:
        if rule_for(name, geometry) is not None:
            key = name if name in tensors else _ADAPTER_PREFIX + name
            shapes[tuple(tensors[key].shape)].append(name)
    ambiguous = {shape: names for shape, names in shapes.items() if len(names) > 1}
    if ambiguous:
        raise ValueError(
            f"value-axis factors are not shape-unique within a layer: {next(iter(ambiguous.values()))}"
        )


def check_axis(
    name: str, rule: Any, tensor: torch.Tensor, geometry: ValueAxisGeometry
) -> None:
    """The factor must be exactly as wide as the geometry says, or the permutation is a guess."""
    expected = rule.offset + rule.length(geometry)
    if int(tensor.shape[rule.axis]) != expected:
        raise SystemExit(
            f"{name}: axis {rule.axis} is {tensor.shape[rule.axis]} wide, the geometry implies {expected}"
        )


def convert_adapter(
    source: dict[str, torch.Tensor], geometry: ValueAxisGeometry
) -> tuple[dict[str, torch.Tensor], int]:
    converted: dict[str, torch.Tensor] = {}
    touched = 0
    for name, tensor in source.items():
        rule = rule_for(strip_prefix(name), geometry)
        if rule is None:
            converted[name] = tensor
            continue
        check_axis(name, rule, tensor, geometry)
        converted[name] = rule.permute(tensor, geometry, GROUPED)
        touched += 1
    return converted, touched


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "checkpoint", type=Path, help="checkpoint directory to convert (left untouched)"
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=None,
        help="output directory (default: <checkpoint>-stock)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing output directory"
    )
    parser.add_argument(
        "--geometry",
        type=str,
        default=None,
        help="num_key_heads:heads_per_key_head:head_dim:key_head_dim (default: read from the base "
        "model recorded in the checkpoint's adapter_config.json)",
    )
    args = parser.parse_args()

    source_dir = args.checkpoint.resolve()
    adapter_path = source_dir / ADAPTER_WEIGHTS
    if not adapter_path.is_file():
        print(
            f"{adapter_path} not found: only safetensors checkpoints are supported",
            file=sys.stderr,
        )
        return 2
    destination = (
        args.dst or source_dir.with_name(source_dir.name + "-stock")
    ).resolve()
    if destination.exists():
        if not args.overwrite:
            print(f"{destination} exists (use --overwrite)", file=sys.stderr)
            return 2
        shutil.rmtree(destination)

    geometry = (
        _parse_geometry(args.geometry)
        if args.geometry
        else _geometry_from_config(source_dir)
    )
    print(
        f"value-axis geometry: {geometry.value_heads} value heads x {geometry.head_dim} "
        f"under {geometry.key_heads} key heads (query/key rows {geometry.query_key_rows})"
    )

    shutil.copytree(source_dir, destination)
    print(f"copied {source_dir} -> {destination}")

    tensors, metadata = load_adapter(source_dir / adapter_path.name)
    names = parameter_names(tensors)
    if len(names) != len(tensors):
        raise SystemExit("adapter factors are not unique per layer")
    converted, touched = convert_adapter(tensors, geometry)
    save_file(converted, str(destination / ADAPTER_WEIGHTS), metadata=metadata)
    print(
        f"adapter: {len(tensors)} factors, {touched} on the value axis permuted to grouped order"
    )

    optimizer_path = source_dir / OPTIMIZER_NAME
    if optimizer_path.is_file():
        state = torch.load(optimizer_path, map_location="cpu", weights_only=False)
        parameters = [
            parameter
            for group in state["param_groups"]
            for parameter in group["params"]
        ]
        entries = state["state"]
        if len(parameters) != len(names):
            raise SystemExit(
                f"optimizer numbers {len(parameters)} parameters, the adapter has {len(names)}"
            )
        per_layer = defaultdict(list)
        for name in names:
            per_layer[layer_index(name)].append(name)
        assigned: list[str] = []
        offset = 0
        for layer in sorted(per_layer):
            block = per_layer[layer]
            shapes = [
                tuple(_quant(entries[offset + index])["state1"].shape)
                for index in range(len(block))
            ]
            verify_unique_value_axis(block, tensors, geometry)
            assigned.extend(assign_shapes(shapes, block, tensors))
            offset += len(block)
        if len(assigned) != len(names):
            raise SystemExit("optimizer entries do not line up with the adapter layers")

        converted_entries = 0
        requantized_entries = 0
        for index, name in enumerate(assigned):
            entry = entries[index]
            shape = tuple(_quant(entry)["state1"].shape)
            converted_entry, re_encoded = permute_state_entry(
                entry, name, shape, geometry, GROUPED
            )
            converted_entries += converted_entry
            requantized_entries += re_encoded
        torch.save(state, destination / OPTIMIZER_NAME)
        print(
            f"optimizer: {len(entries)} state entries, {converted_entries} on the value axis "
            f"permuted ({requantized_entries} of them re-encoded around 256-element blocks)"
        )
    else:
        print("optimizer: none")

    code = _verify(source_dir, destination, geometry)
    if code == 0:
        print("verification passed")
    return code


def _geometry_from_config(source_dir: Path) -> ValueAxisGeometry:
    try:
        from transformers import AutoConfig

        config = json.loads(
            (source_dir / "adapter_config.json").read_text(encoding="utf-8")
        )
        base = config.get("base_model_name_or_path")
        if not base:
            raise ValueError("adapter_config.json has no base_model_name_or_path")
        return ValueAxisGeometry.from_config(AutoConfig.from_pretrained(base))
    except Exception as exc:  # noqa: BLE001 - the checkpoint may not carry a loadable base config
        print(
            "could not read the GDN head geometry from the base config "
            f"({type(exc).__name__}: {exc}). Using Qwen3.8-Flash-Next's 16x3x128/128 "
            "(override with --geometry)",
            file=sys.stderr,
        )
        return ValueAxisGeometry(16, 3, 128, 128)


def _parse_geometry(text: str) -> ValueAxisGeometry:
    key_heads, per_key, head_dim, key_head_dim = (int(part) for part in text.split(":"))
    return ValueAxisGeometry(key_heads, per_key, head_dim, key_head_dim)


def _verify(source_dir: Path, destination: Path, geometry: ValueAxisGeometry) -> int:
    """Convert the result back and compare it with the source."""
    original, _ = load_adapter(source_dir / ADAPTER_WEIGHTS)
    converted, _ = load_adapter(destination / ADAPTER_WEIGHTS)
    if set(original) != set(converted):
        print("adapter keys changed", file=sys.stderr)
        return 1
    mismatched = [
        name
        for name in original
        if not torch.equal(
            _back_to_tiled(converted[name], strip_prefix(name), geometry),
            original[name],
        )
    ]
    print(
        f"adapter round trip: {len(original) - len(mismatched)}/{len(original)} factors bit-identical"
    )
    if mismatched:
        print(f"  mismatched: {mismatched[:3]}", file=sys.stderr)
        return 1

    if not (source_dir / OPTIMIZER_NAME).is_file():
        return 0
    original_state = torch.load(
        source_dir / OPTIMIZER_NAME, map_location="cpu", weights_only=False
    )
    converted_state = torch.load(
        destination / OPTIMIZER_NAME, map_location="cpu", weights_only=False
    )
    names = parameter_names(original)
    per_layer = defaultdict(list)
    for name in names:
        per_layer[layer_index(name)].append(name)
    entries_a, entries_b = original_state["state"], converted_state["state"]
    assigned: list[str] = []
    offset = 0
    for layer in sorted(per_layer):
        block = per_layer[layer]
        shapes = [
            tuple(_quant(entries_a[offset + index])["state1"].shape)
            for index in range(len(block))
        ]
        assigned.extend(assign_shapes(shapes, block, original))
        offset += len(block)

    worst_relative = 0.0
    worst_rms = 0.0
    exact = 0
    for index, name in enumerate(assigned):
        if rule_for(name, geometry) is None:
            continue
        entry_a = entries_a[index]
        entry_b = entries_b[index]
        quant_a, quant_b = _quant(entry_a), _quant(entry_b)
        permute_state_entry(
            entry_b, name, tuple(quant_a["state1"].shape), geometry, TILED
        )
        for key in ("state1", "state2"):
            a, b = quant_a[key], quant_b[key]
            if a.dtype == torch.uint8:
                dense_a = _decode(quant_a, key)
                dense_b = _decode(quant_b, key)
            else:
                dense_a, dense_b = a.float(), b.float()
            scale = float(dense_a.abs().max()) or 1.0
            worst_relative = max(
                worst_relative, float((dense_a - dense_b).abs().max()) / scale
            )
            rms = float((dense_a - dense_b).pow(2).mean().sqrt()) / (
                float(dense_a.pow(2).mean().sqrt()) or 1.0
            )
            worst_rms = max(worst_rms, rms)
            if torch.equal(a, b):
                exact += 1
        for key in ("exp_avg", "exp_avg_sq"):
            if key in entry_a and torch.equal(entry_a[key], entry_b[key]):
                exact += 1
    print(
        f"optimizer round trip: {exact} tensors bit-identical, worst relative deviation "
        f"{worst_relative:.2e} (rms {worst_rms:.2e})"
    )
    if worst_relative > 4e-2 or worst_rms > 2e-2:
        print(
            "  optimizer state deviates by more than a re-encoding step",
            file=sys.stderr,
        )
        return 1
    return 0


def _back_to_tiled(
    tensor: torch.Tensor, name: str, geometry: ValueAxisGeometry
) -> torch.Tensor:
    rule = rule_for(name, geometry)
    return tensor if rule is None else rule.permute(tensor, geometry, TILED)


def _quant(entry: dict[str, Any]) -> dict[str, Any]:
    """The dict holding `state1`/`absmax1`/`qmap1`: bitsandbytes nests it for FSDP safety."""
    nested = entry.get("__bnb_optimizer_quant_state__")
    return nested if isinstance(nested, dict) else entry


def _decode(quant: dict[str, Any], key: str) -> torch.Tensor:
    import bitsandbytes.functional as bnb_functional

    suffix = key[-1]
    return bnb_functional.dequantize_blockwise(
        quant[key],
        absmax=quant[f"absmax{suffix}"],
        code=quant[f"qmap{suffix}"],
        blocksize=256,
    )


if __name__ == "__main__":
    raise SystemExit(main())
