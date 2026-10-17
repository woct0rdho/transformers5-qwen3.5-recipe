#!/usr/bin/env python3
"""Check a LoRA GGUF against the (stock-layout) safetensors it was converted from.

`convert_lora_to_gguf.py` assumes transformers' layout, which is what `gdn_stock_checkpoints`
writes (and what `convert_checkpoint_to_stock_layout.py` produces for older checkpoints). This
script rebuilds the mapping from the base converter (`conversion/qwen.py`,
`conversion/qwen4exp.py`) and compares every factor, in f16, with what the GGUF actually holds:
- `in_proj_qkv.lora_B` value rows, `in_proj_z.lora_B` rows and `out_proj.lora_A` columns are
  reordered into llama.cpp's tiled value-head order.
- the fused expert `gate_up` pair is split into the file's `ffn_gate_exps`/`ffn_up_exps`.
- everything else is copied as is.

python verify_lora_gguf.py OUT_QWEN38_STOCK/checkpoint-14800 MODEL.gguf [--geometry 16:3:128:128]
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import gguf
import numpy as np
import torch
from safetensors import safe_open

from gdn_value_axis import TILED, ValueAxisGeometry, rule_for

ADAPTER_WEIGHTS = "adapter_model.safetensors"
_PREFIX = "base_model.model."
# Factors are stored in f16: relative rounding is 2^-11, and values below 6e-5 land on the
# subnormal grid, whose step is 6e-8.
_RTOL = 1e-3
_ATOL = 2e-7


def read_gguf(path: Path) -> dict[str, torch.Tensor]:
    reader = gguf.GGUFReader(str(path))
    tensors: dict[str, torch.Tensor] = {}
    for tensor in reader.tensors:
        array = tensor.data
        if array.dtype == np.uint16:
            array = array.view(np.float16)
        tensors[tensor.name] = torch.from_numpy(np.array(array)).float()
    return tensors


def read_adapter(path: Path) -> dict[str, torch.Tensor]:
    with safe_open(str(path), "pt") as handle:
        return {
            name.removeprefix(_PREFIX): handle.get_tensor(name).float()
            for name in handle.keys()  # noqa: SIM118  # safe_open is not iterable
        }


def read_geometry(override: str | None) -> ValueAxisGeometry:
    if not override:
        raise SystemExit(
            "pass --geometry num_key_heads:heads_per_key_head:head_dim:key_head_dim"
        )
    key_heads, per_key, head_dim, key_head_dim = (
        int(part) for part in override.split(":")
    )
    return ValueAxisGeometry(key_heads, per_key, head_dim, key_head_dim)


def expected_tensors(
    adapter: dict[str, torch.Tensor], geometry: ValueAxisGeometry, intermediate: int
) -> dict[str, tuple[str, str, int]]:
    """GGUF tensor name -> (adapter factor, operation, argument)."""
    plan: dict[str, tuple[str, str, int]] = {}
    for name in adapter:
        parts = name.split(".")
        if len(parts) < 4 or parts[0] != "model" or parts[1] != "layers":
            raise SystemExit(f"unexpected adapter factor {name!r}")
        layer = int(parts[2])
        block = f"blk.{layer}"
        suffix = name.split(".", 3)[3]
        leaf = "lora_a" if ".lora_A." in name else "lora_b"
        if suffix.startswith("linear_attn.in_proj_qkv."):
            plan[f"{block}.attn_qkv.weight.{leaf}"] = (name, "value", 0)
        elif suffix.startswith("linear_attn.in_proj_z."):
            plan[f"{block}.attn_gate.weight.{leaf}"] = (name, "value", 0)
        elif suffix.startswith("linear_attn.out_proj."):
            plan[f"{block}.ssm_out.weight.{leaf}"] = (name, "value", 0)
        elif suffix.startswith("self_attn.q_proj."):
            plan[f"{block}.attn_q.weight.{leaf}"] = (name, "copy", 0)
        elif suffix.startswith("self_attn.k_proj."):
            plan[f"{block}.attn_k.weight.{leaf}"] = (name, "copy", 0)
        elif suffix.startswith("self_attn.v_proj."):
            plan[f"{block}.attn_v.weight.{leaf}"] = (name, "copy", 0)
        elif suffix.startswith("self_attn.o_proj."):
            plan[f"{block}.attn_output.weight.{leaf}"] = (name, "copy", 0)
        elif suffix.startswith("mlp.shared_expert.gate_proj."):
            plan[f"{block}.ffn_gate_shexp.weight.{leaf}"] = (name, "copy", 0)
        elif suffix.startswith("mlp.shared_expert.up_proj."):
            plan[f"{block}.ffn_up_shexp.weight.{leaf}"] = (name, "copy", 0)
        elif suffix.startswith("mlp.shared_expert.down_proj."):
            plan[f"{block}.ffn_down_shexp.weight.{leaf}"] = (name, "copy", 0)
        elif suffix.startswith("mlp.experts.lora_A_down."):
            plan[f"{block}.ffn_down_exps.weight.lora_a"] = (name, "copy", 0)
        elif suffix.startswith("mlp.experts.lora_B_down."):
            plan[f"{block}.ffn_down_exps.weight.lora_b"] = (name, "copy", 0)
        elif suffix.startswith(("mlp.experts.lora_A.", "mlp.experts.lora_B.")):
            # The checkpoint fuses gate and up. The file keeps them apart, gate first.
            for half, key in ((0, "ffn_gate_exps"), (1, "ffn_up_exps")):
                if ".lora_A." in suffix:
                    plan[f"{block}.{key}.weight.lora_a"] = (name, "copy", 0)
                else:
                    plan[f"{block}.{key}.weight.lora_b"] = (
                        name,
                        "split",
                        half * intermediate,
                    )
        else:
            raise SystemExit(f"no GGUF target known for adapter factor {name!r}")
    return plan


def apply(
    adapter: dict[str, torch.Tensor],
    name: str,
    operation: str,
    argument: int,
    geometry: ValueAxisGeometry,
    intermediate: int,
):
    factor = adapter[name]
    if operation == "copy":
        return factor
    if operation == "split":
        return factor[:, argument : argument + intermediate, :]
    rule = rule_for(name, geometry)
    if rule is None:
        return factor
    # The stock checkpoint is grouped. The file is tiled, which is what the converter writes.
    return rule.permute(factor, geometry, TILED)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "checkpoint", type=Path, help="stock-layout checkpoint directory"
    )
    parser.add_argument(
        "gguf", type=Path, help="adapter GGUF produced by convert_lora_to_gguf.py"
    )
    parser.add_argument(
        "--geometry",
        type=str,
        default=None,
        help="num_key_heads:heads_per_key:head_dim:key_head_dim",
        required=True,
    )
    parser.add_argument(
        "--expert-intermediate",
        type=int,
        default=640,
        help="moe_intermediate_size, i.e. the width of one half of the fused expert gate/up pair",
    )
    args = parser.parse_args()

    geometry = read_geometry(args.geometry)
    adapter = read_adapter(args.checkpoint / ADAPTER_WEIGHTS)
    actual = read_gguf(args.gguf)
    plan = expected_tensors(adapter, geometry, args.expert_intermediate)

    missing = sorted(set(plan) - set(actual))
    unexpected = sorted(set(actual) - set(plan))
    print(
        f"adapter factors: {len(adapter)}   GGUF tensors: {len(actual)}   expected: {len(plan)}"
    )
    if missing or unexpected:
        print(f"missing: {missing[:4]}", file=sys.stderr)
        print(f"unexpected: {unexpected[:4]}", file=sys.stderr)

    status = 0
    worst: list[tuple[float, str]] = []
    too_precise: list[str] = []
    seen: dict[str, int] = defaultdict(int)
    for gguf_name, (factor_name, operation, argument) in plan.items():
        if gguf_name not in actual:
            continue
        seen[factor_name] += 1
        reference = apply(
            adapter,
            factor_name,
            operation,
            argument,
            geometry,
            args.expert_intermediate,
        )
        stored = actual[gguf_name]
        if tuple(stored.shape) != tuple(reference.shape):
            print(
                f"shape mismatch {gguf_name}: {tuple(stored.shape)} vs {tuple(reference.shape)}",
                file=sys.stderr,
            )
            status = 1
            continue
        excess = (
            ((stored - reference).abs() - (_ATOL + _RTOL * reference.abs()))
            .max()
            .item()
        )
        worst.append(
            (
                (stored - reference).abs().max().item()
                / (reference.abs().max().item() or 1.0),
                gguf_name,
            )
        )
        if excess > 0:
            too_precise.append(gguf_name)
    unmapped = sorted(name for name in adapter if name not in seen)
    if unmapped:
        print(f"adapter factors with no GGUF tensor: {unmapped[:4]}", file=sys.stderr)
        status = 1
    worst.sort(reverse=True)
    for deviation, gguf_name in worst[:5]:
        print(f"  {deviation:.2e}  {gguf_name}")
    print(f"tensors compared: {len(worst)}   outside f16 tolerance: {len(too_precise)}")
    if too_precise:
        print(f"  worst: {too_precise[:4]}", file=sys.stderr)
        status = 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())
