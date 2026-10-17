"""The GDN value-head axis in both of the layouts it appears in.

llama.cpp keeps the value heads of the gated delta net in tiled order - value heads outer,
key heads inner - because that is what its key-to-value broadcast (`ggml_repeat_4d`) and its
output projection consume. Transformers' convention, and the layout `convert_hf_to_gguf.py`
starts from, is grouped order: the value heads of one key head are contiguous.

`gdn_tiled_value_heads` makes training keep the file's tiled order end to end, which is what
lets the packed base weights run without a per-call input gather. The consequence is that
every adapter factor on the value axis is stored tiled as well, so a checkpoint boundary that
wants transformers' layout has to permute them (`gdn_stock_checkpoints`).

This module owns the geometry of that axis and the two permutations, so the training-side
boundary and the offline checkpoint converter share exactly one definition. It imports neither
peft nor transformers.
"""

from dataclasses import dataclass
from math import prod

import torch

GROUPED = "grouped"
TILED = "tiled"


@dataclass(frozen=True)
class ValueAxisGeometry:
    """Head geometry of the GDN value axis, from the `linear_*` config fields."""

    key_heads: int
    heads_per_key_head: int
    head_dim: int
    key_head_dim: int

    @property
    def value_heads(self) -> int:
        return self.key_heads * self.heads_per_key_head

    @property
    def value_dim(self) -> int:
        """Length of the value axis: `linear_num_value_heads * linear_value_head_dim`."""
        return self.value_heads * self.head_dim

    @property
    def query_key_rows(self) -> int:
        """Rows of `in_proj_*` that precede the value block (`q` and `k` halves)."""
        return 2 * self.key_heads * self.key_head_dim

    @classmethod
    def from_config(cls, config) -> "ValueAxisGeometry":
        text = config
        get_text_config = getattr(config, "get_text_config", None)
        if callable(get_text_config):
            text = get_text_config()
        try:
            key_heads = int(text.linear_num_key_heads)
            value_heads = int(text.linear_num_value_heads)
            head_dim = int(text.linear_value_head_dim)
            key_head_dim = int(text.linear_key_head_dim)
        except AttributeError as exc:
            raise TypeError(
                f"{type(config).__name__} carries no GDN head geometry "
                "(linear_num_key_heads / linear_num_value_heads / linear_value_head_dim / "
                "linear_key_head_dim)"
            ) from exc
        if key_heads <= 0 or key_head_dim <= 0 or head_dim <= 0:
            raise ValueError(
                f"GDN head geometry must be positive, got {key_heads=} {key_head_dim=} {head_dim=}"
            )
        if value_heads <= 0 or value_heads % key_heads:
            raise ValueError(
                f"linear_num_value_heads={value_heads} is not a multiple of linear_num_key_heads={key_heads}"
            )
        return cls(
            key_heads=key_heads,
            heads_per_key_head=value_heads // key_heads,
            head_dim=head_dim,
            key_head_dim=key_head_dim,
        )

    def tiled_from_grouped(self) -> torch.Tensor:
        """Index array with `x_tiled = x_grouped[perm]` - the base converter's permutation."""
        indices = torch.arange(self.value_dim).reshape(
            self.key_heads, self.heads_per_key_head, self.head_dim
        )
        return indices.permute(1, 0, 2).reshape(-1)

    def grouped_from_tiled(self) -> torch.Tensor:
        """Index array with `x_grouped = x_tiled[perm]` - the inverse of `tiled_from_grouped`."""
        return torch.argsort(self.tiled_from_grouped())

    def index(self, to_layout: str) -> torch.Tensor:
        """Permutation that moves a value-axis vector from the tiled layout to `to_layout`."""
        if to_layout == GROUPED:
            return self.grouped_from_tiled()
        if to_layout == TILED:
            return self.tiled_from_grouped()
        raise ValueError(f"unknown value-head layout {to_layout!r}")


@dataclass(frozen=True)
class ValueAxisRule:
    """Where one adapter factor keeps its value-axis elements.

    `axis` is the factor dimension that indexes value elements and `offset` the number of
    elements on that axis that precede the value block (non-zero only for the `k`/`v` fused
    `in_proj_qkv`, whose first rows are the `q` and `k` halves).
    """

    axis: int
    offset: int = 0

    def length(self, geometry: ValueAxisGeometry) -> int:
        return geometry.value_dim

    def full_index(
        self, total: int, geometry: ValueAxisGeometry, to_layout: str
    ) -> torch.Tensor:
        """Index array over the whole `axis` dimension."""
        length = self.length(geometry)
        if total < self.offset + length:
            raise ValueError(
                f"value-axis rule {self.axis}/{self.offset} needs {self.offset + length} elements, tensor has {total}"
            )
        selected = geometry.index(to_layout) + self.offset
        if self.offset == 0 and total == length:
            return selected
        return torch.cat(
            [
                torch.arange(self.offset, dtype=torch.long),
                selected.to(torch.long),
                torch.arange(self.offset + length, total, dtype=torch.long),
            ]
        )

    def permute(
        self, tensor: torch.Tensor, geometry: ValueAxisGeometry, to_layout: str
    ) -> torch.Tensor:
        """Move one adapter factor (or one optimizer state tensor) into `to_layout`."""
        if tensor.ndim <= self.axis:
            raise ValueError(f"tensor {tuple(tensor.shape)} has no axis {self.axis}")
        index = self.full_index(int(tensor.shape[self.axis]), geometry, to_layout)
        return tensor.index_select(self.axis, index.to(tensor.device))

    def chunks(
        self, shape: tuple[int, ...], geometry: ValueAxisGeometry, to_layout: str
    ) -> tuple[tuple[int, int, int], ...]:
        """Whole-unit moves over the flattened tensor: `(src, dst, length)` in elements.

        Each value head is one unit of `head_dim` contiguous elements on its axis, so the
        permutation moves whole units. Laid out flat, the moves are contiguous runs, which is
        what lets a packed 8-bit optimizer state be relabelled without requantizing.
        """
        length = self.length(geometry)
        unit = geometry.head_dim
        if int(shape[self.axis]) < self.offset + length:
            raise ValueError(
                f"shape {shape} is too small for the value-axis rule {self}"
            )
        index = geometry.index(to_layout).tolist()
        for start in range(0, length, unit):
            if index[start] % unit:
                raise ValueError("value-axis permutation does not move whole heads")
        moves: list[tuple[int, int, int]] = []
        if self.axis == 0:
            row = prod(shape[1:]) if len(shape) > 1 else 1
            for start in range(0, length, unit):
                moves.append(
                    (
                        (self.offset + start) * row,
                        (self.offset + index[start]) * row,
                        unit * row,
                    )
                )
        elif self.axis == 1 and len(shape) == 2:
            rows, cols = int(shape[0]), int(shape[1])
            for row in range(rows):
                base = row * cols + self.offset
                for start in range(0, length, unit):
                    moves.append((base + start, base + index[start], unit))
        else:
            raise ValueError(
                f"value-axis rule {self} cannot chunk a tensor of shape {shape}"
            )
        return tuple(moves)


# Adapter factors that live on the value axis. `in_proj_z` produces one gate row per value
# element, `out_proj` consumes them, and `in_proj_qkv` produces `q`, `k`, then the value block.
_QKV = ValueAxisRule(axis=0)
_Z = ValueAxisRule(axis=0)
_OUT = ValueAxisRule(axis=1)

_SUFFIXES: tuple[tuple[str, ValueAxisRule], ...] = (
    (".linear_attn.in_proj_qkv.lora_B.", _QKV),
    (".linear_attn.in_proj_z.lora_B.", _Z),
    (".linear_attn.out_proj.lora_A.", _OUT),
)


# Factors on the value axis whose other side is the hidden axis: nothing to relabel.
_UNMAPPED_SUFFIXES = (
    ".linear_attn.in_proj_qkv.lora_A.",
    ".linear_attn.in_proj_z.lora_A.",
    ".linear_attn.out_proj.lora_B.",
)

# GDN factors on the value axis that this repository does not adapt and has no rule for. Their
# layouts differ from the three above (`conv1d` is the fused q/k/v channel axis, `in_proj_a/b`
# have one element per value head), so refuse rather than silently write the wrong basis.
_UNSUPPORTED = (
    ".linear_attn.conv1d.",
    ".linear_attn.in_proj_a.",
    ".linear_attn.in_proj_b.",
)


def rule_for(name: str, geometry: ValueAxisGeometry) -> ValueAxisRule | None:
    """The value-axis rule for one parameter / adapter tensor name, or None if it has none."""
    # Names arrive either fully qualified (`model.layers.0.linear_attn...`) or module-relative
    # (`linear_attn...`), so match them with a leading dot.
    probe = name if name.startswith(".") else "." + name
    for suffix, rule in _SUFFIXES:
        if suffix in probe:
            if rule is _QKV:
                return ValueAxisRule(axis=0, offset=geometry.query_key_rows)
            return rule
    if any(suffix in probe for suffix in _UNMAPPED_SUFFIXES):
        return None
    if any(suffix in probe for suffix in _UNSUPPORTED):
        raise ValueError(
            f"{name!r} is a GDN value-axis tensor this repository has no layout rule for. "
            "Adapters may only target in_proj_qkv/in_proj_z/out_proj on the value axis"
        )
    return None
