"""The tiled value-head convention is the same function as the grouped one, with no gather.

Two things have to hold for `gdn_tiled_value_heads` to be a pure relabeling: the loader has to stop
reordering the value-indexed tensors, and the GatedDeltaNet broadcast has to tile the key heads instead
of interleaving them. These tests run a real `Qwen4ExpTextGatedDeltaNet` with tiny dimensions both ways
and compare the results, then check the loader's mapping and the reach of the rewrite.
"""

from types import SimpleNamespace
from typing import Any, cast

import torch
from transformers.integrations.gguf import utils as gguf_utils
from transformers.integrations.gguf.gguf_conversion_mapping import (
    TiledToGroupedInputs,
    TiledToGroupedRows,
    _qwen35,
)
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextGatedDeltaNet

from gdn_tiled_value_heads import (
    configure_tiled_value_heads,
    grouped_broadcast,
    report,
)
from test_support import assert_close_mixed_precision, require_grad

_H_K = 2
_H_V = 4
_D_K = 16
_D_V = 32
_HEADS_PER_K = _H_V // _H_K
_HIDDEN = 2 * _H_K * _D_K + _H_V * _D_V
_QUERY_KEY_ROWS = 2 * _D_K * _H_K
_ROWS = 32
_VALUE_REORDER_OPERATIONS = ("TiledToGroupedRows", "TiledToGroupedInputs")


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=_HIDDEN,
        linear_num_value_heads=_H_V,
        linear_num_key_heads=_H_K,
        linear_key_head_dim=_D_K,
        linear_value_head_dim=_D_V,
        linear_conv_kernel_dim=4,
        hidden_act="silu",
        num_hidden_layers=1,
        output_gate_type="sigmoid",
        rms_norm_eps=1e-6,
        layer_types=["linear_attention"],
    )


def _gdn() -> Qwen4ExpTextGatedDeltaNet:
    torch.manual_seed(20261208)
    module = (
        Qwen4ExpTextGatedDeltaNet(cast(Any, _config()), 0).cuda().to(torch.bfloat16)
    )
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(std=0.05)
        # `A_log` drives `-exp(A_log)`, which wants to stay small.
        module.A_log.normal_(mean=-1.0, std=0.1)
    return module


def _random_hidden(seed: int) -> torch.Tensor:
    return torch.randn(
        1,
        _ROWS,
        _HIDDEN,
        generator=torch.Generator(device="cuda").manual_seed(seed),
        device="cuda",
        dtype=torch.bfloat16,
    )


def _reorder_rows(
    weight: torch.Tensor, permutation: torch.Tensor, offset: int
) -> torch.Tensor:
    """The row reorder `PermuteRows` applies: the head is kept, the tail is permuted."""

    if offset == 0:
        return weight[permutation].contiguous()
    return torch.cat((weight[:offset], weight[offset:][permutation])).contiguous()


def _to_grouped(module: Qwen4ExpTextGatedDeltaNet) -> None:
    """Rewrite one module's file-order (tiled) weights into the grouped convention, in place."""

    device = module.in_proj_qkv.weight.device
    value = TiledToGroupedRows(_H_K, _HEADS_PER_K, _D_V).permutation.to(device)
    value_offset = TiledToGroupedRows(
        _H_K, _HEADS_PER_K, _D_V, offset=_QUERY_KEY_ROWS
    ).permutation.to(device)
    heads = TiledToGroupedRows(_H_K, _HEADS_PER_K).permutation.to(device)
    columns = TiledToGroupedInputs(_H_K, _HEADS_PER_K, _D_V).permutation.to(device)
    with torch.no_grad():
        module.in_proj_qkv.weight.data = _reorder_rows(
            module.in_proj_qkv.weight.data, value_offset, _QUERY_KEY_ROWS
        )
        module.conv1d.weight.data = _reorder_rows(
            module.conv1d.weight.data, value_offset, _QUERY_KEY_ROWS
        )
        module.in_proj_z.weight.data = _reorder_rows(
            module.in_proj_z.weight.data, value, 0
        )
        module.in_proj_a.weight.data = _reorder_rows(
            module.in_proj_a.weight.data, heads, 0
        )
        module.in_proj_b.weight.data = _reorder_rows(
            module.in_proj_b.weight.data, heads, 0
        )
        module.A_log.data = module.A_log.data[heads].contiguous()
        module.dt_bias.data = module.dt_bias.data[heads].contiguous()
        module.out_proj.weight.data = module.out_proj.weight.data[
            :, columns
        ].contiguous()


def test_tiled_and_grouped_conventions_agree() -> None:
    configure_tiled_value_heads()
    module = _gdn()
    hidden = _random_hidden(7)

    tiled = module(hidden)

    _to_grouped(module)
    with grouped_broadcast():
        grouped = module(hidden)

    assert tiled.shape == grouped.shape == (1, _ROWS, _HIDDEN)
    assert_close_mixed_precision(
        tiled, grouped, minimum_cosine=0.999, maximum_relative_rmse=5e-3
    )


def test_tiled_convention_gradients_flow() -> None:
    configure_tiled_value_heads()
    module = _gdn()
    leaf = _random_hidden(11).detach().requires_grad_(True)
    module(leaf).float().square().mean().backward()

    gradient = require_grad(leaf)
    assert bool(torch.isfinite(gradient).all())
    assert bool(torch.any(gradient != 0))
    # The weights are trainable in this toy, so the output projection collects its own gradient.
    assert module.out_proj.weight.grad is not None
    assert bool(torch.isfinite(module.out_proj.weight.grad).all())


def test_loader_drops_exactly_the_value_reorders() -> None:
    configure_tiled_value_heads()
    config = _config()
    config.get_text_config = lambda: config  # type: ignore[attr-defined]

    def operations(mapping: list[Any]) -> list[str]:
        return [
            type(operation).__name__
            for entry in mapping
            for operation in (getattr(entry, "operations", None) or [])
        ]

    before = operations(_qwen35(cast(Any, config)))
    after = operations(
        gguf_utils.get_gguf_conversion_mapping("qwen35", cast(Any, config))
    )

    assert [name for name in before if name not in _VALUE_REORDER_OPERATIONS] == after
    assert (
        before.count("TiledToGroupedRows") + before.count("TiledToGroupedInputs") == 8
    )
    assert report()["value_reorders_dropped"] == 8
    # Everything else the mapping does survives, including the log negation on `A_log` and the
    # conv1d unsqueeze, which rides with a dropped reorder.
    assert after.count("LogNegate") == 1
    assert after.count("SubtractOne") == 5
    assert after.count("Unsqueeze") == 1


def test_the_rewrite_only_happens_inside_a_gated_delta_net_forward() -> None:
    configure_tiled_value_heads()
    tensor = torch.arange(3 * _H_K, dtype=torch.float32).reshape(1, 3, _H_K, 1)
    # Outside a forward the method keeps its grouped, interleaved semantics.
    outside = tensor.repeat_interleave(_HEADS_PER_K, dim=2)
    assert outside[0, 0, :, 0].tolist() == [0.0, 0.0, 1.0, 1.0]

    module = _gdn()
    before = report()
    with torch.no_grad():
        module(_random_hidden(13))
    after = report()
    # One more layer took the tiled broadcast, with two expansions: the query and the key.
    assert after["layers"] == before["layers"] + 1
    assert after["tiled_rewrites"] >= before["tiled_rewrites"] + 2
