"""Tests for the checkpoint boundary that keeps stock-layout adapters around a tiled model.

Everything here is a permutation identity plus the two boundaries it feeds: the adapter state
dict peft saves/loads, and `optimizer.pt`. The permutations must be exact where the tiled and
grouped layouts agree on quantization blocks, and a single re-rounding where they do not.
"""

from pathlib import Path

import pytest
import torch
from torch import nn
from transformers import PretrainedConfig

from gdn_stock_checkpoints import (
    GROUPED,
    TILED,
    configure_stock_checkpoint_layout,
    loading_optimizer_state,
    permute_state_entry,
    saving_optimizer_state,
    stock_layout_state_dict,
)
from gdn_value_axis import ValueAxisGeometry, rule_for

GEOMETRY = ValueAxisGeometry(
    key_heads=2, heads_per_key_head=3, head_dim=4, key_head_dim=5
)
VALUE_DIM = GEOMETRY.value_dim  # 24
QUERY_KEY_ROWS = GEOMETRY.query_key_rows  # 20
HIDDEN = 6
RANK = 2


class _Config(PretrainedConfig):
    model_type = "gdn_stock_layout_test"

    def __init__(self) -> None:
        super().__init__()
        self.linear_num_key_heads = GEOMETRY.key_heads
        self.linear_num_value_heads = GEOMETRY.value_heads
        self.linear_value_head_dim = GEOMETRY.head_dim
        self.linear_key_head_dim = GEOMETRY.key_head_dim

    def get_text_config(self, *args, **kwargs):
        return self


class _Model(nn.Module):
    """Carrier for the three value-axis factors plus one factor with no value axis."""

    def __init__(self) -> None:
        super().__init__()
        self.config = _Config()
        linear_attn = nn.Module()
        linear_attn.in_proj_qkv = nn.Linear(
            HIDDEN, QUERY_KEY_ROWS + VALUE_DIM, bias=False
        )
        linear_attn.in_proj_z = nn.Linear(HIDDEN, VALUE_DIM, bias=False)
        linear_attn.out_proj = nn.Linear(VALUE_DIM, HIDDEN, bias=False)
        self.linear_attn = linear_attn
        self_attn = nn.Module()
        self_attn.q_proj = nn.Linear(HIDDEN, 2 * HIDDEN, bias=False)
        self.self_attn = self_attn

        for module in (
            linear_attn.in_proj_qkv,
            linear_attn.in_proj_z,
            linear_attn.out_proj,
        ):
            module.lora_A = nn.Linear(module.in_features, RANK, bias=False)
            module.lora_B = nn.Linear(RANK, module.out_features, bias=False)
        self_attn.q_proj.lora_A = nn.Linear(HIDDEN, RANK, bias=False)
        self_attn.q_proj.lora_B = nn.Linear(RANK, 2 * HIDDEN, bias=False)

    def factors(self) -> dict[str, torch.Tensor]:
        return {
            f"{prefix}.{leaf}.weight": getattr(module, leaf).weight
            for prefix, module in (
                ("linear_attn.in_proj_qkv", self.linear_attn.in_proj_qkv),
                ("linear_attn.in_proj_z", self.linear_attn.in_proj_z),
                ("linear_attn.out_proj", self.linear_attn.out_proj),
                ("self_attn.q_proj", self.self_attn.q_proj),
            )
            for leaf in ("lora_A", "lora_B")
        }


def _model() -> _Model:
    torch.manual_seed(0)
    return _Model()


def test_tiled_and_grouped_permutations_are_inverse() -> None:
    tiled_from_grouped = GEOMETRY.tiled_from_grouped()
    grouped_from_tiled = GEOMETRY.grouped_from_tiled()
    identity = torch.arange(VALUE_DIM)
    assert torch.equal(tiled_from_grouped[grouped_from_tiled], identity)
    assert not torch.equal(tiled_from_grouped, grouped_from_tiled), (
        "a self-inverse permutation proves nothing"
    )


def test_permutation_moves_whole_value_heads() -> None:
    for start in GEOMETRY.tiled_from_grouped().tolist()[:: GEOMETRY.head_dim]:
        assert start % GEOMETRY.head_dim == 0


def test_rules_pick_the_value_axis_side() -> None:
    qkv_b = rule_for(".linear_attn.in_proj_qkv.lora_B.weight", GEOMETRY)
    z_b = rule_for(".linear_attn.in_proj_z.lora_B.weight", GEOMETRY)
    out_a = rule_for(".linear_attn.out_proj.lora_A.weight", GEOMETRY)
    assert qkv_b is not None
    assert z_b is not None
    assert out_a is not None
    assert (qkv_b.axis, qkv_b.offset) == (0, QUERY_KEY_ROWS)
    assert (z_b.axis, z_b.offset) == (0, 0)
    assert (out_a.axis, out_a.offset) == (1, 0)
    for name in (
        ".linear_attn.in_proj_qkv.lora_A.weight",
        ".linear_attn.in_proj_z.lora_A.weight",
        ".linear_attn.out_proj.lora_B.weight",
        ".self_attn.q_proj.lora_A.weight",
        ".mlp.experts.lora_A.weight",
    ):
        assert rule_for(name, GEOMETRY) is None, name


def test_unsupported_value_axis_factors_fail_closed() -> None:
    with pytest.raises(ValueError, match="no layout rule"):
        rule_for(".linear_attn.conv1d.weight", GEOMETRY)


def test_permute_leaves_the_hidden_side_alone() -> None:
    rule = rule_for(".linear_attn.in_proj_qkv.lora_B.weight", GEOMETRY)
    assert rule is not None
    factor = torch.arange((QUERY_KEY_ROWS + VALUE_DIM) * RANK, dtype=torch.float32)
    factor = factor.reshape(QUERY_KEY_ROWS + VALUE_DIM, RANK)
    moved = rule.permute(factor, GEOMETRY, GROUPED)
    assert torch.equal(moved[:QUERY_KEY_ROWS], factor[:QUERY_KEY_ROWS]), (
        "q/k rows must not move"
    )
    assert not torch.equal(moved[QUERY_KEY_ROWS:], factor[QUERY_KEY_ROWS:])
    assert torch.equal(rule.permute(moved, GEOMETRY, TILED), factor)


def test_state_dict_round_trip_is_exact() -> None:
    model = _model()
    source = model.factors()
    grouped = stock_layout_state_dict(model, source, to_layout=GROUPED)
    assert set(grouped) == set(source)
    assert not torch.equal(
        grouped["linear_attn.in_proj_qkv.lora_B.weight"],
        source["linear_attn.in_proj_qkv.lora_B.weight"],
    )
    assert torch.equal(
        grouped["self_attn.q_proj.lora_A.weight"],
        source["self_attn.q_proj.lora_A.weight"],
    )
    back = stock_layout_state_dict(model, grouped, to_layout=TILED)
    for name, tensor in source.items():
        assert torch.equal(back[name], tensor), name


def test_stock_layout_is_the_converter_permutation() -> None:
    """`_reorder_v_heads` turns grouped into tiled, so grouped must be `tiled[index]` reversed."""
    model = _model()
    tiled = model.factors()["linear_attn.in_proj_z.lora_B.weight"]
    grouped = stock_layout_state_dict(
        model, {"linear_attn.in_proj_z.lora_B.weight": tiled}, to_layout=GROUPED
    )["linear_attn.in_proj_z.lora_B.weight"]
    assert torch.equal(grouped[GEOMETRY.tiled_from_grouped()], tiled)


def _state_after_one_step(model: nn.Module) -> tuple[torch.optim.Optimizer, dict]:
    for parameter in model.parameters():
        parameter.grad = torch.randn_like(parameter)
    optimizer = torch.optim.AdamW(list(model.parameters()), lr=1e-3)
    optimizer.step()
    snapshot = {
        index: {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in optimizer.state[parameter].items()
        }
        for index, parameter in enumerate(_flat(optimizer))
    }
    return optimizer, snapshot


def _flat(optimizer: torch.optim.Optimizer) -> list[torch.Tensor]:
    return [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]


def test_float_optimizer_state_round_trip(tmp_path: Path) -> None:
    configure_stock_checkpoint_layout()
    model = _model()
    optimizer, snapshot = _state_after_one_step(model)

    with saving_optimizer_state(optimizer, model):
        saved = optimizer.state_dict()
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    flat = _flat(optimizer)
    moved = [
        names[id(flat[index])]
        for index, entry in saved["state"].items()
        if not torch.equal(entry["exp_avg"], snapshot[index]["exp_avg"])
    ]
    assert sorted(moved) == sorted(
        [
            "linear_attn.in_proj_qkv.lora_B.weight",
            "linear_attn.in_proj_z.lora_B.weight",
            "linear_attn.out_proj.lora_A.weight",
        ]
    )

    reloaded = torch.optim.AdamW(list(model.parameters()), lr=1e-3)
    with loading_optimizer_state(reloaded, model):
        reloaded.load_state_dict(saved)
    # `accelerate` reloads the optimizer state onto its device, so compare on the host.
    for index, parameter in enumerate(_flat(reloaded)):
        for key, value in reloaded.state[parameter].items():
            if torch.is_tensor(value) and value.dtype.is_floating_point:
                assert torch.equal(value.cpu(), snapshot[index][key].cpu()), (
                    index,
                    key,
                )


def _quantized_entry(shape: tuple[int, ...], scale: float = 0.01) -> dict:
    bnb = pytest.importorskip("bitsandbytes.functional")
    pytest.importorskip("bitsandbytes")
    qmap = bnb.create_dynamic_map(signed=True)
    dense = torch.randn(shape) * scale
    codes, state = bnb.quantize_blockwise(dense, code=qmap, blocksize=256)
    return {
        "state1": codes,
        "absmax1": state.absmax,
        "qmap1": qmap,
        "state2": codes.clone(),
        "absmax2": state.absmax.clone(),
        "qmap2": bnb.create_dynamic_map(signed=False),
        "step": 1,
    }


def test_quantized_state_is_exact_where_blocks_align() -> None:
    """`in_proj_qkv`/`in_proj_z` rows move in 512-element runs, so codes and absmax move untouched."""
    bnb = pytest.importorskip("bitsandbytes.functional")
    pytest.importorskip("bitsandbytes")
    geometry = ValueAxisGeometry(
        key_heads=2, heads_per_key_head=2, head_dim=128, key_head_dim=4
    )
    shape = (geometry.value_dim, 4)
    entry = _quantized_entry(shape)
    reference = {
        key: value.clone() for key, value in entry.items() if torch.is_tensor(value)
    }

    rule = rule_for(".linear_attn.in_proj_z.lora_B.weight", geometry)
    assert rule is not None
    assert all(
        src % 256 == 0 and dst % 256 == 0 and length % 256 == 0
        for src, dst, length in rule.chunks(shape, geometry, GROUPED)
    )
    permute_state_entry(
        entry, ".linear_attn.in_proj_z.lora_B.weight", shape, geometry, GROUPED
    )
    assert not torch.equal(entry["state1"], reference["state1"])
    permute_state_entry(
        entry, ".linear_attn.in_proj_z.lora_B.weight", shape, geometry, TILED
    )
    for key, tensor in reference.items():
        assert torch.equal(entry[key], tensor), key
    del bnb


def test_quantized_state_survives_the_straddling_columns() -> None:
    """`out_proj.lora_A` columns are 128 elements, so they split the 256-element blocks."""
    bnb = pytest.importorskip("bitsandbytes.functional")
    pytest.importorskip("bitsandbytes")
    geometry = ValueAxisGeometry(
        key_heads=2, heads_per_key_head=2, head_dim=128, key_head_dim=4
    )
    shape = (4, geometry.value_dim)
    entry = _quantized_entry(shape)
    reference = bnb.dequantize_blockwise(
        entry["state1"], absmax=entry["absmax1"], code=entry["qmap1"], blocksize=256
    )

    permute_state_entry(
        entry, ".linear_attn.out_proj.lora_A.weight", shape, geometry, GROUPED
    )
    permute_state_entry(
        entry, ".linear_attn.out_proj.lora_A.weight", shape, geometry, TILED
    )
    decoded = bnb.dequantize_blockwise(
        entry["state1"], absmax=entry["absmax1"], code=entry["qmap1"], blocksize=256
    )
    # Re-encoding half-blocks re-rounds each element once, i.e. the same 1.4%-of-absmax step the
    # optimizer kernel applies on every step - no worse.
    deviation = float((decoded - reference).abs().max()) / float(reference.abs().max())
    assert deviation < 0.03, deviation
    rms = float((decoded - reference).pow(2).mean().sqrt()) / float(
        reference.pow(2).mean().sqrt()
    )
    assert rms < 2e-2, rms


def test_peft_save_writes_grouped_and_load_restores_tiled(tmp_path: Path) -> None:
    pytest.importorskip("peft")
    from peft import LoraConfig, get_peft_model
    from safetensors import safe_open

    configure_stock_checkpoint_layout()
    model = get_peft_model(
        _model(),
        LoraConfig(
            r=RANK,
            lora_alpha=RANK,
            target_modules=["in_proj_qkv", "in_proj_z", "out_proj", "q_proj"],
            init_lora_weights=False,
        ),
    )
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.add_(torch.randn_like(parameter) * 0.01)
    tiled = {
        name.removeprefix("base_model.model."): parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    directory = tmp_path / "checkpoint-1"
    model.save_pretrained(str(directory))

    with safe_open(str(directory / "adapter_model.safetensors"), "pt") as handle:
        stored = {
            name: handle.get_tensor(name)
            for name in handle.keys()  # noqa: SIM118  # safe_open is not iterable
        }
    assert len(stored) == len(tiled)
    for name, tensor in stored.items():
        parameter = (
            name.removeprefix("base_model.model.")
            .replace(".lora_A.", ".lora_A.default.")
            .replace(".lora_B.", ".lora_B.default.")
        )
        rule = rule_for(name, GEOMETRY)
        expected = (
            tiled[parameter]
            if rule is None
            else rule.permute(tiled[parameter], GEOMETRY, GROUPED)
        )
        assert torch.equal(tensor, expected), name

    model.load_adapter(str(directory), adapter_name="default", is_trainable=True)
    for name, parameter in model.named_parameters():
        key = name.removeprefix("base_model.model.")
        if key in tiled:
            assert torch.equal(parameter.detach(), tiled[key]), key


def test_hooks_survive_accelerates_optimizer_wrapper(tmp_path: Path) -> None:
    """`Trainer` saves through `AcceleratedOptimizer`, which forwards `state_dict`/`load_state_dict`."""
    accelerate = pytest.importorskip("accelerate")
    configure_stock_checkpoint_layout()
    model = _model()
    optimizer, snapshot = _state_after_one_step(model)
    wrapped = accelerate.Accelerator().prepare(optimizer)

    with saving_optimizer_state(wrapped, model):
        saved = wrapped.state_dict()
    flat = _flat(optimizer)
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    moved = {
        names[id(flat[index])]
        for index, entry in saved["state"].items()
        if not torch.equal(entry["exp_avg"], snapshot[index]["exp_avg"])
    }
    assert moved == {
        "linear_attn.in_proj_qkv.lora_B.weight",
        "linear_attn.in_proj_z.lora_B.weight",
        "linear_attn.out_proj.lora_A.weight",
    }

    reloaded = torch.optim.AdamW(list(model.parameters()), lr=1e-3)
    wrapped_reload = accelerate.Accelerator().prepare(reloaded)
    with loading_optimizer_state(wrapped_reload, model):
        wrapped_reload.load_state_dict(saved)
    # `accelerate` reloads the optimizer state onto its device, so compare on the host.
    for index, parameter in enumerate(_flat(reloaded)):
        for key, value in reloaded.state[parameter].items():
            if torch.is_tensor(value) and value.dtype.is_floating_point:
                assert torch.equal(value.cpu(), snapshot[index][key].cpu()), (
                    index,
                    key,
                )
