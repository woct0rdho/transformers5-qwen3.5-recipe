"""The widened `dqkwg` launcher must reproduce FLA's kernel's four outputs.

`gdn_bwd_dqkwg.chunk_bwd_dqkwg` is a drop-in for `fla.ops.common.chunk_o.chunk_bwd_dqkwg`: FLA's
kernel, unchanged, launched with `BK=128` and `BV=32` instead of the tiles its collapsed
`CONST_TILING` allows on this device. The change is a different accumulation order, so the gate is a
tolerance and not equality, and the tests also pin the shape contract the caller depends on (the key
head reduction, the gate's axes, the value-head axis on `dw`) and the guards that keep an untrained
configuration from returning plausible numbers.
"""

import pytest
import torch
from fla.ops.common.chunk_o import chunk_bwd_dqkwg as reference_dqkwg

from gdn_bwd_dqkwg import (
    BLOCK_K,
    BLOCK_V,
    chunk_bwd_dqkwg,
    report,
    require_matching_reference,
)

# (batch, sequence, key heads, value heads, key dim, value dim)
SHAPES = (
    (1, 128, 2, 6, 128, 128),
    (2, 256, 4, 12, 128, 128),
    (1, 512, 16, 48, 128, 128),
)
# Wider key tiles sum the same products in a different order, so this is the audit's gradient band.
REFERENCE_RMSE = 1e-2


def _inputs(
    batch: int,
    sequence: int,
    key_heads: int,
    value_heads: int,
    key_dim: int,
    value_dim: int,
    seed: int = 0,
):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    query = torch.randn(
        batch,
        sequence,
        key_heads,
        key_dim,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    key = torch.randn(
        batch,
        sequence,
        key_heads,
        key_dim,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    value = torch.randn(
        batch,
        sequence,
        value_heads,
        value_dim,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    weight = torch.randn(
        batch,
        sequence,
        value_heads,
        key_dim,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    dout = torch.randn(
        batch,
        sequence,
        value_heads,
        value_dim,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    dvalue = torch.randn(
        batch,
        sequence,
        value_heads,
        value_dim,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    gate = (
        -torch.rand(
            batch,
            sequence,
            value_heads,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
        * 0.1
    )
    chunks = sequence // 64
    state = (
        torch.randn(
            batch,
            chunks,
            value_heads,
            key_dim,
            value_dim,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.1
    )
    dstate = (
        torch.randn(
            batch,
            chunks,
            value_heads,
            key_dim,
            value_dim,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.1
    )
    return query, key, value, weight, dout, dvalue, gate, state, dstate


def _run(launcher, query, key, value, weight, dout, dvalue, gate, state, dstate):
    return launcher(
        q=query,
        k=key,
        v=value,
        do=dout,
        h=state,
        dh=dstate,
        w=weight,
        g=gate,
        dv=dvalue,
        scale=128**-0.5,
        chunk_size=64,
        state_v_first=False,
    )


@pytest.mark.parametrize("shape", SHAPES)
def test_matches_fla_reference(shape):
    """All four outputs, with the shapes the caller expects for each."""

    tensors = _inputs(*shape)
    batch, sequence, key_heads = shape[0], shape[1], shape[2]
    value_heads, key_dim = shape[3], shape[4]

    want = _run(reference_dqkwg, *tensors)
    got = _run(chunk_bwd_dqkwg, *tensors)

    assert got[0].shape == (batch, sequence, key_heads, key_dim)
    assert got[1].shape == (batch, sequence, key_heads, key_dim)
    assert got[2].shape == (batch, sequence, value_heads, key_dim)
    assert got[3].shape == (batch, sequence, value_heads)
    assert got[3].dtype == torch.float32
    require_matching_reference(got, want, maximum_relative_rmse=REFERENCE_RMSE)


def test_deterministic():
    """One writer per output element and a fixed reduction order."""

    tensors = _inputs(1, 256, 4, 12, 128, 128)
    first = _run(chunk_bwd_dqkwg, *tensors)
    second = _run(chunk_bwd_dqkwg, *tensors)
    for got, again in zip(first, second, strict=True):
        assert torch.equal(got, again)


def test_reports_its_configuration():
    """The gate reads this to record which kernels were measured."""

    configuration = report()
    assert configuration["block_k"] == BLOCK_K
    assert configuration["block_v"] == BLOCK_V
    assert configuration["installed"] is False


def test_fails_closed_on_an_untrained_configuration():
    """A layout this launcher cannot honour must raise rather than return numbers."""

    tensors = _inputs(1, 256, 4, 12, 128, 128)
    query, key, value, weight, dout, dvalue, gate, state, dstate = tensors
    with pytest.raises(NotImplementedError):
        chunk_bwd_dqkwg(
            q=query,
            k=key,
            v=value,
            do=dout,
            h=state,
            dh=dstate,
            w=weight,
            g=gate,
            dv=dvalue,
            scale=128**-0.5,
            state_v_first=True,
        )
    with pytest.raises(NotImplementedError):
        chunk_bwd_dqkwg(
            q=query,
            k=key,
            v=value,
            do=dout,
            h=state,
            dh=dstate,
            w=weight,
            g=gate,
            dv=dvalue,
            scale=128**-0.5,
            chunk_size=32,
        )
    with pytest.raises(NotImplementedError):
        chunk_bwd_dqkwg(
            q=query, k=key, v=value, do=dout, h=state, dh=dstate, scale=128**-0.5
        )
