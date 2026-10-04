"""The rewritten state-gradient walk must reproduce FLA's kernel, in values and in layout.

`gdn_bwd_dhu.chunk_gated_delta_rule_bwd_dhu` is a drop-in for FLA's function of the same name,
called from inside FLA's chunked backward where the call is not tracked by autograd. It differs
from FLA's kernel in one place that matters: the decayed query operand is cast back to BF16
before its dot. FLA multiplies the BF16 operand by an FP32 decay and then casts to the dtype it
now has, which is FP32, so its dot runs as FP32 FMA on the VALU instead of on the matrix cores.

These tests run both on the same tensors and compare the two returned tensors, then check the
contract the caller depends on: the output layout, that the walk is deterministic, and that a
configuration this kernel was not written for fails closed instead of returning plausible
numbers.
"""

import pytest
import torch
from fla.ops.common.chunk_delta_h import (
    chunk_gated_delta_rule_bwd_dhu as reference_bwd_dhu,
)

from gdn_bwd_dhu import (
    BLOCK_T,
    chunk_gated_delta_rule_bwd_dhu,
    report,
    require_matching_reference,
)

# (batch, sequence, key heads, value heads, key dim, value dim)
SHAPES = (
    (1, 128, 2, 6, 128, 128),
    (2, 256, 4, 12, 128, 128),
    (1, 512, 16, 48, 128, 128),
)
# The BF16 query operand is a precision change, not a reordering: it moves the dot from FP32
# FMA to the matrix cores, so the tolerance is the one the audit's gradient gates use.
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
    return query, key, weight, dout, dvalue, gate


def _run(launcher, q, k, w, do, dv, g, sequence):
    return launcher(
        q=q,
        k=k,
        w=w,
        g=g,
        do=do,
        dv=dv,
        scale=128**-0.5,
        chunk_size=BLOCK_T,
        state_v_first=False,
    )


@pytest.mark.parametrize("shape", SHAPES)
def test_matches_fla_reference(shape):
    """Both returned tensors, and their dtypes and shapes, against FLA's kernel."""

    query, key, weight, dout, dvalue, gate = _inputs(*shape)
    sequence = shape[1]

    want = _run(reference_bwd_dhu, query, key, weight, dout, dvalue, gate, sequence)
    got = _run(
        chunk_gated_delta_rule_bwd_dhu, query, key, weight, dout, dvalue, gate, sequence
    )

    assert got[0].shape == want[0].shape
    assert got[2].shape == want[2].shape
    assert got[0].dtype == want[0].dtype
    assert got[2].dtype == want[2].dtype
    require_matching_reference(got, want, maximum_relative_rmse=REFERENCE_RMSE)


def test_deterministic():
    """The walk keeps one writer per output element and no atomics, so two runs agree bitwise."""

    query, key, weight, dout, dvalue, gate = _inputs(1, 256, 4, 12, 128, 128)
    first = _run(
        chunk_gated_delta_rule_bwd_dhu, query, key, weight, dout, dvalue, gate, 256
    )
    second = _run(
        chunk_gated_delta_rule_bwd_dhu, query, key, weight, dout, dvalue, gate, 256
    )
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[2], second[2])


def test_reports_its_configuration():
    """The gate reads this to record which kernels were measured."""

    configuration = report()
    assert configuration["block_t"] == BLOCK_T
    assert configuration["query_operand_dtype"] == "bfloat16"
    assert configuration["installed"] is False


def test_fails_closed_on_an_untrained_configuration():
    """Refusing is the point: a layout this kernel cannot honour must not return numbers."""

    query, key, weight, dout, dvalue, gate = _inputs(1, 256, 4, 12, 128, 128)
    with pytest.raises(NotImplementedError):
        chunk_gated_delta_rule_bwd_dhu(
            q=query,
            k=key,
            w=weight,
            g=gate,
            do=dout,
            dv=dvalue,
            scale=128**-0.5,
            state_v_first=True,
        )
    with pytest.raises(NotImplementedError):
        chunk_gated_delta_rule_bwd_dhu(
            q=query,
            k=key,
            w=weight,
            g=gate,
            do=dout,
            dv=dvalue,
            scale=128**-0.5,
            chunk_size=32,
        )
    with pytest.raises(NotImplementedError):
        chunk_gated_delta_rule_bwd_dhu(
            q=query,
            k=key,
            w=weight,
            g=gate,
            do=dout,
            dv=dvalue,
            scale=128**-0.5,
            h0=torch.zeros(1, 12, 128, 128, device="cuda", dtype=torch.float32),
        )
