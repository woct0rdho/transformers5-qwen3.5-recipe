"""The W/U recomputation must equal FLA's kernel, which is the one the step runs today.

`gdn_wu_recompute.recompute_w_u` is a drop-in for `fla.ops.gated_delta_rule.wy_fast.recompute_w_u_fwd`:
same three inputs, same two outputs, called from inside FLA's own autograd function where the call is
not tracked by autograd. These tests run both kernels on the same tensors and compare, then check the
contract the caller depends on -- the head axis each output carries, the gate-free path, determinism,
and that a tiling that cannot be honoured fails closed instead of summing part of a chunk.
"""

import pytest
import torch
from fla.ops.gated_delta_rule.wy_fast import recompute_w_u_fwd

from gdn_wu_recompute import BLOCK_D, NUM_WARPS, recompute_w_u, report
from test_support import mixed_precision_metrics

# (batch, sequence, key heads, value heads, key dim, value dim, chunk)
SHAPES = (
    (1, 128, 2, 4, 32, 32, 64),
    (2, 64, 4, 4, 64, 64, 64),
    (1, 256, 16, 48, 128, 128, 64),
)
REFERENCE_RMSE = 1e-2


def _inputs(
    batch: int,
    sequence: int,
    key_heads: int,
    value_heads: int,
    key_dim: int,
    value_dim: int,
    chunk: int,
    seed: int = 0,
):
    generator = torch.Generator(device="cuda").manual_seed(seed)
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
    beta = torch.rand(
        batch,
        sequence,
        value_heads,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    # The solved inverse the solve kernel leaves: lower triangular with a unit diagonal.
    lower = (
        torch.randn(
            batch,
            sequence,
            value_heads,
            chunk,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.3
    )
    solved = torch.tril(lower, -1).clone()
    solved.diagonal(dim1=-2, dim2=-1).fill_(1.0)
    # `chunk_local_cumsum` accumulates inside each chunk, in log2 space.
    steps = (
        -torch.rand(batch, sequence, value_heads, device="cuda", generator=generator)
        * 0.05
    )
    decay = (
        steps.reshape(batch, -1, chunk, value_heads)
        .cumsum(2)
        .reshape(batch, sequence, value_heads)
    )
    return key, value, beta, solved.contiguous(), decay.to(torch.float32)


@pytest.mark.parametrize("shape", SHAPES)
def test_matches_the_fla_kernel(shape: tuple[int, ...]) -> None:
    key, value, beta, solved, decay = _inputs(*shape)
    reference = recompute_w_u_fwd(k=key, v=value, beta=beta, A=solved, g=decay)
    candidate = recompute_w_u(key, value, beta, solved, decay)

    for name, got, want in zip(("w", "u"), candidate, reference, strict=True):
        assert got.shape == want.shape
        assert got.dtype == want.dtype
        _, relative_rmse = mixed_precision_metrics(got, want)
        assert relative_rmse <= REFERENCE_RMSE, (name, relative_rmse)

    # `w` carries the value-head axis (one copy per value head, which is what the state pass reads) and
    # `u` is shaped like `v`. Getting this backwards walks off one of the buffers.
    assert candidate[0].shape[2] == shape[3]
    assert candidate[1].shape == value.shape


def test_matches_without_a_gate() -> None:
    shape = SHAPES[0]
    key, value, beta, solved, _ = _inputs(*shape)
    reference = recompute_w_u_fwd(k=key, v=value, beta=beta, A=solved, g=None)
    candidate = recompute_w_u(key, value, beta, solved, None)
    for got, want in zip(candidate, reference, strict=True):
        _, relative_rmse = mixed_precision_metrics(got, want)
        assert relative_rmse <= REFERENCE_RMSE


def test_determinism() -> None:
    shape = SHAPES[1]
    key, value, beta, solved, decay = _inputs(*shape)
    first = recompute_w_u(key, value, beta, solved, decay)
    second = recompute_w_u(key, value, beta, solved, decay)
    for left, right in zip(first, second, strict=True):
        assert torch.equal(left, right)


def test_rejects_shapes_it_cannot_tile() -> None:
    shape = SHAPES[0]
    key, value, beta, solved, decay = _inputs(*shape)
    with pytest.raises(ValueError, match="4-D"):
        recompute_w_u(key, value, beta, solved[:, :, :, 0], decay)
    with pytest.raises(ValueError, match="multiples of"):
        recompute_w_u(key, value, beta, solved, decay, block_d=3)
    # Same shape, different layout: a strided view the kernel would index as if it were contiguous.
    buffer = torch.zeros(
        *solved.shape[:-1],
        2 * solved.shape[-1],
        dtype=solved.dtype,
        device=solved.device,
    )
    buffer[..., ::2] = solved
    strided = buffer[..., ::2]
    assert strided.shape == solved.shape and not strided.is_contiguous()
    with pytest.raises(ValueError, match="contiguous"):
        recompute_w_u(key, value, beta, strided, decay)
    with pytest.raises(ValueError, match="power of two"):
        odd = torch.zeros_like(solved)[..., :48].contiguous()
        recompute_w_u(key, value, beta, odd, decay)
    with pytest.raises(ValueError, match="shortest axis"):
        tiny = torch.zeros_like(solved)[..., :8].contiguous()
        recompute_w_u(key, value, beta, tiny, decay)
    # Head counts that are not divisible: the value axis cannot be mapped onto the key axis.
    uneven_key, uneven_value, uneven_beta, uneven_solved, uneven_decay = _inputs(
        1, 64, 3, 4, 32, 32, 64
    )
    with pytest.raises(ValueError, match="value heads"):
        recompute_w_u(
            uneven_key, uneven_value, uneven_beta, uneven_solved, uneven_decay
        )


def test_report_describes_the_deployed_tiling() -> None:
    summary = report()
    assert summary["block_d"] == BLOCK_D
    assert summary["num_warps"] == NUM_WARPS
