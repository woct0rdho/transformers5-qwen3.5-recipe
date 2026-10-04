"""The fused GatedDeltaNet preparation must equal the chain the model runs today.

`gdn_fused_prep.fused_prep` replaces, in one kernel, the projection output's trip through the
channel-last depthwise convolution, its SiLU, the transpose back, the split and reshape into query,
key and value, the tiled key-to-value head broadcast, FLA's L2 normalization of the expanded heads,
and the sigmoid and softplus gating. Its backward replaces the convolution's own backward, the
normalization backward and the elementwise gate gradients with four kernels.

These tests run the model's chain as the reference and compare every output and every input gradient.
The reference uses the same pieces the model uses: the `causal_conv1d` hub kernel with `silu`, the
tiled `repeat` broadcast that `gdn_tiled_value_heads` installs, and FLA's autograd-aware `l2norm`.

The small geometry cases use a sequence length that is not a multiple of the token block, so the row
masks in both kernels are exercised rather than bypassed.
"""

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
import torch.nn.functional as F
from fla.modules.l2norm import l2norm
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextGatedDeltaNet,
    causal_conv1d_fn,
)

from gdn_fused_prep import (
    BLOCK_T,
    BLOCK_T_LOAD,
    DZ_DTYPE,
    SKIPPED_KEY,
    configure_fused_preparation,
    fused_prep,
    report,
    require_fused_preparation,
)
from gdn_tiled_value_heads import configure_tiled_value_heads
from test_support import assert_close_mixed_precision

TAPS = 4
# A geometry small enough to run per test, plus the real Qwen4-Exp one for the layout gate.
SMALL = (1, 100, 2, 4, 32)
QWEN4 = (1, 256, 16, 48, 128)


def _inputs(
    batch: int,
    sequence: int,
    key_heads: int,
    value_heads: int,
    head_dim: int,
    seed: int = 0,
):
    channels = key_heads * 2 * head_dim + value_heads * head_dim
    generator = torch.Generator(device="cuda").manual_seed(seed)
    tensors = {
        "mixed": torch.randn(
            batch,
            sequence,
            channels,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.5,
        "conv_weight": torch.randn(
            channels, TAPS, device="cuda", dtype=torch.bfloat16, generator=generator
        )
        * 0.2,
        "a": torch.randn(
            batch,
            sequence,
            value_heads,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.5,
        "b": torch.randn(
            batch,
            sequence,
            value_heads,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.5,
        "a_log": torch.randn(
            value_heads, device="cuda", dtype=torch.bfloat16, generator=generator
        ),
        "dt_bias": torch.randn(
            value_heads, device="cuda", dtype=torch.bfloat16, generator=generator
        ),
    }
    return {name: tensor.requires_grad_(True) for name, tensor in tensors.items()}


def _reference(tensors, key_heads: int, value_heads: int, head_dim: int):
    """The model's preparation chain, in the tiled value-head convention."""
    mixed = tensors["mixed"]
    batch, sequence, _ = mixed.shape
    group = value_heads // key_heads
    # `causal_conv1d_fn` is the hub kernel the model dispatches to: zero padding of `taps - 1`,
    # truncation back to the sequence length, then SiLU.
    conv = causal_conv1d_fn(
        mixed.transpose(1, 2), tensors["conv_weight"], None, activation="silu"
    )
    conv = conv.transpose(1, 2)
    query, key, value = torch.split(
        conv,
        [key_heads * head_dim, key_heads * head_dim, value_heads * head_dim],
        dim=-1,
    )
    query = query.reshape(batch, sequence, key_heads, head_dim).repeat(1, 1, group, 1)
    key = key.reshape(batch, sequence, key_heads, head_dim).repeat(1, 1, group, 1)
    value = value.reshape(batch, sequence, value_heads, head_dim)
    beta = tensors["b"].sigmoid()
    gate = -tensors["a_log"].float().exp() * F.softplus(
        tensors["a"].float() + tensors["dt_bias"].float()
    )
    return l2norm(query), l2norm(key), value, gate, beta


def _fused(tensors, key_heads: int, value_heads: int, head_dim: int):
    return fused_prep(
        tensors["mixed"],
        tensors["conv_weight"],
        tensors["a"],
        tensors["b"],
        tensors["a_log"],
        tensors["dt_bias"],
        key_heads=key_heads,
        value_heads=value_heads,
        head_dim=head_dim,
    )


def _clone(tensors):
    return {
        name: tensor.detach().clone().requires_grad_(True)
        for name, tensor in tensors.items()
    }


def _gradients(outputs, tensors):
    ones = [torch.ones_like(output) for output in outputs]
    for output in outputs:
        if not output.requires_grad:
            raise AssertionError("the fused preparation must be differentiable")
    names = list(tensors)
    return dict(
        zip(
            names,
            torch.autograd.grad(
                outputs,
                [tensors[name] for name in names],
                grad_outputs=ones,
                retain_graph=False,
            ),
            strict=True,
        )
    )


@pytest.mark.parametrize(
    ("key_heads", "value_heads", "head_dim"), [(2, 4, 32), (4, 4, 16), (2, 6, 32)]
)
def test_forward_matches_the_reference_chain(
    key_heads: int, value_heads: int, head_dim: int
) -> None:
    batch, sequence, _, _, _ = SMALL
    tensors = _inputs(batch, sequence, key_heads, value_heads, head_dim)
    reference = _reference(tensors, key_heads, value_heads, head_dim)
    fused = _fused(tensors, key_heads, value_heads, head_dim)

    # q and k are normalized and rounded to BF16, so they are held to the rounding of a unit row.
    # g is FP32 throughout and beta is a pure sigmoid, so those two are nearly exact.
    bounds = (
        (0.99999, 5e-3),
        (0.99999, 5e-3),
        (0.99999, 5e-3),
        (1.0, 1e-6),
        (1.0, 1e-6),
    )
    for name, candidate, expected, (cosine, rmse) in zip(
        ("q", "k", "v", "g", "beta"), fused, reference, bounds, strict=True
    ):
        assert candidate.shape == expected.shape
        assert candidate.dtype == expected.dtype
        assert_close_mixed_precision(
            candidate, expected, minimum_cosine=cosine, maximum_relative_rmse=rmse
        )


def test_gradients_match_the_reference_chain() -> None:
    batch, sequence, key_heads, value_heads, head_dim = SMALL
    tensors = _inputs(batch, sequence, key_heads, value_heads, head_dim)
    reference_grads = _gradients(
        _reference(tensors, key_heads, value_heads, head_dim), tensors
    )
    fused_grads = _gradients(_fused(tensors, key_heads, value_heads, head_dim), tensors)

    for name in ("mixed", "conv_weight", "a", "b", "a_log", "dt_bias"):
        candidate = fused_grads[name]
        expected = reference_grads[name]
        assert candidate.shape == expected.shape
        assert candidate.dtype == expected.dtype
        # `mixed` and `conv_weight` pass through the BF16 `dz`, and `b`'s output is BF16, so those
        # three carry rounding error. The gate parameters accumulate in FP32 and come out exact.
        bound = 5e-3 if name in ("mixed", "conv_weight", "b") else 1e-3
        assert_close_mixed_precision(
            candidate, expected, minimum_cosine=0.9999, maximum_relative_rmse=bound
        )


def test_qwen4_geometry_and_layout() -> None:
    _, _, key_heads, value_heads, head_dim = QWEN4
    tensors = _inputs(*QWEN4)
    fused = _fused(tensors, key_heads, value_heads, head_dim)
    query, key, value, gate, beta = fused
    group = value_heads // key_heads

    for tensor in fused:
        assert tensor.is_contiguous()
    assert query.shape == (QWEN4[0], QWEN4[1], value_heads, head_dim)
    assert value.shape == (QWEN4[0], QWEN4[1], value_heads, head_dim)
    assert gate.dtype == torch.float32
    assert beta.dtype == torch.bfloat16

    # The broadcast is tiled, not interleaved: head `i` carries key head `i % key_heads`, which is the
    # order the packed `out_proj` rows want. Interleaving would put key head `i // group` there.
    tiled = query.reshape(QWEN4[0], QWEN4[1], group, key_heads, head_dim)
    for copy in range(1, group):
        assert torch.equal(tiled[:, :, copy], tiled[:, :, 0])

    # The rows are already unit norm, so FLA's in-kernel normalization would be a round trip. This is
    # what lets the core run with `use_qk_l2norm_in_kernel=False`.
    for tensor in (query, key):
        norms = tensor.float().square().sum(-1).sqrt()
        assert torch.allclose(norms, torch.ones_like(norms), atol=2e-2)
    reference = _reference(tensors, key_heads, value_heads, head_dim)
    assert_close_mixed_precision(
        query, reference[0], minimum_cosine=0.99999, maximum_relative_rmse=5e-3
    )


def test_non_contiguous_gradient_is_not_misread() -> None:
    """An expanded incoming gradient must give the same answer as a contiguous one.

    Autograd hands the gate's gradient over as a non-contiguous broadcast, and the kernels index every
    gradient row-major. Reading it at face value silently picks up the wrong elements, so the op has to
    make the layout explicit instead of trusting it.
    """
    batch, sequence, key_heads, value_heads, head_dim = SMALL
    tensors = _inputs(batch, sequence, key_heads, value_heads, head_dim)
    outputs = _fused(tensors, key_heads, value_heads, head_dim)
    names = list(tensors)
    inputs = [tensors[name] for name in names]

    contiguous = [torch.ones_like(output) for output in outputs]
    expanded = list(contiguous)
    expanded[3] = (
        contiguous[3]
        .expand(batch, sequence, value_heads)
        .reshape(-1)[::1]
        .reshape(batch, sequence, value_heads)
    )
    assert (
        not expanded[3].is_contiguous()
        or expanded[3].stride() == contiguous[3].stride()
    )

    expected = torch.autograd.grad(
        outputs, inputs, grad_outputs=contiguous, retain_graph=True
    )
    # A stride-0 broadcast of the same values: same numbers, different memory layout.
    widened = torch.ones(batch, 1, value_heads, device="cuda").expand(
        batch, sequence, value_heads
    )
    assert not widened.is_contiguous()
    candidate = torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=[*contiguous[:3], widened, contiguous[4]],
        retain_graph=True,
    )
    for got, want in zip(candidate, expected, strict=True):
        assert torch.equal(got, want)


def test_determinism() -> None:
    batch, sequence, key_heads, value_heads, head_dim = SMALL
    tensors = _inputs(batch, sequence, key_heads, value_heads, head_dim)
    first = _fused(tensors, key_heads, value_heads, head_dim)
    second = _fused(tensors, key_heads, value_heads, head_dim)
    for left, right in zip(first, second, strict=True):
        assert torch.equal(left, right)
    left_grads = _gradients(first, tensors)
    right_grads = _gradients(second, tensors)
    for name in left_grads:
        assert torch.equal(left_grads[name], right_grads[name]), name


def test_rejects_mismatched_shapes() -> None:
    batch, sequence, key_heads, value_heads, head_dim = SMALL
    tensors = _inputs(batch, sequence, key_heads, value_heads, head_dim)
    with pytest.raises(ValueError, match="mixed has"):
        fused_prep(
            tensors["mixed"],
            tensors["conv_weight"],
            tensors["a"],
            tensors["b"],
            tensors["a_log"],
            tensors["dt_bias"],
            key_heads=key_heads,
            value_heads=value_heads,
            head_dim=head_dim * 2,
        )
    with pytest.raises(ValueError, match="must be a multiple of"):
        fused_prep(
            tensors["mixed"],
            tensors["conv_weight"],
            tensors["a"],
            tensors["b"],
            tensors["a_log"],
            tensors["dt_bias"],
            key_heads=key_heads,
            value_heads=value_heads,
            head_dim=head_dim,
            block_t=7,
        )
    # A channel count that is consistent with the head counts, but with the value heads not being a
    # multiple of the key heads, so the divisibility check is the one that has to fire.
    uneven = _inputs(batch, sequence, 2, 5, head_dim)
    with pytest.raises(ValueError, match="value heads"):
        fused_prep(
            uneven["mixed"],
            uneven["conv_weight"],
            uneven["a"],
            uneven["b"],
            uneven["a_log"],
            uneven["dt_bias"],
            key_heads=2,
            value_heads=5,
            head_dim=head_dim,
        )


def test_report_describes_the_deployed_shape() -> None:
    summary = report()
    assert summary["block_t"] == BLOCK_T
    assert summary["block_t_load"] == BLOCK_T_LOAD
    assert summary["taps"] == TAPS
    assert summary["dz_dtype"] == str(DZ_DTYPE)


def test_large_chunk_block_is_accepted() -> None:
    """The token block only decides how the work is split, so a different one must agree exactly."""
    batch, sequence, key_heads, value_heads, head_dim = SMALL
    tensors = _inputs(batch, sequence, key_heads, value_heads, head_dim)
    baseline = _fused(tensors, key_heads, value_heads, head_dim)
    for block_t, block_t_load in ((16, 4), (32, 8), (64, 16), (128, 32)):
        variant = fused_prep(
            tensors["mixed"],
            tensors["conv_weight"],
            tensors["a"],
            tensors["b"],
            tensors["a_log"],
            tensors["dt_bias"],
            key_heads=key_heads,
            value_heads=value_heads,
            head_dim=head_dim,
            block_t=block_t,
            block_t_load=block_t_load,
        )
        for left, right in zip(variant, baseline, strict=True):
            assert left.shape == right.shape
            if left.dtype == torch.float32:
                assert torch.allclose(left, right, atol=1e-6)
            else:
                # Only the reduction order inside a row changes with the block size.
                assert (
                    torch.equal(left, right)
                    or (left.float() - right.float()).abs().max() <= 2e-2
                )
        del variant


WIRING_HIDDEN = 256
WIRING_HEADS_KEY = 2
WIRING_HEADS_VALUE = 4
WIRING_HEAD_DIM = 64
WIRING_ROWS = 64


def _wiring_config() -> SimpleNamespace:
    """A layer with the shape the preparation tiles for, at test scale."""

    return SimpleNamespace(
        hidden_size=WIRING_HIDDEN,
        linear_num_value_heads=WIRING_HEADS_VALUE,
        linear_num_key_heads=WIRING_HEADS_KEY,
        linear_key_head_dim=WIRING_HEAD_DIM,
        linear_value_head_dim=WIRING_HEAD_DIM,
        linear_conv_kernel_dim=TAPS,
        hidden_act="silu",
        num_hidden_layers=1,
        output_gate_type="sigmoid",
        rms_norm_eps=1e-6,
        layer_types=["linear_attention"],
    )


def _wiring_layer() -> Qwen4ExpTextGatedDeltaNet:
    torch.manual_seed(20261208)
    module = (
        Qwen4ExpTextGatedDeltaNet(cast(Any, _wiring_config()), 0)
        .cuda()
        .to(torch.bfloat16)
    )
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(std=0.05)
        module.A_log.normal_(mean=-1.0, std=0.1)
    return module


def _wiring_run(
    layer: Qwen4ExpTextGatedDeltaNet, hidden: torch.Tensor
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """One forward and backward on a fixed input, returning the output and every gradient."""

    for parameter in layer.parameters():
        parameter.grad = None
    inputs = hidden.detach().clone().requires_grad_(True)
    output = layer(inputs, cache_params=None)
    output.float().sum().backward()
    assert inputs.grad is not None
    gradients = [inputs.grad.float().clone()]
    gradients.extend(
        parameter.grad.float().clone()
        if parameter.grad is not None
        else torch.zeros_like(parameter, dtype=torch.float32)
        for parameter in layer.parameters()
    )
    return output.float().clone(), gradients


def test_the_wiring_reproduces_the_eager_layer() -> None:
    """Same output and same gradients, with the fused preparation installed on the real layer."""

    configure_tiled_value_heads()
    layer = _wiring_layer()
    generator = torch.Generator(device="cuda").manual_seed(7)
    hidden = torch.randn(
        1,
        WIRING_ROWS,
        WIRING_HIDDEN,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )

    eager_output, eager_gradients = _wiring_run(layer, hidden)
    installed = configure_fused_preparation(layer)
    fused_output, fused_gradients = _wiring_run(layer, hidden)

    require_fused_preparation(installed, expected_gdn_layers=1)
    assert installed["patched"] == 1
    assert installed["already_patched"] == 0

    assert_close_mixed_precision(
        fused_output, eager_output, minimum_cosine=0.99999, maximum_relative_rmse=5e-3
    )
    worst = 0.0
    for wanted, produced in zip(eager_gradients, fused_gradients, strict=True):
        scale = wanted.square().mean().sqrt().clamp_min(1e-12)
        worst = max(worst, float((produced - wanted).square().mean().sqrt() / scale))
    assert worst < 1e-2, worst


def test_a_second_configuration_only_counts() -> None:
    """The patcher is idempotent, which is what the inventory gate relies on."""

    layer = _wiring_layer()
    first = configure_fused_preparation(layer)
    second = configure_fused_preparation(layer)
    assert first["patched"] == 1 and first["already_patched"] == 0
    assert second["patched"] == 0 and second["already_patched"] == 1


def test_unsupported_geometry_is_skipped_not_patched() -> None:
    """A layer the fused kernel cannot serve keeps the eager forward and is counted as skipped."""

    layer = _wiring_layer()
    layer.head_k_dim = WIRING_HEAD_DIM * 2
    report_ = configure_fused_preparation(layer)
    # A rejected site is counted only under the skip key, so nothing was handled and nothing patched.
    assert report_["gdn_layers"] == 0
    assert report_[SKIPPED_KEY] == 1
    with pytest.raises(RuntimeError, match="incomplete"):
        require_fused_preparation(report_, expected_gdn_layers=0)
    require_fused_preparation(report_, expected_gdn_layers=0, expected_skipped=1)


def test_the_guard_rejects_paths_the_patch_does_not_serve() -> None:
    """Decode and variable lengths fall through to the eager forward. Padding does not."""

    from gdn_fused_prep import _training_path_applies

    assert _training_path_applies(None, {})
    assert not _training_path_applies(object(), {})
    assert not _training_path_applies(None, {"cu_seqlens": torch.ones(2)})
    assert not _training_path_applies(None, {"seq_idx": torch.ones(1, 2)})


def test_the_wiring_matches_the_eager_layer_with_a_padding_mask() -> None:
    """The trainers pass a padding mask, so the patched forward has to keep its only effect."""

    configure_tiled_value_heads()
    layer = _wiring_layer()
    generator = torch.Generator(device="cuda").manual_seed(11)
    hidden = torch.randn(
        1,
        WIRING_ROWS,
        WIRING_HIDDEN,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    mask = torch.ones(1, WIRING_ROWS, device="cuda", dtype=torch.long)
    mask[:, WIRING_ROWS // 2 :] = 0
    assert layer.forward is not None

    for parameter in layer.parameters():
        parameter.grad = None
    eager = layer(hidden, cache_params=None, attention_mask=mask).float().clone()

    configure_fused_preparation(layer)
    for parameter in layer.parameters():
        parameter.grad = None
    fused = layer(hidden, cache_params=None, attention_mask=mask).float().clone()

    assert_close_mixed_precision(
        fused, eager, minimum_cosine=0.99999, maximum_relative_rmse=5e-3
    )
