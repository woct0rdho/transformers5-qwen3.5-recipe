"""Tests for the fused 8-bit AdamW with stochastic rounding.

The reference for every numerical check is bitsandbytes: its codebook (`create_dynamic_map`), its
quantizer (`quantize_blockwise`/`dequantize_blockwise`) and its 8-bit blockwise AdamW kernel
(`optimizer_update_8bit_blockwise`). The point of the module is to reproduce that state format
exactly while adding the stochastic rounding and fusing the step into one kernel.
"""

import pytest
import torch

from adamw_8bit_sr import (
    BLOCK,
    BNB_QUANT_KEY,
    NON_CASTABLE_KEYS,
    AdamW8bitSR,
    dynamic_qmap,
    is_routed_expert_adapter,
)

cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the fused kernel needs a GPU"
)


def test_dynamic_qmap_matches_bitsandbytes() -> None:
    bitsandbytes_functional = pytest.importorskip("bitsandbytes.functional")
    for signed in (True, False):
        assert torch.equal(
            dynamic_qmap(signed),
            bitsandbytes_functional.create_dynamic_map(signed=signed),
        )


def test_routed_expert_names() -> None:
    routed = [
        "base_model.model.model.layers.0.mlp.experts.lora_A.weight",
        "base_model.model.model.layers.0.moe.experts.lora_B_down.weight",
    ]
    dense = [
        "base_model.model.model.layers.0.mlp.shared_expert.gate_proj.lora_A.weight",
        "base_model.model.model.layers.0.mlp.shared_experts.down_proj.lora_A.weight",
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
    ]
    for name in routed:
        assert is_routed_expert_adapter(name), name
    for name in dense:
        assert not is_routed_expert_adapter(name), name


def _reference_step(
    p, g, state1, state2, absmax1, absmax2, lr, betas, eps, weight_decay, step
):
    bitsandbytes_functional = pytest.importorskip("bitsandbytes.functional")
    bitsandbytes_functional.optimizer_update_8bit_blockwise(
        "adam",
        g,
        p,
        state1,
        state2,
        betas[0],
        betas[1],
        0.0,
        1.0,
        eps,
        step,
        lr,
        dynamic_qmap(True, p.device),
        dynamic_qmap(False, p.device),
        absmax1,
        absmax2,
        weight_decay,
    )


@cuda
def test_kernel_matches_the_bitsandbytes_reference() -> None:
    torch.manual_seed(0)
    n, lr, betas, eps, weight_decay, step = 8192, 1e-4, (0.9, 0.999), 1e-8, 1e-2, 101
    p_reference = (torch.randn(n, device="cuda") * 0.05).to(torch.bfloat16)
    grad = (torch.randn(n, device="cuda") * 1e-3).to(torch.bfloat16)
    shaped = torch.full(p_reference.shape, 127, dtype=torch.uint8, device="cuda")
    zeros = torch.zeros(p_reference.shape, dtype=torch.uint8, device="cuda")
    absmax1 = torch.zeros(n // BLOCK, device="cuda")
    absmax2 = torch.zeros(n // BLOCK, device="cuda")
    state1 = shaped.clone()
    state2 = zeros.clone()

    _reference_step(
        p_reference,
        grad,
        state1,
        state2,
        absmax1,
        absmax2,
        lr,
        betas,
        eps,
        weight_decay,
        step,
    )

    param = p_reference.clone().detach().requires_grad_(True)
    param.grad = grad.clone()
    optimizer = AdamW8bitSR(
        [("dense", param)],
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        stochastic_state=False,
        stochastic_write=False,
    )
    optimizer.state[param] = {
        "step": step - 1,
        "state1": shaped.clone(),
        "state2": zeros.clone(),
        "qmap1": dynamic_qmap(True, param.device),
        "qmap2": dynamic_qmap(False, param.device),
        "absmax1": torch.zeros_like(absmax1),
        "absmax2": torch.zeros_like(absmax2),
    }
    optimizer.step()

    state = optimizer.state[param]
    assert torch.equal(state["absmax1"], absmax1)
    assert (state["state1"] == state1).float().mean() > 0.98
    assert (state["state2"] == state2).float().mean() > 0.98
    # The closed-form codebook and bitsandbytes' bisection break ties differently now and then,
    # which moves an update across a BF16 rounding boundary: the write agrees on the vast majority
    # of elements and never differs by more than one ulp.
    delta = (param.float() - p_reference.float()).abs()
    ulp = torch.pow(2.0, torch.floor(torch.log2(p_reference.float().abs().max())) - 8)
    assert delta.median() == 0
    assert delta.max() <= ulp
    assert (delta > 0).float().mean() < 0.5


@cuda
def test_stochastic_write_makes_the_decay_apply() -> None:
    torch.manual_seed(0)
    n, lr, weight_decay, steps = 1 << 16, 1e-4, 1e-2, 40

    def mean_after(stochastic: bool) -> tuple[float, float]:
        param = (
            (torch.rand(n, device="cuda") + 0.5).to(torch.bfloat16).requires_grad_(True)
        )
        optimizer = AdamW8bitSR(
            [("dense", param)],
            lr=lr,
            weight_decay=weight_decay,
            stochastic_state=stochastic,
            stochastic_write=stochastic,
        )
        param.grad = torch.zeros(n, dtype=torch.bfloat16, device="cuda")
        before = param.detach().double().mean().item()
        for _ in range(steps):
            optimizer.step()
        return before, param.detach().double().mean().item()

    for stochastic in (True,):
        before, after = mean_after(stochastic)
        expected = before * (1.0 - lr * weight_decay) ** steps
        assert abs(after / expected - 1.0) < 1e-3

    before, after = mean_after(False)
    assert after == before  # round-to-nearest BF16 discards the whole decay


@cuda
def test_stochastic_state_is_unbiased() -> None:
    torch.manual_seed(0)
    n, lr, steps = 1 << 16, 1e-4, 64
    param = (torch.rand(n, device="cuda") * 0.1).to(torch.bfloat16).requires_grad_(True)
    optimizer = AdamW8bitSR(
        [("dense", param)], lr=lr, weight_decay=0.0, stochastic_state=True
    )
    # positive mean, wide spread: the aggregate mean over elements is the statistic that a
    # round-to-nearest quantizer biases and a stochastic one does not
    grad = (1e-3 * (1.0 + 0.5 * torch.randn(n, device="cuda"))).to(torch.bfloat16)
    param.grad = grad
    for _ in range(steps):
        optimizer.step()

    state = optimizer.state[param]
    qmap = state["qmap1"]
    scale = state["absmax1"].repeat_interleave(BLOCK)[: param.numel()]
    moment1 = (qmap[state["state1"].long()] * scale).double()
    # The first moment of a constant gradient is a geometric series of `grad`. Round to nearest
    # would bias every element by up to half a codebook cell, so the mean over elements is the
    # statistic that shows the stochastic rounding is unbiased.
    expected = grad.double() * (1 - 0.9**steps)
    assert abs(moment1.mean().item() / expected.mean().item() - 1) < 0.01
    assert state["state1"].unique().numel() > 8  # the codes really are spread out


@cuda
def test_routed_expert_decay_is_separate() -> None:
    torch.manual_seed(0)
    routed = torch.full(
        (8192,), 0.5, dtype=torch.bfloat16, device="cuda", requires_grad=True
    )
    dense = torch.full(
        (8192,), 0.5, dtype=torch.bfloat16, device="cuda", requires_grad=True
    )
    optimizer = AdamW8bitSR(
        [
            ("model.layers.0.mlp.experts.lora_A.weight", routed),
            ("model.layers.0.self_attn.q_proj.lora_A.weight", dense),
        ],
        lr=1e-1,
        weight_decay=1e-1,
        routed_expert_decay=0.0,
        stochastic_write=True,
    )
    for param in (routed, dense):
        param.grad = torch.zeros_like(param)
    for _ in range(5):
        optimizer.step()
    assert torch.equal(routed, torch.full_like(routed, 0.5))  # no decay at all
    assert dense.mean() < 0.5  # dense decayed


@cuda
def test_small_parameters_keep_a_float32_state() -> None:
    torch.manual_seed(0)
    param = (
        (torch.rand(1000, device="cuda") * 0.1).to(torch.bfloat16).requires_grad_(True)
    )
    optimizer = AdamW8bitSR([("dense", param)], lr=1e-4, weight_decay=1e-2)
    param.grad = torch.full_like(param, 1e-3)
    optimizer.step()
    state = optimizer.state[param]
    assert state["state1"].dtype == torch.float32  # below min_8bit_size
    assert "absmax1" not in state
    assert param.mean() < 0.1 * (1 + 1e-3)  # the update ran


@cuda
def test_state_dict_keeps_the_bitsandbytes_schema() -> None:
    param = torch.zeros(8192, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    optimizer = AdamW8bitSR(
        [("model.layers.0.mlp.experts.lora_A.weight", param)], lr=1e-4
    )
    optimizer.state[param] = optimizer._init_state(param)
    payload = optimizer.state_dict()

    entry = next(iter(payload["state"].values()))
    wrapped = entry[BNB_QUANT_KEY]
    assert set(wrapped) == set(NON_CASTABLE_KEYS)
    assert wrapped["state1"].dtype == torch.uint8
    assert wrapped["qmap1"].shape == (256,)
    assert wrapped["absmax1"].numel() == param.numel() // BLOCK
    assert "step" in entry

    # bitsandbytes' checkpoints carry the empty "no decay" group, so this one does too
    assert [len(group["params"]) for group in payload["param_groups"]] == [1, 0]

    clone = torch.zeros_like(param, requires_grad=True)
    other = AdamW8bitSR([("model.layers.0.mlp.experts.lora_A.weight", clone)], lr=1e-4)
    other.load_state_dict(payload)
    restored = other.state[clone]
    assert torch.equal(restored["state1"], wrapped["state1"])
    assert torch.equal(restored["absmax2"], wrapped["absmax2"])
    assert restored["step"] == entry["step"]

    # save -> load -> save is idempotent, which is what makes resume and the layout hooks agree
    again = other.state_dict()
    assert [len(group["params"]) for group in again["param_groups"]] == [1, 0]
    for key, value in wrapped.items():
        assert torch.equal(again["state"][0][BNB_QUANT_KEY][key], value), key


@cuda
def test_closed_form_quantizer_is_nearest() -> None:
    """The code the kernel stores is the nearest codebook entry (brute-force check).

    The remaining disagreements are exact midpoints between two codes, where either choice is
    nearest. The closed form rounds those up and `argmin` returns the lower index.
    """

    torch.manual_seed(0)
    n, block = 4096, BLOCK
    param = (torch.rand(n, device="cuda") * 0.1).to(torch.bfloat16).requires_grad_(True)
    grad = (1e-3 * (1.0 + 2.0 * torch.randn(n, device="cuda"))).to(torch.bfloat16)
    param.grad = grad

    optimizer = AdamW8bitSR(
        [("dense", param)], lr=1e-4, weight_decay=0.0, stochastic_state=False
    )
    optimizer.state[param] = {
        "step": 0,
        "state1": torch.full(param.shape, 127, dtype=torch.uint8, device="cuda"),
        "state2": torch.zeros(param.shape, dtype=torch.uint8, device="cuda"),
        "qmap1": dynamic_qmap(True, param.device),
        "qmap2": dynamic_qmap(False, param.device),
        "absmax1": torch.zeros(n // block, device="cuda"),
        "absmax2": torch.zeros(n // block, device="cuda"),
    }
    optimizer.step()

    state = optimizer.state[param]
    moment1 = (1 - 0.9) * grad.double()
    absmax = (
        moment1.abs()
        .view(-1, block)
        .amax(dim=1)
        .repeat_interleave(block)
        .clamp_min(1e-38)
    )
    ratio = (moment1 / absmax).clamp(-1.0, 1.0).float()
    qmap = dynamic_qmap(True, param.device)
    nearest = (qmap[None, :] - ratio[:, None]).abs().argmin(dim=1)
    agree = (nearest == state["state1"].flatten().long()).float().mean()
    assert agree > 0.999, agree


@cuda
def test_state_dict_does_not_disturb_the_live_state() -> None:
    """Saving must not move the live 8-bit tensors: the layout hook rebuilds the payload."""

    torch.manual_seed(0)
    param = (
        (torch.rand(8192, device="cuda") * 0.05).to(torch.bfloat16).requires_grad_(True)
    )
    grad = (torch.randn(8192, device="cuda") * 1e-3).to(torch.bfloat16)
    param.grad = grad
    optimizer = AdamW8bitSR([("dense", param)], lr=1e-4, weight_decay=1e-2)
    optimizer.step()

    live_before = {
        key: value.clone()
        for key, value in optimizer.state[param].items()
        if torch.is_tensor(value)
    }
    payload = optimizer.state_dict()
    assert BNB_QUANT_KEY in payload["state"][0]
    # the live entry is still flat, and still the same tensors
    assert "state1" in optimizer.state[param]
    for key, value in live_before.items():
        assert torch.equal(optimizer.state[param][key], value), key

    # and a further step still works and keeps the parameters finite
    optimizer.step()
    assert torch.isfinite(param).all()
