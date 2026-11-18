"""AdamW with an 8-bit blockwise state, one fused Triton kernel, stochastic rounding.

`optimizer.pt` keeps bitsandbytes' layout: per parameter `{"step", state1, qmap1, state2, qmap2,
absmax1, absmax2}` with `state1`/`state2` uint8 codes over 256-element blocks, `absmax*` float32
per block and `qmap*` the 256-entry codebook, wrapped under `__bnb_optimizer_quant_state__` when
saved (see `state_dict`). The checkpoints, `gdn_stock_checkpoints` and
`convert_checkpoint_to_stock_layout.py` therefore keep working unchanged.

Two things differ from bitsandbytes:
- the codebook is evaluated in closed form, so the kernel needs no per-element table search.
  bitsandbytes' "dynamic" map is 7 decades `10**(i-6)`, each covering `[0.1, 1]` times its scale
  with `2**i` (signed) or `2**(i+1)` (unsigned) uniform cells whose reconstruction points are the
  cell means, plus the entries 0.0 and 1.0. Both directions are a handful of arithmetic ops
  (`dequantize`, `quantize`), which is why this fits in plain Triton: no shared-memory gather, no
  HIP extension, no inline asm.
- `stochastic_state` chooses the code stochastically (probability equal to the discarded part) and
  `stochastic_write` rounds the parameter write stochastically. Without them a 1e-6 relative decay,
  or any Adam step below half a BF16 ulp, is discarded by the store: at step 32950 of
  Qwen3.8-Flash-Next, 76 % of the adapter elements and 55 % of the update mass were sub-half-ulp.

The kernel follows bitsandbytes' arithmetic exactly (`correction1 = 1 - beta1**step`,
`correction2 = sqrt(1 - beta2**step)`, `p += -lr*correction2/correction1 * m/(sqrt(v) + correction2*eps)`,
then `p *= 1 - lr*weight_decay`), so a run with both stochastic flags off reproduces the reference
optimizer and only differs where the closed-form codebook breaks a tie differently.

Weight decay can be split by family: `routed_expert_decay` applies to the parameters whose names
match `is_routed_expert_adapter`, the rest use `weight_decay`. A routed expert sees
`top_k/experts` of the rows a dense projection sees, so the same decay is a much stronger prior
there (`torch-ggml-ops/docs/expert_distribution_prior.md`).
"""

import copy
import math
from collections.abc import Iterable

import torch
import triton
import triton.language as tl

BLOCK = 256  # bitsandbytes' absmax block size, and one program per block below
LOG2_10 = 3.321928094887362
BNB_QUANT_KEY = "__bnb_optimizer_quant_state__"
NON_CASTABLE_KEYS = ("state1", "qmap1", "state2", "qmap2", "absmax1", "absmax2")
ROUTED_EXPERT_ADAPTER_TOKEN = ".experts."


def is_routed_expert_adapter(name: str) -> bool:
    """Whether a parameter belongs to the packed routed-expert adapter of an MoE block."""

    return ROUTED_EXPERT_ADAPTER_TOKEN in name and "shared_expert" not in name


def dynamic_qmap(signed: bool, device: torch.device | None = None) -> torch.Tensor:
    """bitsandbytes' `create_dynamic_map`, evaluated arithmetically (bit-identical)."""

    total_bits, max_exponent_bits = 8, 7
    non_sign_bits = total_bits - 1
    values: list[float] = []
    for i in range(max_exponent_bits):
        if signed:
            fraction_items = 2 ** (i + non_sign_bits - max_exponent_bits) + 1
        else:
            fraction_items = 2 ** (i + non_sign_bits - max_exponent_bits + 1) + 1
        edges = torch.linspace(0.1, 1.0, fraction_items, dtype=torch.float32)
        means = (edges[:-1] + edges[1:]) / 2.0
        scale = 10.0 ** (-(max_exponent_bits - 1) + i)
        values += (scale * means).tolist()
        if signed:
            values += (-scale * means).tolist()
    values += [0.0, 1.0]
    values.sort()
    assert len(values) == 1 << total_bits
    return torch.tensor(values, dtype=torch.float32, device=device)


@triton.jit
def _pow10(exponent):
    return tl.exp2(exponent.to(tl.float32) * 3.321928094887362)


@triton.jit
def _decade_scale(decade):
    """10**(decade - 6) for decade in 0..6, by selection (no exp)."""

    s = 1e-6
    s = tl.where(decade >= 1, 1e-5, s)
    s = tl.where(decade >= 2, 1e-4, s)
    s = tl.where(decade >= 3, 1e-3, s)
    s = tl.where(decade >= 4, 1e-2, s)
    s = tl.where(decade >= 5, 1e-1, s)
    s = tl.where(decade >= 6, 1.0, s)
    return s


@triton.jit
def _inverse_decade_scale(decade):
    """1 / 10**(decade - 6), by the same selection."""

    s = 1e6
    s = tl.where(decade >= 1, 1e5, s)
    s = tl.where(decade >= 2, 1e4, s)
    s = tl.where(decade >= 3, 1e3, s)
    s = tl.where(decade >= 4, 1e2, s)
    s = tl.where(decade >= 5, 1e1, s)
    s = tl.where(decade >= 6, 1.0, s)
    return s


@triton.jit
def _decade_lower_edge(decade, signed: tl.constexpr):
    """Ratio below which the previous decade's last cell is the nearest code.

    The cells of decade `i` start at `10**(i-6) * 0.1` and the last cell of decade `i-1` ends at
    `10**(i-7)`, so the two grids are not contiguous: the switch happens at the midpoint of the
    two adjacent reconstruction points, not at the decade boundary.
    """

    if signed:
        edge = 0.0
        edge = tl.where(decade >= 1, 1.9e-6, edge)
        edge = tl.where(decade >= 2, 1.45e-5, edge)
        edge = tl.where(decade >= 3, 1.225e-4, edge)
        edge = tl.where(decade >= 4, 1.1125e-3, edge)
        edge = tl.where(decade >= 5, 1.05625e-2, edge)
        edge = tl.where(decade >= 6, 0.1028125, edge)
    else:
        edge = 0.0
        edge = tl.where(decade >= 1, 1.45e-6, edge)
        edge = tl.where(decade >= 2, 1.253e-5, edge)
        edge = tl.where(decade >= 3, 1.1266e-4, edge)
        edge = tl.where(decade >= 4, 1.0633e-3, edge)
        edge = tl.where(decade >= 5, 1.0316e-2, edge)
        edge = tl.where(decade >= 6, 0.10158, edge)
    return edge


@triton.jit
def _dequantize(codes, absmax, signed: tl.constexpr):
    """uint8 code -> float32 value, from the closed-form codebook."""

    code = codes.to(tl.int32)
    if signed:
        index = tl.where(code >= 128, code - 128, 126 - code)
        # cumulative cells per decade are 2**i - 1: 0, 1, 3, 7, 15, 31, 63
        decade = (
            (index >= 1).to(tl.int32)
            + (index >= 3).to(tl.int32)
            + (index >= 7).to(tl.int32)
            + (index >= 15).to(tl.int32)
            + (index >= 31).to(tl.int32)
            + (index >= 63).to(tl.int32)
        )
        before = (1 << decade) - 1
        cells = (1 << decade).to(tl.float32)
    else:
        index = code - 1
        decade = (
            (index >= 2).to(tl.int32)
            + (index >= 6).to(tl.int32)
            + (index >= 14).to(tl.int32)
            + (index >= 30).to(tl.int32)
            + (index >= 62).to(tl.int32)
            + (index >= 126).to(tl.int32)
        )
        before = (2 << decade) - 2
        cells = (2 << decade).to(tl.float32)
    within = tl.minimum(tl.maximum(index - before, 0), (cells - 1).to(tl.int32))
    t = 0.1 + (within.to(tl.float32) + 0.5) * (0.9 / cells)
    value = t * _decade_scale(decade)
    if signed:
        value = tl.where(code >= 128, value, -value)
        value = tl.where(code == 127, 0.0, value)
    else:
        value = tl.where(code == 0, 0.0, value)
    value = tl.where(code == 255, 1.0, value)
    return absmax * value


@triton.jit
def _quantize(values, absmax, uniform, signed: tl.constexpr, stochastic: tl.constexpr):
    """float32 value -> nearest uint8 code, or a stochastic choice when `stochastic`."""

    ratio = tl.minimum(tl.abs(values) / tl.maximum(absmax, 1e-38), 1.0)
    # decades are [1e-7, 1e-6), [1e-6, 1e-5), ... [0.1, 1]. Six comparisons, no logarithm
    decade = (
        (ratio >= 1e-6).to(tl.int32)
        + (ratio >= 1e-5).to(tl.int32)
        + (ratio >= 1e-4).to(tl.int32)
        + (ratio >= 1e-3).to(tl.int32)
        + (ratio >= 1e-2).to(tl.int32)
        + (ratio >= 1e-1).to(tl.int32)
    )
    decade = decade - (ratio < _decade_lower_edge(decade, signed)).to(tl.int32)
    if signed:
        cells = (1 << decade).to(tl.float32)
        before = (1 << decade) - 1
    else:
        cells = (2 << decade).to(tl.float32)
        before = (2 << decade) - 2
    t = ratio * _inverse_decade_scale(decade)
    position = tl.maximum((t - 0.1) / (0.9 / cells) - 0.5, 0.0)
    if stochastic:
        position = tl.floor(position + uniform)
    else:
        position = tl.floor(position + 0.5)
    position = tl.minimum(position, cells - 1.0)
    index = tl.minimum(before + position.to(tl.int32), 127 if signed else 254)
    if signed:
        code = tl.where(values >= 0, 128 + index, 126 - index)
        # the table's 0.0 and 1.0 entries are not on the uniform grid. They win near the extremes
        code = tl.where(ratio >= 0.99648, tl.where(values >= 0, 255, code), code)
        code = tl.where(ratio < 2.75e-7, 127, code)
    else:
        code = index + 1
        code = tl.where(ratio >= 0.99648, 255, code)
        code = tl.where(ratio < 1.625e-7, 0, code)
    return code.to(tl.uint8)


@triton.jit
def _stochastic_bits(value, uniform):
    """Round float32 `value` onto its 16-bit grid with probability equal to the discarded part."""

    bits = value.to(tl.int32, bitcast=True) + uniform
    return (bits & -65536).to(tl.float32, bitcast=True)


@triton.jit
def _adamw_kernel(
    p_ptr,
    g_ptr,
    s1_ptr,
    s2_ptr,
    am1_ptr,
    am2_ptr,
    n_elements,
    lr,
    beta1,
    beta2,
    correction1,
    correction2,
    eps,
    weight_decay,
    seed,
    BLOCK_SIZE: tl.constexpr,
    SR_STATE: tl.constexpr,
    SR_WRITE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    grad = tl.load(g_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    param = tl.load(p_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    absmax1 = tl.load(am1_ptr + pid).to(tl.float32)
    absmax2 = tl.load(am2_ptr + pid).to(tl.float32)
    moment1 = _dequantize(
        tl.load(s1_ptr + offsets, mask=mask, other=127), absmax1, True
    )
    moment2 = _dequantize(tl.load(s2_ptr + offsets, mask=mask, other=0), absmax2, False)

    # bitsandbytes drops non-finite gradients by zeroing both moments
    # NaN and inf both fail this comparison, so one test covers both
    finite = tl.abs(grad) <= 3.4028234663852886e38
    moment1 = beta1 * moment1 + (1.0 - beta1) * grad
    moment2 = beta2 * moment2 + (1.0 - beta2) * grad * grad
    moment1 = tl.where(finite, moment1, 0.0)
    moment2 = tl.where(finite, moment2, 0.0)

    new_absmax1 = tl.maximum(
        tl.max(tl.where(mask, tl.abs(moment1), 0.0), axis=0), 1e-38
    )
    new_absmax2 = tl.maximum(
        tl.max(tl.where(mask, tl.abs(moment2), 0.0), axis=0), 1e-38
    )

    # one counter-based word per element, sliced into the three uniforms this step needs
    word = (tl.rand(seed, offsets) * 4294967296.0).to(tl.int32)
    uniform1 = (word & 255).to(tl.float32) * (1.0 / 256.0)
    uniform2 = ((word >> 8) & 255).to(tl.float32) * (1.0 / 256.0)
    uniform3 = ((word >> 16) & 65535).to(tl.float32) * (1.0 / 65536.0)
    code1 = _quantize(moment1, new_absmax1, uniform1, True, SR_STATE)
    code2 = _quantize(moment2, new_absmax2, uniform2, False, SR_STATE)

    update = (
        (-lr * correction2 / correction1)
        * moment1
        / (tl.sqrt(moment2) + correction2 * eps)
    )
    new_param = param + update
    new_param = new_param * (1.0 - lr * weight_decay)
    if SR_WRITE:
        new_param = _stochastic_bits(new_param, (uniform3 * 65536.0).to(tl.int32))

    tl.store(s1_ptr + offsets, code1, mask=mask)
    tl.store(s2_ptr + offsets, code2, mask=mask)
    tl.store(am1_ptr + pid, new_absmax1)
    tl.store(am2_ptr + pid, new_absmax2)
    tl.store(p_ptr + offsets, new_param.to(p_ptr.dtype.element_ty), mask=mask)


def _stochastic_round_to(values: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Torch reference: round `values` onto `dtype`'s grid with stochastic rounding."""

    bits = values.float().view(torch.int32) + torch.randint(
        0, 1 << 16, values.shape, dtype=torch.int32, device=values.device
    )
    mantissa_bits = 16 if dtype == torch.bfloat16 else 13
    mask = -1 << mantissa_bits
    return (bits & mask).view(torch.float32).to(dtype)


class AdamW8bitSR(torch.optim.Optimizer):
    """8-bit blockwise AdamW in one Triton kernel, with stochastic rounding.

    `named_parameters` is `model.named_parameters()`: the names decide which parameters take
    `routed_expert_decay` instead of `weight_decay`. Parameters below `min_8bit_size` keep a
    float32 state, exactly like bitsandbytes, and are stepped in torch.
    """

    def __init__(
        self,
        named_parameters: Iterable[tuple[str, torch.Tensor]],
        *,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        routed_expert_decay: float | None = None,
        min_8bit_size: int = 4096,
        stochastic_state: bool = True,
        stochastic_write: bool = True,
        seed: int = 0,
    ):
        parameters = list(named_parameters)
        selected = [p for _, p in parameters if p.requires_grad]
        if not selected:
            raise ValueError("AdamW8bitSR needs at least one trainable parameter")
        # bitsandbytes' checkpoints carry a second, empty "no decay" group (`[N, 0]`), so the
        # optimizer keeps one too: `optimizer.pt` then has the same `param_groups` shape.
        super().__init__(
            [{"params": selected}, {"params": []}],
            defaults={
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
            },
        )
        self.routed_expert_decay = routed_expert_decay
        self.routed_ids = {
            id(p)
            for name, p in parameters
            if p.requires_grad and is_routed_expert_adapter(name)
        }
        self.min_8bit_size = min_8bit_size
        self.stochastic_state = stochastic_state
        self.stochastic_write = stochastic_write
        self.seed = seed
        self._qmap = dynamic_qmap(True)
        self._qmap_unsigned = dynamic_qmap(False)
        self._tick = 0

    def _init_state(self, param: torch.Tensor) -> dict:
        state = self.state[param]
        state["step"] = 0
        # CPU parameters (tests, smoke runs) take the float32 path: the fused kernel is GPU only.
        if param.numel() >= self.min_8bit_size and param.device.type == "cuda":
            blocks = (param.numel() + BLOCK - 1) // BLOCK
            device = param.device
            state["state1"] = torch.full_like(param, 127, dtype=torch.uint8)
            state["state2"] = torch.zeros_like(param, dtype=torch.uint8)
            state["qmap1"] = self._qmap.to(device)
            state["qmap2"] = self._qmap_unsigned.to(device)
            state["absmax1"] = torch.zeros(blocks, dtype=torch.float32, device=device)
            state["absmax2"] = torch.zeros(blocks, dtype=torch.float32, device=device)
        else:
            state["state1"] = torch.zeros_like(param, dtype=torch.float32)
            state["state2"] = torch.zeros_like(param, dtype=torch.float32)
        return state

    def _decay(self, param: torch.Tensor, default: float) -> float:
        if self.routed_expert_decay is not None and id(param) in self.routed_ids:
            return self.routed_expert_decay
        return default

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = self.state[param] or self._init_state(param)
                state["step"] += 1
                step = state["step"]
                correction1 = 1.0 - beta1**step
                correction2 = math.sqrt(1.0 - beta2**step)
                weight_decay = self._decay(param, group["weight_decay"])
                grad = param.grad
                if grad.dtype != param.dtype:
                    grad = grad.to(param.dtype)
                grad = grad.contiguous()
                param.data = param.data.contiguous()
                self._tick += 1
                seed = (self.seed + self._tick) % (2**31 - 1)

                if "absmax1" not in state:
                    self._step_torch(
                        param,
                        grad,
                        state,
                        lr,
                        beta1,
                        beta2,
                        correction1,
                        correction2,
                        eps,
                        weight_decay,
                    )
                    continue

                blocks = state["absmax1"].numel()
                _adamw_kernel[(blocks,)](
                    param,
                    grad,
                    state["state1"],
                    state["state2"],
                    state["absmax1"],
                    state["absmax2"],
                    param.numel(),
                    lr,
                    beta1,
                    beta2,
                    correction1,
                    correction2,
                    eps,
                    weight_decay,
                    seed,
                    BLOCK_SIZE=BLOCK,
                    SR_STATE=self.stochastic_state,
                    SR_WRITE=self.stochastic_write,
                )
        return loss

    def _step_torch(
        self,
        param,
        grad,
        state,
        lr,
        beta1,
        beta2,
        correction1,
        correction2,
        eps,
        weight_decay,
    ) -> None:
        """float32 state for parameters below `min_8bit_size`, same arithmetic."""

        moment1 = state["state1"]
        moment2 = state["state2"]
        g = grad.float()
        finite = torch.isfinite(g)
        moment1.mul_(beta1).add_(
            torch.where(finite, g, torch.zeros_like(g)), alpha=1 - beta1
        )
        moment2.mul_(beta2).addcmul_(g, g, value=1 - beta2)
        moment1.mul_(finite)
        moment2.mul_(finite)
        update = (
            (-lr * correction2 / correction1)
            * moment1
            / (moment2.sqrt() + correction2 * eps)
        )
        new_param = (param.float() + update) * (1.0 - lr * weight_decay)
        if self.stochastic_write:
            param.copy_(_stochastic_round_to(new_param, param.dtype))
        else:
            param.copy_(new_param.to(param.dtype))

    def state_dict(self) -> dict:
        payload = super().state_dict()
        # `Optimizer.state_dict` shares the per-parameter entry dicts with the live state, and the
        # layout hook reassigns keys inside them while saving. Copy each entry first, exactly as
        # bitsandbytes does, so saving cannot move the live 8-bit tensors into the other layout.
        wrapped_state = {}
        for index, entry in payload["state"].items():
            if not isinstance(entry, dict):
                wrapped_state[index] = entry
                continue
            entry = dict(entry)
            wrapped = {key: entry.pop(key) for key in NON_CASTABLE_KEYS if key in entry}
            if wrapped:
                entry[BNB_QUANT_KEY] = wrapped
            wrapped_state[index] = entry
        payload["state"] = wrapped_state
        return payload

    def load_state_dict(self, state_dict: dict, *args, **kwargs) -> None:
        payload = copy.deepcopy(state_dict)
        for entry in payload.get("state", {}).values():
            if isinstance(entry, dict) and BNB_QUANT_KEY in entry:
                entry.update(entry.pop(BNB_QUANT_KEY))

        groups = self.param_groups
        saved_groups = payload["param_groups"]
        sizes = [len(g["params"]) for g in groups if g["params"]]
        saved_sizes = [len(g["params"]) for g in saved_groups if g["params"]]
        if sizes != saved_sizes:
            raise ValueError(
                "loaded state dict has a different parameter group layout: "
                f"{[len(g['params']) for g in saved_groups]} against "
                f"{[len(g['params']) for g in groups]}"
            )

        flat = [p for group in groups for p in group["params"]]
        restored: dict[int, dict] = {}
        for index, entry in payload["state"].items():
            if not isinstance(index, int):
                continue
            param = flat[index]
            restored[index] = {
                key: (value.to(param.device) if torch.is_tensor(value) else value)
                for key, value in entry.items()
            }

        for index, entry in restored.items():
            param = flat[index]
            state = self.state[param]
            state.clear()
            state.update(entry)
        # The live hyperparameters (decay, betas) stay in charge: the checkpoint's values are what
        # the previous run used, and silently restoring them is how a new decay value gets lost.
        for group in groups:
            group.setdefault("lr", self.defaults["lr"])
