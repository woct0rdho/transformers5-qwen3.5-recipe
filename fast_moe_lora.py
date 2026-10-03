"""Fast persistent-GGUF MoE LoRA with packed MMQ and AITER.

The backend is shared by Qwen3.5-MoE and DeepSeek V4. Model-specific behavior
stays in the owning experts class (`_prepare_expert_hidden_states` and
`_apply_split_gate`) and in each model's wrapper/registration module.

Packed expert forward projections run directly through grouped gfx1151 MMQ.
Gate and up share one dynamic Q8_1 activation workspace. Frozen base input
gradients decode active packed experts directly into BF16 WMMA fragments.
Gate and up accumulate into one FP32 route-gradient accumulator. Rank-small
LoRA branches retain AITER `gmm` and factor gradients retain AITER `ptgmm`.
The gate/up factor rebuilds its routed rows from the routing index in backward
instead of retaining the gathered activation.
No logical base matrix, full expert LoRA delta, or effective expert-weight
gradient is constructed.

PEFT targets each complete `GgufExperts` module rather than its packed
physical parameters. One wrapper owns the combined gate/up and down factors,
which keeps the combined gate/up LoRA semantics while avoiding nested
parameter wrappers and transient state on the expert module.
"""

import math
from dataclasses import dataclass
from typing import Any, cast

import torch
from aiter.ops.triton.gmm import gmm, ptgmm
from peft import LoraConfig
from peft.tuners.lora.layer import LoraLayer
from torch_ggml_ops import grouped_mmq, grouped_mmq_pair
from transformers.integrations.gguf.gguf_quantized_parameter import (
    GgufQuantizedParameter,
)
from transformers.integrations.gguf.moe import ALL_GGUF_EXPERTS_FUNCTIONS, GgufExperts

from fast_moe_routing import finalize_expert_routing, prepare_expert_routing
from moe_gmm_configs import gmm_config as _gmm_config
from moe_gmm_configs import ptgmm_config as _ptgmm_config

QWEN3_5_MOE_EXPERTS_IMPLEMENTATION = "qwen3_5_moe_gguf_mmq_aiter_lora"
_LORA_WEIGHTS_KWARG = "_gguf_moe_lora_weights"
_GGUF_EXPERTS_TYPE = cast(type[Any], GgufExperts)


def _aiter_forward(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    group_sizes: torch.Tensor,
    *,
    expert_prior: str,
) -> torch.Tensor:
    return gmm(
        lhs,
        rhs,
        group_sizes,
        preferred_element_type=lhs.dtype,
        config=_gmm_config(
            lhs.shape[0],
            lhs.shape[1],
            rhs.shape[-1],
            rhs.stride(1) == 1,
            expert_prior=expert_prior,
        ),
    )


def _aiter_input_grad(
    grad_output: torch.Tensor,
    rhs: torch.Tensor,
    group_sizes: torch.Tensor,
    *,
    expert_prior: str,
) -> torch.Tensor:
    input_rhs = rhs.transpose(1, 2)
    return gmm(
        grad_output,
        input_rhs,
        group_sizes,
        preferred_element_type=grad_output.dtype,
        config=_gmm_config(
            grad_output.shape[0],
            grad_output.shape[1],
            input_rhs.shape[-1],
            input_rhs.stride(1) == 1,
            expert_prior=expert_prior,
        ),
    )


def _aiter_weight_grad(
    lhs: torch.Tensor,
    grad_output: torch.Tensor,
    group_sizes: torch.Tensor,
    *,
    expert_prior: str,
) -> torch.Tensor:
    return ptgmm(
        lhs.T,
        grad_output,
        group_sizes,
        preferred_element_type=lhs.dtype,
        config=_ptgmm_config(
            lhs.shape[0],
            lhs.shape[1],
            grad_output.shape[1],
            expert_prior=expert_prior,
        ),
    )


class _AiterGroupedMM(torch.autograd.Function):
    """Autograd-capable `(M,K) @ (E,K,N)` grouped matrix multiplication.

    `lhs` may be rebuilt instead of retained. When `lhs_source` and
    `lhs_permutation` are supplied they satisfy
    `lhs == lhs_source[lhs_permutation // lhs_top_k]`, so the source rows plus
    the index are saved and the gather is replayed in backward. Routing repeats
    every token `top_k` times, which makes the gathered activation several
    times larger than the token rows it copies. The replay is bitwise identical.
    """

    @staticmethod
    def forward(
        ctx,
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        group_sizes: torch.Tensor,
        expert_prior: str,
        lhs_source: torch.Tensor | None,
        lhs_permutation: torch.Tensor | None,
        lhs_top_k: int,
    ) -> torch.Tensor:
        if lhs.ndim != 2 or rhs.ndim != 3 or group_sizes.ndim != 1:
            raise ValueError(
                "AITER grouped MM expects lhs [M,K], rhs [E,K,N], and group_sizes [E]."
            )
        if lhs.shape[1] != rhs.shape[1]:
            raise ValueError(
                f"Grouped-MM K mismatch: lhs has {lhs.shape[1]}, rhs has {rhs.shape[1]}."
            )
        if rhs.shape[0] != group_sizes.numel():
            raise ValueError(
                f"Grouped-MM expert mismatch: rhs has {rhs.shape[0]}, group_sizes has {group_sizes.numel()}."
            )
        if lhs.dtype not in (torch.float16, torch.bfloat16) or rhs.dtype != lhs.dtype:
            raise TypeError(
                f"AITER grouped MM requires matching FP16/BF16 inputs, got {lhs.dtype} and {rhs.dtype}."
            )

        if lhs.stride() != (lhs.shape[1], 1):
            raise ValueError("AITER grouped MM lhs must be row-major.")
        if group_sizes.device != lhs.device:
            raise ValueError("AITER grouped MM group_sizes must share the lhs device.")
        if group_sizes.dtype != torch.int32 or group_sizes.stride() != (1,):
            raise ValueError("AITER grouped MM group_sizes must be contiguous int32.")

        rebuild_lhs = lhs_source is not None
        if rebuild_lhs:
            if lhs_permutation is None:
                raise ValueError(
                    "Grouped-MM lhs rebuild requires the routing permutation."
                )
            if lhs_top_k < 1:
                raise ValueError(
                    f"Grouped-MM lhs rebuild requires top_k >= 1, got {lhs_top_k}."
                )
            if (
                lhs_source.ndim != 2
                or lhs_source.dtype != lhs.dtype
                or lhs_source.shape[1] != lhs.shape[1]
                or lhs_source.stride() != (lhs_source.shape[1], 1)
            ):
                raise ValueError(
                    "Grouped-MM lhs rebuild requires a row-major source whose features "
                    "match lhs."
                )
            if (
                lhs_permutation.ndim != 1
                or lhs_permutation.numel() != lhs.shape[0]
                or lhs_permutation.dtype not in (torch.int32, torch.int64)
            ):
                raise ValueError(
                    "Grouped-MM lhs rebuild requires one row index per lhs row."
                )

        ctx.expert_prior = expert_prior
        ctx.rebuild_lhs = rebuild_lhs
        ctx.lhs_top_k = lhs_top_k
        ctx.save_for_backward(
            rhs,
            group_sizes,
            lhs_source if rebuild_lhs else lhs,
            lhs_permutation,
        )
        return _aiter_forward(lhs, rhs, group_sizes, expert_prior=ctx.expert_prior)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # ty: ignore[invalid-method-override]
        rhs, group_sizes, lhs_or_source, lhs_permutation = ctx.saved_tensors
        if grad_output.stride() != (grad_output.shape[1], 1):
            raise ValueError("AITER grouped MM output gradient must be row-major.")
        grad_lhs = (
            _aiter_input_grad(
                grad_output, rhs, group_sizes, expert_prior=ctx.expert_prior
            )
            if ctx.needs_input_grad[0]
            else None
        )
        grad_rhs = None
        if ctx.needs_input_grad[1]:
            if ctx.rebuild_lhs:
                if lhs_permutation is None:
                    raise RuntimeError(
                        "Grouped-MM lhs rebuild lost its routing permutation."
                    )
                # `detach` keeps the replay a value-level substitute for a saved
                # tensor. It must not add a second path to the source gradient.
                lhs = lhs_or_source.detach()[lhs_permutation // ctx.lhs_top_k]
            else:
                lhs = lhs_or_source
            grad_rhs = _aiter_weight_grad(
                lhs,
                grad_output,
                group_sizes,
                expert_prior=ctx.expert_prior,
            )
        return grad_lhs, grad_rhs, None, None, None, None, None


def aiter_grouped_mm(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    group_sizes: torch.Tensor,
    *,
    expert_prior: str,
    lhs_source: torch.Tensor | None = None,
    lhs_permutation: torch.Tensor | None = None,
    lhs_top_k: int = 1,
) -> torch.Tensor:
    """Apply the autograd-capable AITER grouped matrix multiplication.

    `lhs_source`/`lhs_permutation`/`lhs_top_k` optionally describe `lhs`
    as a routed row gather, so backward replays the gather instead of holding it.
    """

    return _AiterGroupedMM.apply(
        lhs,
        rhs,
        group_sizes,
        expert_prior,
        lhs_source,
        lhs_permutation,
        lhs_top_k,
    )


def _native_grouped_arguments(
    hidden_states: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_offsets: torch.Tensor,
    compute_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if compute_dtype != torch.bfloat16:
        raise RuntimeError("Grouped GGUF MMQ requires BF16 compute dtype.")
    hidden_states = hidden_states.to(compute_dtype).contiguous()
    expert_indices = expert_indices.to(
        device=hidden_states.device, dtype=torch.int64
    ).contiguous()
    expert_offsets = expert_offsets.to(
        device=hidden_states.device, dtype=torch.int32
    ).contiguous()
    return hidden_states, expert_indices, expert_offsets


def _base_grouped_linear(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_offsets: torch.Tensor,
    group_sizes: torch.Tensor,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    del group_sizes
    if not isinstance(weight, GgufQuantizedParameter):
        raise TypeError(
            "Fast MoE base projections require packed GGUF weights and the "
            "exported torch_ggml_ops grouped_mmq API."
        )
    if weight.requires_grad:
        raise RuntimeError(
            "Fast GGUF expert execution requires frozen packed base weights."
        )
    hidden_states, expert_indices, expert_offsets = _native_grouped_arguments(
        hidden_states, expert_indices, expert_offsets, compute_dtype
    )
    quant_type = int(cast(Any, weight.quant_type))
    logical_shape = cast(tuple[int, ...], weight.logical_shape)
    return grouped_mmq(
        hidden_states,
        weight.as_subclass(torch.Tensor),
        expert_indices,
        expert_offsets,
        quant_type,
        int(logical_shape[-2]),
    )


def _base_grouped_pair(
    hidden_states: torch.Tensor,
    first_weight: torch.Tensor,
    second_weight: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_offsets: torch.Tensor,
    group_sizes: torch.Tensor,
    compute_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(first_weight, GgufQuantizedParameter) or not isinstance(
        second_weight, GgufQuantizedParameter
    ):
        raise TypeError("Fast MoE paired base projections require packed GGUF weights.")
    if (
        first_weight.quant_type == second_weight.quant_type
        and first_weight.logical_shape == second_weight.logical_shape
    ):
        if first_weight.requires_grad or second_weight.requires_grad:
            raise RuntimeError(
                "Fast GGUF expert execution requires frozen packed base weights."
            )
        hidden_states, expert_indices, expert_offsets = _native_grouped_arguments(
            hidden_states, expert_indices, expert_offsets, compute_dtype
        )
        quant_type = int(cast(Any, first_weight.quant_type))
        logical_shape = cast(tuple[int, ...], first_weight.logical_shape)
        return grouped_mmq_pair(
            hidden_states,
            first_weight.as_subclass(torch.Tensor),
            second_weight.as_subclass(torch.Tensor),
            expert_indices,
            expert_offsets,
            quant_type,
            int(logical_shape[-2]),
        )
    return (
        _base_grouped_linear(
            hidden_states,
            first_weight,
            expert_indices,
            expert_offsets,
            group_sizes,
            compute_dtype,
        ),
        _base_grouped_linear(
            hidden_states,
            second_weight,
            expert_indices,
            expert_offsets,
            group_sizes,
            compute_dtype,
        ),
    )


class _ExpertFactor(torch.nn.Module):
    """One PEFT-visible expert factor with a conventional logical weight layout."""

    def __init__(
        self, shape: tuple[int, ...], *, device: torch.device, dtype: torch.dtype
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(shape, device=device, dtype=dtype))


@dataclass(frozen=True)
class _ExpertLoraWeights:
    gate_up_a: torch.Tensor  # [experts, rank, hidden]
    gate_up_b: torch.Tensor  # [experts, 2 * intermediate, rank]
    down_a: torch.Tensor  # [experts, rank, intermediate]
    down_b: torch.Tensor  # [experts, hidden, rank]
    scaling: float
    expert_prior: str


def _prepare_packed_expert_execution(
    routing_plan: Any,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return expert IDs, int32 offsets, and int32 group sizes for every expert.

    `routing_plan.expert_indices` holds one sorted expert id per routed row, so
    the per-expert row counts are a scatter-add into a fixed-size tensor and the
    offsets a prefix sum over it. Covering every expert keeps all three tensors a
    fixed length, which is what lets the routed row counts stay on the device:
    `torch.unique_consecutive` sizes its output on the host and synchronized once
    per MoE layer. Emptied groups are inert in the grouped kernels, which accept
    `num_groups <= num_experts` and drop any group whose row range is empty.
    """

    device = routing_plan.expert_indices.device
    expert_ids = torch.arange(num_experts, device=device, dtype=torch.int64)
    group_sizes = torch.zeros(num_experts, device=device, dtype=torch.int32)
    group_sizes.index_add_(
        0,
        routing_plan.expert_indices,
        torch.ones_like(routing_plan.expert_indices, dtype=torch.int32),
    )
    # cumsum promotes integral inputs to int64 unless told otherwise, and the
    # grouped kernels require int32 offsets.
    return expert_ids, group_sizes.cumsum(0, dtype=torch.int32), group_sizes


class FastGgufMoeLora(torch.nn.Module, LoraLayer):
    """PEFT LoRA wrapper owning all factors for one packed `GgufExperts` module."""

    adapter_layer_names = ("lora_A", "lora_B", "lora_A_down", "lora_B_down")

    def _get_in_out_features(self, module: torch.nn.Module) -> tuple[int, int]:
        module = module.get_base_layer() if isinstance(module, LoraLayer) else module
        if not isinstance(module, _GGUF_EXPERTS_TYPE):
            raise TypeError(
                f"Fast GGUF MoE LoRA requires GgufExperts, got {type(module).__name__}."
            )
        module = cast(Any, module)
        return int(module.hidden_dim), 2 * int(module.intermediate_dim)

    def __init__(
        self,
        base_layer: torch.nn.Module,
        adapter_name: str,
        *,
        config: LoraConfig,
        r: int,
        lora_alpha: int,
        **kwargs: Any,
    ) -> None:
        ephemeral_gpu_offload = bool(kwargs.pop("ephemeral_gpu_offload", False))
        super().__init__()
        LoraLayer.__init__(
            self, base_layer, ephemeral_gpu_offload=ephemeral_gpu_offload, **kwargs
        )
        experts = self.get_base_layer()
        expert_prior = experts.__dict__.get("_aiter_expert_prior")
        if not isinstance(expert_prior, str):
            raise TypeError(
                "Fast GGUF MoE LoRA requires a prior bound to its expert module."
            )
        self._expert_prior = expert_prior
        self.lora_A_down = torch.nn.ModuleDict()
        self.lora_B_down = torch.nn.ModuleDict()
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, lora_alpha, config=config, **kwargs)

    def update_layer(
        self,
        adapter_name: str,
        r: int,
        lora_alpha: int,
        config: LoraConfig,
        **kwargs: Any,
    ) -> None:
        del kwargs
        if r <= 0:
            raise ValueError(f"LoRA rank must be positive, got {r}.")
        if config.lora_dropout != 0.0:
            raise ValueError("Fast GGUF MoE LoRA currently requires lora_dropout=0.")
        if config.lora_bias:
            raise ValueError("Fast GGUF MoE LoRA does not support LoRA bias.")
        if config.use_dora or config.alora_invocation_tokens is not None:
            raise ValueError(
                "Fast GGUF MoE LoRA does not support DoRA or aLoRA variants."
            )

        experts = cast(Any, self.get_base_layer())
        if not isinstance(experts, _GGUF_EXPERTS_TYPE):
            raise TypeError(
                f"Fast GGUF MoE LoRA requires GgufExperts, got {type(experts).__name__}."
            )
        device = cast(torch.device, experts.gate_proj.device)
        dtype = cast(torch.dtype, experts.compute_dtype)
        e = int(experts.num_experts)
        h = int(experts.hidden_dim)
        i = int(experts.intermediate_dim)

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        self.scaling[adapter_name] = lora_alpha / (
            math.sqrt(r) if config.use_rslora else r
        )
        self.use_rslora[adapter_name] = config.use_rslora
        self.use_dora[adapter_name] = False
        self.lora_bias[adapter_name] = False
        self.lora_dropout[adapter_name] = torch.nn.Identity()
        self.lora_A[adapter_name] = _ExpertFactor((e, r, h), device=device, dtype=dtype)
        self.lora_B[adapter_name] = _ExpertFactor(
            (e, 2 * i, r), device=device, dtype=dtype
        )
        self.lora_A_down[adapter_name] = _ExpertFactor(
            (e, r, i), device=device, dtype=dtype
        )
        self.lora_B_down[adapter_name] = _ExpertFactor(
            (e, h, r), device=device, dtype=dtype
        )

        lora_a = cast(Any, self.lora_A[adapter_name])
        lora_b = cast(Any, self.lora_B[adapter_name])
        lora_a_down = cast(Any, self.lora_A_down[adapter_name])
        lora_b_down = cast(Any, self.lora_B_down[adapter_name])
        init = config.init_lora_weights
        if init is True:
            torch.nn.init.kaiming_uniform_(lora_a.weight, a=math.sqrt(5))
            torch.nn.init.kaiming_uniform_(lora_a_down.weight, a=math.sqrt(5))
            torch.nn.init.zeros_(lora_b.weight)
            torch.nn.init.zeros_(lora_b_down.weight)
        elif init == "gaussian":
            torch.nn.init.normal_(lora_a.weight, std=1 / r)
            torch.nn.init.normal_(lora_a_down.weight, std=1 / r)
            torch.nn.init.zeros_(lora_b.weight)
            torch.nn.init.zeros_(lora_b_down.weight)
        elif init is not False:
            raise ValueError(
                f"Fast GGUF MoE LoRA does not support init_lora_weights={init!r}."
            )

        self.set_adapter(self.active_adapters, inference_mode=config.inference_mode)

    def merge(
        self, safe_merge: bool = False, adapter_names: list[str] | None = None
    ) -> None:
        raise RuntimeError(
            "GGUF expert LoRA adapters cannot be merged into packed base weights."
        )

    def unmerge(self) -> None:
        raise RuntimeError(
            "GGUF expert LoRA adapters cannot be unmerged because merging is unsupported."
        )

    def get_delta_weight(
        self, adapter_name: str, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        raise RuntimeError(
            "GGUF expert LoRA does not materialize a full expert delta weight."
        )

    def _active_lora_weights(self, adapter_names: Any) -> _ExpertLoraWeights | None:
        if self.disable_adapters:
            return None
        if adapter_names is not None:
            raise RuntimeError(
                "Fast GGUF MoE LoRA does not support mixed-adapter batches."
            )
        if len(self.active_adapters) != 1:
            raise RuntimeError(
                "Fast GGUF MoE LoRA requires exactly one active adapter."
            )
        adapter_name = self.active_adapters[0]
        if adapter_name not in self.lora_A:
            return None
        lora_a = cast(Any, self.lora_A[adapter_name])
        lora_b = cast(Any, self.lora_B[adapter_name])
        lora_a_down = cast(Any, self.lora_A_down[adapter_name])
        lora_b_down = cast(Any, self.lora_B_down[adapter_name])
        return _ExpertLoraWeights(
            gate_up_a=lora_a.weight,
            gate_up_b=lora_b.weight,
            down_a=lora_a_down.weight,
            down_b=lora_b_down.weight,
            scaling=float(self.scaling[adapter_name]),
            expert_prior=self._expert_prior,
        )

    def forward(
        self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        adapter_names = kwargs.pop("adapter_names", None)
        lora_weights = self._active_lora_weights(adapter_names)
        experts = cast(Any, self.get_base_layer())
        if experts.config._experts_implementation != QWEN3_5_MOE_EXPERTS_IMPLEMENTATION:
            raise RuntimeError(
                f"Fast GGUF MoE LoRA requires experts_implementation="
                f"{QWEN3_5_MOE_EXPERTS_IMPLEMENTATION!r}, "
                f"got {experts.config._experts_implementation!r}."
            )
        kwargs[_LORA_WEIGHTS_KWARG] = lora_weights
        return self.base_layer(hidden_states, *args, **kwargs)


FastMoeParamWrapper = FastGgufMoeLora


def _lora_grouped_linear(
    hidden_states: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    group_sizes: torch.Tensor,
    *,
    expert_prior: str,
    gather_source: torch.Tensor | None = None,
    gather_permutation: torch.Tensor | None = None,
    gather_top_k: int = 1,
) -> torch.Tensor:
    """Run both rank-small factors over the routed rows.

    `gather_source`/`gather_permutation`/`gather_top_k` describe
    `hidden_states` as routed rows of the token activations, which lets the
    first factor rebuild them in backward instead of retaining the gather.
    """

    rank_states = aiter_grouped_mm(
        hidden_states,
        lora_a.transpose(1, 2),
        group_sizes,
        expert_prior=expert_prior,
        lhs_source=gather_source,
        lhs_permutation=gather_permutation,
        lhs_top_k=gather_top_k,
    )
    return aiter_grouped_mm(
        rank_states,
        lora_b.transpose(1, 2),
        group_sizes,
        expert_prior=expert_prior,
    )


def gguf_mmq_aiter_lora_forward(
    self: Any,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    _gguf_moe_lora_weights: _ExpertLoraWeights | None = None,
) -> torch.Tensor:
    """Persistent packed GGUF base projections with separated AITER LoRA GEMMs."""

    if not isinstance(self, _GGUF_EXPERTS_TYPE):
        raise TypeError(
            f"Packed GGUF MoE LoRA requires GgufExperts, got {type(self).__name__}."
        )
    if self.projection_layout != "split_gate_up" or self.has_bias or not self.has_gate:
        raise RuntimeError(
            "The GGUF AITER LoRA path requires split, bias-free gate/up expert projections."
        )

    compute_hidden_states = self._prepare_expert_hidden_states(hidden_states)
    routing_plan = prepare_expert_routing(
        compute_hidden_states,
        top_k_index,
        top_k_weights,
    )
    expert_indices, expert_offsets, group_sizes = _prepare_packed_expert_execution(
        routing_plan, self.num_experts
    )

    selected_hidden_states = routing_plan.selected_hidden_states
    expert_indices = expert_indices.to(
        device=selected_hidden_states.device, dtype=torch.int64
    ).contiguous()
    expert_offsets = expert_offsets.to(
        device=selected_hidden_states.device, dtype=torch.int32
    ).contiguous()

    gate, up = _base_grouped_pair(
        selected_hidden_states,
        self.gate_proj,
        self.up_proj,
        expert_indices,
        expert_offsets,
        group_sizes,
        self.compute_dtype,
    )

    lora_weights = _gguf_moe_lora_weights
    if lora_weights is not None:
        # `group_sizes` already covers every expert, so the grouped MM skips the
        # empty groups without selecting or gathering expert factors.
        gate_up_delta = _lora_grouped_linear(
            selected_hidden_states,
            lora_weights.gate_up_a,
            lora_weights.gate_up_b,
            group_sizes,
            expert_prior=lora_weights.expert_prior,
            # These rows are the token activations repeated top_k times, so keep
            # the token rows and the routing index and replay the gather in
            # backward instead of retaining a routed-size activation.
            gather_source=compute_hidden_states,
            gather_permutation=routing_plan.permutation,
            gather_top_k=routing_plan.num_top_k,
        )
        gate_delta, up_delta = gate_up_delta.chunk(2, dim=-1)
        gate.add_(gate_delta, alpha=lora_weights.scaling)
        up.add_(up_delta, alpha=lora_weights.scaling)

    intermediate = self._apply_split_gate(gate, up)
    output = _base_grouped_linear(
        intermediate,
        self.down_proj,
        expert_indices,
        expert_offsets,
        group_sizes,
        self.compute_dtype,
    )
    if lora_weights is not None:
        down_delta = _lora_grouped_linear(
            intermediate,
            lora_weights.down_a,
            lora_weights.down_b,
            group_sizes,
            expert_prior=lora_weights.expert_prior,
        )
        output.add_(down_delta, alpha=lora_weights.scaling)

    return finalize_expert_routing(output, hidden_states, routing_plan, None)


def register_fast_moe_lora(
    lora_config: LoraConfig,
    model: torch.nn.Module,
    *,
    expert_prior: str,
) -> LoraConfig:
    """Register the GGUF backend and bind its routed expert prior."""

    if expert_prior != "qwen-learned":
        raise ValueError("Qwen expert registration requires prior='qwen-learned'.")
    register = getattr(lora_config, "_register_custom_module", None)
    if register is None:
        raise RuntimeError(
            "This PEFT version has no LoraConfig._register_custom_module API. "
            "Cannot install fast GGUF MoE LoRA without a global PEFT monkey patch."
        )
    if lora_config.target_parameters:
        raise ValueError(
            "Persistent GGUF expert LoRA targets the complete 'experts' module, not target_parameters."
        )
    if isinstance(lora_config.target_modules, str):
        raise TypeError(
            "Fast GGUF MoE LoRA requires an explicit target_modules collection, not a regex string."
        )

    target_modules = set(lora_config.target_modules or ())
    target_modules.add("experts")
    lora_config.__dict__["target_modules"] = target_modules

    ALL_GGUF_EXPERTS_FUNCTIONS[QWEN3_5_MOE_EXPERTS_IMPLEMENTATION] = (
        gguf_mmq_aiter_lora_forward
    )
    cast(Any, model).set_experts_implementation(QWEN3_5_MOE_EXPERTS_IMPLEMENTATION)
    for module in model.modules():
        if isinstance(module, _GGUF_EXPERTS_TYPE):
            module.__dict__["_aiter_expert_prior"] = expert_prior
    register({GgufExperts: FastGgufMoeLora})
    return lora_config
