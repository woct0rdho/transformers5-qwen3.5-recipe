"""Fast persistent-GGUF Qwen3.5-MoE LoRA using packed MMQ and AITER.

Packed expert forward projections run directly through grouped gfx1151 MMQ.
Gate and up share one dynamic Q8_1 activation workspace. Frozen base input
gradients decode active packed experts directly into BF16 WMMA fragments.
Gate and up accumulate into one FP32 route-gradient accumulator. Rank-small
LoRA branches retain AITER ``gmm`` and factor gradients retain AITER ``ptgmm``.
No logical base matrix, full expert LoRA delta, or effective expert-weight
gradient is constructed.

PEFT targets each complete ``GGUFExperts`` module rather than its packed
physical parameters. One wrapper owns the combined gate/up and down factors,
which keeps the Qwen3.5 shared-A gate/up LoRA semantics while avoiding nested
parameter wrappers and transient state on the expert module.
"""

import math
from dataclasses import dataclass
from typing import Any

import torch
from aiter.ops.triton.gmm import gmm, ptgmm
from peft import LoraConfig
from peft.tuners.lora.layer import LoraLayer
from torch_ggml_ops import grouped_mmq, grouped_mmq_pair
from transformers.integrations.gguf import ALL_GGUF_EXPERTS_FUNCTIONS, GGUFExperts
from transformers.integrations.gguf_dequant import (
    GGUFQuantizedTensor,
    dequantize_gguf_tensor,
)

from fast_moe_routing import finalize_expert_routing, prepare_expert_routing

EXPERTS_IMPLEMENTATION = "qwen3_5_moe_gguf_mmq_aiter_lora"
_LORA_WEIGHTS_KWARG = "_qwen3_5_moe_gguf_lora_weights"


def _gmm_config(k: int, n: int) -> dict[str, int]:
    """Return the current provisional gfx1151 AITER GMM configuration."""

    if n <= 16:
        return {
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_K": 128,
            "BLOCK_SIZE_N": 16 if k >= 2048 else 32,
            "GROUP_SIZE": 1,
            "GRID_DIM": 256,
            "num_warps": 4,
            "num_stages": 1,
        }
    if k <= 16:
        return {
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_K": 32 if n >= 2048 else 16,
            "BLOCK_SIZE_N": 256 if n >= 2048 else 64,
            "GROUP_SIZE": 1,
            "GRID_DIM": 256,
            "num_warps": 4,
            "num_stages": 1,
        }
    if k >= 2048 and n <= 512:
        return {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_K": 64,
            "BLOCK_SIZE_N": 128,
            "GROUP_SIZE": 1,
            "GRID_DIM": 256,
            "num_warps": 8,
            "num_stages": 1,
        }
    return {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_K": 64,
        "BLOCK_SIZE_N": 128,
        "GROUP_SIZE": 1,
        "GRID_DIM": 256,
        "num_warps": 8,
        "num_stages": 1,
    }


def _ptgmm_config(k: int, n: int) -> dict[str, int]:
    """Return the current provisional gfx1151 AITER PTGMM configuration."""

    if n <= 16:
        return {
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_K": 256 if k >= 1024 else 128,
            "BLOCK_SIZE_N": 16,
            "GROUP_SIZE": 1,
            "GRID_DIM": 256,
            "num_warps": 4 if k >= 1024 else 8,
            "num_stages": 1,
        }
    if k <= 16:
        return {
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_K": 16,
            "BLOCK_SIZE_N": 256 if n >= 2048 else 128,
            "GROUP_SIZE": 1,
            "GRID_DIM": 256,
            "num_warps": 4,
            "num_stages": 1,
        }
    return {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_K": 256,
        "BLOCK_SIZE_N": 256,
        "GROUP_SIZE": 1,
        "GRID_DIM": 256,
        "num_warps": 8,
        "num_stages": 1,
    }


def _aiter_forward(
    lhs: torch.Tensor, rhs: torch.Tensor, group_sizes: torch.Tensor
) -> torch.Tensor:
    return gmm(
        lhs,
        rhs,
        group_sizes,
        preferred_element_type=lhs.dtype,
        config=_gmm_config(lhs.shape[1], rhs.shape[-1]),
    )


def _aiter_input_grad(
    grad_output: torch.Tensor,
    rhs: torch.Tensor,
    group_sizes: torch.Tensor,
) -> torch.Tensor:
    grad_output = grad_output.contiguous()
    rhs_transposed = rhs.transpose(1, 2)
    if rhs.shape[1] == rhs.shape[2]:
        # AITER rejects square row-major metadata because it is ambiguous
        # between its non-transposed and transposed-layout-2 cases.
        rhs_transposed = rhs_transposed.transpose(1, 2).contiguous().transpose(1, 2)
    return gmm(
        grad_output,
        rhs_transposed,
        group_sizes,
        preferred_element_type=grad_output.dtype,
        config=_gmm_config(rhs.shape[-1], rhs.shape[1]),
    )


def _aiter_weight_grad(
    lhs: torch.Tensor,
    grad_output: torch.Tensor,
    group_sizes: torch.Tensor,
) -> torch.Tensor:
    return ptgmm(
        lhs.T,
        grad_output.contiguous(),
        group_sizes,
        preferred_element_type=lhs.dtype,
        config=_ptgmm_config(lhs.shape[1], grad_output.shape[1]),
    )


class _AiterGroupedMM(torch.autograd.Function):
    """Autograd-capable ``(M,K) @ (E,K,N)`` grouped matrix multiplication."""

    @staticmethod
    def forward(
        ctx,
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        group_sizes: torch.Tensor,
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

        lhs = lhs.contiguous()
        group_sizes = group_sizes.to(device=lhs.device, dtype=torch.int32).contiguous()
        ctx.save_for_backward(lhs, rhs, group_sizes)
        return _aiter_forward(lhs, rhs, group_sizes)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        lhs, rhs, group_sizes = ctx.saved_tensors
        grad_lhs = (
            _aiter_input_grad(grad_output, rhs, group_sizes)
            if ctx.needs_input_grad[0]
            else None
        )
        grad_rhs = (
            _aiter_weight_grad(lhs, grad_output, group_sizes)
            if ctx.needs_input_grad[1]
            else None
        )
        return grad_lhs, grad_rhs, None


def aiter_grouped_mm(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    group_sizes: torch.Tensor,
) -> torch.Tensor:
    """Apply the autograd-capable AITER grouped matrix multiplication."""

    return _AiterGroupedMM.apply(lhs, rhs, group_sizes)


def _dequantize_selected_experts(
    weight: torch.Tensor,
    expert_indices: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(weight, GGUFQuantizedTensor):
        payload = weight.as_subclass(torch.Tensor)
        selected = payload.index_select(0, expert_indices.to(payload.device))
        return dequantize_gguf_tensor(
            selected, weight.quant_type, dtype=dtype, device=device
        )
    if not weight.is_floating_point():
        raise RuntimeError(
            "GGUF expert weights must be packed or floating point before AITER execution."
        )
    return weight.index_select(0, expert_indices.to(weight.device)).to(
        device=device, dtype=dtype
    )


class _AiterGGUFExpertProjection(torch.autograd.Function):
    """Run packed grouped MMQ forward and packed BF16-WMMA input backward."""

    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_offsets: torch.Tensor,
        group_sizes: torch.Tensor,
        compute_dtype: torch.dtype,
    ) -> torch.Tensor:
        if weight.requires_grad:
            raise RuntimeError(
                "Fast GGUF expert execution requires frozen packed base weights."
            )
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(compute_dtype).contiguous()
        expert_indices = expert_indices.to(
            device=hidden_states.device, dtype=torch.long
        ).contiguous()
        expert_offsets = expert_offsets.to(
            device=hidden_states.device, dtype=torch.int32
        ).contiguous()
        group_sizes = group_sizes.to(
            device=hidden_states.device, dtype=torch.int32
        ).contiguous()

        ctx.compute_dtype = compute_dtype
        ctx.input_dtype = input_dtype
        ctx.quant_type = (
            weight.quant_type if isinstance(weight, GGUFQuantizedTensor) else None
        )
        ctx.is_packed = isinstance(weight, GGUFQuantizedTensor)
        ctx.in_features = (
            int(weight.logical_shape[-1]) if ctx.is_packed else int(weight.shape[-1])
        )
        if ctx.needs_input_grad[0]:
            payload = weight.as_subclass(torch.Tensor) if ctx.is_packed else weight
            grouped_metadata = expert_offsets if ctx.is_packed else group_sizes
            ctx.save_for_backward(payload, expert_indices, grouped_metadata)

        if ctx.is_packed:
            if compute_dtype != torch.bfloat16:
                raise RuntimeError("Grouped GGUF MMQ requires BF16 compute dtype.")
            payload = weight.as_subclass(torch.Tensor)
            out_features = int(weight.logical_shape[-2])
            return grouped_mmq(
                hidden_states,
                payload,
                expert_indices,
                expert_offsets,
                int(weight.quant_type),
                out_features,
            )

        dense_weight = _dequantize_selected_experts(
            weight, expert_indices, compute_dtype, hidden_states.device
        )
        return _aiter_forward(hidden_states, dense_weight.transpose(1, 2), group_sizes)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_hidden = None
        if ctx.needs_input_grad[0]:
            payload, expert_indices, grouped_metadata = ctx.saved_tensors
            if ctx.is_packed:
                grad_hidden = torch.ops.torch_ggml_ops.grouped_mmq_grad_input.default(
                    grad_output.contiguous(),
                    payload,
                    expert_indices,
                    grouped_metadata,
                    int(ctx.quant_type),
                    ctx.in_features,
                )
            else:
                dense_weight = payload.index_select(
                    0, expert_indices.to(payload.device)
                ).to(
                    device=grad_output.device,
                    dtype=ctx.compute_dtype,
                )
                grad_hidden = _aiter_input_grad(
                    grad_output, dense_weight.transpose(1, 2), grouped_metadata
                )
            grad_hidden = grad_hidden.to(ctx.input_dtype)
        return grad_hidden, None, None, None, None, None


class _AiterGGUFExpertPairProjection(torch.autograd.Function):
    """Fuse packed gate/up forward workspaces and backward input accumulation."""

    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        first_weight: GGUFQuantizedTensor,
        second_weight: GGUFQuantizedTensor,
        expert_indices: torch.Tensor,
        expert_offsets: torch.Tensor,
        group_sizes: torch.Tensor,
        compute_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if first_weight.requires_grad or second_weight.requires_grad:
            raise RuntimeError(
                "Fast GGUF expert execution requires frozen packed base weights."
            )
        if compute_dtype != torch.bfloat16:
            raise RuntimeError("Grouped GGUF MMQ requires BF16 compute dtype.")
        if first_weight.quant_type != second_weight.quant_type:
            raise RuntimeError(
                "Paired grouped GGUF MMQ requires one quantization type."
            )
        if first_weight.logical_shape != second_weight.logical_shape:
            raise RuntimeError(
                "Paired grouped GGUF MMQ requires matching logical shapes."
            )

        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(compute_dtype).contiguous()
        expert_indices = expert_indices.to(
            device=hidden_states.device, dtype=torch.long
        ).contiguous()
        expert_offsets = expert_offsets.to(
            device=hidden_states.device, dtype=torch.int32
        ).contiguous()
        group_sizes = group_sizes.to(
            device=hidden_states.device, dtype=torch.int32
        ).contiguous()
        first_payload = first_weight.as_subclass(torch.Tensor)
        second_payload = second_weight.as_subclass(torch.Tensor)

        ctx.compute_dtype = compute_dtype
        ctx.input_dtype = input_dtype
        ctx.quant_type = first_weight.quant_type
        ctx.in_features = int(first_weight.logical_shape[-1])
        if ctx.needs_input_grad[0]:
            ctx.save_for_backward(
                first_payload,
                second_payload,
                expert_indices,
                expert_offsets,
            )

        return grouped_mmq_pair(
            hidden_states,
            first_payload,
            second_payload,
            expert_indices,
            expert_offsets,
            int(first_weight.quant_type),
            int(first_weight.logical_shape[-2]),
        )

    @staticmethod
    def backward(
        ctx,
        first_grad_output: torch.Tensor | None,
        second_grad_output: torch.Tensor | None,
    ):
        grad_hidden = None
        if ctx.needs_input_grad[0]:
            (
                first_payload,
                second_payload,
                expert_indices,
                expert_offsets,
            ) = ctx.saved_tensors
            if first_grad_output is not None and second_grad_output is not None:
                grad_hidden = (
                    torch.ops.torch_ggml_ops.grouped_mmq_pair_grad_input.default(
                        first_grad_output.contiguous(),
                        second_grad_output.contiguous(),
                        first_payload,
                        second_payload,
                        expert_indices,
                        expert_offsets,
                        int(ctx.quant_type),
                        ctx.in_features,
                    )
                )
            elif first_grad_output is not None:
                grad_hidden = torch.ops.torch_ggml_ops.grouped_mmq_grad_input.default(
                    first_grad_output.contiguous(),
                    first_payload,
                    expert_indices,
                    expert_offsets,
                    int(ctx.quant_type),
                    ctx.in_features,
                )
            elif second_grad_output is not None:
                grad_hidden = torch.ops.torch_ggml_ops.grouped_mmq_grad_input.default(
                    second_grad_output.contiguous(),
                    second_payload,
                    expert_indices,
                    expert_offsets,
                    int(ctx.quant_type),
                    ctx.in_features,
                )
            if grad_hidden is not None:
                grad_hidden = grad_hidden.to(ctx.input_dtype)
        return grad_hidden, None, None, None, None, None, None


def _base_grouped_linear(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_offsets: torch.Tensor,
    group_sizes: torch.Tensor,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    return _AiterGGUFExpertProjection.apply(
        hidden_states,
        weight,
        expert_indices,
        expert_offsets,
        group_sizes,
        compute_dtype,
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
    if (
        isinstance(first_weight, GGUFQuantizedTensor)
        and isinstance(second_weight, GGUFQuantizedTensor)
        and first_weight.quant_type == second_weight.quant_type
        and first_weight.logical_shape == second_weight.logical_shape
    ):
        return _AiterGGUFExpertPairProjection.apply(
            hidden_states,
            first_weight,
            second_weight,
            expert_indices,
            expert_offsets,
            group_sizes,
            compute_dtype,
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


class FastGGUFMoeLora(torch.nn.Module, LoraLayer):
    """PEFT LoRA wrapper owning all factors for one packed ``GGUFExperts`` module."""

    adapter_layer_names = ("lora_A", "lora_B", "lora_A_down", "lora_B_down")

    def _get_in_out_features(self, module: torch.nn.Module) -> tuple[int, int]:
        module = module.get_base_layer() if isinstance(module, LoraLayer) else module
        if not isinstance(module, GGUFExperts):
            raise TypeError(
                f"Fast GGUF MoE LoRA requires GGUFExperts, got {type(module).__name__}."
            )
        return module.hidden_dim, 2 * module.intermediate_dim

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

        experts = self.get_base_layer()
        if not isinstance(experts, GGUFExperts):
            raise TypeError(
                f"Fast GGUF MoE LoRA requires GGUFExperts, got {type(experts).__name__}."
            )
        device = experts.gate_proj.device
        dtype = experts.compute_dtype
        e = experts.num_experts
        h = experts.hidden_dim
        i = experts.intermediate_dim

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

        init = config.init_lora_weights
        if init is True:
            torch.nn.init.kaiming_uniform_(
                self.lora_A[adapter_name].weight, a=math.sqrt(5)
            )
            torch.nn.init.kaiming_uniform_(
                self.lora_A_down[adapter_name].weight, a=math.sqrt(5)
            )
            torch.nn.init.zeros_(self.lora_B[adapter_name].weight)
            torch.nn.init.zeros_(self.lora_B_down[adapter_name].weight)
        elif init == "gaussian":
            torch.nn.init.normal_(self.lora_A[adapter_name].weight, std=1 / r)
            torch.nn.init.normal_(self.lora_A_down[adapter_name].weight, std=1 / r)
            torch.nn.init.zeros_(self.lora_B[adapter_name].weight)
            torch.nn.init.zeros_(self.lora_B_down[adapter_name].weight)
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
        return _ExpertLoraWeights(
            gate_up_a=self.lora_A[adapter_name].weight,
            gate_up_b=self.lora_B[adapter_name].weight,
            down_a=self.lora_A_down[adapter_name].weight,
            down_b=self.lora_B_down[adapter_name].weight,
            scaling=float(self.scaling[adapter_name]),
        )

    def forward(
        self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        adapter_names = kwargs.pop("adapter_names", None)
        lora_weights = self._active_lora_weights(adapter_names)
        experts = self.get_base_layer()
        if experts.config._experts_implementation != EXPERTS_IMPLEMENTATION:
            raise RuntimeError(
                f"Fast GGUF MoE LoRA requires experts_implementation={EXPERTS_IMPLEMENTATION!r}, "
                f"got {experts.config._experts_implementation!r}."
            )
        kwargs[_LORA_WEIGHTS_KWARG] = lora_weights
        return self.base_layer(hidden_states, *args, **kwargs)


FastMoeParamWrapper = FastGGUFMoeLora


def _group_sizes_from_offsets(offsets: torch.Tensor) -> torch.Tensor:
    if offsets.ndim != 1:
        raise ValueError(
            f"GGUF grouped execution requires one-dimensional offsets, got {offsets.shape}."
        )
    previous = torch.cat((offsets.new_zeros(1), offsets[:-1]))
    return (offsets - previous).to(torch.int32).contiguous()


def _lora_grouped_linear(
    hidden_states: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    expert_indices: torch.Tensor,
    group_sizes: torch.Tensor,
) -> torch.Tensor:
    selected_a = lora_a.index_select(0, expert_indices).transpose(1, 2)
    selected_b = lora_b.index_select(0, expert_indices).transpose(1, 2)
    rank_states = aiter_grouped_mm(hidden_states, selected_a, group_sizes)
    return aiter_grouped_mm(rank_states, selected_b, group_sizes)


def qwen3_5_moe_gguf_mmq_aiter_lora_forward(
    self: GGUFExperts,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    _qwen3_5_moe_gguf_lora_weights: _ExpertLoraWeights | None = None,
) -> torch.Tensor:
    """Persistent packed GGUF base projections with separated AITER LoRA GEMMs."""

    if not isinstance(self, GGUFExperts):
        raise TypeError(
            f"{EXPERTS_IMPLEMENTATION} requires GGUFExperts, got {type(self).__name__}."
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
    execution_plan = self._prepare_expert_execution(routing_plan, "grouped_mm")
    if execution_plan.offsets is None:
        raise RuntimeError("GGUF AITER LoRA requires grouped expert offsets.")

    selected_hidden_states = routing_plan.selected_hidden_states
    expert_indices = execution_plan.expert_indices.to(
        device=selected_hidden_states.device, dtype=torch.long
    ).contiguous()
    expert_offsets = execution_plan.offsets.to(
        device=selected_hidden_states.device, dtype=torch.int32
    ).contiguous()
    group_sizes = _group_sizes_from_offsets(expert_offsets)

    gate, up = _base_grouped_pair(
        selected_hidden_states,
        self.gate_proj,
        self.up_proj,
        expert_indices,
        expert_offsets,
        group_sizes,
        self.compute_dtype,
    )

    lora_weights = _qwen3_5_moe_gguf_lora_weights
    if lora_weights is not None:
        gate_up_delta = _lora_grouped_linear(
            selected_hidden_states,
            lora_weights.gate_up_a,
            lora_weights.gate_up_b,
            expert_indices,
            group_sizes,
        )
        gate_delta, up_delta = gate_up_delta.chunk(2, dim=-1)
        gate = torch.add(gate, gate_delta, alpha=lora_weights.scaling)
        up = torch.add(up, up_delta, alpha=lora_weights.scaling)

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
            expert_indices,
            group_sizes,
        )
        output = torch.add(output, down_delta, alpha=lora_weights.scaling)

    return finalize_expert_routing(
        output,
        hidden_states,
        routing_plan,
        execution_plan.output_mask,
    )


def register_fast_moe_lora(
    lora_config: LoraConfig, model: torch.nn.Module
) -> LoraConfig:
    """Register the process-wide GGUF backend and its config-local PEFT wrapper."""

    register = getattr(lora_config, "_register_custom_module", None)
    if register is None:
        raise RuntimeError(
            "This PEFT version has no LoraConfig._register_custom_module API; "
            "cannot install fast GGUF MoE LoRA without a global PEFT monkey patch."
        )
    if lora_config.target_parameters:
        raise ValueError(
            "Persistent GGUF expert LoRA targets the complete 'experts' module, not target_parameters."
        )
    if isinstance(lora_config.target_modules, str):
        raise ValueError(
            "Fast GGUF MoE LoRA requires an explicit target_modules collection, not a regex string."
        )

    target_modules = set(lora_config.target_modules or ())
    target_modules.add("experts")
    lora_config.target_modules = target_modules

    ALL_GGUF_EXPERTS_FUNCTIONS[EXPERTS_IMPLEMENTATION] = (
        qwen3_5_moe_gguf_mmq_aiter_lora_forward
    )
    model.set_experts_implementation(EXPERTS_IMPLEMENTATION)
    register({GGUFExperts: FastGGUFMoeLora})
    return lora_config


__all__ = [
    "EXPERTS_IMPLEMENTATION",
    "FastGGUFMoeLora",
    "FastMoeParamWrapper",
    "aiter_grouped_mm",
    "qwen3_5_moe_gguf_mmq_aiter_lora_forward",
    "register_fast_moe_lora",
]
