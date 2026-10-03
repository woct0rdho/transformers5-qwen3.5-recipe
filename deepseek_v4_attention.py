import math
from collections.abc import Callable
from typing import Any, cast

import torch
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.deepseek_v4.configuration_deepseek_v4 import (
    DeepseekV4Config,
)
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4Attention,
    DeepseekV4Model,
    apply_rotary_pos_emb,
)

from deepseek_v4_csa import deepseek_v4_csa_attention, deepseek_v4_csa_compress
from deepseek_v4_hca import deepseek_v4_hca_attention, deepseek_v4_hca_compress
from deepseek_v4_sliding_attention import deepseek_v4_sliding_attention
from module_patching import (
    ModulePatchSpec,
    patch_module_forwards,
    require_complete_inventory,
)

_ATTENTION_IMPLEMENTATION = "deepseek_v4_project"
_SUPPORTED_BATCHES = frozenset({1, 4, 16})
_SEQUENCE_LENGTH = 2048
_QUERY_HEADS = 64
_KV_HEADS = 1
_HEAD_DIM = 512
_COMPRESSED_LENGTH = 512
_HCA_COMPRESSED_LENGTH = 16
_WINDOW = 128
_SOFTMAX_SCALE = 1.0 / math.sqrt(_HEAD_DIM)
_EXPECTED_SLIDING = 2
_EXPECTED_CSA = 21
_EXPECTED_HCA = 20
_SUBJECT = "DeepSeek V4 attention"
_CONFIG_MARKER = "_patched_attention"
_CSA_CONFIG_MARKER = "_patched_csa"
_HCA_CONFIG_MARKER = "_patched_hca"
_MODEL_HOOK_MARKER = "_patched_attention_input_hook"


def _canonical_training_mask(
    batch_size: int,
    q_length: int,
    kv_length: int,
    q_offset: int = 0,
    kv_offset: int = 0,
    mask_function=None,
    attention_mask: torch.Tensor | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    **kwargs: Any,
) -> torch.Tensor:
    """Return the mask view the family-owned kernels expect, with no mask data.

    No family-owned kernel reads a mask value: sliding attention ignores its mask
    argument, and the compressed families only check this view's shape. Building
    a real `[1, 1, S, S]` mask therefore costs an allocation and a device
    synchronization per step - transformers' eager mask creates its zero scalar
    on the device - to produce data nothing consumes. The canonical object is
    instead one poisoned element broadcast to the expected shape, so the
    metadata stays checkable while an accidental future read yields NaN instead
    of a mask that silently looks plausible.

    `mask_function` and the interface keyword arguments are accepted and
    unused. The remaining checks still reject padding, cache offsets, and any
    batch or sequence length the fixed-shape kernels do not support.
    """

    if batch_size not in _SUPPORTED_BATCHES:
        raise ValueError(f"unsupported DeepSeek V4 attention batch {batch_size}")
    if q_length != _SEQUENCE_LENGTH or kv_length != _SEQUENCE_LENGTH:
        raise ValueError(
            "DeepSeek V4 attention requires query and KV sequence length 2048"
        )
    if q_offset != 0 or kv_offset != 0:
        raise ValueError(
            "DeepSeek V4 training attention does not support cache offsets"
        )
    if attention_mask is not None:
        expected_shape = (batch_size, _SEQUENCE_LENGTH)
        if tuple(attention_mask.shape) != expected_shape:
            raise ValueError(
                f"attention_mask must have shape {expected_shape}, "
                f"got {tuple(attention_mask.shape)}"
            )
    if mask_function is None:
        raise ValueError("DeepSeek V4 attention requires a canonical mask function")

    # `expand` cannot infer the leading sizes from a one-element base, so the
    # canonical shape is spelled out. The block dimension stays 1, as the input
    # hook has already proven the batch shares one mask.
    return torch.full((1, 1, 1, 1), float("nan"), device=device, dtype=dtype).expand(
        batch_size, 1, q_length, kv_length
    )


def _same_tensor_view(left: torch.Tensor, right: torch.Tensor) -> bool | torch.SymBool:
    return (
        left.device == right.device
        and left.dtype == right.dtype
        and left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
        and left.storage_offset() == right.storage_offset()
        and tuple(left.shape) == tuple(right.shape)
        and tuple(left.stride()) == tuple(right.stride())
    )


def _deepseek_v4_attention_forward(
    module: DeepseekV4Attention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Any,
):
    if module.layer_type != "sliding_attention":
        raise RuntimeError(
            "DeepSeek V4 compressed attention must use its family-owned module forward"
        )

    if not getattr(module, _CONFIG_MARKER, False):
        raise RuntimeError("DeepSeek V4 sliding attention was not configured")
    if float(dropout) != 0.0 or module.attention_dropout != 0.0:
        raise ValueError("DeepSeek V4 sliding attention requires dropout zero")
    if module.sliding_window != _WINDOW:
        raise ValueError("DeepSeek V4 sliding attention requires window size 128")
    if not math.isclose(float(scaling), _SOFTMAX_SCALE, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("DeepSeek V4 sliding attention received an invalid scale")
    if not _same_tensor_view(key, value):
        raise ValueError("DeepSeek V4 sliding attention requires one shared K=V tensor")
    expected_mask = (query.shape[0], 1, _SEQUENCE_LENGTH, _SEQUENCE_LENGTH)
    if (
        not isinstance(attention_mask, torch.Tensor)
        or tuple(attention_mask.shape) != expected_mask
    ):
        raise ValueError(
            "DeepSeek V4 sliding attention requires the configured canonical "
            f"mask view with shape {expected_mask}"
        )
    auxiliary_sink = kwargs.pop("s_aux", None)
    if auxiliary_sink is not None and not _same_tensor_view(
        auxiliary_sink, module.sinks
    ):
        raise ValueError("DeepSeek V4 sliding attention received a foreign sink tensor")
    output = deepseek_v4_sliding_attention(query, key, module.sinks)
    return output, None


def _deepseek_v4_csa_module_forward(
    module: DeepseekV4Attention,
    hidden_states: torch.Tensor,
    position_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]]
    | tuple[torch.Tensor, torch.Tensor],
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    past_key_values=None,
    **kwargs: Any,
):
    if not getattr(module, _CSA_CONFIG_MARKER, False):
        raise RuntimeError("DeepSeek V4 CSA was not configured")
    batch, sequence_length, hidden_width = hidden_states.shape
    if batch not in _SUPPORTED_BATCHES or sequence_length != _SEQUENCE_LENGTH:
        raise ValueError("DeepSeek V4 CSA requires B in {1,4,16} and S=2048")
    if (
        hidden_width != module.config.hidden_size
        or hidden_states.dtype != torch.bfloat16
    ):
        raise ValueError("DeepSeek V4 CSA requires model-width BF16 hidden states")
    if past_key_values is not None:
        raise ValueError("DeepSeek V4 CSA does not support caches")
    if module.training and module.attention_dropout != 0.0:
        raise ValueError("DeepSeek V4 CSA requires dropout zero")
    expected_mask = (batch, 1, _SEQUENCE_LENGTH, _SEQUENCE_LENGTH)
    if (
        not isinstance(attention_mask, torch.Tensor)
        or tuple(attention_mask.shape) != expected_mask
    ):
        raise ValueError(
            f"DeepSeek V4 CSA requires canonical mask shape {expected_mask}"
        )
    if (
        not isinstance(position_ids, torch.Tensor)
        or position_ids.shape[0] not in (1, batch)
        or tuple(position_ids.shape[1:]) != (_SEQUENCE_LENGTH,)
    ):
        raise ValueError("DeepSeek V4 CSA requires canonical position_ids")
    if (
        not isinstance(position_embeddings, dict)
        or "compress" not in position_embeddings
    ):
        raise ValueError("DeepSeek V4 CSA requires compress position embeddings")
    if kwargs.get("output_attentions", False):
        raise ValueError("DeepSeek V4 CSA does not materialize attention weights")

    compressor = module.compressor
    if (
        compressor is None
        or compressor.compress_rate != 4
        or cast(Any, compressor.indexer).index_topk != _COMPRESSED_LENGTH
    ):
        raise ValueError("DeepSeek V4 CSA requires rate 4 and top-k 512")

    cos, sin = position_embeddings["compress"]
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, _HEAD_DIM)
    query_residual = module.q_a_norm(module.q_a_proj(hidden_states))
    query = module.q_b_proj(query_residual).view(*hidden_shape).transpose(1, 2)
    query = module.q_b_norm(query)
    query = apply_rotary_pos_emb(query, cos, sin)

    local_kv = (
        module.kv_norm(module.kv_proj(hidden_states))
        .view(*hidden_shape)
        .transpose(1, 2)
    )
    local_kv = apply_rotary_pos_emb(local_kv, cos, sin)
    compressor_kv = compressor.kv_proj(hidden_states)
    compressor_gate = compressor.gate_proj(hidden_states)
    compressed_kv = deepseek_v4_csa_compress(
        compressor_kv,
        compressor_gate,
        compressor.position_bias,
        compressor.kv_norm.weight,
        cos,
        sin,
        module.config.rms_norm_eps,
    )
    attention_output = deepseek_v4_csa_attention(
        query, local_kv, compressed_kv, module.sinks
    )
    attention_output = apply_rotary_pos_emb(
        attention_output.transpose(1, 2), cos, -sin
    ).transpose(1, 2)
    grouped = attention_output.reshape(*input_shape, module.config.o_groups, -1)
    grouped = module.o_a_proj(grouped).flatten(2)
    return module.o_b_proj(grouped), None


def _deepseek_v4_hca_module_forward(
    module: DeepseekV4Attention,
    hidden_states: torch.Tensor,
    position_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]]
    | tuple[torch.Tensor, torch.Tensor],
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    past_key_values=None,
    **kwargs: Any,
):
    if not getattr(module, _HCA_CONFIG_MARKER, False):
        raise RuntimeError("DeepSeek V4 HCA was not configured")
    batch, sequence_length, hidden_width = hidden_states.shape
    if batch not in _SUPPORTED_BATCHES or sequence_length != _SEQUENCE_LENGTH:
        raise ValueError("DeepSeek V4 HCA requires B in {1,4,16} and S=2048")
    if (
        hidden_width != module.config.hidden_size
        or hidden_states.dtype != torch.bfloat16
    ):
        raise ValueError("DeepSeek V4 HCA requires model-width BF16 hidden states")
    if past_key_values is not None:
        raise ValueError("DeepSeek V4 HCA does not support caches")
    if module.training and module.attention_dropout != 0.0:
        raise ValueError("DeepSeek V4 HCA requires dropout zero")
    expected_mask = (batch, 1, _SEQUENCE_LENGTH, _SEQUENCE_LENGTH)
    if (
        not isinstance(attention_mask, torch.Tensor)
        or tuple(attention_mask.shape) != expected_mask
    ):
        raise ValueError(
            f"DeepSeek V4 HCA requires canonical mask shape {expected_mask}"
        )
    if (
        not isinstance(position_ids, torch.Tensor)
        or position_ids.shape[0] not in (1, batch)
        or tuple(position_ids.shape[1:]) != (_SEQUENCE_LENGTH,)
    ):
        raise ValueError("DeepSeek V4 HCA requires canonical position_ids")
    if (
        not isinstance(position_embeddings, dict)
        or "compress" not in position_embeddings
    ):
        raise ValueError("DeepSeek V4 HCA requires compress position embeddings")
    if kwargs.get("output_attentions", False):
        raise ValueError("DeepSeek V4 HCA does not materialize attention weights")

    compressor = module.compressor
    if compressor is None or compressor.compress_rate != 128:
        raise ValueError("DeepSeek V4 HCA requires rate 128")

    cos, sin = position_embeddings["compress"]
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, _HEAD_DIM)
    query_residual = module.q_a_norm(module.q_a_proj(hidden_states))
    query = module.q_b_proj(query_residual).view(*hidden_shape).transpose(1, 2)
    query = module.q_b_norm(query)
    query = apply_rotary_pos_emb(query, cos, sin)

    local_kv = (
        module.kv_norm(module.kv_proj(hidden_states))
        .view(*hidden_shape)
        .transpose(1, 2)
    )
    local_kv = apply_rotary_pos_emb(local_kv, cos, sin)
    compressor_kv = compressor.kv_proj(hidden_states)
    compressor_gate = compressor.gate_proj(hidden_states)
    compressed_kv = deepseek_v4_hca_compress(
        compressor_kv,
        compressor_gate,
        compressor.position_bias,
        compressor.kv_norm.weight,
        cos,
        sin,
        module.config.rms_norm_eps,
    )
    if compressed_kv.shape[2] != _HCA_COMPRESSED_LENGTH:
        raise RuntimeError("DeepSeek V4 HCA producer must emit exactly 16 entries")
    attention_output = deepseek_v4_hca_attention(
        query, local_kv, compressed_kv, module.sinks
    )
    attention_output = apply_rotary_pos_emb(
        attention_output.transpose(1, 2), cos, -sin
    ).transpose(1, 2)
    grouped = attention_output.reshape(*input_shape, module.config.o_groups, -1)
    grouped = module.o_a_proj(grouped).flatten(2)
    return module.o_b_proj(grouped), None


def _argument_map(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    names = (
        "input_ids",
        "attention_mask",
        "position_ids",
        "past_key_values",
        "inputs_embeds",
        "use_cache",
    )
    values = dict(kwargs)
    for name, value in zip(names, args):
        values.setdefault(name, value)
    return values


def _validate_model_inputs(
    module: DeepseekV4Model,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    values = _argument_map(args, kwargs)
    input_ids = values.get("input_ids")
    inputs_embeds = values.get("inputs_embeds")
    inputs = inputs_embeds if inputs_embeds is not None else input_ids
    if not isinstance(inputs, torch.Tensor) or inputs.ndim < 2:
        raise ValueError("DeepSeek V4 fixed attention requires tensor model inputs")
    batch, sequence_length = inputs.shape[:2]
    if batch not in _SUPPORTED_BATCHES or sequence_length != _SEQUENCE_LENGTH:
        raise ValueError(
            "DeepSeek V4 fixed attention requires B in {1,4,16} and S=2048, "
            f"got B={batch}, S={sequence_length}"
        )
    if values.get("past_key_values") is not None or bool(
        values.get("use_cache", False)
    ):
        raise ValueError("DeepSeek V4 fixed training attention does not support caches")

    attention_mask = values.get("attention_mask")
    if attention_mask is not None:
        expected_shape = (batch, _SEQUENCE_LENGTH)
        if (
            not isinstance(attention_mask, torch.Tensor)
            or tuple(attention_mask.shape) != expected_shape
        ):
            raise ValueError(
                f"attention_mask must have shape {expected_shape}, "
                f"got {getattr(attention_mask, 'shape', None)}"
            )
        torch._assert_async(
            torch.all(attention_mask != 0),
            "DeepSeek V4 fixed attention does not support padding",
        )

    position_ids = values.get("position_ids")
    if position_ids is not None:
        if not isinstance(position_ids, torch.Tensor) or position_ids.ndim != 2:
            raise ValueError("position_ids must be a rank-2 tensor")
        if (
            position_ids.shape[0] not in (1, batch)
            or position_ids.shape[1] != _SEQUENCE_LENGTH
        ):
            raise ValueError("position_ids must have shape [1,2048] or [B,2048]")
        canonical = torch.arange(
            _SEQUENCE_LENGTH, device=position_ids.device, dtype=position_ids.dtype
        )
        torch._assert_async(
            torch.all(position_ids == canonical),
            "DeepSeek V4 fixed attention requires canonical unpacked positions",
        )


def _validate_config(config: DeepseekV4Config) -> None:
    expected = {
        "num_attention_heads": _QUERY_HEADS,
        "head_dim": _HEAD_DIM,
        "qk_rope_head_dim": 64,
        "sliding_window": _WINDOW,
        "attention_dropout": 0.0,
        "index_topk": _COMPRESSED_LENGTH,
        "use_cache": False,
    }
    mismatches = {
        name: (value, getattr(config, name, None))
        for name, value in expected.items()
        if getattr(config, name, None) != value
    }
    compress_rates = config.compress_rates or {}
    if compress_rates.get("compressed_sparse_attention") != 4:
        mismatches["compress_rate_csa"] = (
            4,
            compress_rates.get("compressed_sparse_attention"),
        )
    if compress_rates.get("heavily_compressed_attention") != 128:
        mismatches["compress_rate_hca"] = (
            128,
            compress_rates.get("heavily_compressed_attention"),
        )
    if mismatches:
        raise RuntimeError(f"unsupported DeepSeek V4 attention config: {mismatches}")


_ATTENTION_FAMILIES = (
    "sliding_attention",
    "compressed_sparse_attention",
    "heavily_compressed_attention",
)


def _validate_attention_families(model: torch.nn.Module) -> None:
    """Reject a layer type that no spec owns before any module is patched."""

    for name, module in model.named_modules():
        if not isinstance(module, DeepseekV4Attention):
            continue
        if module.layer_type not in _ATTENTION_FAMILIES:
            raise RuntimeError(
                f"unknown DeepSeek V4 attention family {module.layer_type!r} at {name!r}"
            )


def _attention_spec(
    family: str,
    *,
    forward: Callable[..., Any] | None,
    marker: str,
) -> ModulePatchSpec[DeepseekV4Attention]:
    """Route one attention layer family to its forward, or mark it when dispatch owns it."""

    def _matches(name: str, module: DeepseekV4Attention) -> bool:
        del name
        return module.layer_type == family

    return ModulePatchSpec(
        module_type=DeepseekV4Attention,
        forward=forward,
        handled_key=family,
        matches=_matches,
        marker=marker,
        freeze_weight=False,
    )


_ATTENTION_SPECS = (
    _attention_spec("sliding_attention", forward=None, marker=_CONFIG_MARKER),
    _attention_spec(
        "compressed_sparse_attention",
        forward=_deepseek_v4_csa_module_forward,
        marker=_CSA_CONFIG_MARKER,
    ),
    _attention_spec(
        "heavily_compressed_attention",
        forward=_deepseek_v4_hca_module_forward,
        marker=_HCA_CONFIG_MARKER,
    ),
)


def _hook_attention_inputs(model: torch.nn.Module) -> int:
    """Attach the shared input validator to every model shell that lacks it."""

    hooked_models = 0
    for module in model.modules():
        if not isinstance(module, DeepseekV4Model):
            continue
        hooked_models += 1
        if not hasattr(module, _MODEL_HOOK_MARKER):
            handle = module.register_forward_pre_hook(
                _validate_model_inputs, with_kwargs=True
            )
            setattr(module, _MODEL_HOOK_MARKER, handle)
    return hooked_models


def configure_deepseek_v4_attention(model: torch.nn.Module) -> dict[str, Any]:
    """Enable project attention dispatch on one already-loaded model instance."""

    config = getattr(model, "config", None)
    if not isinstance(config, DeepseekV4Config):
        raise TypeError("configure_deepseek_v4_attention requires DeepseekV4Config")
    _validate_config(config)
    _validate_attention_families(model)
    ALL_ATTENTION_FUNCTIONS.register(
        _ATTENTION_IMPLEMENTATION, _deepseek_v4_attention_forward
    )
    ALL_MASK_ATTENTION_FUNCTIONS.register(
        _ATTENTION_IMPLEMENTATION, _canonical_training_mask
    )

    report = patch_module_forwards(model, _ATTENTION_SPECS)
    report["paths"] = {
        key: sorted(names) for key, names in report["handled_by_key"].items()
    }
    report["hooked_models"] = _hook_attention_inputs(model)
    report["implementation"] = _ATTENTION_IMPLEMENTATION
    config._attn_implementation = _ATTENTION_IMPLEMENTATION
    return report


def require_complete_deepseek_v4_attention(report: dict[str, Any]) -> None:
    """Fail closed unless every fixed DeepSeek V4 attention site was configured."""

    require_complete_inventory(
        report,
        {
            "sliding_attention": _EXPECTED_SLIDING,
            "compressed_sparse_attention": _EXPECTED_CSA,
            "heavily_compressed_attention": _EXPECTED_HCA,
            "hooked_models": 1,
            "implementation": _ATTENTION_IMPLEMENTATION,
        },
        subject=_SUBJECT,
    )
