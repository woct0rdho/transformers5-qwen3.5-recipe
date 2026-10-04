#!/usr/bin/env python3
"""Qwen4-Exp full-step correctness, memory, gradient, and profiling audit."""

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
from typing import Any, cast

import bitsandbytes as bnb
import torch
from datasets import Dataset, load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM
from transformers.integrations.gguf.dequant import GGML_BLOCK
from transformers.integrations.gguf.gguf_quantized_parameter import (
    GgufQuantizedParameter,
)
from transformers.integrations.gguf.modules import (
    GgufEmbedding,
    GgufLinear,
    GgufQwen4ExpIndexerLinear,
)
from transformers.integrations.gguf.moe import GgufExperts
from transformers.integrations.gguf.reader import GgufHeader
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextAttention,
    Qwen4ExpTextDecoderLayer,
    Qwen4ExpTextGatedResidual,
    Qwen4ExpTextPLELayer,
    Qwen4ExpTextRMSNorm,
    Qwen4ExpTextRMSNormGated,
    Qwen4ExpTextTopKRouter,
)

from fast_lora import FastGgufLoraLinear, FastLoraLinear
from fast_moe_ranking import configure_fast_moe_ranking
from fla_tuning import configure_qwen4_exp_fla, require_complete_qwen4_exp_fla
from gdn_bwd_dhu import install as install_gdn_bwd_dhu
from gdn_bwd_dqkwg import install as install_gdn_bwd_dqkwg
from gdn_tiled_value_heads import configure_tiled_value_heads, require_tiled_value_heads
from gdn_tiled_value_heads import report as tiled_value_head_report
from gdn_wu_recompute import install as install_gdn_wu_recompute
from gguf_dequant_compile import configure_compiled_gguf_dequantize
from ple_disk_residency import (
    configure_ple_disk_residency,
    disk_state,
    require_ple_disk_residency,
)
from qwen4_exp_attention import (
    configure_qwen4_exp_qsa_attention,
    reference_attention_output,
    require_complete_qwen4_exp_qsa_attention,
)
from qwen4_exp_fused_norms import (
    configure_qwen4_exp_fused_norms,
    grouped_rms_norm,
    grouped_rms_norm_serves,
    require_complete_qwen4_exp_fused_norms,
)
from qwen4_exp_indexer import (
    configure_qwen4_exp_indexer_fast_path,
    reference_indexer_mask,
    require_complete_qwen4_exp_indexer,
)
from qwen4_exp_liger_hc import (
    configure_qwen4_exp_hc_norm,
    hc_norm_serves,
    reference_hc_output,
    require_complete_qwen4_exp_hc_norm,
)
from qwen4_exp_liger_loss import apply_qwen4_exp_liger_fused_linear_cross_entropy
from qwen4_exp_lora import (
    EXPECTED_ADAPTER_TENSORS,
    EXPECTED_EXPERT_WRAPPERS,
    EXPECTED_NATIVE_MMQ_WRAPPERS,
    EXPECTED_ORDINARY_WRAPPERS,
    EXPECTED_RANK4_PARAMETERS,
    QWEN4_EXP_TARGET_MODULES_PATTERN,
    configure_qwen4_exp_frozen_mmq,
    is_qwen4_exp_target,
    register_qwen4_exp_adapters,
    require_complete_qwen4_exp_frozen_mmq,
)
from qwen4_exp_moe_lora import (
    QWEN4_EXP_EXPERTS_IMPLEMENTATION,
    Qwen4ExpGgufMoeLora,
)
from qwen4_exp_profiler import profile_warmed_training_update
from training_audit import (
    audit_optimizer,
    audit_training_contract,
    clean_loading_info,
    complete_training_update,
    memory_snapshot,
    representative_packed_state,
    run_phase,
    summarize_b_updates,
    summarize_gradients,
    summarize_timeline,
    validate_first_gradients,
    validate_loss_output,
    validate_second_gradients,
)

# The packed payload, the logical parameter count, and the tensor count are properties of the
# checkpoint file, so the gate reads them from its header instead of holding a snapshot of one
# quantization recipe. Only the module graph is architecture-derived, and that stays a constant.
# `layer_multipliers`, `ngram_heads_vocab_sizes`, `ngram_heads_offsets`
PLE_STATE_BUFFERS = 3
EXPECTED_GGUF_LINEARS = 737
EXPECTED_INDEXER_LINEARS = 12
EXPECTED_GGUF_EMBEDDINGS = 2
EXPECTED_EXPERT_MODULES = 48
EXPECTED_DECODER_LAYERS = 48
EXPECTED_QSA_LAYERS = 12
EXPECTED_GATED_DELTA_NET_LAYERS = 36

_GGUF_EXPERTS_TYPE = cast(type[Any], GgufExperts)

EXPECTED_PLE_SITES = 1
PLE_DECODER_LAYER_INDEX = 1
PLE_TABLE_ROWS = 320_001_536
PLE_TABLE_DIM = 160
PLE_TABLE_QUANT_TYPE = 20
PLE_NGRAM_SIZE = 3
PLE_HEADS_PER_NGRAM = 8

# Fragments that must never carry a trainable adapter on this architecture.
_FORBIDDEN_TRAINABLE_FRAGMENTS = (
    ".self_attn.indexer.",
    ".ple.",
    "hyper_connection",
    ".mlp.gate.",
    ".shared_expert_gate.",
    "embed_tokens",
    "lm_head",
    "A_log",
    "dt_bias",
    ".conv1d.",
    "norm",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir", type=Path, default=Path("~/models/qwen4").expanduser()
    )
    parser.add_argument("--gguf-file", default="Qwen3.8-Flash-Next-GSQ-RCO-Q2_0.gguf")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data_tokenized_qwen3.5",
    )
    parser.add_argument(
        "--report-output", type=Path, default=Path("qwen4_training_report.json")
    )
    parser.add_argument("--profile-output", type=Path)
    parser.add_argument(
        "--ple-on-disk",
        action="store_true",
        help="keep the PLE n-gram table in the GGUF file and gather its rows per forward",
    )
    parser.add_argument("--batch-size", type=int, choices=(1,), default=1)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--row-start", type=int, default=0)
    parser.add_argument("--seed", type=int, default=19_260_817)
    return parser.parse_args()


def packed_identity(name: str, parameter: GgufQuantizedParameter) -> dict[str, Any]:
    """Identify one packed payload so an update cannot silently rewrite it."""

    payload = parameter.as_subclass(torch.Tensor).detach().reshape(-1)
    chunk = min(payload.numel(), 1024)
    middle = max((payload.numel() - chunk) // 2, 0)
    sample = torch.cat(
        (
            payload[:chunk].cpu(),
            payload[middle : middle + chunk].cpu(),
            payload[-chunk:].cpu(),
        )
    ).numpy()
    return {
        "name": name,
        "data_ptr": parameter.data_ptr(),
        "version": parameter._version,
        "sample_sha256": hashlib.sha256(sample.tobytes()).hexdigest(),
        "bytes": parameter.numel() * parameter.element_size(),
        "quant_type": int(parameter.quant_type),
        "logical_shape": list(parameter.logical_shape),
    }


def file_inventory(header: GgufHeader) -> dict[str, Any]:
    """What the checkpoint file says the loaded model must contain."""
    packed = [info for info in header.tensors if info.ggml_type in GGML_BLOCK]
    quant_types: dict[str, int] = {}
    for info in packed:
        key = str(info.ggml_type)
        quant_types[key] = quant_types.get(key, 0) + 1
    return {
        "state_tensors": len(header.tensors) + PLE_STATE_BUFFERS,
        "logical_parameters": sum(math.prod(info.shape) for info in header.tensors),
        "packed_parameters": len(packed),
        "packed_bytes": sum(info.nbytes for info in packed),
        "packed_quant_type_counts": quant_types,
    }


def audit_loaded_model(
    model: torch.nn.Module, loading_info: dict[str, Any], gguf_path: Path
) -> dict[str, Any]:
    packed = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if isinstance(parameter, GgufQuantizedParameter)
    ]
    quant_types: dict[str, int] = {}
    for _, parameter in packed:
        key = str(int(parameter.quant_type))
        quant_types[key] = quant_types.get(key, 0) + 1
    observed = {
        "state_tensors": len(model.state_dict()),
        "logical_parameters": sum(
            parameter.logical_numel
            if isinstance(parameter, GgufQuantizedParameter)
            else parameter.numel()
            for parameter in model.parameters()
        ),
        "packed_parameters": len(packed),
        "packed_bytes": sum(parameter.numel() for _, parameter in packed),
        "gguf_linears": sum(
            isinstance(module, GgufLinear) for module in model.modules()
        ),
        "indexer_linears": sum(
            isinstance(module, GgufQwen4ExpIndexerLinear) for module in model.modules()
        ),
        "gguf_embeddings": sum(
            isinstance(module, GgufEmbedding) for module in model.modules()
        ),
        "expert_modules": sum(
            isinstance(module, _GGUF_EXPERTS_TYPE) for module in model.modules()
        ),
        "decoder_layers": sum(
            isinstance(module, Qwen4ExpTextDecoderLayer) for module in model.modules()
        ),
    }
    expected = {
        **file_inventory(GgufHeader.from_file(str(gguf_path))),
        "gguf_linears": EXPECTED_GGUF_LINEARS,
        "indexer_linears": EXPECTED_INDEXER_LINEARS,
        "gguf_embeddings": EXPECTED_GGUF_EMBEDDINGS,
        "expert_modules": EXPECTED_EXPERT_MODULES,
        "decoder_layers": EXPECTED_DECODER_LAYERS,
    }
    state = disk_state()
    if state is not None:
        # The PLE table is served from the file, so the model holds no packed parameter for it, but it is
        # still part of the checkpoint: count it where the header counts it, or the inventory would read as
        # one packed tensor and 28.80 GiB short instead of as intentionally elsewhere.
        observed["logical_parameters"] += state.geometry.rows * state.geometry.columns
        observed["packed_parameters"] += 1
        observed["packed_bytes"] += state.geometry.nbytes
        key = str(state.geometry.ggml_type)
        quant_types[key] = quant_types.get(key, 0) + 1
    observed["packed_quant_type_counts"] = quant_types
    errors = [
        f"{key}: expected {value}, found {observed[key]}"
        for key, value in expected.items()
        if observed[key] != value
    ]
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        if loading_info.get(key):
            errors.append(f"{key}={loading_info[key]}")
    outside_cuda = [
        name
        for name, tensor in (*model.named_parameters(), *model.named_buffers())
        if tensor.device.type != "cuda"
    ]
    if outside_cuda:
        errors.append(f"model tensors outside cuda:0: {outside_cuda[:8]}")
    if errors:
        raise RuntimeError("Qwen4-Exp load audit failed: " + "; ".join(errors))
    return observed | {
        "loading_info": clean_loading_info(loading_info),
        "file_inventory": file_inventory(GgufHeader.from_file(str(gguf_path))),
    }


def audit_ple_table(model: torch.nn.Module) -> dict[str, Any]:
    """Validate the single PLE site, its n-gram table, and the table's frozen state."""

    ple_layers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, Qwen4ExpTextPLELayer)
    ]
    errors = []
    if len(ple_layers) != EXPECTED_PLE_SITES:
        errors.append(
            f"expected {EXPECTED_PLE_SITES} PLE sites, found {len(ple_layers)}"
        )
    name, layer = ple_layers[0]
    expected_name = f"model.layers.{PLE_DECODER_LAYER_INDEX}.ple"
    if not name.endswith(expected_name):
        errors.append(f"PLE site is {name!r}, expected a site at {expected_name!r}")
    embedding = layer.ple_embedding
    table = embedding.ngram_embedding
    weight = table.weight
    if not isinstance(table, GgufEmbedding):
        errors.append(
            f"PLE n-gram table is {type(table).__name__}, expected GgufEmbedding"
        )
    state = disk_state()
    if state is not None:
        # The payload is in the file: the parameter is an empty placeholder and the table's geometry is the
        # one the reader resolved from the checkpoint header.
        if weight.numel():
            errors.append(
                f"PLE table still holds {weight.numel()} payload bytes while it is served from disk"
            )
        if int(state.geometry.ggml_type) != PLE_TABLE_QUANT_TYPE:
            errors.append(
                f"PLE table quant type is {int(state.geometry.ggml_type)}, "
                f"expected {PLE_TABLE_QUANT_TYPE}"
            )
        if (state.geometry.rows, state.geometry.columns) != (
            PLE_TABLE_ROWS,
            PLE_TABLE_DIM,
        ):
            errors.append(
                f"PLE table is [{state.geometry.rows}, {state.geometry.columns}], "
                f"expected {[PLE_TABLE_ROWS, PLE_TABLE_DIM]}"
            )
    elif not isinstance(weight, GgufQuantizedParameter):
        errors.append("PLE n-gram table is not a packed parameter")
    else:
        if int(weight.quant_type) != PLE_TABLE_QUANT_TYPE:
            errors.append(
                f"PLE table quant type is {int(weight.quant_type)}, "
                f"expected {PLE_TABLE_QUANT_TYPE}"
            )
        if list(weight.logical_shape) != [PLE_TABLE_ROWS, PLE_TABLE_DIM]:
            errors.append(
                f"PLE table logical shape is {list(weight.logical_shape)}, "
                f"expected {[PLE_TABLE_ROWS, PLE_TABLE_DIM]}"
            )
    if table.num_embeddings != PLE_TABLE_ROWS or table.embedding_dim != PLE_TABLE_DIM:
        errors.append(
            f"PLE table is [{table.num_embeddings}, {table.embedding_dim}], "
            f"expected [{PLE_TABLE_ROWS}, {PLE_TABLE_DIM}]"
        )
    if weight.requires_grad:
        errors.append("PLE n-gram table must stay frozen")
    if embedding.ngram_size != PLE_NGRAM_SIZE:
        errors.append(
            f"PLE ngram_size is {embedding.ngram_size}, expected {PLE_NGRAM_SIZE}"
        )
    if embedding.heads_per_ngram != PLE_HEADS_PER_NGRAM:
        errors.append(
            f"PLE heads_per_ngram is {embedding.heads_per_ngram}, "
            f"expected {PLE_HEADS_PER_NGRAM}"
        )
    if errors:
        raise RuntimeError("Qwen4-Exp PLE audit failed: " + "; ".join(errors))
    return {
        "site": name,
        "decoder_layer_index": PLE_DECODER_LAYER_INDEX,
        "ngram_size": embedding.ngram_size,
        "heads_per_ngram": embedding.heads_per_ngram,
        "context_len": embedding.context_len,
        "head_vocab_sizes": len(embedding.head_vocab_sizes),
        "table_rows": table.num_embeddings,
        "table_dim": table.embedding_dim,
        "table_quant_type": int(weight.quant_type)
        if state is None
        else int(state.geometry.ggml_type),
        "table_payload_bytes": state.geometry.nbytes
        if (state := disk_state()) is not None
        else weight.numel() * weight.element_size(),
        "table_disk_backed": state is not None,
        "identity": state.identity
        if state is not None
        else packed_identity(f"{name}.ple_embedding.ngram_embedding.weight", weight),
    }


def audit_qsa_indexer_skip(
    model: torch.nn.Module, config: Any, sequence_length: int
) -> dict[str, Any]:
    """Prove the short circuit returns the mask the reference indexer would have returned.

    The selection is exhaustive at the audited length, so the mask does not depend on the scores and
    this gate can compare against the causal-plus-padding mask directly. The padded case is what the
    training collator produces for a short sample, and the reference is the module's own forward,
    which the patch keeps.
    """

    text_config = getattr(config, "text_config", config)
    indexers = [
        (name, module)
        for name, module in model.named_modules()
        if name.endswith(".self_attn.indexer")
    ]
    if len(indexers) != EXPECTED_QSA_LAYERS:
        raise RuntimeError(
            f"expected {EXPECTED_QSA_LAYERS} QSA indexers, found {len(indexers)}"
        )
    name, indexer = indexers[0]
    hidden_size = int(text_config.hidden_size)
    head_dim = int(text_config.head_dim)
    rope_parameters = getattr(text_config, "rope_parameters", None) or {}
    rotary_dim = int(head_dim * rope_parameters.get("partial_rotary_factor", 1.0))
    if rotary_dim > int(indexer.index_head_dim):
        raise RuntimeError(
            f"Qwen4-Exp attention rotary dim {rotary_dim} does not fit the index head "
            f"{indexer.index_head_dim}"
        )
    parameter = next(indexer.parameters())
    device, dtype = parameter.device, parameter.dtype
    generator = torch.Generator(device="cpu").manual_seed(19_260_817)
    hidden_states = torch.randn(
        (1, sequence_length, hidden_size), generator=generator, dtype=torch.float32
    ).to(device=device, dtype=dtype)
    cos = torch.randn(
        (1, sequence_length, rotary_dim), generator=generator, dtype=torch.float32
    ).to(device=device, dtype=dtype)
    sin = torch.randn(
        (1, sequence_length, rotary_dim), generator=generator, dtype=torch.float32
    ).to(device=device, dtype=dtype)
    positions = torch.arange(sequence_length, device=device)
    causal = positions[:, None] >= positions[None, :]
    cases = {
        "all_visible": torch.ones(
            (1, sequence_length), dtype=torch.bool, device=device
        ),
        "right_padded": (positions < sequence_length - 64)[None, :],
    }
    report: dict[str, Any] = {"indexer": name, "sequence_length": sequence_length}
    for label, valid in cases.items():
        attention_mask = (causal[None, None] & valid[:, None, None, :]).contiguous()
        reference_mask = reference_indexer_mask(
            indexer, hidden_states, (cos, sin), attention_mask
        )
        patched_mask = indexer(hidden_states, (cos, sin), attention_mask, None)
        matches_reference = bool(torch.equal(reference_mask, attention_mask))
        returns_mask = patched_mask is attention_mask
        if not matches_reference or not returns_mask:
            raise RuntimeError(
                f"QSA indexer short circuit disagrees with the reference for {label}: "
                f"reference_equals_mask={matches_reference}, patched_returns_mask={returns_mask}"
            )
        report[label] = {
            "reference_equals_mask": matches_reference,
            "patched_returns_mask": returns_mask,
        }
    return report


def audit_qsa_attention_equivalence(
    model: torch.nn.Module, config: Any, sequence_length: int
) -> dict[str, Any]:
    """Prove the QSA kernels agree with the model's SDPA path on the same inputs.

    The indexer is already short circuited, so the only difference between the two paths is the
    attention itself. The right-padded case is what the collator produces for a short sample. Only
    the rows that carry an incoming gradient are compared, because the collator masks the labels of
    the padded rows.
    """

    text_config = getattr(config, "text_config", config)
    attentions = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, Qwen4ExpTextAttention)
    ]
    if len(attentions) != EXPECTED_QSA_LAYERS:
        raise RuntimeError(
            f"expected {EXPECTED_QSA_LAYERS} QSA attention layers, found {len(attentions)}"
        )
    name, attention = attentions[0]
    head_dim = int(text_config.head_dim)
    rope_parameters = getattr(text_config, "rope_parameters", None) or {}
    rotary_dim = int(head_dim * rope_parameters.get("partial_rotary_factor", 1.0))
    # The layer's parameters are GGUF-quantized bytes, so the activation dtype comes from the load
    # instead: the audit loads bf16 and the QSA kernels only serve bf16.
    device = next(attention.parameters()).device
    dtype = torch.bfloat16
    generator = torch.Generator(device="cpu").manual_seed(19_260_818)
    hidden_states = torch.randn(
        (1, sequence_length, int(text_config.hidden_size)),
        generator=generator,
        dtype=torch.float32,
    ).to(device=device, dtype=dtype)
    cos = torch.randn(
        (1, sequence_length, rotary_dim), generator=generator, dtype=torch.float32
    ).to(device=device, dtype=dtype)
    sin = torch.randn(
        (1, sequence_length, rotary_dim), generator=generator, dtype=torch.float32
    ).to(device=device, dtype=dtype)
    positions = torch.arange(sequence_length, device=device)
    causal = positions[:, None] >= positions[None, :]
    cases = {
        "all_visible": torch.ones(
            (1, sequence_length), dtype=torch.bool, device=device
        ),
        "right_padded": (positions < sequence_length - 64)[None, :],
    }
    report: dict[str, Any] = {"attention": name, "sequence_length": sequence_length}
    for label, valid in cases.items():
        attention_mask = (causal[None, None] & valid[:, None, None, :]).contiguous()
        # The module's own forward always receives the combined mask, indexer included, which is
        # never None on this path.
        with torch.no_grad():
            patched = attention(hidden_states, (cos, sin), attention_mask, None)[0]
            reference = reference_attention_output(
                attention, hidden_states, (cos, sin), attention_mask
            )
        rows = int(valid.sum())
        left = patched[:, :rows].float().flatten()
        right = reference[:, :rows].float().flatten()
        difference = left - right
        rmse = float(
            difference.square().mean().sqrt() / (right.square().mean().sqrt() + 1e-12)
        )
        cosine = float(torch.nn.functional.cosine_similarity(left, right, dim=0))
        if rmse > 0.02 or cosine < 0.999:
            raise RuntimeError(
                f"QSA attention disagrees with the reference on {label}: "
                f"rmse {rmse}, cosine {cosine}"
            )
        report[label] = {"rmse": rmse, "cosine": cosine}
    return report


def audit_hc_norm_equivalence(
    model: torch.nn.Module, config: Any, sequence_length: int
) -> dict[str, Any]:
    """Prove the fused grouped RMSNorm is as accurate as the module's own norm.

    Three checks. The kernel's output and `dX` are compared against an fp64 truth taken from autograd
    on a transcription of the module's own expression, because both the kernel and the module sit on
    the bf16 store floor there and a truth that repeats the kernel's algebra agrees with any error of
    its own. The module forward and gradients are then compared against the module's kept reference
    with a loose RMSE and a tight cosine, which is what catches a wiring mistake such as a wrong
    group, a dropped `(1 + weight)` or a transposed layout. Finally the kernel predicate is required,
    so a silent fall back cannot pass this gate.
    """

    connections = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, Qwen4ExpTextGatedResidual)
    ]
    if not connections:
        raise RuntimeError("no Qwen4-Exp hyper-connections found to check")
    name, connection = connections[0]
    device = next(connection.parameters()).device
    weight = connection.hc_norm.weight
    group = connection.hc_norm.group_size
    eps = float(connection.hc_norm.eps)
    flat = int(weight.numel())
    rows = sequence_length
    if group is None or flat % group != 0:
        raise RuntimeError(f"{name!r} has no usable group size")
    generator = torch.Generator(device="cpu").manual_seed(19_260_819)

    def rand(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float32).to(
            device=device, dtype=torch.bfloat16
        )

    hyper_input = rand(rows, flat)
    if not hc_norm_serves(connection, hyper_input):
        raise RuntimeError(
            f"the fused grouped RMSNorm does not serve {name!r}: the hyper-connection is outside "
            f"the shapes, the dtype or the frozen weight the kernel requires"
        )

    report: dict[str, Any] = {
        "connection": name,
        "rows": rows,
        "connections": len(connections),
        "group_size": group,
    }

    # The norm kernel against fp64 truth, forward and backward, where the truth is autograd on the
    # module's own expression rather than a second transcription of the kernel's algebra: a
    # transcription that leaves the scale vector out of the mean agrees with itself.
    cotangent = rand(rows, flat)
    leaf = hyper_input.detach().requires_grad_(True)
    fused_norm = grouped_rms_norm(leaf, weight, eps, group)
    fused_norm.backward(cotangent)
    x64 = hyper_input.double().view(rows, -1, group).requires_grad_(True)
    w64 = 1.0 + weight.double().view(-1, group)
    truth_leaf = x64 * torch.rsqrt(x64.pow(2).mean(-1, keepdim=True) + eps) * w64
    truth_leaf.backward(cotangent.double().view(rows, -1, group))
    truth_norm = truth_leaf.detach().view(rows, flat)
    truth_dx = x64.grad.view(rows, flat)
    for label, got, want in (
        ("norm_output_vs_fp64", fused_norm.detach(), truth_norm),
        ("norm_dx_vs_fp64", leaf.grad, truth_dx),
    ):
        left = got.float().flatten()
        right = want.float().flatten()
        difference = left - right
        rmse = float(
            difference.square().mean().sqrt() / (right.square().mean().sqrt() + 1e-12)
        )
        cosine = float(torch.nn.functional.cosine_similarity(left, right, dim=0))
        if rmse > 5e-3 or cosine < 0.99999:
            raise RuntimeError(
                f"fused grouped RMSNorm is not within the bf16 floor of fp64 on {label}: "
                f"rmse {rmse}, cosine {cosine}"
            )
        report[label] = {"rmse": rmse, "cosine": cosine}

    # The reference chain's own distance from fp64, for context on the loose threshold below.
    reference_leaf = hyper_input.detach().requires_grad_(True)
    connection.hc_norm(reference_leaf).backward(cotangent)
    reference_dx = reference_leaf.grad.float().flatten()
    report["reference_norm_dx_vs_fp64"] = float(
        (reference_dx - truth_dx.float().flatten()).square().mean().sqrt()
        / (truth_dx.float().square().mean().sqrt() + 1e-12)
    )

    # The patched module against its kept reference, on the same activations.
    def run(function):
        sample = hyper_input.detach().requires_grad_(True)
        mixed, _streams, injection = function(sample)
        (mixed.float().square().mean() + injection.float().square().mean()).backward()
        return mixed.detach(), injection.detach(), sample.grad

    reference = run(lambda sample: reference_hc_output(connection, sample))
    fused = run(lambda sample: connection(sample))
    for label, got, want in (
        ("mixed_input", fused[0], reference[0]),
        ("injection_weights", fused[1], reference[1]),
        ("input_gradient", fused[2], reference[2]),
    ):
        left = got.float().flatten()
        right = want.float().flatten()
        difference = left - right
        rmse = float(
            difference.square().mean().sqrt() / (right.square().mean().sqrt() + 1e-12)
        )
        cosine = float(torch.nn.functional.cosine_similarity(left, right, dim=0))
        if rmse > 3e-2 or cosine < 0.9999:
            raise RuntimeError(
                f"patched hyper-connection disagrees with its reference on {label}: "
                f"rmse {rmse}, cosine {cosine}"
            )
        report[label] = {"rmse": rmse, "cosine": cosine}
    return report


def audit_fused_norm_equivalence(
    model: torch.nn.Module, sequence_length: int
) -> dict[str, Any]:
    """Check the three norm contracts the fused patch installs.

    Every contract is compared against autograd on an fp64 transcription of the module's own
    expression, forward and backward, so the check cannot repeat the algebra of the kernel it checks.
    The scale vector is what all three have in common and where they differ: the module upcasts it,
    and it is `1 + weight` for the plain and grouped norms but a plain `weight` for the gated one.
    The grouped sites must also satisfy the kernel's predicate, so a silent fall back cannot pass.
    """

    plain = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, Qwen4ExpTextRMSNorm) and module.group_size is None
    ]
    grouped = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, Qwen4ExpTextRMSNorm)
        and module.group_size is not None
        and not name.endswith(".hc_norm")
    ]
    gated = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, Qwen4ExpTextRMSNormGated)
    ]
    for label, sites in (("plain", plain), ("grouped", grouped), ("gated", gated)):
        if not sites:
            raise RuntimeError(f"no Qwen4-Exp {label} norm found to check")

    device = next(model.parameters()).device
    generator = torch.Generator(device="cpu").manual_seed(19_260_819)
    report: dict[str, Any] = {
        "plain_sites": len(plain),
        "grouped_sites": len(grouped),
        "gated_sites": len(gated),
    }

    def rand(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float32).to(
            device=device, dtype=torch.bfloat16
        )

    def activation_of(module: Qwen4ExpTextRMSNormGated):
        if module.activation in ("swish", "silu"):
            return torch.nn.functional.silu
        if module.activation == "sigmoid":
            return torch.sigmoid
        raise RuntimeError(f"unsupported activation {module.activation!r}")

    def truth(x, weight, cotangent, eps, group, offset, gate, activation):
        rows = x.numel() // x.shape[-1]
        x64 = x.double().view(rows, -1, group).requires_grad_(True)
        scale = offset + weight.double().view(-1, group)
        x_norm = x64 * torch.rsqrt(x64.pow(2).mean(-1, keepdim=True) + eps)
        output = x_norm * scale
        gate64 = None
        if gate is not None:
            gate64 = gate.double().view(rows, -1, group).requires_grad_(True)
            output = output * activation(gate64)
        output.backward(cotangent.double().view(rows, -1, group))
        return (
            output.detach().view(rows, -1),
            x64.grad.view(rows, -1),
            None if gate64 is None else gate64.grad.view(rows, -1),
        )

    def compare(label, got, want):
        left, right = got.float().flatten(), want.float().flatten()
        difference = left - right
        rmse = float(
            difference.square().mean().sqrt() / (right.square().mean().sqrt() + 1e-12)
        )
        cosine = float(torch.nn.functional.cosine_similarity(left, right, dim=0))
        if rmse > 5e-3 or cosine < 0.99999:
            raise RuntimeError(
                f"fused norm {label} is not within the bf16 floor of fp64: "
                f"rmse {rmse}, cosine {cosine}"
            )
        return {"rmse": rmse, "cosine": cosine}

    def check_site(label, name, module, group, offset, eps, gate=None, activation=None):
        width = int(module.weight.numel())
        x = rand(sequence_length, width)
        cotangent = rand(sequence_length, width)
        leaf = x.detach().requires_grad_(True)
        gate_leaf = None if gate is None else gate.detach().requires_grad_(True)
        output = module(leaf) if gate_leaf is None else module(leaf, gate_leaf)
        output.backward(cotangent)
        want, want_dx, want_dgate = truth(
            x, module.weight, cotangent, eps, group, offset, gate, activation
        )
        entry: dict[str, Any] = {
            "site": name,
            "rows": sequence_length,
            "width": width,
        }
        entry["output_vs_fp64"] = compare(f"{label} output", output.detach(), want)
        entry["input_gradient_vs_fp64"] = compare(
            f"{label} input gradient", leaf.grad, want_dx
        )
        if gate_leaf is not None:
            entry["gate_gradient_vs_fp64"] = compare(
                f"{label} gate gradient", gate_leaf.grad, want_dgate
            )
        return entry

    name, module = plain[0]
    report["plain"] = check_site(
        "plain", name, module, int(module.weight.numel()), 1.0, float(module.eps)
    )

    name, module = gated[0]
    width = int(module.weight.numel())
    activation = activation_of(module)
    report["gated"] = check_site(
        "gated",
        name,
        module,
        width,
        0.0,
        float(module.variance_epsilon),
        gate=rand(sequence_length, width),
        activation=activation,
    )
    report["gated"]["activation"] = module.activation

    name, module = grouped[0]
    group = module.group_size
    if group is None:
        raise RuntimeError(f"{name!r} has no group size")
    if not grouped_rms_norm_serves(
        rand(sequence_length, int(module.weight.numel())), module.weight, group
    ):
        raise RuntimeError(
            f"the grouped kernel does not serve {name!r}: the site is outside the shapes, the "
            f"dtype or the frozen weight the kernel requires"
        )
    report["grouped"] = check_site(
        "grouped", name, module, group, 1.0, float(module.eps)
    )
    return report


# Router logits are FP32 by contract: the selection runs in FP32 and narrows to the activation
# dtype on the way out, so the gate module is the one place a non-BF16 tensor is expected.
_AUTOCAST_FP32_MODULES = ("mlp.gate", "indexer", "lm_head", "loss")


def audit_autocast_dtypes(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Run one update under the trainer's autocast and require one activation dtype.

    `train_qwen4_exp.py` trains with `bf16=True`, so accelerate runs the model under
    `torch.autocast`, and a custom op has no autocast policy: whatever dtype reaches it is what its
    kernel validates. Autocast's FP32 list includes the reductions, so an op like `sum` can promote
    an activation and take the whole residual stream with it - which is how the PLE once turned
    every layer after it FP32 and failed the packed projections' BF16 contract.

    Every leaf module's forward outputs and their cotangents are checked, so a promotion is caught
    wherever it appears rather than in whatever kernel happens to validate first. The cotangents are
    read with output-tensor hooks rather than with full backward hooks, which would make every
    module input require a gradient and cost tens of gigabytes. The router gate is allowed to be
    FP32, because its selection is FP32 by contract and it narrows on the way out. The gradients
    this pass creates are cleared before the measured updates.

    The gate has to run in the trainer's own state, so the caller runs it with the adapters injected
    and gradient checkpointing on. Without checkpointing the same pass holds every layer's
    activations and peaks about 39 GiB above the model instead of three.
    """

    offenders: dict[str, list[str]] = {"forward": [], "backward": []}
    checked = {"forward": 0, "backward": 0}
    names: dict[int, str] = {}

    def allowed(name: str) -> bool:
        return any(fragment in name for fragment in _AUTOCAST_FP32_MODULES)

    def record(where: str, name: str, dtype: torch.dtype) -> None:
        if dtype == torch.bfloat16 or allowed(name):
            return
        entry = f"{name} ({dtype})"
        if entry not in offenders[where]:
            offenders[where].append(entry)

    def forward_hook(module, args, output):
        name = names[id(module)]
        values = output if isinstance(output, (tuple, list)) else (output,)
        for value in values:
            if not isinstance(value, torch.Tensor) or not value.is_floating_point():
                continue
            checked["forward"] += 1
            record("forward", name, value.dtype)
            if value.requires_grad:
                value.register_hook(cotangent_hook(name))

    def cotangent_hook(name: str):
        def hook(gradient: torch.Tensor) -> None:
            checked["backward"] += 1
            record("backward", name, gradient.dtype)

        return hook

    handles = []
    for name, module in model.named_modules():
        if list(module.children()):
            continue
        names[id(module)] = name
        handles.append(module.register_forward_hook(forward_hook))

    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(**batch, use_cache=False)
        loss = output.loss
        if loss is None:
            raise RuntimeError("the autocast gate needs a forward that returns a loss")
        loss.backward()
    loss_value = float(loss.detach())

    for handle in handles:
        handle.remove()
    model.zero_grad(set_to_none=True)

    if offenders["forward"] or offenders["backward"]:
        raise RuntimeError(
            "the autocast pass produced an activation outside the model's BF16 contract: "
            f"forward {offenders['forward'][:6]}, backward {offenders['backward'][:6]}"
        )
    return {
        "loss": loss_value,
        "tokens": int(batch["input_ids"].shape[-1]),
        "leaf_modules": len(names),
        "tensors_checked": checked,
        "non_bf16": offenders,
        "allowed_fp32_modules": list(_AUTOCAST_FP32_MODULES),
    }


def audit_qsa_indexer(
    model: torch.nn.Module, config: Any, sequence_length: int
) -> dict[str, Any]:
    """Report the QSA schedule and whether its selection is exhaustive at this length.

    With `indexer_budget // compress_ratio` block slots and at most
    `sequence_length // compress_ratio` complete blocks, the `topk` selects every complete block, so
    the returned mask equals the causal-plus-padding mask and the indexer is a no-op at this shape.
    The reference stack relies on the same property to delete the equivalent work in DeepSeek V4.
    """

    qsa_layers = [
        index
        for index, kind in enumerate(config.layer_types)
        if kind == "indexed_attention"
    ]
    if len(qsa_layers) != EXPECTED_QSA_LAYERS:
        raise RuntimeError(
            f"expected {EXPECTED_QSA_LAYERS} QSA layers, found {len(qsa_layers)}"
        )
    ratio = int(config.indexer_compress_ratio)
    budget = int(config.indexer_budget)
    block_topk = budget // ratio
    max_complete_blocks = sequence_length // ratio
    exhaustive = block_topk >= max_complete_blocks
    indexer_parameters = [
        name
        for name, parameter in model.named_parameters()
        if ".self_attn.indexer." in name
    ]
    indexer_linears = [
        name
        for name, module in model.named_modules()
        if isinstance(module, GgufQwen4ExpIndexerLinear)
    ]
    if len(indexer_linears) != len(qsa_layers):
        raise RuntimeError(
            f"expected one indexer projection per QSA layer, found "
            f"{len(indexer_linears)} for {len(qsa_layers)} layers"
        )
    # Whether these parameters are frozen is checked after adapter injection, where the whole
    # non-adapter surface must be frozen. At load time floating parameters still require grad.
    return {
        "qsa_layers": len(qsa_layers),
        "qsa_layer_indices": qsa_layers,
        "compress_ratio": ratio,
        "indexer_budget": budget,
        "block_topk": block_topk,
        "max_complete_blocks": max_complete_blocks,
        "selection_exhaustive_at_sequence_length": exhaustive,
        "indexer_parameters": len(indexer_parameters),
        "indexer_linears": len(indexer_linears),
    }


def attach_ranking_value_check(model: torch.nn.Module):
    """Compare the installed selection with the model's own selection on real router logits.

    The installed selection is deterministic, which gradient checkpointing needs, but it is not
    required to reproduce `torch.topk`'s choice among tied experts. Expert ids are therefore counted
    and never compared: the gate is the selected weights against the model's own, with a tolerance,
    and the requirement that every selected score is at or above the kth threshold.
    """

    state = {
        "layers": 0,
        "rows": 0,
        "weight_squared_error": 0.0,
        "weight_squared_reference": 0.0,
        "threshold_violations": 0,
        "id_differences": 0,
        "rows_with_id_differences": 0,
    }
    handles = []

    def make_hook(module: Qwen4ExpTextTopKRouter):
        def hook(_module, _args, output):
            logits, weights, indices = output
            probabilities = torch.softmax(logits.detach().float(), dim=-1)
            native_values, native_indices = torch.topk(
                probabilities, module.top_k, dim=-1
            )
            native_weights = native_values / native_values.sum(dim=-1, keepdim=True)
            candidate = weights.detach().float()
            reference = native_weights.to(weights.dtype).float()
            state["layers"] += 1
            state["rows"] += int(candidate.shape[0])
            difference = candidate - reference
            state["weight_squared_error"] += float(difference.square().sum())
            state["weight_squared_reference"] += float(reference.square().sum())
            selected = probabilities.gather(1, indices)
            state["threshold_violations"] += int(
                (selected < native_values[:, -1:]).sum().item()
            )
            id_difference = native_indices != indices
            state["id_differences"] += int(id_difference.sum().item())
            state["rows_with_id_differences"] += int(
                id_difference.any(dim=-1).sum().item()
            )

        return hook

    for _name, module in model.named_modules():
        if isinstance(module, Qwen4ExpTextTopKRouter):
            handles.append(module.register_forward_hook(make_hook(module)))
    return state, handles


def audit_adapter_injection(model: torch.nn.Module) -> dict[str, Any]:
    get_base_model = getattr(model, "get_base_model", None)
    base = get_base_model() if callable(get_base_model) else model
    ordinary = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, FastLoraLinear)
    ]
    native_ordinary = [
        (name, module)
        for name, module in ordinary
        if isinstance(module, FastGgufLoraLinear) and module.uses_packed_mmq()
    ]
    experts = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, Qwen4ExpGgufMoeLora)
    ]
    wrapped = {
        name
        for name, module in model.named_modules()
        if isinstance(module, FastLoraLinear | Qwen4ExpGgufMoeLora)
    }
    targeted = {name for name, _ in model.named_modules() if is_qwen4_exp_target(name)}
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    errors = []
    counts = {
        "ordinary_wrappers": len(ordinary),
        "native_mmq_ordinary_wrappers": len(native_ordinary),
        "generic_packed_ordinary_wrappers": len(ordinary) - len(native_ordinary),
        "expert_wrappers": len(experts),
        "trainable_tensors": len(trainable),
        "trainable_parameters": sum(parameter.numel() for _, parameter in trainable),
        "targeted_modules": len(targeted),
    }
    expected = {
        "ordinary_wrappers": EXPECTED_ORDINARY_WRAPPERS,
        "native_mmq_ordinary_wrappers": EXPECTED_NATIVE_MMQ_WRAPPERS,
        "generic_packed_ordinary_wrappers": (
            EXPECTED_ORDINARY_WRAPPERS - EXPECTED_NATIVE_MMQ_WRAPPERS
        ),
        "expert_wrappers": EXPECTED_EXPERT_WRAPPERS,
        "trainable_tensors": EXPECTED_ADAPTER_TENSORS,
        "trainable_parameters": EXPECTED_RANK4_PARAMETERS,
        "targeted_modules": EXPECTED_ORDINARY_WRAPPERS + EXPECTED_EXPERT_WRAPPERS,
    }
    errors.extend(
        f"{key}: expected {value}, found {counts[key]}"
        for key, value in expected.items()
        if counts[key] != value
    )
    missed = sorted(targeted - wrapped)
    extra = sorted(wrapped - targeted)
    if missed:
        errors.append(f"targeted modules without an adapter: {missed[:8]}")
    if extra:
        errors.append(f"adapters outside the target pattern: {extra[:8]}")
    implementation = getattr(base.config, "_experts_implementation", None)
    if implementation != QWEN4_EXP_EXPERTS_IMPLEMENTATION:
        errors.append(
            f"experts implementation is {implementation!r}, "
            f"expected {QWEN4_EXP_EXPERTS_IMPLEMENTATION!r}"
        )
    invalid = [name for name, _ in trainable if ".lora_" not in name]
    forbidden = [
        name
        for name, _ in trainable
        if any(fragment in name for fragment in _FORBIDDEN_TRAINABLE_FRAGMENTS)
    ]
    non_bf16 = [
        name for name, parameter in trainable if parameter.dtype != torch.bfloat16
    ]
    non_cuda = [
        name for name, parameter in trainable if parameter.device.type != "cuda"
    ]
    packed_trainable = [
        name
        for name, parameter in model.named_parameters()
        if isinstance(parameter, GgufQuantizedParameter) and parameter.requires_grad
    ]
    if invalid:
        errors.append(f"non-adapter trainable tensors: {invalid[:8]}")
    if forbidden:
        errors.append(f"frozen-family trainable tensors: {forbidden[:8]}")
    if non_bf16:
        errors.append(f"non-BF16 adapters: {non_bf16[:8]}")
    if non_cuda:
        errors.append(f"adapters outside cuda:0: {non_cuda[:8]}")
    if packed_trainable:
        errors.append(f"trainable packed parameters: {packed_trainable[:8]}")
    if errors:
        raise RuntimeError("Qwen4-Exp adapter audit failed: " + "; ".join(errors))
    return counts | {
        "adapter_dtypes": ["torch.bfloat16"],
        "device": "cuda:0",
        "experts_implementation": implementation,
    }


def load_fixed_batch(
    dataset_dir: Path,
    *,
    row_start: int,
    batch_size: int,
    sequence_length: int,
) -> tuple[Dataset, dict[str, torch.Tensor], dict[str, Any]]:
    if sequence_length <= 1 or sequence_length > 2048:
        raise ValueError(
            "The fixed Qwen dataset supports sequence lengths in [2,2048]."
        )
    dataset = load_from_disk(str(dataset_dir))
    if not isinstance(dataset, Dataset):
        raise TypeError(f"expected a Dataset at {dataset_dir}, got DatasetDict")
    if row_start < 0 or row_start + batch_size > len(dataset):
        raise IndexError(
            f"requested rows [{row_start},{row_start + batch_size}) outside "
            f"dataset of {len(dataset)} rows"
        )
    selected = dataset.select(range(row_start, row_start + batch_size))
    rows = [selected[index] for index in range(batch_size)]
    input_ids = torch.tensor(
        [row["input_ids"][:sequence_length] for row in rows],
        dtype=torch.int64,
        device="cuda:0",
    )
    valid_lengths = torch.tensor(
        [min(int(row["num_tokens"]), sequence_length) for row in rows],
        device=input_ids.device,
    )
    positions = torch.arange(sequence_length, device=input_ids.device).unsqueeze(0)
    attention_mask = (positions < valid_lengths.unsqueeze(1)).long()
    labels = input_ids.masked_fill(attention_mask == 0, -100)
    return (
        selected,
        {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels},
        {
            "dataset_rows": len(dataset),
            "selected_rows": list(range(row_start, row_start + batch_size)),
            "num_tokens": valid_lengths.cpu().tolist(),
            "input_shape": list(input_ids.shape),
        },
    )


def main() -> None:
    args = parse_args()
    if args.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "status": "running",
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "environment": {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "device": torch.cuda.get_device_name(0),
            "pid": os.getpid(),
        },
        "timeline": {},
        "memory_before": memory_snapshot(),
    }

    def persist() -> None:
        report["performance_summary"] = summarize_timeline(report["timeline"])
        args.report_output.write_text(json.dumps(report, indent=2) + "\n")

    persist()
    torch.manual_seed(args.seed)
    report["static_configuration"] = {
        "compiled_gguf_dequant": configure_compiled_gguf_dequantize(),
        "tiled_value_heads": configure_tiled_value_heads(),
        "fla_cache_entries": configure_qwen4_exp_fla(),
        "gdn_bwd_dhu": install_gdn_bwd_dhu(),
        "gdn_bwd_dqkwg": install_gdn_bwd_dqkwg(),
        "gdn_wu_recompute": install_gdn_wu_recompute(),
        "ple_disk": configure_ple_disk_residency(
            checkpoint=args.model_dir / args.gguf_file, enabled=args.ple_on_disk
        ),
        "target_modules_pattern": QWEN4_EXP_TARGET_MODULES_PATTERN,
    }

    def load_model():
        return AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            gguf_file=args.gguf_file,
            gguf_mmap_policy="pread",
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation=None,
            device_map={"": "cuda:0"},
            output_loading_info=True,
        )

    model, loading_info = run_phase(report["timeline"], "load", load_model)
    model.config.use_cache = False
    model.config.output_router_logits = False
    model.config.router_aux_loss_coef = 0.0
    if args.ple_on_disk:
        # Before the inventory: the load audit counts what the model holds, and the disk-backed PLE table is
        # held by the file, so its install has to be visible by then.
        report["ple_disk"] = require_ple_disk_residency(
            model, expected_rows=PLE_TABLE_ROWS, expected_dim=PLE_TABLE_DIM
        )
    report["load_audit"] = audit_loaded_model(
        model, loading_info, args.model_dir / args.gguf_file
    )
    report["ple"] = audit_ple_table(model)
    report["qsa"] = audit_qsa_indexer(model, model.config, args.sequence_length)
    report["ranking"] = configure_fast_moe_ranking(model)
    report["frozen_mmq"] = configure_qwen4_exp_frozen_mmq(model)
    require_complete_qwen4_exp_frozen_mmq(report["frozen_mmq"])
    report["qsa_indexer"] = configure_qwen4_exp_indexer_fast_path(model)
    report["qsa_indexer"]["applied"] = args.sequence_length <= int(
        report["qsa_indexer"]["exhaustive_upto"]
    )
    require_complete_qwen4_exp_indexer(
        report["qsa_indexer"],
        sequence_length=args.sequence_length
        if report["qsa_indexer"]["applied"]
        else None,
    )
    report["qsa_indexer_skip"] = run_phase(
        report["timeline"],
        "qsa_indexer_skip",
        lambda: audit_qsa_indexer_skip(model, model.config, args.sequence_length),
    )
    report["qsa_attention"] = configure_qwen4_exp_qsa_attention(
        model, sequence_length=args.sequence_length
    )
    require_complete_qwen4_exp_qsa_attention(
        report["qsa_attention"], sequence_length=args.sequence_length
    )
    report["qsa_attention_equivalence"] = run_phase(
        report["timeline"],
        "qsa_attention_equivalence",
        lambda: audit_qsa_attention_equivalence(
            model, model.config, args.sequence_length
        ),
    )
    report["tiled_value_heads"] = dict(
        report["static_configuration"]["tiled_value_heads"]
    )
    report["tiled_value_heads"].update(
        require_tiled_value_heads(
            model, expected_gdn_layers=EXPECTED_GATED_DELTA_NET_LAYERS
        )
    )
    report["fused_norms"] = configure_qwen4_exp_fused_norms(model)
    require_complete_qwen4_exp_fused_norms(report["fused_norms"])
    report["fused_norm_equivalence"] = run_phase(
        report["timeline"],
        "fused_norm_equivalence",
        lambda: audit_fused_norm_equivalence(model, args.sequence_length),
    )
    report["hc_norm"] = configure_qwen4_exp_hc_norm(model)
    require_complete_qwen4_exp_hc_norm(report["hc_norm"])
    report["memory_after_load"] = memory_snapshot()
    persist()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=QWEN4_EXP_TARGET_MODULES_PATTERN,
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.0,
        bias="none",
        use_rslora=False,
        init_lora_weights=True,
    )

    def inject_adapters():
        register_qwen4_exp_adapters(lora_config, model)
        return get_peft_model(model, lora_config, autocast_adapter_dtype=False)

    model = run_phase(report["timeline"], "adapter_injection", inject_adapters)
    apply_qwen4_exp_liger_fused_linear_cross_entropy(model)
    report["liger_loss"] = {
        "patched": True,
        "lm_head_quant_type": int(
            cast(GgufQuantizedParameter, model.lm_head.weight).quant_type
        ),
    }
    report["injection_audit"] = audit_adapter_injection(model)
    # The norm weights are frozen by the adapter injection, which is the state the kernel and its
    # predicate expect, so this gate runs after it rather than beside the other patches.
    report["hc_norm_equivalence"] = run_phase(
        report["timeline"],
        "hc_norm_equivalence",
        lambda: audit_hc_norm_equivalence(model, model.config, args.sequence_length),
    )
    initial_b = [
        torch.count_nonzero(parameter.detach())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and ".lora_B" in name
    ]
    initial_b_nonzero = torch.stack(initial_b).cpu()
    report["injection_audit"]["initial_b_tensors"] = len(initial_b)
    report["injection_audit"]["initial_b_nonzero_tensors"] = int(
        torch.count_nonzero(initial_b_nonzero)
    )
    if bool(torch.any(initial_b_nonzero).item()):
        raise RuntimeError("LoRA-B factors must be zero-initialized")

    persist()

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    model.train()
    report["checkpointing"] = audit_training_contract(
        model,
        decoder_type=Qwen4ExpTextDecoderLayer,
        expected_layers=EXPECTED_DECODER_LAYERS,
    ) | {"policy": "per_decoder_layer"}
    report["memory_after_adapters"] = memory_snapshot()

    selected_dataset, batch, data_report = load_fixed_batch(
        args.dataset_dir,
        row_start=args.row_start,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
    )
    report["data"] = data_report
    # The trainer runs this model under autocast (`bf16=True`), so the audit checks the trainer's own
    # activation dtype here rather than only the no-autocast path the measurements use. It runs in
    # the state the trainer has: adapters injected, model in train mode, checkpointing on.
    report["autocast_dtype"] = run_phase(
        report["timeline"],
        "autocast_dtype",
        lambda: audit_autocast_dtypes(model, batch),
    )
    optimizer = run_phase(
        report["timeline"],
        "optimizer_create",
        lambda: bnb.optim.AdamW8bit(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        ),
    )
    report["optimizer"] = audit_optimizer(optimizer, model)
    report["memory_after_optimizer_create"] = memory_snapshot()
    packed_before = representative_packed_state(model)
    report["packed_before"] = packed_before
    # Snapshot the PLE table here rather than at load time: adapter injection freezes the base
    # parameters, and freezing a parameter bumps its version counter.
    ple_before = audit_ple_table(model)["identity"]
    persist()

    # The equivalence gates dequantize a whole dense expert axis per module, which grows the
    # allocator's reserved pool and leaves the machine short of free memory. Release it here, before
    # the first measured step, so the step numbers describe the step and not the audit's own gates.
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    report["memory_before_first_step"] = memory_snapshot()

    require_complete = args.sequence_length == 2048
    optimizer.zero_grad(set_to_none=True)
    ranking_state, ranking_handles = attach_ranking_value_check(model)
    first_output = run_phase(
        report["timeline"], "first_forward", lambda: model(**batch, use_cache=False)
    )
    for handle in ranking_handles:
        handle.remove()
    report["ranking"].update(ranking_state)
    tiled_state = tiled_value_head_report()
    report["tiled_value_heads"]["tiled_rewrites"] = tiled_state["tiled_rewrites"]
    report["tiled_value_heads"]["layers_taking_the_tiled_broadcast"] = tiled_state[
        "layers"
    ]
    if tiled_state["layers"] != EXPECTED_GATED_DELTA_NET_LAYERS:
        raise RuntimeError(
            "the tiled value-head broadcast did not run in every GatedDeltaNet layer: "
            f"{tiled_state['layers']} of {EXPECTED_GATED_DELTA_NET_LAYERS}"
        )
    if ranking_state["layers"] != report["ranking"]["qwen4"]:
        raise RuntimeError(
            f"expected {report['ranking']['qwen4']} routed layers in the selection check, "
            f"saw {ranking_state['layers']}"
        )
    if ranking_state["threshold_violations"]:
        raise RuntimeError(
            "the streaming selection returned an expert below the kth score: "
            f"{ranking_state['threshold_violations']} violations"
        )
    relative_rmse = math.sqrt(
        ranking_state["weight_squared_error"]
        / (ranking_state["weight_squared_reference"] + 1e-12)
    )
    ranking_state["relative_rmse"] = relative_rmse
    if relative_rmse > 3e-2:
        raise RuntimeError(
            "the streaming selection changed a selected routing weight: "
            f"relative RMSE {relative_rmse}"
        )
    persist()
    # The chunked packed Q5_K loss owns the head boundary, so no logits tensor reaches this gate.
    validate_loss_output(first_output, "first forward")
    first_loss = float(first_output.loss.detach())
    run_phase(report["timeline"], "first_backward", first_output.loss.backward)
    report["fla_tuning"] = require_complete_qwen4_exp_fla(
        report["static_configuration"]["fla_cache_entries"]
    )
    first_gradients = summarize_gradients(model, "first_backward")
    validate_first_gradients(first_gradients, require_complete=require_complete)
    report["first_backward"] = {
        "loss": first_loss,
        "gradients": first_gradients,
    }
    del first_output
    persist()

    clip_norm = run_phase(
        report["timeline"],
        "gradient_clip",
        lambda: torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            args.max_grad_norm,
        ),
    )
    if not bool(torch.isfinite(clip_norm).item()):
        raise RuntimeError("gradient clipping returned a nonfinite norm")
    report["clip_norm_before"] = float(clip_norm)
    run_phase(report["timeline"], "optimizer_step", optimizer.step)
    first_update = summarize_b_updates(model)
    expected_unchanged = {
        name for name in first_gradients["missing"] if ".lora_B" in name
    }
    if set(first_update["unchanged"]) != expected_unchanged:
        raise RuntimeError("LoRA-B update coverage differs from gradient coverage")
    if require_complete and first_update["changed_tensors"] != first_update["tensors"]:
        raise RuntimeError(
            f"some LoRA-B tensors did not update: {first_update['unchanged'][:8]}"
        )
    report["first_update"] = first_update
    persist()

    optimizer.zero_grad(set_to_none=True)
    gc.collect()
    second_output = run_phase(
        report["timeline"], "second_forward", lambda: model(**batch, use_cache=False)
    )
    validate_loss_output(second_output, "second forward")
    second_loss = float(second_output.loss.detach())
    run_phase(report["timeline"], "second_backward", second_output.loss.backward)
    second_gradients = summarize_gradients(model, "second_backward")
    validate_second_gradients(second_gradients, require_complete=require_complete)
    report["second_backward"] = {
        "loss": second_loss,
        "gradients": second_gradients,
    }
    del second_output

    packed_after = representative_packed_state(model)
    report["packed_after"] = packed_after
    if packed_before != packed_after:
        raise RuntimeError("representative packed payload identity or checksum changed")
    ple_after = audit_ple_table(model)["identity"]
    report["ple_after"] = ple_after
    if ple_before != ple_after:
        raise RuntimeError("the PLE n-gram table identity changed during the update")
    report["memory_after_measured_steps"] = memory_snapshot()
    persist()

    def complete_update(label: str) -> dict[str, Any]:
        return complete_training_update(
            model,
            optimizer,
            batch,
            label=label,
            max_grad_norm=args.max_grad_norm,
        )

    for step in range(1, args.max_steps):
        report.setdefault("extra_steps", []).append(
            complete_update(f"extra_step_{step}")
        )
        persist()

    if args.profile_output is not None:
        report["profile"] = profile_warmed_training_update(
            model,
            complete_update,
            output_path=args.profile_output,
            metadata={
                "batch_size": args.batch_size,
                "sequence_length": args.sequence_length,
                "rank": args.rank,
                "checkpointing": report["checkpointing"],
                "experts_implementation": QWEN4_EXP_EXPERTS_IMPLEMENTATION,
            },
        )

    del selected_dataset
    if args.ple_on_disk:
        # The counters are per-forward, so they are read now rather than at install time.
        ple_state = disk_state()
        if ple_state is None:
            raise RuntimeError("the PLE disk state disappeared during the run")
        report["ple_disk"] = ple_state.report()
    report["status"] = "passed"
    report["memory_final"] = memory_snapshot()
    persist()
    print(f"QWEN4-EXP GATE PASS: {args.report_output}", flush=True)


if __name__ == "__main__":
    main()
