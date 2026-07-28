#!/usr/bin/env python3

import argparse
import gc
import json
import math
import time

import torch

from deepseek_v4_csa import deepseek_v4_csa_attention, deepseek_v4_csa_compress

_SEQUENCE_LENGTH = 2048
_QUERY_HEADS = 64
_HEAD_DIM = 512
_WINDOW = 128
_COMPRESSED_LENGTH = 512


def metrics(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    candidate_f32 = candidate.detach().float().flatten()
    reference_f32 = reference.detach().float().flatten()
    difference = candidate_f32 - reference_f32
    return {
        "relative_rmse": float(
            difference.square().mean().sqrt()
            / reference_f32.square().mean().sqrt().clamp_min(1e-12)
        ),
        "cosine": float(
            torch.nn.functional.cosine_similarity(candidate_f32, reference_f32, dim=0)
        ),
        "max_abs": float(difference.abs().max()),
    }


def reference_csa_attention(
    query: torch.Tensor,
    local_kv: torch.Tensor,
    compressed_kv: torch.Tensor,
    sink: torch.Tensor,
    block_m: int = 32,
) -> torch.Tensor:
    """Blockwise FP32 oracle. Inputs are BHSD and output is BSHD."""
    outputs = []
    scale = 1.0 / math.sqrt(_HEAD_DIM)
    for query_start in range(0, _SEQUENCE_LENGTH, block_m):
        query_end = min(query_start + block_m, _SEQUENCE_LENGTH)
        rows = torch.arange(query_start, query_end, device=query.device)
        local_start = max(0, query_start - (_WINDOW - 1))
        local_end = query_end
        query_block = query[:, :, query_start:query_end].float()
        local_block = local_kv[:, :, local_start:local_end].float()
        local_positions = torch.arange(local_start, local_end, device=query.device)
        local_scores = torch.matmul(query_block, local_block.transpose(-1, -2)) * scale
        local_visible = (local_positions[None, :] <= rows[:, None]) & (
            local_positions[None, :] > rows[:, None] - _WINDOW
        )
        local_scores = local_scores.masked_fill(
            ~local_visible[None, None], float("-inf")
        )

        compressed_end = query_end // 4
        compressed_block = compressed_kv[:, :, :compressed_end].float()
        compressed_scores = (
            torch.matmul(query_block, compressed_block.transpose(-1, -2)) * scale
        )
        compressed_positions = torch.arange(compressed_end, device=query.device)
        compressed_visible = compressed_positions[None, :] < (rows[:, None] + 1) // 4
        compressed_scores = compressed_scores.masked_fill(
            ~compressed_visible[None, None], float("-inf")
        )
        sink_logits = sink[None, :, None, None].expand(
            query.shape[0], -1, query_end - query_start, -1
        )
        probabilities = torch.softmax(
            torch.cat((local_scores, compressed_scores, sink_logits), dim=-1),
            dim=-1,
        )
        local_probabilities = probabilities[..., : local_scores.shape[-1]]
        compressed_probabilities = probabilities[
            ...,
            local_scores.shape[-1] : local_scores.shape[-1]
            + compressed_scores.shape[-1],
        ]
        output = torch.matmul(
            local_probabilities.to(torch.bfloat16),
            local_kv[:, :, local_start:local_end],
        )
        output += torch.matmul(
            compressed_probabilities.to(torch.bfloat16),
            compressed_kv[:, :, :compressed_end],
        )
        outputs.append(output.transpose(1, 2))
    return torch.cat(outputs, dim=1)


def reference_csa_compress(
    kv: torch.Tensor,
    gate: torch.Tensor,
    position_bias: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rms_eps: float,
) -> torch.Tensor:
    batch = kv.shape[0]
    kv_windows = kv.view(batch, _COMPRESSED_LENGTH, 4, 2 * _HEAD_DIM)
    gate_windows = (
        gate.view(batch, _COMPRESSED_LENGTH, 4, 2 * _HEAD_DIM) + position_bias
    ).float()
    overlap_kv = torch.zeros(
        batch,
        _COMPRESSED_LENGTH,
        8,
        _HEAD_DIM,
        device=kv.device,
        dtype=kv.dtype,
    )
    overlap_gate = torch.full(
        overlap_kv.shape,
        float("-inf"),
        device=kv.device,
        dtype=torch.float32,
    )
    overlap_kv[:, :, 4:] = kv_windows[..., _HEAD_DIM:]
    overlap_gate[:, :, 4:] = gate_windows[..., _HEAD_DIM:]
    overlap_kv[:, 1:, :4] = kv_windows[:, :-1, :, :_HEAD_DIM]
    overlap_gate[:, 1:, :4] = gate_windows[:, :-1, :, :_HEAD_DIM]
    probabilities = overlap_gate.softmax(dim=2)
    pooled = (overlap_kv.float() * probabilities).sum(dim=2)
    normalized = pooled * torch.rsqrt(
        pooled.square().mean(dim=-1, keepdim=True) + rms_eps
    )
    normalized *= weight.float()

    positions = torch.arange(_COMPRESSED_LENGTH, device=kv.device) * 4
    cos_values = cos[0, positions].float()
    sin_values = sin[0, positions].float()
    rope = normalized[..., -64:].reshape(batch, _COMPRESSED_LENGTH, 32, 2)
    even, odd = rope[..., 0], rope[..., 1]
    rotated = torch.stack(
        (
            even * cos_values - odd * sin_values,
            odd * cos_values + even * sin_values,
        ),
        dim=-1,
    ).flatten(-2)
    return torch.cat((normalized[..., :-64], rotated), dim=-1).to(torch.bfloat16)


def make_rope(batch: int, device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(_SEQUENCE_LENGTH, device=device, dtype=torch.float32)[
        :, None
    ]
    frequencies = torch.exp(
        -torch.arange(32, device=device, dtype=torch.float32)[None, :] / 32
    )
    cos = torch.cos(positions * frequencies).to(torch.bfloat16).unsqueeze(0)
    sin = torch.sin(positions * frequencies).to(torch.bfloat16).unsqueeze(0)
    if batch == 1:
        return cos, sin
    return cos.expand(batch, -1, -1), sin.expand(batch, -1, -1)


def make_inputs(batch: int, seed: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed + batch)
    query = torch.randn(
        batch,
        _QUERY_HEADS,
        _SEQUENCE_LENGTH,
        _HEAD_DIM,
        device="cuda",
        dtype=torch.bfloat16,
    )
    local_kv = torch.randn(
        batch,
        1,
        _SEQUENCE_LENGTH,
        _HEAD_DIM,
        device="cuda",
        dtype=torch.bfloat16,
    )
    compressed_kv = torch.randn(
        batch,
        1,
        _COMPRESSED_LENGTH,
        _HEAD_DIM,
        device="cuda",
        dtype=torch.bfloat16,
    )
    sink = torch.randn(_QUERY_HEADS, device="cuda", dtype=torch.float32) * 0.2
    compressor_kv = torch.randn(
        batch,
        _SEQUENCE_LENGTH,
        2 * _HEAD_DIM,
        device="cuda",
        dtype=torch.bfloat16,
    )
    compressor_gate = torch.randn_like(compressor_kv) * 0.5
    position_bias = (
        torch.randn(4, 2 * _HEAD_DIM, device="cuda", dtype=torch.bfloat16) * 0.2
    )
    weight = torch.randn(_HEAD_DIM, device="cuda", dtype=torch.bfloat16) * 0.5
    cos, sin = make_rope(batch)
    return {
        "query": query,
        "local_kv": local_kv,
        "compressed_kv": compressed_kv,
        "sink": sink,
        "compressor_kv": compressor_kv,
        "compressor_gate": compressor_gate,
        "position_bias": position_bias,
        "weight": weight,
        "cos": cos,
        "sin": sin,
    }


def run_correctness(batch: int, seed: int) -> dict[str, object]:
    values = make_inputs(batch, seed)
    output_gradient = torch.randn_like(values["query"]).transpose(1, 2) * 0.03
    candidate_inputs = tuple(
        values[name].clone().requires_grad_()
        for name in ("query", "local_kv", "compressed_kv", "sink")
    )
    candidate_output = deepseek_v4_csa_attention(*candidate_inputs)
    candidate_gradients = torch.autograd.grad(
        candidate_output, candidate_inputs, output_gradient
    )
    reference_inputs = tuple(
        tensor.detach().clone().requires_grad_() for tensor in candidate_inputs
    )
    reference_output = reference_csa_attention(*reference_inputs)
    reference_gradients = torch.autograd.grad(
        reference_output, reference_inputs, output_gradient
    )

    producer_gradient = (
        torch.randn(
            batch,
            1,
            _COMPRESSED_LENGTH,
            _HEAD_DIM,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.03
    )
    candidate_producer_inputs = tuple(
        values[name].clone().requires_grad_()
        for name in ("compressor_kv", "compressor_gate")
    )
    candidate_compressed = deepseek_v4_csa_compress(
        *candidate_producer_inputs,
        values["position_bias"],
        values["weight"],
        values["cos"],
        values["sin"],
        1e-6,
    )
    candidate_producer_gradients = torch.autograd.grad(
        candidate_compressed, candidate_producer_inputs, producer_gradient
    )
    reference_producer_inputs = tuple(
        tensor.detach().clone().requires_grad_() for tensor in candidate_producer_inputs
    )
    reference_compressed = reference_csa_compress(
        *reference_producer_inputs,
        values["position_bias"],
        values["weight"],
        values["cos"],
        values["sin"],
        1e-6,
    )
    reference_producer_gradients = torch.autograd.grad(
        reference_compressed,
        reference_producer_inputs,
        producer_gradient.squeeze(1),
    )
    torch.cuda.synchronize()

    report = {"attention_output": metrics(candidate_output, reference_output)}
    for name, candidate, reference in zip(
        (
            "query_gradient",
            "local_kv_gradient",
            "compressed_kv_gradient",
            "sink_gradient",
        ),
        candidate_gradients,
        reference_gradients,
    ):
        report[name] = metrics(candidate, reference)
    report["producer_output"] = metrics(
        candidate_compressed.squeeze(1), reference_compressed
    )
    for name, candidate, reference in zip(
        ("producer_kv_gradient", "producer_gate_gradient"),
        candidate_producer_gradients,
        reference_producer_gradients,
    ):
        report[name] = metrics(candidate, reference)
    del values
    gc.collect()
    torch.cuda.empty_cache()
    return report


def median_ms(function, iterations: int) -> float:
    samples = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        function()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    return float(torch.tensor(samples).median())


def run_benchmark(
    batch: int, seed: int, warmup: int, iterations: int
) -> dict[str, float]:
    values = make_inputs(batch, seed)
    query = values["query"].requires_grad_()
    local_kv = values["local_kv"].requires_grad_()
    sink = values["sink"].requires_grad_()
    compressor_kv = values["compressor_kv"].requires_grad_()
    compressor_gate = values["compressor_gate"].requires_grad_()
    output_gradient = torch.randn_like(query).transpose(1, 2)

    def forward_only() -> None:
        with torch.no_grad():
            compressed = deepseek_v4_csa_compress(
                compressor_kv,
                compressor_gate,
                values["position_bias"],
                values["weight"],
                values["cos"],
                values["sin"],
                1e-6,
            )
            deepseek_v4_csa_attention(query, local_kv, compressed, sink)

    def complete() -> None:
        compressed = deepseek_v4_csa_compress(
            compressor_kv,
            compressor_gate,
            values["position_bias"],
            values["weight"],
            values["cos"],
            values["sin"],
            1e-6,
        )
        output = deepseek_v4_csa_attention(query, local_kv, compressed, sink)
        torch.autograd.grad(
            output,
            (query, local_kv, compressor_kv, compressor_gate, sink),
            output_gradient,
        )

    for _ in range(warmup):
        forward_only()
        complete()
    torch.cuda.synchronize()
    forward_ms = median_ms(forward_only, iterations)
    complete_ms = median_ms(complete, iterations)

    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    complete()
    torch.cuda.synchronize()
    peak_incremental = torch.cuda.max_memory_allocated() - baseline
    return {
        "forward_ms": forward_ms,
        "backward_ms": complete_ms - forward_ms,
        "complete_ms": complete_ms,
        "peak_incremental_mib": peak_incremental / (1024**2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 4, 16])
    parser.add_argument("--correctness-batches", nargs="+", type=int, default=[1])
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = {"correctness": {}, "benchmark": {}}
    for batch in args.correctness_batches:
        report["correctness"][str(batch)] = run_correctness(batch, args.seed)
    for batch in args.batches:
        report["benchmark"][str(batch)] = run_benchmark(
            batch, args.seed, args.warmup, args.iterations
        )
        gc.collect()
        torch.cuda.empty_cache()
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")


if __name__ == "__main__":
    main()
