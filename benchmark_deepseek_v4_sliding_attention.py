#!/usr/bin/env python3

import argparse
import gc
import json
import math
import time

import torch

from deepseek_v4_sliding_attention import deepseek_v4_sliding_attention

_SEQUENCE_LENGTH = 2048
_QUERY_HEADS = 64
_HEAD_DIM = 512
_WINDOW = 128


def reference_sliding_attention(
    query: torch.Tensor,
    shared_kv: torch.Tensor,
    sink: torch.Tensor,
) -> torch.Tensor:
    query_f32 = query.float()
    kv_f32 = shared_kv.float()
    outputs = []
    scale = 1.0 / math.sqrt(_HEAD_DIM)
    for query_start in range(0, _SEQUENCE_LENGTH, _WINDOW):
        query_end = query_start + _WINDOW
        key_start = max(0, query_start - (_WINDOW - 1))
        key_end = query_end
        query_block = query_f32[:, :, query_start:query_end]
        key_block = kv_f32[:, :, key_start:key_end]
        scores = torch.matmul(query_block, key_block.transpose(-1, -2)) * scale
        query_positions = torch.arange(query_start, query_end, device=query.device)[
            :, None
        ]
        key_positions = torch.arange(key_start, key_end, device=query.device)[None, :]
        visible = (key_positions <= query_positions) & (
            key_positions > query_positions - _WINDOW
        )
        scores = scores.masked_fill(~visible[None, None], float("-inf"))
        sink_logits = sink[None, :, None, None].expand(query.shape[0], -1, _WINDOW, -1)
        probabilities = torch.softmax(torch.cat((scores, sink_logits), dim=-1), dim=-1)[
            ..., :-1
        ]
        outputs.append(torch.matmul(probabilities, key_block))
    return torch.cat(outputs, dim=2).transpose(1, 2)


def metrics(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    candidate_f32 = candidate.detach().float().flatten()
    reference_f32 = reference.detach().float().flatten()
    difference = candidate_f32 - reference_f32
    return {
        "relative_rmse": float(
            difference.square().mean().sqrt()
            / (reference_f32.square().mean().sqrt() + 1e-12)
        ),
        "cosine": float(
            torch.nn.functional.cosine_similarity(candidate_f32, reference_f32, dim=0)
        ),
        "max_abs": float(difference.abs().max()),
    }


def make_inputs(batch: int, seed: int):
    torch.manual_seed(seed + batch)
    query = torch.randn(
        batch,
        _QUERY_HEADS,
        _SEQUENCE_LENGTH,
        _HEAD_DIM,
        device="cuda",
        dtype=torch.bfloat16,
    )
    shared_kv = torch.randn(
        batch,
        1,
        _SEQUENCE_LENGTH,
        _HEAD_DIM,
        device="cuda",
        dtype=torch.bfloat16,
    )
    sink = torch.randn(_QUERY_HEADS, device="cuda", dtype=torch.float32)
    output_gradient = torch.randn_like(query).transpose(1, 2)
    return query, shared_kv, sink, output_gradient


def run_correctness(batch: int, seed: int) -> dict[str, object]:
    query, shared_kv, sink, output_gradient = make_inputs(batch, seed)

    candidate_query = query.clone().requires_grad_()
    candidate_kv = shared_kv.clone().requires_grad_()
    candidate_sink = sink.clone().requires_grad_()
    candidate_output = deepseek_v4_sliding_attention(
        candidate_query,
        candidate_kv,
        candidate_sink,
    )
    candidate_gradients = torch.autograd.grad(
        candidate_output,
        (candidate_query, candidate_kv, candidate_sink),
        output_gradient,
    )

    reference_query = query.clone().requires_grad_()
    reference_kv = shared_kv.clone().requires_grad_()
    reference_sink = sink.clone().requires_grad_()
    reference_output = reference_sliding_attention(
        reference_query,
        reference_kv,
        reference_sink,
    )
    reference_gradients = torch.autograd.grad(
        reference_output,
        (reference_query, reference_kv, reference_sink),
        output_gradient.float(),
    )
    torch.cuda.synchronize()

    report = {"output": metrics(candidate_output, reference_output)}
    for name, candidate, reference in zip(
        ("query_gradient", "shared_kv_gradient", "sink_gradient"),
        candidate_gradients,
        reference_gradients,
    ):
        report[name] = metrics(candidate, reference)
    del (
        query,
        shared_kv,
        sink,
        output_gradient,
        candidate_query,
        candidate_kv,
        candidate_sink,
        candidate_output,
        candidate_gradients,
        reference_query,
        reference_kv,
        reference_sink,
        reference_output,
        reference_gradients,
    )
    gc.collect()
    torch.cuda.empty_cache()
    return report


def elapsed_ms(function, iterations: int) -> float:
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        function()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000.0 / iterations


def run_benchmark(
    batch: int, seed: int, warmup: int, iterations: int
) -> dict[str, float]:
    query, shared_kv, sink, output_gradient = make_inputs(batch, seed)
    query.requires_grad_()
    shared_kv.requires_grad_()
    sink.requires_grad_()

    def forward_only() -> None:
        with torch.no_grad():
            deepseek_v4_sliding_attention(query, shared_kv, sink)

    def complete() -> None:
        output = deepseek_v4_sliding_attention(query, shared_kv, sink)
        torch.autograd.grad(
            output,
            (query, shared_kv, sink),
            output_gradient,
        )

    for _ in range(warmup):
        forward_only()
        complete()
    torch.cuda.synchronize()

    forward_ms = elapsed_ms(forward_only, iterations)
    complete_ms = elapsed_ms(complete, iterations)

    torch.cuda.synchronize()
    baseline_allocated = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    complete()
    torch.cuda.synchronize()
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    return {
        "forward_ms": forward_ms,
        "forward_backward_ms": complete_ms,
        "incremental_peak_allocated_mib": (peak_allocated - baseline_allocated) / 2**20,
        "incremental_peak_reserved_mib": (peak_reserved - baseline_reserved) / 2**20,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", default="1,4,16")
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    batches = tuple(int(value) for value in args.batches.split(","))
    if not batches or any(batch not in (1, 4, 16) for batch in batches):
        raise ValueError("--batches must contain only 1, 4, and 16")

    device = torch.cuda.get_device_properties(0)
    report: dict[str, object] = {
        "environment": {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "device": device.name,
            "arch": device.gcnArchName,
        },
        "batches": {},
    }
    for batch in batches:
        entry: dict[str, object] = {}
        if not args.skip_correctness:
            entry["correctness"] = run_correctness(batch, args.seed)
            print(
                f"batch {batch} correctness",
                json.dumps(entry["correctness"]),
                flush=True,
            )
        entry["benchmark"] = run_benchmark(
            batch, args.seed, args.warmup, args.iterations
        )
        report["batches"][str(batch)] = entry
        print(f"batch {batch} benchmark", json.dumps(entry["benchmark"]), flush=True)
        gc.collect()
        torch.cuda.empty_cache()

    serialized = json.dumps(report, indent=2) + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output_file:
            output_file.write(serialized)
    print(serialized, end="")


if __name__ == "__main__":
    main()
