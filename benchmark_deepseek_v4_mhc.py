"""Standalone DeepSeek V4 mHC forward/backward benchmark.

This intentionally benchmarks only the component boundary requested by
``deepseek_v4_liger_mhc``. It does not load the model or collect a full training
profile.
"""

import argparse
import gc
import json
import statistics
import time
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F

from deepseek_v4_liger_mhc import deepseek_v4_mhc_merge, deepseek_v4_mhc_prepare

_HC = 4
_HIDDEN = 4096
_FLAT = _HC * _HIDDEN
_MIX = 24
_EPS = 1e-6


def _reference(
    x: torch.Tensor,
    branch: torch.Tensor,
    fn: torch.Tensor,
    base: torch.Tensor,
    scale: torch.Tensor,
):
    rows = x.numel() // _FLAT
    flat = x.reshape(rows, _FLAT).float()
    invr = torch.rsqrt(flat.square().mean(dim=-1, keepdim=True) + _EPS)
    mix = F.linear(flat, fn) * invr
    pre = torch.sigmoid(mix[:, :_HC] * scale[0] + base[:_HC]) + _EPS
    post = 2.0 * torch.sigmoid(mix[:, _HC : 2 * _HC] * scale[1] + base[_HC : 2 * _HC])
    comb = (
        torch.softmax(
            (mix[:, 2 * _HC :] * scale[2] + base[2 * _HC :]).view(rows, _HC, _HC),
            dim=-1,
        )
        + _EPS
    )
    comb = comb / (comb.sum(dim=-2, keepdim=True) + _EPS)
    for _ in range(19):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + _EPS)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + _EPS)
    collapsed = (pre.unsqueeze(-1) * x.reshape(rows, _HC, _HIDDEN).float()).sum(dim=1)
    merged = post.unsqueeze(-1) * branch.reshape(rows, _HIDDEN).float().unsqueeze(
        1
    ) + torch.matmul(comb.transpose(-1, -2), x.reshape(rows, _HC, _HIDDEN).float())
    return collapsed.to(x.dtype), merged.to(x.dtype)


def _measure(
    forward: Callable,
    x_data: torch.Tensor,
    branch_data: torch.Tensor,
    grad_collapsed: torch.Tensor,
    grad_merged: torch.Tensor,
    *,
    warmup: int,
    repetitions: int,
) -> dict[str, Any]:
    def update():
        x = x_data.clone().requires_grad_()
        branch = branch_data.clone().requires_grad_()
        merged_cotangent = grad_merged.clone()
        collapsed, merged = forward(x, branch)
        torch.autograd.backward(
            (collapsed, merged),
            (grad_collapsed, merged_cotangent),
        )
        return x, branch, collapsed, merged, merged_cotangent

    for _ in range(warmup):
        values = update()
        del values
    torch.cuda.synchronize()

    forward_samples = []
    backward_samples = []
    for _ in range(repetitions):
        x = x_data.clone().requires_grad_()
        branch = branch_data.clone().requires_grad_()
        merged_cotangent = grad_merged.clone()
        torch.cuda.synchronize()
        start = time.perf_counter()
        collapsed, merged = forward(x, branch)
        torch.cuda.synchronize()
        forward_samples.append((time.perf_counter() - start) * 1000)
        start = time.perf_counter()
        torch.autograd.backward(
            (collapsed, merged),
            (grad_collapsed, merged_cotangent),
        )
        torch.cuda.synchronize()
        backward_samples.append((time.perf_counter() - start) * 1000)
        del x, branch, collapsed, merged, merged_cotangent

    # Peak instrumentation perturbs short forward timings on ROCm, so collect
    # it in a separate untimed update after the warm timing loop.
    gc.collect()
    x = x_data.clone().requires_grad_()
    branch = branch_data.clone().requires_grad_()
    merged_cotangent = grad_merged.clone()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    collapsed, merged = forward(x, branch)
    torch.autograd.backward(
        (collapsed, merged),
        (grad_collapsed, merged_cotangent),
    )
    torch.cuda.synchronize()
    peak_increment_mib = (torch.cuda.max_memory_allocated() - baseline) / (1024 * 1024)
    del x, branch, collapsed, merged, merged_cotangent

    totals = [
        forward_ms + backward_ms
        for forward_ms, backward_ms in zip(
            forward_samples, backward_samples, strict=True
        )
    ]
    return {
        "forward_ms": statistics.median(forward_samples),
        "backward_ms": statistics.median(backward_samples),
        "update_ms": statistics.median(totals),
        "peak_increment_mib": peak_increment_mib,
        "forward_samples_ms": forward_samples,
        "backward_samples_ms": backward_samples,
    }


def _result(
    forward: Callable,
    x_data: torch.Tensor,
    branch_data: torch.Tensor,
    grad_collapsed: torch.Tensor,
    grad_merged: torch.Tensor,
):
    x = x_data.detach().clone().requires_grad_()
    branch = branch_data.detach().clone().requires_grad_()
    collapsed, merged = forward(x, branch)
    torch.autograd.backward(
        (collapsed, merged),
        (grad_collapsed, grad_merged.clone()),
    )
    return collapsed.detach(), merged.detach(), x.grad.detach(), branch.grad.detach()


def _metric(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    candidate = candidate.float().flatten()
    reference = reference.float().flatten()
    delta = candidate - reference
    return {
        "cosine": float(F.cosine_similarity(candidate, reference, dim=0)),
        "relative_rmse": float(
            delta.square().mean().sqrt() / (reference.square().mean().sqrt() + 1e-12)
        ),
        "max_abs": float(delta.abs().max()),
    }


def _saved_state(forward: Callable, x_data: torch.Tensor, branch_data: torch.Tensor):
    records = []

    def pack(tensor):
        records.append(tensor)
        return tensor

    x = x_data.clone().requires_grad_()
    branch = branch_data.clone().requires_grad_()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        collapsed, merged = forward(x, branch)
    unique = {}
    for tensor in records:
        storage = tensor.untyped_storage()
        unique[storage.data_ptr()] = max(
            unique.get(storage.data_ptr(), 0), storage.nbytes()
        )
    result = {
        "occurrences": len(records),
        "occurrence_mib": sum(
            tensor.numel() * tensor.element_size() for tensor in records
        )
        / (1024 * 1024),
        "unique_storage_mib": sum(unique.values()) / (1024 * 1024),
        "shapes": [
            {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
            for tensor in records
        ],
    }
    del x, branch, collapsed, merged
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", nargs="+", type=int, default=[2048, 8192, 32768])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--output")
    args = parser.parse_args()
    unsupported = set(args.rows) - {2048, 8192, 32768}
    if unsupported:
        parser.error(f"unsupported row counts: {sorted(unsupported)}")

    torch.manual_seed(2026)
    fn_f16 = (torch.randn(_MIX, _FLAT, device="cuda", dtype=torch.float32) * 1e-4).to(
        torch.float16
    )
    fn_fp32 = fn_f16.float()
    base = (torch.randn(_MIX, device="cuda") * 0.1).float().contiguous()
    scale = (torch.randn(3, device="cuda") * 0.1).float().contiguous()
    report = {
        "device": torch.cuda.get_device_properties(0).gcnArchName,
        "fn_source_dtype": str(fn_f16.dtype),
        "rows": {},
    }

    for rows in args.rows:
        x_data = torch.randn(rows, _HC, _HIDDEN, device="cuda", dtype=torch.bfloat16)
        branch_data = torch.randn(rows, _HIDDEN, device="cuda", dtype=torch.bfloat16)
        grad_collapsed = torch.randn_like(branch_data)
        grad_merged = torch.randn_like(x_data)

        def candidate(x, branch):
            residual, coefficients, collapsed = deepseek_v4_mhc_prepare(
                x, fn_f16, base, scale
            )
            return collapsed, deepseek_v4_mhc_merge(residual, branch, coefficients)

        def reference(x, branch):
            return _reference(x, branch, fn_fp32, base, scale)

        candidate_bench = _measure(
            candidate,
            x_data,
            branch_data,
            grad_collapsed,
            grad_merged,
            warmup=args.warmup,
            repetitions=args.repetitions,
        )
        reference_bench = _measure(
            reference,
            x_data,
            branch_data,
            grad_collapsed,
            grad_merged,
            warmup=args.warmup,
            repetitions=max(2, args.repetitions // 2),
        )
        candidate_values = _result(
            candidate, x_data, branch_data, grad_collapsed, grad_merged
        )
        reference_values = _result(
            reference, x_data, branch_data, grad_collapsed, grad_merged
        )
        names = ("collapsed", "merged", "grad_x", "grad_branch")
        accuracy = {
            name: _metric(candidate_value, reference_value)
            for name, candidate_value, reference_value in zip(
                names, candidate_values, reference_values, strict=True
            )
        }
        row_report = {
            "candidate": candidate_bench,
            "reference": reference_bench,
            "speedup": reference_bench["update_ms"] / candidate_bench["update_ms"],
            "peak_reduction_mib": reference_bench["peak_increment_mib"]
            - candidate_bench["peak_increment_mib"],
            "accuracy": accuracy,
        }
        if rows == 2048:
            row_report["candidate_saved_state"] = _saved_state(
                candidate, x_data, branch_data
            )
        report["rows"][str(rows)] = row_report
        print(json.dumps({str(rows): row_report}, indent=2), flush=True)
        del (
            x_data,
            branch_data,
            grad_collapsed,
            grad_merged,
            candidate_values,
            reference_values,
        )
        gc.collect()
        torch.cuda.empty_cache()

    if args.output:
        with open(args.output, "w") as handle:
            json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
