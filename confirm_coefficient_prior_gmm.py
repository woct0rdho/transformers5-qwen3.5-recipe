#!/usr/bin/env python3
"""Confirm accepted coefficient-prior AITER configurations."""

import argparse
import gc
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

import tune_coefficient_prior_gmm as tuner
from aiter_benchmark_protocol import AdaptiveTimingPolicy, adaptive_pair_timings
from expert_distribution_prior import route_bank_for_routed_rows


def _route_bank_matches(source: dict[str, Any], routes: tuple[Any, ...]) -> bool:
    expected = source.get("prior", {}).get("vectors")
    if not expected:
        return False
    actual = [route.to_mapping() for route in routes]
    return actual == expected


def confirm(
    path: Path,
    repeats: int | None,
    max_samples: int | None,
    min_gain: float,
) -> dict[str, Any]:
    source = json.loads(path.read_text(encoding="utf-8"))
    target = source["target"]
    prior = source["prior"]
    protocol = source["protocol"]
    adaptive = protocol["adaptive"]
    policy = AdaptiveTimingPolicy(**adaptive)
    if repeats is not None:
        policy = replace(policy, min_samples=repeats)
    if max_samples is not None:
        policy = replace(policy, max_samples=max_samples)
    route_count = int(prior["vector_count"])
    if policy.max_samples > route_count:
        raise ValueError("confirmation max samples cannot exceed the source route bank")
    if policy.max_samples < policy.min_samples:
        raise ValueError("confirmation max samples must be at least repeats")
    route_bank = route_bank_for_routed_rows(
        prior["law"],
        target["rows"],
        route_count,
        seed=prior["route_seed"],
    )
    if not _route_bank_matches(source, route_bank):
        raise ValueError(f"route bank does not match source report: {path}")
    transposed = bool(target["transposed_rhs"]) if target["op"] == "gmm" else False
    launch_case = tuner.build_case(
        target["op"],
        target["rows"],
        target["k"],
        target["n"],
        route_bank,
        transposed,
    )
    baseline = source["baseline"]["config"]
    candidate = source["selected"]["config"]
    timing, adaptive_report = adaptive_pair_timings(
        {
            "baseline": lambda: launch_case.launch(baseline),
            "candidate": lambda: launch_case.launch(candidate),
        },
        policy=policy,
        warmup=int(protocol["warmup"]),
        launches_per_sample=int(protocol["launches_per_sample"]),
        flops=2 * target["rows"] * target["k"] * target["n"],
        select_sample=launch_case.select_sample,
        sample_capacity=len(route_bank),
    )
    speedup = math.exp(
        float(timing["baseline"]["median_log_ms"])
        - float(timing["candidate"]["median_log_ms"])
    )
    correctness = tuner.correctness_check(
        target["op"],
        baseline,
        candidate,
        target["k"],
        target["n"],
        transposed,
        experts=len(route_bank[0].group_sizes),
    )
    passed = (
        candidate != baseline
        and correctness["bitwise_equal"]
        and speedup > 1.0 + min_gain
    )
    result = {
        "source": str(path),
        "target": target,
        "prior": {
            "law": prior["law"],
            "route_seed": prior["route_seed"],
            "route_bank_match": True,
            "vector_count": len(route_bank),
        },
        "protocol": {
            "adaptive": policy.to_mapping(),
            "warmup": int(protocol["warmup"]),
            "launches_per_sample": int(protocol["launches_per_sample"]),
        },
        "baseline_config": baseline,
        "candidate_config": candidate,
        "timing": timing,
        "adaptive_report": adaptive_report,
        "speedup": speedup,
        "correctness": correctness,
        "passed": passed,
    }
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--min-gain", type=float, default=0.02)
    args = parser.parse_args()
    if args.repeats is not None and args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    torch.cuda.set_device(0)
    paths = []
    for path in sorted(args.input_dir.glob("*.json")):
        if path.name == "campaign_manifest.json":
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        if (
            isinstance(report, dict)
            and report.get("accepted")
            and "prior" in report
            and "protocol" in report
        ):
            paths.append(path)
    results = []
    for index, path in enumerate(paths, start=1):
        print(f"START {index}/{len(paths)} {path.name}", flush=True)
        result = confirm(path, args.repeats, args.max_samples, args.min_gain)
        results.append(result)
        print(
            f"{path.name}: {result['speedup']:.4f}x passed={result['passed']}",
            flush=True,
        )
    report = {
        "method": "single-prior-route-bank-adaptive-paired-confirmation",
        "repeats_override": args.repeats,
        "max_samples_override": args.max_samples,
        "min_gain": args.min_gain,
        "candidate_count": len(results),
        "passed_count": sum(item["passed"] for item in results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"candidates={report['candidate_count']} passed={report['passed_count']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
