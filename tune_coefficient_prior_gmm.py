#!/usr/bin/env python3
"""Tune one exact AITER GMM/PTGMM key with coefficient-only route priors.

The script deliberately keeps the search local: it evaluates one field at a time
from the current exact-key config. Group sizes are sampled from the fitted
model-level learned priors and the capture-free DeepSeek hash prior. No route
captures, production-frequency weights, or Cartesian candidate products are
used.
"""

import argparse
import gc
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from aiter.ops.triton.gmm import gmm, ptgmm

from aiter_benchmark_protocol import AdaptiveTimingPolicy, adaptive_pair_timings
from expert_distribution_prior import (
    EXPERT_PRIOR_NAMES,
    ExpertPrior,
    RouteVector,
    expert_prior_experts,
    expert_prior_metadata,
    route_bank_for_routed_rows,
)
from moe_gmm_configs import gmm_config, ptgmm_config

FIELDS = (
    "BLOCK_SIZE_M",
    "BLOCK_SIZE_K",
    "BLOCK_SIZE_N",
    "GROUP_SIZE",
    "GRID_DIM",
    "num_warps",
    "num_stages",
)

FAMILIES = {
    "deepseek": {
        "rows": {1: 12288, 4: 49152, 16: 196608},
        "gmm_base": (
            (4096, 2048, True),
            (2048, 4096, True),
            (2048, 4096, False),
            (4096, 2048, False),
        ),
        "gmm_lora": (
            (4096, 4, True),
            (2048, 4, True),
            (4, 4096, True),
            (4, 4096, False),
            (4, 2048, False),
            (4096, 4, False),
        ),
        "ptgmm_base": (
            (4096, 2048),
            (2048, 4096),
        ),
        "ptgmm_lora": (
            (4096, 4),
            (2048, 4),
            (4, 4096),
        ),
    },
    "qwen": {
        "rows": {1: 16384, 4: 65536, 16: 262144},
        "gmm_base": (
            (2048, 512, True),
            (512, 2048, True),
            (512, 2048, False),
            (2048, 512, False),
        ),
        "gmm_lora": (
            (2048, 4, True),
            (512, 4, True),
            (4, 1024, True),
            (4, 2048, True),
            (4, 2048, False),
            (4, 512, False),
            (1024, 4, False),
            (2048, 4, False),
        ),
        "ptgmm_base": (
            (2048, 512),
            (512, 2048),
        ),
        "ptgmm_lora": (
            (2048, 4),
            (512, 4),
            (4, 1024),
            (4, 2048),
        ),
    },
    "qwen3.8": {
        "rows": {1: 20480, 4: 81920, 16: 327680},
        "gmm_base": (
            (2560, 640, True),
            (640, 2560, True),
            (640, 2560, False),
            (2560, 640, False),
        ),
        "gmm_lora": (
            (2560, 4, True),
            (640, 4, True),
            (4, 1280, True),
            (4, 2560, True),
            (4, 2560, False),
            (4, 640, False),
            (1280, 4, False),
            (2560, 4, False),
        ),
        "ptgmm_base": (
            (2560, 640),
            (640, 2560),
        ),
        "ptgmm_lora": (
            (2560, 4),
            (640, 4),
            (4, 1280),
            (4, 2560),
        ),
    },
}

# One learned or hash law per routed family. The expert count comes from the law.
FAMILY_PRIORS = {
    "qwen": ExpertPrior.QwenLearned.value,
    "qwen3.8": ExpertPrior.Qwen38Learned.value,
    "deepseek": ExpertPrior.DeepSeekLearned.value,
}
PRIOR_FAMILIES = {prior: family for family, prior in FAMILY_PRIORS.items()}
PRIOR_FAMILIES[ExpertPrior.DeepSeekHash.value] = "deepseek"


@dataclass
class LaunchCase:
    routes: tuple[RouteVector, ...]
    groups: tuple[torch.Tensor, ...]
    lhs: torch.Tensor
    rhs: torch.Tensor
    out: torch.Tensor
    op: str
    selected_route: int = 0

    def select_sample(self, index: int) -> None:
        if not 0 <= index < len(self.routes):
            raise IndexError(f"route vector index {index} is outside the route bank")
        self.selected_route = index

    def launch(self, config: dict[str, int]) -> torch.Tensor:
        group_sizes = self.groups[self.selected_route]
        if self.op == "gmm":
            return gmm(
                self.lhs,
                self.rhs,
                group_sizes,
                preferred_element_type=torch.bfloat16,
                existing_out=self.out,
                config=config,
            )
        return ptgmm(
            self.lhs,
            self.rhs,
            group_sizes,
            preferred_element_type=torch.bfloat16,
            existing_out=self.out,
            config=config,
        )


def make_route_bank(
    expert_prior: str,
    rows: int,
    count: int,
    seed: int | None,
) -> tuple[RouteVector, ...]:
    return route_bank_for_routed_rows(expert_prior, rows, count, seed=seed)


def routed_group_median(routes: tuple[RouteVector, ...]) -> float:
    """Median non-empty routed group size, averaged over the route bank."""

    medians = []
    for route in routes:
        sizes = sorted(size for size in route.group_sizes if size)
        if sizes:
            medians.append(sizes[len(sizes) // 2])
    if not medians:
        raise ValueError("route bank has no non-empty expert group")
    return sum(medians) / len(medians)


def config_values(
    op: str,
    field: str,
    k: int,
    n: int,
    m: int,
    current: int,
    experts: int,
    group_median: float,
) -> list[int]:
    """Candidate values for one config field, pruned by kernel geometry.

    The lists are the search space of the tuning protocol. The geometry filters
    only remove values the kernel cannot use well on this shape: an M tile that
    is several times the routed group size spends its rows on padding, a K or N
    tile wider than its dimension has no work, and a one-warp program cannot
    hide the K-loop pipeline. These candidates would be probed out anyway.
    Removing them keeps the shortlist and the recorded provenance honest.
    `num_warps` is not filtered: the two migrated families' measured tables
    contain one-warp winners for rank-small and for B1 base shapes.
    """

    if field == "BLOCK_SIZE_M":
        values = [16, 32, 64, 128, 256]
        if op == "ptgmm" and m // experts >= 512:
            values.append(512)
        cap = max(64, 2 * int(group_median))
        values = [value for value in values if value <= cap]
    elif field == "BLOCK_SIZE_K":
        if k <= 16:
            values = [16]
        elif op == "gmm":
            values = [value for value in (32, 64, 128, 256) if value <= k]
        else:
            values = [value for value in (64, 128, 256, 512) if value <= k]
    elif field == "BLOCK_SIZE_N":
        if n <= 16:
            values = [16, 32]
        elif op == "gmm" and k > 16 and n > 16:
            values = [32, 64, 128, 256]
        else:
            values = [32, 64, 128, 256, 512]
        values = [value for value in values if value <= max(16, n)]
    elif field == "GROUP_SIZE":
        values = [1, 2, 4, 8]
    elif field == "GRID_DIM":
        values = [20, 40, 80, 160, 256]
        if experts > 256:
            # A 512-expert layer owns at least 512 M tiles per N tile, so the persistent grid can
            # be twice as wide as the 256-expert families'.
            values.append(512)
    elif field == "num_warps":
        values = [1, 2, 4, 8]
    elif field == "num_stages":
        values = [1, 2, 3]
    else:
        raise KeyError(field)
    if current not in values:
        values = [current, *values]
    return list(dict.fromkeys(values))


def valid_config(op: str, config: dict[str, int]) -> bool:
    if config["BLOCK_SIZE_K"] * config["BLOCK_SIZE_N"] > 65536:
        return False
    return not (op == "gmm" and config["BLOCK_SIZE_M"] * config["BLOCK_SIZE_N"] > 65536)


def build_case(
    op: str,
    m: int,
    k: int,
    n: int,
    route_bank: tuple[RouteVector, ...],
    transposed: bool = False,
) -> LaunchCase:
    if not route_bank:
        raise ValueError("route bank must not be empty")
    expert_counts = {len(route.group_sizes) for route in route_bank}
    if len(expert_counts) != 1:
        raise ValueError("route bank mixes physical expert counts")
    experts = expert_counts.pop()
    device = "cuda"
    dtype = torch.bfloat16
    generator = torch.Generator(device=device).manual_seed(20260824)
    groups = tuple(
        torch.tensor(route.group_sizes, device=device, dtype=torch.int32)
        for route in route_bank
    )
    if op == "gmm":
        lhs = torch.randn((m, k), generator=generator, device=device, dtype=dtype)
        if transposed:
            rhs_storage = torch.randn(
                (experts, n, k), generator=generator, device=device, dtype=dtype
            )
            rhs = rhs_storage.transpose(1, 2)
        else:
            rhs = torch.randn(
                (experts, k, n), generator=generator, device=device, dtype=dtype
            )
        out = torch.empty((m, n), device=device, dtype=dtype)
        return LaunchCase(route_bank, groups, lhs, rhs, out, op)
    lhs_storage = torch.randn((m, k), generator=generator, device=device, dtype=dtype)
    lhs = lhs_storage.transpose(0, 1)
    rhs = torch.randn((m, n), generator=generator, device=device, dtype=dtype)
    out = torch.empty((experts, k, n), device=device, dtype=dtype)
    return LaunchCase(route_bank, groups, lhs, rhs, out, op)


def benchmark_pair(
    case: LaunchCase,
    baseline: dict[str, int],
    candidate: dict[str, int],
    policy: AdaptiveTimingPolicy,
    warmup: int,
    launches_per_sample: int,
) -> dict[str, Any]:
    if not valid_config(case.op, candidate):
        return {"status": "pruned", "score_ms": math.inf}
    timing, adaptive = adaptive_pair_timings(
        {
            "baseline": lambda: case.launch(baseline),
            "candidate": lambda: case.launch(candidate),
        },
        policy=policy,
        warmup=warmup,
        launches_per_sample=launches_per_sample,
        flops=2 * case.lhs.shape[0] * case.rhs.shape[-1] * case.lhs.shape[1],
        select_sample=case.select_sample,
        sample_capacity=len(case.routes),
    )
    log_speedup = float(timing["baseline"]["median_log_ms"]) - float(
        timing["candidate"]["median_log_ms"]
    )
    return {
        "status": "ok",
        "score_ms": math.exp(float(timing["candidate"]["median_log_ms"])),
        "speedup": math.exp(log_speedup),
        "ci_low_speedup": float(adaptive["rounds"][-1]["ci_low_speedup"]),
        "timing": timing,
        "adaptive": adaptive,
    }


def target_kind_for(family: str, op: str, target: tuple[int, ...]) -> str:
    for target_kind in ("base", "lora"):
        if target in FAMILIES[family][f"{op}_{target_kind}"]:
            return target_kind
    raise ValueError(f"unknown {op} target for {family}: {target}")


def correctness_check(
    op: str,
    baseline: dict[str, int],
    selected: dict[str, int],
    k: int,
    n: int,
    transposed: bool = False,
    *,
    experts: int,
) -> dict[str, Any]:
    device = "cuda"
    dtype = torch.bfloat16
    generator = torch.Generator(device=device).manual_seed(2037)
    small_groups = torch.zeros(experts, device=device, dtype=torch.int32)
    small_groups[0] = 2
    small_groups[7] = 3
    if op == "gmm":
        lhs = torch.randn((5, k), generator=generator, device=device, dtype=dtype)
        if transposed:
            rhs_storage = torch.randn(
                (experts, n, k), generator=generator, device=device, dtype=dtype
            )
            rhs = rhs_storage.transpose(1, 2)
        else:
            rhs = torch.randn(
                (experts, k, n), generator=generator, device=device, dtype=dtype
            )
        out_baseline = gmm(
            lhs,
            rhs,
            small_groups,
            preferred_element_type=dtype,
            config=baseline,
        )
        out_selected = gmm(
            lhs,
            rhs,
            small_groups,
            preferred_element_type=dtype,
            config=selected,
        )
    else:
        lhs_storage = torch.randn(
            (5, k), generator=generator, device=device, dtype=dtype
        )
        lhs = lhs_storage.transpose(0, 1)
        rhs = torch.randn((5, n), generator=generator, device=device, dtype=dtype)
        out_baseline = ptgmm(
            lhs, rhs, small_groups, preferred_element_type=dtype, config=baseline
        )
        out_selected = ptgmm(
            lhs, rhs, small_groups, preferred_element_type=dtype, config=selected
        )
    equal = bool(torch.equal(out_baseline, out_selected))
    delta = (out_baseline.float() - out_selected.float()).abs()
    result = {"bitwise_equal": equal, "max_abs": float(delta.max())}
    del lhs, rhs, out_baseline, out_selected, small_groups
    if "rhs_storage" in locals():
        del rhs_storage
    if "lhs_storage" in locals():
        del lhs_storage
    gc.collect()
    torch.cuda.empty_cache()
    return result


def parse_seed_config(value: str) -> dict[str, int]:
    """Read a replacement seed config from an inline JSON object or a JSON file."""

    text = value
    candidate = Path(value)
    if candidate.exists():
        text = candidate.read_text(encoding="utf-8")
    loaded = json.loads(text)
    if not isinstance(loaded, dict) or set(loaded) != set(FIELDS):
        raise ValueError(f"--seed-config must define exactly: {', '.join(FIELDS)}")
    return {field: int(loaded[field]) for field in FIELDS}


def run(args: argparse.Namespace) -> dict[str, Any]:
    family = PRIOR_FAMILIES[args.expert_prior]
    experts = expert_prior_experts(args.expert_prior)
    rows = FAMILIES[family]["rows"][args.batch]
    final_repeats = args.final_repeats or args.repeats
    final_max_samples = args.final_max_samples or args.max_samples
    final_sample_step = args.final_sample_step or args.sample_step
    route_bank = make_route_bank(
        args.expert_prior,
        rows,
        max(args.max_samples, final_max_samples),
        args.route_seed,
    )
    route_seed = route_bank[0].profile.seed
    if args.op == "gmm":
        target = (args.k, args.n, args.trans)
        current = gmm_config(rows, args.k, args.n, args.trans, args.expert_prior)
    else:
        target = (args.k, args.n)
        current = ptgmm_config(rows, args.k, args.n, args.expert_prior)
    if args.seed_config is not None:
        current = parse_seed_config(args.seed_config)
    target_kind = args.target_kind or target_kind_for(family, args.op, target)
    if target not in FAMILIES[family][f"{args.op}_{target_kind}"]:
        raise ValueError(f"unknown {args.op} {target_kind} target {target}")

    launch_case = build_case(
        args.op,
        rows,
        args.k,
        args.n,
        route_bank,
        args.trans if args.op == "gmm" else False,
    )
    group_median = routed_group_median(route_bank)
    screen_policy = AdaptiveTimingPolicy(
        min_samples=args.repeats,
        max_samples=args.max_samples,
        sample_step=args.sample_step,
        confidence=args.adaptive_confidence,
        epsilon_pct=args.adaptive_epsilon_pct,
        stable_rounds=args.adaptive_stable_rounds,
        noise_floor_pct=args.adaptive_noise_floor_pct,
    )
    final_policy = AdaptiveTimingPolicy(
        min_samples=final_repeats,
        max_samples=final_max_samples,
        sample_step=final_sample_step,
        confidence=args.adaptive_confidence,
        epsilon_pct=args.adaptive_epsilon_pct,
        stable_rounds=args.adaptive_stable_rounds,
        noise_floor_pct=args.adaptive_noise_floor_pct,
    )
    probe_policy = None
    if args.probe_slowdown > 1.0:
        probe_policy = AdaptiveTimingPolicy(
            min_samples=args.probe_samples,
            max_samples=args.probe_max_samples,
            sample_step=args.probe_sample_step,
            confidence=args.adaptive_confidence,
            epsilon_pct=args.probe_epsilon_pct,
            stable_rounds=args.adaptive_stable_rounds,
            noise_floor_pct=args.adaptive_noise_floor_pct,
        )
    evaluated: dict[tuple[tuple[str, int], ...], dict[str, Any]] = {}
    baseline = dict(current)
    # The fastest configuration measured so far and its screen score. The probe uses the
    # configuration as its reference and never the score.
    best_config = dict(baseline)
    best_score = math.inf

    def key(config: dict[str, int]) -> tuple[tuple[str, int], ...]:
        return tuple(sorted(config.items()))

    def evaluate(config: dict[str, int]) -> float:
        nonlocal best_config, best_score
        config = dict(config)
        config_key = key(config)
        if config_key not in evaluated:
            probe = None
            if (
                probe_policy is not None
                and config_key != key(baseline)
                and config_key != key(best_config)
            ):
                # Staged probe: one short paired comparison against the fastest config known so
                # far, under its own timing policy. It removes the catastrophically slow tail and
                # is never pooled with the screen or final timings.
                probe = benchmark_pair(
                    launch_case,
                    best_config,
                    config,
                    probe_policy,
                    args.warmup,
                    args.launches_per_sample,
                )
                screened = probe["status"] == "pruned" or (
                    probe["status"] == "ok"
                    and float(probe.get("ci_low_speedup", math.inf))
                    > args.probe_slowdown
                )
                if screened:
                    evaluated[config_key] = {
                        "config": config,
                        "status": "probed_out",
                        "score_ms": math.inf,
                        "probe": probe,
                    }
                    print(json.dumps(evaluated[config_key], sort_keys=True), flush=True)
                    return math.inf
            measured = benchmark_pair(
                launch_case,
                baseline,
                config,
                screen_policy,
                args.warmup,
                args.launches_per_sample,
            )
            evaluated[config_key] = {"config": config, **measured}
            if probe is not None:
                evaluated[config_key]["probe"] = probe
            print(json.dumps(evaluated[config_key], sort_keys=True), flush=True)
            score = float(evaluated[config_key]["score_ms"])
            if score < best_score:
                best_config = config
                best_score = score
        return float(evaluated[config_key]["score_ms"])

    evaluate(baseline)
    best_score = evaluated[key(baseline)]["score_ms"]
    rounds = []
    for round_index in range(args.rounds):
        round_start = dict(current)
        for field in args.fields:
            choices = []
            for value in config_values(
                args.op,
                field,
                args.k,
                args.n,
                rows,
                current[field],
                experts,
                group_median,
            ):
                candidate = dict(current)
                candidate[field] = value
                choices.append((evaluate(candidate), value, candidate))
            finite = [item for item in choices if math.isfinite(item[0])]
            if not finite:
                continue
            _, _, current = min(finite, key=lambda item: (item[0], item[1]))
        current_score = evaluate(current)
        rounds.append(
            {
                "round": round_index + 1,
                "start": round_start,
                "winner": dict(current),
                "score_ms": current_score,
            }
        )
        if current == round_start:
            break

    final = benchmark_pair(
        launch_case,
        baseline,
        current,
        final_policy,
        args.warmup,
        args.launches_per_sample,
    )
    accepted = (
        current != baseline
        and final["status"] == "ok"
        and float(final["speedup"]) > 1.0 + args.min_gain
    )
    selected = dict(current) if accepted else dict(baseline)
    correctness = correctness_check(
        args.op,
        baseline,
        selected,
        args.k,
        args.n,
        args.trans if args.op == "gmm" else False,
        experts=experts,
    )
    if accepted and not correctness["bitwise_equal"]:
        accepted = False
        selected = dict(baseline)
        correctness = correctness_check(
            args.op,
            baseline,
            selected,
            args.k,
            args.n,
            args.trans if args.op == "gmm" else False,
            experts=experts,
        )

    selected_name = "candidate" if accepted else "baseline"
    selected_score = (
        float(final["timing"][selected_name]["median_log_ms"])
        if final["status"] == "ok"
        else math.inf
    )
    prior = {
        **expert_prior_metadata(args.expert_prior),
        "kind": "coefficient_only",
        "route_seed": route_seed,
        "vector_count": len(route_bank),
        "vectors": [route.to_mapping() for route in route_bank],
        "mixed_deepseek_learned_hash": False,
    }
    report = {
        "target": {
            "family": family,
            "target_kind": target_kind,
            "expert_prior": args.expert_prior,
            "batch": args.batch,
            "op": args.op,
            "rows": rows,
            "k": args.k,
            "n": args.n,
            "transposed_rhs": args.trans if args.op == "gmm" else None,
            "search_fields": list(args.fields),
            "rounds_requested": args.rounds,
            "min_gain": args.min_gain,
        },
        "prior": prior,
        "protocol": {
            "sample_unit": "route_vector",
            "route_bank_generated_before_timing": True,
            "one_timing_sample_per_route_vector": True,
            "launches_per_sample": args.launches_per_sample,
            "warmup": args.warmup,
            "adaptive": final_policy.to_mapping(),
            "screen_adaptive": screen_policy.to_mapping(),
            "final_adaptive": final_policy.to_mapping(),
            "probe": {
                "enabled": probe_policy is not None,
                "slowdown": args.probe_slowdown,
                "adaptive": None if probe_policy is None else probe_policy.to_mapping(),
                "pooled_with_main_timing": False,
            },
            "route_order": "alternating implementation order",
            "numeric_inputs_fixed": True,
        },
        "baseline": {
            "config": baseline,
            "screen": evaluated[key(baseline)],
            "final": {
                "timing": final.get("timing"),
                "adaptive": final.get("adaptive"),
            },
        },
        "rounds": rounds,
        "search_space": {
            "group_median": group_median,
            "candidates": {
                field: config_values(
                    args.op,
                    field,
                    args.k,
                    args.n,
                    rows,
                    current[field],
                    experts,
                    group_median,
                )
                for field in args.fields
            },
        },
        "screen_winner": {
            "config": dict(current),
            "score_ms": float(evaluated[key(current)]["score_ms"]),
        },
        "final_candidate": {"config": dict(current), **final},
        "accepted": accepted,
        "selected": {"config": selected, "score_ms": math.exp(selected_score)},
        "correctness": correctness,
        "evaluations": list(evaluated.values()),
        "probed_out": sum(
            1 for item in evaluated.values() if item.get("status") == "probed_out"
        ),
    }
    gc.collect()
    torch.cuda.empty_cache()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert-prior", choices=EXPERT_PRIOR_NAMES, required=True)
    parser.add_argument("--target-kind", choices=("base", "lora"), default=None)
    parser.add_argument("--batch", choices=(1, 4, 16), type=int, required=True)
    parser.add_argument("--op", choices=("gmm", "ptgmm"), required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--trans", action="store_true")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--fields", default=",".join(FIELDS))
    parser.add_argument("--route-seed", type=int, default=None)
    parser.add_argument(
        "--seed-config",
        default=None,
        help="JSON object or path to one with the seven config fields to start from",
    )
    parser.add_argument("--repeats", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--sample-step", type=int, default=16)
    parser.add_argument("--final-repeats", type=int, default=None)
    parser.add_argument("--final-max-samples", type=int, default=None)
    parser.add_argument("--final-sample-step", type=int, default=None)
    parser.add_argument("--adaptive-confidence", type=float, default=0.90)
    parser.add_argument("--adaptive-epsilon-pct", type=float, default=2.0)
    parser.add_argument("--adaptive-stable-rounds", type=int, default=2)
    parser.add_argument("--adaptive-noise-floor-pct", type=float, default=0.5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--launches-per-sample", type=int, default=1)
    parser.add_argument(
        "--probe-slowdown",
        type=float,
        default=0.0,
        help="screen a candidate when the probe's lower log-gap bound exceeds this factor",
    )
    parser.add_argument("--probe-samples", type=int, default=2)
    parser.add_argument("--probe-max-samples", type=int, default=4)
    parser.add_argument("--probe-sample-step", type=int, default=2)
    parser.add_argument("--probe-epsilon-pct", type=float, default=5.0)
    parser.add_argument("--min-gain", type=float, default=0.01)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.fields = tuple(
        field.strip() for field in args.fields.split(",") if field.strip()
    )
    if not args.fields or any(field not in FIELDS for field in args.fields):
        parser.error(f"--fields must contain only: {', '.join(FIELDS)}")
    if args.rounds <= 0:
        parser.error("--rounds must be positive")
    if args.repeats <= 0 or args.max_samples < args.repeats:
        parser.error("--max-samples must be at least positive --repeats")
    if args.sample_step <= 0:
        parser.error("--sample-step must be positive")
    if args.final_repeats is not None and args.final_repeats <= 0:
        parser.error("--final-repeats must be positive")
    if args.final_max_samples is not None and args.final_max_samples <= 0:
        parser.error("--final-max-samples must be positive")
    if args.final_sample_step is not None and args.final_sample_step <= 0:
        parser.error("--final-sample-step must be positive")
    final_repeats = args.final_repeats or args.repeats
    final_max_samples = args.final_max_samples or args.max_samples
    final_sample_step = args.final_sample_step or args.sample_step
    if final_max_samples < final_repeats:
        parser.error("final max samples must be at least final repeats")
    if final_sample_step > final_max_samples:
        parser.error("final sample step cannot exceed final max samples")
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    if args.launches_per_sample <= 0:
        parser.error("--launches-per-sample must be positive")
    if args.probe_slowdown < 0.0:
        parser.error("--probe-slowdown must be nonnegative")
    if args.probe_slowdown > 1.0:
        if args.probe_samples <= 0 or args.probe_max_samples < args.probe_samples:
            parser.error("probe max samples must be at least positive probe samples")
        if not 0 < args.probe_sample_step <= args.probe_max_samples:
            parser.error(
                "probe sample step must be positive and within probe max samples"
            )
        if args.probe_epsilon_pct <= 0.0:
            parser.error("--probe-epsilon-pct must be positive")
        if args.probe_max_samples > args.max_samples:
            parser.error("probe max samples cannot exceed the screen max samples")
    if args.min_gain < 0.0:
        parser.error("--min-gain must be nonnegative")
    torch.cuda.set_device(0)
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "target": report["target"],
                "accepted": report["accepted"],
                "selected": report["selected"],
                "correctness": report["correctness"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
