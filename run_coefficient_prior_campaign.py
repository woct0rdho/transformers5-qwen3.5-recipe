#!/usr/bin/env python3
"""Run the bounded coefficient-only AITER GMM/PTGMM target campaign."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import cast

SCRIPT_DIR = Path(__file__).resolve().parent
TUNER = SCRIPT_DIR / "tune_coefficient_prior_gmm.py"
FAMILIES = {
    "deepseek": {
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
ROWS = {
    "deepseek": {1: 12288, 4: 49152, 16: 196608},
    "qwen": {1: 16384, 4: 65536, 16: 262144},
    "qwen3.8": {1: 20480, 4: 81920, 16: 327680},
}
PRIOR_FAMILIES = {
    "qwen-learned": "qwen",
    "qwen3.8-learned": "qwen3.8",
    "deepseek-learned": "deepseek",
    "deepseek-hash": "deepseek",
}
EXPERT_PRIORS = (
    "qwen-learned",
    "qwen3.8-learned",
    "deepseek-learned",
    "deepseek-hash",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expert-prior", choices=EXPERT_PRIORS)
    parser.add_argument("--target-kind", choices=("base", "lora"))
    parser.add_argument("--batch", choices=(1, 4, 16), type=int)
    parser.add_argument("--op", choices=("gmm", "ptgmm"))
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--sample-step", type=int, default=16)
    parser.add_argument("--final-repeats", type=int)
    parser.add_argument("--final-max-samples", type=int)
    parser.add_argument("--final-sample-step", type=int)
    parser.add_argument("--adaptive-confidence", type=float, default=0.90)
    parser.add_argument("--adaptive-epsilon-pct", type=float, default=2.0)
    parser.add_argument("--adaptive-stable-rounds", type=int, default=2)
    parser.add_argument("--adaptive-noise-floor-pct", type=float, default=0.5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--launches-per-sample", type=int, default=1)
    parser.add_argument("--route-seed", type=int)
    parser.add_argument("--min-gain", type=float, default=0.01)
    parser.add_argument(
        "--budget-seconds",
        type=int,
        default=240,
        help="hard wall-clock limit for each target experiment (maximum 300)",
    )
    args = parser.parse_args()
    if args.repeats <= 0 or args.max_samples < args.repeats:
        parser.error("--max-samples must be at least positive --repeats")
    if args.sample_step <= 0 or args.warmup < 0 or args.launches_per_sample <= 0:
        parser.error(
            "sample-step and launches must be positive, warmup must be nonnegative"
        )
    if args.final_repeats is not None and args.final_repeats <= 0:
        parser.error("--final-repeats must be positive")
    if args.final_max_samples is not None and args.final_max_samples <= 0:
        parser.error("--final-max-samples must be positive")
    if args.final_sample_step is not None and args.final_sample_step <= 0:
        parser.error("--final-sample-step must be positive")
    if (
        args.final_max_samples is not None
        and args.final_repeats is not None
        and args.final_max_samples < args.final_repeats
    ):
        parser.error("--final-max-samples must be at least --final-repeats")
    if not 0 < args.budget_seconds <= 300:
        parser.error("--budget-seconds must be between 1 and 300")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    batches = (args.batch,) if args.batch else (1, 4, 16)
    ops = (args.op,) if args.op else ("gmm", "ptgmm")
    target_kinds = (args.target_kind,) if args.target_kind else ("base", "lora")
    expert_priors = (args.expert_prior,) if args.expert_prior else EXPERT_PRIORS
    results = []
    for expert_prior in expert_priors:
        family = PRIOR_FAMILIES[expert_prior]
        for target_kind in target_kinds:
            for batch in batches:
                for op in ops:
                    targets = FAMILIES[family][f"{op}_{target_kind}"]
                    for target in targets:
                        if op == "gmm":
                            k, n, transposed = cast(tuple[int, int, bool], target)
                            layout = "t" if transposed else "n"
                            name = f"gmm_{target_kind}_{expert_prior}_b{batch}_k{k}_n{n}_{layout}"
                        else:
                            k, n = target[0], target[1]
                            transposed = False
                            name = (
                                f"ptgmm_{target_kind}_{expert_prior}_b{batch}_k{k}_n{n}"
                            )
                        output = args.output_dir / f"{name}.json"
                        log = output.with_suffix(".log")
                        command = [
                            sys.executable,
                            str(TUNER),
                            "--expert-prior",
                            expert_prior,
                            "--target-kind",
                            target_kind,
                            "--batch",
                            str(batch),
                            "--op",
                            op,
                            "--k",
                            str(k),
                            "--n",
                            str(n),
                            "--rounds",
                            str(args.rounds),
                            "--repeats",
                            str(args.repeats),
                            "--max-samples",
                            str(args.max_samples),
                            "--sample-step",
                            str(args.sample_step),
                            "--adaptive-confidence",
                            str(args.adaptive_confidence),
                            "--adaptive-epsilon-pct",
                            str(args.adaptive_epsilon_pct),
                            "--adaptive-stable-rounds",
                            str(args.adaptive_stable_rounds),
                            "--adaptive-noise-floor-pct",
                            str(args.adaptive_noise_floor_pct),
                            "--warmup",
                            str(args.warmup),
                            "--launches-per-sample",
                            str(args.launches_per_sample),
                            "--min-gain",
                            str(args.min_gain),
                            "--output",
                            str(output),
                        ]
                        if args.final_repeats is not None:
                            command.extend(("--final-repeats", str(args.final_repeats)))
                        if args.final_max_samples is not None:
                            command.extend(
                                ("--final-max-samples", str(args.final_max_samples))
                            )
                        if args.final_sample_step is not None:
                            command.extend(
                                ("--final-sample-step", str(args.final_sample_step))
                            )
                        if args.route_seed is not None:
                            command.extend(("--route-seed", str(args.route_seed)))
                        if transposed:
                            command.append("--trans")
                        print(f"START {name}", flush=True)
                        timed_out = False
                        try:
                            with log.open("w", encoding="utf-8") as handle:
                                completed = subprocess.run(
                                    command,
                                    stdout=handle,
                                    stderr=subprocess.STDOUT,
                                    timeout=args.budget_seconds,
                                    check=False,
                                )
                            returncode = completed.returncode
                        except subprocess.TimeoutExpired:
                            timed_out = True
                            returncode = 124
                        record = {
                            "name": name,
                            "family": family,
                            "target_kind": target_kind,
                            "expert_prior": expert_prior,
                            "batch": batch,
                            "op": op,
                            "k": k,
                            "n": n,
                            "transposed_rhs": transposed if op == "gmm" else None,
                            "returncode": returncode,
                            "timed_out": timed_out,
                            "budget_seconds": args.budget_seconds,
                            "output": str(output),
                            "log": str(log),
                        }
                        if not timed_out and output.exists():
                            report = json.loads(output.read_text(encoding="utf-8"))
                            if not isinstance(report, dict):
                                raise TypeError(
                                    f"{output} does not contain a JSON object"
                                )
                            record.update(
                                {
                                    "accepted": report.get("accepted"),
                                    "selected": report.get("selected"),
                                    "correctness": report.get("correctness"),
                                }
                            )
                        results.append(record)
                        print(json.dumps(record, sort_keys=True), flush=True)
    manifest = {
        "method": "single-prior-route-bank-adaptive-paired-coordinate-search",
        "priors": EXPERT_PRIORS,
        "rows": ROWS,
        "target_kinds": target_kinds,
        "budget_seconds": args.budget_seconds,
        "target_count": len(results),
        "results": results,
    }
    (args.output_dir / "campaign_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    failed = [
        record for record in results if record["returncode"] != 0 or record["timed_out"]
    ]
    print(f"completed={len(results)} failed={len(failed)}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
