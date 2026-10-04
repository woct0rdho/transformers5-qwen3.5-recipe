# AITER GMM and PTGMM coefficient-prior tuning

This document defines the reproducible tuning protocol for the AITER `gmm` and `ptgmm` configurations in this repository. The production authority remains `moe_gmm_configs.py`. AITER source and operator contracts are not modified by the tuner.

The tuner searches exact supported shape keys with bounded coordinate descent. It is a benchmark tool, not a production route dispatcher and not a claim about model quality or training-wide route frequency.

## One prior law per run

Every tuner invocation selects exactly one value of `--expert-prior`:

| Prior | Family | Top-k | Law |
|---|---|---:|---|
| `qwen-learned` | Qwen | 8 | fitted learned support and rank curve |
| `deepseek-learned` | DeepSeek | 6 | fitted learned support and rank curve |
| `deepseek-hash` | DeepSeek | 6 | fitted persistent-head plus Dirichlet body |
| `qwen3.8-learned` | Qwen3.8 (Qwen4-Exp) | 10 | fitted learned support and rank curve, 512 experts |

The fitted Qwen3.8 law (final result. The fit and its validation are documented in `torch-ggml-ops/docs/expert_distribution_prior.md`) is:

```text
x = log(T / 2048)
y_A = a0 + a_log_tokens*x + epsilon_A
A = round(513*sigmoid(y_A) - 0.5), clipped to [10, 512]
log(alpha) = b0 + b_log_tokens*x + b_active_residual*epsilon_A + epsilon_alpha
q_r = min(1, C*(r + shift)^(-alpha)), 1 <= r <= A, sum(q_r) = 10
```

| Coefficient | Value |
| --- | ---: |
| `shift` | `20.7163112483` |
| `h` | `1.1946652586` |
| `a0` | `0.7076361999` |
| `a_log_tokens` | `0.9702739281` |
| `b0` | `0.7505135414` |
| `b_log_tokens` | `-0.0229817472` |
| `b_active_residual` | `-0.2302367069` |

| Residual | `nu` | `loc` | `scale` | Clip |
| --- | ---: | ---: | ---: | ---: |
| `epsilon_A` | `7.4647474580` | `0.0130546014` | `0.4915605168` | `[-1.4935692978,1.9212931553]` |
| `epsilon_alpha` (dispersion calibrated, `x0.6`) | `8469463127.5627` | `0.0002737558` | `0.0870558989` | `[-0.2081819082,0.2166008918]` |

`expert_distribution_prior.py` implements all four laws, and `torch-ggml-ops/bench/workload_prior.py` is numerically identical to it (same rows and metadata for every law and seed). The expert count is a property of the law rather than a module constant: `LearnedLaw.experts` is 256 for the Qwen and DeepSeek laws and 512 for `qwen3.8-learned`, `expert_prior_experts()` exposes it, and the support transform is `A = round((E + 1)*sigmoid(y_A) - 0.5)` clipped to `[top_k, E]`. The 256-expert rendering is unchanged for the two migrated families.

`moe_gmm_configs.py` carries the measured `qwen3.8-learned` entries for all 54 exact keys (36 GMM, 18 PTGMM) at physical B1/B4/B16. The campaign that produced them is described under "Qwen3.8 (Qwen4-Exp) campaign result".

The coefficient definitions of `qwen-learned`, `deepseek-learned` and `deepseek-hash` are implemented in `expert_distribution_prior.py` and numerically match `torch-ggml-ops/bench/workload_prior.py`. A DeepSeek learned draw and a DeepSeek hash draw are separate campaigns. They are never averaged, pooled, or combined with a `40/43` versus `3/43` production weighting. The campaign runner defaults to one Qwen learned campaign and two separate DeepSeek campaigns.

The only workload input to a fitted law is the physical token count `T = physical_batch * 2048`. The sampler uses the fitted residual laws, exact constrained largest-remainder rounding, and a deterministic expert-identity permutation. It preserves exactly `top_k * T` routed rows and keeps every hash expert active. Seeds are recorded in each report.

DeepSeek V4 uses both route laws in one model. `model.layers[0]`, `model.layers[1]`, and `model.layers[2]` are `hash_moe` layers with `DeepseekV4HashRouter`, so their routed expert modules use `deepseek-hash`. `model.layers[3]` through `model.layers[42]` are ordinary `moe` layers with `DeepseekV4TopKRouter`, so their routed expert modules use `deepseek-learned`. The two prior values select configurations for the same B1/B4/B16 row counts and expert matrix geometries. They must not be pooled or assigned by a model-wide compromise. The dense `shared_experts` MLP and the router's own linear are outside this GMM/PTGMM route-prior table.

The current GGUF training path uses packed MMQ for frozen base expert projections and input gradients, while AITER GMM/PTGMM handles the rank-4 LoRA factors. Base-shaped entries are retained for future non-packed base-kernel tuning and are not evidence that the current packed path invokes those entries.

The routed row counts are:

| Family | B1 | B4 | B16 |
|---|---:|---:|---:|
| DeepSeek top-6 | 12,288 | 49,152 | 196,608 |
| Qwen top-8 | 16,384 | 65,536 | 262,144 |
| Qwen3.8 top-10 | 20,480 | 81,920 | 327,680 |

No captured histogram, checkpoint, layer identity, training step, or route corpus is consumed by the coefficient prior.

## Route bank

`route_bank_for_routed_rows()` creates the complete route bank before any timed AITER call. The default bank has 128 vectors. The first route seed is `8,314,159 + physical_batch * 104,729`, with the DeepSeek hash offset applied by the canonical prior module. An explicit `--route-seed` is also supported.

A `RouteVector` records:
- the selected prior, seed, token count, top-k, row total, and row digest.
- compact active `expert_indices`.
- cumulative `expert_offsets` over the compact active experts.
- `group_sizes`, a full `experts`-entry tuple indexed by physical expert ID (256 for the Qwen and DeepSeek laws, 512 for `qwen3.8-learned`).

AITER receives the full `group_sizes` tensor because its GMM/PTGMM calls own one RHS plane per physical expert. The compact indices and offsets are retained as route provenance and make the active-order interpretation explicit. They are not silently substituted for the AITER group-size contract.

A route vector is one complete workload: every expert group size is selected as a unit. The route bank is generated once for a target and reused for every configuration comparison in that target. Numeric input tensors are also created once per target with a deterministic CUDA generator and are reused by both configurations. Route generation, tensor allocation, validation, and kernel compilation are outside measured CUDA events.

## Timing protocol

The defaults are:

| Setting | Default |
|---|---:|
| initial samples (`--repeats`) | 16 |
| maximum samples (`--max-samples`) | 128 |
| expansion (`--sample-step`) | 16 |
| confidence | 90% |
| estimate epsilon | 2% |
| stable rounds | 2 |
| noise floor | 0.5% |
| warmup launches per route | 2 |
| timed launches per sample | 1 |

The sample unit is a route vector, not a profile aggregate. For route index `i`, exactly one baseline timing sample and one candidate timing sample are recorded. The second launch is first on odd indices and the baseline is first on even indices. This alternating order is part of the report and reduces systematic position effects.

`--launches-per-sample N` runs `N` launches for each implementation inside the same timed event pair and stores the average milliseconds as the one sample for that route vector. The route, inputs, and output contract remain fixed across those launches. It does not create `N` independent route samples and does not permit mixing route vectors.

For every implementation, raw route samples are retained. Summaries are computed in log-time space with a robust scale estimate using standard deviation, MAD, IQR, and the configured noise floor. Adaptive expansion stops only after the configured number of consecutive rounds satisfies both:
- the paired log-speedup confidence interval is within epsilon.
- the cumulative log-speedup estimate changed by no more than epsilon from the previous round.

The primary paired speedup is:

`exp(median(log(baseline_ms)) - median(log(candidate_ms)))`

The report contains raw samples, per-round confidence bounds, stop reason, route order, sample count, policy values, and the full deterministic route bank. A candidate is accepted only when the final paired result exceeds `1 + --min-gain` and the existing correctness check passes.

## Search and correctness

Each exact target starts from its current production configuration. The tuner changes only the supported fields `BLOCK_SIZE_M`, `BLOCK_SIZE_K`, `BLOCK_SIZE_N`, `GROUP_SIZE`, `GRID_DIM`, `num_warps`, and `num_stages`. It uses bounded coordinate descent: one field at a time, one shortlist per field, never a Cartesian product across fields. Invalid shared-memory products and launch errors reject only that candidate.

### Geometry-pruned shortlist

The candidate values of a field are filtered by the kernel geometry before any timing. An M tile that is more than twice the median routed group size spends most of its rows on padding, a K or N tile wider than its own dimension has no work to do, and a PTGMM output tile wider than its output dimension is pure waste. `num_warps` is deliberately not filtered, because the two migrated families' measured tables contain one-warp winners for rank-small outputs and for one B1 base shape. The filter is recorded in every report under `search_space`, together with the median group size it used, and the current value is always kept in the shortlist, so pruning can never silently rewrite a configuration.

### Staged probe screen

`--probe-slowdown F` adds the staged probe of the EvoTensile screening design. A candidate is first measured against the fastest configuration already known for that target under a short paired policy (default `--probe-samples 2`, `--probe-max-samples 4`, `--probe-sample-step 2`, `--probe-epsilon-pct 5`), and it is screened out only when the lower confidence bound of its log-time gap exceeds `F`. The campaign used `1.25` for the B1/B4 screens and `1.15` for the B16 screens. Probe timing has its own policy identity, is never pooled with screen or final timing, and is reported per candidate. A screened candidate consumes no screen budget. A survivor is then measured under the screen policy exactly as before. Probe screening is a timing-allocation decision, not validity evidence, and a screened candidate can be retried in a later run.

### Winner refinement and independent confirmation

A sweep winner is re-measured once with a single-field `num_warps` refinement on a fixed 16-sample paired budget, because the shortlist and the seed's own value decide which candidates the sweep compared, and `num_warps` is the field most sensitive to that choice. The refined configuration is then confirmed against the seed it replaces by an independent paired replay on the full 128-vector route bank with `--repeats 32` and adaptive expansion to 128 samples, under the same bitwise correctness check. A confirmation that fails the bitwise gate or a `1.02` speedup reverts the key to its seed. Confirmation evidence is separate from screening evidence: a sweep winner that fails it is not a production configuration.

Correctness compares the baseline and selected configuration on a deterministic sparse boundary route with group sizes `[2, 0, 3]`. This check is separate from performance route-bank sampling and does not change the selected prior law.

## Commands

A single target can be run as follows. The expert prior determines the route family:

```bash
python tune_coefficient_prior_gmm.py \
  --expert-prior deepseek-learned \
  --batch 4 --op ptgmm --k 4096 --n 4 \
  --output results/deepseek-learned-ptgmm.json
```

The campaign runner passes the same protocol arguments to each target. Its output names include the prior, so learned and hash results cannot overwrite or be mistaken for one another:

```bash
python run_coefficient_prior_campaign.py \
  --output-dir results/gmm_campaign --expert-prior deepseek-learned
```

Confirmation replays the source report's one prior law, route seed, complete route bank, warmup, launch multiplicity, and adaptive policy. It refuses to time if regenerated route metadata does not exactly match the source report:

```bash
python confirm_coefficient_prior_gmm.py \
  --input-dir results/gmm_campaign \
  --output results/gmm_campaign/confirmation.json
```

The Qwen3.8 campaign adds two flags to the single-target form: `--probe-slowdown F` (staged probe screen) and, when a measured seed exists, `--seed-config` (an inline JSON object or a file with the seven fields). `--fields` restricts a run to one knob, which is how the `num_warps` refinement is run (`--fields num_warps`).

## Qwen3.8 (Qwen4-Exp) campaign result

Qwen3.8-Flash-Next (`qwen4exp`) is the third routed family: 512 experts, top-10, one model state, 48 MoE layers, and its own fitted law. The target inventory is 36 GMM keys and 18 PTGMM keys over physical B1/B4/B16.

Method per key, in order:
- seed = the Qwen3.5 entry of the same shape class under the shape bijection `2048 <-> 2560`, `1024 <-> 1280`, `512 <-> 640` (the placeholder that this change replaces).
- probe-screened bounded coordinate sweep with the geometry-pruned shortlist under the fitted 512-expert route law (`--rounds 1`).
- single-field `num_warps` refinement of the winner.
- independent confirmation of the final configuration against that seed, on the full 128-vector route bank at `--repeats 32` (adaptive to 128 samples), with the bitwise correctness gate.

All 54 keys were measured. 33 keys gained a confirmed improvement (median `1.088x`, largest `1.560x`), 6 sweep winners measured about `2%` slower than their seed under the independent replay and were reverted, and 15 keys kept their seed because no candidate beat the sweep's `1%` acceptance margin. The six reverts are the expected outcome of screening at 16 samples: their confirmed gap was stable in the wrong direction at a confidence half-width below `1.6%`, so the seed is the better choice.

| Batch | Kind | Keys | Confirmed improvements | Median confirmed gain |
| --- | --- | ---: | ---: | ---: |
| B1 | base | 6 | 5 | `1.094x` |
| B1 | LoRA factors | 12 | 8 | `1.045x` |
| B4 | base | 6 | 3 | `1.112x` |
| B4 | LoRA factors | 12 | 7 | `1.050x` |
| B16 | base | 6 | 2 | `1.187x` |
| B16 | LoRA factors | 12 | 8 | `1.070x` |

Largest confirmed gains: gate-up LoRA-A forward at B1 `1.560x`, its PTGMM counterpart at B1 `1.282x`, the gate-up LoRA-A forward at B16 `1.222x`, the gate-up LoRA-B forward at B1 `1.199x`, and the two B16 base PTGMM shapes `1.195x` and `1.179x`.

Two shape-level observations are worth keeping:
- The base shapes converge on `BLOCK_SIZE_M` 64-128 with `num_warps` 8, while the rank-4 factors keep the small tiles the migrated tables use (`BLOCK_SIZE_N` 16 for an output of 4, `BLOCK_SIZE_M` 16-32). A key that kept its seed kept it against at least twenty fully measured or probed candidates.
- `num_warps = 1` won nowhere at 512 experts across the 54 keys, including the K=4 shapes where the migrated tables use it at 256 experts.

Artifacts, all under `~/tmp/test_no_unsloth/`: `aiter_qwen38_campaign_v2/` (one probe-screened sweep report per key, with per-candidate timing, probe evidence, and the recorded `search_space`), `aiter_qwen38_refine/` and `aiter_qwen38_refine_b4/` (`num_warps` refinements), `aiter_qwen38_confirm/` (independent confirmations plus `confirmation_state.json`), `aiter_qwen38_final_configs.json` and `aiter_qwen38_final_report.json` (the assembled per-key result with its provenance: confirmed, confirmed-reverted, or placeholder-retained), `aiter_qwen38_table_seeds.json` (the placeholder snapshot), and the drivers `sweep_qwen38_aiter.py`, `assemble_qwen38_aiter.py`, `confirm_qwen38_aiter.py`, `write_qwen38_table.py`, and `check_qwen38_table_launches.py`. The last one is the fail-closed launch gate for the production table: it builds every key from the fitted law, launches the table configuration, and requires bitwise equality with the placeholder seed on the sparse boundary route (`54` keys, `0` failures).

## Evidence boundary

The fitted laws are workload priors. Deterministic route banks make paired configuration comparisons reproducible, but they do not exhaust the residual distributions or establish production frequency. The bounded coordinate search does not prove global optimality. Historical result documents in `torch-ggml-ops` remain unchanged and are not rewritten by this protocol.
