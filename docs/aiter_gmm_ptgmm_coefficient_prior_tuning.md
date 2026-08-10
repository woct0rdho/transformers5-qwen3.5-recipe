# AITER GMM and PTGMM coefficient-prior tuning

## Status and scope

This record owns the gfx1151 tuning method and evidence for the exact AITER `gmm` and `ptgmm` configurations used by routed MoE LoRA training in this repository. The production authority is `moe_gmm_configs.py`. `~/torch-ggml-ops/bench/aiter_gmm_heuristics.py` is kept numerically identical so grouped-MMQ comparator tooling and training cannot diverge on an exact key. AITER source code was not modified.

The complete table contains 66 GMM keys `(total routed rows, K, N, RHS layout)` and 33 PTGMM keys `(total routed rows, K, N)`. A prior route-aware campaign had already covered the 24 full-rank base-model GMM keys. This campaign therefore screened the remaining 42 rank-4 LoRA GMM keys and all 33 PTGMM keys, including both rank-4 and full-rank matrix shapes.

'Base model' and 'LoRA' identify matrix-shape families only. They are not router priors, runtime labels, checkpoint states, adapter states, or production-frequency classes. A full-rank PTGMM benchmark key does not make the frozen packed base weights trainable. Runtime dispatch remains exact shape and layout dispatch and fails closed for unknown keys.

The routed row counts at sequence length 2,048 are:

| Family | Physical B1 | Physical B4 | Physical B16 |
|---|---:|---:|---:|
| DeepSeek top-6 | 12,288 | 49,152 | 196,608 |
| Qwen top-8 | 16,384 | 65,536 | 262,144 |

The newly screened GMM shape inventory is:

| Family | Transposed RHS | Row-major RHS |
|---|---|---|
| DeepSeek rank-4 | `4096x4`, `2048x4`, `4x4096` | `4x4096`, `4x2048`, `4096x4` |
| Qwen rank-4 | `2048x4`, `512x4`, `4x1024`, `4x2048` | `4x2048`, `4x512`, `1024x4`, `2048x4` |

The PTGMM inventory is:

| Family | Rank-4 shapes | Full-rank shapes |
|---|---|---|
| DeepSeek | `4096x4`, `2048x4`, `4x4096` | `4096x2048`, `2048x4096` |
| Qwen | `2048x4`, `512x4`, `4x1024`, `4x2048` | `2048x512`, `512x2048` |

## Coefficient-only priors

No captured expert histogram, model checkpoint, layer identity, training step, or route corpus is consumed by the tuner. The only workload variable supplied to a learned prior is `T = physical_batch * sequence_length`, with reference token count `T0 = 2048`.

For a learned router, active support `A` and ranked exponent `alpha` are sampled from:

`x = log(T / 2048)`

`log((A + 0.5) / (256 - A + 0.5)) = a0 + a_log_tokens * x + epsilon_A`

`log(alpha) = b0 + b_log_tokens * x + b_active_residual * epsilon_A + epsilon_alpha`

For ranks `r = 1..A`, the sampler computes `q_r = min(1, C * (r + shift)^(-alpha))`, chooses `C` so that `sum(q_r) = top_k`, adjusts the first rank by the fitted head multiplier while preserving the total, applies constrained largest-remainder rounding to `T * q_r`, and randomly permutes physical expert identities. Ranks above `A` receive zero rows.

The learned coefficients are:

| Parameter | Qwen learned | DeepSeek learned |
|---|---:|---:|
| `top_k` | 8 | 6 |
| `shift` | 11.5465232873 | 6.3223820835 |
| head multiplier | 1.1255020052 | 1.0643729189 |
| `a0` | 1.3568429238 | 2.2468973539 |
| `a_log_tokens` | 1.1367804236 | 0.8906164814 |
| `b0` | 0.7376182182 | 0.4081577973 |
| `b_log_tokens` | -0.0215993943 | -0.0211298377 |
| `b_active_residual` | -0.1730074363 | -0.0754167931 |

Qwen uses `epsilon_A ~ clip(StudentT(8.3978226526, -0.0335118652, 0.9830493404), [-2.5036492445, 3.7809143778])` and `epsilon_alpha ~ clip(StudentT(5.2038953632, 0.0054814884, 0.1025706842), [-0.4463245132, 0.3833201702])`.

DeepSeek learned routing uses `epsilon_A ~ clip(StudentT(33.5987235964, -0.0013194870, 0.8328268568), [-1.6829685257, 2.7587218851])` and `epsilon_alpha ~ clip(StudentT(5.1551932778, -0.0071110386, 0.0923527684), [-0.3180690433, 0.3604795507])`.

The capture-free DeepSeek hash prior samples a persistent head share `rho` and exchangeable Dirichlet body. Its coefficients are `mu_rho = 0.076568603515625`, `kappa_rho = 111.40091020461985`, `body_a0 = 6.660030508273053`, and `body_log_tokens_exponent = 0.4771750368253468`. For `B = T / 2048`, the beta concentration is `B * (kappa_rho + 1) - 1`, the body concentration is `body_a0 * B^body_log_tokens_exponent`, six randomly permuted head experts divide `rho`, and bounded largest-remainder rounding produces exactly `6T` rows with every hash expert active.

The tuning profiles are deliberately small and unweighted. For physical batch `B`, the base seed is `8,314,159 + B * 104,729`. Qwen learned profile A uses that seed and profile B adds `1,000,003`, while DeepSeek learned uses the base seed and DeepSeek hash adds `31,000`. Qwen therefore uses two deterministic learned-prior draws. DeepSeek uses one deterministic learned-prior draw and one deterministic hash-prior draw with equal standing. The production layer ratio of 40 learned routers to 3 hash routers is not used. This is per-kernel tuning, not an estimate of training-wide frequency.

The generated profile geometry was:

| Family/profile | B1 active / max rows | B4 active / max rows | B16 active / max rows |
|---|---:|---:|---:|
| Qwen learned A | 233 / 1,297 | 253 / 4,192 | 253 / 20,802 |
| Qwen learned B | 241 / 941 | 253 / 5,055 | 243 / 32,768 |
| DeepSeek learned | 244 / 982 | 253 / 3,441 | 254 / 12,476 |
| DeepSeek hash | 256 / 264 | 256 / 942 | 256 / 3,765 |

## Search method

Every target is tuned independently from its current exact-key configuration. The seven serialized fields are `BLOCK_SIZE_M`, `BLOCK_SIZE_K`, `BLOCK_SIZE_N`, `GROUP_SIZE`, `GRID_DIM`, `num_warps`, and `num_stages`.

The search is one bounded coordinate pass rather than a Cartesian product. For each field, all other fields remain fixed at the current coordinate winner. Candidate domains are powers of two already supported by AITER: M tiles `16, 32, 64, 128, 256` with PTGMM `512` only for sufficiently large B16 groups, K tiles `16, 32` for rank-small K and otherwise `32, 64, 128, 256` for GMM or `64, 128, 256, 512` for PTGMM, N tiles `16, 32` for rank-small N and otherwise `32, 64, 128, 256, 512`, group sizes `1, 2, 4, 8`, grid dimensions `20, 40, 80, 160, 256`, warps `1, 2, 4, 8`, and stages `1, 2, 3`. The current value is always retained even when it lies outside a reduced slice. Shared-memory-impossible products are pruned and compile or launch errors reject only that candidate.

The screen used 5 ms warmup, 8 ms measurement windows per coefficient profile, and a 25 ms final baseline-versus-candidate measurement. A provisional candidate needed more than a one-percent aggregate gain. This screen produced 35 provisional candidates from 75 targets.

Every provisional candidate then received 25 alternating-order single-launch timings on each coefficient profile. Promotion required bitwise equality to the old config on a 256-group sparse boundary control, aggregate speedup greater than two percent, and no coefficient profile below `0.99x`. Twenty-five candidates passed: eight GMM and seventeen PTGMM. Fifty targets retained their prior configurations, including ten provisional screen winners that failed the longer confirmation.

Each promoted config was also checked against an independent per-group `torch.float32` matmul reference rounded to BF16 on group sizes `[2,0,3]`, then rerun after an input mutation. All 25 passed. The maximum relative RMSE was `6.173e-5`, the minimum cosine was `0.99999988`, and every mutation changed the expected active output.

## Confirmed changes

Config tuples below use `(BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, GROUP_SIZE, GRID_DIM, num_warps, num_stages)`.

| Operator | Exact key | Old config | New config | Aggregate speedup | Minimum profile |
|---|---|---|---|---:|---:|
| GMM | `(12288,2048,4,T)` | `(64,256,16,4,20,4,2)` | `(16,256,16,2,40,4,3)` | 1.2444x | 1.1924x |
| GMM | `(12288,4,4096,T)` | `(32,16,64,4,256,2,3)` | `(32,16,64,8,256,2,2)` | 1.0213x | 1.0143x |
| GMM | `(49152,4,4096,T)` | `(32,16,64,8,160,2,1)` | `(64,16,32,1,160,2,2)` | 1.0853x | 1.0534x |
| GMM | `(49152,4,4096,N)` | `(16,16,128,2,256,1,1)` | `(16,16,128,1,160,1,1)` | 1.0466x | 1.0357x |
| GMM | `(65536,4,2048,T)` | `(32,16,64,4,160,2,2)` | `(32,16,64,1,160,2,2)` | 1.0493x | 1.0414x |
| GMM | `(65536,4,2048,N)` | `(16,16,128,8,160,1,1)` | `(16,16,128,2,160,1,2)` | 1.0348x | 1.0322x |
| GMM | `(196608,4096,4,N)` | `(16,128,16,4,160,2,1)` | `(16,64,16,1,160,1,1)` | 1.0218x | 1.0218x |
| GMM | `(262144,4,1024,T)` | `(32,16,64,8,160,2,2)` | `(32,16,64,1,160,2,1)` | 1.0384x | 1.0358x |
| PTGMM | `(12288,2048,4)` | `(16,128,16,2,160,4,1)` | `(16,64,16,1,160,1,1)` | 1.0471x | 1.0390x |
| PTGMM | `(12288,4,4096)` | `(64,16,256,8,160,4,1)` | `(64,16,256,1,80,4,1)` | 1.0249x | 1.0046x |
| PTGMM | `(49152,4096,4)` | `(16,512,16,1,80,8,1)` | `(32,512,16,2,80,8,1)` | 1.0409x | 1.0288x |
| PTGMM | `(49152,4096,2048)` | `(16,64,256,8,80,4,1)` | `(16,64,256,8,80,4,3)` | 1.0376x | 1.0101x |
| PTGMM | `(49152,2048,4096)` | `(16,64,512,2,40,8,1)` | `(16,64,512,2,40,8,3)` | 1.1029x | 1.0528x |
| PTGMM | `(196608,4096,2048)` | `(16,64,256,1,80,4,1)` | `(16,64,256,4,80,4,1)` | 1.0507x | 0.9978x |
| PTGMM | `(16384,2048,4)` | `(16,128,16,8,80,4,2)` | `(16,256,16,8,80,4,1)` | 1.0487x | 1.0391x |
| PTGMM | `(16384,2048,512)` | `(16,64,128,1,80,8,3)` | `(16,64,256,1,80,8,2)` | 1.0672x | 1.0666x |
| PTGMM | `(16384,512,2048)` | `(16,64,128,4,80,8,3)` | `(16,64,256,4,80,8,1)` | 1.0798x | 1.0779x |
| PTGMM | `(65536,2048,4)` | `(64,512,16,1,80,8,1)` | `(32,512,16,4,80,8,1)` | 1.0327x | 0.9985x |
| PTGMM | `(65536,512,4)` | `(64,128,16,1,20,4,1)` | `(64,128,16,1,80,4,2)` | 1.1481x | 1.1474x |
| PTGMM | `(65536,4,1024)` | `(16,16,128,8,80,2,1)` | `(32,16,64,4,80,2,2)` | 1.1090x | 1.1036x |
| PTGMM | `(262144,2048,4)` | `(64,256,16,1,40,4,1)` | `(64,256,16,1,256,4,1)` | 1.0408x | 1.0278x |
| PTGMM | `(262144,512,4)` | `(64,256,16,2,20,8,2)` | `(64,128,16,4,160,8,2)` | 1.1467x | 1.1367x |
| PTGMM | `(262144,4,1024)` | `(16,16,128,2,80,2,1)` | `(32,16,64,1,80,4,3)` | 1.1642x | 1.1641x |
| PTGMM | `(262144,4,2048)` | `(64,16,256,1,40,4,1)` | `(64,16,256,8,256,4,1)` | 1.0391x | 1.0238x |
| PTGMM | `(262144,512,2048)` | `(32,128,256,2,40,8,3)` | `(32,128,256,1,40,8,3)` | 1.1313x | 1.1163x |

The two sub-one minimum-profile ratios are bounded route-component compromises, not hidden production weighting: DeepSeek B16 PTGMM `(4096,2048)` measured `0.9978x` on its weaker equal-standing profile while gaining `1.0507x` in the unweighted aggregate, and Qwen B4 PTGMM `(2048,4)` measured `0.9985x` on its weaker profile while gaining `1.0327x` in aggregate. Both satisfy the predefined `0.99x` per-profile floor.

## Evidence boundary

The fit coefficients are benchmark priors. They do not establish production route frequency, checkpoint invariance, layer invariance, or model-quality effects. The deterministic profiles make config comparisons reproducible but do not exhaust the fitted residual distributions. Captured routes were intentionally excluded from candidate generation, ranking, and confirmation.

The search is bounded coordinate descent and does not prove global optimality. It does not test Cartesian combinations that require moving several fields simultaneously before any intermediate coordinate improves. Rejected screen candidates remain rejected for this campaign even if a shorter timing window looked favorable.

The campaign does not alter AITER kernels, operator contracts, factor layouts, group-size ownership, autograd behavior, or dispatch dimensions. It changes only exact config values for already-supported keys. Unknown shapes still fail closed.

## Artifacts

Per-target screen reports and logs are under `~/tmp/test_no_unsloth/gmm_coefficient_campaign/`.

The 25-repeat confirmation artifact is `~/tmp/test_no_unsloth/gmm_coefficient_campaign/confirmation_25.json`.

The compact campaign authority is `~/tmp/test_no_unsloth/gmm_coefficient_campaign/campaign_summary.json`.

The independent-reference and mutation artifact is `~/tmp/test_no_unsloth/gmm_coefficient_campaign/correctness.json`.

The self-contained coefficient sampler and tuner are `~/tmp/test_no_unsloth/tune_coefficient_prior_gmm.py`, `~/tmp/test_no_unsloth/run_coefficient_prior_campaign.py`, and `~/tmp/test_no_unsloth/confirm_coefficient_prior_gmm.py`.
