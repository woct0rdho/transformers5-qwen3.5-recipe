# DeepSeek V4 Sliding-Attention Backward Plan

Status: accepted and optimization-exhausted for the exact production geometry on `gfx1151`.

This document owns the dense sliding-attention backward algorithm, launch tables, measurements, compiler evidence, and rejected branches. The forward state contract is in `plan_deepseek_v4_sliding_attention_forward.md`.

## Production contract

Backward returns:

- BF16 `dQ` `[B,64,2048,512]`;
- one BF16 combined shared-input gradient `[B,1,2048,512]` for K=V;
- FP32 differentiable sink gradient `[64]`.

It consumes BF16 Q/shared-KV/output/`dO`, FP32 LSE, and one raw FP16 score band `[B,64,2048,128]`. Production `dO` may be a last-dimension-contiguous non-contiguous view. No repeated K/V heads, S-by-S probability tensor, output atomics, per-head KV gradient, or full partial-gradient workspace is permitted.

The stock AITER gfx1151 D512 backward requests 128 KiB LDS, while the accepted project path stays within 64 KiB. After final selection, the AITER fallback, score-disabled recomputation path, runtime toggles, and their launch tables were removed from the runtime module.

## Accepted schedule

The backward uses four deterministic launches in this order:

1. Value owner: key-owned split `dV` reads raw scores, reconstructs probabilities from FP32 LSE, and writes the compact shared-KV gradient buffer.
2. Query owner: grouped `dQ` computes Delta and sink partials, reconstructs probabilities, forms `dS`, accumulates direct `dQ`, and overwrites the now-dead FP16 score band with `dS`.
3. Key owner: one-dot `dK` reads FP16 `dS`, computes `scale * dS^T @ Q`, and adds it into the existing `dV` buffer.
4. Sink reducer: a fixed-order FP32 reduction emits `[64]`.

For visible probability `P` and `Delta_i = sum_d(O_i,d * dO_i,d)`:

```text
dP_ij = dO_i dot KV_j
dS_ij = P_ij * (dP_ij - Delta_i)
dQ_i  = scale * sum_j dS_ij * KV_j
dK_j  = scale * sum_i dS_ij * Q_i
dV_j  = sum_i P_ij * dO_i
dKV_j = dK_j + dV_j
```

The query owner reconstructs `sum(P * dP)` for the sink derivative. Replacing it with BF16-output-derived Delta is algebraically valid but was slower after compilation and increased sink RMSE by roughly 10x, so the reconstruction is retained.

## Launch tables

| Batch | `dQ` M/N | Head group | Warps | Waves/EU | `dV` M/N | Warps | Waves/EU | Unroll | `dK` M/N | Warps | Waves/EU | Unroll |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8/16 | 8 | 8 | compiler | 32/32 | 8 | 1 | 1 | 32/32 | 4 | 2 | 1 |
| 4 | 8/16 | 8 | 8 | compiler | 32/32 | 8 | 1 | 2 | 32/32 | 4 | 2 | 1 |
| 16 | 16/16 | 4 | 8 | 1 | 32/32 | 8 | 1 | 2 | 32/32 | 8 | compiler | 2 |

Traversal bounds remain dynamic and exact. Fixed nine-step query and five-step key loops issued excess masked work and were rejected.

## Correctness

Final maxima across B1/B4/B16 are:

| Quantity | Maximum relative RMSE | Minimum cosine |
|---|---:|---:|
| Q gradient | `0.002644` | `0.99999648` |
| Shared-KV gradient | `0.003026` | `0.99999541` |
| Sink gradient | `0.000234` | `0.99999988` |

Candidate screening used format-aware Q/shared-KV/sink RMSE limits of `0.005/0.005/0.003`. The retained production tests are tighter: `0.0030/0.0035/0.0005`, with cosine floors `0.99999/0.99999/0.999999`. The exact shared K=V cotangent is combined inside the key owners. Focused coverage includes production-strided `dO`, boundary rows, deterministic checkpoint replay, layout-copy guards, and unsupported-input failure.

## Performance

Authoritative final medians are:

| Batch | AITER backward | Project backward | AITER complete | Project complete | Complete speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | `84.400 ms` | `9.149 ms` | `97.976 ms` | `14.690 ms` | `6.67x` |
| 4 | `333.463 ms` | `35.903 ms` | `384.719 ms` | `57.696 ms` | `6.67x` |
| 16 | `1327.220 ms` | `145.404 ms` | `1553.851 ms` | `239.203 ms` | `6.50x` |

Relative to the pre-reopening project path, complete time falls from `42.739/173.067/696.944 ms` to `14.690/57.696/239.203 ms`. Relative to the output-only grouped-forward plus recomputation baseline `35.747/147.937/606.944 ms`, the raw-score handoff is `2.43x/2.56x/2.54x` faster.

A three-call device profile attributes approximately:

| Batch | Grouped `dQ` + Delta + sink | Split `dV+dK` | Sink reduce | Profiled backward/call |
|---:|---:|---:|---:|---:|
| 1 | `6.592 ms` | `2.558 ms` | `0.0017 ms` | `9.152 ms` |
| 4 | `26.483 ms` | `10.138 ms` | `0.0025 ms` | `36.623 ms` |
| 16 | `106.288 ms` | `37.218 ms` | `0.0038 ms` | `143.510 ms` |

Isolated tuning places `dV` near `1.12/4.37/17.55 ms` and one-dot `dK` near `1.31/5.22/19.65 ms`. Complete-boundary timings are authoritative because producer-consumer cache locality changed some isolated rankings.

After removing the obsolete persistent Delta workspace, incremental complete-boundary allocation is `290.563/1162.250/4648.500 MiB`, including the 32 MiB-per-batch-element score/`dS` band. The in-place handoff adds no second state workspace.

## Compiler evidence

Final retained variants compile to:

| Kernel family | LDS | VGPR range | Spill range | Warps | Assembly instruction lines |
|---|---:|---:|---:|---:|---:|
| Grouped forward | 64 KiB | 256 | 212-213 | 8 | 2380-3180 |
| Grouped `dQ`/Delta/sink | 64 KiB | 256 | 248-265 | 8 | 3253-3982 |
| Split `dV` | 32 KiB | 167-204 | 0 | 8 | 1065-1549 |
| One-dot `dK` | 32 KiB | 187-256 | 0-20 | 4 or 8 | 2680-2797 |
| Sink reduce | 32 B | 4-10 | 0 | 8 | 56-76 |

The score handoff removes the expensive two-dot key owner. The remaining dominant kernel is grouped `dQ`, where D512 FP32 accumulation reaches both the 64 KiB LDS ceiling and 256-VGPR limit.

## Work and roofline

Useful backward dot work is `66.589` GFLOP per batch element. Tile issuance is `79.054` GFLOP per batch element: grouped query ownership issues `1.125x` useful pairs and key ownership issues `1.250x`. Useful backward throughput is `7.28/7.42/7.33 TFLOP/s`; issued throughput is `8.64/8.81/8.70 TFLOP/s`.

At the complete boundary, useful throughput is `6.80/6.92/6.68 TFLOP/s` and issued throughput is `7.93/8.08/7.79 TFLOP/s`, only about 13% of the `59.4 TFLOP/s` WMMA peak. Masked issuance is therefore not the primary gap. The limiting factors are:

- 64 KiB LDS and 256 VGPRs in forward and `dQ`, allowing low occupancy;
- 212-265 spills in those D512 FP32-accumulator kernels;
- narrow N16/N32 local-window dots rather than large dense GEMMs;
- online-softmax `exp2`, reductions, masking, and state traffic outside WMMA accounting;
- serial traversal of 64 query heads in compact key owners;
- FP16 score/`dS` traffic, which is necessary to remove two larger QK recomputations.

Larger tiles exceed LDS, smaller accumulators and split dimensions lost absolute time, and lower precision failed either sink accuracy or complete-boundary value. The remaining roofline gap is explained by resource occupancy and mixed scalar/matrix work, not an untested launch parameter.

## Optimization history

Accepted progression:

- query-owned `dQ`/Delta/sink and key-owned compact shared-KV gradients;
- group-4 recomputation `dQ`, then larger groups after QK removal;
- compact raw FP16 scores removing both backward QK dots;
- split `dV` and `dK` after probability reconstruction made `dV` lightweight;
- `dV -> dQ/dS -> dK` ordering and in-place score-to-`dS` handoff;
- one-dot `dK` retuning and complete-boundary producer-consumer finalist selection.

Rejected branches include:

- fixed traversal counts;
- 8-row key tiles and earlier 8-row query ownership under recomputation;
- pre-handoff split or serial `dK/dV`;
- `dK`-first instruction ordering;
- `matrix_instr_nonkdim={16,32}` overrides;
- per-head KV atomics, repeated KV heads, and full partial-gradient workspaces;
- BF16 score state due sink cancellation;
- FP32 score state due doubled memory and slower complete time;
- normalized FP16 probabilities due bandwidth cost;
- output-derived Delta for sink due slower code and worse sink accuracy;
- lower-precision LSE/Delta/sink state.

All rejected normalization, alternate saved-state, score-disabled recomputation, fused `dK/dV`, and AITER fallback branches have been removed from production. The runtime contains only the accepted raw-score `dV -> dQ/dS -> dK` schedule. A post-cleanup 50-iteration verification measured `14.778/58.543/240.564 ms` complete at B1/B4/B16, within 1.5% of the accepted complete medians, with unchanged correctness and no geometry changes.

## Post-HCA review

The later CSA/HCA work does not expose a new sliding-specific owner. Sliding already has 64 local key tiles at B1 with nearly uniform 128-row work, so HCA's bounded compressed head partials do not address under-occupancy. Arithmetic packing would save only the short initial-window triangle, approximately 1 MiB per batch element, while disturbing the accepted 256-byte score-row alignment. CSA's two-phase dS/D256 dQ schedule won by replacing a 971-spill owner over a 512-entry triangular prefix with a 17-spill feature-sliced owner. Sliding has no compressed prefix, only 248-265 spills, and much less dQ traversal, so extra dS/K reads have no comparable repayment. HCA's measured split loss further confirms that a CSA win is not sufficient evidence to reopen lower-pressure families. Sliding remains closed.

The in-place score-to-dS handoff deliberately supports one backward per forward. Production checkpoint replay satisfies this contract; repeated backward through one retained graph is unsupported.

## External review and exhaustion

AITER's backward uses the same Delta/dS identities and independently retains a `delta_recomp` sink path; it offers no compact reusable `dS` handoff and its stock D512 launch exceeds local LDS. llama.cpp, DS4, vLLM, and SGLang remain forward/cache implementations without compatible Q/KV/sink training backward. Triton's available gfx1151 warp, wave, stage, and MFMA controls were screened; no additional control removes the measured accumulator pressure.

The final review found no viable unmeasured algorithm that preserves direct compact gradients, deterministic FP32 sinks, FP32 LSE/Delta, no atomics, no dense probabilities, and no additional large workspace. Backward tuning is closed.

## Validation and evidence

The focused suite passes `10 passed`. `py_compile`, Ruff, and `git diff --check` pass. The earlier physical-B1/S2048 real-model audit remains valid for integration and packed-payload immutability; a final focused checkpoint replay was rerun after the kernel changes.

Primary artifacts:

```text
/tmp/deepseek_v4_sliding_raw_score_handoff_production_final.json
/tmp/deepseek_v4_sliding_raw_score_handoff_metadata.json
/tmp/deepseek_v4_sliding_specialized_profile.json
/tmp/deepseek_v4_sliding_raw_score_complete_finalists.json
/tmp/deepseek_v4_sliding_grad_score_complete_finalists.json
/tmp/deepseek_v4_sliding_grad_score_dk_tuning.json
/tmp/deepseek_v4_sliding_raw_score_dq_tuning.json
/tmp/deepseek_v4_sliding_raw_score_dv_tuning.json
/tmp/deepseek_v4_sliding_grad_score_state_comparison.json
/tmp/deepseek_v4_sliding_sink_delta_identity.json
```
