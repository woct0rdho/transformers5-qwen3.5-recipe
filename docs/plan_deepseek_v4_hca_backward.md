# DeepSeek V4 HCA backward plan for gfx1151

## Status

Accepted and optimization-exhausted for exact physical batches 1/4/16 at S2048 on `gfx1151`. Custom backward covers fused local-plus-16-entry attention and the non-overlapping rate-128 producer. Exact component, non-reentrant checkpoint, compiler/allocation, and real GGUF update gates pass. The conditional B16 local/compressed dQ split was measured after CSA validated phase splitting, failed its 5% gate, and was removed.

## Gradient contract

Attention backward returns:

- BF16 `dQ [B,64,2048,512]`;
- one combined BF16 local shared-KV gradient `[B,1,2048,512]` because local K=V is one activation;
- one combined BF16 compressed shared-KV gradient `[B,1,16,512]` because compressed K=V is one activation;
- FP32 `dSink [64]`.

Producer backward consumes combined `dCompressedKV` and returns BF16 gradients for compressor KV and gate projections `[B,2048,512]`. Frozen position bias, RMS scale, and RoPE tables receive no gradient. Calls with trainable frozen metadata fail closed.

Model-native Q/KV storage is BHSD and internal views are BSHD. Production `dO` may be a last-dimension-contiguous non-contiguous view. Layouts requiring a copy fail closed instead of calling `.contiguous()`. Inputs and output gradients are checked for signed-int32 element-offset fit and `gfx1151` dispatch.

## Accepted attention schedule

Forward retains separate FP16 raw-score states for 128 local and 16 compressed slots. Backward uses this ordered handoff:

1. local key-owned `dV` reconstructs probabilities from local scores and shared FP32 LSE;
2. compressed head-group owners reconstruct compressed probabilities and write bounded FP32 dV partials;
3. one deterministic compact reducer writes combined compressed dV;
4. one grouped query owner computes `Delta = sum_d O*dO`, reconstructs both regions, accumulates one dQ, writes deterministic sink partials, and overwrites both score states with FP16 dS;
5. local key-owned one-dot dK consumes local dS and adds into local dV;
6. compressed head-group owners consume compressed dS and reuse the same compact FP32 partial buffer for dK;
7. the compact reducer adds scaled dK into compressed dV;
8. one deterministic FP32 sink reduction combines query-tile partials.

Local, compressed, and sink logits share one softmax. There are no atomics, per-head KV outputs, repeated KV heads, dense probabilities, persistent Delta, a second score/dS state, or query-sized partial-gradient workspaces.

### Compact partial ownership

The initial direct all-head compressed owner launched only one workgroup per batch and cost `4.80/5.48/11.05 ms` for dV and `7.44/7.80/10.12 ms` for dK in profiler runs. It was especially under-occupied at B1.

The accepted schedule splits only the 64 query heads. Each head group writes `[16,512]` FP32 partials, and one deterministic row reducer combines them. The same buffer is reused after dQ for dK. Workspace is bounded to `0.5/0.5/2.0 MiB` at B1/B4/B16 and is not proportional to query sequence length. B1 compressed owners fall to about `0.55 ms` each; B4 to `2.20 ms`; B16 to `8.7-8.8 ms`.

## Accepted launch tables

Grouped query owner:

| Batch | Query M | Head group | Local N | Compressed N | Warps | waves/EU |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 16 | 16 | 16 | 8 | 0 |
| 4 | 4 | 16 | 16 | 16 | 8 | 0 |
| 16 | 8 | 8 | 16 | 16 | 8 | 0 |

B1/B4 M4 doubles sink query tiles relative to M8 but wins the repeated complete boundary. B16 M8/group-8 uses 1 MiB sink partials. Interleaved B16 M4/M8 zero/one-wave candidates cluster within 0.2%; M8/group-8 with zero waves is retained for the smaller sink state. Two waves/EU is consistently slower.

Local shared-KV owners reuse the accepted sliding geometry after HCA-specific complete-boundary validation:

| Batch | Kernel | Query M | Key N | Warps | waves/EU | Head unroll |
|---:|---|---:|---:|---:|---:|---:|
| 1 | dV | 32 | 32 | 8 | 1 | 1 |
| 1 | dK | 32 | 32 | 4 | 2 | 1 |
| 4 | dV | 32 | 32 | 8 | 1 | 2 |
| 4 | dK | 32 | 32 | 4 | 2 | 1 |
| 16 | dV | 32 | 32 | 8 | 1 | 2 |
| 16 | dK | 32 | 32 | 8 | 0 | 2 |

Compressed partial owners:

| Batch | Kernel | Query M | Key N | Head group | Warps | waves/EU |
|---:|---|---:|---:|---:|---:|---:|
| 1 | dV | 32 | 16 | 4 | 4 | 2 |
| 1 | dK | 32 | 16 | 4 | 4 | 2 |
| 4 | dV | 32 | 16 | 16 | 8 | 2 |
| 4 | dK | 32 | 16 | 16 | 8 | 2 |
| 16 | dV | 16 | 16 | 16 | 8 | 2 |
| 16 | dK | 16 | 16 | 16 | 8 | 2 |

Each compressed key `c` starts at query `128*c+127`; visibility remains arithmetic within every query tile. Reductions use four warps and no LDS.

## Producer backward

One program reconstructs the 128 featurewise gate logits and pooled pre-norm vector in FP32, reverses compressed-position RoPE, reverses weighted RMSNorm, and recomputes softmax probabilities. For each source row:

- `dKV_i = p_i * dPooled`;
- `dGate_i = p_i * dPooled * (KV_i - pooled)`.

Every source row belongs to exactly one compression window. Gradients are direct unique writes with no atomics, zero fill, or reduction workspace. Dynamic C128 loops retain compact code and complete within the normal Triton compile path; static unrolling was removed after violating the 60-second compile bound.

## Correctness gates

Exact final B1/B4/B16 relative RMSE against the independent FP32-logit autograd oracle:

| Tensor | B1 | B4 | B16 | Minimum cosine |
|---|---:|---:|---:|---:|
| Q gradient | 0.003180 | 0.003173 | 0.003174 | 0.999995 |
| Local shared-KV gradient | 0.005217 | 0.005221 | 0.005221 | 0.999986 |
| Compressed shared-KV gradient | 0.011834 | 0.011538 | 0.011515 | 0.999930 |
| Sink gradient | 0.001709 | 0.001746 | 0.001214 | 0.999998 |
| Producer KV gradient | 0.002410 | 0.002417 | 0.002440 | 0.999997 |
| Producer gate gradient | 0.002391 | 0.002403 | 0.002415 | 0.999997 |

The compact-KV threshold is intentionally looser than local/Q because each of only 16 vectors accumulates up to 1,921 query rows across 64 heads. The residual is BF16 D512 dot/reduction ordering, consistent across exact batches. All outputs and gradients are finite.

Direct and non-reentrant checkpoint outputs plus all four attention gradients are bitwise equal, including the production B16 phase schedule. Unsupported dO layouts are rejected without a copy. Final evidence is in `/tmp/deepseek_v4_hca_phase_final.json` and `/tmp/deepseek_v4_hca_phase_checkpoint_b16.json`.

## Timing, allocation, and profile

Final producer-plus-attention medians:

| Batch | Forward | Backward-only | Complete | Incremental peak allocation |
|---:|---:|---:|---:|---:|
| 1 | 6.459 ms | 11.767 ms | 18.226 ms | 295.157 MiB |
| 4 | 26.621 ms | 44.958 ms | 71.579 ms | 1,179.125 MiB |
| 16 | 59.588 ms | 130.526 ms | 190.114 ms | 4,715.000 MiB |

Against the previous accepted B16 boundary, the operation phases improve backward by 36.0% and complete time by 39.6%. The final one-run profiler attribution is:

| Kernel | B1 | B4 | B16 |
|---|---:|---:|---:|
| Local dV | 1.19 ms | 4.63 ms | 16.66 ms |
| Compressed dV partial/reduce | 0.55 ms | 2.20 ms | 8.82 ms |
| Query dS/Delta/sink | 8.03 ms | 33.33 ms | 56.15 ms |
| Query dQ from dS | included above | included above | 22.05 ms |
| Local dK | 1.26 ms | 4.99 ms | 18.91 ms |
| Compressed dK partial/reduce | 0.55 ms | 2.22 ms | 8.66 ms |
| Producer backward | 0.09 ms | 0.13 ms | 0.71 ms |

Profiler overhead and sustained GPU state make its totals larger than event-timed medians. The B16 query-gradient boundary falls from `149.17 ms` in the pre-split profile to `78.20 ms` across dS and D256 dQ.

## Compiler evidence

Final production-only metadata is `/tmp/deepseek_v4_hca_phase_metadata.json`:

- B1/B4 grouped forward: 64 KiB LDS, 256 registers, 255-261 spills, zero scratch;
- B16 QK/LSE: 64 KiB LDS, 256 registers, 124 spills, zero scratch;
- B16 D256 output: 8 KiB LDS, 256 registers, 10 spills, zero scratch;
- B1/B4 grouped dQ: 64 KiB LDS, 256 registers, 338-351 spills, zero scratch;
- B16 dS/Delta/sink: 64 KiB LDS, 256 registers, 334 spills, zero scratch;
- B16 D256 dQ: 8 KiB LDS, 256 registers, zero spills/scratch;
- local dV/dK: 32 KiB LDS, 166-256 registers, zero spills except 19 in B1 dK, zero scratch;
- compressed dV/dK: 16-32 KiB LDS, 98-256 registers, zero spills/scratch;
- compact reducer: zero LDS, 17-67 registers, zero spills/scratch;
- producer: 16-32 bytes LDS, 24-49 registers, zero spills/scratch;
- sink reducer: 32 bytes LDS, 4-18 registers, zero spills/scratch.

No retained launch exceeds the 64 KiB LDS limit or uses global scratch.

## Optimization record

Retained:

- separate local/compressed FP16 raw-score states overwritten in place with dS;
- local Delta with no persistent workspace;
- direct local dV plus one-dot dK;
- bounded deterministic compressed head-group partials reused for dV and dK;
- one combined dQ owner over local, compressed, and sink regions at B1/B4;
- B16 operation-phase dS/Delta/sink ownership followed by direct D256 dQ slices;
- deterministic compact and sink reductions;
- dynamic C128 producer recomputation with direct unique writes;
- exact B1/B4/B16 query/head, M/N, warp, and waves/EU tables.

Rejected:

- direct all-head compressed key owners were under-occupied and made initial B1 complete time 31.52 ms;
- compressed N4/N8 tiles lost to N16 at every batch because extra launches and MFMA padding outweighed tighter causal starts;
- head groups 1/2/8/16 lost at B1; group 16 wins B4/B16 while minimizing workspace;
- compressed M8 incurred traversal/launch overhead; M64 caused instruction growth and regressions; M32/M16 finalists remain;
- query M2/M32 and 32-row logical owners lost through extra programs or wider local traversal;
- B16 two-wave dQ lost to zero/one-wave variants in interleaved complete tests;
- splitting local and compressed dQ was measured after CSA's phase split succeeded; screened M4/group-16, M8/group-8, and M16/group-4 schedules measured best at `327.8 ms` complete versus accepted `314.1-314.7 ms`, so rereading output/dO/LSE/Delta and the BF16 accumulation boundary did not repay the shorter owner;
- fusing compressed dV/dK is impossible without retaining probabilities across dQ or recomputing them: dV needs raw score/LSE before dQ overwrites state, while dK needs dS afterward;
- atomics and query-sized partial workspaces were excluded; the accepted compact partial is only 0.5/0.5/2 MiB;
- FP32/FP8/probability score formats, repeated KV heads, dense masks/probabilities, persistent Delta, and eager fallback remain removed;
- LSE-only recomputation remains a future B16 low-memory policy only if end-to-end attribution identifies HCA's 576 MiB score state as a limit.

## Integration evidence

`/tmp/deepseek_v4_training_hca_report.json` passes:

- all 20 HCA modules and all previously accepted sliding/CSA modules dispatch;
- 43 non-reentrant decoder checkpoints;
- zero missing/nonfinite expected gradients;
- all 469 LoRA-B tensors update after the first optimizer step;
- immutable sampled packed pointers, versions, and hashes;
- no grouped frozen-output gradients;
- warmed full-model second forward/backward `6.064/12.977 s`, improved from post-CSA `7.467/17.178 s`;
- peak allocation/reservation `87.625/88.441 GiB`, down from `90.624/93.232 GiB`;
- zero process swap.

## B16 local/compressed dQ experiment result

CSA's accepted phase split triggered the conditional HCA screen. The candidate kept the local owner as the first direct BF16 dQ writer, then launched one compressed owner to recompute Delta, overwrite compressed scores with dS, add compressed dQ into final BF16 dQ, and add deterministic sink partials. It used no full dQ partial and all tested static variants compiled within the 60-second cap.

The natural M8/group-8 schedule measured `338.6 ms` complete. M16/group-4 variants measured `334.3-414.2 ms`; M4/group-16 was best at `327.8 ms`. All lose to the accepted combined owner's `314.1-314.7 ms`, so the candidate misses the required 5% win before correctness becomes a retention question. The experimental kernel, selector, and launch table were removed. HCA retains one combined local/compressed dQ owner. The screen record is `/tmp/deepseek_v4_hca_dq_split_screen.json`.

## Reopened B16 dS/D256 dQ experiment

The rejected local/compressed dQ split duplicated Delta, output, dO, and LSE work to shorten ownership for only 5.7% of HCA's visible pairs. CSA's accepted operation-phase split suggests a different B16-only schedule:

1. one grouped owner computes Delta, both probability regions, local and compressed dS, and deterministic sink partials without a D512 dQ accumulator;
2. D128/D256 feature-sliced owners consume both existing dS states and both KV regions and write disjoint final BF16 dQ slices directly;
3. existing local and compressed dV/dK ownership remains unchanged, and no dQ partial workspace or probability recomputation is introduced.

Start from M16/group-4, four warps for dS and D256 dQ, with zero/one/two waves. Screen M8/group-8 and D128 only if the first candidate is competitive and compiles within 60 seconds. Acceptance requires current exact output/Q/local-KV/compressed-KV/sink/producer gates, bitwise B16 non-reentrant checkpoint replay, no material allocation increase, at most 64 KiB LDS, zero global scratch, and at least a 5% complete producer-plus-attention improvement after any accepted forward change. Losing kernels and selectors must be removed.

### Result: accepted

M16/group-4, four-warp, zero-wave dS/Delta/sink plus M16/group-4, four-warp, one-wave D256 dQ is retained at B16. It measures `131.337 ms` backward and `190.888 ms` complete in the exact candidate report; final production medians are `130.526/190.114 ms`. Zero/one/two-wave controls placed backward around `131.5-136.8 ms`; D128 measured `146.311 ms`, M8/group-8 dS measured `296.801 ms`, and M8/group-8 dQ measured `139.106 ms`. Losing variants were removed.

The retained schedule does not recompute probabilities in dQ and introduces no dQ partial workspace. Its M16 sink partial reduces B16 incremental allocation by 0.5 MiB. Exact B1/B4/B16 gates pass, and B16 checkpoint output plus Q/local-KV/compressed-KV/sink gradients are bitwise equal. Evidence is in `/tmp/deepseek_v4_hca_phase_screen.json`, `/tmp/deepseek_v4_hca_backward_phase_b16.json`, `/tmp/deepseek_v4_hca_phase_final.json`, `/tmp/deepseek_v4_hca_phase_profile.txt`, `/tmp/deepseek_v4_hca_phase_checkpoint_b16.json`, and `/tmp/deepseek_v4_hca_phase_metadata.json`.

## Autograd boundary note

The raw-score-to-dS overwrite supports one backward per forward. Production checkpoint replay satisfies that contract; repeated backward through one retained graph is unsupported.

## Remaining bottleneck

HCA backward launch and ownership tuning is optimization-exhausted for the retained contract. B16 dS must reconstruct local and compressed probabilities and compute dP/dS over 17.24 million visible head-pairs per batch; D256 dQ then rereads dS and both KV regions. The dS owner still reaches 64 KiB LDS and 334 spills, but separating D512 dQ reduces the query-gradient profile from `149.17` to `78.20 ms`. Smaller ownership increases KV/program traffic; larger ownership expands traversal/register pressure; the measured local/compressed split repeats output/dO/LSE/Delta traffic; direct compact gradients under-occupy; atomics/full partials violate the contract.

No available AITER, vLLM, SGLang, llama.cpp, DS4, or Triton path provides a reusable compact shared-KV HCA training backward. Reopen this stage only for a larger projection/RoPE/output fusion or a measured B16 memory policy with complete-model evidence.
