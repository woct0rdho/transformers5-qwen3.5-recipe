# DeepSeek V4 HCA forward plan for gfx1151

## Status

Accepted and optimization-exhausted for the fixed S2048 training workload on Strix Halo `gfx1151`. Exact B1/B4/B16 component correctness, timing, allocation, compiler, checkpoint, and real 81 GiB GGUF integration gates pass.

Production code is in `deepseek_v4_hca.py`; family integration is in `deepseek_v4_attention.py`; focused correctness and timing are in `test_deepseek_v4_hca.py` and `benchmark_deepseek_v4_hca.py`.

## Fixed contract

The path is specialized to:

- physical batches 1, 4, and 16;
- sequence length 2,048;
- 64 query heads, one shared K=V head, and head dimension 512;
- BF16 projected activations, attention output, and activation gradients;
- FP32 sink, LSE, online-softmax state, producer reductions, and frozen checkpoint position bias/RMS scale where present;
- partial interleaved RoPE over the trailing 64 dimensions;
- local causal window 128, represented internally as `WINDOW_LEFT=127`;
- non-overlapping rate-128 compression to exactly 16 shared-KV entries;
- stateless training with canonical positions, no cache, no padding, and no dropout;
- `gfx1151` runtime dispatch only, signed-int32 element offsets, and at most 64 KiB LDS per workgroup.

The attention input is Q `[B,64,2048,512]`, local shared KV `[B,1,2048,512]`, compressed shared KV `[B,1,16,512]`, and sink `[64]`. It returns contiguous BSHD `[B,2048,64,512]`.

## Exact HCA semantics

Compression windows are independent and non-overlapping. For compressed entry `c` and feature `d`:

1. source rows are `128*c <= t < 128*(c+1)`;
2. compressor gate plus frozen position bias `[128,512]` is reduced with an FP32 featurewise softmax over 128 rows;
3. the FP32 weighted KV pool receives frozen weighted RMSNorm;
4. the normalized result is quantized to BF16 at the Transformers norm boundary;
5. trailing interleaved RoPE pairs are rotated at absolute position `128*c`.

Query row `t` sees compressed entry `c` iff `c < floor((t+1)/128)`. Rows 0-126 see no compressed entry, row 127 first sees entry zero, and row 2047 sees all 16 entries. HCA has no indexer, ranking, top-k, overlap, sentinel repair, or selective metadata.

## Accepted producer

One workgroup owns one `(batch, compressed entry)` row and distributes all 512 features over wave32 lanes. It streams the 128 source rows with per-feature FP32 maximum, denominator, and weighted sum, then performs one D512 FP32 RMS reduction and register-only RoPE. Every input row is read once by its unique output owner.

Accepted launch policy:

| Batch | Warps | waves/EU |
|---:|---:|---:|
| 1 | 4 | 0 |
| 4 | 8 | 1 |
| 16 | 8 | 2 |

Dynamic `tl.range` is deliberate. Fully unrolled `static_range(128)` expanded forward from a few hundred instructions to 8,625-17,440 instructions and failed the required B1 forward-plus-backward compile within 60 seconds. It was removed.

Measured producer-only complete medians are approximately `0.177/0.201/1.246 ms`. A multi-wave source-split/LDS merge or head-dimension split cannot improve the complete HCA boundary by more than about 0.4% even if producer time vanished, while it would add a pool workspace, another finalize launch, and a more complex backward merge. Those inference-oriented AITER/vLLM candidates are therefore not viable at this training boundary.

Compiler metadata for the retained producer has 16-32 bytes LDS, 24-42 forward registers, 27-49 backward registers, zero spills, and zero global scratch.

## Accepted two-region forward

One grouped query owner traverses:

1. local keys `max(0,t-127) <= k <= t`;
2. compressed prefix `0 <= c < floor((t+1)/128)`;
3. one FP32 per-head sink logit.

All regions share one FP32 online-softmax maximum, denominator, and D512 output accumulator. There are 254,016 local and 15,376 compressed visible pairs per head, or 269,392 total. The kernel does not concatenate KV, repeat the shared KV head, read a block mask, build `[B,1,2048,2064]` bias, or materialize probabilities.

Forward stores raw log2-domain scores as two FP16 states:

- local `[B,64,2048,128]` with a 256-byte row stride;
- compressed `[B,64,2048,16]` with a 32-byte row stride.

Together they retain exactly 144 slots, or 36 MiB per batch element. Backward overwrites both states with `dS`. Splitting the physical states from a unified 144-slot row preserved allocation and correctness while improving the same-table complete boundary from about `18.80/73.88/307.19 ms` to `18.77/72.90/304.91 ms` before the final nearby launch check.

Accepted forward launch table:

| Batch | Query M | Head group | Local N | Compressed N | Warps | waves/EU |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 8 | 16 | 16 | 8 | 1 |
| 4 | 8 | 8 | 16 | 16 | 8 | 1 |
| 16 | 4 | 16 | 16 | 16 | 8 | 0 |

B1/B4 retain this combined owner. B16 now uses an M16/group-4, four-warp, two-wave QK/LSE owner followed by M16/group-4, four-warp, one-wave D256 output owners. The B1/B4 combined owner uses 64 KiB LDS, 256 registers, and 255-261 spills. B16 QK/LSE uses 64 KiB LDS, 256 registers, and 124 spills; D256 output uses 8 KiB LDS, 256 registers, and 10 spills. None uses global scratch.

## Correctness and performance

The independent oracle uses FP32 logits/softmax, BF16 probability/value products, the Transformers BF16 norm boundary, and autograd. Exact final B1/B4/B16 relative RMSE is:

| Tensor | B1 | B4 | B16 | Minimum cosine |
|---|---:|---:|---:|---:|
| Attention output | 0.003405 | 0.003403 | 0.002951 | 0.999994 |
| Producer output | 0.002387 | 0.002246 | 0.002310 | 0.999997 |
| Producer KV gradient | 0.002410 | 0.002417 | 0.002440 | 0.999997 |
| Producer gate gradient | 0.002391 | 0.002403 | 0.002415 | 0.999997 |

Threshold rows 0, 126, 127, 128, and 2047 are explicitly covered. Frozen FP32 bias/RMS weight and strided bias/RoPE metadata also pass.

Final producer-plus-attention measurements from `/tmp/deepseek_v4_hca_phase_final.json` are:

| Batch | Forward | Complete F+B | Incremental peak allocation |
|---:|---:|---:|---:|
| 1 | 6.459 ms | 18.226 ms | 295.157 MiB |
| 4 | 26.621 ms | 71.579 ms | 1,179.125 MiB |
| 16 | 59.588 ms | 190.114 ms | 4,715.000 MiB |

B1/B4 remain on the combined owner and move only within measurement variance. Against the previous accepted B16 boundary, phase splitting improves forward by 46.1% and complete time by 39.6% without increasing allocation. Relative to tuned FlexAttention complete time `217/616/2630 ms`, HCA is now `11.9x/8.6x/13.8x` faster while retaining the full producer and training backward. The portable forward-only AITER DSV4 CSR reports `91.30 ms` at B16, versus `59.588 ms` for this training forward including raw-score handoff.

## Rejected forward/state candidates

- Unified 144-slot score rows lost to separate 128/16 states at unchanged element count.
- Padding unified score rows to 152, 160, 176, 192, or 256 slots did not improve the complete boundary; B16 complete regressed from about 308.0 ms to 312.2-326.0 ms.
- M2/group-32 and M32/group-2 forward ownership lost to M4/M8 finalists; 32-row owners doubled program overhead, while larger query M traversed more local keys.
- 32-row logical ownership with four/eight warps lost by 8-52% depending on batch.
- Fully unrolled C128 producer loops violated the 60-second compilation bound and caused extreme instruction growth.
- Producer head-dimension splitting, AITER-style multi-wave LDS source splitting, and split pool/norm finalize cannot materially improve a producer below 0.4% of the boundary and would add state/launches.
- FP32 score state, normalized probability state, and FP8 state are not carried as dormant branches. Accepted sliding/CSA evidence already shows FP32 does not repair BF16 D512 reduction order, normalized state adds a bandwidth pass, and FP8 materially damages gradients.
- Materialized concatenated KV, repeated heads, dense bias/probabilities, generic mask interpretation, cache paths, and eager fallback are excluded.

## Integration evidence

`/tmp/deepseek_v4_training_hca_report.json` passes with:

- exactly 2 sliding, 21 CSA, and 20 HCA modules configured;
- all 43 decoder layers checkpointed with `use_reentrant=False`;
- finite loss and gradients;
- all 469 LoRA-B tensors nonzero after the first update;
- zero grouped frozen-output gradients;
- unchanged sampled packed pointers, versions, and SHA-256 payload hashes;
- warmed second forward/backward of `6.064/12.977 s`, versus `7.467/17.178 s` before fused HCA;
- peak allocation/reservation of `87.625/88.441 GiB`, versus `90.624/93.232 GiB` before fused HCA;
- zero process swap.

## Reopened B16 operation-phase experiment

CSA's accepted B16 QK/LSE plus D256 output ownership changes the evidence behind HCA's forward closure. HCA's combined forward has the same 64 KiB LDS, 256-register, 255-261-spill D512 accumulator pattern as pre-split CSA, while sustaining only about 5.1 useful TFLOP/s.

Test a B16-only two-phase schedule:

1. one grouped owner computes local and compressed QK, writes the existing separate raw-score states, and carries the shared local/compressed/sink max and denominator to FP32 LSE without a D512 output accumulator;
2. D128/D256 feature-sliced owners reconstruct normalized probabilities from raw scores/LSE, traverse both KV regions, and write disjoint BF16 output slices;
3. no probability tensor, second score state, output partial workspace, repeated KV head, or QK recomputation is permitted.

Screen M16/group-4 with four warps and zero/one/two waves first, then bounded M8/group-8 and D128 alternatives only where compilation remains below 60 seconds. Acceptance requires unchanged exact threshold-row and B1/B4/B16 gates, bitwise non-reentrant checkpoint replay, no material allocation increase, at most 64 KiB LDS, zero global scratch, and at least a 5% complete producer-plus-attention improvement against `314.660 ms`. Losing kernels, launch tables, and selectors must be removed after recording evidence.

### Result: accepted

M16/group-4 QK/LSE with four warps and two waves plus M16/group-4 D256 output with four warps and one wave is retained for B16. The first candidate measured `59.476/202.483/261.959 ms` forward/backward/complete before the backward reopening, versus `110.599/204.061/314.660 ms` previously. Zero/one/two-wave controls clustered near `59.4-64.9 ms` forward; D128 measured `68.529 ms`, M8/group-8 QK measured `189.243 ms`, and M8/group-8 output measured `61.666 ms`. Losing variants were removed.

Exact B16 output/Q/local-KV/compressed-KV/sink RMSE after both accepted phases is `0.002951/0.003205/0.005225/0.011517/0.001214`. Production B16 checkpoint replay is bitwise equal for output and all four attention gradients. Evidence is in `/tmp/deepseek_v4_hca_phase_screen.json`, `/tmp/deepseek_v4_hca_forward_phase_b16.json`, `/tmp/deepseek_v4_hca_phase_final.json`, `/tmp/deepseek_v4_hca_phase_checkpoint_b16.json`, and `/tmp/deepseek_v4_hca_phase_metadata.json`.

## Final resource review

Transformers remains the semantic authority. AITER's wave32 gfx1250 FlyDSL compressor contributed the multi-wave source-split candidate, but its architecture-gated inference wrapper was not copied. vLLM and SGLang confirm C128 featurewise compression, occupancy splitting, fused norm/RoPE, and online reduction, but provide no matching training cotangent boundary. llama.cpp confirms deterministic HCA positions and local-plus-compressed graph semantics. DS4 and AITER sparse attention confirm one online-softmax state across regions. Accepted sliding/CSA work supplied the grouped compact-MQA, raw-score handoff, and direct shared-KV ownership patterns.

No viable unmeasured forward candidate remains. The retained B16 schedule is limited by 269,392 visible D512 pairs per head, one additional KV traversal, raw-score writes/reads, and scalar online-softmax work, but it removes the long-lived D512 output accumulator from QK/LSE and halves output feature ownership. Reopening HCA should require a larger fusion that removes neighboring projection/RoPE/output work with complete-boundary evidence.
