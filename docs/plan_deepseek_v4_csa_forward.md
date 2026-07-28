# DeepSeek V4 CSA forward plan for gfx1151

## Status

Accepted and optimization-exhausted for the fixed S2048 training workload on Strix Halo `gfx1151`. B1/B4 retain the combined online-softmax/output owner with dense compressed-score rows. B16 uses exact arithmetic-packed triangular scores plus separate QK/LSE and D256 output owners. Component correctness, allocation, compiler, non-reentrant checkpoint, and the prior real 81 GiB GGUF integration gates pass. A new full-model B4/B16 audit remains part of the later adapter stage.

Production code is in `deepseek_v4_csa.py`. Module-family integration is in `deepseek_v4_attention.py`. Focused correctness and timing live in `test_deepseek_v4_csa.py` and `benchmark_deepseek_v4_csa.py`.

## Fixed contract

This path is specialized to:
- physical batches 1, 4, and 16.
- sequence length 2,048.
- 64 query heads, one shared K=V head, and head dimension 512.
- BF16 projected activations, attention output, and activation gradients.
- FP32 sink, LSE, online-softmax state, producer reductions, and frozen checkpoint position bias/RMS scale where present.
- partial interleaved RoPE over the trailing 64 dimensions.
- local causal window 128, represented internally as `WINDOW_LEFT=127`.
- rate-4 compression to 512 shared-KV entries.
- stateless training with canonical positions, no cache, no padding, and no dropout.

The final attention input is Q `[B,64,2048,512]`, local shared KV `[B,1,2048,512]`, compressed shared KV `[B,1,512,512]`, and sink `[64]`. The attention output is contiguous BSHD `[B,2048,64,512]`.

## Indexer-elimination proof

At the supported shape:
- `compressed_len = 2048 / 4 = 512`.
- configured `index_topk = 512`.
- the eager indexer scatters selected identities into a visibility mask, so ordering of all 512 selected identities cannot change attention.
- only causal readiness remains: query `t` sees compressed entry `c` iff `c < floor((t+1)/4)`.

Production therefore bypasses indexer projections, indexer compression, scoring, ReLU/weighted reduction, top-k, sentinel repair, and dense block-bias construction. Dispatch verifies S2048, rate 4, top-k 512, canonical positions, no cache/padding, and the fixed model geometry. A selective `index_topk < compressed_len` configuration fails closed.

The real audit confirms all 21 CSA instances were patched. The profile contains no indexer or top-k operation. The project adapter target set does not require inactive indexer adapters. All 469 expected LoRA-B tensors still update.

## Accepted producer

The producer consumes existing packed/LoRA projection outputs `[B,2048,1024]` for KV and gate. Packed dequantization and rank-4 LoRA remain outside this kernel.

For compressed entry `c` and every feature independently:
- slots 0-3 read Ca from raw window `c-1`. Entry zero supplies zero KV and `-inf` gate.
- slots 4-7 read Cb from raw window `c`.
- the corresponding row of frozen `[4,1024]` position bias is added in FP32.
- an FP32 online softmax over valid slots weights an FP32 KV reduction.
- frozen weighted RMSNorm is evaluated in FP32.
- interleaved RoPE rotates the trailing 64 dimensions at absolute position `4*c`.
- the compressed vector is stored in BF16.

One program owns one `(batch, compressed entry)` row. RMSNorm output is quantized at the required BF16 boundary, reshaped into interleaved pairs in registers, rotated, and stored once. This replaces the earlier global store/reload and `tl.debug_barrier()` workaround. Producer backward reconstructs the pool, directly writes unique previous-Ca/current-Cb owners into empty gradient tensors, and explicitly zeros only the final four unused Ca rows.

Accepted producer launch policy:

| Batch | Warps | waves/EU |
|---:|---:|---:|
| 1 | 8 | 2 |
| 4 | 8 | 2 |
| 16 | 8 | 2 |

The post-cleanup screen measured complete producer medians of approximately `0.229/0.789/2.651 ms`. Producer cost is below one percent of the complete CSA boundary. Accepted compiler metadata has 32 bytes LDS, 84/106 forward/backward registers, no global scratch, and no spills.

## Accepted two-region forward

Every batch traverses one shared softmax over:
- recent local keys `max(0,t-127) <= k <= t`.
- compressed prefix `0 <= c < floor((t+1)/4)`.
- one FP32 per-head sink logit in the same denominator.

B1/B4 use one grouped query owner carrying FP32 max, denominator, and D512 output accumulator. B16 splits ownership: a QK/LSE owner carries only max/denominator while writing raw scores, then two D256 output owners reconstruct normalized probabilities from scores/LSE and write disjoint output feature slices. The split creates no probability tensor, second score state, output partial workspace, repeated KV head, or QK recomputation.

Raw log2-domain FP16 scores use separate local and compressed states and are overwritten in place with `dS` by backward. B1/B4 retain a dense 128-slot local band plus direct `[2048,512]` compressed rows, totaling 160 MiB per batch element. B16 retains the same 32 MiB local band but arithmetic-packs exactly 523,776 causally ready compressed entries per head, so total score state is 95.9375 MiB per batch element.

Accepted launch table:

| Batch | Phase | Query M | Head group | Feature D | Local N | Compressed N | Warps | waves/EU |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | Combined | 16 | 4 | 512 | 16 | 16 | 8 | 0 |
| 4 | Combined | 16 | 4 | 512 | 16 | 16 | 8 | 0 |
| 16 | QK/LSE | 16 | 4 | 512 | 16 | 16 | 4 | 2 |
| 16 | Output | 16 | 4 | 256 | 16 | 16 | 4 | 2 |

B1/B4 combined forward uses 64 KiB LDS, 256 registers, and 267 spills. B16 QK/LSE uses 64 KiB LDS, 256 registers, and 106 spills. Its output owner uses 8 KiB LDS, 256 registers, and 10 spills. No launch uses global scratch.

## Correctness and performance

The blockwise oracle uses FP32 logits/softmax, one shared sink denominator, BF16 probability/value products, and autograd. Relative RMSE is stable across exact batches:

| Tensor | B1 | B4 | B16 | Minimum cosine |
|---|---:|---:|---:|---:|
| Attention output | 0.003442 | 0.003441 | 0.003061 | 0.999994 |
| Producer output | 0.001992 | 0.001984 | 0.002015 | 0.999997 |
| Producer KV gradient | 0.002034 | 0.002030 | 0.002052 | 0.999997 |
| Producer gate gradient | 0.001777 | 0.001751 | 0.001778 | 0.999998 |

The retained focused tests use measured BF16 gates rather than the tighter sliding thresholds. The larger compact-KV reduction errors are documented in the backward plan. A production-shaped B16 non-reentrant checkpoint replay is bitwise equal for output and all four attention gradients with finite state. Its report is `~/tmp/test_no_unsloth/deepseek_v4_csa_forward_phase_checkpoint_b16.json`.

Final producer-plus-attention medians from `~/tmp/test_no_unsloth/deepseek_v4_csa_forward_phase_final.json`:

| Batch | Forward | Backward | Complete F+B | Incremental peak allocation |
|---:|---:|---:|---:|---:|
| 1 | 15.464 ms | 24.729 ms | 40.193 ms | 451.563 MiB |
| 4 | 60.201 ms | 97.693 ms | 157.894 ms | 1,806.250 MiB |
| 16 | 109.168 ms | 323.732 ms | 432.900 ms | 6,199.500 MiB |

The B1/B4 allocation increase is the accepted reusable deterministic compressed-gradient workspace described in the backward plan. At B16 that 512 MiB workspace is more than repaid by 1,025 MiB of packed score-state savings, for a net 513 MiB reduction versus the earlier direct-owner boundary.

The portable AITER DSV4 forward reference is `14.37/57.34/224.49 ms`. It remains 5-8% faster at B1/B4 because it emits no training score handoff, but the retained B16 phase split is about 2.06 times faster while still emitting raw scores and LSE. AITER supplies no reusable training backward/state ownership, so it remains an inference reference rather than a substitute.

Against the pre-CSA real-model audit `~/tmp/test_no_unsloth/deepseek_v4_training_sliding_specialized_report.json`, the cleaned fused path changes warmed full-model second forward from `9.612 s` to `7.467 s`, backward from `22.909 s` to `17.178 s`, peak allocation from `91.654` to `90.624 GiB`, and peak reservation from `96.133` to `93.232 GiB`. The accepted cleanup report is `~/tmp/test_no_unsloth/deepseek_v4_training_csa_cleanup_report.json`.

## Rejected forward/state candidates

- Compressed N32 forward tiles were 32-37% slower than N16 at all batches.
- Alternate equal-64-row query/head groupings were close. M16/group-4 won B1/B4 and M4/group-16 won B16 at the complete boundary.
- FP32 raw scores doubled state and changed Q/local/compressed/sink gradient RMSE only from approximately `0.003074/0.005013/0.009931/0.001153` to `0.003043/0.004899/0.009888/0.001138`. The error is dot/reduction order, not FP16 state quantization.
- FP8 E4M3 score state preserved forward output because the state is consumed only by backward, but degraded Q/local/compressed/sink gradient RMSE to `0.0508/0.0705/0.0894/0.0213`. It was rejected.
- A full raw-score-to-normalized-FP16 probability pass preserved accuracy but increased complete medians from about `53.45/186.59/906.90 ms` to `56.23/195.21/934.31 ms`. Saved exponentials did not repay the pass.
- Separate local and compressed query-owner launches increased complete medians to `58.01/201.93/1089.87 ms` by rereading output, `dO`, LSE, and Delta state.
- Fusing packed projection/dequantization into compression was rejected structurally: it would duplicate accepted packed operators and obscure LoRA input-gradient ownership for a sub-one-percent producer.
- Materialized concatenated KV, dense `[B,1,2048,2560]` bias, CSR indices, repeated 64-head KV, and dense probability tensors were never viable production candidates.

## Integration evidence

The real physical-B1/S2048 81 GiB GGUF audit passed with:
- exactly 2 sliding, 21 fused CSA, and 20 eager HCA layer instances.
- all 43 decoder layers checkpointed with `use_reentrant=False`.
- finite loss and gradients.
- all 469 expected LoRA-B tensors updated.
- zero grouped frozen-output gradients.
- unchanged sampled packed pointers, versions, and SHA-256 payload hashes.
- 90.628 GiB peak allocated, 93.232 GiB peak reserved, and zero process swap.

The first two failed integration attempts were strict-contract findings: frozen position bias/RMS scale are FP32 in the checkpoint, and frozen position/RoPE metadata may be strided. Production now accepts BF16 or FP32 only for the frozen bias/scale and honors explicit metadata strides. It still requires contiguous-last-dimension BF16 projected activations and introduces no copies.

## Packed triangular compressed-score result

The experiment separated the dense 128-slot local score band from compressed scores and screened exact packing plus 16-, 64-, and 512-score row alignment. Exact packing stores `sum_t floor((t+1)/4) = 523,776` compressed entries per head, or 63.9375 MiB per batch element. Including the local band, B16 score state falls from 160 to 95.9375 MiB per batch element without a row-offset table, metadata tensor, second score state, or QK recomputation.

Packing is batch-specific. B1/B4 use compile-time dense rows because irregular arithmetic regressed their latency and their end-to-end allocation is not yet limiting. B16 uses exact packing. Production exposes only this dense/exact boolean contract. The tuning-era 16-, 64-, and 512-score alignment selector and its intermediate launch-policy table have been removed. Under the final B16 forward/backward phase schedule, exact packing measures `109.168/323.732/432.900 ms` forward/backward/complete at `6199.5 MiB`. The dense-row control measures `118.949/397.847/516.796 ms` at `7224.5 MiB`. Post-cleanup B16 is `107.898/325.751/433.649 ms` at the same allocation, a 0.17% complete-time difference within sustained variance. Earlier exact/16-aligned/dense ownership screens remain in `~/tmp/test_no_unsloth/deepseek_v4_csa_packed_phase_layout_screen.json`. Final cleanup evidence is `~/tmp/test_no_unsloth/deepseek_v4_csa_production_cleanup.json`.

The local and compressed score states still form one raw-score-to-dS handoff and deliberately support one backward per forward. Direct query-row access and compressed key-owner access use compile-time arithmetic only.

## B16 forward-efficiency experiment results

Generated TTIR retains a helper call textually inside the compressed loop, but final ISA hoists loop-invariant packed-row work into the preheader. Replacing the generic formula with an exact closed form reduced a few dozen instructions but measured `258.733/327.636/586.369 ms` versus the fresh `257.991/327.824/585.814 ms` baseline. It failed the 5% gate and was removed. A row-offset tensor would add a load without removing repeated arithmetic and was not implemented.

The two-phase forward is accepted. M4/group-16, M8/group-8, and eight-warp M16/group-4 score owners measured about `698-712 ms` complete and were rejected. Four-warp M16/group-4 changes the resource regime. D128 and M8/group-8 output owners measured about `443-450 ms`. D256 M16/group-4 with four warps and two waves/EU wins at `432.8-434.9 ms` complete.

The retained schedule lowers B16 forward from the fresh `257.991 ms` baseline to `109.168 ms` and complete time from `585.814` to `432.900 ms`, a 26.1% complete-boundary improvement with unchanged allocation. Exact B16 output/Q/local-KV/compressed-KV/sink RMSE is `0.003061/0.003114/0.004999/0.009953/0.002009`. All variants compiled within 60 seconds. Retained metadata has no global scratch. The screen record is `~/tmp/test_no_unsloth/deepseek_v4_csa_forward_reopen_screen.json`.

## Final resource review

AITER's wave32 gfx1250 FlyDSL compressor and generic Triton DSV4 sparse attention supplied portable dataflow ideas, but architecture-gated inference wrappers were not copied and no AITER training backward exists. SGLang's Triton compressor and paged sparse prefill confirm the eight-slot overlap and shared online-softmax state, but are inference/cache oriented. llama.cpp confirms previous-Ca/current-Cb graph ownership, deterministic compressed positions, and causal thresholds. DS4 confirms one online-softmax state over local and compressed regions. vLLM contributes inference integration and no reusable D512 training backward. Triton top-k is irrelevant under `topk == compressed_len`. Its generated gfx1151 code confirms that the accepted grouped D512 owners saturate LDS/VGPR resources.

The sliding-forward record was reviewed again: grouped compact-MQA, dynamic traversal, N16, FP16 raw scores, and complete-boundary selection all transfer. Normalized probability state, FP32 scores, and generic fallback branches do not. No unmeasured forward-compute geometry remains. B16 score packing and QK/output phase ownership are accepted together for time and allocation. Remaining B16 forward cost is explained by one 64 KiB QK/LSE owner with 106 spills, an 8 KiB D256 output owner with 10 spills, raw-score traffic, narrow D512 QK dots, and scalar online-softmax work.

At the end of this plan, review the current CSA forward kernels, all completed producer/attention optimizations, the accepted and rejected sliding-forward evidence, and the relevant AITER, llama.cpp, DS4, vLLM, SGLang, Transformers, and Triton resources. Any genuinely new speed or memory idea must be recorded here before implementation and evaluated at the complete producer-plus-attention forward/backward boundary. CSA forward optimization is exhausted for this contract.
