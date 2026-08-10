# DeepSeek V4 Flash GGUF LoRA training plan for gfx1151

## Current state

The project trains `DeepseekV4ForCausalLM` from the frozen local GGUF checkpoint with rank-4 ordinary and routed-expert LoRA on one `gfx1151` device. The implementation is specialized for this checkpoint, architecture, sequence length, and the physical batches listed below. It is not a generic GGUF training abstraction.

| Area | State |
|---|---|
| Packed GGUF load and payload audit | Complete |
| Dense and grouped packed MMQ | Complete |
| Packed LM-head loss | Complete |
| Ordinary rank-4 LoRA | Complete |
| Full-256-group routed-expert LoRA | Complete |
| Target-specific AITER GMM/PTGMM dispatch | Complete |
| Top-k ranking, route gather, and route combine | Complete |
| Frozen-weight RMSNorm | Complete |
| mHC layer and head kernels | Complete |
| Sliding attention | Complete |
| Compressed sparse attention (CSA) | Complete |
| Heavily compressed attention (HCA) | Complete |
| Final physical-B1 model audit and profile | Complete |
| Physical-B4/B16 full update | Pending |
| Batch-16 low-memory policy | Pending |
| llama.cpp adapter conversion and cross-check | Pending |

The production stack is assembled and accepted at physical B1/S2048. The next milestone is a physical B4 update followed by the initial physical B16 attempt. No new standalone kernel tuning or low-memory policy should precede those measurements.

## Production contract

### Model

The frozen checkpoint is:

```text
~/models/ds4/DeepSeek-V4-Flash-IQ2XXS.gguf
```

The fixed model geometry is:
- 43 decoder layers.
- hidden size 4,096.
- vocabulary size 129,280.
- four mHC residual streams and 20 Sinkhorn iterations.
- 64 query heads, head dimension 512, and one shared K=V head.
- Q projection path `4096 -> 1024 -> 32768`.
- eight grouped output-A projections, each `4096 -> 1024`.
- output-B projection `8192 -> 4096`.
- 256 routed experts per layer with top-six selection.
- routed expert dimensions `4096 -> 2048 -> 4096`.
- one shared expert per layer.
- three hash-routed layers and 40 learned-router layers.
- clamped split SwiGLU with limit 10.0.
- two sliding, 21 CSA, and 20 HCA attention layers.
- sliding window 128.
- CSA compression rate 4 and 512 compressed entries.
- HCA compression rate 128 and 16 non-overlapping compressed entries.

### Target workload

- device `cuda:0`, architecture `gfx1151`, wave32, 64 KiB LDS limit.
- BF16 activations and rank-4 BF16 adapters.
- sequence length 2,048.
- physical batches 1, 4, and 16.
- no CPU offload.
- per-decoder-layer non-reentrant checkpointing.
- checkpoint-specific packed tensor types and shapes.

| Physical batch | Tokens | Selected route rows | Mean rows per expert | LM-loss rows |
|---:|---:|---:|---:|---:|
| 1 | 2,048 | 12,288 | 48 | about 2,048 |
| 4 | 8,192 | 49,152 | 192 | about 8,192 |
| 16 | 32,768 | 196,608 | 768 | about 32,768 |

Important BF16 activation sizes:

| Tensor | B1 | B4 | B16 |
|---|---:|---:|---:|
| One mHC stream tensor `[B,S,4,4096]` | 0.0625 GiB | 0.25 GiB | 1.0 GiB |
| Routed hidden/output `[B*S*6,4096]` | 0.09375 GiB | 0.375 GiB | 1.5 GiB |
| Routed gate/up activation `[B*S*6,2048]` | 0.046875 GiB | 0.1875 GiB | 0.75 GiB |
| Attention Q output `[B,S,32768]` | 0.125 GiB | 0.5 GiB | 2.0 GiB |
| Full logical LM logits | about 0.49 GiB | about 1.97 GiB | about 7.9 GiB |

### Adapter surface

The accepted adapter surface contains:
- 383 ordinary LoRA wrappers.
- 43 complete routed-expert wrappers.
- 426 wrapped modules.
- 938 trainable adapter tensors.
- 645,609,472 BF16 parameters, approximately 1.20 GiB before gradients and optimizer state.

Ordinary targets include Q-A, Q-B, KV, output-B, compressor KV/gate, and shared-expert gate/up/down projections. Routed wrappers preserve combined gate/up factors, DeepSeek clamping, expert-major planes, and packed base projections.

The following remain frozen and adapter-free:
- grouped `o_a_proj`.
- indexer projections.
- learned and hash router weights.
- norm scales.
- mHC controls.
- attention sinks and position biases.
- token embeddings.
- packed LM-head weights.

Grouped `o_a_proj` is a three-dimensional eight-group boundary, not an ordinary two-dimensional linear layer. It remains explicitly unsupported for LoRA until the training representation, converter, and llama.cpp loader all support the same eight independent factor pairs.

### Required semantics

- Packed GGUF payloads are the frozen source of truth. Production must not materialize logical dequantized base weights.
- Learned and hash routing, selected-expert order, score normalization, and clamped SwiGLU semantics are unchanged.
- Production Q/KV storage is BHSD and attention output is contiguous BSHD.
- FP32 sinks, LSE, Delta, RMS reductions, softmax reductions, and deterministic compact-KV reductions are preserved.
- RMSNorm-to-RoPE boundaries remain BF16. Input RoPE is partial and output rotation is conjugate.
- Sliding, CSA, and HCA own separate dispatch, launch tables, workspaces, score layouts, and custom backward paths.
- Attention raw FP16 scores are overwritten in place with `dS`. One backward is supported per forward or checkpoint replay. Repeated backward through one retained graph is unsupported.
- Shared K=V gradients are combined directly without repeated KV heads.
- Unsupported architecture, batch, sequence, layout, dtype, mask, cache, or grouped-GEMM shape fails closed.
- `TrainingArguments(use_liger_kernel=False)` remains required. DeepSeek-specific RMSNorm, mHC, and loss integrations are project-local.

## Latest accepted results

### Full-model B1 update

The authoritative model-level boundary is physical B1/S2048 with 43 non-reentrant decoder checkpoints.

| Update | Forward | Backward | Gradient clip | Optimizer | Total | Throughput |
|---|---:|---:|---:|---:|---:|---:|
| Warm untraced | 5.892 s | 12.681 s | 38.8 ms | 110.5 ms | 18.723 s | 109.4 tokens/s |
| Kineto traced | 5.962 s | 12.744 s | 33.3 ms | 109.7 ms | 18.849 s | 108.7 tokens/s |

The accepted audit confirms:
- first loss `3.0208380222320557` and post-update second loss `3.009669780731201`.
- 2 sliding, 21 CSA, and 20 HCA modules use project-owned dispatch.
- all 469 LoRA-B tensors changed on the first update.
- all 938 adapter tensors had finite, nonzero gradients on the second backward.
- all 474 packed parameters retained their payload pointers, versions, storage, and checksums.
- all 43 grouped output-A modules remained frozen and gradient-free.
- all 86 mHC connections, 43 decoder boundaries, and the final head were patched before PEFT.
- process swap remained zero.

### Whole-update attribution

The traced update contains 18.565 s of GPU-kernel time. Nested module annotations overlap, so exact kernel-name aggregation is authoritative for percentages.

| Kernel family | GPU time | Share |
|---|---:|---:|
| Packed frozen MMQ | 11.059 s | 59.6% |
| PyTorch elementwise kernels | 2.835 s | 15.3% |
| Sliding, CSA, and HCA attention | 1.533 s | 8.3% |
| hipBLASLt/rocBLAS GEMM | 0.837 s | 4.5% |
| Explicit copy and concatenation | 0.512 s | 2.8% |
| mHC Triton kernels | 0.512 s | 2.8% |
| AITER routed-LoRA GMM/PTGMM | 0.504 s | 2.7% |
| RMSNorm kernels | 0.278 s | 1.5% |

Packed MMQ subcomponents are 4.029 s input backward, 2.925 s IQ2_XXS gate/up forward and replay, 1.702 s Q2_K down forward and replay, 1.297 s dense forward and replay, 0.868 s grouped output-A forward and replay, and 0.239 s activation quantization.

The largest generic elementwise kernel is BF16 direct copy/conversion at 1.054 s across 1,936 launches. Vector BF16 add contributes 0.497 s. These launches require ownership attribution before any fusion work.

### Memory

The ROCm-visible device capacity is 125 GiB.

| Boundary | Allocated | Reserved or free |
|---|---:|---:|
| Packed model loaded | 80.834 GiB | 39.816 GiB free |
| Adapters injected | 81.973 GiB | 82.100 GiB reserved |
| Complete-update peak | 87.615 GiB | 87.955 GiB reserved |
| After traced update | 84.559 GiB | 32.229 GiB free |

The high-water mark occurs during backward. Peak allocation is 6.781 GiB above the loaded model and 5.641 GiB above the adapter-injected state. Final profiler-process RSS is 4.288 GiB and process swap is zero.

### Attention components

All timings include each family's producer and attention boundary. Complete time is the authoritative selection metric.

#### Sliding attention

| Phase | B1 | B4 | B16 |
|---|---:|---:|---:|
| Forward | 5.542 ms | 21.793 ms | 93.799 ms |
| Backward | 9.149 ms | 35.903 ms | 145.404 ms |
| Production complete | 14.778 ms | 58.543 ms | 240.564 ms |

The retained score state is 0.03125 GiB per batch element. Exact output/Q/shared-KV/sink checks, production layout checks, checkpoint replay, compiler metadata, and fail-closed dispatch pass.

#### Compressed sparse attention

| Phase | B1 | B4 | B16 |
|---|---:|---:|---:|
| Forward | 15.464 ms | 60.201 ms | 107.898 ms |
| Backward | 24.729 ms | 97.693 ms | 325.751 ms |
| Production complete | 40.193 ms | 157.894 ms | 433.649 ms |
| Incremental peak allocation | 0.441 GiB | 1.764 GiB | 6.054 GiB |

B1/B4 use dense compressed-score rows. B16 uses exact triangular packing, separate QK/LSE and D256 output owners, and separate `dS`/Delta/sink and D256 `dQ` owners. Compressed KV gradients use deterministic group-2 partials with no atomics.

#### Heavily compressed attention

| Phase | B1 | B4 | B16 |
|---|---:|---:|---:|
| Forward | 6.459 ms | 26.621 ms | 59.588 ms |
| Backward | 11.767 ms | 44.958 ms | 130.526 ms |
| Complete | 18.226 ms | 71.579 ms | 190.114 ms |
| Incremental peak allocation | 0.288 GiB | 1.151 GiB | 4.604 GiB |

HCA uses 16 non-overlapping C128 compressed entries and visibility `c < floor((t+1)/128)`. B16 separates QK/LSE from D256 output and separates `dS`/Delta/sink from D256 `dQ`. Exact output/Q/local-KV/compressed-KV/sink/producer checks and bitwise checkpoint replay pass.

### Routed LoRA and grouped GEMM

DeepSeek checkpoint-charged routed-LoRA complete medians are:

| B1 | B4 | B16 |
|---:|---:|---:|
| 12.067 ms | 37.342 ms | 140.504 ms |

The retained implementation always passes all 256 expert factor planes and uses a contiguous 256-entry `int32` group-size vector with zero sizes for inactive experts. Packed frozen MMQ remains compact-active. There are no factor `index_select` allocations, factor-scatter autograd paths, implicit contiguous repairs, or square-matrix workarounds.

`moe_gmm_configs.py` owns exactly 66 GMM keys `(M,K,N,RHS-layout)` and 33 PTGMM keys `(M,K,N)` for the required DeepSeek/Qwen B1/B4/B16 shapes. Layout is part of GMM identity. Unsupported shapes fail closed because the measured table does not support a reliable general heuristic. The coefficient-only priors, bounded per-key search, and confirmation evidence are recorded in [aiter_gmm_ptgmm_coefficient_prior_tuning.md](aiter_gmm_ptgmm_coefficient_prior_tuning.md).

Outputs, hidden gradients, active factor gradients, inactive zero gradients, sparse routes, and non-reentrant checkpoint replay are bitwise equal to the compact correctness control.

### RMSNorm and mHC

Weighted RMSNorm patches 235 norms at widths 4,096, 1,024, 512, and 128. Q-B uses 43 scale-free norms. Frozen scales receive no gradients, and the project-local weighted backward computes only `dX`.

Accepted `[2048,width]` forward-plus-backward medians are 0.359 ms at width 4,096, 0.109 ms at width 1,024, and 0.110 ms at width 512. Width-128 accepted medians are 0.132 ms for `[512,128]`, 0.128 ms for `[2048,128]`, and 0.088 ms for `[8192,128]`.

mHC complete component results are:

| Rows | Forward | Backward | Complete | Incremental peak allocation |
|---:|---:|---:|---:|---:|
| 2,048 | 1.524 ms | 3.105 ms | 4.619 ms | 0.095 GiB |
| 8,192 | 5.595 ms | 10.975 ms | 16.574 ms | 0.380 GiB |
| 32,768 | 21.839 ms | 42.541 ms | 64.386 ms | 1.521 GiB |

mHC preserves FP32 projection/control accumulation, F16 controls, BF16 activation boundaries, compact Sinkhorn state, activation-only backward, and the original 20-iteration semantics.

### Routing

Accepted routing medians are:

| Operation | B1 | B4 | B16 |
|---|---:|---:|---:|
| DeepSeek top-six selection, forward plus backward | 0.121 ms | 0.227 ms | 0.551 ms |
| DeepSeek route gather, forward plus backward | 1.195 ms | 5.740 ms | 22.255 ms |
| DeepSeek weighted combine, forward plus backward | 1.978 ms | 7.478 ms | 30.454 ms |

The path preserves learned-router sqrt-softplus scoring, correction bias, hash routing, tie-aware thresholds, selected order, FP32 weighted accumulation, direct inverse permutation, and custom gather/combine backward.

## Completed optimizations

### Packed model execution

- Frozen Q8_0 dense projections use packed MMQ forward and exact packed input-gradient kernels.
- Grouped output-A uses fixed eight-group Q8_0 MMQ and remains adapter-free.
- Routed gate/up uses paired IQ2_XXS grouped MMQ.
- Routed down uses Q2_K grouped MMQ.
- Packed execution does not create logical expert matrices, packed transposes, packed gradients, or CPU route metadata.
- Capability and shape guards fail closed instead of silently dequantizing or copying operands.

Rejected or deferred:
- Production logical dequantization was rejected because it breaks packed-weight ownership and creates persistent logical matrices.
- Packed transposes, selected-expert weight copies, and packed gradients were rejected because they add recurrent allocation/copy work without changing the frozen base contract.
- Quantizing backward cotangents was rejected because exact packed input gradients already own the boundary and the additional approximation is unnecessary.
- Standalone MMQ and grouped-MMQ retuning is closed. Reopen only for a complete fusion that removes adjacent quantization, activation, routing, residual, or LoRA traffic.

### Packed LM-head loss

`apply_deepseek_v4_liger_loss()` processes 512 hidden rows at a time, applies packed Q8_0 MMQ, computes fused cross-entropy, and returns the packed hidden-state input gradient. It does not materialize full logits or the approximately 0.99 GiB logical BF16 LM head. Materialized helpers are bounded correctness oracles only.

Rejected or deferred:
- Full-logit loss was rejected because it materializes about 0.49 GiB at B1, 1.97 GiB at B4, and 7.9 GiB at B16 before loss backward.
- Production LM-head dequantization was rejected because the logical BF16 head is about 0.99 GiB and duplicates the packed frozen source.
- Materialized loss and dequantization remain test-only oracles and must not become fallback training paths.

### Attention ownership

Sliding, CSA, and HCA have separate production implementations and launch tables.

Common accepted properties:
- compact shared-KV traversal with no repeated 64-head K/V tensor.
- no dense attention masks or probability tensors.
- FP32 online-softmax state and sinks.
- FP16 raw score storage overwritten in place with `dS`.
- direct shared K=V gradient accumulation.
- deterministic sink and compressed-KV reductions.
- no atomics, global scratch, or full query-sized partial-gradient workspace.
- at most 64 KiB LDS per retained launch.
- metadata-only BHSD/BSHD layout handling.
- custom backward and non-reentrant checkpoint replay.

CSA additionally owns rate-4 compression, compressed-position RoPE, fixed-geometry indexer elimination, causal-prefix visibility, batch-specific score storage, and deterministic compressed-gradient partials.

HCA owns rate-128 featurewise compression, 16-entry C128 visibility, producer backward, separate local/compressed score state, and compact deterministic partials. It has no indexer, top-k, overlap, or sentinel path.

Rejected or deferred:
- Stock AITER D512 backward was rejected because it requests 128 KiB LDS, above the retained 64 KiB limit. AITER DSV4 sparse attention was rejected as a training backend because it lacks the required backward and reusable LSE contract.
- A shared eager, FlexAttention, or generic streaming backend was rejected because it either retained dense/repeated state, failed ROCm lowering for required variants, or lost the complete family boundary. The three attention semantics remain separately owned.
- Sliding score recomputation, score-disabled backward, and broader fused-gradient variants lost to the retained raw-score-to-`dS` boundary. AITER sliding complete time was about 6.5x slower at the target batches.
- CSA generic top-k/indexer work was removed after the fixed S2048 geometry proved `compressed_len == index_topk == 512`. Dense B16 score storage was rejected in favor of exact triangular packing and phase ownership.
- HCA local/compressed `dQ` splitting measured 327.8 ms complete versus 314.1-314.7 ms for the then-combined owner. It duplicated control work while compressed pairs represented only 5.7% of visible pairs, so the split was removed.
- Static C128 HCA producer expansion was rejected because compile time and instruction size exceeded the bounded production contract. Dynamic producer loops remain retained.
- Reopen an attention family only when a larger fusion beats its complete producer-plus-attention boundary and preserves its exact gradients, state ownership, and checkpoint replay.

### Routing

- `fast_moe_ranking.py` owns Qwen top-eight and DeepSeek top-six/hash ranking.
- `fast_moe_routing.py` owns sorting, direct inverse permutation, gather, weighted combine, and custom backward.
- Production uses static expert/top-k/hidden-width geometry and device-resident metadata.
- Tie-aware correctness compares selected thresholds and sorted selected weights rather than unstable near-tie expert identities.

Rejected or deferred:
- Exact `torch.topk` expert-ID matching was rejected as a correctness rule because tied or near-tied experts may exchange identity while preserving selected scores and model behavior.
- Approximate expert selection was rejected. Fixed top-six/hash semantics and selected-score normalization remain exact.
- Eager gather, inverse permutation, and weighted combine were rejected as production ownership because they materialize intermediates and fragment forward/backward into many generic kernels.
- Routing is closed as a standalone target. The final trace leaves only 4.3 ms in sort/scan work. Reopen only as part of a measured surrounding fusion.

### RMSNorm and mHC

- Frozen weighted and Q-B RMSNorm use project-local Liger-compatible kernels.
- Weighted backward writes `dX` over the incoming cotangent and omits frozen-scale scratch and reduction.
- Width 128 has a dedicated launch geometry inside the existing kernel.
- mHC prepare, merge, final-head, and activation-only backward are specialized for HC=4 and rows 2,048/8,192/32,768.
- mHC aliases the residual stream where valid, reuses dead cotangent storage, keeps F16 controls contiguous, and avoids full FP32 hidden-stream copies.
- All 86 connections, 43 layer boundaries, and the final head are configured before PEFT and checked after wrapping.

Rejected or deferred:
- Generic model-wide Liger patching was rejected because the installed registry has no DeepSeek V4 contract and cannot preserve project-owned clamp, partial-RoPE, packed-MoE, mHC, and packed-loss behavior.
- Stock weighted RMSNorm backward was rejected for frozen scales because it allocates and reduces unnecessary `dW` scratch. The retained path computes only `dX`.
- A grouped-token tensor-core mHC candidate measured about 9.47 ms complete at B1, versus 4.619 ms for the retained token-program boundary.
- Fully static HC=4 merge expansion increased code size, while dynamic Sinkhorn-loop variants were unstable in B1 backward. Cache-hint variants regressed target rows. These branches were removed.
- Lower-precision controls, Sinkhorn state, and control accumulation were rejected because they change the fixed mHC numerical contract.
- Collapse-plus-RMSNorm, cross-layer mHC fusion, and producer epilogues remain deferred until B4/B16 attribution identifies a complete-boundary opportunity.

### LoRA

- Ordinary LoRA uses the existing hipBLASLt BF16 path and fused LoRA-B plus residual addition where applicable.
- Routed LoRA uses one unconditional full-256-group implementation.
- Original expert factor planes are passed through supported transpose metadata views.
- AITER `gmm` computes forward and input gradients. `ptgmm` writes complete expert-major factor gradients directly.
- Exact target dispatch includes row count and RHS layout and rejects unmeasured shapes.
- PEFT names, rank/alpha scaling, combined gate/up-A semantics, and adapter-only serialization are preserved.

Rejected or deferred:
- Per-route factor `index_select` was rejected because it allocates gathered factor planes and introduces `IndexSelectBackward` zero/scatter work during both original forward and checkpoint replay.
- A compact/full active-count selector recovered only small component differences. Compact sparse B1 was 2.6% faster for DeepSeek and 2.3% faster for Qwen, but the dual ownership was not justified. One unconditional full-group path is retained.
- Implicit `.contiguous()` repair and the square-matrix input-gradient workaround were rejected because they hide unsupported layouts and can introduce copies or incorrect transpose ownership.
- A general GMM/PTGMM heuristic was rejected because the accepted targets use 54 distinct GMM configs across 66 keys and 30 PTGMM configs across 33 keys, with non-monotonic batch and layout changes. Unmeasured shapes fail closed.
- Large Cartesian tuning was rejected in favor of bounded one-parameter coordinate slices followed by coefficient-profile reversed-order confirmation. Ordinary LoRA GEMM retuning remains closed because hipBLASLt already owns that BF16 boundary.

### Integration and cleanup

- Base-model attention, routing, MMQ, RMSNorm, and mHC patches are installed before PEFT.
- Packed LM-head loss is mandatory.
- Training keeps `use_liger_kernel=False` because generic Liger has no DeepSeek V4 registry contract.
- Production contains one retained path per accepted boundary. Experimental selectors and losing branches are absent.
- The repository suite passes `106/106`. Ruff, `py_compile`, and `git diff --check` pass.

Rejected or deferred:
- Runtime experimental switches were removed after selection so checkpoint replay and user-facing training cannot diverge from the accepted path.
- Generic fallbacks that repair unsupported layouts, dequantize packed weights, or silently change model semantics were rejected in favor of explicit failure.
- Repeated backward through one retained attention graph remains unsupported because forward score storage is intentionally transferred to `dS` ownership. A new forward or checkpoint replay is required.
- Validation on additional ROCm architectures is deferred. Current launch and compiler evidence is specific to `gfx1151`.

## Validation requirements

Every full-model acceptance run must verify:
- finite loss and finite trainable gradients.
- every expected adapter gradient present.
- no gradients on packed parameters, norms, routers, mHC controls, embeddings, sinks, or LM head.
- packed payload pointer, version, storage, and checksum immutability.
- exact hash-router indices.
- sorted selected routing-weight agreement for learned routers.
- all 43 decoder layers using non-reentrant checkpointing.
- all intended attention modules using project-owned dispatch.
- no hidden CPU offload or process swap growth.
- first-step LoRA-B updates and second-backward nonzero A/B gradients.

Continuous numerical gates are:
- loss relative error at most `5e-3`.
- final-hidden cosine at least `0.999`.
- final-hidden relative L2 error at most `3e-2`.
- concatenated adapter-gradient cosine at least `0.999`.
- concatenated adapter-gradient relative L2 error at most `3e-2`.
- major adapter-family cosine at least `0.99`.
- first-update cosine at least `0.995`.
- updated-adapter relative L2 error at most `1e-2`.
- no NaN, Inf, missing, or unexpectedly zero gradient.

For near-tied learned-router selections, compare sorted selected weights without expert IDs. Require cosine at least `0.999`, relative RMSE at most `3e-2`, bounded maximum weight error, and bounded normalized row-sum error. A near-tie exception never permits incomplete or nonfinite gradients.

Attention component gates additionally require exact-shape output and Q/local-KV/compressed-KV/sink/producer-gradient checks, threshold rows, checkpoint replay, compiler metadata, zero global scratch, bounded LDS, and fail-closed layout/mask/cache behavior.

## Remaining work

### Batch-16 memory policy, only if required

If B16 does not fit or leaves insufficient operating headroom, test contiguous non-reentrant checkpoint segments of 1, 2, 4, and 8 decoder layers. Measure recomputation time, boundary activation size, peak allocation/reservation, and gradient agreement.

If segmentation is insufficient, attribute the peak before changing code. Candidate policies are:
- stricter lifetime control for attention, route, and mHC workspaces.
- serialized reuse of temporary workspaces.
- family-specific reduction of retained LSE, metadata, producer state, or backward scratch.
- checkpoint segment placement around expensive CSA/HCA boundaries.

Do not fork mathematical kernels, lower precision, reintroduce dense masks/probabilities, or add CPU offload solely for B16.

B16 is accepted only after one complete update with finite gradients, immutable packed payloads, and zero process swap.

### llama.cpp adapter interoperability

Implement ordinary and routed-expert PEFT-name conversion to llama.cpp GGUF LoRA. Validate:
- ordinary target aliases.
- combined expert gate/up factor splitting.
- shared gate/up-A semantics.
- expert-major three-dimensional layouts.
- rank/alpha scaling exactly once.
- missing expert, wrong expert count, and incompatible shape rejection.
- ordinary-only, expert-only, and combined fixed-prompt outputs.
- adapter-disabled and scale-zero equivalence.
- adapter switching with KV/compressor state reset.
- adapter-only save and reload.

Grouped `o_a_proj` must remain explicitly unsupported rather than silently omitted. Adapter merge remains rejected because the frozen source is packed.

### Optional attribution-driven fusion

Consider additional fusion only after B4/B16 establishes the next limiting boundary.

Current candidates are:
- BF16 direct copy/conversion and vector-add traffic attributed to its owning packed-input-gradient, residual, LoRA, mHC, or producer boundary.
- packed-MMQ fusion that removes adjacent quantization, activation, routing, residual, or LoRA traffic.
- workspace-lifetime fusion required by a measured B16 peak.

Do not reopen standalone MMQ, grouped MMQ, ordinary LoRA GEMM, routing, RMSNorm, mHC, sliding, CSA, or HCA tuning without complete-boundary evidence. Additional ROCm architecture validation is outside the current scope.

## Benchmark policy

Component benchmarks must:
- use deterministic BF16 inputs and real packed payloads where applicable.
- use actual B1/B4/B16 route distributions and exact shapes.
- run correctness before timing.
- use at least three warmups and 20 timed samples for small kernels.
- report median forward, backward, complete time, and incremental peak GiB.
- validate checkpoint replay and unsupported-shape rejection.
- keep each tuning experiment under five minutes unless separately justified.

Full-model benchmarks must:
- warm one complete update.
- measure at least three untraced updates for acceptance timing.
- trace one separate update for attribution.
- report phase timing, throughput, allocated/reserved GiB, device-free GiB, RSS GiB, and swap.
- verify gradients, updates, dispatch coverage, and packed immutability on a measured update.

Do not accept isolated throughput that regresses complete-update time, memory, correctness, layout ownership, or model semantics.

## Code ownership

Training and full-model audit:
- `train_deepseek_v4.py`.
- `audit_deepseek_v4_training_step.py`.
- `deepseek_v4_profiler.py`.

Attention:
- `deepseek_v4_attention.py`.
- `deepseek_v4_sliding_attention.py`.
- `deepseek_v4_csa.py`.
- `deepseek_v4_hca.py`.
- `benchmark_deepseek_v4_sliding_attention.py`.
- `benchmark_deepseek_v4_csa.py`.
- `benchmark_deepseek_v4_hca.py`.

Adapters and grouped GEMM:
- `deepseek_v4_lora.py`.
- `deepseek_v4_moe_lora.py`.
- `fast_moe_lora.py`.
- `moe_gmm_configs.py`.

Packed loss, routing, norm, and mHC:
- `deepseek_v4_liger_loss.py`.
- `deepseek_v4_liger_rmsnorm.py`.
- `deepseek_v4_liger_mhc.py`.
- `deepseek_v4_routing.py`.
- `fast_moe_ranking.py`.
- `fast_moe_routing.py`.

Focused tests are co-located as `test_*.py`. Attention implementation detail and retained launch tables live in the six focused attention plans linked above.
