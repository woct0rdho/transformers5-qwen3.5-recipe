# Qwen4-Exp QSA Backward Triton Kernel Plan

Status: implemented, tuned and tested. Faster than AITER's Triton backward and the ROCm flash-attention package at every training batch, for both full and padded chunks. This document owns the backward contract for the dense QSA forward that `qwen4_exp_qsa_attention.py` implements, the prior art that was inspected, the accepted design, and the measurement and tuning protocol. The forward is implemented and tuned already. It emits the FP32 LSE this backward consumes.

## Production contract

| quantity | value |
| --- | --- |
| batches | 1, 4, 16 |
| sequence length | 2048, fixed |
| Q, K, V | `[B, 24, 2048, 256]` and `[B, 2, 2048, 256]` BF16 BHSD contiguous |
| forward state | output `[B, 2048, 6144]` BF16 in the `o_proj` layout, LSE `[B, 24, 2048]` FP32 natural log, per-sample `key_end [B]` int32 for right padding |
| incoming gradient | `[B, 2048, 6144]` BF16 in the same layout as the forward output |
| outputs | `dQ [B, 24, 2048, 256]`, `dK [B, 2, 2048, 256]`, `dV [B, 2, 2048, 256]` BF16 BHSD, each element owned by exactly one program |
| absent | dropout, KV cache, page tables, varlen, sinks, the gate |

The gate stays outside this contract. The forward kernel can fuse `sigmoid(gate)` as an epilogue, but the differentiable path keeps `out = attn * sigmoid(gate)` in torch, exactly as the model writes it today, because its backward is a cheap elementwise pass rather than a reason to carry a second saved activation through the attention kernels.

## What the current path costs

The step attribution in `plan_qwen4_exp.md` puts QSA attention at 1140.6 ms of backward self time for the twelve layers, which is 17.0% of the 6.71 s of own backward work and 6.5x its 174.4 ms forward. Direct measurement of the masked SDPA forward is 32.1 ms per layer at batch 1 against the 14.5 ms per layer that attribution implies, so the family's true cost has to be measured at the model level once the kernels are wired in. The forward kernel already removed the mask and the transpose copy. The backward is where most of what is left sits.

## Prior art

| source | what it establishes |
| --- | --- |
| `~/feather-attn/docs/featherattn_gfx1151_bwd_plan.md` and `csrc/featherattn_bwd_fused_d{64,128}.cu` | the accepted gfx1151 backward shape: `Delta = rowsum(dO * O)` once, then seven GEMM-equivalent phases split into a KV owner (QK score, dO/V dP, dS/Q dK, P/dO dV) and a Q owner (QK score, dO/V dP, dS/K dQ). One owner per output element, FP32 accumulation in registers, gradients stored directly as FP16, hence no atomics, no FP32 workspace, no clear or reduction kernel. Probability reconstructed in base 2 as `P = exp2(score * scale * log2(e) - LSE * log2(e))`. Measured 1.21x (D64) and 2.99x (D128) over AITER's Triton backward |
| `~/test_no_unsloth/docs/plan_deepseek_v4_sliding_attention_backward.md` and `deepseek_v4_sliding_attention.py` (`_sliding_dq_delta_kernel`, `_sliding_dv_kernel`, `_sliding_dk_kernel`, `_sliding_backward`, `_DeepseekV4SlidingAttentionFunction`) | this repository's own pattern for the same problem: a `torch.autograd.Function` whose forward calls the Triton forward, split dQ and dK/dV kernels, an FP32 score workspace where the shapes allow it, and the config and validation house style |
| `~/aiter/aiter/ops/triton/attention/{mha.py,mha_fused_bwd.py,mha_onekernel_bwd.py}` with `configs/gfx1151/triton/attention/mha/DEFAULT.json` | a training-capable Triton attention (`flash_attn_func(..., causal=True, deterministic=True)`) with gfx1151 fwd and both backward kernels already configured. The baseline for this work, and a possible fallback |
| `~/llama.cpp/docs_gfx1151/qsa_pp_tg_margins.md` | the gfx1151 attention lessons already applied to the forward: instruction and latency bound rather than bandwidth bound, and the cost of splitting the head dimension across warps or joining rows behind a barrier |
| `~/sglang/python/sglang/srt/layers/attention/qsa/` | a QSA-specific Triton implementation, but sparse and serving-only, with no backward |

## Accepted design

Three kernels plus the existing forward, each launched with the forward's BHSD/BSHD conventions and no atomics anywhere:
- `_qsa_delta_kernel`: `Delta[b, h, m] = rowsum(dO[b, m, h] * O[b, m, h])` in FP32, `[B, 24, 2048]`, one row reduction over the 256-wide head. It reads the two 25 MB tensors once, so it costs about 0.2 ms per layer at batch 1.
- `_qsa_dq_kernel`: one program per (query tile, head group, batch) using the forward's row layout, looping over key tiles up to the causal and padding bound. Per tile it recomputes `s = dot(q, k^T) * (scale * log2e)`, reconstructs `p = exp2(s - LSE * log2e)`, computes `dP = dot(dO, v^T)`, then `dS = p * (dP - Delta)` and accumulates `dQ += dot(dS, k)` in FP32. Q rows, dO rows and the `Delta` row are reused across key tiles, so they stay in registers.
- `_qsa_dkdv_kernel`: one program per (key tile, KV head, batch). It owns both `dK` and `dV` for that tile and loops over the twelve query heads of the group and their query tiles, so no cross-program reduction is needed. Per step it recomputes `s`, `p`, `dP` and `dS` from the query tile and `v`, then accumulates `dV += dot(p^T, dO)` and `dK += dot(dS^T, q)` in FP32. The causal bound makes the work per key tile decrease with the tile index, which is the one load-balance cost of the ownership scheme.

Both kernels reconstruct probability from the stored natural-log LSE with the same base-2 identity the forward uses, so `P` matches the forward's own `exp2` path up to the rounding of the reconstructing dot. Gradients accumulate in FP32 and are stored BF16, which is what the bf16 parameter path consumes.

Split accumulators, atomics, a workspace and a reduction kernel are all rejected for the first implementation: each output element has one owner, which keeps the backward deterministic, which gradient checkpointing's replay and the audit's identity gates depend on. If the KV owner's load imbalance turns out to matter, the deterministic fix is splitting the query range and reducing in a second small kernel, not atomics.

Tiles are per batch from a measured table, as in the forward: the query owner wants the forward's row layout with `head_group x block_m <= 64`, and the KV owner wants a key tile that keeps two FP32 `[block_n, 256]` accumulators (dK and dV) plus their tiles inside the register file, which puts `block_n` at 16-64.

## Implementation steps

- Write the three kernels and the `torch.autograd.Function` wrapper in `qwen4_exp_qsa_attention.py`, keeping the existing kernel-level entry point for the benchmark and adding a differentiable entry point for the tests.
- Correctness against the FP32 oracle's own autograd, in the same test file as the forward.
- Screening and finalist measurement of the backward configs per batch, in the forward's harness.
- A model-level measurement once the forward and backward are wired in.

## Correctness gates

- `dQ`, `dK` and `dV` against `torch.autograd.grad` on the blockwise FP32 oracle: relative RMSE at or below 0.005 and cosine at or above 0.9999, the range the DeepSeek backward plan used after its own selection.
- The padded case: `key_end` below the sequence length must give the same gradients as the same rows computed without padding, and padding rows must produce no gradient contribution.
- Determinism: identical bytes across repeated calls, which follows from single ownership and the absence of atomics.
- The forward's own gates stay green, and the forward's accuracy is unchanged.

## Measurement protocol

The same staged protocol as the forward and the earlier rounds: screen every candidate at one low-fidelity sample, keep the top few plus the incumbent, re-measure finalists with 15 log-space medians and MAD-based standard errors, and only displace an incumbent for a win above 2%, per `~/evotensile/docs/noisy_measurements.md`. Tune per batch at `S=2048` over `block_n` for both kernels, the query owner's `head_group x block_m`, the KV owner's query tile, and `num_warps`, in small interacting groups rather than a full Cartesian product. `num_stages=1` is the forward's measured winner and the first thing to confirm for the backward.

Baselines at the same shapes, per batch, timed fwd+bwd together and backward alone: masked SDPA (today's path), eager attention, the ROCm `flash_attn` package, and AITER's Triton `flash_attn_func` with its gfx1151 backward configs. AITER's kernel is the one that decides whether this work is worth keeping beyond the forward's gain.

## Results

Measured on gfx1151 with `~/tmp/test_no_unsloth/bench_qsa_backward.py`: warm medians, 15 samples for the kernel and 5 for the baselines at batch 16. Forward plus backward together, which is what the model pays per layer, and the backward alone for this kernel.

| batch | SDPA (today) | eager | flash_attn | AITER Triton | kernel fwd+bwd | kernel bwd |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 238.09 ms | 245.39 ms | 82.71 ms | 78.70 ms | 34.14 ms | 30.99 ms |
| 4 | 799.73 ms | 835.49 ms | 567.43 ms | 327.47 ms | 133.93 ms | 122.58 ms |
| 16 | 3203.47 ms | 3359.90 ms | 2201.36 ms | 1212.40 ms | 529.80 ms | 484.30 ms |

That is 6.8-8.0x faster than the masked SDPA path the model uses today, 2.1-4.0x faster than the ROCm flash-attention package, and 2.1-2.2x faster than AITER's Triton attention with its own gfx1151 backward configs. Gradients against the eager backward: `dQ` relative RMSE 0.0027, `dK` 0.0032, `dV` 0.0032, cosine at or above 0.99999, comfortably inside the gates this document set.

Launch table, per batch:

| batch | dQ head group x BLOCK_M | dQ BLOCK_N | dQ warps/stages | dKdV BLOCK_N x BLOCK_M | dKdV warps/stages | dKdV split |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4 x 16 | 16 | 4 / 1 | 16 x 32 | 2 / 1 | 4 |
| 4 | 4 x 16 | 16 | 4 / 1 | 16 x 32 | 2 / 1 | 4 |
| 16 | 4 x 16 | 16 | 4 / 1 | 16 x 32 | 2 / 1 | 4 |

What the sweeps established. `BLOCK_N=16` wins for both owners. 32 and 64 are 1.2-2x slower for dQ and 1.2-2.2x slower for dKdV, because the fused dKdV owner keeps two FP32 `[BLOCK_N, 256]` accumulators plus their tiles in registers. `num_warps=2` beats 4 and 8 for dKdV. One stage beats two for dQ. Tiles are launched heavy-first, and the KV owner splits its query range four ways with a deterministic reduction over FP32 partials, which was worth 9% at batch 1 and 5-8% at the larger batches.

## Optimization log

Adopted, in order of measured value:

| change | effect |
| --- | --- |
| split the KV owner's query range four ways with a reduction kernel, so the grid's makespan stops being set by the key tile that owns the whole causal prefix | 33.44 -> 30.87 ms of backward at batch 1 |
| run dQ tiles heavy-first instead of light-first | 33.44 -> 33.02 ms |
| one stage for dQ, `num_warps=2` for dKdV, `BLOCK_N=16` everywhere | included above. At batch 16, one stage for dQ measured about 2% faster than two on the backward alone |

Tried and rejected, with the measurement that rejected it:

| change | result |
| --- | --- |
| split dK and dV into two launches of one kernel, halving the accumulator pressure and allowing a larger `BLOCK_N` | 35.3 ms against 32.0 ms: the extra score and dP dots cost more than the Q re-reads they save |
| branch inside the loops to drop the causal comparison on tiles that are entirely off the diagonal | 31.58 ms against 30.83 ms |
| unroll the inner loops by two | 40.50 ms against 30.93 ms |
| `BLOCK_N` 32, 64, 128 and `BLOCK_M` 64, 128 for dKdV. `num_warps` 1, 4, 8, `num_stages` 2, 3, 4. Two stages for dQ at the larger batches | all slower, by 1.2-3x |
| splitting the head dimension across programs so a larger `BLOCK_N` fits | not viable: dP needs the full head dimension, so the split cannot be made without a second reduction of every score |
| a VGPR cap (`maxnreg`) to trade occupancy against spills | this Triton rejects the launch keyword |

The remaining headroom was then measured rather than estimated, in `plan_qwen4_exp_qsa_kv_owner_rewrite.md`. A mode decomposition of the dK/dV owner shows that 66.6% of its 23.0 ms at batch 1 is Q and dO streaming: 3.17 GB at 207 GB/s, which is this device's practical DRAM rate at these shapes. The arithmetic and the elementwise work are a few percent each, and removing either makes the kernel slower, because the schedule and not the math is what is being spent.

The compiled code agrees: per loop body there are 200 `ds_load_b128`, 467 spilled-register accesses and 128 `v_perm_b32` against 64 `v_wmma` and 8 `exp2`, and every configuration compiles to exactly the 256-VGPR ceiling and then spills 400 to 2531 slots. Traffic scales as `1/BLOCK_N`, so `BLOCK_N=64` would cut the streaming to 3.8 ms, but two FP32 `[BLOCK_N, 256]` accumulators plus their tiles cannot be held in registers and Triton's layout conversions add a second copy of every tile that crosses a boundary. Every Triton-side attempt is in the log of that document, with its measurement.

The dK/dV owner has since been rewritten in Gluon, `qwen4_exp_qsa_gluon.py` plus the dispatch in `qwen4_exp_qsa_attention.py`, which is bit-identical to the Triton kernel and 12-15% faster. The table above is with that owner in place. The Triton kernel is kept for reference, and the rewrite document carries the log, whose `AMDWMMALayout(transposed=...)` and `gl.amd.rdna3.wmma` exist in the installed `triton 3.8.0` and remove both the spill pressure and the conversion traffic.

## Padding, and why no fallback is needed

The kernels take a per-sample `key_end`, so a batch whose chunks are only partly filled is handled natively. `~/test_unsloth/data_tokenized_qwen3.5_padding_lengths.npz` records 11,503 padded chunks out of 1,961,644, 0.586%, with a maximum padding of 975 tokens and a mean of 301 over the padded ones, so a shuffled batch contains at least one with probability 0.59%, 2.33% and 8.98% at batches 1, 4 and 16. There is therefore no fallback path: the same kernels run for every batch.

Two details make that safe. The collator masks the padded rows' labels, so their incoming gradient is exactly zero and the KV owner can skip their contribution. And `lse` and `delta` are defined for every row the forward produced, padded rows included, so the KV owner loads them unmasked: masking them to zero is what made `exp2` overflow to infinity before the fix.

Skipping that contribution is not the same as leaving the position out of the output, and that distinction is the third kernel bug this document records. `dQ` and `dK`/`dV` arrive from `torch.empty_like` and the split workspace from `torch.empty`, and the owners store only the keys below `key_end`, so the padded rows of the returned gradients held whatever the allocator last recycled. On a padded batch those positions then carried a recycled value into the next kernel, and the 48 layers amplified it by three to four orders of magnitude, which is what appeared as `grad_norm` spikes of 1e3 to 1e6 every few dozen steps of a Qwen3.8 training run while the loss stayed normal, and occasionally as a non-finite step that poisoned the optimizer state. Measured on the step-25 batch of that run, a single-step replay gave `grad_norm` 3.2e3 to 1.7e4 where the same batch through the module's own attention path gives 0.39, and the gradient's extremes sat exactly in the 49 padded rows of the short chunk, with padding-to-valid ratios of 29x to 8.6e3x per layer. Batches whose chunks are all full never showed it, which is why the A/B and stress suites that used a full `key_end` were clean.

The fix leaves the dK/dV owners byte-identical, because removing their store mask moved the Gluon compiler's schedule and cost 3% for no reason, and instead zeroes the rows they skipped in the reduction kernel, which already stores every row. Its partial loads stay unmasked and the store becomes `tl.where(offs_n < key_end, summed, 0)`, which writes the zeros the gradient must have and also discards a non-finite recycled value rather than propagating it through an accumulate. On a batch with no padding the predicate is unchanged, so that path's gradient is bit-identical to before, and the cost is one vector select over dK/dV: an interleaved in-process A/B at batch 4 gives +0.13 ms on a 108.4 ms backward with no padding (+0.12%) and +0.05 ms padded, against the 0.6-0.8 ms the reduction kernel itself costs.

`test_qsa_backward_padding_rows_are_zero` covers it for both owners, and it primes the allocator with a full-length backward first: without that, a fresh process's zeroed pages hide the difference and the assertion passes on the broken kernels. With the priming it fails on them (`dK has 0.48 on padded rows`) and passes on the fix, which is what the test is for.

Two real bugs were found on the way, one of them in the test rather than the kernel, plus the padding-store bug above, which was found later. The kernel bug was the scale of `dS`: it is the gradient with respect to the raw logit, so it takes the plain softmax scale, and using the forward's internal `scale * log2e` made every gradient 44% too large. The test bug was the eager reference's visibility mask, which broadcast a four-dimensional mask incorrectly and made the padded cases look broken while the kernel was right. Both are fixed, and the padded gradient cases pass at `key_end` 1234 and 2000.

## Open questions

- Whether the KV owner's causal load imbalance needs the split-query reduction, which the first measurement will show as a gap between the sum of per-program times and the kernel's wall time.
- Whether `dQ` benefits from grouping query heads at all, since it shares no accumulator across heads. The forward's grouping helped by halving K/V loads, and the same argument should hold.
- Whether the backward should also consume the raw scores: it cannot, because a full causal band is 201 MB per layer at batch 1 against 0.2 MB for LSE, which is why the forward stores LSE only.
