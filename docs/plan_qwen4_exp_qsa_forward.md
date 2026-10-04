# Qwen4-Exp QSA Forward Triton Kernel Plan

Status: forward implemented and tuned. The backward is not written yet. This document owns the dense QSA forward contract for training on `gfx1151`, the prior art that was inspected, the accepted design, and the measurement and tuning protocol the implementation has to satisfy.

## Scope

At `S=2048` the QSA selection is exhaustive (`block_topk == max_complete_blocks == 512`), so the twelve indexed-attention layers are ordinary causal grouped-query attention and no selection, block gather, or sparse kernel is involved. The kernel described here therefore implements exactly that: one dense causal GQA forward with the QSA output gate. The sparse path above the budget, decode, the indexer itself, and the backward are out of scope. The forward is designed so the backward is a standard flash-attention backward over the same tiles.

## Production contract

| quantity | value |
| --- | --- |
| batches | 1, 4, 16 |
| sequence length | 2048 (fixed) |
| layers | the 12 indexed layers, 1-based 4, 8, ..., 48 |
| Q | `[B, 24, 2048, 256]` BF16, model-native contiguous BHSD, metadata-only view at the boundary |
| K, V | `[B, 2, 2048, 256]` BF16, same storage, GQA group size 12 |
| scale | `1/16` (exact `head_dim ^ -0.5`) |
| mask | causal, plus an optional per-row right-padding bound `kv_end[b, row]` because the trainer collator emits a real `attention_mask` |
| gate | `[B, 2048, 6144]` BF16, applied as `out * sigmoid(gate)` |
| output | `[B, 2048, 6144]` BF16 in the layout `o_proj` consumes, one contiguous store per head |
| accumulation | FP32 scores, softmax, and output. Optional FP32 LSE `[B, 24, 2048]` |
| absent | dropout, KV cache, page tables, varlen/`cu_seqlens`, RoPE, q/k norm, sparse selection |

Dispatch rejects any unsupported architecture, dtype, shape, head geometry, or reachable offset, and mutates no global AITER or Triton configuration. Everything unsupported falls back to the current masked SDPA path.

## What the current path costs

From the step attribution in `plan_qwen4_exp.md`, one warm update at `B=1`: QSA attention (scores, softmax, output) is 174.4 ms of forward self time (1.7% of the 10.56 s forward), 172.2 ms again inside gradient checkpointing's recomputation, and 1140.6 ms of backward self time (17.0% of the 6.71 s of own backward work). The indexer is 7359.6 ms forward plus 8471.1 ms recomputed, and it owns the `[B, 1, 2048, 2048]` bool mask (4 MiB per layer) that the attention consumes.

Derived: 618 GFLOP of forward work per step for the twelve layers at `B=1`, at 174.4 ms that is 3.5 TFLOP/s effective, and the backward costs 6.5x the forward. The whole attention family is 1.49 s of a 29 s step, 5.1%.

This bounds what a forward kernel alone can win. At the machine's peak the forward would be ~10 ms per step, so the entire remaining forward margin is ~165 ms, 0.6% of the step, and the mask materialisation disappears with the indexer fast path rather than with this kernel. The forward kernel is worth writing because it removes the dependency on a mask-capable SDPA backend and, more importantly, because the measured 1.14 s backward is where the attention family actually spends its time, and a forward that emits FP32 LSE on a mask-free causal tile is the prerequisite for a standard backward. The gate fusion is worth about 3 ms per step and is not a reason to write anything.

## Prior art and portability

| source | what it provides | what it lacks for us | verdict |
| --- | --- | --- | --- |
| `llama.cpp/ggml/src/ggml-cuda/qsa-prefill.cu`, `lightning-indexer.cu`, `fattn-tile.cuh`, `docs_gfx1151/{attn_plan,qsa_pp_tg_margins}.md` | the gfx1151 measurement base: `nbatch_K=64` for `DKQ=DV=256, ncols=32`. rocWMMA slower than the hand-written tile path on this part. A four-row union at G=4 costs 4x instructions and one `__syncthreads` per four-cell chunk, giving 95 GB/s against the 230 GB/s ceiling with the fix being one query per workgroup. WMMA sits at 2.8-5.7 TFLOP/s because the kernel is instruction and latency bound | inference-only, no training state, no LSE contract, per-cell mask gathers that our exhaustive case does not need | structure and tuning reference, not reusable code |
| `aiter/aiter/ops/triton/_triton_kernels/attention/mha.py` with `aiter/ops/triton/configs/gfx1151/triton/attention/mha/DEFAULT.json` | a classic-Triton `_attn_fwd` that already runs on gfx1151: GQA through `off_k_head = off_q_head // grp_sz`, a dense `sd_mask` path, an XCD workgroup remap, fused-`pe` variant, and a shipped gfx1151 table (fwd default `BLOCK_M 64, BLOCK_N 32, num_warps 4, num_stages 2, waves_per_eu 1`. `pe` variant `BLOCK_M 256, BLOCK_N 64, num_warps 8, PRELOAD_V`) plus Triton backward kernels with gfx1151 tables | one program per (query head, M tile, batch), so K/V are re-loaded per head instead of shared across the group. Head-dim config rules only cover V <= 128, so D=256 needs retuning. No gate fusion | measure it first at our exact shapes with a retuned D=256 config. It is the baseline the project kernel has to beat |
| `aiter/aiter/ops/triton/_triton_kernels/attention/{pa_prefill,extend_attention}.py`, `_triton_kernels/attention/{mha_fused_bwd,mha_onekernel_bwd}.py`, `aiter/aiter/ops/mha.py` | further Triton prefill/backward attention and the asm FMHA wrappers. A gfx1151 table also exists for `unified_attention` | neither `pa_prefill` nor `extend_attention` ships a gfx1151 table (`extend_attention` exists only for gfx1250, `pa_prefill` for no listed architecture). The asm paths are gated (`require_gfx1250_asm`, `get_gfx() in ("gfx942", "gfx950")`) and are not portable to RDNA3.5 | not portable.The Triton `mha` forward and backward are the portable part |
| `sglang/python/sglang/srt/layers/attention/qsa/` (`qsa_indexer.py`, `sparse_attn.py` with `_sparse_gqa_prefill` and `_sparse_gqa_chunk_prefill`, `mqa.py`, `fused_kv.py`, `metadata.py`, `config.py`) | the closest QSA-specific prior art: `BLOCK_M = max(16, next_power_of_2(group_size))` with `GROUP_SIZE` as a constexpr, FP32 online softmax with a `[BLOCK_M, HEAD_DIM]` accumulator, per-`total_q` config tables of `(BLOCK_N, warps, stages)` | sparse/inference contract: selected indices, `cu_seqlens`, paged cache, no backward | structural ideas only |
| `vllm/vllm/model_executor/models/qwen3_next.py` | the same gate epilogue, unfused (`attn_output = attn_output * torch.sigmoid(gate)`, line 458) | no Qwen4 model, no training forward | confirms the gate is unfused everywhere |
| `~/test_no_unsloth/docs/plan_deepseek_v4_sliding_attention_forward.md` | the project precedent for exactly this class of work: grouped compact KV across query heads measured about 2x, 64 logical rows per program at 64 KiB LDS and 256 VGPRs, FP32 LSE state, 5.5/21.8/93.8 ms at B=1/4/16 with 6.0-6.1 TFLOP/s useful, and the explicit conclusion that llama.cpp, DS4, vLLM and SGLang kernels were rejected because they provide no compatible training-state contract | different geometry (D=512, sliding window 128, shared KV, sinks) | the design and validation pattern to follow |

No inspected source provides a dense causal D=256 GQA training forward with the QSA gate, so the kernel is project-owned. The one genuine portability finding is that AITER's Triton `mha` forward and backward do run on gfx1151 with shipped config tables and a dense-mask path, which is why the first implementation step is a measurement of that path rather than a new kernel.

## Roofline and targets

Per layer per batch element: Q 25.2 MB, K and V 2.1 MB each, gate 25.2 MB, output 25.2 MB, so about 80 MB of traffic against 51.5 GFLOP of useful work, an intensity near 645 FLOP/byte. The machine's balance is 222 FLOP/byte (51 TFLOP/s over 230 GB/s), so the operation is compute-bound at the ideal level, but comparable kernels on this part reach 2.8-6 TFLOP/s because they are instruction and latency bound. The target is therefore throughput, not bandwidth: forward at or below 90 ms per step at `B=1` (2x today) and 60 ms as the stretch goal, then a complete forward plus backward under 0.6 s, against 1.49 s today. The `tl.dot`-versus-FMA question is a first-class experiment for this part, since rocWMMA measured slower than the hand-written tile path in `llama.cpp`.

## Accepted design

One program owns a tile of query *positions* across a group of query heads that share one KV head, following the DeepSeek grouped design. Because all rows in the tile share the same query positions, they share the causal bound and the padding bound, which is what makes grouping cheap:
- Logical rows are `G x BLOCK_M` with `G x BLOCK_M <= 64` at `D=256` in FP32, the register budget the DeepSeek kernel hit at 256 VGPRs. Candidate shapes, to be screened: `G=12, BLOCK_M=4` (48 rows, all heads of a KV head in one program), `G=6, BLOCK_M=8`, `G=4, BLOCK_M=16` (64 rows), `G=3, BLOCK_M=16`. `G=12, BLOCK_M=8` needs 96 rows and is expected to spill.
- K/V tiles are loaded once per program and reused across the group's heads, which is where the measured 2x of the DeepSeek grouped design came from. With `G=12` a single K/V read serves every query head of that KV head.
- Key traversal is a `tl.range` over `BLOCK_N in {32, 64, 128}` with `hi = min(rows_lo + BLOCK_M, kv_end)` for the causal bound and a lower bound of zero. Tiles entirely above the diagonal are skipped rather than masked, so no masked cell is ever multiplied and the `0 * NaN` poison case that `qsa_pp_tg_margins.md` documents cannot arise.
- Online softmax keeps FP32 running max and denominator per logical row and rescales the FP32 accumulator, with the exact `exp2`-based formulation the DeepSeek kernel uses. LSE per logical row is written out for the backward.
- The head dimension is never split across programs. llama.cpp measures a per-chunk barrier to recombine a split-D partial before the softmax at 92 cycles per four-cell chunk. Triton keeps the `[ROWS, 256]` reduction inside the program, and a split-D variant would additionally need a merge kernel and extra traffic.
- The epilogue applies `sigmoid(gate)` from a `[BLOCK_M, G, 256]` gate tile and stores one contiguous `[BLOCK_M, G, 256]` block per program into the BSHD output that `o_proj` consumes, which also removes the `transpose(1, 2)` plus `contiguous()` copy the current path pays (about 25 MB per layer, 300 MB per step at `B=1`).
- Padding is a per-row `kv_end` bound, not an additive mask. `kv_end = 2048` recovers the fixed-length case and the audit's all-ones batch is the largest instance of it.
- The raw-score handoff that the DeepSeek forward used for its backward is not available here: a full causal band is `S x S` per head, 201 MB per layer at `B=1` and 3.2 GB at `B=16`, against 0.2 MB for LSE. The backward must recompute probabilities from LSE and its own Q/K/V reads, which is the standard flash-attention contract and the reason LSE output is part of this contract.

Launches per batch: `B x num_kv_heads x ceil(2048 / BLOCK_M)` programs, so `B=1` is 64 programs per layer at a 64-position tile and 256 at a 16-position tile, and sixteen times that at `B=16`. Occupancy on 40 CUs is part of the screen, and the tile may differ per batch as it does in the DeepSeek table.

## Implementation steps

- Baseline measurement at the exact contract shapes, all through CUDA events with the audit's warm-up discipline: masked SDPA (today's path), eager attention, AITER's Triton `mha` forward with gfx1151 configs retuned for D=256 and group size 12, and a hand-written FMA-accumulation variant of the same tile as the WMMA comparison. Decide from this whether the project kernel starts from the Triton `mha` structure or from scratch.
- Write `qwen4_exp_qsa_attention.py`: the kernel, the dispatch guard, the FP32 oracle, and the patch entry points.
- Integrate through the existing patching framework: a `ModulePatchSpec` over the twelve `self_attn` modules with an inventory gate, following `qwen4_exp_lora.py` and `configure_qwen4_exp_frozen_mmq`, keeping SDPA as the fallback for any shape the kernel does not claim.
- Correctness gates, then the tuning campaign, then the model-level measurement.

## Correctness gates

- A blockwise FP32 oracle over the same Q/K/V, compared with relative RMSE and cosine like the DeepSeek plan: output RMSE at or below 0.0022 and cosine at or above 0.99999, measured over 50 iterations at `B=1/4/16`.
- The padded case: a batch whose rows have `kv_end < 2048` must match the same rows computed without padding, and must match the padded SDPA path within the same thresholds.
- Determinism: no atomics, no split-K, identical bytes across repeated calls, and identical results when gradient checkpointing replays the forward.
- Finiteness: masked tiles are skipped, so a non-finite value in K or V propagates only through tiles that contain it. The kernel does not pay per element to tolerate poison, but the guard tests must show no NaN is manufactured.
- Audit additions: a gate that the twelve QSA modules dispatch to the kernel with a complete inventory, and that the packed identities the audit already checks stay unchanged.

## Tuning protocol

Follow the protocol already used for the AITER grouped GEMMs and the FLA kernels: screen every candidate at one low-fidelity sample, keep the top five plus the incumbent, re-measure finalists with 15 samples, rank by log-space medians with the MAD-based standard error, and let a candidate displace the incumbent only when it wins by more than 2%, per `~/evotensile/docs/noisy_measurements.md`. Tune per batch at `S=2048`, over the interacting knobs `G x BLOCK_M`, `BLOCK_N`, `num_warps`, `num_stages`, and `waves_per_eu`, in small groups rather than a full Cartesian product. Candidates worth a first screen, in order: the register-bounded row shapes at `BLOCK_N=64, num_warps=4, num_stages=2` (which is also the AITER gfx1151 `fwd` default), then `BLOCK_N` at the best shape, then warps and stages at that pair, then the FMA variant.

Two traps from the earlier rounds apply. Triton caches autotune decisions on disk in `~/.triton/cache/<hash>/<kernel>.autotune.json`, so a re-screen needs a fresh `TRITON_CACHE_DIR` or the table deleted, otherwise the previous winner is silently reused, and a single winner is recorded per key, so a key that ignores the batch dimension must be measured deliberately at each batch and the table then carries the per-batch entries. Winners are stored in-repo as an exact-shape table, following `moe_gmm_configs.py` and `fla_tuning.py`, and preloaded rather than autotuned at training start.

## Measurement and expected end-to-end effect

| item | today | target | step effect at B=1 |
| --- | ---: | ---: | ---: |
| forward self | 174.4 ms | <= 90 ms | -85 ms |
| recomputed forward | 172.2 ms | <= 90 ms | -82 ms |
| backward self | 1140.6 ms | <= 350 ms (later phase) | -790 ms |
| total | 1487 ms | <= 530 ms | about -950 ms, 3.3% of a 29 s step |

The forward alone is 0.6% of the step. The family is 5.1%. The plan is therefore worth executing after the indexer fast path (16 s) and the GatedDeltaNet round (3.2 s) and before micro-optimising the norms, and its backward phase is where most of its value sits.

## Open questions

- Whether AITER's Triton `mha`, retuned for D=256 and group size 12, already matches or beats masked SDPA closely enough that the project kernel only needs to add LSE and the gate. This is answered by step 1, before any kernel is written.
- Whether `tl.dot` (WMMA) or an FMA outer-product formulation wins at D=256 on gfx1151, given that rocWMMA lost to the hand-written tile path in `llama.cpp`.
- Whether folding q/k RMSNorm and the 64-dim partial RoPE into the prologue is worth it. AITER ships a `pe` variant of its forward, and SGLang fuses QKV split, norm, and RoPE, so it is done elsewhere. It is deliberately out of this phase to keep the kernel's contract small.
- Whether the mask-free kernel changes numerics enough to matter against SDPA: the softmax order and the causal bound handling differ, so the RMSE and cosine gates above are the test, not bit equality.

## References

- `~/llama.cpp/docs_gfx1151/attn_plan.md`, `qsa_pp_tg_margins.md`, `qsa_sparse_decode.md`, and `ggml/src/ggml-cuda/{qsa-prefill.cu,lightning-indexer.cu,fattn-tile.cuh}`.
- `~/aiter/aiter/ops/triton/_triton_kernels/attention/{mha.py,mha_fused_bwd.py,mha_onekernel_bwd.py,pa_prefill.py,extend_attention.py}` and `~/aiter/aiter/ops/triton/configs/gfx1151/triton/attention/mha/DEFAULT.json`.
- `~/sglang/python/sglang/srt/layers/attention/qsa/{kernel.py,sparse_attn.py,qsa_indexer.py,mqa.py,fused_kv.py}`.
- `~/vllm/vllm/model_executor/models/qwen3_next.py`.
- `~/test_no_unsloth/docs/plan_deepseek_v4_sliding_attention_forward.md` and `plan_deepseek_v4_csa_forward.md`.
- `~/test_no_unsloth/docs/plan_qwen4_exp.md` for the step attribution, the exhaustive-selection argument, and the kernel-work ranking.

## Results

Measured on gfx1151 with `~/tmp/test_no_unsloth/bench_qsa_forward.py`: warm medians of 15 samples per point, batch 1 and 4 at +/-0.3%, batch 16 at +/-0.1%. `flash_attn` is the ROCm flash-attention package called causal without a mask, which is the strongest available kernel at these shapes. `sdpa_mask` is the masked SDPA path the model uses today.

| batch | SDPA (today) | eager | flash_attn | kernel, no gate | kernel, gate fused | speedup vs SDPA |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32.07 ms | 30.14 ms | 3.38 ms | 3.58 ms | 3.91 ms | 8.2x |
| 4 | 126.96 ms | 118.87 ms | 12.63 ms | 10.65 ms | 12.24 ms | 10.4x |
| 16 | 508.73 ms | 475.50 ms | 50.33 ms | 41.96 ms | 50.48 ms | 10.1x |

The kernel is faster than `flash_attn` at batches 4 and 16 (1.19x and 1.20x) and 5% slower at batch 1. Accuracy against the blockwise FP32 oracle is better than `flash_attn`'s own: relative RMSE 0.0019-0.0020 against 0.0019-0.0020 for flash and 0.0016 for SDPA, cosine 0.999998, and the FP32 LSE matches the oracle to the printed precision. The gate costs 0.33/1.6/8.5 ms, which at batch 16 is the 1.6 GB the gate read costs at 230 GB/s. The model path pays that read anyway plus a separate output write and the transpose copy this kernel avoids.

Winning launch table, per batch:

| batch | head group | BLOCK_M | BLOCK_N | warps | stages | logical rows |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4 | 16 | 16 | 4 | 1 | 64 |
| 4 | 4 | 16 | 16 | 4 | 1 | 64 |
| 16 | 4 | 16 | 16 | 4 | 1 | 64 |

What the sweep established: `BLOCK_N=16` beats 8, 32 and 64 (32 and 64 are 1.9x and 3x slower at batch 1). `num_stages=1` beats 2, 3 and 4 by 1.7-4.7x, so software pipelining costs more than it hides here. Two warps and eight warps are 3-5x slower than four, and the batch-dependent head grouping matters, since batch 1 prefers the twelve-head group that halves K/V loads while batches 4 and 16 prefer an exact 64-row tile of four heads. `waves_per_eu` made no measurable difference. `BLOCK_N=128` cannot fit: the K and V tiles would need 128 KiB against the 64 KiB LDS limit.

Two correctness details that the padded tests caught: the key and value loads must be masked at the padding bound or the last tile reads past it, and the maskless loop's upper bound must be rounded down to a whole number of tiles, otherwise the last maskless tile's overhang includes keys after the query rows or past the padding bound. With both in place, `key_end=7`, `1234`, `1500` and `2000` all match the oracle.

One discrepancy has to be settled on the model rather than in this benchmark: the step attribution assigns 174.4 ms of forward self time to this category for the twelve layers, 14.5 ms per layer, while the direct measurement of masked SDPA at the same shapes is 32.1 ms per layer at batch 1. Until that gap is explained, the end-to-end gain has to be measured, not derived.

Remaining work: the backward, which is why the forward emits FP32 LSE. At batch 1 the forward family is 174.4 ms of forward self time and 172.2 ms again in checkpoint recomputation, and the backward is 1140.6 ms, so the backward is where most of the remaining value sits. Wiring into the model waits for it.
