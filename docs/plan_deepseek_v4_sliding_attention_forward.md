# DeepSeek V4 Sliding-Attention Forward Plan

Status: accepted and optimization-exhausted for the exact production geometry on `gfx1151`.

This document owns the dense sliding-attention forward contract, implementation, measurements, compiler evidence, and rejected branches. The coordinated backward is documented in `plan_deepseek_v4_sliding_attention_backward.md`.

## Production contract

The path is deliberately narrow:
- `gfx1151`, wave32, and at most 64 KiB LDS per workgroup.
- physical batches 1, 4, and 16, sequence length 2,048.
- Q `[B,64,2048,512]` and shared K=V `[B,1,2048,512]` in model-native contiguous BHSD storage.
- metadata-only BHSD-to-BSHD views at the kernel boundary.
- BF16 Q/KV/output, FP32 sink `[64]`, and FP32 LSE.
- scale `1/sqrt(512)`, causal window 128, and inclusive left offset 127.
- dropout zero, no padding, cache, custom positions, dense mask, or repeated K/V heads.
- partial RoPE before attention and conjugate output rotation afterward.

Dispatch rejects any unsupported architecture, dtype, shape, head geometry, sequence length, mask/cache/position state, or reachable element offset outside signed int32. No AITER global configuration is mutated.

## Accepted design

`_sliding_grouped_forward_kernel` is project-owned Triton. Each program owns a query-position tile across a group of query heads. It loads compact shared KV once for the group while keeping independent FP32 online-softmax max, denominator, and D512 output state per logical row. Direct interval bounds restrict traversal to the causal 128-token window.

The production forward emits:
- contiguous BF16 BSHD output.
- FP32 LSE `[B,64,2048]`.
- one raw FP16 scaled-score band `[B,64,2048,128]`.

The score band is 32 MiB per batch element. It is not a probability tensor and never expands to S-by-S. Backward consumes it in place: `dV` reads scores, `dQ` reads scores and overwrites them with FP16 `dS`, and `dK` consumes `dS`. This lifetime reuse removes two backward QK dots without another workspace.

The selected launch table is:

| Batch | Head group | Rows/head | N tile | Logical rows | Warps | Waves/EU | LDS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 4 | 16 | 64 | 8 | 2 | 64 KiB |
| 4 | 8 | 8 | 16 | 64 | 8 | 2 | 64 KiB |
| 16 | 16 | 4 | 16 | 64 | 8 | 2 | 64 KiB |

Only the project-owned grouped forward remains in the runtime module. The earlier direct AITER comparison launcher and its tuning table were removed after final selection.

## Correctness

The final 50-iteration B1/B4/B16 benchmark against the blockwise FP32 oracle reports:

| Quantity | Maximum relative RMSE | Minimum cosine |
|---|---:|---:|
| Output | `0.002049` | `0.99999785` |
| Q gradient | `0.002644` | `0.99999648` |
| Shared-KV gradient | `0.003026` | `0.99999541` |
| Sink gradient | `0.000234` | `0.99999988` |

Candidate screening allowed format-aware Q/shared-KV RMSE up to `0.005` and sink RMSE up to `0.003`. After selection, the production tests were tightened to output/Q/shared-KV/sink RMSE limits of `0.0022/0.0030/0.0035/0.0005`. Cosine floors are `0.99999` for output/Q/shared KV and `0.999999` for sink. Non-reentrant checkpoint replay remains deterministic, and the focused suite covers exact shared-KV gradient combination and production-strided `dO`.

## Performance

Authoritative warm medians on Radeon 8060S Graphics are:

| Batch | AITER bridge forward | Earlier direct forward | Final grouped + score forward | Backward | Complete |
|---:|---:|---:|---:|---:|---:|
| 1 | `13.576 ms` | `11.824 ms` | `5.542 ms` | `9.149 ms` | `14.690 ms` |
| 4 | `51.256 ms` | `45.928 ms` | `21.793 ms` | `35.903 ms` | `57.696 ms` |
| 16 | `226.630 ms` | `184.809 ms` | `93.799 ms` | `145.404 ms` | `239.203 ms` |

The raw score store costs only about `0.29/1.00/1.00 ms` relative to the best output-only grouped-forward measurements, while enabling the much larger backward reduction.

Post-cleanup incremental peak allocation is `290.563/1162.250/4648.500 MiB`. Relative to the no-score project boundary, the increase is approximately the expected 32 MiB per batch element. The obsolete persistent Delta workspace is gone. There is no full probability matrix or partial-gradient workspace.

## Work and roofline

There are 254,016 visible pairs per head. Useful forward dot work is `33.294` GFLOP per batch element. Tile issuance is `37.447` GFLOP per batch element, only `1.125x` useful work. Final useful forward throughput is `6.01/6.11/5.68 TFLOP/s`. Issued throughput is `6.76/6.87/6.39 TFLOP/s`.

The gap to the `59.4 TFLOP/s` WMMA peak is not primarily masking. Final compiler metadata shows every grouped-forward variant at 64 KiB LDS, 256 VGPRs, and 212-213 spills, with roughly 2,380-3,180 assembly instruction lines. D512 FP32 output state fills the LDS limit and register file, limiting occupancy. The 128-token window also forces narrow N16 dots interleaved with online-softmax exponentials, reductions, predicates, loads, and stores. The advertised WMMA peak does not cover those operations.

## Optimization history

Accepted progression:
- replaced the generic AITER forward allocation/launch boundary with checked int32 addressing and `torch.empty` state.
- grouped compact KV across query heads, reducing forward by about 2x.
- extended the grouped screen through head groups 16, 32, and 64.
- selected batch-specific group-16/group-8 geometries with 64 logical rows.
- saved a compact raw FP16 score band.
- reused that band in place for the backward `dS` handoff.

Rejected or bounded:
- group 32/64 forward, because reduced KV issuance did not offset weaker dot geometry and occupancy.
- N32 grouped-forward finalists, because N16 was faster at the complete boundary.
- 128-row D512 tiles, because they require 128 KiB LDS.
- two-pass and smaller-logical-row variants, because extra work or lower dot efficiency lost.
- BF16 scores, because sink cancellation produced materially larger error.
- FP32 scores, because 64 MiB per batch element and slower stores did not beat FP16.
- normalized FP16 probabilities, because the `0.08/1.19/4.70 ms` bandwidth pass cost more than two local backward `exp2` streams.
- a fused normalization epilogue, because raw scores already won every complete boundary.
- lower-precision LSE, because its memory saving is negligible and it weakens all probability reconstruction.
- inference kernels from llama.cpp, DS4, vLLM, and SGLang, because they provide no compatible training-state/backward contract.

## External review and exhaustion

AITER confirms direct sliding bounds, LSE/Delta state, and sink recomputation but its generic D512 backward exceeds the LDS limit. llama.cpp and DS4 confirm grouped compact-KV and sink-aware online softmax, but remain inference-only. vLLM and SGLang confirm inclusive window and LSE semantics, but their kernels target cache/extend serving paths. No inspected source provides a D512 compact-MQA training state handoff beyond the measured project design.

The final review found no untested forward candidate that preserves compact KV, differentiable FP32 sinks, FP32 LSE, the 64 KiB limit, and direct gradients while plausibly reducing the measured register/LDS bottleneck. Forward tuning is closed.

## Validation and evidence

The focused suite passes `10 passed`, including exact-shape oracle checks, failure guards, metadata-only layout conversion, checkpoint replay, dispatch, and layer inventory. `py_compile`, Ruff, and `git diff --check` pass. The earlier real B1/S2048 81 GiB GGUF audit remains the model-level gate: both sliding layers dispatched, all 43 non-reentrant checkpoints ran, losses and gradients were finite, all 469 LoRA-B tensors updated, packed hashes were unchanged, frozen grouped outputs had no gradients, and process swap remained zero.

Primary artifacts:

```text
~/tmp/test_no_unsloth/deepseek_v4_sliding_raw_score_handoff_production_final.json
~/tmp/test_no_unsloth/deepseek_v4_sliding_raw_score_handoff_metadata.json
~/tmp/test_no_unsloth/deepseek_v4_sliding_specialized_profile.json
~/tmp/test_no_unsloth/deepseek_v4_sliding_raw_score_complete_finalists.json
~/tmp/test_no_unsloth/deepseek_v4_sliding_grad_score_dk_tuning.json
~/tmp/test_no_unsloth/deepseek_v4_sliding_raw_score_dq_tuning.json
~/tmp/test_no_unsloth/deepseek_v4_sliding_raw_score_dv_tuning.json
~/tmp/test_no_unsloth/deepseek_v4_sliding_probability_normalize_tuning.json
~/tmp/test_no_unsloth/deepseek_v4_training_sliding_specialized_report.json
```
