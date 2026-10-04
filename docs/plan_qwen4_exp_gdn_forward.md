# Qwen4-Exp GatedDeltaNet Forward Plan

This is the forward half of the recurrent layer's kernel work: what it costs now, what has been accepted and rejected with the measurement behind each decision, and what is left. The backward half is in `plan_qwen4_exp_gdn_backward.md`, which also carries the cross-plan priority order for the Gluon rewrites. Both plans were reorganised around their results. The numbers below are the deployed and measured ones, not estimates, except where a line says it is an estimate.

## Production contract

The forward of `Qwen4ExpTextGatedDeltaNet` is the target: 36 of the 48 layers, at batch 1 with a sequence length of 2048, hidden size 2560, 16 key heads, 48 value heads, head dimension 128 for both, conv width 4, BF16 activations and weights with an FP32 recurrent state.

Because this repository only has to serve the shapes it trains, the forward may assume all of that. There is no variable-length path, no attention mask beyond the chunk-causal bound, `NT = 32` chunks of 64 exactly, `H/HV = 1/3` exactly, and no tail chunk. Everything that a general kernel would branch on can be a constexpr.

Three constraints are not negotiable:
- Every forward runs twice per step. All 48 layers are checkpointed with `torch.utils.checkpoint` (`policy: per_decoder_layer`), so each layer's forward kernels execute once in the forward phase and once inside the backward phase as recomputation. Forward work is worth double wall time. That is why this document exists separately from the backward one.
- The value-head convention stays llama.cpp's tiled order (`gdn_tiled_value_heads.py`). That convention is what removes `out_proj`'s input gather and lets it run the native MMQ base, and it is gated by `require_tiled_value_heads`, so any change to the forward must keep its outputs under that order.
- The path stays deterministic. The audit compares losses and clip norms across steps, and the checkpoint replay depends on reproducible forward values. Anything that reorders accumulation inside a tolerance is acceptable. Anything that makes a kernel's output depend on scheduling is not.

Acceptance is the training audit's gates, not bitwise equality: arithmetic order and compute dtype may change as long as the losses, the clip norms and the gradient gates stay in their bands. The fused preparation is the one place this round changed a compute dtype on purpose, and its gates are in `## Accepted`.

## Current measurements

All of it is one layer at batch 1 and sequence 2048, medians over interleaved samples, in the harness that reproduces the model's own kernel times within 2% (`~/tmp/test_no_unsloth/gdn_isolated_bench.py`. The model's `chunk_fwd_kernel_o` was 5.13 ms against the harness's 5.15 before this round). The scripts are listed in `## Measurement protocol`.

### The forward kernels, against their rooflines

Machine numbers from `~/ComfyUI-FeatherOps/docs/gfx1151_reference.md`: 59.4 TFLOPS of BF16 WMMA (80 SIMDs x 256 FLOP/cycle x 2.9 GHz, one 16x16x16 WMMA per SIMD every 32 cycles), 14.8 TFLOPS of FP32 VALU, 928 Gop/s of transcendental, 256 GB/s of DRAM (207 GB/s under load), 128 B/cycle of LDS per WGP, and 1536 VGPR per SIMD allocated in blocks of 24. `MB` is the unique DRAM traffic the formulation requires, counting the value-head replication the chunked form implies.

| kernel | calls per step | us/call | GFLOP | MB | compute us | memory us | measured us | margin | status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `recompute_w_u` | 3 | 3154.1 | 3.22 | 105.3 | 54.2 | 508.5 | 3154.1 | 6.20x | replaced by `gdn_wu_recompute.py` at 0.551x |
| `chunk_fwd_o` | 2 | 2922.2 | 6.44 | 151.4 | 108.5 | 731.3 | 2922.2 | 4.00x | the one Gluon candidate on this side |
| `kkt_solve` | 2 | 734.4 | 1.61 | 38.1 | 27.1 | 184.3 | 734.4 | 3.99x | left alone, see `## Rejected` |
| `conv1d` forward | 2 | 1274.8 | 0.34 | 83.8 | 5.7 | 404.8 | 1274.8 | 3.15x | removed by the fused preparation |
| `chunk_fwd_h` | 3 | 1221.0 | 6.44 | 151.4 | 108.5 | 731.3 | 1221.0 | 1.67x | F4, algorithmic |
| `_prep_fwd_kernel` | 2 | 662 | 0.20 | 119.0 | 3.4 | 575 | 662 | 1.15x | deployed when F1 is wired |
| `_prep_bwd_kernel` | 1 | 916 | 0.34 | 165.0 | 5.7 | 797 | 916 | 1.15x | deployed when F1 is wired |
| `_prep_dmixed_kernel` | 1 | 399 | 0.17 | 83.8 | 2.9 | 405 | 399 | 0.98x | deployed when F1 is wired |
| `wu_recompute` | 3 | 1650 | 3.22 | 113.4 | 54.2 | 548 | 1650 | 3.01x | replaces `recompute_w_u`, wired |

Every one of the FLA kernels is memory-bound in the model's sense - its FLOPs are 5-13% of what the same time could buy - and every one is also far above its memory roofline. None is limited by arithmetic, and only `chunk_fwd_h` is close to limited by bandwidth. The fused preparation's kernels are the opposite: three of them are at 1.15x, 1.15x and 0.98x of their traffic roofline, which is what a memory-bound kernel with a four-shifted-load convolution can be. Its two small kernels are launch-bound (27 us with 16 programs, 37 us for the reduction) and are far below the level where anything else in the layer would notice.

### What the kernels run with

`rocprofv3 --kernel-trace --stats --pmc L2CacheHit VALUInsts LDSBankConflict` over the one-layer harness, at the deployed configurations (`~/tmp/test_no_unsloth/gdn_prof2/gdn_results.db`):

| kernel | VGPR | LDS | workgroup | occupancy cap | waves/SIMD | L2 hit | VALU units |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| `chunk_fwd_o` | 256 | 16384 | 128 | VGPR: 264 per wave | 5 (31%) | 67.1% | 1489 |
| `chunk_fwd_h` | 248 | 8192 | 128 | VGPR | 5 (31%) | 86.8% | 20690 |
| `kkt_solve` | 256 | 4096 | 32 | VGPR | 5 (31%) | 47.8% | 9549 |
| `recompute_w_u` | 256 | 8192 | 64 | VGPR | 5 (31%) | 72.2% | 5724 |
| `conv1d` backward | 248 | 38160 | 128 | LDS: 3 workgroups, 4 waves each | 3 (19%) | 50.2% | 3265 |
| `conv1d` forward | 56 | 9648 | 128 | LDS: 13 workgroups, 4 waves each | 13 (81%) | - | - |

Triton pins every one of these at the 256-VGPR ceiling: `ceil(256/24)*24 = 264`, so `floor(1536/264) = 5` waves per SIMD, a 31% occupancy cap, the same for all of them regardless of how the tile knobs are set. The pressure is structural - each kernel keeps a `[BT, BT]` FP32 accumulator or a `[BK, BV]` state tile live for the whole program, and at 64 lanes that tile alone is 64 registers per lane. LDS bank conflicts are not the problem: 37-49 per dispatch against 1.5k-20k VALU instruction counts. Three of the four spill, from 1.3 KB per thread in `kkt_solve` to 1.8 KB in `recompute_w_u`.

Before this round's screens the numbers were worse in the two places that mattered: `chunk_fwd_kernel_o` at the `BK=BV=128` tile it was deploying was at 216 VGPR and 49 KB of LDS, which capped it at two workgroups per WGP, and `recompute_w_u_fwd_kernel` was at 256 VGPR with 1,652 B of spill per thread.

### What the step pays, before and after

The forward-side family costs, with the per-kernel numbers above and the deltas the two accepted changes carry:

| kernel | before this round | deployed now | after F1 and F2 are wired |
| --- | ---: | ---: | ---: |
| `chunk_fwd_kernel_o` | 5.13 | 2.96 | 2.96 |
| `recompute_w_u_fwd` | 2.93 | 2.93 | 1.74 |
| `causal_conv1d` forward | 1.58 | 1.58 | removed |
| `chunk_gated_delta_rule_fwd_h` | 1.22 | 1.22 | 1.22 |
| `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 0.68 | 0.68 | 0.68 |
| `l2norm_fwd` (q and k) | 0.43 | 0.43 | removed |
| `layer_norm_gated_fwd` | 0.33 | 0.33 | 0.33 |
| `chunk_local_cumsum_scalar` | 0.01 | 0.01 | 0.01 |
| elementwise kernels and copies inside the module | ~1.2 | ~1.2 | ~1.2 (the fused preparation owns most of them) |
| total, one pass | 14.2 | 12.0 | 8.1 |

| state | per layer per step | per step over 36 layers | share of the 10.14 s warm update |
| --- | ---: | ---: | ---: |
| before this round | 28.7 ms | 1.03 s | 10.2% |
| deployed (F0) | 24.0 ms | 0.87 s | 8.5% |
| F1 and F2 wired | 10.2 ms | 0.37 s | 3.6% |

The projection family of this layer (`in_proj_qkv`, `in_proj_z`, `in_proj_a/b`, `out_proj`) is costed in `plan_qwen4_exp.md` and is not repeated here. The model-level gate is still the audit's: it is not run in this round, so the deployed share is measured at the kernel level and predicted at the step level.

### The measurements behind the analysis

Three things shape everything below, and all three are measurements rather than readings.

The kernels wait, and they wait inside each program. `chunk_fwd_h`, the best-behaved of them, is 151.4 MB at 124 GB/s. `chunk_fwd_o` is 151.4 MB at 52 GB/s. With L2 hit rates of 48-87% the DRAM side is doing even less work than that. A scale test settles what the limit is: `gdn_roofline.py` runs the same kernels at batch 16 and sequence 2048, which is 16x the work, and they take 15.1x to 16.2x the time. Sixteen times the programs, sixteen times the bytes in flight in aggregate, no economy of scale at all. Occupancy and wave count are not what is missing.

Every knob that moves work between programs loses. For `chunk_fwd_o` a 4x larger tile moves 9% (5.135 ms at `BK=BV=128` with 8 warps against 2.957 ms at `BK=16, BV=128` with 4 warps and two stages), and the stage count dominates the tile shape. For `recompute_w_u`, splitting its two matmuls across separate programs to halve the live set and double the program count is worse at every point (1.779 ms at best against 1.760 ms unsplit) and by far more than the 11% of extra traffic the second `A` load costs. Shrinking the output tile below 32 loses too.

The one thing that does work in Triton is more independent work per program, and it saturates. Two chunks per program, unrolled, so there are two independent load-to-dot chains in flight, is worth 6.4% on `wu_recompute` (1.764 -> 1.652 ms). Four chunks is already worse than two.

What is left, then, is shared-memory traffic, and reading the kernels settles where it comes from. Every `tl.dot` stages its operands through LDS, so a tile *computed* in one layout and consumed in another - which is what a transpose of an accumulator is - costs a store and a load of the whole tile.

`chunk_fwd_o` on this path transposes nothing at all: it loads `q` as `[BT, BK]` and `k` as `[BK, BT]` with the index expressions swapped, which Triton turns into the right operand layout without a conversion, and its only accumulator-to-operand move is `b_A` feeding the intra-chunk dot. `bwd_dqkwg` transposes `b_ds`, computed inside the kernel from `do` and `v`, so no choice of load pattern avoids it. `prepare_wy_repr_bwd` transposes both a loaded tile, which a transposed load could in principle avoid, and a dot result, which it could not.

`bwd_dhu` transposes a loaded `dh` tile and the state tiles it accumulates, and `chunk_fwd_h` transposes its state tiles. The remaining conversions are therefore mostly on computed tiles, which is the case that needs explicit layout control rather than a better index expression. The reference's own note about `ds_load_b128` describes the resulting failure: LDS processes 128 bytes per cycle per WGP, so several resident waves issuing block-sized LDS instructions saturate the LDS instruction queue and the sequencer stalls on the instruction rather than on the data.

That is what `AMDWMMALayout` in Gluon removes, by telling the compiler the layout the dot wants instead of converting into it - but the backward plan's B8 is the reminder to look one level lower first. Its `bwd_dhu` carried the family's largest VALU count by a factor of five, and the cause was not the arithmetic this section would have blamed: a decay multiply promoted a BF16 operand to FP32, so `tl.dot` was handed two FP32 operands and lowered to FP32 FMA on the VALU instead of the matrix cores, which is also what made the tile spill.

One cast, worth 2.3x. Every forward kernel was checked for the same pattern and casts its dot operands back to BF16 before the dot - `chunk_fwd_kernel_o` after the score tile, `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` after the decay multiply, and `gdn_wu_recompute.py` at both matmuls - so the forward side is clean. The diagnostic itself, an opcode histogram read as an FP32-FMA-to-WMMA ratio, is in the measurement protocol.

Two cost centres that the kernel table does not show:
- The chunk walk. The same arithmetic and the same kernels at a shallower walk: `(B=1, S=2048)` 26.7 ms of forward against `(B=16, S=128)` 20.7 ms, so 6.3 ms/layer is the serialization itself. `fwd_h` and the `o` pass each walk 32 chunks with a dependent state and no pipelining across chunks. The depth sweep puts a ceiling on what restructuring that can recover: cutting the walk from 32 chunks to 2 while keeping the total work and the program count identical buys about 20%, which is the 10.2 ms the two walk kernels cost between them.
- The conv. 2048 tokens x 10240 channels x 4 taps is 168 MFLOP, and it costs 1.54 ms per pass in a channel-last kernel with 128-channel tiles and 9.6 KB of LDS, shaped for a much larger shared-memory budget. That is the whole reason the fused preparation is worth 3.4% of the step.

## Prior art

| source | what it establishes |
| --- | --- |
| `fla/ops/gated_delta_rule/chunk.py`, `chunk_fwd.py`, `wy_fast.py`, `../common/chunk_h.py`, `../common/chunk_o.py`, `../common/chunk_scaled_dot_kkt.py` | the current decomposition: `chunk_local_cumsum` -> fused KKT/solve (and a separate `recompute_w_u`) -> `fwd_h` state pass -> `chunk_fwd_o`. The WY preparation is the fusion target `## Rejected` settles, and `chunk_fwd_kernel_o` is the kernel whose launch configuration this plan retuned, from the 128x128 tile FLA's own candidate list collapses to on this device down to `BK=16, BV=128`. |
| `~/vllm/vllm/third_party/flash_linear_attention/ops/fused_gdn_prefill_post_conv.py` | the accepted single-kernel preparation: "Replaces the chain: split -> rearrange -> contiguous x 3 -> l2norm x 2 -> gating with a single Triton kernel", `BLOCK_T = 16`, `grid = (ceil(L/BLOCK_T), H + HV)`, `num_warps = 4`. The fusion F1 implements, in the serving layout. |
| `~/vllm/vllm/model_executor/layers/mamba/ops/gdn_chunk_cutedsl/kernel_kkt_inv_uw.py` | `Sm100ChunkUWKernel`, "Compute per-chunk KKT inverse preprocessing and U/W tiles": one kernel producing `A`, `U` and `W` from `K`, `V`, `g` and `beta`. Blackwell-only, and `## Rejected` explains why the same fusion is not worth writing here. |
| `~/vllm/vllm/model_executor/layers/mamba/ops/gdn_chunk_cutedsl/kernel_h.py` | their state pass: `grid = (Hv, batch, 1)` with `num_stages = 2` and TMA, kept serial per (value head, batch) and parallelised across value heads only. That is the parallelism limit F4 attacks. |
| `~/vllm/vllm/model_executor/layers/mamba/ops/ssd_state_passing.py` | the mamba2 path's state passing kernel, which turns a per-chunk recurrence into per-segment local passes plus one small combine. The precedent for the two-level scan, and the reason to expect the combine to be cheap. |
| `~/sglang/python/sglang/kernels/ops/attention/triton_gdn_fused_proj.py` | `fused_qkvzba_split_reshape_cat_kernel`, one pass from the fused projection to `q/k/v/z/b/a`, `NUM_HEADS_QK = 16`, `NUM_HEADS_V = 48`: the same preparation idea with the group ratio baked in. |
| `~/llama.cpp/ggml/src/ggml-cuda/gated_delta_net.cu`, `gdn-conv.cu` | the register-resident state (warp per column, `DPP` cross-lane reductions, `keep_rs_t` snapshots) and `gdn_conv_direct_kernel` with a fused SiLU. Inference is token-serial, but its state ownership is the shape a chunk-parallel `h` pass should keep for its own tile. |
| `~/test_no_unsloth/docs/plan_qwen4_exp_qsa_forward.md`, `plan_qwen4_exp_qsa_backward.md`, `plan_qwen4_exp_qsa_kv_owner_rewrite.md` | this project's own attention kernels: owner-per-output-element rather than atomics, FP32 accumulation with BF16 storage, small tiles winning, Triton hitting the 256-VGPR ceiling and spilling, and the Gluon rewrite that removed both and was 12-15% faster and bit-identical. |
| `~/ComfyUI-FeatherOps/docs/gfx1151_reference.md` | the hardware budget and the rocprofv3 recipes used throughout this document. |

## Accepted

Two kernels and one table entry, all verified at the kernel and gradient level. `gdn_config_check.py` reports the worst relative L2 over the ten parameter gradients at 8.5e-07 with nine of them exactly zero.

### F0: `chunk_fwd_kernel_o`'s launch configuration (deployed)

Table entry in `fla_tuning.py`, keyed by kernel name and Triton's cache key and injected through `Autotuner.run`: `BK=16, BV=128, num_warps=4, num_stages=2`, up from what the model was running (`BK=BV=128, w8, s3`, an entry that had itself replaced `BK=32, BV=64, w4, s3`).

| config | kernel ms | against the entry it replaced |
| --- | ---: | ---: |
| BK16 BV128 w4 s2 (deployed) | 2.957 | 0.70x |
| BK32 BV64 w4 s2 | 3.138 | 0.75x |
| BK16 BV64 w4 s3 | 3.206 | 0.76x |
| BK32 BV128 w4 s2 | 3.338 | 0.79x |
| BK64 BV128 w4 s2 | 3.579 | 0.85x |
| BK32 BV64 w8 s2 | 3.804 | 0.90x |
| BK32 BV128 w4 s3 | 4.150 | 0.99x |
| BK32 BV64 w4 s3 (the entry two rounds back) | 4.209 | 1.000x |
| BK32 BV32 w4 s2 | 4.211 | 1.00x |
| BK64 BV64 w4 s2 | 4.813 | 1.14x |
| BK128 BV128 w8 s3 (what the model was running) | 5.135 | 1.22x |
| BK64 BV64 w4 s3 | 6.330 | 1.50x |
| BK64 BV32 w4 s3 | 6.820 | 1.62x |

`chunk_fwd_kernel_o`'s grid is `lambda meta: (cdiv(V, meta['BV']), NT, B*HV)`, so a `BK`/`BV` change is grid-safe and a table entry is enough to deploy it. That is not true of every kernel in this family, and the backward plan records the two that need their launcher's constants changed instead. 5.135 -> 2.957 ms per call, and the kernel runs in the forward pass and again in the checkpoint recomputation, so -4.36 ms per layer per step.

The stage count matters more than the tile: 4.209 ms at three stages against 3.138 ms at two for the same 32x64 tile, and a 128x128 tile with 8 warps is 1.22x slower than the deployed entry because of what it does to the register and LDS budget.

The original expectation for F0 named two more entries, both of which the paired screen reverted (see `## Rejected`). Its estimate was -5.7 ms/layer/step, 2.0%. The measured result is smaller because two thirds of it did not exist.

### F1: the fused preparation (measured, tested, not wired: 0.35 s of step time for 6.3 GiB)

Wired into the trainers and audits, measured twice, and disconnected again in the same round. The two runs agree on what it does: with it installed the audit's first warm update is 8.96 s against 9.31 s for the same process without it, so its kernels are worth 0.35 s, 3.6% of the step - and every run with it allocates 6.3 GiB more (72.28 GiB against 65.97 GiB after the measured steps), which is what then costs the step: that run follows its 8.96 s update with a 13.41 s one, while the run without it repeats 9.31 s. The isolated harness sees only the first half of that trade, which is why it measured a clear win and why this round wired it before the model-level gate ran.

The 6.3 GiB is the op's saved state. `_GdnPrepFunction` keeps `mixed` and its five outputs for its backward, about 120 MB per layer at this geometry, and gradient checkpointing means that state is produced during the recomputation and held until the backward consumes it, so all 48 layers' worth is live at the peak. The eager chain saves nothing comparable, because its intermediates are recomputed rather than kept. The fix is the one the small-items list already names: let the backward recompute what it can from the four shifted loads and keep only what it cannot, which is the difference between 120 MB and a few megabytes per layer. Until then the preparation stays out, and the deployed state is the 9.31 s one.

`gdn_fused_prep.py` replaces the whole preparation of the chunked core with one forward kernel and four backward ones: the projection output's trip through the channel-last depthwise convolution, its SiLU, the transpose back, the split and reshape into `q/k/v`, the tiled key-to-value head broadcast, FLA's L2 normalization of the expanded heads, and the sigmoid and softplus gating. Its outputs are what the core consumes: `q` and `k` normalized and expanded in the tiled order, `v` convolved, `g` in FP32 log space, `beta`.

Measured against the chain the model runs, with the same gradient seeding and no loss reduction inside the timed region:

| | forward ms | backward ms | total ms |
| --- | ---: | ---: | ---: |
| the model's chain (`causal_conv1d` hub kernel, `repeat`, FLA `l2norm`, gating) | 2.362 | 7.875 | 10.237 |
| the fused preparation | 0.676 | 1.553 | 2.229 |
| saving | 1.685 | 6.322 | 8.008 (78%) |

Per layer per step it runs twice in the forward and once backwards, so 2 x 0.676 + 1.553 = 2.906 ms against 12.599 ms: -9.69 ms per layer per step, -0.35 s per step, 3.4% of the warm update, which is the 8.96 s the model-level run measured. The original estimate was 1.1-1.3%, for the conv and the elementwise tail alone. The measured result is larger because the backward side of the preparation turned out to be the bigger half.

Accuracy against the model's chain, flat FP32 cosine and relative RMSE:

| tensor | cosine | relative RMSE | | gradient | cosine | relative RMSE |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| q | 1.00000000 | 2.3e-05 | | d mixed | 0.99999380 | 3.5e-03 |
| k | 1.00000000 | 0.0 | | d conv weight | 0.99999744 | 2.3e-03 |
| v | 1.00000000 | 1.4e-05 | | d a | 1.00000000 | 0.0 |
| g | 0.99999994 | 8.9e-08 | | d b | 0.99999845 | 1.7e-03 |
| beta | 0.99999994 | 0.0 | | d A_log, d dt_bias | 1.00000000 | 0.0 |

The gate parameters' gradients come out bit-exact. The three that carry error all pass through the BF16 `dz` intermediate or are a BF16 output themselves, which is where the 1e-3-level error comes from. `dz` is BF16 by choice, measured against FP32 (2.33 ms) and FP16 (1.72 ms) for the backward, because BF16 cannot overflow where FP16 can. The kernel reads the conv's weights, `[10240, 1, 4]`, from memory rather than keeping them in registers, and it takes the token block as a constexpr: 128 rows with 4 rows per iteration measured best (0.686 + 1.385 ms against 0.672 + 1.562 for 64 rows, with 32 and 256 both worse).

`test_gdn_fused_prep.py` gates all of it in fifteen tests: the five outputs and six gradients against the model's chain, the tiled head order, the unit row norms that let the core run with `use_qk_l2norm_in_kernel=False`, determinism, non-contiguous gradients, the argument validation, that a different token block gives the same answer, and the wiring itself - the patched layer's output and gradients against the eager one, with and without a padding mask, the idempotence of the patch, and the skip counter for a geometry it cannot serve.

Three defects that only measuring found, all of them silent, are recorded here because each is a trap for the next kernel in this family:
- The normalization backward first summed `y` over the broadcast copies as well as the gradient, which tripled the sum of squares and shrank `rstd` by sqrt(3). The copies hold the same row, so only the gradient sums.
- Autograd hands the gate's gradient over as a non-contiguous broadcast. The kernels index every gradient row-major, so reading it at face value picked up 72000-sized values from unrelated memory while `d b` stayed correct, which made it look like a gate-kernel bug rather than a layout one. The backward now makes each incoming gradient contiguous.
- The transposed convolution reads *ahead* of the row it writes, so its `source < T` bound can never be dropped. Unifying the row masks dropped it, and the sweep then read past the end of `dz` and faulted the device. The test geometry, sequence 100 against a 64-row block, had exercised the same path without faulting, which is why the bound belongs in the code rather than in a comment.

Wiring it needs two things beyond the patch itself: the model must call the core with `use_qk_l2norm_in_kernel=False`, because the preparation normalizes, and the head order must be the tiled one the `out_proj` weights are stored in. Both are in `configure_fused_preparation`, which patches the layer forwards through the shared module-patching protocol, and the patch records the broadcast it performs so the tiled gate keeps reading 36 of 36 layers.

### F2, first half: the W and U recomputation (measured, tested, wired)

`gdn_wu_recompute.py` is a drop-in for `fla.ops.gated_delta_rule.wy_fast.recompute_w_u_fwd`: same inputs, same outputs, called from inside FLA's own autograd function where the call is not tracked. It replaces two per-chunk matmuls, `u = A @ (v * beta)` and `w = A @ (k * beta * exp2(g))`, that FLA runs with `BK = BV = 64` and two warps, so a 64x64 accumulator spreads over 64 lanes.

| output tile | warps | chunks per program | ms | against FLA's kernel | relative RMSE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 1 | 2 (deployed) | 1.650 | 0.551x | 4.7e-03 |
| 32 | 1 | 4 | 1.698 | 0.564x | 4.7e-03 |
| 32 | 1 | 1 | 1.764 | 0.586x | 4.7e-03 |
| 16 | 2 | 1 | 1.955 | 0.645x | 4.7e-03 |
| 16 | 1 | 1 | 2.000 | 0.660x | 4.7e-03 |
| 64 | 2 | 1 | 2.910 | 0.960x | 4.7e-03 |
| 64 | 1 | 1 | 2.932 | 0.967x | 4.7e-03 |
| 128 | 4 | 1 | 3.196 | 1.054x | 4.7e-03 |

The kernel runs `block_d=32`, one warp, two stages and two chunks per program: 2.995 -> 1.650 ms per call, so 0.551x of what it replaces, and the two calls a step it serves - one in the forward, one inside the backward's recomputation - make it -2.7 ms per layer per step. The same inverted answer as the attention sweeps applies - the small tile with one warp wins and the widest tile loses - and two chunks per program was added after the roofline work showed these kernels are limited by the serial load-to-dot chain inside a program rather than by wave count.

Two defects the tests caught, both of which would have read plausible numbers: the token block is not a free tiling parameter (the contraction of both matmuls is the chunk, so a smaller block sums part of the chunk and a larger one reads into the next chunk, and the sweep's apparent 0.535x winner at 16 rows was computing a quarter of the sum), and `w` carries the *value* head axis while `k` carries the key axis, so indexing one with the other's offset walks off the buffer. `test_gdn_wu_recompute.py` gates the comparison against FLA's kernel, the layout, the gate-free path, determinism and the argument validation.

## Rejected and closed

Everything here was measured or reasoned to a decision, and the reason is recorded so the next round does not repeat it.

F0's other two entries: reverted. Both measured *slower* than the entries already deployed, so the table keeps what it had:

| kernel | deployed | candidate | candidate ms | deployed ms | ratio |
| --- | --- | --- | ---: | ---: | ---: |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | BV32 w4 s1 | BV64 w8 s2 | 2.871 | 2.432 | 1.18x |
| | | BV32 w8 s1 | 2.662 | | 1.09x |
| | | BV16 w4 s1 | 2.856 | | 1.17x |
| | | BV32 w2 s1 | 3.407 | | 1.40x |
| | | BV128 w8 s1 | 5.757 | | 2.37x |
| `recompute_w_u_fwd_kernel` | w2 s3 | w2 s2 | 6.536 | 5.821 | 1.12x |
| | | w1 s3 | 6.627 | | 1.14x |
| | | w4 s3 | 8.068 | | 1.39x |

`chunk_gated_delta_rule_fwd_kkt_solve_kernel` keeps its entry too: `BK=64, num_warps=1, num_stages=3` measures 0.695 ms against 0.684 ms at two stages, inside the noise. `dv_local` is on the backward side and its screen is there.

The first screen of the round reported the opposite for all of these, because its baseline called FLA's autotuner directly instead of the tuning table's injection hook, so it measured every candidate against FLA's collapsed fallback. That fallback is 6.36 ms for `chunk_fwd_kernel_o` (not the 5.14 ms the model ran) and 9.20 ms per pair for `recompute_w_u_fwd` (not 5.82 ms).

Every earlier "win" in this plan's tables has to be read as "better than FLA's fallback", which is not the same question as "better than what is deployed". `gdn_config_probe.py` now installs the table before wrapping the runner, and `gdn_fwd_screen.py` interleaves candidates so drift cannot favour one.

Re-enabling FLA's own device gates instead of extending the table. FLA builds its candidate lists from `check_shared_mem(arch)`. gfx1151 reports 65,536 bytes per workgroup, so every tier reads false and `BKV_LIST` collapses to `[32]`. Patching the gate alone measures 101.03 ms against 100.84 ms per layer, because the table pins the config of every kernel it covers. The gates only matter where the table has no entry. `~/tmp/test_no_unsloth/gdn_gate_probe.py` is the experiment.

Widening `dqkwg`'s tiles through a launcher constant (the backward plan's B1). `gdn_tile64_probe.py` patches `check_shared_mem` inside `fla.ops.common.chunk_o`, which is where both `dqkwg` and `dv_local` read it, and the widest tier takes `CONST_TILING` to 128: `dqkwg` then measures 10.02 ms against the deployed 32-wide tile's 10.92 ms, 8%, worth 0.75 ms per layer per step. Two things make that a poor trade on its own: the tile is part of the autotune key, so the widened launcher needs its own table entries for two kernels, and 8% is what a 4x wider tile buys a kernel that is not limited by it at all. B9 in the backward plan supersedes this route: the same widening through a launcher of our own, with 8 warps and two stages, measures 22% (`BK128 BV32 w8 s2`: 10.833 -> 8.459 ms) and is deployed. The difference is the warps and stages the module-wide patch leaves to the autotuner.s tile at all.

Fusing the block triangular solve with the W/U matmuls (F2's second half). The fusion removes from memory the `A` round trip (12.6 MB written and 12.6 MB read, 0.12 ms) and `k`'s second read (about 25 MB, 0.12 ms): 0.24 ms per call, 0.7 ms per layer per step, 0.25% of the update. It fixes neither kernel's margin, because both are limited by their per-program chains and their LDS staging rather than by that traffic, and a fused kernel inherits the same chains.

Against that it has to re-implement FLA's blocked triangular solve in a language that cannot slice a tile: FLA's kernel builds the ten 16x16 blocks of the KKT matrix with ten separate dots and forward-substitutes the four diagonal blocks in place, precisely because Triton offers no other way to reach a sub-block, and the fused version would then have to consume a 64x64 inverse it cannot assemble from those blocks and hand to a large dot. The two kernels it would replace cost 2.38 ms per call together, and one of them is already replaced at 0.551x of its half.

F3, the chunk size: closed. FLA accepts only `chunk_size` 16, 32 or 64 (`chunk_gated_delta_rule_fwd_intra` raises for anything else), and 32 faults on this device inside `recompute_w_u_fwd_kernel`. `chunk_size=64` is not a choice, it is the only value that runs, which also means the cache-key interaction that would have made this expensive is moot.

Other shapes through the same kernels. `BV=32, num_warps=4, num_stages=3` for the state pass needs 106 KB of LDS against the 64 KB per-workgroup limit, so the stage-and-tile budget is real and the two have to be traded against each other rather than combined. For `chunk_fwd_o`, `BK=64, BV=32` was 1.074x slower than the entry of the time while `BK=32, BV=64` was 0.664x faster: the asymmetry comes from the accumulator shape, not from the tile area. `num_warps` 4 or 8 with two to four stages for `recompute_w_u_fwd` is 0.86-1.06x of the incumbent everywhere except `w2, s2`.

A bitwise-identical rewrite. Not required and counterproductive: the acceptance gate is the audit's bands, and every accuracy figure in this document is against the model's own chain, not against bit-equality.

## What remains

### F4: the two-level scan for the state pass (algorithmic, no Gluon needed)

`chunk_gated_delta_rule_fwd_h` walks 32 chunks serially per (value-head, tile) block, and the forward pays 6.3 ms/layer for that serialization. A two-level version keeps the per-chunk algebra and changes only the schedule: each block handles a segment of consecutive chunks and writes its local state pairs, one small combine kernel reduces the segment boundaries across the sequence, and then the state pass or the `o` pass applies the segment prefix. vLLM's mamba2 path does exactly this in `ssd_state_passing.py`, so the algebra is known.

Worth: recovering half of the forward's 6.3 ms/layer is -3 ms/layer/step, -0.11 s/step, 1.1%. The measured ceiling for the whole idea is the 10.2 ms the two walk kernels cost between them, because the depth sweep says only about 20% of the walk's cost is the walk. Risk: more state traffic, a second launch, and a new place for an accumulation-order difference.

One thing decides the shape of the combine. The chunk recurrence is `h_out = h_in * decay + k^T (u - w @ h_in)`, which is affine but not *scalar-decay* affine: it is `h_out = (decay * I - k^T w) @ h_in + k^T u`, so a segment's transition from its incoming state to its outgoing state is a full `[K, K]` matrix per (segment, head), not a scalar. Composing one costs `K^3 = 2.1M` MACs per chunk-head, the same order as the state step itself.

For four segments and 48 value heads that is 302 MMAC per layer, about 0.01 ms at this machine's WMMA rate, which is cheap, but the kernel has to be written as a `[K, K]` composition rather than as a scalar prefix product. FLA's context-parallel path does exactly this and calls the result `compress_h0`/`expand_h0`, so it is where to start reading. `chunk_gated_delta_rule_bwd_dhu_pre_process` and the segment handling in `chunk_delta_h.py` are the same machinery.

### The Gluon candidate on this side

`chunk_fwd_o` is the only forward kernel among the four large margins: 2.93 ms per call, one call per step, 2.93 ms per layer per step at 4.00x its roofline, with a `[BT, BT]` score accumulator and a `[BK, BV]` state tile live at once. It is the one kernel in the family that transposes nothing on the deployed path - its `q` and `k` loads use swapped index expressions, which Triton turns into the right operand layout for free - so its 4.00x is not a conversion problem: the two accumulators hold the tile count down, which is a register-and-schedule problem of the same kind as the other three.

A rewrite that reached half the roofline would take it to 1.83 ms per call, a prize of 1.10 ms per layer per step over its single call, 0.4% of the step. The other three of the original four are in the backward plan, which carries the priority order: `bwd_dqkwg` and `wy_repr_bwd` are still open there, and `bwd_dhu` was closed at the source by its B8 rather than by a rewrite.

`chunk_fwd_h` is not a Gluon candidate: it is the best-behaved kernel in the family at 86.8% L2 hit and 1.67x its roofline, and what remains for it is F4.

### Wiring, then the audit

Both accepted new kernels are drop-ins for FLA's own functions and neither is wired in yet, because the wiring was deliberately deferred to the round that also brings the backward's replacements. That round is also the first one that can run the model-level gate, and until it does, the step-level shares in `## Current measurements` are predictions from kernel measurements rather than audit numbers.

### Small items

- `use_gate_in_kernel` in the FLA call would fold the gate's cumsum pass into the chunk machinery. The pass is 0.008 ms/layer, so this is a launch-count item, not a time item.
- The gated norm before `out_proj` is already the FLA fused kernel (0.33 ms/pass) and its output feeds a packed MMQ projection that quantizes to Q8_1. A fused "gated norm + Q8_1 quantize" kernel would remove one pass over 25 MB, about 0.2 ms/layer. Low priority.
- The fused preparation's `dz` could be removed entirely: the backward already recomputes the convolution's pre-activation from the same four shifted loads, so the transposed convolution could do the same and take a three-row halo of redundant work instead of a 40 MB round trip, about 0.4 ms of the 1.38 ms backward. Both kernels are at 1.15x of their roofline already, so this is the difference between good and slightly better.

## Correctness gates

- Kernel-level: `~/tmp/test_no_unsloth/gdn_config_check.py`'s comparison of all ten parameter and input gradients between the incumbent and the candidate, relative L2 at or below 1e-5. It reports 8.5e-07 for the deployed table entries.
- New kernels: `test_gdn_fused_prep.py` (ten tests) and `test_gdn_wu_recompute.py` (seven) gate each replacement against the FLA kernel or the model's chain that it replaces, plus layout, determinism and argument validation. Every accuracy figure in `## Accepted` comes from those tests or from the harnesses they share.
- Layer-level: forward output and input gradient against the untouched module, and `test_gdn_tiled_value_heads.py` for the value-head conventions.
- Model-level, still pending: `audit_qwen4_exp_training_step.py` must reach `GATE PASS` with the losses inside the recorded band (3.4896 and 3.4860 for the two measured updates), the clip norms inside theirs (0.2734, 0.2832, 0.3301), and the warm update at or below 10.14 s. Once F0's -0.55 s and the two kernels' further -0.48 s land, the number to beat is that 10.14 s, not a new estimate.
- Determinism: two identical runs of the isolated layer must produce identical outputs and gradients, which both new kernels satisfy in their tests.

## Measurement protocol

- One layer at the training geometry, in the harness that reproduces the model's kernel times: `python gdn_isolated_bench.py --repeats 5 [--kernel-table] [--depth-sweep] [--batch 16]`.
- The roofline table and the scale test: `python gdn_roofline.py --iterations 5`.
- Paired config screens, which interleave their candidates inside one session because a cross-session comparison on this APU drifts by several percent: `python gdn_fwd_screen.py --kernels o,h,w_u,wy,dv_local,kkt --repeats 5`, and `python gdn_config_probe.py --kernel <name> --table`.
- The new kernels: `python bench_gdn_fused_prep.py --repeats 25`, `python check_gdn_wu.py --repeats 15 --sweep`, `python profile_gdn_fused_prep.py`.
- Resource counters: `rocprofv3 --kernel-trace --stats --pmc L2CacheHit VALUInsts LDSBankConflict -d gdn_prof2 -o gdn -- python gdn_isolated_bench.py --repeats 3`.
- What a kernel is actually doing, in three steps: `analyze_gdn_stalls.py` for the stall and traffic counters, `analyze_gdn_isa.py <kernel>.amdgcn` for the opcode histogram out of Triton's cache, and `analyze_gdn_dots.py <cache>` for the FP32-FMA-to-WMMA ratio. A kernel with many FP32 FMA and few WMMA has a dot on the VALU, which is the diagnostic that found the backward plan's B8.
- End-to-end: `~/test_no_unsloth/train_qwen4_exp.py` for the step time and `audit_qwen4_exp_training_step.py --max-steps 3 --profile-output ...` for the attribution.
- Evidence for this document: `~/tmp/test_no_unsloth/gdn_isolated_bench.py`, `gdn_roofline.py`, `gdn_fwd_screen.py`, `gdn_config_probe.py`, `gdn_config_check.py`, `gdn_candidates.py`, `gdn_gate_probe.py`, `gdn_tile64_probe.py`, `bench_gdn_fused_prep.py`, `check_gdn_wu.py`, `profile_gdn_fused_prep.py`, `analyze_qwen4_gdn_layer.py`, `analyze_qwen4_gdn_kernels.py`, `gdn_prof2/gdn_results.db` (the counters above. `gdn_prof/gdn_results.db` is the earlier run at the previous configurations), `qwen4_outproj_profile.json` / `.trace.json`.

## Open questions

- How much of the 6.3 ms/layer depth penalty survives the fused preparation's own pass, and what F4 is then worth. Answer with `gdn_isolated_bench.py --depth-sweep` on the wired model.
- Whether the two-level scan should live in `fwd_h` (materialise segment prefixes and let the `o` pass read them) or grow into a single fused pass that produces `h` and `o` together, as llama.cpp's register-resident state suggests.
- Whether `chunk_fwd_o` is worth a Gluon rewrite before or after the backward's three, which are larger prizes but harder. The ordering in the backward plan puts it last.
- What the fused preparation looks like in Gluon. Its kernels are at 1.15x of their traffic roofline with a plain elementwise structure, so there is less there than in the FLA kernels, but it is the same kind of rewrite.
- Whether the preparation should stop at the projection's output, which it does, or also own the projection and with it the Q8_1 quantization of its input.
