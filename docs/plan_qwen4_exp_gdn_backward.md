# Qwen4-Exp GatedDeltaNet Backward Plan

## Production contract

The target is the backward of `Qwen4ExpTextGatedDeltaNet`: 36 of the 48 layers, batch 1, sequence length 2048, hidden size 2560, 16 key heads, 48 value heads, head dimension 128 for both, conv width 4, BF16 activations with an FP32 recurrent state. As in the forward plan, only the shapes this repository trains have to work: `NT = 32` chunks of 64 exactly, no variable lengths, no tail chunk, `H/HV = 1/3` exactly, and every dimension a constexpr.

Four constraints shape the plan:
- Every backward follows a recomputed forward. All 48 layers are checkpointed, so the backward pass begins with a second forward, and the passes that FLA re-derives (`recompute_w_u_fwd`, `chunk_gated_delta_rule_fwd_h`) run inside this phase. Of the 56.87 ms/layer the backward costs today, about 4.5 ms is forward work repeated inside it: `recompute_w_u_fwd` at 3.27 ms for its one call here and `fwd_h` at 1.22 ms.
- The backward is 2.3x the forward at the deployed state (56.87 ms against 24.79 ms). That ratio, not the total, is what makes this the largest single optimisation target in the model.
- Determinism is required. The audit's step-to-step comparisons and the checkpoint replay depend on it, which rules out atomics and split-reduction schemes that do not fix an ownership order. Every kernel on this path is currently deterministic.
- The value-head convention stays llama.cpp's tiled order (`gdn_tiled_value_heads.py`), gated by `require_tiled_value_heads`.

Acceptance is the audit's gates (losses, clip norms, gradient gates, warm step time), not bitwise equality against the incumbent kernels: reordering accumulation inside a tolerance is expected and allowed.

## Current measurements

One layer at the training geometry, both deployed launchers installed, five warm iterations of `~/tmp/test_no_unsloth/gdn_current_measurements.py`: forward 24.785 ms, backward 56.87 ms per layer per step. The same harness, one session, before and after this round's two replacements: 64.9 ms of backward to 56.6 ms, so they are worth -8.3 ms/layer/step in total, 12.8% of the backward and 2.9% of the 10.14 s warm update.

### The backward's kernels, as they stand

| kernel | ms/layer/step | calls | what it was | status |
| --- | ---: | ---: | --- | --- |
| `chunk_bwd_kernel_dqkwg` | 8.27 | 1 | 10.94 at the tiles FLA's collapsed constant allows | B9 launcher, deployed |
| `prepare_wy_repr_bwd_kernel` | 6.97 | 1 | 7.87 at the table entry before the warp/stage screen | table entry, deployed |
| `recompute_w_u_fwd_kernel` | 6.55 | 2 | - | B2/F2 replacement `gdn_wu_recompute.py` wired |
| `causal_conv1d_channellast_bwd_kernel` | 5.98 | 1 | - | F1 fused preparation written, tested, not wired: it measured slower in the model |
| `_dhu_kernel` (B8) | 3.66 | 1 | 9.12 for FLA's kernel | deployed |
| `chunk_fwd_kernel_o` | 2.93 | 1 | 5.14 two rounds back | forward plan's F0, deployed |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 2.43 | 2 | - | the backward's own call is 1.22 ms of it |
| `chunk_bwd_kernel_dv_local` | 1.38 | 1 | - | B7: no change, the deployed entry wins |
| `l2norm_bwd_kernel` | 0.73 | 2 | - | F1 fused preparation written, tested, not wired: it measured slower in the model |
| `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 0.69 | 1 | - | forward side |

Two call counts here correct earlier notes in these plans: `recompute_w_u_fwd` runs twice a step (once in the forward, once inside the backward's recomputation) and `chunk_fwd_o` runs once, which changes what their replacements are worth: the W/U replacement is -2.94 ms/layer/step over its two calls, not -4.25 over three.

### Counters, at the deployed configurations

rocprofv3 over the one-layer harness with both launchers installed (`analyze_gdn_stalls.py`, one counter group per `--pmc`). The vendor GEMM is the scale reference: it is the only kernel here that spends its cycles issuing rather than waiting.

| kernel | L2 MB/call | GB/s | its tensors need | ratio | wait VMEM | wait LGKM | wait barrier | VALU starve |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `chunk_bwd_kernel_dqkwg` | 429.8 | 50 | 327.9 | 1.3x | 48.7% | 21.8% | 0.0% | 24.1% |
| `prepare_wy_repr_bwd_kernel` | 1423.1 | 203 | 164.0 | 8.7x | 85.3% | 7.7% | 0.0% | 23.3% |
| `_dhu_kernel` | 893.4 | 243 | 226.9 | 3.9x | 64.1% | 7.0% | 0.0% | 21.2% |
| `causal_conv1d_channellast_bwd_kernel` | 126.5 | 21 | 125.7 | 1.0x | 76.8% | 0.2% | 0.0% | 31.5% |
| hipBLASLt BF16 GEMM, for scale | 943.8 | 241 | (its own sizes) | - | 2.0% | 2.8% | 0.0% | 28.0% |

Three readings. `wy_repr_bwd` and what is left of `dhu` move several times the bytes their tensors account for, and the excess is register spill traffic: their assembly carries 861 and 335 scratch instructions respectively (`analyze_gdn_isa.py`). `dqkwg` is at 1.3x its necessary traffic with no spills at all, so its cost is latency and scaffolding rather than bytes. And the conv backward moves exactly what its tensors need but sustains only 21 GB/s, which is the signature of a kernel waiting on dependent loads rather than on the memory system - the case the fused preparation replaces outright.

| kernel | instructions | WMMA | FP32 FMA | LDS ops | scratch (spill) ops |
| --- | ---: | ---: | ---: | ---: | ---: |
| `_dhu_kernel` | 6261 | 64 | 96 | 645 | 335 |
| `prepare_wy_repr_bwd_kernel` | 12417 | 268 | 223 | 1943 | 861 |

`dhu`'s row is the fixed kernel: FLA's carried 1930 FP32 FMA against 48 WMMA, because the multiply by the FP32 decay promotes the BF16 query operand and the cast written to keep it narrow runs afterwards, so that dot lowers to FP32 FMA on the VALU. B8 below is that one cast. `dqkwg`'s LDS traffic halved with B9's wider tile - the share of its cycles waiting on LDS drops from 39.3% to 21.8% and its traffic from 622 MB to 430 MB - which is why the same kernel now has 0.0% of its cycles at its barrier where it used to spend 14.3% there.

### Rooflines, and how far the kernels are from them

Machine numbers from `~/ComfyUI-FeatherOps/docs/gfx1151_reference.md`: 59.4 TFLOPS of BF16 WMMA, 14.8 TFLOPS of FP32 VALU, 928 Gop/s of transcendental, 256 GB/s of DRAM (207 GB/s under load), 128 B/cycle of LDS per WGP, and 1536 VGPR per SIMD allocated in blocks of 24.

| kernel | us/call | GFLOP | MB needed | compute us | memory us | roofline us | measured us | margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `bwd_dqkwg` | 8266 | 6.44 | 327.9 | 108.5 | 1584.3 | 1584.3 | 8266 | 5.2x |
| `bwd_dhu` (B8 kernel) | 3662 | 6.44 | 226.9 | 108.5 | 1096.1 | 1096.1 | 3662 | 3.3x |
| `wy_repr_bwd` | 6965 | 3.22 | 164.0 | 54.2 | 792.1 | 792.1 | 6965 | 8.8x |
| `recompute_w_u` | 3272 | 3.22 | 105.3 | 54.2 | 508.5 | 508.5 | 3272 | 6.4x, replaced by F2 |
| `conv1d` backward | 5983 | 0.67 | 125.7 | 11.3 | 607.2 | 607.2 | 5983 | 9.9x, replaced by F1 |
| `dv_local` | 1382 | 3.22 | 100.7 | 54.2 | 486.3 | 486.3 | 1382 | 2.8x |

None is close to its compute roofline, and the three that matter are 3.3x to 8.8x their memory roofline. `dhu` is the one that moved: 8.19x to 3.3x, because the FP32 dot removed both the VALU work and most of the spill traffic, and the value-block retile halved what was left of it.

### Two structural measurements

The backward's serial-depth penalty is 11.5 ms/layer: the same work at `(B=16, S=128)` costs 45.7 ms and at `(B=32, S=64)` 45.2 ms, against 56.7 ms at `(B=1, S=2048)`. The projections do not depend on the chunk count, so that is the chunked recurrent structure itself, and it is the ceiling for any two-level-scan work - see B5, which the two walk kernels bound at roughly 5 ms of the 11.5.

There is no economy of scale to be had from more programs either. At batch 16, which is 16x the work, `bwd_dqkwg` takes 15.7x as long, `wy_repr_bwd` 15.9x, the B8 walk 16.2x and `recompute_w_u` 16.3x, so the machine is already saturated at batch 1 and the limit is inside each program. The conv backward is the exception and the wrong way: 25.4x for 16x the work.

### Audited

`audit_qwen4_exp_training_step.py --max-steps 3` passes its gates with B8 installed (`static_configuration.gdn_bwd_dhu.installed` is true in the report) and records a warm step of 9.495 s against the 10.14 s baseline: 2.677 s of forward, 6.677 s of backward, 0.031 s of clipping and 0.110 s of optimizer, so -0.645 s, 6.4%. Loss 3.4838, the second backward's clip norm 0.2832, zero missing, non-finite or zero gradient tensors, and 65.97 GiB allocated / 68.99 GiB reserved after the measured steps. That run used the 64-wide value block. The 32-wide block and B9's launcher that followed it are layer-verified only, and the next audit is their gate.

## Prior art

| source | what it establishes |
| --- | --- |
| `fla/ops/gated_delta_rule/chunk.py` (`chunk_gated_delta_rule_bwd`) | the current decomposition: `recompute_w_u_fwd` -> `fwd_h` recompute -> `bwd_dv_local` -> `bwd_dhu` -> `bwd_dqkwg` -> `prepare_wy_repr_bwd` -> reverse `chunk_local_cumsum` -> optional gate backward. The three backward-side passes that could merge into one walk are steps 4, 5 and 6. |
| `fla/ops/common/chunk_delta_h.py` (`chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64`) | the kernel B8 replaces: FP32 state-gradient tiles alive across a 32-chunk reverse walk, `BV` from the gated candidate list, `num_stages=1`, and the `_pre_process` helpers that already implement a split-sequence combine for context parallelism - the machinery a two-level scan needs, currently unused outside the CP path. |
| `fla/ops/common/chunk_o.py` (`chunk_bwd_dqkwg`, `chunk_bwd_dv_local`) | per-chunk kernels whose tiles come from `BKV_LIST`, with `CONST_TILING` deciding `NK`/`NV` in the launcher. The reason a tile change here is a launcher of our own (B9) and not a table edit. |
| `~/vllm/vllm/model_executor/layers/mamba/ops/gdn_chunk_cutedsl/` | vLLM's CuTe DSL GDN is forward-only (three kernels: `kernel_h`, `kernel_kkt_inv_uw`, `kernel_o`), so there is no serving-side backward to copy. What transfers is the fusion pattern and the state-pass shape. |
| `~/vllm/vllm/model_executor/layers/mamba/ops/ssd_checkpoint.py`, `ssd_state_passing.py` | the mamba2 path's chunked state handling: replay from checkpoints instead of storing every state, and a dedicated state-passing kernel for split sequences. Both are relevant to B3 (what to stash and what to replay) and B5 (how to split the scan). |
| `~/llama.cpp/ggml/src/ggml-cuda/gated_delta_net.cu` | the register-resident state with warp-per-column ownership and `DPP` cross-lane reductions, plus `keep_rs_t` snapshotting. Inference-only and token-serial, but it is the shape a fused chunk walk should keep for its state-gradient tile. |
| `plan_qwen4_exp_qsa_backward.md`, `plan_qwen4_exp_qsa_kv_owner_rewrite.md` | this project's attention backward, the closest measured precedent for a rewrite on this device: an owner-per-output design with no atomics, FP32 accumulation and BF16 stored gradients. `BLOCK_N=16`, `num_warps=2`, one stage won while 32/64/128 and 2-4 stages lost. The Gluon rewrite with `AMDWMMALayout(transposed=...)` and `gl.amd.rdna3.wmma` removed both the spills and the conversion traffic and was 12-15% faster, bit-identical. |
| `~/feather-attn/docs/featherattn_gfx1151_bwd_plan.md` | the older CK/CUDA attention backward on the same device: a KV owner and a Q owner, gradients stored directly as FP16, probability reconstructed in base 2, no atomics and no workspace. 1.21x (D64) and 2.99x (D128) over AITER's Triton backward. |
| `~/ComfyUI-FeatherOps/docs/gfx1151_reference.md` | the hardware budget: 40 CUs, wave32, 64 KB LDS per CU / 128 KB per WGP / 64 KB per workgroup, 1536 VGPR per SIMD with 16 waves maximum, `vgpr_allocated = ceil(vgpr/24)*24`, WMMA 16x16x16 at 32 cycles, 256 GB/s, and the rocprofv3 recipes used above. |

## Accepted

Three changes are deployed, each verified at the kernel and gradient level, and all three are now wired at their FLA call sites in both trainers and both audits. The wired state measures 9.31 s per update against the 10.14 s before this plan's work, -0.83 s or 8.2%, with the allocation unchanged at 65.97 GiB allocated and 68.99 GiB reserved. The fused preparation is the exception and is written, tested and disconnected: in the model it measured *slower* than the chain it replaces (the wired run's backward 7.90 s against 6.51 s without it, and 6.3 GiB more allocation), while in the isolated harness it is a clear win, so the round left it out rather than ship a regression and the forward plan's F1 carries the bisect.

### B8: the walk kernel's FP32 query operand (deployed)

`chunk_gated_delta_rule_bwd_dhu` hands `tl.dot` two FP32 operands, because the cast written to keep the decayed query narrow runs after the multiply that widens it, so its `q @ do` term lowers to FP32 FMA on the VALU. Casting the decayed operand back to BF16 first is the whole change. The recurrence is untouched. `gdn_bwd_dhu.py` carries it as a specialised drop-in, `install()` rebinds the name FLA's autograd function calls, and `test_gdn_bwd_dhu.py` compares both outputs against FLA's kernel on three shapes, checks determinism, and pins the fail-closed guards.

| configuration | dhu ms | vs FLA | gradient rmse |
| --- | ---: | ---: | ---: |
| FLA kernel (before) | 9.156 | 1.000 | - |
| faithful copy of FLA's kernel | 9.267 | 1.012 | 9.7e-08 |
| faithful copy + `disallow_acc_multi_buffer` | 9.181 | 1.003 | 3.4e-09 |
| BF16 query operand | 4.043 | 0.442 | 6.4e-04 |
| BF16 query operand, value block 32 (deployed) | 3.597 | 0.393 | 5.2e-04 |

Worth -6.02 ms/layer/step end to end (65.054 -> 59.032 in the paired harness). The third row is a useful negative: accumulator multi-buffering is not what spills here, so the register pressure was a symptom of the FP32 operand rather than a cause of anything. The value block does not change the recurrence - each value column of the state evolves on its own - but it halves the two FP32 state tiles, which is worth another 12% because the kernel still spills 335 instructions per loop body.

Risk, and the reason the gate is a tolerance: a BF16 operand is a precision change rather than a reordering, so the layer's own gradients at 5.2e-04 relative RMSE are the check, not equality.

### B9: `dqkwg`'s tiles, from a launcher of our own (deployed)

FLA sizes this kernel's key and value blocks from `CONST_TILING`, which is 32 here because gfx1151's shared-memory budget fails every `check_shared_mem` gate its candidate list is built from. The kernel itself is unchanged and its tiles are plain constexpr arguments, so `gdn_bwd_dqkwg.py` carries a launcher of its own and leaves FLA's kernel in place. `BK=128` removes a fourfold redundancy: one score tile per chunk and head instead of four, and one read of `v`, `do` and `dv` instead of four.

| tiling | dqkwg ms | vs deployed | gradient rmse |
| --- | ---: | ---: | ---: |
| `BK32 BV32 w4 s2` (FLA's launcher) | 10.833 | 1.000 | - |
| `BK128 BV32 w8 s2` (deployed) | 8.459 | 0.781 | 7.3e-04 |
| `BK128 BV32 w8 s3` | 8.509 | 0.786 | - |
| `BK128 BV64 w8 s1` | 8.612 | 0.790 | - |
| `BK128 BV32 w8 s1` | 8.959 | 0.827 | - |
| `BK128 BV16 w8 s2` | 9.305 | 0.859 | - |
| `BK128 BV64 w8 s2` | 9.703 | 0.890 | - |
| `BK128 BV128 w8 s2` | 10.036 | 0.916 | - |
| `BK64 BV64 w4 s2` | 10.244 | 0.935 | - |
| `BK128 BV32 w16 s2` | 11.546 | 1.066 | - |
| `BK128 BV64 w4 s2` | 14.284 | 1.311 | - |

Worth -2.53 ms/layer/step (10.905 -> 8.373 in the same harness, 0.768x). The widened tile changes the accumulation order of `dq` and `dk`, so the gate is the layer's gradients at 7.3e-04. With `NK=1` the kernel's partial-gradient axis is a single slice, so the workspace the reference allocates, writes and then sums over becomes the output tensor itself.

### B0 and the `wy_repr_bwd` entry: the tuning table's backward entries (deployed)

| change | effect |
| --- | --- |
| `chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64` -> `BV=64, num_warps=8, num_stages=2` | 19.054 -> 9.323 ms/call, -9.73 ms/layer/step. Inert while B8's module is installed. It is the fallback if that module is removed |
| `prepare_wy_repr_bwd_kernel` -> `num_warps=2, num_stages=2` | 7.872 -> 6.759 ms/call, -1.11 ms/layer/step |
| `chunk_bwd_kernel_dv_local` -> no change (B7) | the deployed entry beats every candidate screened |

`gdn_config_check.py` verified all three at 8.5e-07 worst relative L2 over the ten parameter gradients, nine of them exactly zero.

### Written, tested, not wired

| change | effect |
| --- | --- |
| `gdn_wu_recompute.py` (F2, serves B2's call as well as the forward's) | 0.551x of FLA's kernel, so 3.27 -> 1.80 ms/call, -2.94 ms/layer/step over its two calls. Wired |
| `gdn_fused_prep.py` (F1, owns the conv and `l2norm` backward that B6 describes) | the model's chain is 12.618 ms/layer/step against the fused op's 3.004 ms, so -9.61 ms/layer/step, 3.4% of the update. Of the fused op's own total, 0.679 ms is forward and 1.646 ms backward |

Together they are 12.55 ms/layer/step, 4.4% of the update, and they wait on the wiring round the forward plan describes rather than on anything here.

## Rejected and closed

Everything here was measured or reasoned to a decision, and the reason is recorded so the next round does not repeat it.

### `dhu`: the whole configuration space around the deployed point

Measured with interleaved samples against the entry that was deployed, so the comparison is not against FLA's fallback. The two screens ran on different days and each measured its own deployed baseline, 9.246 ms in the first and 9.329 ms in the second, which is why the ratios below use the baseline of their own run:

| config | kernel ms | against the deployed entry |
| --- | ---: | ---: |
| BV64 w8 s2 (deployed entry, first screen) | 9.246 | 1.000x |
| BV16 w4 s1 | 9.648 | 0.509x at the old baseline, 1.043x here |
| BV32 w4 s2 | 11.314 | 1.224x |
| BV32 w8 s2 | 11.607 | 1.255x |
| BV32 w8 s1 | 16.357 | 1.768x |
| BV64 w4 s2 | 18.100 | 1.957x |
| BV32 w4 s1 (the previous entry) | 18.956 | 2.049x |
| BV128 w8 s1 | 26.939 | 2.912x |
| BV64 w4 s1 | 28.363 | 3.066x |
| BV32 w2 s1 | 28.550 | 3.086x |
| BV64 w16 s2 | 11.885 | 1.274x |
| BV32 w16 s2 | 9.684 | 1.038x |
| BV16 w8 s2 | 13.977 | 1.498x |
| BV64 w16 s1 | 18.703 | 2.005x |
| BV32 w4 s3, BV64 w8 s3, BV32 w8 s3 and every other 3-stage point | does not compile: 114,688 B of LDS against the 65,536 B limit | |

The deployed entry is the optimum of the space. Deeper pipelining is unavailable for a structural reason rather than a tuning one: Triton stages dot operands through shared memory, so the stage count and the tile size trade against each other and both are at their limit.

### The knobs that looked obvious and are not

| knob | kernel | measurement | why it closed |
| --- | --- | ---: | --- |
| `disallow_acc_multi_buffer` | `bwd_dhu` | 1.003x | accumulator multi-buffering is not the spill source. The FP32 operand was |
| `disallow_acc_multi_buffer` | `prepare_wy_repr_bwd` | 0.999x | same |
| dropping `tl.debug_barrier()` | `prepare_wy_repr_bwd` | 0.996x | the barrier is not what costs |
| dropping `tl.debug_barrier()` | `bwd_dqkwg` | 0.989x | worth 1.1%, and its 14.3% barrier stall in the old profile was overlapped with other stalls. The 0.0% barrier reading after B9 is a tiling change, not this edit |
| `loop_unroll_factor=2` | `prepare_wy_repr_bwd` | 1.764x | much worse |
| one 128-row key block instead of two 64-row ones | the B8 walk | 1.027x | worse. The split helps the allocator |
| `maxnreg` | any | rejected by this Triton | - |
| re-enabling FLA's own device gates without a table entry | any | no effect | the table pins the config of every kernel it covers |

### Tile and warp questions that the launcher answers

- B1's original route, patching `check_shared_mem` inside `fla.ops.common.chunk_o` so `CONST_TILING` becomes 128, measured 8% for `dqkwg` - but it also moves `dv_local`, needs new table entries for both widened cache keys, and leaves the warp and stage count at whatever the autotuner picked. B9's launcher, with the same widening plus 8 warps and two stages, measures 22%. B1 is closed as superseded.
- B2's config question is closed with no win: the paired screen in the forward plan found `recompute_w_u_fwd` already at its best table entry (`w2, s3`. `w2, s2` is 1.12x slower) and `fwd_h` already at its best (`BV=32, w4, s1`. The `BV=64, w8, s2` this plan first proposed is 1.18x slower against it). What is left at those call sites is F2's replacement, not a table entry.
- Every `num_warps`/`num_stages` variant for `dqkwg` and `prepare_wy_repr_bwd` at their own tiles: 1.000x or worse, so their warp/stage space is exhausted.
- `dqkwg`'s other wide-tile points lost to the deployed one: `BK128 BV128 w8 s2` 0.916x against the old launcher but 1.12x against the new one, `BK128 BV32 w16 s2` 1.066x, `BK128 BV64 w4 s2` 1.311x.

### Design routes that were measured and are worse

- Splitting a KV owner into two kernels to halve its accumulators (the QSA record): 35.3 against 32.0 ms, the extra score and `dP` dots cost more than the re-reads they save. The same arithmetic applies to splitting `dqkwg`.
- Unrolling an inner loop by two (the QSA record): 40.5 against 30.9 ms.
- Splitting the W/U recomputation's two matmuls across programs: worse at every point (1.779 ms at best against 1.760 ms unsplit). Halving the live set does not pay for loading `A` twice.
- Atomics and split reductions with a non-deterministic combine: excluded by the determinism contract, as they were for the attention backward.

## What remains

### The wiring round: 12.55 ms/layer/step, 4.4% of the update

F1's fused preparation (-9.61) and F2's W/U replacement (-2.94) are written, tested against the chain the model runs, and waiting to enter the step together with the forward plan's kernels. Nothing in this plan blocks them, and B8/B9 are installed the same way (an `install()` next to the tuning table).

### The Gluon round: the layout work Triton cannot express

Three of the kernels below are at 3.9x to 8.8x their memory rooflines for reasons a better configuration cannot reach, and the fourth is the fixed walk, whose remaining spills are 3.9x its own tensors.

| kernel | now ms/layer/step | target | prize ms/layer/step | prize % of the step |
| --- | ---: | ---: | ---: | ---: |
| `wy_repr_bwd` (B4) | 6.97 | 1.98 | 4.99 | 1.8% |
| `bwd_dqkwg` (B4, retiled by B9) | 8.27 | 3.96 | 4.31 | 1.5% |
| `chunk_fwd_o` (a forward item) | 2.93 | 1.83 | 1.10 | 0.4% |
| `bwd_dhu` (B8's kernel) | 3.66 | 2.74 | 0.92 | 0.3% |
| total | 21.83 | 10.51 | 11.32 | 4.0% |

The target column is 2.5x each kernel's memory roofline, the same convention as the round that built this table, and the `now` column is today's measurement per layer per step, so `chunk_fwd_o` counts its single call. The percentage is of the 10.14 s warm update across all 36 recurrent layers.

`wy_repr_bwd` first, reversing an earlier order: it is the worst spill case left (861 scratch instructions, 8.7x its necessary traffic) on top of the LDS work its computed-tile transposes cost, so it is the one where explicit layouts have the most to remove. `bwd_dqkwg` second: no spills and 1.3x its traffic, so its win has to come from the scaffolding around the dots (21.8% of its cycles still wait on LDS, 48.7% on memory) and from registers the allocator spends on four live accumulators. `chunk_fwd_o` third is the forward plan's. It transposes nothing and its cost is the two accumulators, so it is a register-and-schedule problem. `bwd_dhu` last: the layout work would remove its remaining spills, but its roofline gap is now the smallest of the four.

B4's original framing - one fused kernel walking the chunks backwards for all three of `dhu`, `dqkwg` and `wy_repr_bwd` - should be re-measured rather than assumed: the three are 18.9 ms/layer now instead of 27.5, so the ceiling shrank with them, and the dh round trip it removes (50 MB written and read) is a smaller share of a smaller total. Treat B4 as "the Gluon round applied to these kernels, with the fusion as one option inside it".

### B3: stop re-deriving `W`, `U` and the states

The forward computes `A`, `W` and `U` and discards `W` and `U`, and the backward recomputes all three. Stashing `W`, `U` and `A` (about 46 MB/layer, 1.7 GB for the model) removes the backward's `recompute_w_u_fwd` call, and stashing the per-chunk states removes the backward's `fwd_h` call as well. The implementation lives in FLA's `ChunkGatedDeltaRuleFunction`: extra `ctx.save_for_backward` tensors, or extra outputs with the non-reentrant checkpoint path. It overlaps F2 (both remove the same call), so the two are alternatives - F2 is measured, B3 is not, and B3 additionally removes the `fwd_h` call at 1.22 ms. Risk: memory, and the correctness of the stash under `torch.utils.checkpoint(use_reentrant=False)`, which is the path the audit exercises.

### B5: two-level scan for the state gradient, if it is still worth it

The measured depth penalty is 11.5 ms/layer, and the two walk kernels are roughly 5 ms of it (`dhu` 3.66 and `fwd_h`'s 2.43 for two calls, of which the backward's is 1.22). So a scan that made both parallel would recover at most 2-3 ms/layer, 0.7-1.1%, and that is before the extra state traffic and the new accumulation order. FLA already contains the split-sequence machinery for context parallelism (`chunk_gated_delta_rule_bwd_dhu_pre_process`, and `compress_h0`/`expand_h0` on the forward side), so the ingredients exist - but the shape of the combine is decided by the forward plan's F4: the chunk recurrence is `h_out = (decay * I - k^T w) @ h_in + k^T u`, so a segment's transition is a full `[K, K]` operator and composing one costs `K^3` per chunk-head, not a scalar prefix product. Re-measure the residual depth penalty after the Gluon round before writing it. If the Gluon work shortens the walk as a side effect, this item may close itself.

### The audit for B9 and the 32-wide value block

Their combined -2.9 ms/layer/step is layer-verified (gradients at 7.3e-04 and 5.2e-04) and awaits the next `audit_qwen4_exp_training_step.py --max-steps 3` run, which is also where the 64-wide-to-32-wide value block gets its model-level check.

## Correctness gates

- Kernel-level: all ten parameter gradients from `gdn_config_check.py` at relative L2 <= 1e-5 against the incumbent. For a kernel replacement, both of its outputs against FLA's kernel on three shapes, determinism, and fail-closed guards - the pattern `test_gdn_bwd_dhu.py` and `test_gdn_bwd_dqkwg.py` implement.
- A tile or operand change is a tolerance gate, not an equality gate: the recorded worst relative RMSEs are 5.2e-04 (walk, BF16 query operand), 7.3e-04 (B9's tiles) and 8.5e-07 (the table entries, which keep the accumulation order).
- Layer-level: the module's input gradient and every parameter gradient against the untouched module, plus `test_gdn_tiled_value_heads.py` for the value-head conventions.
- Model-level: `audit_qwen4_exp_training_step.py` must reach `GATE PASS`, with `second_backward` reporting zero missing, non-finite or zero gradients, and the warm update at or below the 9.495 s the B8-era run recorded. The B8-era values for reference: loss 3.4838, second-backward clip norm 0.2832.
- Determinism: two runs of the isolated layer, and two runs of the audit, must agree bitwise on the gradients.
- Memory: the B3 stash must be visible in the audit's `memory_after_measured_steps` and must not push the peak past the recorded 65.97 GiB allocated / 68.99 GiB reserved.

## Measurement protocol

- One layer at the training geometry: `python ~/tmp/test_no_unsloth/gdn_current_measurements.py`, which installs the deployed launchers and prints the phase totals, the full kernel table, the depth sweep and the batch-16 scale test in one session. `gdn_isolated_bench.py --repeats 5 [--kernel-table]` is the same harness without the launchers.
- Kernel-level attribution through `torch.profiler` in that harness, and through `rocprofv3 --kernel-trace --pmc ...` when a resource question decides the design.
- What a kernel is actually doing, in three steps: `analyze_gdn_stalls.py <profile-root>` for the stall and traffic counters (`rocprofv3` accepts one counter group per `--pmc` and writes `pass_N/` directories to merge), `analyze_gdn_isa.py <kernel>.amdgcn` for the opcode histogram out of Triton's cache, and `analyze_gdn_dots.py <cache>` for the FP32-FMA-to-WMMA ratio that found B8. A kernel with many FP32 FMA and few WMMA has a dot on the VALU.
- Direct A/B of two implementations with the paired protocol from `~/torch-ggml-ops/bench/`: even sample counts, alternating order, blocks, medians. Interleave candidates inside one session, because a cross-session comparison on this APU drifts by several percent. Every point should be verified against its reference through the layer's gradients on a *fixed* input first: two random draws differ by more than any kernel change, and a comparison that redraws the input measures the seed.
- End-to-end through `~/test_no_unsloth/train_qwen4_exp.py` and `audit_qwen4_exp_training_step.py --max-steps 3 --profile-output ...`, with the attribution reconstructed by `~/tmp/test_no_unsloth/analyze_qwen4_gdn_kernels.py` and `analyze_qwen4_gdn_layer.py`.
- Reproducing the tables above:

```bash
python ~/tmp/test_no_unsloth/gdn_current_measurements.py --iterations 5     # totals, kernel table, depth sweep, scale test
python ~/tmp/test_no_unsloth/dhu_retile_probe.py --samples 3                # B8's value-block screen
python ~/tmp/test_no_unsloth/dqkwg_tile_probe.py --samples 3                # B9's tile screen
python ~/tmp/test_no_unsloth/dhu_stage_probe.py --samples 3                 # the walk's warp/stage space
python ~/tmp/test_no_unsloth/dhu_loop_probe.py                              # acc multi-buffer, unroll, flatten
python ~/tmp/test_no_unsloth/wy_repr_acc_probe.py --samples 3               # the same knobs on wy_repr_bwd
python ~/tmp/test_no_unsloth/dqkwg_barrier_probe.py --samples 3             # the barrier, from FLA's own source
python ~/tmp/test_no_unsloth/dhu_singleblock_probe.py --samples 2           # one key block instead of two
python ~/tmp/test_no_unsloth/gdn_config_probe.py --kernel dhu --table       # the table entries, against the deployed ones
python ~/tmp/test_no_unsloth/check_gdn_wu.py --repeats 15                   # the W/U replacement
```

## Open questions

- How much of the 11.5 ms/layer depth penalty is left after the Gluon round, and therefore whether B5 is worth writing at all. The two walk kernels bound it at roughly 5 ms.
- Whether B4 should be a fused walk for all three kernels or the same layout work applied kernel by kernel, which depends on how much register pressure the `h` and `A` tiles add to the state-gradient tile under Gluon's explicit allocation. The three kernels are 18.9 ms/layer now, against 27.5 when the fusion was first estimated.
- Whether B3's stash should live for all 36 layers or only for the layers the backward needs at a given time, which changes the memory/time trade from 1.7 GB to a smaller sliding set.
- Whether the remaining spill traffic in the B8 walk (893 MB/call, 3.9x its tensors) is worth a further pass, since its whole roofline gap is now 0.92 ms/layer.
- Whether `dqkwg`'s 21.8% of cycles waiting on LDS and 48.7% on memory respond to explicit layouts, or whether the kernel's four live accumulators are the binding constraint and the answer is a different decomposition.
