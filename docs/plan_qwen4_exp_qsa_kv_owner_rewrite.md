# Rewriting the QSA dK/dV owner: evidence, diagnosis, design and result

Status: implemented, tested and shipped. The Gluon kernel is `qwen4_exp_qsa_gluon.py`, and `qwen4_exp_qsa_attention.py` dispatches to it. The Triton owner is kept for reference and for the config sweeps, which select it with a `dkdv_override`. The dK/dV owner is 12-15% faster than the Triton one at every batch size and produces bit-identical output, so the accuracy gates, the padding contract and the deterministic split-and-reduce structure all carry over unchanged.

This document is the record of the whole effort. "What was measured" and "Diagnosis" say what limited the Triton kernel and why a bigger key tile, the one lever the traffic analysis offers, could not be taken inside Triton. "What was tried inside Triton" is the list of attempts with their scores. "The design for a rewrite" is the plan as it stood before implementation, with a note on where the built kernel deviates from it. "Building it" is the port itself: what the Gluon layout API required, the micro-tests that established its behaviour, the two bugs, and the negative result. "Results" and "What did not work" close it out.

## What was measured

All at batch 1, `BLOCK_N=16`, `BLOCK_M=32`, `num_warps=2`, `SPLIT=4`, 4 samples, medians, on the real `_qsa_dkdv_kernel` (22.99 ms) and on probe variants of it that differ in exactly one dimension (`~/tmp/test_no_unsloth/probe_qsa_kv_modes.py`).

| variant | time | what it removes | share of the kernel |
| --- | ---: | --- | ---: |
| full | 22.99 ms | - | - |
| free-loads (every iteration loads the same query tile, so Q/dO hit L1) | 7.67 ms | the streaming traffic of Q and dO | 66.6% |
| loads-only | 2.97 ms | all tile math (the loads get hoisted, so this is a floor, not a roofline) | 87.1% |
| no-elementwise (no exp2, no causal mask, no delta) | 25.01 ms | 2.2 ms of elementwise work, and it got *slower* | -8.8% |
| no-dots (row sums instead of `tl.dot`) | 37.50 ms | the MMA work, and it got much slower | -63.1% |
| transposed-acc (accumulators as `[D, BLOCK_N]`, transposed loads) | 30.41 ms | the two per-iteration `tl.trans` calls | -32.3% |

Removing work makes the kernel slower in two of the six variants: the schedule, not the arithmetic, is what is being spent. That is consistent with the compiled code. Instruction mix per loop body, counted from the `amdgcn` assembly:

| config | instructions | wmma | exp2 | ds_load_b128 | scratch (spills) | v_perm_b32 | regs / spills |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `bn16 bm32 w2` (shipped) | 2946 | 64 | 8 | 200 | 467 | 128 | 256 / 511 |
| `bn32 bm32 w4` | 3175 | 96 | 16 | 264 | 310 | 0 | 256 / 708 |
| `bn32 bm32 w8` | 2585 | 80 | 16 | 232 | 198 | 0 | 256 / 600 |
| `bn64 bm32 w8` | 4841 | 160 | 32 | 392 | 648 | 0 | 256 / 1263 |
| `bn128 bm16 w8` | 8796 | 288 | 64 | 816 | 1330 | 0 | 256 / 2531 |

The loop body is shared-memory traffic, spilled registers and layout conversions. The matrix math and the exponentials are a few percent of it. Every configuration in the matrix compiles to exactly 256 VGPRs, the hardware maximum, and then spills 400 to 2531 slots.

## Diagnosis

The kernel is DRAM-traffic bound on re-reading Q and dO. Each tile pair reads `BLOCK_M x 256` of Q and the same of dO, 32 KB in bf16, and there are `(S^2/2) / (BLOCK_M * BLOCK_N)` pairs per head. The traffic is

```
bytes = 2 * H * (S^2 / 2) * 256 * 2 / BLOCK_N = S^2 * H * 512 / BLOCK_N
```

increasingly independent of `BLOCK_M` and linear in `1 / BLOCK_N`. At batch 1, `BLOCK_N=16`: 3.17 GB. The measured 15.3 ms of streaming time over it is 207 GB/s, which is this device's practical DRAM rate, and the working set that would have to stay cached is 25 MB of Q plus dO against a small L2. So the only lever on two thirds of the kernel is a larger key tile, and the arithmetic says exactly what it is worth:

| `BLOCK_N` | tile pairs (batch 1) | Q+dO traffic | streaming time at 207 GB/s |
| ---: | ---: | ---: | ---: |
| 16 (shipped) | 49920 | 3.17 GB | 15.3 ms |
| 32 | 24960 | 1.59 GB | 7.7 ms |
| 64 | 12480 | 0.79 GB | 3.8 ms |
| 128 | 6240 | 0.40 GB | 1.9 ms |

Those numbers also give the arithmetic intensity, `2 * BLOCK_N` FLOP per byte of streaming, so at `BLOCK_N=16` the device's 207 GB/s supports only 6.6 TFLOP/s of the 208 GFLOP that the backward needs, which is why the traffic dominates every other term.

`BLOCK_N` cannot be raised because the live set does not fit in registers, and Triton's layout machinery makes it much worse than the algorithmic minimum. At `BLOCK_N=32`, `BLOCK_M=32`, 4 warps the minimum live set is about 320 VGPRs per lane against 256 available:

| live value | elements | VGPRs per lane (4 warps) |
| --- | ---: | ---: |
| dK `[32, 256]` fp32 | 8192 | 64 |
| dV `[32, 256]` fp32 | 8192 | 64 |
| K and V `[32, 256]` bf16, resident | 8192 | 64 |
| Q and dO `[32, 256]` bf16, streamed | 8192 | 64 |
| p, dP, dS `[32, 32]` fp32 | 3072 | 24 |
| lse, delta, offsets, mailboxes | - | 40 |

The two accumulators are irreducible in a fused kernel, and the 128 VGPRs they need are what a key tile wider than 16 rows cannot pay for. Triton then adds a second copy of every tile that crosses a layout boundary, which is why the measured spill count is far above what the table suggests and why all five larger-tile configurations land between 26 ms and 340 ms instead of the 8-15 ms the traffic model wants.

## What was tried inside Triton, and what it scored

| change | result |
| --- | --- |
| `BLOCK_N` 32, 64, 128 with `BLOCK_M` 16, 32, 64 and 4, 8 warps, `SPLIT` 4 and 8 | best is 26.04 ms (`bn32 bm64 w8 s8`), against 22.99 ms at `BLOCK_N=16. The rest are 37-340 ms, all spilling 600-2531 slots |
| `num_stages` 2 on the pipelined inner loop (`tl.range(..., num_stages=2)`) | 56.66 against 56.93 ms at `bn32 bm32 w4`. No effect |
| reloading K and V inside the inner loop so they need not stay in registers | 22.96 against 23.00 ms at `BLOCK_N=16`, and the compiler hoists the loads back out. No effect at `BLOCK_N=32` either |
| transposed accumulators `[D, BLOCK_N]` fed by transposed loads, so `p` and `dS` enter the MMA as the B operand in the layout they were computed in | 30.41 ms, 32% worse, and more spills |
| chunking the 256-wide Q/dO loads into 128-wide halves to halve the transient tiles | not tried: the arithmetic above shows it saves 32 of the 64 needed VGPRs per lane at `BLOCK_M=32`, so it cannot reach `BLOCK_N=32` on its own |
| split dK and dV into separate launches (halves the accumulator, allows a larger tile) | measured earlier at 35.3 ms against 32.0 ms: the second pass over Q and dO costs more than the traffic the larger tile saves |
| `maxnreg` | this Triton rejects the launch keyword |
| 8-bit dO | 1.6 GB less traffic, but e4m3 has three mantissa bits against a 0.3% gradient tolerance |

Rejected analytically, without building them: a query-owning dK/dV with atomics streams the same bytes (each pair still reads one 32 KB operand, only the direction changes) and gives up the deterministic one-owner-per-element accumulation that the audit identity gates need. Splitting the head dimension across programs doubles the streams unless L2 catches the second half, and dP needs a full-head-dimension reduction anyway.

## The design for a rewrite

Three changes together, none of which Triton can express. Where the built kernel deviates from this plan is noted in "Building it":
- Key and value tiles live in shared memory, not registers. That removes 64 VGPRs per lane at `BLOCK_N=32` (128 at 64) and makes the wmma operands LDS reads in whatever layout the instruction wants. Expected cost: LDS bandwidth, which the assembly above shows is already the largest single consumer and is far from saturated.
- `p` and `dS` reach the accumulation dots as the A operand without a register layout conversion and without a `v_perm_b32` pass. On RDNA3 the wmma B operand can be taken transposed, which is exactly the layout `p^T` and `dS^T` want in `dV += p^T dO` and `dK += dS^T Q`. Gluon has no register transpose, so in the built kernel this is one shared-memory store and one transposed read. The point is that it is the only round trip those tiles make, and that the streamed operands come out of shared memory directly in their operand layouts.
- Q and dO are streamed through shared memory with double buffering, so their liveness in registers is one buffer, not the whole tile, and the DRAM requests are issued early enough to cover the latency that the `s_waitcnt` density in the assembly suggests is being lost. This was not implemented. It is the main piece of the plan that is still open.

With 1-3, the register budget at `BLOCK_N=64`, `BLOCK_M=32`, 8 warps is dK and dV at 32 VGPRs per lane each (16384 words over 256 lanes), Q/dO as one shared-memory buffer, and K/V in shared memory: about 120-150 VGPRs, which leaves room for the elementwise path. The traffic model then predicts 3.8 ms of streaming instead of 15.3 ms, so a target of 8-10 ms against the current 23 ms is reasonable, and the whole backward would go from 31 ms to about 16-18 ms at batch 1. The negative result in "What did not work" is that the shared-memory budget, not the register budget, is what actually blocks this.

### Gluon mapping

Everything needed exists in `triton 3.8.0` as installed. The version number is 1 for RDNA3 (version 2 is RDNA4, as the frontend tests show):

| need | primitive |
| --- | --- |
| transposed wmma operand without a layout conversion | `triton.experimental.gluon.language.amd.AMDWMMALayout(version=1, transposed=True/False, warp_bases=..., reg_bases=..., instr_shape=[16, 16, 16])` |
| the MMA itself | `gl.amd.rdna3.wmma` |
| key/value resident on chip | `gl.allocate_shared_memory`, `SwizzledSharedLayout` or `PaddedSharedLayout` |
| streaming of Q/dO through shared memory | `gl.amd.buffer_load`/`buffer_store`, `gl.async_copy` plus `mbarrier`, or `gl.amd.warp_pipeline`/`warp_pipeline_stage` for warp specialization |
| explicit control over which layout each value is in | `gl.BlockedLayout`, `gl.DotOperandLayout`, `gl.convert_layout` (so the cost is visible in the IR rather than inserted silently) |
| synchronising a shared-memory store with a cross-layout read | `gl.barrier()`. The compiler does not insert these inside a Gluon kernel |

Tutorials worth following, in this tree: `python/tutorials/gluon/01-intro.py`, `02-layouts.py`, `03-async-copy.py`, `07-persistence.py`, `08-warp-specialization.py`, and the frontend tests `python/test/gluon/test_frontend.py` for `AMDWMMALayout` (around line 2898 and 2935) and `test_amd_rdna3_wmma` (line 3575).

### HIP fallback

If Gluon's RDNA3 wmma path had turned out to be incomplete for bf16 with fp32 accumulators, the same design is straightforward in HIP, which is what the `gfx1151_reference.md` facts are for: `__builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32` for the four dots, `__builtin_amdgcn_ds_*` for the shared-memory staging, `buffer_load_dwordx4` for the Q/dO streams (16-byte vector loads are what the contiguous 256-wide rows want), `s_waitcnt`/`sched_group_barrier` for the pipeline, and 15 waves per SIMD rather than the current 8 as the target. The reference document's warnings apply directly: watch for `v_perm_b32` with inline literals choking instruction fetch, and for `ds_load_b128` queue-full stalls when many waves issue wide LDS reads. The Gluon path proved complete, so this stays on the shelf.

## Building it

The kernel took a full owner's worth of code: the same grid, split, head loop and query-tile range as the Triton kernel, key and value staged in shared memory with `SwizzledSharedLayout(8, 1, 8, [1, 0])` and read as wmma B operands through `.permute([1, 0])`, the streamed query and dO loaded directly in the score dot's A-operand layout, and the computed `p` and `dS` stored to shared memory and read back transposed as the A operands of the accumulation dots. Accumulators are `gl.amd.AMDWMMALayout` tiles and the four dots are `gl.amd.rdna3.wmma` calls.

What the port needed from the API, none of it obvious from the tutorials:

Layouts cannot be passed as nested Python lists, because the frontend hashes constexpr arguments and lists are unhashable. They have to be constructed on the host and passed as `gl.constexpr` layout objects, which is what the CDNA5 examples do for exactly this reason. An `lse` or `delta` vector that will broadcast against the score tile must be loaded in a slice of the *score* layout, not in a slice of the operand layout the streamed loads use, or the broadcast fails with a layout mismatch.

A dot's A operand must be converted or loaded in `DotOperandLayout(0, parent, 16)` explicitly. The parent's warp bases then decide how the operand's K axis is covered. `num_warps` changes the warp bases that are legal, so the layout set is built per warp count.

The micro-tests in `~/tmp/test_no_unsloth/gluon_wmma_probe.py` settled the rest before the kernel was debugged, and they are the reason the port was tractable:

`gl.amd.rdna3.wmma` with explicit `AMDWMMALayout`s is exact against a torch matmul (rmse 0.000000) for both `transposed=True` and `transposed=False`, with the operands loaded straight from global memory into `DotOperandLayout` form.

A shared-memory round trip is faithful for blocked, MMA-accumulator and dot-operand store layouts alike, and `.permute([1, 0])` on a shared descriptor really does transpose the logical tile, in both directions.

The two paths the owner actually depends on - a `[BN, BM]` A operand read transposed out of a `[BM, BN]` shared tile, and a `[BM, 256]` B operand read out of a shared tile of the same shape - both work at the kernel's own shapes with the gradient accumulator's layout.

### The two bugs

The first was mine and mechanical, and it is why the closed-form test exists. The key/value offsets dropped the `start_n` term, so all 1024 programs loaded and stored the *first* key tile and raced on the same addresses. The symptom was that only key tile 0 held output and everything else was exactly zero, and the raw workspace dump showed row 0 written by key tiles 80 through 86. What localised it was a degenerate case with a closed form: setting `p = dS = 1` turns both accumulations into plain sums of the streamed operands, so the expected value of every key row is a suffix sum that can be computed in torch and compared row by row. That test is worth keeping as the first thing to run on any rewrite of this kernel.

The second is a property of the hardware path rather than of the code. The kernel needs two workgroup barriers per iteration, one after staging all four tiles and one immediately before the accumulation reads. Removing the second one made it 60% slower - 30.8 ms against 19.4 ms at the same configuration - and the effect reproduced in both directions with 9 samples and spreads under 0.2%. The barrier does not only synchronise shared memory. On this target it also fixes the schedule the compiler picks for the shared-memory reads.

## Results

The dK/dV owner alone, against the Triton kernel at the same shapes, medians of 9 samples:

| configuration | Triton | Gluon |
| --- | ---: | ---: |
| batch 1, the shipped tile `BLOCK_N=16 BLOCK_M=32` | 23.33 ms | 19.77 ms (+15.3%) |
| batch 4, same shape | 91.33 ms | 79.61 ms (+12.8%) |
| batch 16, same shape | 362.09 ms | 308.15 ms (+14.9%) |

The whole layer through the module, forward plus backward per layer:

| batch | Triton owner | Gluon owner | speedup against masked SDPA |
| ---: | ---: | ---: | ---: |
| 1 | 34.14 ms | 29.69 ms | 8.0x |
| 4 | 133.93 ms | 118.00 ms | 6.8x |
| 16 | 529.80 ms | 468.11 ms | 6.8x |

Gradient accuracy is unchanged, because the output is bit-identical to the Triton kernel at full length and with right padding: rounding to nearest in the shared-memory staging reproduces Triton's `.to(bfloat16)` exactly, and the accumulation order is the same. Right padding is handled natively through `key_end`, with the probabilities masked per element and the store masked per key row, so the Triton owner is kept only for reference and for the config sweeps. Two regression tests in `test_qwen4_exp_qsa_attention.py` pin the two owners together, at full length and with padding.

## What did not work

Larger key tiles, the prize the traffic model predicted, do not pay off in this design. `BLOCK_N=32` needs 32 KB for K and V plus the staging buffers, which overflows the 64 KiB limit at `BLOCK_M=32`. Letting the compiler convert the B operands instead of staging them (`B_FROM_CONVERT=1`) does fit, and even then the best `BLOCK_N=32` configuration runs at 36.8 ms against the 19.8 ms of `BLOCK_N=16`. The traffic saving is real but the wider tile loses more to its half-sized grid, to per-iteration staging traffic and to the conversion path than it gains. `BLOCK_N=64` does not fit at all, and the whole point of the design - reaching a key tile wide enough to cut the streaming severalfold - is blocked by shared-memory capacity rather than by registers.

Everything else in that sweep was measured too: `num_warps` 2 and 8 are worse than 4, splits 2 and 8 are worse than 4, and the compiler's conversion path (`conv1`) is 2x worse than explicit staging wherever both fit. The configuration that ships is `BLOCK_M=32`, `BLOCK_N=16`, `num_warps=4`, using the module's split of 4.

## What is not worth rewriting

The other two backward kernels are already small: the Delta kernel is a row-sum over dO and O, and the dQ kernel measures at 1-6 ms across the configs in the backward sweep, against 23 ms for the KV owner. The launch table, the padding contract (`key_end`), the split-and-reduce structure and the FP32 LSE/delta interface should be kept as they are, since a rewritten KV owner plugs into them unchanged.

## Reproducing the measurements

Triton-side diagnosis:

```
python ~/tmp/test_no_unsloth/probe_qsa_kv_modes.py --samples 4 --batch 1
python ~/tmp/test_no_unsloth/probe_qsa_kv_modes.py --matrix scan --samples 4 --batch 1
python ~/tmp/test_no_unsloth/probe_qsa_kv_modes.py --matrix isa --batch 1
```

Gluon kernel, checked against the Triton kernel and swept over configurations:

```
python ~/tmp/test_no_unsloth/gluon_qsa_dkdv.py --check --block-n 16 --block-m 32 --warps 4 --split 4
python ~/tmp/test_no_unsloth/gluon_qsa_dkdv.py --check --padded 137 --block-n 16 --block-m 32 --warps 4 --split 4
python ~/tmp/test_no_unsloth/gluon_qsa_dkdv.py --bench --samples 9 --batch 1 --block-n 16 --block-m 32 --warps 4 --split 4
python ~/tmp/test_no_unsloth/gluon_qsa_dkdv.py --sweep --batch 1
```

Gluon layout API micro-tests:

```
python ~/tmp/test_no_unsloth/gluon_wmma_probe.py
```

The mode probe builds its own inputs at the real shapes, times the shipped kernel and its variants in the same process, and reports register, spill and shared-memory counts per configuration together with an instruction-class histogram from the `amdgcn` assembly. The Gluon harness does the same for its kernel, including the padded case, and the micro-test file is four tiny kernels with known expected values.
