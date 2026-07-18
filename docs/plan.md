# Qwen3.6-35B-A3B LoRA training under 16 GiB

## Result

The target has been demonstrated: Qwen3.6-35B-A3B rank-4 LoRA can complete full-model batch-1, sequence-2048 training updates with both live PyTorch allocation and allocator reservation below 16 GiB.

Production configuration:
- checkpoint: `~/models/qwen3.6/Qwen3.6-35B-A3B-APEX-I-Mini.gguf`.
- architecture: text-only `Qwen3_5MoeForCausalLM`.
- tokenizer: `Qwen/Qwen3.5-35B-A3B`.
- complete model on `cuda:0`.
- BF16 compute and BF16 LoRA adapters.
- batch size 1, sequence length 2048.
- rank-4 ordinary and routed-expert LoRA.
- non-reentrant gradient checkpointing on all 40 decoder layers.
- bitsandbytes `adamw_8bit` for adapter parameters only.
- top-8 routing active, with router-logit retention and the auxiliary balancing objective disabled.

Current sequence-2048 two-pass measurement:

| Phase | Peak allocated | Peak reserved | Time |
| --- | ---: | ---: | ---: |
| Model load | 13.323 GiB | 13.465 GiB | 11.36 s |
| Adapter injection | 13.721 GiB | 13.738 GiB | 0.51 s |
| First forward | 14.527 GiB | 14.584 GiB | 6.33 s |
| First backward | 14.960 GiB | 15.020 GiB | 8.85 s |
| AdamW8bit step | 14.792 GiB | 15.146 GiB | 0.09 s |
| Second forward | 15.050 GiB | 15.209 GiB | 3.32 s |
| Second backward | 15.399 GiB | 15.463 GiB | 8.62 s |

The steady-state maximum is now the second backward rather than an avoidable forward materialization. A direct one-step `SFTTrainer.train()` memory-validation run measured 14.928 GiB allocated, 15.150 GiB reserved, and 15.24 seconds. That one-step run includes cold compilation, autotuning, and optimizer-state initialization and is not the steady-state runtime result. The warmed profile is recorded in section 8.

The full audit produced all 660 finite trainable gradients, no frozen or packed-weight gradients, updates to all 330 LoRA-B tensors on the first step, and finite nonzero A/B gradients after the first update. Both packed-loss forwards returned `logits=None` and `aux_loss=None`.

Primary report: `/tmp/qwen36_packed_q8_lm_head_final_report.json`.

## Fixed training contract

The implementation is intentionally narrow. Future optimization must preserve these rules:
- GGUF is the sole base-weight representation.
- bitsandbytes is used only by `adamw_8bit`.
- Packed base parameters remain frozen and never receive gradients.
- The complete model remains on `cuda:0`. Arbitrary offload, sharding, and expert parallelism are unsupported.
- Serialized adapters retain ordinary PEFT names and rank-4 shapes.
- LoRA-A always consumes the original unquantized BF16 activation.
- Packed merge/unmerge and persistent packed-base `save_pretrained` remain rejected.
- Mixed-adapter expert batches, DoRA, aLoRA, LoRA bias, and unsupported PEFT modes fail explicitly.
- Gradient checkpointing remains non-reentrant and is verified on all 40 layers after trainer construction.
- Batch size remains 1 and sequence length remains 2048 for the validated memory claim.
- Router-logit retention and the router auxiliary loss remain disabled. Top-8 dispatch remains active.
- The real project dataset is not scanned, aggregated, regenerated, or rewritten without explicit approval.
- Native operators remain gfx1151-specific, asynchronous on the current Torch stream, and fail rather than inserting hidden operand copies.

## Completed optimizations

### Persistent GGUF model residency

The checkpoint remains compressed after loading:
- 733 checkpoint tensors.
- 34,660,610,688 logical parameters.
- 14,216,723,456 packed payload bytes.
- 351 `GGUFLinear` modules.
- 40 `GGUFExperts` modules.
- 432 frozen `GGUFQuantizedTensor` parameters.
- approximately 13.282 GiB live allocation immediately after load.

Reusable Transformers support provides:
- `GGUFLinear.materialize_logical_weight()` as a compatibility boundary for consumers that genuinely need a canonical floating matrix.
- Qwen3.5 recurrent input/output layout handling.
- capability-validated private expert backend names for specialized `GGUFExperts` execution.

The production ordinary, routed-expert, and LM-head paths no longer materialize logical base matrices. The remaining users of logical materialization are the 90 recurrent-layout projections that do not yet have permutation-aware MMQ kernels and explicit compatibility/reference paths.

### Ordinary packed LoRA

`fast_lora.py` installs PEFT-native wrappers without patching installed PEFT sources.

Current model composition:
- 250 ordinary LoRA wrappers.
- 160 ordinary packed projections using native MMQ in forward and backward.
- 90 GatedDeltaNet/recurrent-layout projections using the generic GGUF compatibility path.
- 250 ordinary LoRA-A and LoRA-B factor pairs included in normal PEFT serialization.

For the 160 native projections:
- the frozen BF16 input is dynamically quantized to Q8_1.
- `torch_ggml_ops::mmq` multiplies it by the authoritative packed GGUF weight.
- LoRA-A runs from the original BF16 input.
- LoRA-B and residual accumulation use framework BF16 GEMM/addition.
- backward calls `torch_ggml_ops::mmq_grad_input` for the frozen logical base Jacobian.
- only adapter factors receive parameter gradients.

The native backward removes logical-weight allocation at the cost of lower isolated GEMM throughput. That trade is intentional: complete-layer and full-model execution benefit from the substantially lower live allocation.

### Routed-expert packed LoRA

`fast_moe_lora.py` wraps each complete `GGUFExperts` module and preserves four rank-4 BF16 factor families per adapter:
- combined gate/up A.
- combined gate/up B.
- down A.
- down B.

The project-private expert backend uses Transformers routing and reduction while replacing the frozen projection work:
- expert-sorted routed rows and cumulative offsets remain on the GPU.
- gate/up uses `grouped_mmq_pair` when both packed tensors have matching geometry and quantization type.
- otherwise gate and up use independent `grouped_mmq` calls so their GGUF metadata remains authoritative.
- down uses `grouped_mmq`.
- paired gate/up backward accumulates both frozen Jacobians in one FP32 accumulator and emits one BF16 route-gradient tensor.
- down backward uses `grouped_mmq_grad_input`.
- LoRA execution and LoRA input gradients use AITER GMM.
- factor gradients use AITER PTGMM.
- no full expert delta, effective trainable expert matrix, selected logical expert copy, or packed gradient is created.

Routing, SiLU, gate/up multiplication, route weighting, inverse permutation, reduction, and LoRA residual accumulation remain framework-level operations. Native fusion stops at the projection boundary.

A matched layer-10 sequence-2048 benchmark measured 93.38 ms for packed forward plus backward versus 195.10 ms for selective materialization plus AITER, a 52.1% complete-layer speedup. Peak allocation and reservation were also more than 1 GiB lower in the packed variant.

### Native gfx1151 GGUF operators

`~/torch-ggml-ops` provides stable-ABI operators for:
- dense MMQ forward.
- dense packed input gradients.
- grouped MMQ forward.
- grouped packed input gradients.
- paired grouped gate/up forward.
- paired grouped gate/up input gradients.

Supported GGUF formats:
- `IQ2_S`.
- `Q3_K`.
- `Q4_K`.
- `Q5_K`.
- `Q6_K`.

Important implementation properties:
- BF16 inputs and outputs.
- Q8_1 activation workspace only in forward.
- BF16 WMMA with FP32 accumulation for frozen-base input gradients.
- no quantization of backward cotangents.
- no logical BF16/F32 weight allocation.
- no packed transpose or lossy repack.
- no CPU route descriptor or route-metadata synchronization.
- exact packed byte-count, dtype, device, alignment, contiguity, and zero-storage-offset validation.
- FakeTensor/meta registration, registered autograd, `torch.library.opcheck`, and `torch.compile` composition.
- explicit higher-order-gradient rejection.

The dense Q6_K backward decoder broadcasts one `d * scale` value across each 16-column WMMA tile. Wider-N and eight-wave schedules were measured and removed because they regressed the real LM-head geometry. The final package has 38 passing tests.

### Packed LM-head loss

The Q6_K language-model head has logical shape `[248320, 2048]`:
- packed payload: 417,177,600 bytes, approximately 0.389 GiB.
- logical BF16 matrix avoided: approximately 0.947 GiB.

`gguf_liger_loss.py` now owns a project-local packed causal-language-model loss:
- flatten and causally shift labels.
- process 256 hidden rows per chunk.
- explicitly clone the at-most 1 MiB BF16 slice because native MMQ requires zero storage offset.
- call `mmq` to produce a 121.25 MiB BF16 logits chunk from Q8_1 activations.
- run Liger's cross-entropy Triton primitive in place so logits become `dLogits`.
- call `mmq_grad_input` against the packed Q6_K head.
- save only the complete BF16 `dHidden` for model backward.

The function preserves:
- causal shifting.
- `ignore_index=-100`.
- mean/sum reduction.
- `num_items_in_batch` scaling.
- token accuracy.
- optional predicted tokens.
- `outputs.logits=None` during fused training.
- a frozen, gradient-free LM head.

It explicitly rejects unsupported geometry, trainable or biased heads, layout permutations, class weighting, token scaling, label smoothing, z-loss, LSE square scaling, logit softcapping, and higher-order differentiation.

Numerical comparison with the logical-BF16 fused reference:
- isolated hidden-gradient cosine: `0.99999988`.
- isolated hidden-gradient relative L2 error: `0.0714%`.
- full-model hidden-gradient cosine: `0.99999779`.
- full-model all-adapter gradient cosine: `0.99998622`.
- full-model all-adapter relative L2 error: `0.527%`.
- paired first AdamW8bit update cosine: `0.997115`.
- complete post-update adapter-state relative L2 difference: `0.0863%`.

The Q8_1 activation forward is therefore retained. Backward remains the logical packed-weight Jacobian rather than differentiation through rounding.

A production-geometry 2,048-row packed-loss benchmark measured:

| Chunk rows | Median core loss time | Peak allocation above resident inputs |
| ---: | ---: | ---: |
| 64 | 312.690 ms | 69.03 MiB |
| 256 | 229.958 ms | 253.57 MiB |

The 256-row schedule is 26.5% faster in the complete MMQ-forward, in-place cross-entropy, and MMQ-backward loop. Its additional approximately 184.5 MiB peak allocation is accepted for production.

### Router auxiliary-memory removal

Top-8 routing is unchanged, but retention of all layer router logits and the generic load-balancing objective are disabled:

- `model.config.output_router_logits = False`.
- `model.config.router_aux_loss_coef = 0.0`.
- `SFTConfig(router_aux_loss_coef=0.0)`.

All three settings are required because current TRL copies its coefficient back into the model configuration. Runtime checks reject a trainer that re-enables router-logit collection or a nonzero auxiliary coefficient.

This removes the large dense one-hot/expanded router auxiliary tensors while preserving actual expert dispatch.

### Trainer, data, attention, and recurrent execution

The trainer also includes:
- `get_peft_model(..., autocast_adapter_dtype=False)` so adapters remain BF16.
- explicit non-reentrant checkpoint enablement after `SFTTrainer` construction.
- verification that all 40 layers are checkpointed.
- `packing=False` and `padding_free=False`.
- compact fixed-length dataset rows containing `input_ids` and `num_tokens`.
- collator reconstruction of attention masks and `-100` labels.
- deterministic guarded Flash Attention 2 choices for the validated geometry.
- 14 exact FLA autotuner preloads across 13 autotuners.
- a project-local compiled GGUF dequantization fallback for operations that do not yet have layout-correct native MMQ, chiefly the 90 recurrent projections.
- Liger RMSNorm only where independently useful.

### Warmed full-step runtime profile

A production-geometry profile was collected after the `torch-ggml-ops` optimizations:
- batch size 1 and sequence length 2048.
- BF16 compute and rank-4 LoRA.
- non-reentrant checkpointing on all 40 decoder layers.
- one complete warm-up AdamW8bit update followed by one Kineto CPU+GPU traced update.
- synchronized wall-clock boundaries around forward, backward, gradient clipping, and the optimizer.
- correlated GPU kernel-duration sums for module and kernel attribution.

The warm-up update took 13.497 seconds. The traced steady-state update measured:

| Phase | Wall time | Step share |
| --- | ---: | ---: |
| Forward | 2.164 s | 26.32% |
| Backward | 5.979 s | 72.70% |
| Gradient clipping | 10.3 ms | 0.13% |
| AdamW8bit step | 46.7 ms | 0.57% |
| Other loop overhead | 24.2 ms | 0.29% |
| Total | 8.224 s | 100% |

The traced times include profiler overhead and are a single warmed sample rather than a benchmark distribution. They are nevertheless suitable for relative attribution because all requested kernel families were correlated to their launching operations.

Forward decoder-layer GPU work totaled 1.457 seconds:

| Module | GPU kernel time | Calls | Average per call |
| --- | ---: | ---: | ---: |
| MoE block | 696.7 ms | 40 | 17.42 ms |
| GatedDeltaNet | 644.7 ms | 30 | 21.49 ms |
| Full attention | 109.9 ms | 10 | 10.99 ms |

The routed experts account for 628.6 ms of the 696.7 ms MoE total, while the shared-expert MLP accounts for 40.2 ms. GatedDeltaNet forward contains 301.2 ms of FLA/recurrent kernels and 255.3 ms of GEMM. The complete Flash Attention forward kernels take only 16.9 ms. Approximately 235 ms outside the decoder layers is the packed LM-head MMQ, packed input gradient, and in-place cross-entropy path.

Backward decoder-layer GPU work totaled 5.299 seconds. Non-reentrant checkpoint recomputation accounts for 1.446 seconds, or 27.3%, while actual autograd work accounts for 3.853 seconds. The exclusive decomposition is:

| Module side | Actual backward | Checkpoint recompute | Inclusive backward phase |
| --- | ---: | ---: | ---: |
| MoE | 1.719 s | 686.8 ms | 2.406 s |
| GatedDeltaNet side | 1.676 s | 647.4 ms | 2.324 s |
| Attention side | 457.9 ms | 111.3 ms | 569.2 ms |

The GatedDeltaNet and attention sides include their adjacent layer-normalization and residual work outside the MoE block. Every forward-path optimization also reduces checkpoint recomputation, so its full-step value is larger than the isolated forward saving.

The kernel-family totals were:

| Kernel family | Forward GPU time | Backward-phase GPU time |
| --- | ---: | ---: |
| FLA/recurrent | 301.2 ms | 1.358 s |
| Routing/indexing | 60.1 ms | 1.023 s |
| Grouped MMQ | 367.2 ms | 771.6 ms |
| GEMM | 300.1 ms | 610.6 ms |
| Dense MMQ | 299.5 ms | 157.1 ms |
| Flash Attention | 16.9 ms | 333.0 ms |
| AITER GMM | 94.5 ms | 163.3 ms |
| AITER PTGMM | 0 ms | 48.2 ms |

Backward-phase totals include checkpoint recomputation. For grouped MMQ, 363.7 ms is recomputed gate/up and down forward work, while the actual paired gate/up and down input-gradient kernels cost 222.6 ms and 185.2 ms respectively, or 407.9 ms combined. AITER's backward-phase total consists of 93.5 ms of recomputed forward GMM, 69.7 ms of input-gradient GMM, and 48.2 ms of PTGMM factor gradients.

The largest unexpected operation is generic indexed-gather backward scatter:
- `aten::_index_put_impl_`: 918.8 ms.
- raw `indexing_backward_kernel`: 916.8 ms.
- 22.97 ms per decoder layer on average, with a 22.84-23.07 ms range across all 40 layers.
- 17.3% of all backward GPU kernel time and 53.8% of actual MoE backward.

This is consistent with backward scatter from the expert-row routing gathers and confirms that framework routing gather/scatter is now a primary bottleneck rather than a speculative fusion target. Other leading raw kernels are the 467.3 ms FLA `dhu` kernel, the 309.0 ms fused-causal Flash Attention backward kernel, the 253.0 ms FLA `dqkwg` kernel, and the 176.7 ms FLA WY-preparation backward kernel.

Profile artifacts:
- refined report: `/tmp/qwen35_training_step_profile_refined.json`.
- Chrome/Kineto trace: `/tmp/qwen35_training_step_trace.json`.
- reproduction driver: `/tmp/profile_qwen35_training_step.py`.

## Remaining work

- Replace the generic MoE routing gather backward scatter.
  - identify the exact indexed gather whose autograd emits the 918.8 ms `aten::_index_put_impl_` path.
  - implement a route-aware inverse permutation/reduction or custom backward that preserves top-8 routing and the existing expert ordering.
  - avoid dense token-by-expert intermediates, CPU route metadata, hidden synchronization, or materialized expert matrices.
  - measure complete-step time and memory because the current cost is almost exactly 23 ms in every decoder layer.
  - do not introduce a stock external MoE replacement merely for architectural neatness.

- Add permutation-aware MMQ for the 90 GatedDeltaNet projections.
  - preserve physical input/output permutations exactly.
  - keep authoritative GGUF values and the original BF16 LoRA input.
  - remove their remaining logical-weight compatibility materializations.
  - target the measured 255.3 ms GatedDeltaNet forward GEMM and 274.3 ms actual-backward GEMM, plus their checkpoint repetition.
  - measure full-model runtime and memory rather than relying only on isolated projections.

- Reduce GatedDeltaNet/FLA backward time.
  - actual GatedDeltaNet-side backward is 1.676 seconds, including 1.057 seconds of FLA/recurrent kernels.
  - prioritize the `chunk_gated_delta_rule_bwd_kernel_dhu`, `chunk_bwd_kernel_dqkwg`, WY-preparation, and causal-convolution backward paths.
  - retain the fixed production geometry and exact recurrence rather than replacing the layer with a different algorithm.
  - account for the additional 647.4 ms GatedDeltaNet-side checkpoint recomputation when evaluating forward changes.

- Reduce grouped packed input-gradient runtime.
  - paired gate/up and down packed input gradients cost 222.6 ms and 185.2 ms per step respectively.
  - preserve the current 8/16/64 MiB-scale live footprints.
  - tune decoder reuse, occupancy, and WMMA scheduling.
  - reject optimizations that require logical matrices, packed transposes, or cotangent quantization.

- Tune rank-4 AITER LoRA execution after the larger bottlenecks.
  - production-route-count profiling is complete: actual input-gradient GMM costs 69.7 ms, PTGMM costs 48.2 ms, and checkpoint recomputation adds 93.5 ms of forward GMM.
  - reduce factor-layout repacking and contiguous intermediates where AITER supports the required layout.
  - investigate LoRA-B accumulation only if AITER exposes safe GEMM alpha/beta or epilogue support.
  - keep this below routing scatter, GatedDeltaNet/FLA, and grouped-MMQ backward in priority.
