# Qwen3.6-35B-A3B LoRA training under 16 GiB

## Result

The target has been demonstrated: Qwen3.6-35B-A3B rank-4 LoRA can complete full-model batch-1, sequence-2048 training updates with both live PyTorch allocation and allocator reservation below 16 GiB.

Configuration:
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

Future optimization must preserve these rules:
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

The ordinary, routed-expert, and LM-head paths no longer materialize logical base matrices. The remaining users of logical materialization are the 90 recurrent-layout projections that do not yet have permutation-aware MMQ kernels and explicit compatibility/reference paths.

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

SiLU, gate/up multiplication, and LoRA residual accumulation remain framework-level operations. The existing routing forward expressions are preserved, while route gathering and weighted combine use the completed Triton autograd path. Native projection fusion stops at the projection boundary.

A matched layer-10 sequence-2048 benchmark measured 93.38 ms for packed forward plus backward versus 195.10 ms for selective materialization plus AITER, a 52.1% complete-layer speedup. Peak allocation and reservation were also more than 1 GiB lower in the packed variant.

### Triton routing autograd kernels

`fast_moe_routing.py` replaces the generic route-gather and weighted-combine backward graphs for:
- contiguous CUDA BF16 hidden states shaped `[T, 2048]`, where `1 <= T <= 32768`.
- top-k indices and BF16 or FP32 weights shaped `[T, 8]`.
- contiguous route output shaped `[T * 8, 2048]`.
- no expert execution output mask.

This range includes sequence-2048 batches 1, 4, and 16. Token, route, allocation, and launch-grid sizes are derived from tensor metadata without device synchronization. Measured full-width launch buckets use 4/8 gather/combine warps through 2,048 tokens, 8/16 through 8,192 tokens, and 16/16 through 32,768 tokens. Hidden-dimension tiling was measured and rejected because repeated route metadata and additional programs made it substantially slower at all three target batches.

The forward expressions and grouped-MMQ calls remain unchanged. The custom autograd path provides:
- deterministic gather backward with expert-route duplicates processed in sorted source-position order and BF16 rounding after every accumulation, matching PyTorch's current indexed-gather gradient.
- weighted-combine backward for expert-output and routing-weight gradients in one Triton kernel.
- generic Torch routing for unsupported shapes, dtypes, devices, layouts, token counts, or masked outputs.
- no host route descriptors, `.item()` synchronization, dense logical expert weights, packed-weight changes, or grouped-MMQ ABI changes.

The latest isolated warmed launch sweep on gfx1151 measured:

| Sequence-2048 batch | Tokens | Gather backward | Combine backward |
| ---: | ---: | ---: | ---: |
| 1 | 2,048 | 0.339 ms | 0.636 ms |
| 4 | 8,192 | 1.325 ms | 3.363 ms |
| 16 | 32,768 | 5.252 ms | 14.691 ms |

The latest full-step profile remains the batch-1 profile below: it measured 14.657 ms for all 40 gather-backward launches and 33.313 ms for all 40 combine-backward launches. Residual `aten::_index_put_impl_` activity was 0.273 ms across 80 metadata operations. Automated tests cover BF16 and FP32 routing weights, exact forward and gradient behavior at fixed and dynamic token counts, batch-1/4/16 dispatch policy, fallback dispatch, and absence of generic indexed-scatter backward. Separate large-grid validation executed exact constant-value forward and gradient checks at batches 4 and 16.

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

A 2,048-row packed-loss benchmark measured:

| Chunk rows | Median core loss time | Peak allocation above resident inputs |
| ---: | ---: | ---: |
| 64 | 312.690 ms | 69.03 MiB |
| 256 | 229.958 ms | 253.57 MiB |

The 256-row schedule is 26.5% faster in the complete MMQ-forward, in-place cross-entropy, and MMQ-backward loop. Its additional approximately 184.5 MiB peak allocation is accepted.

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

The latest profile uses:
- batch size 1 and sequence length 2048.
- BF16 compute and rank-4 LoRA.
- non-reentrant checkpointing on all 40 decoder layers.
- one complete warm-up AdamW8bit update followed by one Kineto CPU+GPU traced update.
- synchronized wall-clock boundaries around forward, backward, gradient clipping, and the optimizer.
- correlated GPU kernel-duration sums for module and kernel attribution.

The warm-up update took 9.998 seconds. The warmed traced update measured:

| Phase | Wall time | Step share |
| --- | ---: | ---: |
| Forward | 2.019 s | 31.04% |
| Backward | 4.406 s | 67.71% |
| Gradient clipping | 10.6 ms | 0.16% |
| AdamW8bit step | 46.7 ms | 0.72% |
| Other loop overhead | 24.0 ms | 0.37% |
| Total | 6.507 s | 100% |

The traced times include profiler overhead and are a single warmed sample rather than a benchmark distribution. They are suitable for relative attribution because all requested kernel families were correlated to their launching operations.

Forward decoder-layer GPU work totaled 1.448 seconds:

| Module | GPU kernel time | Calls | Average per call |
| --- | ---: | ---: | ---: |
| MoE block | 694.2 ms | 40 | 17.35 ms |
| GatedDeltaNet | 638.4 ms | 30 | 21.28 ms |
| Routed experts | 626.1 ms | 40 | 15.65 ms |
| Full attention | 109.8 ms | 10 | 10.98 ms |
| Shared-expert MLP | 40.2 ms | 40 | 1.01 ms |

GatedDeltaNet forward contains 296.0 ms of FLA/recurrent kernels and 253.4 ms of GEMM. The complete Flash Attention forward kernels take 16.9 ms. Approximately 235 ms outside the decoder layers is the packed LM-head MMQ, packed input gradient, and in-place cross-entropy path.

Backward decoder-layer GPU work totaled 4.311 seconds. Non-reentrant checkpoint recomputation accounts for 1.441 seconds, or 33.4%, while actual autograd work accounts for 2.870 seconds. The exclusive decomposition is:

| Module side | Actual backward | Checkpoint recompute | Inclusive backward phase |
| --- | ---: | ---: | ---: |
| MoE | 742.6 ms | 688.0 ms | 1.431 s |
| GatedDeltaNet side | 1.670 s | 641.8 ms | 2.311 s |
| Attention side | 458.2 ms | 111.2 ms | 569.4 ms |

The kernel-family totals are:

| Kernel family | Forward GPU time | Backward-phase GPU time |
| --- | ---: | ---: |
| FLA/recurrent | 296.0 ms | 1.348 s |
| Grouped MMQ | 364.9 ms | 771.4 ms |
| GEMM | 298.3 ms | 609.8 ms |
| Dense MMQ | 296.1 ms | 157.5 ms |
| Flash Attention | 16.9 ms | 333.6 ms |
| AITER GMM | 93.9 ms | 168.5 ms |
| AITER PTGMM | 0 ms | 48.1 ms |
| Routing/indexing | 60.0 ms | 118.9 ms |

Backward-phase totals include checkpoint recomputation. The remaining leading opportunities are GatedDeltaNet/FLA, grouped-MMQ input gradients, GEMM, and Flash Attention.

Profile artifacts:
- refined report: `/tmp/qwen35_training_step_profile_refined_routing.json`.
- Chrome/Kineto trace: `/tmp/qwen35_training_step_trace_routing.json`.
- reproduction driver: `/tmp/profile_qwen35_training_step.py`.

## Remaining work

- Add permutation-aware MMQ for the 90 GatedDeltaNet projections.
  - preserve physical input/output permutations exactly.
  - keep authoritative GGUF values and the original BF16 LoRA input.
  - remove their remaining logical-weight compatibility materializations.
  - target the measured 253.4 ms GatedDeltaNet forward GEMM and 274.2 ms actual-backward GEMM, plus their checkpoint repetition.
  - measure full-model runtime and memory rather than relying only on isolated projections.

- Reduce GatedDeltaNet/FLA backward time.
  - actual GatedDeltaNet-side backward is 1.670 seconds, including 1.053 seconds of FLA/recurrent kernels.
  - prioritize the `chunk_gated_delta_rule_bwd_kernel_dhu`, `chunk_bwd_kernel_dqkwg`, WY-preparation, and causal-convolution backward paths.
  - retain the fixed and exact recurrence rather than replacing the layer with a different algorithm.
  - account for the additional 641.8 ms GatedDeltaNet-side checkpoint recomputation when evaluating forward changes.

- Reduce grouped packed input-gradient runtime.
  - paired gate/up and down packed input gradients cost 223.1 ms and 183.9 ms per step respectively.
  - preserve the current 8/16/64 MiB-scale live footprints.
  - tune decoder reuse, occupancy, and WMMA scheduling.
  - reject optimizations that require logical matrices, packed transposes, or cotangent quantization.

- Tune rank-4 AITER LoRA execution after the larger bottlenecks.
  - route-count profiling is complete: actual input-gradient GMM costs 74.5 ms, PTGMM costs 48.1 ms, and checkpoint recomputation adds 93.9 ms of forward GMM.
  - reduce factor-layout repacking and contiguous intermediates where AITER supports the required layout.
  - investigate LoRA-B accumulation only if AITER exposes safe GEMM alpha/beta or epilogue support.
  - keep this below GatedDeltaNet/FLA and grouped-MMQ backward in priority.
