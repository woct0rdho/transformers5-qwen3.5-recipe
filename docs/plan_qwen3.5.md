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
- Liger RMSNorm on all 101 Qwen3.5-MoE norms and FLA's fused gated RMSNorm on the 30 GatedDeltaNet norms.

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
- Fused norm patches remain instance-local, keep module classes, parameter names, shapes, and dtypes unchanged, and fail closed unless all 101 plain and 30 gated norms are handled.
- The real project dataset is not scanned, aggregated, regenerated, or rewritten without explicit approval.
- Native operators remain gfx1151-specific, asynchronous on the current Torch stream, and fail rather than inserting hidden operand copies.

## Completed optimizations

### Persistent GGUF model residency

The checkpoint remains compressed after loading:
- 733 checkpoint tensors.
- 34,660,610,688 logical parameters.
- 14,216,723,456 packed payload bytes.
- 351 `GgufLinear` modules.
- 40 `GgufExperts` modules.
- 432 frozen `GgufQuantizedParameter` parameters.
- approximately 13.283 GiB live allocation immediately after load.

Reusable Transformers support provides:
- `GgufLinear.materialize_logical_weight()` as a compatibility boundary for consumers that genuinely need a canonical floating matrix.
- Qwen3.5 recurrent input/output layout handling.
- capability-validated private expert backend names for specialized `GgufExperts` execution: the architecture-wide registry rejects them, but the model-level check accepts a private name once every applicable expert module's own `_validate_supported_experts_implementation` validator accepts it.

The ordinary, routed-expert, and LM-head paths no longer materialize logical base matrices, and the 30 GatedDeltaNet output projections have left the generic compiled-dequant path as well: `gdn_tiled_value_heads.py` keeps llama.cpp's tiled value-head order through the load and through the broadcast, so `linear_attn.out_proj` consumes the packed columns in their own order, and `torch-ggml-ops` now deploys `(n=2048, k=4096)` for Q3_K in both directions, which is the key 25 of the 30 layers needed. `in_proj_a` and `in_proj_b` are the only measured projections left on the dequantize-and-multiply base, on purpose: their `[48,2048]` shape is below the granularity of a deployed key. The recurrent input and gate projections used to be held back with it because `torch-ggml-ops` had no exact dense deployment for `linear_attn.in_proj_z` (`N=4096`, `K=2048`, Q3_K/Q4_K) or for one Q5_K `linear_attn.in_proj_qkv`. Those keys exist now, their row reorder is folded into the packed payload (`PermuteRows`/`TiledToGroupedRows` are packed-safe), so they look permutation-free at runtime and run the native base.

### Ordinary packed LoRA

`fast_lora.py` installs PEFT-native wrappers without patching installed PEFT sources.

Current model composition:
- 250 ordinary LoRA wrappers.
- 220 ordinary packed projections on the native dense MMQ forward and backward (the attention/MLP projections plus the GatedDeltaNet input and gate projections).
- 0 ordinary projections on the generic compiled-dequant base: all 250 run the native dense MMQ base in both directions.
- 250 ordinary LoRA-A and LoRA-B factor pairs included in normal PEFT serialization.

`FastGgufLoraLinear.uses_packed_mmq()` is the single source of truth for the path decision. It refuses any module whose `GgufLinear.input_permutation` is set, which is the only remaining reason a packed projection would leave the native base. With `gdn_tiled_value_heads.py` installed the loaded model carries no permutation at all.

For the 220 native projections:
- the frozen BF16 input is dynamically quantized to Q8_1.
- `torch_ggml_ops::mmq` multiplies it by the authoritative packed GGUF weight.
- LoRA-A runs from the original BF16 input.
- LoRA-B and residual accumulation use framework BF16 GEMM/addition.
- backward calls `torch_ggml_ops::mmq_grad_input` for the frozen logical base Jacobian.
- only adapter factors receive parameter gradients.

The native backward removes logical-weight allocation at the cost of lower isolated GEMM throughput. That trade is intentional: complete-layer and full-model execution benefit from the substantially lower live allocation.

### Routed-expert packed LoRA

`fast_moe_lora.py` wraps each complete `GgufExperts` module and preserves four rank-4 BF16 factor families per adapter:
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

The current full-step profile measured 15.244 ms for all 40 gather-backward launches, 26.817 ms for all 40 combine-backward launches, 22.833 ms of gather forward, and 32.475 ms of combine forward. Generic indexed-scatter backward (`aten::_index_put_impl_`) is now absent from the step. Automated tests cover BF16 and FP32 routing weights, exact forward and gradient behavior at fixed and dynamic token counts, batch-1/4/16 dispatch policy, fallback dispatch, and absence of generic indexed-scatter backward. Separate large-grid validation executed exact constant-value forward and gradient checks at batches 4 and 16.

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

The dense Q6_K backward decoder broadcasts one `d * scale` value across each 16-column WMMA tile. Wider-N and eight-wave schedules were measured and removed because they regressed the real LM-head geometry. The package test suite covers exact deployment membership, validation boundaries, registered autograd, and the grouped and paired paths. Its `test_grouped_mmq.py` and `test_public_deployment_correctness.py` modules still import the pre-refactor `transformers.integrations.gguf_dequant` path and no longer collect against the current Transformers.

### Packed LM-head loss

The Q6_K language-model head has logical shape `[248320, 2048]`:
- packed payload: 417,177,600 bytes, approximately 0.389 GiB.
- logical BF16 matrix avoided: approximately 0.947 GiB.

`packed_liger_loss.py` owns the shared chunked packed causal-language-model loss. `gguf_liger_loss.py` owns the Qwen3.5-MoE entry points and its validated constants (hidden size 2048, Q6_K head, 256-row chunks):
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

### Fused RMSNorm

`qwen3_5_fused_norms.py` installs both Qwen3.5-MoE norms on the module instances before PEFT wrapping, using the same shared patch protocol as the DeepSeek kernels:
- 101 `Qwen3_5MoeRMSNorm` instances (40 input, 40 post-attention, 10 query, 10 key, 1 final) use Liger's `LigerRMSNormFunction` with `offset=1.0`, `casting_mode="gemma"`, and `in_place=False`, preserving `x_norm * (1 + weight)` in FP32 with the input dtype restored.
- 30 `Qwen3_5MoeRMSNormGated` instances use FLA's `LayerNormGatedFunction` (`is_rms_norm=True`, SiLU gate), preserving `weight * x_norm * silu(gate)` for the GatedDeltaNet output norm.

The patched module classes, parameter names, shapes, and dtypes are unchanged, so PEFT serialization, the frozen-base contract, and the audit inventories stay valid. `require_complete_qwen35_fused_norms` fails closed unless exactly 101 plain and 30 gated norms are handled, and the audit records the complete patch inventory.

Complete fused norm kernel time is 42.6 ms per traced step: 12.2 ms gated forward, 14.5 ms gated backward, 4.6 ms Liger row-norm forward, 7.1 ms Liger row-norm backward, and 4.3 ms of block-norm variants. This replaces the eager FP32 norm graphs, which accounted for the forward `reduction` (20.3 ms), `copy_init` (69.9 ms), and part of the `elementwise` (129.8 ms) families, plus the backward `copy_init` (145.4 ms), `elementwise` (296.7 ms), and `reduction` (31.4 ms) families in the earlier profile.

### Router auxiliary-memory removal

Top-8 routing is unchanged, but retention of all layer router logits and the generic load-balancing objective are disabled:
- `model.config.output_router_logits = False`.
- `model.config.router_aux_loss_coef = 0.0`.

The driver sets the same coefficient before wrapping, and the audit rejects a model that re-enables router-logit collection or a nonzero auxiliary coefficient.

This removes the large dense one-hot/expanded router auxiliary tensors while preserving actual expert dispatch.

### Trainer, data, attention, and recurrent execution

The `train_qwen3_5_35b.py` driver and its `BF16AdapterTrainer` include:
- `get_peft_model(..., autocast_adapter_dtype=False)` so adapters remain BF16, and adapter restore keeps that dtype on checkpoint resume.
- `gradient_checkpointing=True` with `gradient_checkpointing_kwargs={"use_reentrant": False}`. The audit then verifies that all 40 layers are checkpointed and non-reentrant.
- compact fixed-length dataset rows containing `input_ids` and `num_tokens`.
- a collator that reconstructs attention masks and `-100` labels.
- deterministic guarded Flash Attention 2 choices for the validated geometry.
- 17 exact FLA autotuner preloads across 13 kernels.
- a project-local compiled GGUF dequantization fallback for operations that do not yet have layout-correct native MMQ, chiefly the 30 GatedDeltaNet output projections.
- the tiled value-head convention (`gdn_tiled_value_heads.py`), which keeps llama.cpp's order for the value axis so the output projection needs no input gather and can run the native base.
- Liger and FLA fused norms on all 131 Qwen3.5-MoE norms (see Fused RMSNorm above).

### Warmed full-step runtime profile

The latest profile uses:
- batch size 1 and sequence length 2048.
- BF16 compute and rank-4 LoRA.
- non-reentrant checkpointing on all 40 decoder layers.
- one complete warm-up AdamW8bit update followed by one Kineto CPU+GPU traced update.
- synchronized wall-clock boundaries around forward, backward, gradient clipping, and the optimizer.
- correlated GPU kernel-duration sums for module and kernel attribution.

The warmed traced update measured:

| Phase | Wall time | Step share |
| --- | ---: | ---: |
| Forward | 1.433 s | 26.89% |
| Backward | 3.830 s | 71.86% |
| Gradient clipping | 15.4 ms | 0.29% |
| AdamW8bit step | 51.4 ms | 0.96% |
| Other loop overhead | 0.2 ms | 0.00% |
| Total | 5.330 s | 100% |

The audit process measured the same update independently at 1.433 s forward and 3.830 s backward. The instrumented deep trace measured 1.434 s and 3.822 s.

After the `out_proj` round the audit measures `1.362 s` forward and `3.619 s` backward on the same batch, a `5.3%` lighter update at `4.98 s`, with all 250 ordinary wrappers on the native dense MMQ base and the losses at `3.2423` and `3.2384`.

The traced times include profiler overhead and are a single warmed sample rather than a benchmark distribution. They are suitable for relative attribution because all requested kernel families were correlated to their launching operations.

Forward decoder-layer GPU work totaled 1.203 seconds:

| Module | GPU kernel time | Calls | Average per call |
| --- | ---: | ---: | ---: |
| MoE block | 521.2 ms | 40 | 13.03 ms |
| GatedDeltaNet | 579.3 ms | 30 | 19.31 ms |
| Routed experts | 462.0 ms | 40 | 11.55 ms |
| Full attention | 97.1 ms | 10 | 9.71 ms |
| Shared-expert MLP | 33.2 ms | 40 | 0.83 ms |

GatedDeltaNet forward contains 289.0 ms of FLA/recurrent kernels and 246.6 ms of GEMM. The complete Flash Attention forward kernels take 23.4 ms. Approximately 205 ms outside the decoder layers is the packed LM-head MMQ, packed input gradient, and in-place cross-entropy path. The 42.6 ms of fused norm kernels are attributed to their owning modules.

Backward decoder-layer GPU work totaled 3.762 seconds. Non-reentrant checkpoint recomputation accounts for 1.195 seconds, or 31.8%, while actual autograd work accounts for 2.567 seconds. The exclusive decomposition is:

| Module side | Actual backward | Checkpoint recompute | Inclusive backward phase |
| --- | ---: | ---: | ---: |
| MoE | 622.2 ms | 513.4 ms | 1.136 s |
| GatedDeltaNet side | 1.550 s | 582.9 ms | 2.132 s |
| Attention side | 395.5 ms | 98.7 ms | 494.2 ms |

The kernel-family totals are:

| Kernel family | Forward GPU time | Backward-phase GPU time |
| --- | ---: | ---: |
| FLA/recurrent | 289.0 ms | 1.304 s |
| Grouped MMQ | 285.0 ms | 588.8 ms |
| GEMM | 293.6 ms | 606.1 ms |
| Dense MMQ | 255.1 ms | 123.5 ms |
| Flash Attention | 23.4 ms | 332.1 ms |
| AITER GMM | 63.1 ms | 129.7 ms |
| AITER PTGMM | 0 ms | 45.9 ms |
| Routing/indexing | 18.4 ms | 61.9 ms |

Backward-phase totals include checkpoint recomputation. The remaining leading opportunities are GatedDeltaNet/FLA backward, grouped-MMQ input gradients, GEMM, and Flash Attention: the generic-dequant GatedDeltaNet projections left the list when `out_proj` moved onto the native base, which also moves that family's share of the forward `GEMM` row into the dense-MMQ row rather than removing the row.

Profile artifacts:
- refined report: `~/tmp/test_no_unsloth/qwen35_deep_refined.json`.
- audit gate report: `~/tmp/test_no_unsloth/qwen35_audit_report.json`.
- Kineto traces: `~/tmp/test_no_unsloth/qwen35_profile.trace.json` (audit) and `~/tmp/test_no_unsloth/qwen35_deep.trace.json` (module and direction annotations).
- reproduction drivers: `audit_qwen3_5_training_step.py` (gates and canonical profile) and `~/tmp/test_no_unsloth/profile_qwen35_step_deep.py` (deep module and direction attribution).
- peak-memory replay: `~/tmp/test_no_unsloth/peak_memory_profile.py` and `~/tmp/test_no_unsloth/qwen35_peak_memory.json`.

### Memory

The ROCm-visible device capacity is 125 GiB.

| Boundary | Allocated | Reserved or free |
| --- | ---: | ---: |
| Packed model loaded | 13.283 GiB | 107.290 GiB free |
| Adapters injected | 13.742 GiB | 13.777 GiB reserved |
| Complete-update peak | 15.228 GiB | 15.354 GiB reserved |
| After traced update | 14.759 GiB | 104.790 GiB free |

The high-water mark occurs during the backward pass. Peak allocation is 1.945 GiB above the loaded model and 1.486 GiB above the adapter-injected state, and the post-update state sits 0.469 GiB below the peak.

Replaying the allocator event stream at the peak shows the live set is the resident 13.283 GiB packed base, 0.441 GiB BF16 adapters, 0.438 GiB BF16 adapter gradients, 0.458 GiB AdamW8bit state, and roughly 0.6 GiB of transient activations and workspaces. The largest transient blocks are AITER GMM outputs (96 MiB), a grouped-MMQ output (64 MiB), `_fused_lora_add` addmm outputs (60 MiB), FLA `l2norm` state (32 MiB), and per-layer Liger/FLA norm buffers (16 MiB each). Final profiler-process RSS is 3.97 GiB and process swap is zero.

Both the live allocation and the allocator reservation stay below the 16 GiB claim, with about 0.77 GiB and 0.65 GiB of headroom respectively.

## Remaining work

- Every GatedDeltaNet projection is done: the input, gate and output projections all run the native dense MMQ base in both directions, every quant type and matrix size they carry is an exact dense deployment key, and those keys are exercised on real payloads by `test_fast_lora.py::test_fast_lora_gdn_projections_take_the_native_base`. Re-measure the family, because the 217/471 ms generic-fallback figure was measured with all 90 projections on that path.
- `linear_attn.out_proj` is done: `(n=2048, k=4096)` for Q3_K is deployed in both directions, the input permutation is gone with the value-head convention, and the 25 Q3_K and 5 Q4_K layers run the native base.
  - keep authoritative packed values and the original BF16 LoRA input. Do not materialize logical matrices for a whole layer.
  - measure full-model runtime and memory rather than relying only on isolated projections.

- Reduce GatedDeltaNet/FLA backward time.
  - the same three replacements Qwen4-Exp uses are wired here as well, in `train_qwen3_5_35b.py` and `audit_qwen3_5_training_step.py`: `gdn_bwd_dhu.py` (the walk), `gdn_bwd_dqkwg.py` (the widened `dqkwg` launcher) and `gdn_wu_recompute.py` (the W/U recomputation). The geometry qualifies (16 key heads, 32 value heads, head and value dim 128, conv kernel 4), and `plan_qwen4_exp_gdn_forward.md` and `plan_qwen4_exp_gdn_backward.md` own their measured effects, but this model has not been re-measured with them, so every number in this bullet and in the kernel-family totals above is the pre-replacement one.
  - re-measure the family first: actual GatedDeltaNet-side backward was 1.550 seconds, including 1.014 seconds of FLA/recurrent kernels and 279 ms of GEMM.
  - what is left after the replacements: WY preparation (163.6 ms backward and 88.0 ms forward replay) and the causal-convolution backward (128.8 ms), which the fused preparation in `gdn_fused_prep.py` targets but cannot yet be wired for its memory cost.
  - retain the fixed and exact recurrence rather than replacing the layer with a different algorithm.
  - account for the additional 583 ms GatedDeltaNet-side checkpoint recomputation when evaluating forward changes.

- Reduce grouped packed input-gradient runtime.
  - paired gate/up and down packed input gradients cost 184.5 ms and 119.8 ms per step respectively.
  - preserve the current 8/16/64 MiB-scale live footprints.
  - tune decoder reuse, occupancy, and WMMA scheduling.
  - reject optimizations that require logical matrices, packed transposes, or cotangent quantization.

- Tune rank-4 AITER LoRA execution after the larger bottlenecks.
  - AITER GMM costs 63.1 ms forward and 129.7 ms backward, and PTGMM costs 45.9 ms backward. Checkpoint recomputation adds its own forward GMM work.
  - reduce factor-layout repacking and contiguous intermediates where AITER supports the required layout.
  - investigate LoRA-B accumulation only if AITER exposes safe GEMM alpha/beta or epilogue support.
  - keep this below GatedDeltaNet/FLA and grouped-MMQ backward in priority.
