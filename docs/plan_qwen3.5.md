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
- 250 ordinary LoRA projections and 40 routed-expert wrappers. All 250 ordinary projections run the native dense-MMQ base in both directions, with no ordinary generic-dequant fallback.
- the three GatedDeltaNet backward launchers (`gdn_bwd_dhu.py`, the widened `gdn_bwd_dqkwg.py`, and `gdn_wu_recompute.py`).
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
- GatedDeltaNet keeps its fixed exact recurrence and the three validated backward launchers. It is not replaced with a different algorithm.
- The real project dataset is not scanned, aggregated, regenerated, or rewritten without explicit approval.
- Native operators remain gfx1151-specific, asynchronous on the current Torch stream, and fail rather than inserting hidden operand copies.

## Audit and profile

```bash
python audit_qwen3_5_training_step.py \
  --batch-size 1 --sequence-length 2048 --max-steps 1 --rank 4 --alpha 4 \
  --report-output ~/tmp/test_no_unsloth/qwen35/audit/qwen3_5_training_report.json \
  --profile-output ~/tmp/test_no_unsloth/qwen35/audit/qwen35_training_step_profile.json
```

`--profile-output` names the profile report. The Kineto trace is written beside it as `qwen35_training_step_profile.trace.json`. All three canonical commands use `--max-steps 1`. The profile's warm-up and traced updates are the measurement. Audit/profile artifacts live under `~/tmp/test_no_unsloth/qwen35/`, with the older profiles in `audit/archive/`.

## Completed optimizations

### Persistent GGUF model residency

The checkpoint remains compressed after loading:
- 733 checkpoint tensors.
- 34,660,610,688 logical parameters.
- 14,216,723,456 packed payload bytes.
- 351 `GgufLinear` modules.
- 40 `GgufExperts` modules.
- 432 frozen `GgufQuantizedParameter` parameters.
- approximately 13.281 GiB live allocation immediately after load.

Reusable Transformers support provides:
- `GgufLinear.materialize_logical_weight()` as a compatibility boundary for consumers that genuinely need a canonical floating matrix.
- Qwen3.5 recurrent input/output layout handling.
- capability-validated private expert backend names for specialized `GgufExperts` execution: the architecture-wide registry rejects them, but the model-level check accepts a private name once every applicable expert module's own `_validate_supported_experts_implementation` validator accepts it.

The ordinary, routed-expert, and LM-head paths no longer materialize logical base matrices. All 250 ordinary LoRA projections now run the native dense-MMQ base in both directions, including the 30 GatedDeltaNet output projections and the recurrent input and gate projections. `gdn_tiled_value_heads.py` keeps llama.cpp's tiled value-head order through the load and through the broadcast, so `linear_attn.out_proj` consumes the packed columns in their own order, and `torch-ggml-ops` deploys `(n=2048, k=4096)` for Q3_K in both directions. The remaining Q4_K layers have exact keys as well.

An earlier profile left `linear_attn.in_proj_a` and `linear_attn.in_proj_b` on the dequantize-and-multiply base because their `[48,2048]` shape was below the deployed key granularity. Those native keys now exist, and the current audit reports 250/250 ordinary wrappers native and 0 on the generic compiled-dequant path. The fallback remains installed as a compatibility path for shapes without a native key, but no ordinary projection selects it.

### Ordinary packed LoRA

`fast_lora.py` installs PEFT-native wrappers without patching installed PEFT sources.

Current model composition:
- 250 ordinary LoRA wrappers.
- 250 ordinary packed projections on the native dense MMQ forward and backward (attention, MLP, and all GatedDeltaNet projections).
- 0 ordinary projections on the generic compiled-dequant base: all 250 run the native dense MMQ base in both directions.
- 250 ordinary LoRA-A and LoRA-B factor pairs included in normal PEFT serialization.

`FastGgufLoraLinear.uses_packed_mmq()` is the single source of truth for the path decision. It refuses any module whose `GgufLinear.input_permutation` is set, which is the only remaining reason a packed projection would leave the native base. With `gdn_tiled_value_heads.py` installed the loaded model carries no permutation at all.

For the 250 native projections:
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

### GatedDeltaNet backward launchers

Three drop-in launchers replace FLA's GatedDeltaNet backward kernels for the trained geometry (16 key heads, 32 value heads, head and value dim 128, convolution width 4):
- `gdn_bwd_dhu.py` replaces the delta-rule state walk (`BT=64`, `BK=64`, `BV=32`, 8 warps, 2 stages).
- `gdn_bwd_dqkwg.py` launches FLA's `chunk_bwd_dqkwg` kernel with `BK=128` and `BV=32`, exposing the wider key tile that FLA's collapsed `CONST_TILING` hides on gfx1151. The launcher uses FLA 0.6's flattened 2-D program grid.
- `gdn_wu_recompute.py` recomputes W/U in `BD=32` blocks with two chunks per program, 1 warp, and 2 stages.

The launchers are installed by both `train_qwen3_5_35b.py` and `audit_qwen3_5_training_step.py`. The audit's `static_configuration` records them as installed. `test_gdn_bwd_dhu.py`, `test_gdn_bwd_dqkwg.py`, and `test_gdn_wu_recompute.py` pin their outputs and guards, and the full-step audit verifies the adapter gradients.

Measured on the current batch-1 sequence-2048 profile:

| Metric | Before replacements | Current |
| --- | ---: | ---: |
| GatedDeltaNet-side actual backward | 1.550 s | 920.1 ms |
| GatedDeltaNet-side checkpoint recompute | 582.9 ms | 522.6 ms |
| GatedDeltaNet-side inclusive backward | 2.132 s | 1.443 s |
| FLA/recurrent backward kernels | 1.304 s | 720.4 ms |
| GatedDeltaNet forward | 579.3 ms | 520.0 ms |
| GatedDeltaNet forward GEMM | 246.6 ms | 49.9 ms |

The GatedDeltaNet-side actual backward is 40.6% lower and the inclusive backward phase is 32.3% lower than the pre-replacement profile. The forward improvement is from the native dense-MMQ deployment of the recurrent input, gate, and output projections. `plan_qwen4_exp_gdn_backward.md` owns the isolated kernel screens. This table is the Qwen3.5 full-step result.

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

The latest isolated warmed launch sweep on gfx1151 (pre-Triton-3.9, retune pending) measured:

| Sequence-2048 batch | Tokens | Gather backward | Combine backward |
| ---: | ---: | ---: | ---: |
| 1 | 2,048 | 0.339 ms | 0.636 ms |
| 4 | 8,192 | 1.325 ms | 3.363 ms |
| 16 | 32,768 | 5.252 ms | 14.691 ms |

The current full-step profile measured 14.688 ms for all 40 gather-backward launches, 27.141 ms for all 40 combine-backward launches, 22.832 ms of gather forward, and 32.522 ms of combine forward. Generic indexed-scatter backward (`aten::_index_put_impl_`) is now absent from the step. Automated tests cover BF16 and FP32 routing weights, exact forward and gradient behavior at fixed and dynamic token counts, batch-1/4/16 dispatch policy, fallback dispatch, and absence of generic indexed-scatter backward. Separate large-grid validation executed exact constant-value forward and gradient checks at batches 4 and 16.

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

Complete fused norm kernel time is 40.6 ms per traced step: 12.3 ms gated forward, 11.7 ms gated backward, 5.2 ms Liger row-norm forward, 7.1 ms Liger row-norm backward, and 4.4 ms of block-norm variants. This replaces the eager FP32 norm graphs, which accounted for the forward `reduction` (20.3 ms), `copy_init` (69.9 ms), and part of the `elementwise` (129.8 ms) families, plus the backward `copy_init` (145.4 ms), `elementwise` (296.7 ms), and `reduction` (31.4 ms) families in the earlier profile.

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
- a project-local compiled GGUF dequantization fallback that remains installed for shapes without a native MMQ deployment. No ordinary LoRA projection selects it now that all 250 run the native base.
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
- Triton 3.9.0. This profile is the 3.9 starting point, not a re-tuned optimum: the FLA autotuner preloads, routing launch buckets, and project Triton kernels were selected under earlier Triton generations and need a retuning pass for 3.9.

The warmed traced update measured:

| Phase | Wall time | Step share |
| --- | ---: | ---: |
| Forward | 1.420 s | 30.52% |
| Backward | 3.171 s | 68.14% |
| Gradient clipping | 14.5 ms | 0.31% |
| AdamW8bit step | 47.7 ms | 1.03% |
| Total | 4.653 s | 100% |

The audit gate passed for this run. Its traced update measured 1.4201 s forward and 3.1706 s backward. The instrumented deep trace measured 1.4208 s and 3.1812 s on a separate run. The audit's warm-up update measured 1.3935 s and 3.1431 s, so the forward spread is up to about 5%.

History: the pre-`out_proj` profile was 1.433 s / 3.830 s (5.330 s full step). The `out_proj` round reduced it to 1.362 s / 3.619 s (4.98 s forward+backward) at losses 3.2423 / 3.2384. The current run is 4.653 s for the full step and 4.591 s forward+backward, 7.8% below the 4.98 s `out_proj` baseline and 12.7% below the 5.330 s pre-`out_proj` full step. Backward is 12.4% below the 3.619 s pre-GDN baseline. Forward is 4.3% above 1.362 s and within the observed run-to-run spread. The current first/second losses are 3.2415 / 3.2394.

The traced times include profiler overhead and are a single warmed sample rather than a benchmark distribution. They are suitable for relative attribution because all requested kernel families were correlated to their launching operations.

Forward decoder-layer GPU work totaled 1.171 seconds. The module rows overlap because the routed experts and shared-expert MLP are nested inside the MoE block:

| Module | GPU kernel time | Calls | Average per call |
| --- | ---: | ---: | ---: |
| MoE block | 546.7 ms | 40 | 13.67 ms |
| GatedDeltaNet | 520.0 ms | 30 | 17.33 ms |
| Routed experts | 488.8 ms | 40 | 12.22 ms |
| Full attention | 98.5 ms | 10 | 9.85 ms |
| Shared-expert MLP | 33.5 ms | 40 | 0.84 ms |

GatedDeltaNet forward contains 288.3 ms of FLA/recurrent kernels, 160.8 ms of native dense MMQ, and 49.9 ms of GEMM. The complete Flash Attention forward kernels take 23.1 ms. Approximately 218.7 ms outside the decoder layers is the packed LM-head MMQ, packed input gradient, and in-place cross-entropy path. The 40.6 ms of fused norm kernels are attributed to their owning modules.

Backward decoder-layer GPU work totaled 3.133 seconds. Non-reentrant checkpoint recomputation accounts for 1.161 seconds, or 37.1%, while actual autograd work accounts for 1.972 seconds. The exclusive decomposition is:

| Module side | Actual backward | Checkpoint recompute | Inclusive backward phase |
| --- | ---: | ---: | ---: |
| MoE | 642.7 ms | 538.6 ms | 1.181 s |
| GatedDeltaNet side | 920.1 ms | 522.6 ms | 1.443 s |
| Attention side | 408.9 ms | 100.3 ms | 509.2 ms |

The kernel-family totals are:

| Kernel family | Forward GPU time | Backward-phase GPU time |
| --- | ---: | ---: |
| FLA/recurrent | 288.3 ms | 720.4 ms |
| Grouped MMQ | 301.7 ms | 607.7 ms |
| GEMM | 95.1 ms | 179.2 ms |
| Dense MMQ | 430.9 ms | 429.9 ms |
| Flash Attention | 23.1 ms | 344.7 ms |
| AITER GMM | 72.8 ms | 151.5 ms |
| AITER PTGMM | 0 ms | 46.0 ms |
| Routing/indexing | 13.1 ms | 42.8 ms |

Backward-phase totals include checkpoint recomputation. Relative to the pre-GDN table, FLA/recurrent backward fell from 1.304 s to 720.4 ms. The GEMM-to-dense-MMQ shift (forward 293.6 to 95.1 ms and 255.1 to 430.9 ms. Backward 606.1 to 179.2 ms and 123.5 to 429.9 ms) is the native deployment of the GDN input, gate, and output projections, not a regression in either family. The remaining leading opportunities are FLA/recurrent backward (720.4 ms), grouped-MMQ (607.7 ms backward), dense MMQ (429.9 ms backward), and Flash Attention backward (344.7 ms). GEMM is no longer a leading family.

Profile artifacts:
- audit gate report: `~/tmp/test_no_unsloth/qwen35/audit/qwen3_5_training_report.json`.
- canonical profile report: `~/tmp/test_no_unsloth/qwen35/audit/qwen35_training_step_profile.json`.
- audit Kineto trace: `~/tmp/test_no_unsloth/qwen35/audit/qwen35_training_step_profile.trace.json`.
- deep refined report: `~/tmp/test_no_unsloth/qwen35/profiles/qwen35_deep_refined.json`.
- deep Kineto trace: `~/tmp/test_no_unsloth/qwen35/profiles/qwen35_deep.trace.json`.
- reproduction drivers: `audit_qwen3_5_training_step.py` (gates and canonical profile), `~/tmp/test_no_unsloth/profile_qwen35_step_deep.py` (deep module and direction attribution. It installs the GDN launchers and tiled value heads), and `~/tmp/test_no_unsloth/refine_qwen35_step_profile.py` (trace refinement).
- earlier peak-memory replay, not rerun for this update: `~/tmp/test_no_unsloth/peak_memory_profile.py` and `~/tmp/test_no_unsloth/qwen35/profiles/qwen35_peak_memory.json`.

### Memory

The ROCm-visible device capacity is 124.0 GiB.

| Boundary | Allocated | Reserved or free |
| --- | ---: | ---: |
| Packed model loaded | 13.281 GiB | 105.80 GiB free |
| Adapters injected | 13.721 GiB | 13.756 GiB reserved |
| Complete-update peak | 15.078 GiB | 15.268 GiB reserved |
| After traced update | 14.608 GiB | 103.40 GiB free |

The high-water mark occurs during the backward pass. Peak allocation is 1.797 GiB above the loaded model and 1.357 GiB above the adapter-injected state, and the post-update state sits 0.470 GiB below the peak.

The live set at the peak is the resident 13.281 GiB packed base, 0.439 GiB BF16 adapters, 0.439 GiB BF16 adapter gradients, approximately 0.46 GiB AdamW8bit state, and roughly 0.6 GiB of transient activations and workspaces. The earlier allocator replay identified the largest transient blocks as AITER GMM outputs (96 MiB), a grouped-MMQ output (64 MiB), `_fused_lora_add` addmm outputs (60 MiB), FLA `l2norm` state (32 MiB), and per-layer Liger/FLA norm buffers (16 MiB each). The GDN launchers change the recurrent workspaces, so treat that block list as historical rather than current. The deep-profile process RSS is 4.02 GiB and process swap is zero. The audit process finishes at 5.31 GiB after exporting the trace.

Both the live allocation and the allocator reservation stay below the 16 GiB claim, with about 0.922 GiB and 0.732 GiB of headroom respectively.

## Remaining work

- Retune the Triton kernels for Triton 3.9.
  - the environment now uses Triton 3.9.0. Re-screen the 17 FLA autotuner preloads in `fla_tuning.py`, the `fast_moe_routing.py` gather/combine launch buckets, and the project Triton kernels (the GDN launchers, routing, and FLA/Liger norm and loss kernels) for `num_warps`, `num_stages`, and tile sizes.
  - treat the current phase and kernel-family numbers as the Triton 3.9 starting point, not as the tuned optimum. Preserve the packed-base, numerical, and memory invariants while retuning.

- Every GatedDeltaNet projection is done: the input, gate and output projections all run the native dense MMQ base in both directions, every quant type and matrix size they carry is an exact dense deployment key, and those keys are exercised on real payloads by `test_fast_lora.py::test_fast_lora_gdn_projections_take_the_native_base`. The current profile measures GatedDeltaNet forward at 520.0 ms, including 288.3 ms FLA/recurrent, 160.8 ms dense MMQ, 49.9 ms GEMM, and 21.0 ms copy/other. Actual GatedDeltaNet-side backward at 920.1 ms, including 432.3 ms FLA/recurrent, 265.8 ms other preparation/walk kernels, 143.5 ms dense MMQ, 44.7 ms GEMM, and 33.8 ms elementwise/reduction/copy. And 522.6 ms of checkpoint recompute. The old 217/471 ms generic-fallback figure is historical.
- `linear_attn.out_proj` is done: `(n=2048, k=4096)` for Q3_K is deployed in both directions, the input permutation is gone with the value-head convention, and the 25 Q3_K and 5 Q4_K layers run the native base.
  - keep authoritative packed values and the original BF16 LoRA input. Do not materialize logical matrices for a whole layer.
  - keep the full-model measurement above as the acceptance gate. Do not reintroduce the generic path for an individual projection.

- Reduce GatedDeltaNet/FLA backward time.
  - the three replacements are installed and measured (see GatedDeltaNet backward launchers): actual GatedDeltaNet-side backward 1.550 s to 920.1 ms, inclusive backward phase 2.132 s to 1.443 s, and FLA/recurrent backward 1.304 s to 720.4 ms.
  - what is left: WY preparation (`prepare_wy_repr_bwd` 130.4 ms and `recompute_w_u_fwd` 87.8 ms in the backward phase. 88.5 ms in the forward pass), causal-convolution backward (139.2 ms), the widened `dqkwg` (203.5 ms), `dhu` (72.5 ms), and `dv_local` (49.6 ms). `gdn_fused_prep.py` targets WY/conv but cannot yet be wired for its memory cost.
  - retain the fixed and exact recurrence rather than replacing the layer with a different algorithm.
  - account for the 522.6 ms GatedDeltaNet-side checkpoint recomputation when evaluating forward changes.

- Reduce grouped packed input-gradient runtime.
  - paired gate/up and down packed input gradients cost 190.3 ms and 114.5 ms per step respectively.
  - preserve the current 8/16/64 MiB-scale live footprints.
  - tune decoder reuse, occupancy, and WMMA scheduling.
  - reject optimizations that require logical matrices, packed transposes, or cotangent quantization.

- Tune rank-4 AITER LoRA execution after the larger bottlenecks.
  - AITER GMM costs 72.8 ms forward and 151.5 ms backward, and PTGMM costs 46.0 ms backward. Checkpoint recomputation adds its own forward GMM work.
  - reduce factor-layout repacking and contiguous intermediates where AITER supports the required layout.
  - investigate LoRA-B accumulation only if AITER exposes safe GEMM alpha/beta or epilogue support.
  - keep this below GatedDeltaNet/FLA and grouped-MMQ backward in priority.
