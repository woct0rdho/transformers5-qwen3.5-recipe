# Qwen3.6-35B-A3B LoRA training under 16 GiB

## Result

The target has been demonstrated: Qwen3.6-35B-A3B rank-4 LoRA can complete full-model batch-1, sequence-2048 training updates with both live PyTorch allocation and allocator reservation below 16 GiB.

Production configuration:

- checkpoint: `~/models/qwen3.6/Qwen3.6-35B-A3B-APEX-I-Mini.gguf`;
- architecture: text-only `Qwen3_5MoeForCausalLM`;
- tokenizer: `~/models/qwen3.5/Qwen3.5-35B-A3B`;
- complete model on `cuda:0`;
- BF16 compute and BF16 LoRA adapters;
- batch size 1, sequence length 2048;
- rank-4 ordinary and routed-expert LoRA;
- non-reentrant gradient checkpointing on all 40 decoder layers;
- bitsandbytes `adamw_8bit` for adapter parameters only;
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
| Second backward | **15.399 GiB** | **15.463 GiB** | 8.62 s |

The steady-state maximum is now the second backward rather than an avoidable forward materialization. A direct one-step `SFTTrainer.train()` run measured 14.928 GiB allocated, 15.150 GiB reserved, and 15.24 seconds.

The full audit produced all 660 finite trainable gradients, no frozen or packed-weight gradients, updates to all 330 LoRA-B tensors on the first step, and finite nonzero A/B gradients after the first update. Both packed-loss forwards returned `logits=None` and `aux_loss=None`.

Primary report: `/tmp/qwen36_packed_q8_lm_head_final_report.json`.

## Fixed training contract

The implementation is intentionally narrow. Future optimization must preserve these rules:

- GGUF is the sole base-weight representation.
- bitsandbytes is used only by `adamw_8bit`.
- Packed base parameters remain frozen and never receive gradients.
- The complete model remains on `cuda:0`; arbitrary offload, sharding, and expert parallelism are unsupported.
- Serialized adapters retain ordinary PEFT names and rank-4 shapes.
- LoRA-A always consumes the original unquantized BF16 activation.
- Packed merge/unmerge and persistent packed-base `save_pretrained` remain rejected.
- Mixed-adapter expert batches, DoRA, aLoRA, LoRA bias, and unsupported PEFT modes fail explicitly.
- Gradient checkpointing remains non-reentrant and is verified on all 40 layers after trainer construction.
- Batch size remains 1 and sequence length remains 2048 for the validated memory claim.
- Router-logit retention and the router auxiliary loss remain disabled; top-8 dispatch remains active.
- The real project dataset is not scanned, aggregated, regenerated, or rewritten without explicit approval.
- Native operators remain gfx1151-specific, asynchronous on the current Torch stream, and fail rather than inserting hidden operand copies.

## Completed optimizations

### 1. Persistent GGUF model residency

The checkpoint remains compressed after loading:

- 733 checkpoint tensors;
- 34,660,610,688 logical parameters;
- 14,216,723,456 packed payload bytes;
- 351 `GGUFLinear` modules;
- 40 `GGUFExperts` modules;
- 432 frozen `GGUFQuantizedTensor` parameters;
- approximately 13.282 GiB live allocation immediately after load.

Reusable Transformers support provides:

- `GGUFLinear.materialize_logical_weight()` as a compatibility boundary for consumers that genuinely need a canonical floating matrix;
- Qwen3.5 recurrent input/output layout handling;
- capability-validated private expert backend names for specialized `GGUFExperts` execution.

The production ordinary, routed-expert, and LM-head paths no longer materialize logical base matrices. The remaining users of logical materialization are the 90 recurrent-layout projections that do not yet have permutation-aware MMQ kernels and explicit compatibility/reference paths.

### 2. Ordinary packed LoRA

`fast_lora.py` installs PEFT-native wrappers without patching installed PEFT sources.

Current model composition:

- 250 ordinary LoRA wrappers;
- 160 ordinary packed projections using native MMQ in forward and backward;
- 90 GatedDeltaNet/recurrent-layout projections using the generic GGUF compatibility path;
- 250 ordinary LoRA-A and LoRA-B factor pairs included in normal PEFT serialization.

For the 160 native projections:

- the frozen BF16 input is dynamically quantized to Q8_1;
- `torch_ggml_ops::mmq` multiplies it by the authoritative packed GGUF weight;
- LoRA-A runs from the original BF16 input;
- LoRA-B and residual accumulation use framework BF16 GEMM/addition;
- backward calls `torch_ggml_ops::mmq_grad_input` for the frozen logical base Jacobian;
- only adapter factors receive parameter gradients.

The native backward removes logical-weight allocation at the cost of lower isolated GEMM throughput. That trade is intentional: complete-layer and full-model execution benefit from the substantially lower live allocation.

### 3. Routed-expert packed LoRA

`fast_moe_lora.py` wraps each complete `GGUFExperts` module and preserves four rank-4 BF16 factor families per adapter:

- combined gate/up A;
- combined gate/up B;
- down A;
- down B.

The project-private expert backend uses Transformers routing and reduction while replacing the frozen projection work:

- expert-sorted routed rows and cumulative offsets remain on the GPU;
- gate/up uses `grouped_mmq_pair` when both packed tensors have matching geometry and quantization type;
- otherwise gate and up use independent `grouped_mmq` calls so their GGUF metadata remains authoritative;
- down uses `grouped_mmq`;
- paired gate/up backward accumulates both frozen Jacobians in one FP32 accumulator and emits one BF16 route-gradient tensor;
- down backward uses `grouped_mmq_grad_input`;
- LoRA execution and LoRA input gradients use AITER GMM;
- factor gradients use AITER PTGMM;
- no full expert delta, effective trainable expert matrix, selected logical expert copy, or packed gradient is created.

Routing, SiLU, gate/up multiplication, route weighting, inverse permutation, reduction, and LoRA residual accumulation remain framework-level operations. Native fusion stops at the projection boundary.

A matched layer-10 sequence-2048 benchmark measured 93.38 ms for packed forward plus backward versus 195.10 ms for selective materialization plus AITER, a 52.1% complete-layer speedup. Peak allocation and reservation were also more than 1 GiB lower in the packed variant.

### 4. Native gfx1151 GGUF operators

`~/torch-ggml-ops` provides stable-ABI operators for:

- dense MMQ forward;
- dense packed input gradients;
- grouped MMQ forward;
- grouped packed input gradients;
- paired grouped gate/up forward;
- paired grouped gate/up input gradients.

Supported GGUF formats:

- `IQ2_S`;
- `Q3_K`;
- `Q4_K`;
- `Q5_K`;
- `Q6_K`.

Important implementation properties:

- BF16 inputs and outputs;
- Q8_1 activation workspace only in forward;
- BF16 WMMA with FP32 accumulation for frozen-base input gradients;
- no quantization of backward cotangents;
- no logical BF16/F32 weight allocation;
- no packed transpose or lossy repack;
- no CPU route descriptor or route-metadata synchronization;
- exact packed byte-count, dtype, device, alignment, contiguity, and zero-storage-offset validation;
- FakeTensor/meta registration, registered autograd, `torch.library.opcheck`, and `torch.compile` composition;
- explicit higher-order-gradient rejection.

The dense Q6_K backward decoder broadcasts one `d * scale` value across each 16-column WMMA tile. Wider-N and eight-wave schedules were measured and removed because they regressed the real LM-head geometry. The final package has 38 passing tests.

### 5. Packed LM-head loss

The Q6_K language-model head has logical shape `[248320, 2048]`:

- packed payload: 417,177,600 bytes, approximately 0.389 GiB;
- logical BF16 matrix avoided: approximately 0.947 GiB.

`gguf_liger_loss.py` now owns a project-local packed causal-language-model loss:

- flatten and causally shift labels;
- process 64 hidden rows per chunk;
- explicitly clone the small BF16 slice because native MMQ requires zero storage offset;
- call `mmq` to produce a 30.31 MiB BF16 logits chunk from Q8_1 activations;
- run Liger's cross-entropy Triton primitive in place so logits become `dLogits`;
- call `mmq_grad_input` against the packed Q6_K head;
- save only the complete BF16 `dHidden` for model backward.

The function preserves:

- causal shifting;
- `ignore_index=-100`;
- mean/sum reduction;
- `num_items_in_batch` scaling;
- token accuracy;
- optional predicted tokens;
- `outputs.logits=None` during fused training;
- a frozen, gradient-free LM head.

It explicitly rejects unsupported geometry, trainable or biased heads, layout permutations, class weighting, token scaling, label smoothing, z-loss, LSE square scaling, logit softcapping, and higher-order differentiation.

Numerical comparison with the logical-BF16 fused reference:

- isolated hidden-gradient cosine: `0.99999988`;
- isolated hidden-gradient relative L2 error: `0.0714%`;
- full-model hidden-gradient cosine: `0.99999779`;
- full-model all-adapter gradient cosine: `0.99998622`;
- full-model all-adapter relative L2 error: `0.527%`;
- paired first AdamW8bit update cosine: `0.997115`;
- complete post-update adapter-state relative L2 difference: `0.0863%`.

The Q8_1 activation forward is therefore retained. Backward remains the logical packed-weight Jacobian rather than differentiation through rounding.

### 6. Router auxiliary-memory removal

Top-8 routing is unchanged, but retention of all layer router logits and the generic load-balancing objective are disabled:

- `model.config.output_router_logits = False`;
- `model.config.router_aux_loss_coef = 0.0`;
- `SFTConfig(router_aux_loss_coef=0.0)`.

All three settings are required because current TRL copies its coefficient back into the model configuration. Runtime checks reject a trainer that re-enables router-logit collection or a nonzero auxiliary coefficient.

This removes the large dense one-hot/expanded router auxiliary tensors while preserving actual expert dispatch.

### 7. Trainer, data, attention, and recurrent execution

The trainer also includes:

- `get_peft_model(..., autocast_adapter_dtype=False)` so adapters remain BF16;
- explicit non-reentrant checkpoint enablement after `SFTTrainer` construction;
- verification that all 40 layers are checkpointed;
- `packing=False` and `padding_free=False`;
- compact fixed-length dataset rows containing `input_ids` and `num_tokens`;
- collator reconstruction of attention masks and `-100` labels;
- deterministic guarded Flash Attention 2 choices for the validated geometry;
- 14 exact FLA autotuner preloads across 13 autotuners;
- a project-local compiled GGUF dequantization fallback for operations that do not yet have layout-correct native MMQ, chiefly the 90 recurrent projections;
- Liger RMSNorm only where independently useful;

## Remaining work

- Add permutation-aware MMQ for the 90 GatedDeltaNet projections
  - preserve physical input/output permutations exactly;
  - keep authoritative GGUF values and the original BF16 LoRA input;
  - remove their remaining logical-weight compatibility materializations;
  - measure full-model runtime and memory rather than relying only on isolated projections.

- Improve Q6_K LM-head input-gradient throughput
  - the packed loss meets memory and quality targets but `mmq_grad_input` remains slower than dense BF16 GEMM;
  - investigate better large-reduction scheduling without increasing the 64-token workspace;
  - consider a row-range operator to remove slice clones only if profiling shows the 256 KiB copies matter;
  - retain the Q8_1 forward unless quality measurements materially change.

- Reduce dense and grouped backward runtime penalties
  - preserve current 8/16/64 MiB-scale live footprints;
  - tune decoder reuse, occupancy, and WMMA scheduling;
  - reject optimizations that require logical matrices, packed transposes, or cotangent quantization.

- Tune rank-4 AITER LoRA execution
  - profile GMM/PTGMM at production route counts;
  - reduce factor-layout repacking and contiguous intermediates where AITER supports the required layout;
  - investigate LoRA-B accumulation only if AITER exposes safe GEMM alpha/beta or epilogue support.

- Profile routing gather/scatter overhead before adding more fusion
  - route gather, weighting, inverse permutation, and reduction remain framework operations;
  - fuse them only if whole-model profiling shows they are a significant remaining cost;
  - do not introduce a stock external MoE replacement merely for architectural neatness.
