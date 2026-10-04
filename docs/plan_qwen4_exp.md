# Qwen4-Exp LoRA training plan for gfx1151

## Result

One complete rank-4 LoRA update on `Qwen4ExpForCausalLM` from the persistent GGUF checkpoint is the unit this project validates: batch 1, sequence length 2048, the complete model on `cuda:0`, and per-decoder-layer non-reentrant gradient checkpointing.

```bash
python audit_qwen4_exp_training_step.py --max-steps 3 --report-output ~/tmp/test_no_unsloth/qwen4_exp_training_report.json --profile-output ~/tmp/test_no_unsloth/qwen4_exp_profile.trace.json
```

The audit passes on the active checkpoint, `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0.gguf`: 61.93 GiB resident, 448 packed parameters, 64,927,247,360 packed bytes, and 699,678,720 trainable parameters in 792 BF16 adapter tensors across 348 wrapped modules. Its report defaults to `qwen4_training_report.json` in the working directory, and the accepted runs write `~/tmp/test_no_unsloth/qwen4_exp_training_report.json`. Temporary scripts, artifacts and profiles for this project live in `~/tmp/test_no_unsloth/`.

One complete update costs 9.31 s: 2.67 s forward, 6.50 s backward, 0.03 s gradient clipping and 0.11 s of the optimizer step, and the two warm steps agree to 3 ms. The load costs 22.6 s, and the first forward of a fresh process costs 4.01 s against the 2.67 s of a warm one, because it still carries the compiled dequantizer's remaining shape specializations.

Memory is 61.93 GiB resident after load, 63.31 GiB with the adapters, and a 67.51 GiB allocated / 68.99 GiB reserved peak across the measured steps in the machine's 125 GiB pool, with 5.27 GiB of process RSS and no swap. An opt-in mode keeps the PLE n-gram table in the GGUF file instead of device memory: `audit_qwen4_exp_training_step.py --ple-on-disk` runs the same update in 9.305 and 9.312 s with 35.10 GiB resident after load, 39.15 GiB allocated after the measured steps and a 40.69 GiB peak, which is what brings this step under 40 GiB of live allocation.

The GatedDeltaNet kernel round is wired end to end: `gdn_bwd_dhu.py`, `gdn_bwd_dqkwg.py` and `gdn_wu_recompute.py` replace three FLA kernels at their call sites and are worth 0.83 s of the update, 10.14 s to 9.31 s, with the allocation unchanged.

No family dominates any more. The routed experts, the GatedDeltaNet family and the hyper-connections are the top three at 2.90 s, 1.99 s and 1.34 s, and the attribution table below is the measured order.

Future optimization must preserve the production contract: packed GGUF payloads stay the only base-weight representation, adapters stay rank-4 BF16 with ordinary PEFT names, the complete model stays on `cuda:0`, checkpointing stays non-reentrant and verified on all 48 layers, and unsupported geometry fails closed rather than falling back silently.

## Current state

| Area | State |
| --- | --- |
| Packed GGUF load, payload audit and `Q2_0` support | Complete |
| Native dense packed MMQ for the ordinary projections | Complete |
| Routed experts on the shared grouped-MMQ base with AITER factors | Complete |
| Packed LM-head loss | Complete |
| Router selection on the shared streaming kernel | Complete |
| QSA attention and the indexer fast path | Complete |
| Fused norms (plain, gated, grouped) | Complete |
| Hyper-connection fused norm | Complete |
| GatedDeltaNet value-head convention and native `out_proj` | Complete |
| GatedDeltaNet kernel replacements (walk, `dqkwg`, W/U) | Complete and wired |
| GatedDeltaNet fused preparation | Written and tested, not wired |
| PLE table on disk (opt-in) | Complete and measured |
| llama.cpp adapter export for this surface | Complete |
| Physical B4 and B16 full update | Pending |

The production stack is assembled and accepted at physical B1/S2048. What is left is attribution-driven work inside `torch-ggml-ops` and in the model rather than new wiring: the routed experts' paired `Q2_0` forward body, the residual-stream arithmetic the decoder layer owns, the hyper-connection elementwise chain, and the fused preparation's 6.3 GiB of saved state, which is why it is written but disconnected. Memory is a solved question at the model's own size: the PLE table can move to the file, which takes the step under 40 GiB, and what remains resident is the 31.6 GiB of routed experts, which every step reads in full.

A physical B4 then B16 full update is the next milestone. The kernels and their isolated measurements already cover both.

## Production contract

### Model

The active checkpoint is `~/models/qwen4/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0.gguf` (66,423,878,496 bytes = 61.86 GiB, `general.file_type = 41`, MOSTLY_Q2_0, quantized by ISTA DASLab).

It is a non-uniform mixed-precision conversion of the same base model as the APEX-I-Nano and IQ4_NL files, and the three are directly comparable at `-c 8192 --chunks 1`: PPL 3.6730 against 3.6257 (APEX-I-Nano) and 3.4149 (IQ4_NL). `general.architecture = qwen4exp`, native model type `qwen4_exp_text`.

| Property | Value |
| --- | --- |
| tensors | 1224 (448 packed, 61.852 GiB total, 60.470 GiB packed) |
| decoder layers | 48 |
| linear attention (GatedDeltaNet) layers | 36 |
| QSA layers | 12 (indices 3, 7, 11, ..., 47, `attention.compress_ratios = 4`) |
| hidden size | 2560 |
| hyper-connections | `hc_count = 4`, `hc_lowrank = 320` (stream width 10240) |
| attention | 24 query heads, 2 KV heads, head dim 256 |
| GatedDeltaNet | 16 key heads, 48 value heads, head and value dim 128, conv kernel 4, sigmoid output gate |
| QSA indexer | 4 query heads, head dim 128, budget 2048, compress ratio 4, `block_topk = 512` |
| MoE | 512 experts, top-10, expert intermediate 640, shared expert 640 |
| PLE | one site, decoder layer index 1 (one-based 2), `ngram_size = 3`, 8 heads per n-gram, head dim 160, conv kernel 4 |
| PLE table | `per_layer_token_embd.weight`, logical `[320001536, 160]` = 51.2 B parameters, IQ4_NL, 26.82 GiB |
| routed experts | 144 tensors (gate, up, down), all `Q2_0`, 32,400 MiB |
| token embedding | `token_embd.weight` Q3_K, 260.5 MiB |
| LM head | `output.weight` Q5_K, 416.8 MiB, untied |
| hyper-connection weights | 194 projections in BF16 (1.18 GiB), the rest F32 |
| context metadata | 262144 (training uses 2048) |

Types present (GGML id: count, packed bytes): Q2_0 202 / 31.71 GiB, IQ4_NL 8 / 26.83 GiB (27.4 of the 26.8 GiB total is the PLE table), BF16 483 / 1.37 GiB, Q3_K 91 / 0.72 GiB, Q5_K 16 / 0.49 GiB, IQ4_XS 56 / 0.41 GiB, Q4_K 38 / 0.22 GiB, Q6_K 12 / 0.08 GiB, Q4_0 16 / 14 MiB, F32 292 / 9 MiB, Q5_0 7 / 7 MiB, Q8_0 2 / 3 MiB, F16 1 / 22 KiB.

`transformers.integrations.gguf.dequant` implements all of them - `Q2_0` (id 42) was added for this checkpoint - so the generic path can execute every tensor in this file.

`Q2_0` is `QK2_0 = 64` weights per block in 18 bytes: an fp16 scale, then sixteen bytes holding four consecutive 2-bit codes each, lowest bits first, and code `q` is the level `q - 1` (so -1, 0, 1, 2, times the scale). There is no codebook, which is why this recipe is both cheap to decode and easy to fuse, and why the expert path stopped paying the IQ1_S/IQ2_XXS codebook lookup that dominated the previous checkpoint.

The decoder was verified against the real files before anything was trained on it, and the checks are part of the contract:
- Structural: the reader derives every tensor's byte size from its block geometry, so a wrong geometry desynchronizes every offset after the first `Q2_0` tensor. With 64 weights per 18 bytes the 1224 tensors lay out to exactly the file's length, 66,423,878,496 bytes, with 0 bytes left over after the last tensor at a 32-byte alignment, and the 202 `Q2_0` tensors account for 34,047,037,440 bytes.
- Cross-file: seven real `Q2_0` tensors (both routed expert shapes, one shared expert, three attention projections, one PLE projection) decode against the same tensors of the `Qwen3.8-Flash-Next-Q8_0.gguf` conversion, which the already-validated `Q8_0` path decodes. Cosine similarity is 0.856 to 0.936, the least-squares slope 0.83 to 1.00, the intercept at most 2e-5 of the scale, and the relative L2 error 0.35 to 0.55, which is what 2.25 bits per weight buys. The intercept is the sharp check: it is the level offset, so a wrong code mapping would move it far more than the quantization noise does.
- Reference: a numpy transcription of `dequantize_row_q2_0` from `ggml/src/ggml-quants.c`, written independently of the torch decoder, is bitwise equal to it on real tensor bytes, on 512 synthetic blocks that cover all four codes and all byte positions, and (under `equal_nan`) on fully random bytes, NaN scales included. `~/tmp/test_no_unsloth/qwen4_q2_0_verify.py` runs all of it.
- Upstream test: `tests/quantization/ggml/test_gguf_integration.py` passes, including the `Q2_0ReferenceTest` and the all-types comparison against `gguf-py`'s numpy decoders, which now names the types `gguf-py` has no numpy decoder for instead of failing on them.

### Target workload

- device `cuda:0`, architecture `gfx1151`, wave32, 64 KiB LDS limit.
- BF16 activations and rank-4 BF16 adapters.
- sequence length 2048, fixed.
- physical batches 1, 4 and 16.
- no CPU offload.
- per-decoder-layer non-reentrant checkpointing.
- checkpoint-specific packed tensor types and shapes.

| Physical batch | Tokens | Routed rows (top-10) | Delta-rule state `[B,48,128,128]` fp32 |
| ---: | ---: | ---: | ---: |
| 1 | 2,048 | 20,480 | 3 MiB |
| 4 | 8,192 | 81,920 | 12 MiB |
| 16 | 32,768 | 327,680 | 50 MiB |

Placement and loading are part of the contract, because this machine's unified memory is what limits the measurements:
- `gguf_mmap_policy="pread"`, never file-backed mmap. On this APU a mapped source is charged twice against the same budget (GPU plus page cache) and the load OOMs, see safetensors#728.
- The reader's pinned staging is bounded. `pread` is fast because the buffer it reads into is pinned, which turns the following transfer into a direct DMA, but PyTorch's host caching allocator retains a freed pinned block per size class, and on a unified-memory device those blocks are charged to the same pool as the model. A 28.8 GB read therefore left a 32 GiB pinned block resident for the rest of the process. `_GgufFileReader` now stages reads above one gibibyte in a pageable buffer (`PINNED_STAGING_LIMIT_BYTES`), so the load keeps the pinned path for every other tensor: the retained footprint falls from 34.1 GiB to 2.1 GiB and the load time is unchanged (`~/transformers/src/transformers/integrations/gguf/reader.py`).
- The PLE table stays on `cuda:0` by default. `_no_placement_params` only affects automatic device maps, and this project passes an explicit `device_map={"": "cuda:0"}`. `--ple-on-disk` (or `QWEN4_PLE_ON_DISK=1` in the trainer) leaves its payload in the GGUF file and gathers rows per forward instead, which is the mode to train in when memory rather than time is the constraint. General host offload remains a later option, not a current one.
- No allocator tuning. `expandable_segments` was tried and is not needed: the end-to-end test in `~/transformers` passes without it, and so does this audit, with the same load time and less retained memory.

The tokenizer is the Qwen3.5/3.6 tokenizer: identical `tokens` and `merges` SHA256 against the Qwen3.6 GGUF, ids `0..248043` and all 26 added tokens identical to the HF `Qwen/Qwen3.5-35B-A3B` tokenizer, and 150 raw corpus documents encode identically. `data_tokenized_qwen3.5` is reused unchanged (1,961,644 rows of `uint32[2048]` plus `num_tokens`).

Padding is id 248044, and the model rewrites padded positions to its EOS before the PLE lookup.

### Adapter surface

Rank 4, `lora_alpha = 4`, no dropout, no bias, no DoRA, no merging. `target_modules` is the single indexer-excluding pattern in `qwen4_exp_lora.py`.

| Family | Pattern | Base shape | Count | A / B per module | Parameters |
| --- | --- | --- | ---: | --- | ---: |
| GatedDeltaNet QKV | `linear_attn.in_proj_qkv` | `[10240, 2560]` | 36 | `[4,2560]` / `[10240,4]` | 1,843,200 |
| GatedDeltaNet gate | `linear_attn.in_proj_z` | `[6144, 2560]` | 36 | `[4,2560]` / `[6144,4]` | 1,253,376 |
| GatedDeltaNet output | `linear_attn.out_proj` | `[2560, 6144]` | 36 | `[4,6144]` / `[2560,4]` | 1,253,376 |
| QSA query | `self_attn.q_proj` | `[12288, 2560]` | 12 | `[4,2560]` / `[12288,4]` | 712,704 |
| QSA key | `self_attn.k_proj` | `[512, 2560]` | 12 | `[4,2560]` / `[512,4]` | 147,456 |
| QSA value | `self_attn.v_proj` | `[512, 2560]` | 12 | `[4,2560]` / `[512,4]` | 147,456 |
| QSA output | `self_attn.o_proj` | `[2560, 6144]` | 12 | `[4,6144]` / `[2560,4]` | 417,792 |
| Shared expert gate | `mlp.shared_expert.gate_proj` | `[640, 2560]` | 48 | `[4,2560]` / `[640,4]` | 614,400 |
| Shared expert up | `mlp.shared_expert.up_proj` | `[640, 2560]` | 48 | `[4,2560]` / `[640,4]` | 614,400 |
| Shared expert down | `mlp.shared_expert.down_proj` | `[2560, 640]` | 48 | `[4,640]` / `[2560,4]` | 614,400 |
| Ordinary total | | | 300 | | 7,618,560 |
| Routed gate/up | `mlp.experts.lora_A` / `lora_B` | `[512, 1280, 2560]` | 48 | `[512,4,2560]` / `[512,1280,4]` | 377,487,360 |
| Routed down | `mlp.experts.lora_A_down` / `lora_B_down` | `[512, 2560, 640]` | 48 | `[512,4,640]` / `[2560,4]` | 314,572,800 |
| Expert total | | | 48 | | 692,060,160 |
| Trainable total | | | 348 wrappers | | 699,678,720 |

The routed expert adapter is one wrapper per `GgufExperts` module, not per projection: the module owns `gate_proj`, `up_proj`, and `down_proj` as packed rank-3 parameters, so only the complete expert operation is a correct target. Gate and up share one family because the checkpoint stores them as one pair.

Factors are per-expert planes, so a single tensor covers all 512 experts and every tensor receives a gradient even when some experts are never selected.

Frozen and adapter-free: the PLE table and all embeddings, the QSA indexer (`index_qk_proj.q_proj`, `index_qk_proj.k_proj`, `q_layernorm`, `k_layernorm` - its selection is a `topk` with no gradient, so an adapter there would never update), `shared_expert_gate`, hyper-connection projections, all 184 norms, `dt_bias`, `A_log`, `linear_attn.conv1d`, `ple.conv1d`, `ple.key_proj`, `ple.value_proj`, and the tied/untied LM head.

The 48 routers are adapter-free in this script too, but their patch does not freeze them: `configure_fast_moe_ranking` replaces the forward only and leaves the gates' gradient flags as it found them, so a run that trains the gates directly keeps them trainable. What freezes them here is PEFT, which turns off every parameter outside its adapter prefixes.

Two of the frozen sites above still get native kernels, because they are executed on every token and are big enough to pay for it.

### Required semantics

- Packed GGUF payloads are the frozen source of truth. Production must not materialize a logical dequantized base matrix for a whole layer.
- Only adapters receive gradients. Every packed parameter, norm, router, embedding and the LM head stay frozen and gradient-free.
- `bitsandbytes` is used only by `adamw_8bit`, for adapter parameters only.
- Serialized adapters keep ordinary PEFT names and rank-4 shapes.
- LoRA-A always consumes the original unquantized BF16 activation.
- GatedDeltaNet keeps llama.cpp's tiled value-head order end to end (`gdn_tiled_value_heads.py`), so `out_proj` consumes the packed columns in their own order and no projection carries an input permutation.
- Fused norm patches stay instance-local, keep module classes, parameter names, shapes and dtypes unchanged, and fail closed unless the expected inventory is handled.
- Router-logit retention and the router auxiliary loss stay disabled. Top-10 dispatch stays active.
- Unsupported architecture, batch, sequence length, layout, dtype, mask, cache or grouped-GEMM shape falls closed to the model's own path or is rejected.
- The real project dataset is not scanned, aggregated, regenerated or rewritten without explicit approval.

## Latest accepted results

### Full-model update

The authoritative model-level boundary is physical B1/S2048 with 48 non-reentrant decoder checkpoints, measured on torch `2.14.0+rocm10.2.0a20261003` / HIP `7.17.26392`.

| Phase | First update | Second update | Warm update |
| --- | ---: | ---: | ---: |
| load | 22.58 s | - | - |
| adapter injection | 0.27 s | - | - |
| optimizer create | 0.01 s | - | - |
| forward | 4.01 s | 2.71 s | 2.67 s |
| backward | 6.87 s | 6.50 s | 6.51 s |
| gradient clipping | 0.08 s | - | 0.03 s |
| optimizer step | 0.14 s | - | 0.11 s |

The load, the adapter injection and the optimizer creation are one-time costs and are not part of an update: the first update totals 11.10 s and the warm update 9.31 s. The losses of the measured updates are `3.4914` and `3.4869`, and the clipping norms before the first update and in the two measured ones are `0.2754`, `0.2832` and `0.3301`.

The first update is the slow one, at 11.10 s, because it carries the compiled dequantizer's autotune. The second measures 9.22 s of forward and backward (it runs no clipping or optimizer step) and the two warm steps after it 9.31 s including both.

The traced update now completes as well. It was OOM-killed before the indexer fast path, because the indexer's 2048-iteration Python loop alone produced 3,265,884 of the step's ~3.6 million profiler events. With the attention wiring skipping the indexer call, the trace exports at 104 MB:

| Update | Forward | Backward | Gradient clip | Optimizer | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Warm untraced | 2.669 s | 6.515 s | 30.6 ms | 108.6 ms | 9.323 s |
| Kineto traced | 2.716 s | 6.567 s | 29.8 ms | 109.6 ms | 9.422 s |

The trace is `~/tmp/test_no_unsloth/qwen4_final_profile.trace.json`, written by the audit's `--profile-output`. Its own buffers add about 5 GiB to the process while it is capturing, which is why the profile block's allocation reading (70.85 GiB) is above the untraced peak of 67.51 GiB.

### Whole-update attribution

The module-category attribution comes from `~/tmp/test_no_unsloth/qwen4_light_profile.py`, which measures the same module ranges with CUDA events and applies the same configuration calls as the audit. The table is the third step of a `--steps 3 --with-optimizer` run: 2.66 s of forward and 6.48 s of backward, whose 2.37 s of recomputed layer forwards are attributed separately and are not part of the backward column.

The categories cover the forward to within 0.20 s, and the backward closes exactly once the recomputation is added back: 4.11 s of backward self plus 2.37 s of recomputation against the 6.48 s wall.

| Category | Forward self | Share | Backward self | Share | Recomputed |
| --- | ---: | ---: | ---: | ---: | ---: |
| routed experts (grouped MMQ base) | 900.5 ms | 33.8% | 1101.7 ms | 17.0% | 899.8 ms |
| GatedDeltaNet (norms, conv, kernels, gates. Projections are children) | 416.3 ms | 15.6% | 1152.3 ms | 17.8% | 417.4 ms |
| ordinary projections (all 300 wrappers) | 503.2 ms | 18.9% | 477.7 ms | 7.4% | 503.7 ms |
| hyper-connection (fused norm, two projections, gates, mean) | 349.5 ms | 13.1% | 644.3 ms | 9.9% | 346.2 ms |
| QSA attention (norms, RoPE, kernels, gate) | 69.4 ms | 2.6% | 381.1 ms | 5.9% | 70.9 ms |
| decoder-layer residual arithmetic (the two injection pairs) | 88.2 ms | 3.3% | 174.4 ms | 2.7% | - |
| shared expert | 55.1 ms | 2.1% | 65.7 ms | 1.0% | 55.0 ms |
| PLE layer (with its three grouped norms) | 42.0 ms | 1.6% | 64.2 ms | 1.0% | 40.1 ms |
| MoE block remainder (gate projection, routing combine) | 32.7 ms | 1.2% | 43.3 ms | 0.7% | 32.9 ms |
| routed-expert dispatch | 0.2 ms | 0.0% | 0.3 ms | 0.0% | 0.2 ms |
| PLE n-gram embedding | 0.6 ms | 0.0% | - | - | 0.7 ms |
| QSA indexer (not called) | 0.0 ms | 0.0% | - | - | 0.0 ms |
| attributed total | 2.46 s | | 4.11 s | | 2.37 s |

All 300 ordinary wrappers are one row because every ordinary projection now runs the native dense MMQ base. The QSA indexer row is zero for a structural reason: with the attention wiring in place the indexer is not called at all, so there is nothing left for the fast path to skip at run time. The PLE row is the resident table's. With the table on disk and prefetched it keeps the same forward time, because the gather overlaps the backward, and without the prefetch it absorbs the 0.2-1.6 s the reads cost.

The tree behind the table is built from the measured event windows rather than from the order the hooks fired.

With gradient checkpointing a layer's recomputation runs inside the enclosing backward pass but outside any backward span that is open at the time, so a stack-built tree attaches it to nothing: its wall time was then reported as the enclosing layer's self time while it also appeared in the recomputed bucket, and the decoder layer's backward row read 2.73 s against the 0.26 s of arithmetic actually there. Nesting the spans by their own windows fixes that.

`~/tmp/test_no_unsloth/decompose_qwen4_remainder.py` and `~/tmp/test_no_unsloth/walk_qwen4_span_tree.py` are the probes that showed the inconsistency, and `~/tmp/test_no_unsloth/isolate_qwen4_layer.py` confirms the corrected numbers against a single real layer with checkpointing on and off: the layer's own backward is 3.6 ms, and the recomputation costs exactly one extra forward, 44-57 ms per layer.

### Memory

The ROCm-visible device capacity is 125 GiB, and on this APU the GPU's memory is the system RAM itself, so the resident model, the page cache and the training state share one pool.

| Boundary | Allocated | Reserved or free |
| --- | ---: | ---: |
| Packed model loaded | 61.93 GiB | 63.63 GiB reserved, 54.73 GiB device free |
| Adapters injected | 63.31 GiB | 64.89 GiB reserved, 53.43 GiB free |
| Complete-update peak | 67.51 GiB | 68.99 GiB reserved |
| After the measured steps | 65.97 GiB | 68.99 GiB reserved, 49.09 GiB free |

The peak is 5.58 GiB above the loaded model and 4.20 GiB above the adapter-injected state. Process RSS is 4.40 GiB after load, 4.47 GiB with the adapters and 5.27 GiB after the updates. `MemAvailable` falls from 52.30 GiB to 46.15 GiB across the run and process swap stays at 0 B throughout.

The same audit with `--ple-on-disk`, which is the mode a smaller machine would run, moves the 26.82 GiB payload into the file:

| Boundary | Allocated | Reserved or free |
| --- | ---: | ---: |
| Packed model loaded | 35.10 GiB | 36.81 GiB reserved, 81.53 GiB device free |
| Adapters injected | 36.49 GiB | 38.07 GiB reserved, 80.23 GiB free |
| Complete-update peak | 40.69 GiB | 42.17 GiB reserved |
| After the measured steps | 39.15 GiB | 42.17 GiB reserved, 75.89 GiB free |

That is 26.82 GiB less at every boundary, `MemAvailable` stays above 72.94 GiB, and process RSS is unchanged (4.42, 4.49 and 5.31 GiB). The mode costs 26.82 GiB of memory because the payload is read from the file, not because anything else about the step changed: the audit's inventory still counts the table from the checkpoint header, and the load gets faster rather than slower (10.79 s against 22.58 s), because 28.80 GB is no longer read into the machine only to be copied to the device.

Two facts about the resident table are worth keeping. First, the three GatedDeltaNet replacements change no tensor the step keeps, so the allocation is identical to the state before them. Second, the transition stall that earlier runs showed - 96.6 s and 70.3 s for the update after the first one, with 2.2 GB of swap - was not the model, the kernels or the audit's gates: it was a 34.1 GiB pinned staging block the loader left behind, which on a unified-memory device came out of the same pool as the model and left the allocator to reclaim on the first `optimizer.step()`. Bounding the reader's staging removed it, and `device_free` after load rose from 24.7 GiB to the 54.73 GiB above.

### Audit gates and what they prove

The audit re-derives every quantity below on the active checkpoint and fails closed on any mismatch. The values are the ones the accepted run recorded.

| Gate | Acceptance |
| --- | --- |
| load inventory: 1227 state tensors (1224 file tensors + 3 persistent PLE buffers), 448 packed parameters, 64,927,247,360 packed bytes and their type histogram, 737 `GgufLinear`, 12 indexers, 2 embeddings, 48 experts, 48 layers | exact, and derived from the checkpoint header |
| all tensors on `cuda:0`, no missing, unexpected or mismatched keys | exact |
| PLE: one site at layer 1, `[320001536, 160]` IQ4_NL, `requires_grad=False`, payload identity stable across the update | exact |
| PLE on disk (when asked for): 1 table patched, 0 skipped, the parameter holds no payload, the geometry comes from the header, and the payload identity is sampled from the file | exact, and equal to the resident run's `sha256` |
| QSA: 12 layers at indices 3..47, budget 2048, compress ratio 4, `block_topk == max_complete_blocks == 512` | selection is exhaustive at the audited length |
| router: 48 patched, 98,304 rows checked, 0 selections below the kth score, selected weights within `1.5e-5` relative RMSE of the model's own `torch.topk` selection, every id difference a tied-expert exchange (46,587 rows, 106,997 ids) | exact threshold, bounded weight error |
| adapters: 300 ordinary + 48 expert wrappers, 792 BF16 tensors, 699,678,720 parameters, 300 native-MMQ ordinary wrappers, 0 generic ones, 0 packed trainable | exact |
| GatedDeltaNet value-head convention: 8 value reorders dropped at load, 36 layers taking the tiled broadcast, 0 modules with an input permutation | exact |
| GatedDeltaNet kernel replacements installed: `gdn_bwd_dhu` (`BT 64`, `BK 64`, `BV 32`, 8 warps, 2 stages, BF16 query operand), `gdn_bwd_dqkwg` (`BK 128`, `BV 32`), `gdn_wu_recompute` | exact |
| FLA autotune table: 20 entries preloaded, the step used 10 of them across 9 kernels | non-zero coverage |
| frozen packed MMQ: 1 module (`model.layers.1.ple.key_proj`), no other frozen projection left on the generic forward | exact |
| LM head: packed loss installed, head is `Q5_K` (type 13) | exact |
| checkpointing: 48/48 layers, non-reentrant, `use_cache=False`, router logits and auxiliary loss off | exact |
| autocast activation dtype: one forward and backward under the trainer's `torch.autocast` in the trainer's own state (adapters injected, checkpointing on), 3,289 leaf modules, 1,939 forward activations and 966 cotangents, every one BF16 except the router gate | exact, no non-BF16 activation |
| first backward: 396 LoRA-A tensors exactly zero and every LoRA-B family nonzero | exact, no missing or nonfinite gradient |
| first update: 396/396 LoRA-B tensors changed | exact |
| second backward: no missing, nonfinite, or zero gradients | exact |
| packed payload identity (one parameter of every type) and PLE table identity across the update | unchanged |
| loss output: no logits tensor and no router auxiliary loss in the training output | exact |

The Qwen3.5-MoE and DeepSeek V4 audits run load, configuration patches, the adapter injection check, the training-contract check, and the measured updates. This audit adds eight architecture-owned passes. Three read module metadata or payload samples and cost nothing measurable (`ple_table`, `qsa_indexer`, `frozen_mmq_patch`). The five that run forwards are:

| Pass | Seconds | Peak allocated above the resident model | What it does |
| --- | ---: | ---: | --- |
| `qsa_indexer_skip` | 1.801 | 81 MiB | installs the short circuit, requires the inventory and the exhaustiveness at the audited length, and runs the reference indexer twice to show that the mask it returns equals the causal-plus-padding mask, all-visible and right-padded |
| `qsa_attention_equivalence` | 2.143 | 271 MiB | installs the QSA attention, requires the twelve-layer inventory, and compares the patched forward against the kept reference forward on real activations, all-visible and right-padded |
| `fused_norm_equivalence` | 0.506 | 1600 MiB | installs Liger on the plain norms, FLA on the gated ones and the project kernel on the grouped ones, requires the 48/3/97/36 inventory, and checks all three contracts against autograd on an fp64 transcription of the module's own forward, including the gated norm's activation gradient |
| `hc_norm_equivalence` | 0.410 | 1490 MiB | installs the fused grouped RMSNorm, requires the 97-connection inventory, and compares its forward and `dX` against the same fp64 truth alongside the patched module against its reference |
| `autocast_dtype` | 10.522 | 3060 MiB | runs one forward and backward under `torch.autocast` with the adapters injected and checkpointing on, and requires every floating activation and cotangent outside the router gate to be BF16 |

The hyper-connection gate runs after the adapter injection, because that is what freezes the norm weight its predicate requires. The audit's `--profile-output` pass is the only heavy one, and it is now the Kineto trace in the phase table above.

The `autocast_dtype` pass exists because the trainer's execution mode is not the one the measurements use: `train_qwen4_exp.py` trains with `bf16=True`, so accelerate runs the model under `torch.autocast`, while the audit's own forward and backward run in BF16 without it. Autocast's FP32 list covers the reductions, so an op like `sum` can promote an activation and take the residual stream with it, and a custom op has no autocast policy to put the dtype back: whatever dtype reaches `torch_ggml_ops` is what its kernel validates. That is exactly how the PLE turned every layer after it FP32 and failed the packed projections' BF16 contract, so the gate now watches the trainer's mode rather than a mode nothing runs in.

## Completed optimizations

### Packed GGUF residency and the ordinary projections

The checkpoint remains compressed after loading: 737 `GgufLinear` modules, 448 frozen `GgufQuantizedParameter` parameters, and no logical base matrix materialized anywhere in the step.

`qwen4_exp_lora.py` registers the ordinary attention and shared-expert projections on the exported dense MMQ entry point (`fast_lora.packed_mmq_linear`), which runs the base projection and leaves the fused LoRA-B plus residual `addmm` path unchanged.

That is all 300 ordinary wrappers, in both directions, and the audit gates the count in both directions so a checkpoint that moves one projection to another base shows up as a mismatch instead of a silent slowdown.

The 36 GatedDeltaNet output projections run the native base as well, which took a convention change: `gdn_tiled_value_heads.py` wraps `get_gguf_conversion_mapping` so the load skips the eight value-head reorders the mapping would apply, and switches the GatedDeltaNet broadcast from `repeat_interleave` (grouped pairing) to `repeat` (tiled pairing) while that forward runs.

The two conventions compute the same function under a relabeling of the value heads, and every other operation on the value axis - the recurrent core, the gated norm, the depthwise conv, `A_log`, `dt_bias` and `in_proj_a`/`in_proj_b` - is per head, so nothing else moved.

The projection then consumes the packed columns in their own order and no input gather happens at all, and `torch-ggml-ops` added the exact `(n=2560, k=6144)` keys for Q3_K and IQ4_XS and `(n=2048, k=4096)` for the Qwen3.6 APEX-I-Mini checkpoint, 18 deployed keys over four HIP controls in both directions, including the IQ4_XS fragment-order forward at the training token count.

Measured at the time: the warm update went from 10.50 s to 10.21 s with the convention and to 10.14 s with the keys.

`in_proj_qkv` and `in_proj_z` carry only the tiled-to-grouped *row* reorder, which the loader applies to their packed rows as whole blocks, so their packed weights are in the model's ordinary layout and every type and matrix size they contain is an exact dense deployment key.

`configure_compiled_gguf_dequantize()` remains for `in_proj_a` and `in_proj_b`, two `[48, 2560]` frozen projections per layer that are below the granularity of a deployed key, and for the LM head's non-training evaluation path. `fast_lora.packed_mmq_weight` refuses any module that carries an `input_permutation`, which is now nothing in this model.

`ple.key_proj` is the one frozen packed projection that is neither adapter-wrapped nor part of the LM head, so `qwen4_exp_lora.py` installs the native forward on it through `ModulePatchSpec` and fails closed on the inventory, in the same shape as DeepSeek V4's fixed grouped projection. Its input is the n-gram embedding lookup, which is untrainable, so the native path needs the forward only.

### PLE table on disk, and the prefetch that pays for it

The PLE n-gram table is one embedding site at layer 1, 320,001,536 rows of 160 values in IQ4_NL, and it is 26.82 GiB of the checkpoint's 61.85 GiB for a forward that reads sixteen 90-byte rows per token. `ple_disk_residency.py` keeps that payload in the GGUF file and gathers only the rows a forward asks for, which `audit_qwen4_exp_training_step.py --ple-on-disk` and `train_qwen4_exp.py` under `QWEN4_PLE_ON_DISK=1` turn on:
- a hook on the GGUF quantizer's state dict replaces the tensor's entry with an empty placeholder, so `from_pretrained` never materializes it, and a second hook on the same class's `param_element_size` reports zero bytes for that parameter. Without the second hook `caching_allocator_warmup` pre-allocates the checkpoint's size and the pool keeps it for the run: measured, the pool sat at 61.99 GiB and the load's peak at 61.85 GiB before the hook, and 36.81 GiB and 36.67 GiB after it.
- the rows are gathered on the host into a pinned buffer and only those rows are uploaded: 90 bytes per row, about 2.2 MiB per forward, 2000x less than the table.
- the gathered payload is dequantized by the same `GgufQuantizedParameter.dequantize` kernel the resident path uses, so the values are identical to it by construction. The audit's file-sourced identity `sha256` equals the resident run's.
- the layer's recomputation under gradient checkpointing gathers nothing, because a payload is cached and matched by the row ids the layer itself computes, so a repeated lookup is free and a wrong prediction costs a read rather than a value.

The reader is the shape llama.cpp uses for its own lazy PLE tensors (`llama-lazy-reader.cpp`): sort the row ids, then let several threads issue one buffered `pread` per row through a shared descriptor with `POSIX_FADV_RANDOM` set. What a step pays is one 4 KiB page read per distinct row, and a 2048-token chunk asks for 32,768 rows, about 25,000 of them distinct and in distinct pages, because a page holds one useful row and only 1.5% of a step's pages repeat the previous step's (14% repeat any of the first 24). A shuffled loop therefore reads ~100 MB of pages per step for 2.2 MiB of payload, and a page cache does not help.

The alternatives were measured and rejected: `O_DIRECT` with 64 threads takes 4.20-4.33 s for those rows, cold and warm, because the file lives on btrfs with `compress=zstd:3` and its extents are `encoded`, so direct I/O cannot bypass the decompression and the per-request path serializes. One `pread` per row on one thread takes 4.2 s. mmap with `MADV_RANDOM` takes 4.3 s and with `MADV_NORMAL` 14.0 s. The buffered sorted multi-threaded reader takes 0.24-0.69 s.

llama.cpp agrees with the measurement: its lazy reader opens its files with `use_direct_io = false`, and its row reader carries an assert that a single-row read meets none of `O_DIRECT`'s alignment rules. Strata's `ple_reader.cpp` does use `O_DIRECT`, but with an 8-way row cache and asynchronous tickets, which is a design for inference, where one token reads a handful of rows and reuses them. Training reads 32,768 rows per step with 1.5% reuse, where a cache could only hold bytes nobody asks for again.

What pays for the reads in a training loop is `prefetch_ple_rows(model, input_ids)`. It holds the layer's n-gram arithmetic as numpy arrays (`NgramIdSource`), so the rows a batch will ask for can be resolved from host token ids, which a loop already has before the forward. A device tensor is refused, because copying one back would wait on work that is already queued. The trainer wiring is its data collator, wrapped so every batch starts its read on the host while the previous step's kernels are still on the device. Three steps of a shuffled loop, measured with the light profile (`--disk-ple --row-stride 7 --prefetch`):

| forward | resident table | on disk, no prefetch | on disk, prefetched |
| ---: | ---: | ---: | ---: |
| step 1 | 4.97 s | 5.18 s | 4.67 s |
| step 2 | 3.04 s | 3.24 s | 2.99 s |
| step 3 | 2.69 s | 4.30 s | 2.66 s |
| the three backwards | 6.70 / 6.45 / 6.47 s | 6.72 / 6.48 / 6.48 s | 6.76 / 6.56 / 6.57 s |

The prefetch removes the whole penalty on the forwards and leaves the backward untouched, because the recomputation is served from the cache. The counters agree: 3 prefetches, 89,046 rows, 3 hits and 0 prediction misses, with 0.43 s of reading of which 0.096 s landed on a forward's critical path and the rest overlapped the backward. The step's losses match the resident run's for the two steps the comparison can use (`3.1939`, `3.4499`). The third differs in its third decimal, which is the step after the first optimizer update through AITER's atomic kernels, and the PLE values themselves are row-verified.

### Routed experts: grouped MMQ base with AITER factors

`qwen4_exp_moe_lora.py` owns `QWEN4_EXP_EXPERTS_IMPLEMENTATION` and the `qwen3.8-learned` prior. Everything else is the shared implementation in `fast_moe_lora.py`:
- `prepare_expert_routing` / `finalize_expert_routing` (project Triton route gather and combine) and one full-cover `group_sizes` vector for all 512 experts, so no routed distribution is captured and no host synchronization is added.
- the frozen gate, up and down projections through `_base_grouped_pair` and `_base_grouped_linear`, the exported `grouped_mmq_pair` and `grouped_mmq` entry points. Gate and up share one packed pair deployment. Down has its own. The routed row counts this step issues are exactly the three deployed ones, and `torch-ggml-ops` fails closed on any other count.
- AITER `gmm` for the rank-4 factor families and AITER `ptgmm` for their gradients, through `_AiterGroupedMM`. The factor `lora_A` rows are rebuilt in backward from the routing index instead of retained, no base-weight gradient is produced, and no logical base matrix is materialized.

The AITER entries come from the `qwen3.8-learned` keys in `moe_gmm_configs.py`, keyed by the fitted 512-expert top-10 route law of this checkpoint and measured with the tuner in `tune_coefficient_prior_gmm.py`.

`docs/aiter_gmm_ptgmm_coefficient_prior_tuning.md` owns that protocol and its evidence: all 54 exact keys carry measured entries, 33 of them a confirmed improvement over their seed (median 1.088x, up to 1.560x), 6 reverted because they measured ~2% slower on the independent replay, and 15 kept their seed because no candidate beat it.

The measured cost of the whole routed-expert path, base projections and factors together, is 0.90 s per forward, 1.10 s of its own backward and 0.90 s recomputed.

The previous round's `_GenericGroupedProjection`, which dequantized the whole expert axis to BF16 and ran one dense grouped GEMM per projection, is gone with the generic checkpoint: it existed only because no grouped kernel covered `Q2_0`. The shared forward no longer takes pluggable base callables either, so `_base_grouped_pair` and `_base_grouped_linear` are the single integration point.

### Router selection on the shared streaming kernel

Qwen4-Exp's router is the same selection the other two architectures have: a BF16 gate projection rounded at the module boundary, then a top-k over the expert axis, then normalization over the selected values only.

`fast_moe_ranking.py` already owned that kernel for Qwen3.5-MoE and DeepSeek V4, so this project reused it instead of keeping a Qwen4-specific path, and the module's expert count and top-k size are now kernel parameters rather than 256-expert constants.

Qwen4-Exp is the widest geometry it serves: 2,560 hidden states, 512 experts, top-10, 98,304 rows per step over its 48 layers.

The old path materialized an FP32 softmax over all 512 experts and ranked it with a stable full argsort, which existed only to make the selection deterministic. Determinism is a real requirement, but only of *our* implementation: a checkpointed layer's replay has to save as many tensors as its forward did, or `torch.utils.checkpoint` rejects it with a tensor-count mismatch.

Reproducing `torch.topk`'s choice among tied experts is not a requirement, and no gate compares expert identities, so the streaming kernel resolves a tie to the lower expert index by construction and the audit compares selected weights with a tolerance plus the requirement that every selected expert scores at or above the kth threshold.

Measured on its own at the real shapes: the selection goes from 0.373 ms to 0.042 ms per call at 2,048 rows, 0.721 to 0.126 ms at 8,192 rows, and 3.77 to 0.509 ms at 32,768 rows, and the full router path (projection, selection, softmax over the ten, gather) from 0.66 to 0.47 ms per call at batch 1, 3.16 to 1.37 ms at batch 4, and 13.05 to 5.42 ms at batch 16, where the FP32 probability tensor and the argsort workspace also stop being materialized.

In the attribution the MoE block remainder went from 41.7/49.2/41.9 ms to 32.4/43.0/32.8 ms of forward, backward and recomputation, 25 ms of the update.

### Packed LM-head loss

`qwen4_exp_liger_loss.py` owns the Qwen4-Exp entry points and constants: hidden size 2560, `Q5_K` head, 256-row chunks. The shared `packed_liger_loss` module owns the calculation, and training never materializes the `[rows, 248320]` logits, the head's logical BF16 matrix, or an FP32 cross-entropy upcast. The backward decodes the frozen head's input Jacobian directly through the split-contraction `Q5_K` records, whose partial buffers `torch-ggml-ops` allocates in Python so a compile graph can plan them. `scoped_packed_causal_lm_loss` keeps evaluation on the head's own forward, so a labels-free call still returns logits.

### QSA attention and the indexer

Both QSA families run project kernels now.

At S=2048 the budget covers the whole chunk, `block_topk == max_complete_blocks`, so every complete block is selected and the selection is exhaustive for every query. The indexer's outputs reach the loss only through `topk` indices and its projections are frozen, so `qwen4_exp_indexer.py` replaces its forward with the identity mask and the audit proves the mask the reference would have produced equals the causal-plus-padding mask on real shapes, including a padded one.

That took the extra steps from 10.48/18.31 s to 3.47/9.83 s and 4.2 GiB off the peak.

`qwen4_exp_attention.py` then patches the twelve indexed layers onto `qwen4_exp_qsa_attention.py`, replacing the whole module forward rather than only the attention interface, which removes the indexer call and the `[B, 1, S, S]` mask combination with it. The patched forward reproduces the projections, q/k RMSNorm and partial RoPE, takes the per-sample key bound from the model's causal mask, calls `qwen4_exp_qsa_attention_autograd`, applies the sigmoid gate and hands the result to `o_proj`. The attention category went from 178.7 ms of forward and 1210.0 ms of backward to 69.4 ms and 381.1 ms with 70.9 ms recomputed, and `_qsa_dkdv_gluon_kernel` is the Gluon dK/dV owner that made the backward a further 12-15% cheaper than the Triton kernel.

The kernel contracts, the designs, the rejected alternatives and the tuning evidence live in `plan_qwen4_exp_qsa_forward.md`, `plan_qwen4_exp_qsa_backward.md` and `plan_qwen4_exp_qsa_kv_owner_rewrite.md`.

What this plan owns is the wiring: the twelve-layer inventory with a kept reference forward, the equivalence gate (relative RMSE `1.06e-4` all-visible and `1.05e-4` right-padded against SDPA, cosine `1.0` in both), the guard that sends anything outside the training contract back to the model's own path, and the measured BHSD-only layout contract - handing the kernels the projections' own BSHD layout is bit-identical and consistently 6% slower end to end, so `_validate_inputs` fails closed on it.

### Fused norms and hyper-connections

`qwen4_exp_liger_hc.py` replaces the grouped RMSNorm of the 97 hyper-connection sites with one program per (row, group) pair, a 4096-wide masked tile over the 2560-wide group and two warps: the forward reads bf16, accumulates `sum(x^2)` in fp32 registers, and applies `rsqrt` and `(1 + weight)` in fp32 before storing bf16, so the stream is read once and written once against the reference's five kernels and an fp32 copy. The backward holds the whole group in registers, so the mean its Jacobian needs is a register reduction and `dX` is one pass. Measured at the real shapes the grouped norm went from 3.82 ms to 0.62 ms forward and from 12.16 ms to 1.99 ms forward plus backward at batch 1, and the whole hyper-connection module from 20.4 ms to 9.8 ms, a 52% gain that is 53.6% and 53.9% at batches 4 and 16.

`qwen4_exp_fused_norms.py` owns the grouped kernel, which moved out of the hyper-connection module, and serves the other 87 norms: 48 plain through Liger (`offset=1.0`, `casting_mode="gemma"`, `in_place=False`), 36 gated through FLA's `LayerNormGatedFunction` (`is_rms_norm=True`, with the activation the checkpoint's config declares, which is sigmoid here rather than silu), and the 3 PLE grouped norms through the project kernel. The 97 hyper-connection norms are counted as skipped, because that module's forward owns them, and the inventory gate requires exactly 48/3/97/36. A grouped site keeps the module's own forward and falls back to it for anything outside the kernel's shape, dtype, row count or frozen-weight contract, and the audit requires the predicate to be true on the loaded model, so a silent fall back cannot pass.

Measured: the gated norms took the GatedDeltaNet family from 3.30 s to 2.80 s, the Q/K head norms took the attention category from 0.55 s to 0.45 s, the PLE norms took that layer from 51.8/97.7/49.1 ms to 43.0/64.8/39.6 ms of forward, backward and recomputation, and the update from 11.1 s to 10.45 s at the light profile's instrument, which is 10.52 s at the audit's own extra steps.

Peak allocation went from 67.8 GiB to 67.5 GiB.

The accuracy gate is fp64, and the truth comes from autograd on a transcription of the module's own expression rather than from a second transcription of the kernel's algebra. Both the kernel and the module sit on the bf16 store floor there: at weight scales 0, 0.1, 1.0 and 3.0 the kernel's `dX` is 1.66e-3 from fp64, the module's own `dX` is the same, and the two agree with each other to 1.5-1.8e-5.

On the loaded checkpoint the audit compares the patched module against its kept reference at a loose RMSE (3e-2) with a tight cosine, while the kernel itself is gated at 5e-3 against fp64.

That truth change is a finding worth keeping: the earlier gate repeated the kernel's backward algebra, and the grouped backward did leave the scale vector out of the mean, so `dX` was 2.6% off at weight scale 3.0 while every transcription-based check agreed with it, and the module-level comparison improved from 3.3e-3 to 8.7e-5 of relative RMSE once the term moved inside the mean.

### GatedDeltaNet: projections, value-head order, and kernels

The family's three projections are done: `in_proj_qkv` and `in_proj_z` carry only the row reorder, `out_proj` runs the native base under the tiled value-head convention, and all of it is the ordinary-MMQ work described above.

The FLA autotune table is its other half, because the Qwen4-Exp geometry (16 key heads, 48 value heads, sigmoid output gate) differs from Qwen3.5, so `fla_tuning.configure_qwen35_fla` does not apply.

The kernels this project owns replace three FLA kernels at their call sites:
- `gdn_bwd_dhu.py` fixes FLA's state-gradient walk, whose decayed query operand was promoted to FP32 so its dot ran as VALU FMA rather than WMMA. The fix is one cast of the decayed operand back to BF16, with `BT 64`, `BK 64`, `BV 32`, 8 warps and 2 stages: -6.02 ms per layer per step.
- `gdn_bwd_dqkwg.py` drives FLA's `chunk_bwd_kernel_dqkwg` with `BK=128, BV=32` from a launcher of our own, which removes the fourfold redundancy of the collapsed 32-wide tiles: -2.53 ms per layer per step, and 22% of that kernel against 8% for the module-wide patch.
- `gdn_wu_recompute.py` replaces `recompute_w_u_fwd_kernel` with its own kernel at 0.551x of the reference cost: -2.94 ms per layer per step over its two calls, since the recomputation runs twice a step and this serves both.

All three are installed next to the tuning table in both trainers and both audits. They are drop-ins with a kernel-level reference test each (`test_gdn_bwd_dhu.py`, `test_gdn_bwd_dqkwg.py`, `test_gdn_wu_recompute.py`), so they need no inventory gate of their own, and the audit records the installed geometries.

Measured at the model level: the update goes from 10.14 s to 9.31 s, -0.83 s or 8.2%, with the allocation unchanged at 65.97/68.99 GiB. The per-kernel measurements, the roofline, the rejected alternatives and what remains are in `plan_qwen4_exp_gdn_forward.md` and `plan_qwen4_exp_gdn_backward.md`.

One more change is written, tested and deliberately disconnected: `gdn_fused_prep.py` fuses the whole preparation of the chunked core (the depthwise convolution, its SiLU, the split and reshape, the tiled head broadcast, FLA's L2 normalization and the gating) into one forward kernel and four backward ones.

Its kernels are worth 0.35 s of the update - the first warm update measures 8.96 s against 9.31 s - but it allocates 6.3 GiB more, and that run's next update crawls to 13.41 s where the run without it repeats 9.31 s.

The 6.3 GiB is the op's saved state: `_GdnPrepFunction` keeps `mixed` and its five outputs for its backward, about 120 MB per layer, and gradient checkpointing means that state is produced during the recomputation and held until the backward consumes it, so all 48 layers' worth is live at the peak. The eager chain it replaces saves nothing comparable because its intermediates are recomputed rather than kept.

The fix is to let the backward recompute what it can from the four shifted loads and keep only what it cannot, which is the difference between 120 MB and a few megabytes per layer. `plan_qwen4_exp_gdn_forward.md` carries the two runs and the evidence.

### FLA autotune table

The kernel workers ship a table for this geometry. `fla_tuning.configure_qwen4_exp_fla()` preloads 20 exact Triton cache keys across 11 kernels, tuned by `~/tmp/test_no_unsloth/tune_gdn_fla.py`, which runs one real layer, records every autotuner call with its grid and arguments, screens all candidate configs on those arguments, and re-measures the finalists with 15 samples per config.

Winners are log-space medians of the per-batch medians and only displace the config Triton had selected when they win by more than 2%, following `~/evotensile/docs/noisy_measurements.md`.

Measured gains over the configs Triton had chosen: `recompute_w_u_fwd_kernel` 1.677x, `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` 1.578x (the incumbent used `num_stages: 2`, the winner `1`), `chunk_gated_delta_rule_fwd_kkt_solve_kernel` 1.429x, `chunk_fwd_kernel_o` 1.211x (the incumbent pooled at `BK: 64, BV: 64` while `BK: 128, BV: 128, num_warps: 8` wins at every batch), and `chunk_local_cumsum_scalar_kernel` 1.222x at batch 1.

The remaining keys kept the incumbent because nothing beat it by 2%.

That is roughly 14 ms per linear-attention layer at batch 1, about 0.5 s of the update, and proportionally more at batches 4 and 16.

Both architectures share one mechanism: a table of `(kernel name, exact Triton cache key, config)` entries injected from `Autotuner.run`.

That replaced Qwen3.5's module-and-attribute addressing, which cannot reach these kernels because the hub `kernels` package loads its own copies of them lazily inside a loader closure, so no importable module exposes the `Autotuner` objects and they exist only after the layer is loaded.

`configure_qwen4_exp_fla()` is called by `train_qwen4_exp.py` next to the dequantizer configuration and by the audit in its static configuration block, and `require_complete_qwen4_exp_fla()` fails the audit when the step applied none of the entries.

A single run uses the keys of its own batch, so partial coverage is expected: with the three replacements installed, the accepted run applied 10 of the 20 entries across 9 kernels, and the table's `chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64` and `chunk_bwd_kernel_dqkwg` entries are now inert because those kernels are ours. They stay as the A/B baseline the replacements were measured against.

Nothing else in the step autotunes. Liger carries `@triton.autotune` only in `ops/dyt.py`, `ops/fused_moe_kernels.py`, `ops/grpo_loss.py`, `ops/mlp.py` and `ops/experimental/mm_int8int2.py`, none of which runs here, so its chunked cross-entropy and RMSNorm use fixed launch parameters.

AITER's gmm and ptgmm and the MMQ bundle are HIP with the tables above, and `bitsandbytes` AdamW, SDPA and the CK flash path are not Triton.

The compiled GGUF dequantizer is Inductor's `max-autotune-no-cudagraphs`, which enables GEMM template autotuning while `max_autotune_pointwise` stays False and the dequantize graph is elementwise, so what it costs is compilation: about 14 specializations in the first forward, and artifacts in `/tmp/torchinductor_wd` that are lost on reboot.

### llama.cpp inference with the trained adapter

Every tensor this plan adapts has both an inference-side LoRA site in `~/llama.cpp` and an export path in `convert_lora_to_gguf.py`, so a trained adapter applies to the GGUF checkpoints as well as to this project's trainer.

| LoRA family (HF) | GGUF tensor | llama.cpp apply site | Adapter export |
| --- | --- | --- | --- |
| `linear_attn.in_proj_qkv` | `blk.N.attn_qkv.weight` | `build_lora_mm` in the GatedDeltaNet block | 1 pair |
| `linear_attn.in_proj_z` | `blk.N.attn_gate.weight` | `build_lora_mm` in the GatedDeltaNet block | 1 pair |
| `linear_attn.out_proj` | `blk.N.ssm_out.weight` | `build_lora_mm` at the mixer output | 1 pair |
| `self_attn.q_proj` | `blk.N.attn_q.weight` | `build_lora_mm` in `build_layer_attn` (holds q and the interleaved gate) | 1 pair |
| `self_attn.k_proj` | `blk.N.attn_k.weight` | `build_lora_mm` in `build_layer_attn` | 1 pair |
| `self_attn.v_proj` | `blk.N.attn_v.weight` | `build_lora_mm` in `build_layer_attn` | 1 pair |
| `self_attn.o_proj` | `blk.N.attn_output.weight` | `build_lora_mm` in `build_layer_attn` | 1 pair |
| `mlp.shared_expert.gate_proj` | `blk.N.ffn_gate_shexp.weight` | `build_ffn` gate branch | 1 pair |
| `mlp.shared_expert.up_proj` | `blk.N.ffn_up_shexp.weight` | `build_ffn` up branch | 1 pair |
| `mlp.shared_expert.down_proj` | `blk.N.ffn_down_shexp.weight` | `build_ffn` down branch | 1 pair |
| `mlp.experts` gate + up | `blk.N.ffn_gate_exps.weight` and `blk.N.ffn_up_exps.weight` | `build_lora_mm_id` in `build_moe_ffn`, one call per target | 1 trained pair, split into 2 |
| `mlp.experts` down | `blk.N.ffn_down_exps.weight` | `build_lora_mm_id` in `build_moe_ffn` | 1 pair |

`build_lora_mm` and `build_lora_mm_id` are the only places llama.cpp applies an adapter, so a family is inference-capable exactly when its build site uses one of them. Both take 3-D per-expert factors for the expert tensors, so the routed families keep their per-expert planes through export. The MoE export needed a converter branch that did not exist: `convert_lora_to_gguf.py` already handled the same fused gate/up PEFT convention for Qwen3.5 and DeepSeek-V4, but not for `QWEN4EXP`, and the fallback path misrouted the fused pair to `ffn_down_exps` and rejected the `lora_A_down` names. This project added the `QWEN4EXP` arch to that mapping plus `_split_qwen4_exp_gate_up`, which follows the DeepSeek-V4 treatment because this checkpoint stores gate and up per expert instead of a fused `ffn_gate_up_exps`: the two targets share the A factor, so only B splits on its output dimension, gate rows first.

The assumption matches every local GGUF of this model (APEX-I-Nano, IQ4_NL, Q8_0, GSQ-RCO-Q2_0 all carry `ffn_gate_exps` + `ffn_up_exps`) and the base conversion, which splits `mlp.experts.gate_up_proj` the same way.

The export was verified against a local checkpoint by converting a synthetic rank-4 adapter with this plan's names and shapes, once per family and once over all 348 wrappers, and running `llama-perplexity` with it loaded: every converted pair satisfies the load-time rule in `llama_adapter_lora_init` (`base.ne[0] == a.ne[0]`, `base.ne[1] == b.ne[1]`, `a.ne[1] == b.ne[0]`, plus the expert axis for 3-D factors), and every family reaches the graph, which shows as a perplexity far from the no-adapter baseline.

The factors are random, so the absolute numbers carry no information. The probe driver is `~/tmp/test_no_unsloth/lora_probe.py` with `lora_probe_ppl.sh`, and it re-runs against whichever checkpoint is active.

Two caveats are part of the record: the verification runs use `-fa off` because of an unrelated, pre-existing assert in the sparse-QSA HIP path (commit `656cc6a41`), and the export assumes the base GGUF has separate gate/up expert tensors, so a base with the fused `ffn_gate_up_exps` would need the single fused pair instead, which the generic `is_gate_up` rename already produces.

### Supporting changes

- `qwen4_exp_lora.py`: the frozen packed-projection patch (`configure_qwen4_exp_frozen_mmq`, `require_complete_qwen4_exp_frozen_mmq`) and the native-MMQ inventory constant. `register_qwen4_exp_adapters` no longer takes a blanket generic flag.
- `fast_lora.py`: the base projection is one shared boundary, `packed_mmq_linear`, called by the LoRA wrapper and by the frozen patch, so the kernel call, the bias and the BF16 rule have one definition. `packed_mmq_weight` gained the `input_permutation` capability check, and `register_fast_lora` lost the `generic_packed` parameter it needed only while no MMQ kernels existed.
- `qwen4_exp_moe_lora.py`: rewritten on the shared grouped-MMQ base. The generic dequantize-then-GEMM backend, its two `Function` methods and the module-level projection helpers are gone.
- `fast_moe_lora.py`: `gguf_aiter_lora_forward` lost the pluggable `base_pair`/`base_linear` parameters, which existed only for that removed backend, and the module docstring now names Qwen4-Exp next to Qwen3.5-MoE and DeepSeek V4.
- `training_audit.py`: `validate_loss_output` and `complete_training_update` no longer take `expect_logits`, because the packed loss removed the only caller that materialized logits, and `memory_snapshot` now reports the machine's unified pool (`MemFree`, `MemAvailable`, `Cached`, `Shmem`, swap) next to the allocator's own numbers.
- `training_profiler.py`: `profile_warmed_training_update` gained `capture_memory`, because this checkpoint traces without the per-allocation memory events.
- `qwen4_exp_profiler.py`: categorizes a wrapper by its capability, so the profiler's categories follow what a module actually runs.
- `~/transformers/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py` and its `modular_qwen4_exp.py` source: the PLE gate's reduction carries its dtype (`sum(..., dtype=key_normed.dtype)`). Autocast runs `sum` in FP32, so the gate was computed in FP32 and promoted the PLE's output, and with it the residual stream of every layer after the PLE site, to FP32 on the trainer's `bf16=True` path. The change is bit-identical without autocast, so the measurements are unaffected, and it follows the model's own convention: the router does the same dance with `softmax(..., dtype=torch.float)` and then `.to(router_logits.dtype)`.
- `~/transformers/src/transformers/integrations/gguf/reader.py`: `_GgufFileReader` keeps its pinned staging below `PINNED_STAGING_LIMIT_BYTES` (1 GiB) and stages larger reads in a pageable buffer.

## Remaining work

Ordered by the measured cost in the attribution table:
- Routed experts, 2.90 s of the update: the largest single block, and the factor side is already tuned, so what is left is kernel work inside `torch-ggml-ops`. The pair forward for `Q2_0` currently launches twice because no fused HIP body exists for it, and the 0.90 s of recomputation is the same forward again. Measure any change at the complete-update boundary, and keep the current live footprint of a few tens of MiB.
- GatedDeltaNet fused preparation, 0.35 s of step time for 6.3 GiB: reconnect it only with the backward recomputing instead of saving, then re-measure both the update and the allocation. The isolated kernel numbers are not in doubt. What fails is the memory trade.
- Residual-stream arithmetic, 0.26 s (88.2 ms of forward and 174.4 ms of backward) for the two injection pairs per layer: `hidden_states.unsqueeze(-2) * injection_weights.unsqueeze(-1)` and `hyper_input + injection.flatten(-2)` at every hyper-connection boundary. One fused kernel could take it to roughly 0.13 s by writing the new stream once instead of three passes and producing all three gradients in one backward pass. Measured in isolation at the real shape and dtype: 0.91 ms of forward and 2.59 ms of forward plus backward per pair against 87 ms and 248 ms for the step's 96 pairs.
- Hyper-connection elementwise chain, part of the family's 1.34 s: the two BF16 projections are 26.8 GFLOP per application and 2.6 TFLOP over a step, and a fused down/SiLU/up/sigmoid/mean/injection kernel could keep the 320-wide intermediate on chip and drop the ~0.3 s of elementwise traffic, but the GEMMs remain the floor of this family.
- Under 0.25 s each and not worth a round of their own: the PLE layer, the packed LM head, the MoE block remainder and the shared expert. What is no longer on any list: the QSA indexer, the QSA attention in both directions, the LM head's logits path, the attention and shared-expert projections, the GatedDeltaNet projections, and the FLA autotune table.
- Memory, for the machines that need it: the PLE table moves to the file for 26.82 GiB, which takes this step to 39.15 GiB allocated and a 40.69 GiB peak against 65.97 and 67.51. What is left resident is the checkpoint itself, 35.10 GiB, of which 31.6 GiB is the routed experts: a step reads all 512 experts in full, so they cannot go to the file the way the PLE table can, and a page cache holding them is the same RAM again. Adapters, gradients and the 8-bit optimizer state are 2.9 GiB together and already minimal, and the allocator knobs are excluded by the contract.
- PLE disk residency: the mode is off by default and the prefetch that pays for its reads is wired only in the trainer's collator. A loop that accumulates several batches at once would want the prefetch ring widened (it holds four payloads), and a loop that can look further ahead than one batch would want the reads started earlier.
- Physical B4 and B16 full updates: the next milestone. The kernels, the AITER tables and the attention padding path already cover both, and the audit currently accepts batch 1 only.
- Do not reopen a standalone kernel without complete-boundary evidence, and do not accept isolated throughput that regresses complete-update time, memory, correctness, layout ownership or model semantics.

## Known limits

- Sequence length 2048 and physical batches 1, 4 and 16 are the whole supported range. Anything else - a different length, a different batch, a cache, a `generate()` call, a mask layout the collator does not produce - runs the model's own forward or is rejected, and the patches say so in their guards.
- The patches belong to the training wiring, not to the model, so any driver that measures this step has to apply `configure_qwen4_exp_indexer_fast_path`, `configure_qwen4_exp_qsa_attention`, `configure_qwen4_exp_fused_norms`, `configure_qwen4_exp_hc_norm` and `configure_fast_moe_ranking`, plus the dequantizer, value-head and GatedDeltaNet installs. The light profile did not at first and reported the indexer at 7.0 s while the audit's own steps had already dropped by 13 s.
- The QSA attention patch needs contiguous q, k and v, so it copies the three transposed views, 30 MB per layer. The partial-RoPE concat in the model's own path materializes query and key as fresh contiguous tensors anyway, so only `value_states` is really copied, and that copy is 0.026 ms per layer. It reads the key bound from the model's causal mask, whose last row it requires to be an ordinary causal row over the whole sequence, which is what the collator's `valid_tokens` mask produces.
- The `qwen3.8-learned` AITER entries exist for the three physical batches' routed row counts and the shapes listed in `docs/aiter_gmm_ptgmm_coefficient_prior_tuning.md`. Other routed row counts fail closed, and so does any dense projection whose type or matrix size leaves the deployment tables.
- `in_proj_a` and `in_proj_b` stay on the compiled dequantizer on purpose: two `[48, 2560]` packed weights per layer are below the granularity of a deployed key. Neither is adapter-wrapped, so this costs the step no trainable path.
- 1.18 GiB of hyper-connection weights are BF16 rather than quantized, so they are read at BF16 width in both directions. That is a bandwidth cost this recipe chose and keeps.
- Nothing in the norm family is unserved: 97 grouped norms belong to the hyper-connection forward, three to the project grouped kernel, 48 plain to Liger and 36 gated to FLA. Of the 100 grouped norms, the 97 hyper-connection ones keep that module's forward because it needs the normalized stream twice.
- Text only. The vision tower is a separate `mmproj` file and is not loaded.
- The machine's unified memory is the binding constraint for headroom, not for correctness: the GPU has 1 GiB of dedicated VRAM and the rest of its memory is the system RAM itself, so the resident model and everything else share one pool. With the reader's pinned staging bounded, an update leaves 46 GiB of `MemAvailable` and touches no swap, and 73 GiB with the PLE table on disk.
- The GatedDeltaNet fused preparation is implemented, tested and not wired, for the memory reason above.
- The PLE table on disk costs what the SSD charges for random 4 KiB pages. A shuffled training loop reads about 100 MB of pages per step for 2.2 MiB of payload, 0.2-1.6 s of forward time when the reads are not prefetched, and a page cache does not help because only 1.5% of a step's pages repeat the previous step's. `prefetch_ple_rows()` takes that off the critical path, and it needs the batch's token ids on the host, which is why the wiring is the data collator. A loop that cannot offer host ids should leave the mode off. The audit reuses one batch, so its own numbers for the mode are warm ones.

## Measurement protocol

- `gguf_mmap_policy="pread"` with the reader's bounded pinned staging is part of the measurement environment, not a tuning knob. No allocator configuration is set.
- Acceptance is the audit's gate set above on a real batch, not a single loss value.
- The warm update is the reference: the measured updates after the first agree to 3 ms. The first update of a fresh process is 1.8 s slower because it carries the compilation of the dequantizer over its ~14 shape and grad-mode specializations, and the 1.5 GiB of 8-bit optimizer state is created on that first step. Run the audit with `--max-steps 3` and compare its `extra_steps`, or use `~/tmp/test_no_unsloth/qwen4_light_profile.py`, whose per-step attribution shows the same numbers without the audit's gates. That driver applies the same configuration calls as the audit and has to be kept in step with them: a driver missing the GatedDeltaNet installs measures the FLA path, and one missing the attention wiring reports the attention category as masked SDPA, 178.7 ms of forward and 1210.0 ms of backward, instead of 69.4 ms and 381.1 ms.
- Attribution uses the same shared mechanism as the other two audits, `training_profiler.profile_warmed_training_update`, whose module ranges carry the same category vocabulary, and its Kineto path now completes on this checkpoint. The per-category numbers in this document come from `~/tmp/test_no_unsloth/qwen4_light_profile.py`, which measures the same ranges with CUDA events, so a capture's own overhead cannot distort them. Its self time is a module's inclusive span minus its categorized children's spans, taken from the measured event windows rather than from the order the hooks fired, which is what keeps a checkpointed layer's recomputation in its own bucket. The forward, backward and recomputed buckets then partition the step: 4.11 s plus 2.37 s against a 6.48 s backward wall in the current run.
- The audit's measured forward and backward run in BF16 without autocast, so its `autocast_dtype` gate is what covers the trainer's `bf16=True` mode: it runs the same update under `torch.autocast` with the adapters injected and checkpointing on and requires one activation dtype. Keep that gate in step with the trainer, because a promotion it does not see is a promotion that fails a kernel's own validation at run time.
- Layout probes: `~/tmp/test_no_unsloth/profile_qsa_attention_wiring.py` breaks one wired attention layer's forward into its RMSNorms, RoPE, copies, kernels and gate, and `~/tmp/test_no_unsloth/bench_qsa_layouts.py` times the QSA kernels under BHSD and BSHD and requires them to agree elementwise. The second is what keeps the BHSD-only contract a measurement rather than a preference.
- Component benchmarks use the real packed payloads and the real routed row counts. `~/tmp/test_no_unsloth/qwen4_gmm_bench.py` is the driver for the decode and `gmm` numbers, and `~/tmp/test_no_unsloth/qwen4_quant_mix.py` prints the per-role quantization mix of a checkpoint.
- PLE disk residency is measured three ways. `~/tmp/test_no_unsloth/ple_row_probe.py` and `ple_gather_probe.py` time the read paths (one `pread` per row, sorted multi-threaded `pread`, mmap with and without advice, `O_DIRECT` at two block sizes) at the real row count and require every path to return identical bytes. `ple_pattern_probe.py` computes the real n-gram ids of dataset rows and reports distinct rows, distinct pages and cross-step reuse. `~/tmp/test_no_unsloth/qwen4_light_profile.py --disk-ple [--prefetch] [--row-stride N]` measures the step itself, where `--row-stride` is what makes it a shuffled loop rather than one repeated batch, and the counters in its report say whether the reads were prefetched or paid for on a forward's critical path.
- Reports, profiles, probes and throwaway scripts for this project live in `~/tmp/test_no_unsloth/`, not in the repository.

## Code ownership

- Training: `train_qwen4_exp.py`.
- Audit: `audit_qwen4_exp_training_step.py`, `qwen4_exp_profiler.py`, `training_profiler.py`, `training_audit.py`.
- Adapters: `qwen4_exp_lora.py` (target pattern, registration, frozen packed-projection patch), `qwen4_exp_moe_lora.py` (expert backend registration and the route prior), `fast_lora.py` and `fast_moe_lora.py` (shared ordinary and expert backends).
- QSA: `qwen4_exp_attention.py` (attention wiring), `qwen4_exp_indexer.py` (exhaustive-selection skip), `qwen4_exp_qsa_attention.py` (forward and backward kernels), `qwen4_exp_qsa_gluon.py` (the Gluon dK/dV owner). The kernel plans are `plan_qwen4_exp_qsa_forward.md`, `plan_qwen4_exp_qsa_backward.md` and `plan_qwen4_exp_qsa_kv_owner_rewrite.md`.
- GatedDeltaNet: `gdn_tiled_value_heads.py` (loader and broadcast patches, plus the gate that requires them), `gdn_bwd_dhu.py` (the walk), `gdn_bwd_dqkwg.py` (the widened `dqkwg` launcher), `gdn_wu_recompute.py` (the W/U recomputation), `gdn_fused_prep.py` (the fused preparation, not wired) and `fla_tuning.py` (both autotune tables). The kernel plans are `plan_qwen4_exp_gdn_forward.md` and `plan_qwen4_exp_gdn_backward.md`.
- Hyper-connections: `qwen4_exp_liger_hc.py` (the module patch and its fused forward).
- Norms: `qwen4_exp_fused_norms.py` (the grouped kernel, Liger for the plain norms, FLA for the gated ones, and the inventory gate).
- PLE table: `ple_disk_residency.py` (the two GGUF quantizer hooks, the row reader, the n-gram id arithmetic and the prefetch), with `test_ple_disk_residency.py` covering the identity of the gathered bytes against the file, the patched forward against a resident payload, the id arithmetic against the real layer, the prefetch's cache, and its failure modes.
- LM head: `qwen4_exp_liger_loss.py` (Qwen4 constants and the forward patch), `packed_liger_loss.py` (shared calculation).
- Routing: `fast_moe_ranking.py` (the streaming selection for all three architectures), `fast_moe_routing.py`.
- Tests: `test_gdn_tiled_value_heads.py`, `test_gdn_bwd_dhu.py`, `test_gdn_bwd_dqkwg.py`, `test_gdn_wu_recompute.py` and `test_gdn_fused_prep.py` for the GatedDeltaNet work, `test_qwen4_exp_lora.py` for the frozen patch and the capability gate, `test_qwen4_exp_qsa_attention.py` for the attention kernels, `test_qwen4_exp_indexer.py`, `test_qwen4_exp_liger_hc.py` and `test_qwen4_exp_fused_norms.py` for the norm family, plus the shared `test_fast_lora.py`, `test_fast_moe_lora.py`, `test_fast_moe_ranking.py` (which carries the Qwen4-Exp router cases) and `test_module_patching.py`. `test_fast_lora.py` pins the native base for every recurrent projection, including `out_proj`, on real payloads.
- Inference export: `~/llama.cpp/convert_lora_to_gguf.py` (QWEN4EXP MoE split), with the probe driver `~/tmp/test_no_unsloth/lora_probe.py` and `lora_probe_ppl.sh`.
- GEMM configuration: `moe_gmm_configs.py`, `expert_distribution_prior.py`.
- Quantization support: `~/transformers/src/transformers/integrations/gguf/dequant.py` (`Q2_0`), with its test in `~/transformers/tests/quantization/ggml/test_gguf_integration.py` and its checkpoint-level verification in `~/tmp/test_no_unsloth/qwen4_q2_0_verify.py`.
- Kernels: `~/torch-ggml-ops` owns the deployment tables, the launch paths and the public entry points. `docs/kernel_bundle.md` there documents them.
- Checkpoint bindings live in `~/transformers/`. This project does not patch installed PEFT or Transformers classes.
