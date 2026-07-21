# Qwen3.5 MoE Low-VRAM LoRA Training With GGUF Base Model

GGUF is going to replace bitsandbytes as the base model format for low-VRAM LoRA training. It's possible to train Qwen3.6-35B-A3B on 16 GiB VRAM with no CPU offload, and it's more than enough to train Qwen3.5-122B-A10B on 64 GiB VRAM, and Qwen3.5-397B-A17B on 192 GiB VRAM. I've tested it on Strix Halo (gfx1151). Currently all kernel parameters in this repo are tuned for Qwen3.6-35B-A3B on Strix Halo. It should not be too hard to port to other GPUs.

Things involved in the training:
- Qwen3.6-35B-A3B with APEX-I-Mini quantization that only takes 13.3 GiB, see https://huggingface.co/mudler/Qwen3.6-35B-A3B-APEX-GGUF/blob/main/Qwen3.6-35B-A3B-APEX-I-Mini.gguf
- Usual training loop with transformers 5 and PEFT
- Transformers with GGUF quantizer, see https://github.com/woct0rdho/transformers/tree/gguf . I'm tracking this in https://github.com/huggingface/transformers/issues/40070 . If I could not merge it into transformers in the end, I'll reimplement it as some monkey patches in this repo
- torch.compile added to GGUF dequant to save VRAM
- Tuned GEMM, see https://github.com/ROCm/rocm-libraries/pull/9385
- MMQ and grouped MMQ like llama.cpp, see https://github.com/woct0rdho/torch-ggml-ops
- Fast LoRA bwd formula like Unsloth for linear layer and MoE layer
- AITER gmm/ptgmm Triton kernels with tuned configs for non-quantized MoE LoRA
- MoE routing like OpenAI triton-kernels
- AITER FlashAttention Triton kernel with tuned configs and the bugfix https://github.com/ROCm/aiter/issues/3551
- GatedDeltaNet with FLA and causal-conv1d, which are automatically chosen by transformers if installed. I've added tuned configs but I've not yet fully considered how to optimize this
- RMSNorm from Liger Kernel
- Chunked cross entropy loss like Liger Kernel, which works with MMQ
- Autoregressive decoding cache and load balancing loss disabled to save VRAM
- Non-reentrant gradient checkpointing
- bitsandbytes AdamW 8-bit optimizer

Other notes:
- Unsloth gradient checkpointing provides fast async CPU-GPU copy. You need it if you actually do CPU offload. But on Strix Halo with unified memory you should just use the usual gradient checkpointing
- It does not affect loading GGUF models, but when loading safetensors models on Strix Halo, you need to patch transformers so it passes `backend="pread"` to safetensors, see https://github.com/safetensors/safetensors/pull/728
