# Low-VRAM LoRA Training With GGUF Base Model

Open weight AI is like open source software. Users not only run the weights, but also modify the weights. It matters to develop training framework for local hardware.

GGUF is going to replace bitsandbytes as the base model format for low-VRAM LoRA training. I've tried to train, with no CPU offload:
- Qwen3.6-35B-A3B in 16 GiB VRAM (implying it's more than enough to train Qwen3.5-122B-A10B in 64 GiB, and Qwen3.5-397B-A17B in 192 GiB)
- DeepSeek-V4-Flash (284B-A13B) in 90 GiB VRAM
- Qwen3.8-Flash-Next (125B-A6B + 51B engram) in 40 GiB VRAM

Currently all kernels and parameters in this repo are tuned for Strix Halo. More work is needed to support other GPUs.

Things involved in the training:
- Usual training loop with transformers 5 and PEFT
- Transformers with GGUF quantizer, see https://github.com/woct0rdho/transformers/tree/gguf . I'm tracking this in https://github.com/huggingface/transformers/issues/40070 . If I could not merge it into transformers in the end, I'll reimplement it as some monkey patches in this repo
- Tuned GEMM, see https://github.com/ROCm/rocm-libraries/pull/9385
- MMQ and grouped MMQ like llama.cpp, see https://github.com/woct0rdho/torch-ggml-ops
- Fast LoRA bwd formula like Unsloth for linear layer and MoE layer
- AITER gmm/ptgmm Triton kernels with tuned configs for non-quantized MoE LoRA
- MoE routing like OpenAI triton-kernels
- RMSNorm from Liger Kernel
- Chunked cross entropy loss like Liger Kernel, which works with MMQ
- Autoregressive decoding cache and load balancing loss disabled to save VRAM
- Non-reentrant gradient checkpointing
- bitsandbytes AdamW 8-bit optimizer

Qwen3.5-specific:
- APEX-I-Mini quantization that only takes 13.3 GiB, see https://huggingface.co/mudler/Qwen3.6-35B-A3B-APEX-GGUF/blob/main/Qwen3.6-35B-A3B-APEX-I-Mini.gguf
- AITER FlashAttention Triton kernel with tuned configs and the bugfix https://github.com/ROCm/aiter/issues/3551
- GatedDeltaNet with MMQ, FLA, causal-conv1d, and custom Triton kernels, including bwd

DeepSeek-specific:
- IQ2_XXS quantization that only takes 81 GiB, see https://huggingface.co/antirez/deepseek-v4-gguf/blob/main/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
- Sliding attention, CSA, HCA, mHC with Triton kernels, including bwd

Qwen4-Exp-specific:
- GSQ-RCO Q2_0 quantization that only takes 35 GiB VRAM + 27 GiB engram, see https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/main/Q2_0
- QSA with Triton kernels, including bwd
- PLE on disk with prefetch

Other notes:
- Unsloth gradient checkpointing provides fast async CPU-GPU copy. You need it if you actually do CPU offload. But on Strix Halo with unified memory you should just use the usual gradient checkpointing
- When loading large models on Strix Halo, you need pread, see https://github.com/safetensors/safetensors/pull/728
- When using transformers with GGUF quantizer, you need torch.compile on the GGUF dequant function to save VRAM
- When inferencing the model with LoRA in llama.cpp, currently llama.cpp does not have a fast LoRA kernel. I've made one, see https://github.com/woct0rdho/llama.cpp/commit/36e9f19a3058cbdc86e824293c0c69ca02ad03ea
