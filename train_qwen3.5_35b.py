#!/usr/bin/env python3

import os

os.environ["TORCH_LOGS"] = "recompiles"
os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

from pathlib import Path

import torch
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

from attention_aiter_tuning import configure_qwen35_flash_attention_2
from fast_lora import register_fast_lora
from fast_moe_lora import register_fast_moe_lora
from fla_tuning import configure_qwen35_fla
from gguf_dequant_compile import configure_compiled_gguf_dequantize
from gguf_liger_loss import apply_gguf_liger_fused_linear_cross_entropy

script_dir = Path(__file__).resolve().parent


# I usually preprocess the dataset into chunks with fixed length. You may change this with your dataset
def fixed_length_lm_collator(examples):
    batch = default_data_collator(examples)
    input_ids = batch["input_ids"].long()
    num_tokens = batch.pop("num_tokens").long()
    positions = torch.arange(input_ids.shape[1]).unsqueeze(0)
    valid_tokens = positions < num_tokens.unsqueeze(1)

    batch["input_ids"] = input_ids
    batch["attention_mask"] = valid_tokens.long()
    batch["labels"] = input_ids.masked_fill(~valid_tokens, -100)
    return batch


def main():
    model_dir = Path.home() / "models/qwen3.6"
    gguf_file = "Qwen3.6-35B-A3B-APEX-I-Mini.gguf"
    tokenizer_id = "Qwen/Qwen3.5-35B-A3B"
    dataset_dir = script_dir / "data_tokenized_qwen3.5"
    output_dir = script_dir / "out_qwen36_35b"
    random_seed = 19260817

    set_seed(random_seed)

    configure_compiled_gguf_dequantize()
    configure_qwen35_flash_attention_2()
    configure_qwen35_fla()

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        gguf_file=gguf_file,
        gguf_mmap_policy="release",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": "cuda:0"},
    )

    # Autoregressive decoding cache is not needed in training
    model.config.use_cache = False

    # Disable load balancing loss to save VRAM
    model.config.output_router_logits = False
    model.config.router_aux_loss_coef = 0.0

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        # It's possible to create a LoRA on the routing gate, but this may make the training unstable
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "down_proj",
            "gate_proj",
            "up_proj",
            "in_proj_qkv",
            "in_proj_z",
            "out_proj",
            "experts",
        ],
        r=4,
        lora_alpha=4,
        use_rslora=False,
    )
    register_fast_lora(lora_config)
    register_fast_moe_lora(lora_config, model)
    model = get_peft_model(model, lora_config, autocast_adapter_dtype=False)

    apply_gguf_liger_fused_linear_cross_entropy(model)

    # for layer in model.base_model.model.model.layers:
    #     layer.forward = torch.compile(
    #         layer.forward,
    #         fullgraph=False,
    #         mode="max-autotune-no-cudagraphs",
    #     )

    model.print_trainable_parameters()

    # Dataset is shuffled by the trainer by default
    dataset = load_from_disk(dataset_dir)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,  # Increase batch size if you have more VRAM
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        weight_decay=1e-3,  # For MoE models this can be smaller than dense models
        max_grad_norm=1,
        num_train_epochs=1,
        lr_scheduler_type="linear",
        warmup_steps=100,
        logging_steps=1,
        save_steps=100,
        save_total_limit=5,
        bf16=True,
        optim="adamw_8bit",
        use_liger_kernel=True,
        liger_kernel_config={
            "rope": False,  # Liger's Qwen3 RoPE patch is wrong on Qwen3.5
            "cross_entropy": False,
            "fused_linear_cross_entropy": False,  # We use our cross entropy patch
            "rms_norm": True,
            "swiglu": False,  # Liger's MoE SwiGLU patch is incompatible with our MoE LoRA patch
        },
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        # torch_compile=True,
        # torch_compile_mode="max-autotune",
        report_to="wandb",
        seed=random_seed,
    )
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
        data_collator=fixed_length_lm_collator,
    )

    # TODO: Why do we need to enable it again?
    trainer.model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    trainer_stats = trainer.train()
    # trainer_stats = trainer.train(resume_from_checkpoint=True)
    print("trainer_stats")
    print(trainer_stats)


if __name__ == "__main__":
    main()
