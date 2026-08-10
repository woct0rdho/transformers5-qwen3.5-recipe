#!/usr/bin/env python3

import os

os.environ["TORCH_LOGS"] = "recompiles"
os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

from pathlib import Path
from typing import Any, cast

import torch
from datasets import Dataset, load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

from bf16_adapter_trainer import BF16AdapterTrainer
from deepseek_v4_attention import configure_deepseek_v4_attention
from deepseek_v4_liger_loss import apply_deepseek_v4_liger_loss
from deepseek_v4_liger_mhc import configure_deepseek_v4_liger_mhc
from deepseek_v4_liger_rmsnorm import configure_deepseek_v4_liger_rmsnorm
from deepseek_v4_lora import (
    DEEPSEEK_V4_TARGET_MODULES_PATTERN,
    configure_deepseek_v4_grouped_mmq,
    register_deepseek_v4_lora,
)
from deepseek_v4_moe_lora import register_deepseek_v4_moe_lora
from fast_moe_ranking import configure_fast_moe_ranking

script_dir = Path(__file__).resolve().parent


# I usually preprocess the dataset into chunks with fixed length. You may change this with your dataset
def fixed_length_lm_collator(examples):
    batch = default_data_collator(examples)
    input_ids = batch["input_ids"].long()
    num_tokens = batch.pop("num_tokens").long()
    positions = torch.arange(input_ids.shape[1]).unsqueeze(0)
    valid_tokens = positions < num_tokens.unsqueeze(1)

    batch["input_ids"] = input_ids
    # Fixed attention requires a full mask. Right-padding cannot affect earlier
    # causal outputs, and the ignored labels keep the padded suffix out of loss.
    batch["attention_mask"] = torch.ones_like(input_ids)
    batch["labels"] = input_ids.masked_fill(~valid_tokens, -100)
    return batch


def main():
    model_dir = Path.home() / "models/ds4"
    gguf_file = "DeepSeek-V4-Flash-IQ2XXS.gguf"
    tokenizer_id = "deepseek-ai/DeepSeek-V4-Flash"
    dataset_dir = script_dir / "data_tokenized_ds4"
    output_dir = script_dir / "out_deepseek_v4"
    random_seed = 19260817

    set_seed(random_seed)

    tokenizer = cast(Any, AutoTokenizer.from_pretrained(tokenizer_id))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        gguf_file=gguf_file,
        gguf_mmap_policy="release",
        dtype=torch.bfloat16,
        attn_implementation="eager",  # attn_implementation="flash_attention_2" is unsupported for DeepSeek V4
        device_map={"": "cuda:0"},
    )

    # Autoregressive decoding cache is not needed in training
    model.config.use_cache = False

    # Disable load balancing loss to save VRAM
    model.config.output_router_logits = False
    model.config.router_aux_loss_coef = 0.0

    configure_deepseek_v4_attention(model)
    configure_deepseek_v4_grouped_mmq(model)
    configure_deepseek_v4_liger_mhc(model)
    configure_deepseek_v4_liger_rmsnorm(model)
    configure_fast_moe_ranking(model)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=DEEPSEEK_V4_TARGET_MODULES_PATTERN,
        r=4,
        lora_alpha=4,
        use_rslora=False,
    )
    register_deepseek_v4_lora(lora_config)
    register_deepseek_v4_moe_lora(lora_config, model)
    model = get_peft_model(model, lora_config, autocast_adapter_dtype=False)

    apply_deepseek_v4_liger_loss(model)

    model.print_trainable_parameters()

    # Dataset is shuffled by the trainer by default
    dataset = load_from_disk(dataset_dir)
    if not isinstance(dataset, Dataset):
        raise TypeError(f"expected a Dataset at {dataset_dir}, got DatasetDict")

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
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        report_to="wandb",
        seed=random_seed,
    )
    trainer = BF16AdapterTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
        data_collator=fixed_length_lm_collator,
    )

    trainer_stats = trainer.train()
    # trainer_stats = trainer.train(resume_from_checkpoint=True)
    print("trainer_stats")
    print(trainer_stats)


if __name__ == "__main__":
    main()
