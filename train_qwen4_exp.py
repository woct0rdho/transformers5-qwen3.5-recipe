#!/usr/bin/env python3

import os

os.environ["TORCH_LOGS"] = "recompiles"
os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from pathlib import Path

import torch
from datasets import Dataset, load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

from bf16_adapter_trainer import BF16AdapterTrainer
from fast_moe_ranking import configure_fast_moe_ranking
from fla_tuning import configure_qwen4_exp_fla
from gdn_bwd_dhu import install as install_gdn_bwd_dhu
from gdn_bwd_dqkwg import install as install_gdn_bwd_dqkwg
from gdn_tiled_value_heads import configure_tiled_value_heads
from gdn_wu_recompute import install as install_gdn_wu_recompute
from gguf_dequant_compile import configure_compiled_gguf_dequantize
from ple_disk_residency import (
    configure_ple_disk_residency,
    prefetch_ple_rows,
    require_ple_disk_residency,
)
from qwen4_exp_attention import configure_qwen4_exp_qsa_attention
from qwen4_exp_fused_norms import configure_qwen4_exp_fused_norms
from qwen4_exp_indexer import configure_qwen4_exp_indexer_fast_path
from qwen4_exp_liger_hc import configure_qwen4_exp_hc_norm
from qwen4_exp_liger_loss import apply_qwen4_exp_liger_fused_linear_cross_entropy
from qwen4_exp_lora import (
    QWEN4_EXP_TARGET_MODULES_PATTERN,
    configure_qwen4_exp_frozen_mmq,
    register_qwen4_exp_adapters,
)

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
    model_dir = Path.home() / "models/qwen4"
    gguf_file = "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0.gguf"
    # The Qwen3.8 tokenizer is the same as Qwen3.5/3.6
    dataset_dir = script_dir / "data_tokenized_qwen3.5"
    output_dir = script_dir / "out_qwen38"
    random_seed = 19260817

    set_seed(random_seed)

    configure_compiled_gguf_dequantize()
    configure_qwen4_exp_fla()
    configure_tiled_value_heads()
    install_gdn_bwd_dhu()
    install_gdn_bwd_dqkwg()
    install_gdn_wu_recompute()
    # Off by default: this machine has room for the table, and keeping it resident is faster. Set
    # QWEN4_PLE_ON_DISK=1 to keep the 26.82 GiB payload in the file and gather its rows per forward.
    configure_ple_disk_residency(
        checkpoint=model_dir / gguf_file,
        enabled=os.environ.get("QWEN4_PLE_ON_DISK") == "1",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        gguf_file=gguf_file,
        local_files_only=True,
    )
    assert isinstance(tokenizer, PreTrainedTokenizerBase)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        gguf_file=gguf_file,
        gguf_mmap_policy="pread",
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation=None,  # attn_implementation has no effect for Qwen4-Exp
        device_map={"": "cuda:0"},
    )

    # Autoregressive decoding cache is not needed in training
    model.config.use_cache = False

    # Disable load balancing loss to save VRAM
    model.config.output_router_logits = False
    model.config.router_aux_loss_coef = 0.0

    if os.environ.get("QWEN4_PLE_ON_DISK") == "1":
        require_ple_disk_residency(model)

    configure_fast_moe_ranking(model)
    configure_qwen4_exp_frozen_mmq(model)
    configure_qwen4_exp_fused_norms(model)
    configure_qwen4_exp_hc_norm(model)
    configure_qwen4_exp_indexer_fast_path(model)
    configure_qwen4_exp_qsa_attention(model)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=QWEN4_EXP_TARGET_MODULES_PATTERN,
        r=4,
        lora_alpha=4,
        use_rslora=False,
    )
    register_qwen4_exp_adapters(lora_config, model)
    model = get_peft_model(model, lora_config, autocast_adapter_dtype=False)

    apply_qwen4_exp_liger_fused_linear_cross_entropy(model)

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

    data_collator = fixed_length_lm_collator
    if os.environ.get("QWEN4_PLE_ON_DISK") == "1":
        # The collator runs on the host before the forward, while the previous step's kernels are still
        # queued, which is where the PLE rows for the next batch are worth reading.
        def prefetching_collator(features):
            batch = fixed_length_lm_collator(features)
            prefetch_ple_rows(model, batch["input_ids"][0])
            return batch

        data_collator = prefetching_collator

    trainer = BF16AdapterTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
        data_collator=data_collator,
    )

    trainer_stats = trainer.train()
    # trainer_stats = trainer.train(resume_from_checkpoint=True)
    print("trainer_stats")
    print(trainer_stats)


if __name__ == "__main__":
    main()
