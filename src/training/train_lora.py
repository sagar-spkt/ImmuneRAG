"""
LoRA/QLoRA Training for Instruction Hierarchy

Implements supervised fine-tuning with LoRA on Llama-3-8B.
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

logger = logging.getLogger(__name__)


class LoRATrainer:
    """LoRA/QLoRA trainer for instruction hierarchy."""

    def __init__(self, config_path: str):
        """
        Initialize trainer.

        Args:
            config_path: Path to training_config.yaml
        """
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.model_config = self.config["model"]
        self.lora_config = self.config["lora"]
        self.training_config = self.config["training"]

        self.model = None
        self.tokenizer = None
        self.trainer = None

    def setup(self) -> None:
        """Setup model, tokenizer, and trainer."""
        logger.info("Setting up model and tokenizer...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config["base_model"],
            trust_remote_code=self.model_config["trust_remote_code"],
        )

        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Quantization config for QLoRA
        if self.lora_config["method"] == "qlora" and self.model_config.get("load_in_4bit", False):
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type=self.lora_config.get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_use_double_quant=self.lora_config.get("bnb_4bit_use_double_quant", True),
            )
        else:
            bnb_config = None

        # Load model
        load_kwargs: Dict[str, Any] = {
            "quantization_config": bnb_config,
            "torch_dtype": torch.bfloat16 if self.model_config["torch_dtype"] == "bfloat16" else torch.float16,
            "device_map": self.model_config["device_map"],
            "trust_remote_code": self.model_config["trust_remote_code"],
        }
        if self.model_config.get("use_flash_attention_2", False):
            try:
                import flash_attn  # noqa: F401
                load_kwargs["attn_implementation"] = "flash_attention_2"
            except ImportError:
                logger.warning(
                    "flash-attn not installed — falling back to sdpa. "
                    "Install with: pip install flash-attn --no-build-isolation"
                )
                load_kwargs["attn_implementation"] = "sdpa"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config["base_model"],
            **load_kwargs,
        )

        # Prepare model for k-bit training
        if bnb_config:
            self.model = prepare_model_for_kbit_training(self.model)

        # Setup LoRA
        lora_config = LoraConfig(
            r=self.lora_config["r"],
            lora_alpha=self.lora_config["lora_alpha"],
            target_modules=self.lora_config["target_modules"],
            lora_dropout=self.lora_config["lora_dropout"],
            bias=self.lora_config["bias"],
            task_type=self.lora_config["task_type"],
        )

        self.model = get_peft_model(self.model, lora_config)

        logger.info("Model setup complete")
        self.model.print_trainable_parameters()

    def load_data(self) -> tuple:
        """Load training and evaluation datasets."""
        logger.info("Loading datasets...")

        # Load train data
        train_dataset = load_dataset(
            "json",
            data_files=self.training_config["train_data"],
            split="train"
        )

        # Load eval data
        eval_dataset = load_dataset(
            "json",
            data_files=self.training_config["eval_data"],
            split="train"
        )

        logger.info(f"Loaded {len(train_dataset)} train, {len(eval_dataset)} eval examples")

        return train_dataset, eval_dataset

    def create_trainer(self, train_dataset, eval_dataset) -> SFTTrainer:
        """Create SFTTrainer."""
        logger.info("Creating trainer...")

        # SFT Trainer
        sft_config = self.config.get("sft", {})

        # Pre-apply chat template so SFTTrainer receives a "text" field.
        tokenizer = self.tokenizer

        def apply_template(example):
            return {"text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )}

        train_dataset = train_dataset.map(apply_template)
        eval_dataset = eval_dataset.map(apply_template)

        # Build a completion-only collator so loss is computed ONLY on
        # assistant-response tokens. Without this, the model is trained to
        # predict injected user/tool tokens (attack content) which increases
        # attack success rate after finetuning.
        model_type = self.model_config.get("model_type", "")
        if "qwen" in model_type.lower():
            response_template = "<|im_start|>assistant\n"
        else:
            # Llama-3.x Instruct format
            response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"

        collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template,
            tokenizer=self.tokenizer,
        )

        # Training arguments (SFTConfig = TrainingArguments + SFT-specific fields)
        training_args = SFTConfig(
            output_dir=self.training_config["output_dir"],
            run_name=self.training_config.get("run_name", "instruction-hierarchy"),
            num_train_epochs=self.training_config["num_train_epochs"],
            per_device_train_batch_size=self.training_config["per_device_train_batch_size"],
            per_device_eval_batch_size=self.training_config["per_device_eval_batch_size"],
            gradient_accumulation_steps=self.training_config["gradient_accumulation_steps"],
            learning_rate=self.training_config["learning_rate"],
            weight_decay=self.training_config["weight_decay"],
            lr_scheduler_type=self.training_config["lr_scheduler_type"],
            warmup_ratio=self.training_config["warmup_ratio"],
            logging_steps=self.training_config["logging_steps"],
            save_strategy=self.training_config["save_strategy"],
            save_steps=self.training_config["save_steps"],
            save_total_limit=self.training_config["save_total_limit"],
            eval_strategy=self.training_config.get("eval_strategy", "steps"),
            eval_steps=self.training_config.get("eval_steps", 250),
            bf16=self.training_config["bf16"],
            fp16=self.training_config["fp16"],
            gradient_checkpointing=self.training_config["gradient_checkpointing"],
            gradient_checkpointing_kwargs=self.training_config.get("gradient_checkpointing_kwargs", {}),
            optim=self.training_config["optim"],
            max_grad_norm=self.training_config["max_grad_norm"],
            seed=self.training_config["seed"],
            report_to=self.training_config.get("report_to", ["tensorboard"]),
            load_best_model_at_end=self.training_config.get("load_best_model_at_end", True),
            metric_for_best_model=self.training_config.get("metric_for_best_model", "eval_loss"),
            dataloader_num_workers=self.training_config.get("dataloader_num_workers", 4),
            remove_unused_columns=self.training_config.get("remove_unused_columns", False),
            # SFT-specific fields
            max_length=sft_config.get("max_seq_length", 4096),
            packing=sft_config.get("packing", False),
            dataset_text_field="text",
        )

        trainer = SFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
        )

        return trainer

    def train(self) -> None:
        """Execute training."""
        logger.info("Starting training...")

        # Setup
        self.setup()

        # Load data
        train_dataset, eval_dataset = self.load_data()

        # Create trainer
        self.trainer = self.create_trainer(train_dataset, eval_dataset)

        # Train
        self.trainer.train()

        # Save final model
        self.save_model()

        logger.info("Training complete!")

    def save_model(self) -> None:
        """Save trained LoRA adapter."""
        output_dir = Path(self.training_config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving model to {output_dir}")

        # Save LoRA adapter
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        logger.info("Model saved successfully")

        # Optionally merge and save full model
        if self.config.get("post_training", {}).get("merge_adapter", False):
            self.merge_and_save()

    def merge_and_save(self) -> None:
        """Merge LoRA weights into base model and save."""
        logger.info("Merging LoRA adapter into base model...")

        merged_path = Path(self.config["post_training"]["merged_model_path"])
        merged_path.mkdir(parents=True, exist_ok=True)

        # Merge weights
        merged_model = self.model.merge_and_unload()

        # Save
        merged_model.save_pretrained(merged_path)
        self.tokenizer.save_pretrained(merged_path)

        logger.info(f"Merged model saved to {merged_path}")


def main():
    """CLI entry point for training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train LoRA for Instruction Hierarchy")
    parser.add_argument(
        "--config",
        default="config/training_config.yaml",
        help="Path to training config",
    )

    args = parser.parse_args()

    # Run training
    trainer = LoRATrainer(args.config)
    trainer.train()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    main()
