"""
Model Prediction for Instruction Hierarchy Evaluation (Phase 2 / Phase 5)

Loads a model (pretrained baseline or finetuned LoRA adapter), runs greedy
inference on the test set, and saves predictions with full metadata for the
downstream LLM-as-Judge evaluation phase.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ModelPredictor:
    """
    Runs inference on test.jsonl and saves predictions.

    Supports both pretrained baselines (Phase 2) and finetuned LoRA
    checkpoints (Phase 5) via the optional adapter_path argument.
    """

    def __init__(
        self,
        model_id: str,
        output_path: str,
        adapter_path: Optional[str] = None,
        max_new_tokens: int = 512,
    ):
        """
        Args:
            model_id: HuggingFace model ID or local path of the base model.
            output_path: Directory where predictions.jsonl will be written.
            adapter_path: Path to a LoRA adapter directory.  If None the base
                          model is used as-is (pretrained baseline).
            max_new_tokens: Maximum tokens the model may generate per example.
        """
        self.model_id = model_id
        self.adapter_path = adapter_path
        self.output_path = Path(output_path)
        self.max_new_tokens = max_new_tokens

        self.model = None
        self.tokenizer = None
        self.predictions: list = []

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def load_model(self) -> None:
        """Load tokenizer and model (base or base + LoRA adapter)."""
        if self.adapter_path:
            logger.info(
                f"Loading base model '{self.model_id}' + LoRA adapter '{self.adapter_path}'"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            base = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            self.model = PeftModel.from_pretrained(base, self.adapter_path)
        else:
            logger.info(f"Loading pretrained model '{self.model_id}'")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()
        logger.info("Model loaded successfully")

    def predict(self, test_data_path: str) -> None:
        """
        Run inference on all examples in test_data_path (test.jsonl).

        Each prediction is stored in self.predictions.  Intermediate results
        are collected in memory; call save() to persist them.

        Args:
            test_data_path: Path to test.jsonl with full metadata.
        """
        logger.info(f"Loading test data from {test_data_path}")
        with open(test_data_path) as f:
            examples = [json.loads(line) for line in f]
        logger.info(f"Loaded {len(examples)} test examples")

        self.predictions = []

        for example in tqdm(examples, desc="Predicting"):
            messages = example["messages"]
            notes = example.get("notes", {})

            # Build prompt: all turns except the last (assistant) turn
            prompt = self.tokenizer.apply_chat_template(
                messages[:-1],
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,  # Greedy for reproducibility
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decode only the newly generated tokens
            new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
            model_output = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

            self.predictions.append(
                {
                    "id": example.get("id", ""),
                    "model_id": self.adapter_path or self.model_id,
                    "scenario": example.get("scenario", ""),
                    "alignment": example.get("alignment", ""),
                    "attack_family": notes.get("attack_family"),
                    "messages_context": messages[:-1],   # for LLM judge
                    "expected_output": messages[-1]["content"],
                    "model_output": model_output,
                }
            )

        logger.info(f"Generated {len(self.predictions)} predictions")

    def save(self) -> None:
        """Write predictions to {output_path}/predictions.jsonl."""
        self.output_path.mkdir(parents=True, exist_ok=True)
        out_file = self.output_path / "predictions.jsonl"

        with open(out_file, "w") as f:
            for pred in self.predictions:
                f.write(json.dumps(pred) + "\n")

        logger.info(f"Saved {len(self.predictions)} predictions to {out_file}")

    def run(self, test_data_path: str) -> None:
        """Load model, predict, and save in one call."""
        self.load_model()
        self.predict(test_data_path)
        self.save()
