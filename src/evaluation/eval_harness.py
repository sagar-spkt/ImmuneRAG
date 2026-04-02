"""
Evaluation Harness for Instruction Hierarchy

Reads test.jsonl (full metadata + messages list), applies the model's chat
template on-the-fly via tokenizer.apply_chat_template(), and runs inference.
The assistant turn is excluded from the prompt so the model generates it.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

from .metrics import HierarchyMetrics

logger = logging.getLogger(__name__)


class EvaluationHarness:
    """Evaluation harness for instruction hierarchy models."""

    def __init__(self, model_path: str, test_data_path: str, output_path: str):
        """
        Initialize evaluation harness.

        Args:
            model_path: Path to the model to evaluate.  For baseline pass the
                        HuggingFace model name / local path directly (no adapter).
                        For finetuned pass the LoRA adapter directory and supply
                        base_model in load_model().
            test_data_path: Path to test.jsonl with full metadata and messages list.
            output_path: Directory to save evaluation results.
        """
        self.model_path = Path(model_path)
        self.test_data_path = Path(test_data_path)
        self.output_path = Path(output_path)

        self.model = None
        self.tokenizer = None
        self.test_data = []
        self.predictions = []

    def load_model(self, base_model: Optional[str] = None) -> None:
        """
        Load model and tokenizer.

        Args:
            base_model: If provided, loads this as the base model and applies the
                        LoRA adapter from self.model_path.  If None, loads
                        self.model_path directly (baseline or merged model).
        """
        if base_model:
            logger.info(f"Loading base model {base_model} + LoRA adapter {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(base_model)
            base = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            self.model = PeftModel.from_pretrained(base, str(self.model_path))
        else:
            logger.info(f"Loading model from {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()
        logger.info("Model loaded successfully")

    def load_test_data(self) -> None:
        """Load test.jsonl (full metadata format with messages list)."""
        logger.info(f"Loading test data from {self.test_data_path}")

        with open(self.test_data_path) as f:
            self.test_data = [json.loads(line) for line in f]

        logger.info(f"Loaded {len(self.test_data)} test examples")

    def run_inference(self, max_new_tokens: int = 512, batch_size: int = 1) -> None:
        """
        Run inference on test data.

        The prompt is built from all message turns except the last (assistant)
        turn using tokenizer.apply_chat_template with add_generation_prompt=True.
        Only the newly generated tokens are decoded as the model response.

        Args:
            max_new_tokens: Maximum tokens to generate.
            batch_size: Batch size for inference (currently 1).
        """
        logger.info("Running inference...")

        self.predictions = []

        for example in tqdm(self.test_data, desc="Inference"):
            messages = example["messages"]

            # All turns except the final assistant turn → model generates the response
            prompt = self.tokenizer.apply_chat_template(
                messages[:-1],
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Greedy decoding for reproducibility
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decode only the newly generated tokens (skip the prompt)
            new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
            response = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

            notes = example.get("notes", {})
            system_content = messages[0]["content"] if messages[0]["role"] == "system" else ""

            prediction = {
                "id": example.get("id", ""),
                "output": response,
                "expected_output": messages[-1]["content"],
                "prompt": prompt,
                # Metadata for metrics
                "scenario": example.get("scenario", ""),
                "alignment": example.get("alignment", ""),
                "system_content": system_content,
                "payload": {"attack_family": notes.get("attack_family")},
                "constraints": notes.get("constraint_type", []),
            }

            self.predictions.append(prediction)

        logger.info(f"Completed inference on {len(self.predictions)} examples")

    def compute_metrics(self) -> Dict[str, Any]:
        """Compute evaluation metrics."""
        logger.info("Computing metrics...")

        metrics_calculator = HierarchyMetrics()
        metrics = metrics_calculator.compute_all_metrics(self.predictions)

        logger.info("Metrics computed")
        return metrics

    def save_results(self, metrics: Dict[str, Any]) -> None:
        """Save evaluation results."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        results = {
            "model_path": str(self.model_path),
            "test_data_path": str(self.test_data_path),
            "metrics": metrics,
            "num_predictions": len(self.predictions),
        }

        # Save results JSON
        results_file = self.output_path / "eval_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        # Save predictions
        predictions_file = self.output_path / "predictions.jsonl"
        with open(predictions_file, "w") as f:
            for pred in self.predictions:
                f.write(json.dumps(pred) + "\n")

        logger.info(f"Results saved to {self.output_path}")

    def run(
        self,
        base_model: Optional[str] = None,
        max_new_tokens: int = 512,
    ) -> Dict[str, Any]:
        """
        Run full evaluation pipeline.

        Args:
            base_model: Optional base model path (for LoRA adapter)
            max_new_tokens: Max tokens to generate

        Returns:
            Evaluation metrics
        """
        # Load model
        self.load_model(base_model)

        # Load test data
        self.load_test_data()

        # Run inference
        self.run_inference(max_new_tokens=max_new_tokens)

        # Compute metrics
        metrics = self.compute_metrics()

        # Save results
        self.save_results(metrics)

        return metrics


def main():
    """CLI entry point for evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Instruction Hierarchy Model")
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to trained model or LoRA adapter",
    )
    parser.add_argument(
        "--base_model",
        default=None,
        help="Base model path (if using LoRA adapter)",
    )
    parser.add_argument(
        "--test_data",
        default="data/final/test.jsonl",
        help="Path to test data (test.jsonl with full metadata and messages list)",
    )
    parser.add_argument(
        "--output",
        default="outputs/evaluation",
        help="Output directory for results",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Max tokens to generate",
    )

    args = parser.parse_args()

    # Run evaluation
    harness = EvaluationHarness(
        model_path=args.model_path,
        test_data_path=args.test_data,
        output_path=args.output,
    )

    metrics = harness.run(
        base_model=args.base_model,
        max_new_tokens=args.max_new_tokens,
    )

    # Print summary
    print("\n=== Evaluation Results ===")
    print(f"ASR (misaligned): {metrics['asr_misaligned']:.3f}")
    print(f"Extraction rate: {metrics['system_extraction_rate']:.3f}")
    print(f"Over-refusal rate: {metrics['over_refusal_rate']:.3f}")
    print(f"Constraint adherence: {metrics['constraint_adherence_aligned']:.3f}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    main()
