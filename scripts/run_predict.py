"""
Phase 2 / Phase 5 — Model Prediction Script

Runs inference on the test set and saves predictions for downstream
LLM-as-Judge evaluation.

Phase 2 (pretrained baseline):
    python scripts/run_predict.py \\
        --model_id meta-llama/Llama-3.1-8B-Instruct \\
        --output_dir outputs/predictions/llama31_pretrained

Phase 5 (finetuned model):
    python scripts/run_predict.py \\
        --model_id meta-llama/Llama-3.1-8B-Instruct \\
        --adapter_path outputs/models/llama31/lora_adapter \\
        --output_dir outputs/predictions/llama31_finetuned

Other examples:
    python scripts/run_predict.py \\
        --model_id Qwen/Qwen2.5-7B-Instruct \\
        --output_dir outputs/predictions/qwen25_pretrained

    python scripts/run_predict.py \\
        --model_id Qwen/Qwen2.5-7B-Instruct \\
        --adapter_path outputs/models/qwen25/lora_adapter \\
        --output_dir outputs/predictions/qwen25_finetuned
"""

import argparse
import logging
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.predict import ModelPredictor


def main():
    parser = argparse.ArgumentParser(
        description="Run model predictions on the test set (Phase 2 / Phase 5)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model_id",
        required=True,
        help="HuggingFace model ID or local path of the base model "
             "(e.g. meta-llama/Llama-3.1-8B-Instruct)",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save predictions.jsonl "
             "(e.g. outputs/predictions/llama31_pretrained)",
    )
    parser.add_argument(
        "--adapter_path",
        default=None,
        help="Path to LoRA adapter directory for finetuned evaluation (Phase 5). "
             "Omit for pretrained baseline (Phase 2).",
    )
    parser.add_argument(
        "--test_data",
        default="data/final/test.jsonl",
        help="Path to test.jsonl with full metadata (default: data/final/test.jsonl)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate per example (default: 512)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    logger = logging.getLogger(__name__)
    phase = "5 (finetuned)" if args.adapter_path else "2 (pretrained baseline)"
    logger.info(f"=== Phase {phase} Prediction ===")
    logger.info(f"Model   : {args.model_id}")
    if args.adapter_path:
        logger.info(f"Adapter : {args.adapter_path}")
    logger.info(f"Test data: {args.test_data}")
    logger.info(f"Output  : {args.output_dir}")

    predictor = ModelPredictor(
        model_id=args.model_id,
        output_path=args.output_dir,
        adapter_path=args.adapter_path,
        max_new_tokens=args.max_new_tokens,
    )

    predictor.run(test_data_path=args.test_data)

    logger.info("Prediction complete.")
    logger.info(f"Next step → run_judge_eval.py --predictions_path {args.output_dir}/predictions.jsonl")


if __name__ == "__main__":
    main()
