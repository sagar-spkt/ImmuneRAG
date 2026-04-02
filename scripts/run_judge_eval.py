"""
Phase 3 / Phase 6 — LLM-as-Judge Evaluation Script

Loads predictions from a previous run_predict.py run, evaluates each with
Mistral-Small as judge, and saves metrics (HAR, ASR, TCR) and per-prediction
verdicts.

Phase 3 (pretrained evaluation):
    python scripts/run_judge_eval.py \\
        --predictions_path outputs/predictions/llama31_pretrained/predictions.jsonl \\
        --output_dir outputs/evaluation/llama31_pretrained

Phase 6 (finetuned evaluation):
    python scripts/run_judge_eval.py \\
        --predictions_path outputs/predictions/llama31_finetuned/predictions.jsonl \\
        --output_dir outputs/evaluation/llama31_finetuned

Other examples:
    # Qwen pretrained
    python scripts/run_judge_eval.py \\
        --predictions_path outputs/predictions/qwen25_pretrained/predictions.jsonl \\
        --output_dir outputs/evaluation/qwen25_pretrained

    # Use 4-bit judge to reduce VRAM
    python scripts/run_judge_eval.py \\
        --predictions_path outputs/predictions/llama31_pretrained/predictions.jsonl \\
        --output_dir outputs/evaluation/llama31_pretrained \\
        --load_in_4bit

    # Compare results after all phases
    python -c "
import json
for tag in ['llama31_pretrained', 'llama31_finetuned']:
    m = json.load(open(f'outputs/evaluation/{tag}/metrics.json'))
    print(tag, '  HAR', m['hierarchy_adherence_rate'], '  ASR', m['attack_success_rate'])
"
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.judge_eval import JudgeEvaluator


def main():
    parser = argparse.ArgumentParser(
        description="Run LLM-as-Judge evaluation on model predictions (Phase 3 / Phase 6)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--predictions_path",
        required=True,
        help="Path to predictions.jsonl produced by run_predict.py",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save judge_results.jsonl and metrics.json",
    )
    parser.add_argument(
        "--judge_model",
        default="mistralai/Mistral-Small-Instruct-2409",
        help="HuggingFace model ID for the judge "
             "(default: mistralai/Mistral-Small-Instruct-2409)",
    )
    parser.add_argument(
        "--torch_dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Torch dtype for the judge model (default: bfloat16)",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load judge in 4-bit quantization to reduce VRAM (~14GB vs ~48GB)",
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
    logger.info("=== LLM-as-Judge Evaluation ===")
    logger.info(f"Predictions : {args.predictions_path}")
    logger.info(f"Judge model : {args.judge_model} (4bit={args.load_in_4bit})")
    logger.info(f"Output      : {args.output_dir}")

    evaluator = JudgeEvaluator(
        predictions_path=args.predictions_path,
        output_path=args.output_dir,
        judge_model=args.judge_model,
        judge_torch_dtype=args.torch_dtype,
        judge_load_in_4bit=args.load_in_4bit,
    )

    evaluator.run()

    logger.info("Evaluation complete.")
    logger.info(f"Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
