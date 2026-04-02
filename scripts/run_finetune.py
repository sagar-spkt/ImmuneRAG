"""
Phase 4 — Finetuning Script

QLoRA finetuning of a base model on the instruction hierarchy train set.
Pass the model-specific config file; adapters are saved to outputs/models/{model}/.

Llama-3.1-8B-Instruct:
    python scripts/run_finetune.py --config config/train_llama31.yaml

Qwen2.5-7B-Instruct:
    python scripts/run_finetune.py --config config/train_qwen25.yaml

Debug run (edit config debug.enabled=true first):
    python scripts/run_finetune.py --config config/train_llama31.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.train_lora import LoRATrainer


def main():
    parser = argparse.ArgumentParser(
        description="Phase 4 — QLoRA finetuning on instruction hierarchy data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to model-specific training config YAML "
             "(e.g. config/train_llama31.yaml or config/train_qwen25.yaml)",
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
    logger.info("=== Phase 4: QLoRA Finetuning ===")
    logger.info(f"Config: {args.config}")

    trainer = LoRATrainer(args.config)
    trainer.train()

    logger.info("Finetuning complete.")
    logger.info("Next step → run_predict.py with --adapter_path to run Phase 5")


if __name__ == "__main__":
    main()
