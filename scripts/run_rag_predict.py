"""
Phase 2b / 5b — RAG prediction script.

Runs the pretrained or finetuned model on the RAG-eligible test cohort
(`tool_output_misaligned`) with documents retrieved from the persisted
ChromaDB collection injected as `tool` / `ipython` role messages.

Phase 2b (pretrained baseline + RAG):
    python scripts/run_rag_predict.py \\
        --model_id meta-llama/Llama-3.1-8B-Instruct \\
        --model_family llama \\
        --output_dir outputs/predictions/llama31_pretrained_rag

Phase 5b (finetuned + RAG):
    python scripts/run_rag_predict.py \\
        --model_id meta-llama/Llama-3.1-8B-Instruct \\
        --model_family llama \\
        --adapter_path outputs/models/llama31/lora_adapter \\
        --output_dir outputs/predictions/llama31_finetuned_rag

Qwen variants: pass --model_family qwen and --model_id Qwen/Qwen2.5-7B-Instruct.
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.rag_predictor import RAGPredictor
from src.rag.retriever import OracleDistractorRetriever


def main():
    parser = argparse.ArgumentParser(
        description="Run RAG-enabled predictions on the RAG-eligible test cohort",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--model_family",
        required=True,
        choices=["llama", "qwen"],
        help="Selects the tool-role for retrieved-doc injection "
             "(ipython for llama, tool for qwen)",
    )
    parser.add_argument(
        "--adapter_path",
        default=None,
        help="LoRA adapter path (omit for pretrained baseline)",
    )
    parser.add_argument(
        "--config",
        default="config/rag_config.yaml",
        help="Path to rag_config.yaml",
    )
    parser.add_argument(
        "--test_data",
        default="data/final/test.jsonl",
        help="Path to test.jsonl",
    )
    parser.add_argument(
        "--persist_dir",
        default=None,
        help="Override config.chroma.persist_dir",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Override config.retrieval.k",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N eligible examples (smoke-test)",
    )
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    logger = logging.getLogger(__name__)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)["rag"]
    if args.persist_dir:
        cfg["chroma"]["persist_dir"] = args.persist_dir
    if args.k is not None:
        cfg["retrieval"]["k"] = args.k

    phase = "5b (finetuned + RAG)" if args.adapter_path else "2b (pretrained + RAG)"
    logger.info(f"=== Phase {phase} ===")
    logger.info(f"Model        : {args.model_id} (family={args.model_family})")
    if args.adapter_path:
        logger.info(f"Adapter      : {args.adapter_path}")
    logger.info(f"Test data    : {args.test_data}")
    logger.info(f"Persist dir  : {cfg['chroma']['persist_dir']}")
    logger.info(f"k            : {cfg['retrieval']['k']}")
    logger.info(f"Output dir   : {args.output_dir}")

    retriever = OracleDistractorRetriever(cfg)
    retriever.load()

    predictor = RAGPredictor(
        model_id=args.model_id,
        output_path=args.output_dir,
        retriever=retriever,
        model_family=args.model_family,
        adapter_path=args.adapter_path,
        max_new_tokens=args.max_new_tokens,
        rag_scenario=cfg["corpus"]["oracle_source_scenario"],
        limit=args.limit,
    )
    predictor.run(test_data_path=args.test_data)

    logger.info("RAG prediction complete.")
    logger.info(
        f"Next step → run_judge_eval.py "
        f"--predictions_path {args.output_dir}/predictions.jsonl"
    )


if __name__ == "__main__":
    main()
