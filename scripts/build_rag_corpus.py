"""
Phase 2b.0 — RAG corpus builder.

Extracts oracle attack documents from `tool_output_misaligned` test rows and a
benign distractor pool from `open_aligned` test rows, embeds both, and
persists them into a ChromaDB collection on disk.

Idempotent: re-running with the same persist_dir is a no-op unless --rebuild
is passed (or the manifest is missing).

Examples:
    # Full corpus
    python scripts/build_rag_corpus.py

    # Smoke-test corpus (10 oracle + 50 distractors)
    python scripts/build_rag_corpus.py \\
        --persist_dir /tmp/rag_smoke \\
        --limit_oracles 10 --limit_distractors 50

    # Force rebuild after dataset change
    python scripts/build_rag_corpus.py --rebuild
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.corpus_builder import CorpusBuilder


def main():
    parser = argparse.ArgumentParser(
        description="Build the ChromaDB corpus for the RAG evaluation track",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        default="config/rag_config.yaml",
        help="Path to rag_config.yaml (default: config/rag_config.yaml)",
    )
    parser.add_argument(
        "--test_data",
        default=None,
        help="Override config.corpus.test_data_path "
             "(default: read from config)",
    )
    parser.add_argument(
        "--persist_dir",
        default=None,
        help="Override config.chroma.persist_dir (default: read from config)",
    )
    parser.add_argument(
        "--limit_oracles",
        type=int,
        default=None,
        help="Cap on oracle docs (smoke-test mode)",
    )
    parser.add_argument(
        "--limit_distractors",
        type=int,
        default=None,
        help="Cap on distractor docs (smoke-test mode)",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete the existing Chroma collection before re-indexing",
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

    with open(args.config) as f:
        cfg = yaml.safe_load(f)["rag"]

    if args.persist_dir:
        cfg["chroma"]["persist_dir"] = args.persist_dir
    test_data = args.test_data or cfg["corpus"]["test_data_path"]

    manifest_path = Path(cfg["output"]["manifest_path"])
    if manifest_path.exists() and not args.rebuild:
        logger.info(
            f"Manifest already exists at {manifest_path} — skipping build. "
            f"Pass --rebuild to force re-indexing."
        )
        return

    logger.info("=== Phase 2b.0 — RAG corpus build ===")
    logger.info(f"Test data    : {test_data}")
    logger.info(f"Persist dir  : {cfg['chroma']['persist_dir']}")
    logger.info(f"Collection   : {cfg['chroma']['collection_name']}")
    logger.info(f"Embedder     : {cfg['embedding']['model']}")

    builder = CorpusBuilder(cfg)
    stats = builder.build(
        test_data_path=test_data,
        limit_oracles=args.limit_oracles,
        limit_distractors=args.limit_distractors,
        rebuild=args.rebuild,
    )
    builder.write_manifest(stats=stats, test_data_path=test_data)

    logger.info("Corpus build complete.")


if __name__ == "__main__":
    main()
