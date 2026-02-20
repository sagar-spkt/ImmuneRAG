#!/usr/bin/env python3
"""
Main Pipeline Orchestrator for Instruction Hierarchy Training

Runs all pipeline stages (A-E) in sequence.
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline import (
    DatasetDownloader,
    SeedNormalizer,
    HierarchyGenerator,
    QualityControl,
    ModelRenderer,
)

logger = logging.getLogger(__name__)


def load_configs(
    manifest_path: str = "config/datasets_manifest.yaml",
    pipeline_path: str = "config/pipeline_config.yaml",
):
    """Load configuration files."""
    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)

    with open(pipeline_path) as f:
        pipeline = yaml.safe_load(f)

    return manifest, pipeline


def run_stage_a(manifest_path: str, config: dict) -> bool:
    """Run Stage A: Dataset Download."""
    logger.info("=" * 60)
    logger.info("STAGE A: Dataset Download")
    logger.info("=" * 60)

    try:
        downloader = DatasetDownloader(manifest_path, config)
        stats = downloader.run()

        logger.info(f"✓ Stage A complete: {stats['datasets_downloaded']} datasets downloaded")
        return True

    except Exception as e:
        logger.error(f"✗ Stage A failed: {e}")
        return False


def run_stage_b(manifest_path: str, config: dict) -> bool:
    """Run Stage B: Seed Normalization."""
    logger.info("=" * 60)
    logger.info("STAGE B: Seed Normalization")
    logger.info("=" * 60)

    try:
        normalizer = SeedNormalizer(manifest_path, config)
        stats = normalizer.run()

        logger.info(f"✓ Stage B complete: {stats['total_seeds']} seeds created")
        return True

    except Exception as e:
        logger.error(f"✗ Stage B failed: {e}")
        return False


def run_stage_c(config: dict) -> bool:
    """Run Stage C: Hierarchy Case Generation."""
    logger.info("=" * 60)
    logger.info("STAGE C: Hierarchy Case Generation")
    logger.info("=" * 60)

    try:
        generator = HierarchyGenerator(config)
        stats = generator.run()

        logger.info(f"✓ Stage C complete: {stats['total_cases']} cases generated")
        return True

    except Exception as e:
        logger.error(f"✗ Stage C failed: {e}")
        return False


def run_stage_d(config: dict) -> bool:
    """Run Stage D: Quality Control."""
    logger.info("=" * 60)
    logger.info("STAGE D: Quality Control & Balancing")
    logger.info("=" * 60)

    try:
        qc = QualityControl(config)
        stats = qc.run()

        logger.info(f"✓ Stage D complete: {stats['train_count']} train, {stats['test_count']} test")
        return True

    except Exception as e:
        logger.error(f"✗ Stage D failed: {e}")
        return False


def run_stage_e(config: dict) -> bool:
    """Run Stage E: Model Rendering."""
    logger.info("=" * 60)
    logger.info("STAGE E: Model-Specific Rendering")
    logger.info("=" * 60)

    try:
        renderer = ModelRenderer(config)
        stats = renderer.run()

        logger.info(f"✓ Stage E complete: {stats['train_rendered']} train, "
                    f"{stats['test_rendered']} test rendered")
        return True

    except Exception as e:
        logger.error(f"✗ Stage E failed: {e}")
        return False


def main():
    """Main orchestrator."""
    parser = argparse.ArgumentParser(
        description="Run Instruction Hierarchy Data Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all stages
  python scripts/run_pipeline.py

  # Run specific stages
  python scripts/run_pipeline.py --stages A B C

  # Run from a specific stage
  python scripts/run_pipeline.py --from-stage C
        """,
    )

    parser.add_argument(
        "--manifest",
        default="config/datasets_manifest.yaml",
        help="Path to datasets manifest",
    )
    parser.add_argument(
        "--config",
        default="config/pipeline_config.yaml",
        help="Path to pipeline config",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["A", "B", "C", "D", "E"],
        help="Specific stages to run",
    )
    parser.add_argument(
        "--from-stage",
        choices=["A", "B", "C", "D", "E"],
        help="Run from this stage onwards",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load configs
    logger.info("Loading configurations...")
    manifest, pipeline_config = load_configs(args.manifest, args.config)

    # Determine which stages to run
    all_stages = ["A", "B", "C", "D", "E"]

    if args.stages:
        stages_to_run = args.stages
    elif args.from_stage:
        start_idx = all_stages.index(args.from_stage)
        stages_to_run = all_stages[start_idx:]
    else:
        stages_to_run = all_stages

    logger.info(f"Running stages: {', '.join(stages_to_run)}")

    # Run stages
    stage_functions = {
        "A": lambda: run_stage_a(args.manifest, pipeline_config["pipeline"]["stage_a_download"]),
        "B": lambda: run_stage_b(args.manifest, pipeline_config["pipeline"]["stage_b_normalize"]),
        "C": lambda: run_stage_c(pipeline_config["pipeline"]["stage_c_hierarchy"]),
        "D": lambda: run_stage_d(pipeline_config["pipeline"]["stage_d_quality"]),
        "E": lambda: run_stage_e(pipeline_config["pipeline"]["stage_e_render"]),
    }

    results = {}
    for stage in stages_to_run:
        success = stage_functions[stage]()
        results[stage] = success

        if not success:
            logger.error(f"Pipeline failed at Stage {stage}")
            sys.exit(1)

    # Summary
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info("All stages completed successfully!")
    logger.info("\nNext steps:")
    logger.info("  1. Review data quality: data/final/stats.json")
    logger.info("  2. Start training: python src/training/train_lora.py")
    logger.info("  3. Evaluate model: python src/evaluation/eval_harness.py")


if __name__ == "__main__":
    main()
