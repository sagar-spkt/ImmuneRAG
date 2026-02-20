"""
Stage D: Quality Gates and Balancing

Implements:
- Deduplication
- Quality filtering
- Train/test splitting (deterministic, hash-based)
- Balancing to target mixture
"""

import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict
import yaml

logger = logging.getLogger(__name__)


class QualityControl:
    """Applies quality gates and balancing to hierarchy cases."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize quality control.

        Args:
            config: Configuration from pipeline_config.yaml (stage_d_quality)
        """
        self.config = config
        self.input_file = Path(config["input_file"])
        self.output_dir = Path(config["output_dir"])

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.cases = []
        self.train_cases = []
        self.test_cases = []

    def run(self) -> Dict[str, Any]:
        """
        Execute Stage D: Apply quality gates and balancing.

        Returns:
            Dictionary containing statistics
        """
        logger.info("Starting Stage D: Quality Control")

        # Load cases
        self._load_cases()

        stats = {
            "input_cases": len(self.cases),
            "after_dedup": 0,
            "after_quality": 0,
            "train_count": 0,
            "test_count": 0,
            "train_by_scenario": {},
            "test_by_scenario": {},
        }

        # Deduplication
        if self.config["deduplication"]["enabled"]:
            self._deduplicate()
        stats["after_dedup"] = len(self.cases)

        # Quality filtering
        self._apply_quality_filters()
        stats["after_quality"] = len(self.cases)

        # Train/test splitting
        self._split_train_test()

        # Balancing
        self._balance_splits()

        # Save final datasets
        self._save_splits()

        # Compute statistics
        self._compute_statistics()

        stats["train_count"] = len(self.train_cases)
        stats["test_count"] = len(self.test_cases)

        logger.info(f"Stage D complete: {stats['train_count']} train, "
                    f"{stats['test_count']} test cases")

        return stats

    def _load_cases(self) -> None:
        """Load hierarchy cases from Stage C."""
        with open(self.input_file) as f:
            self.cases = [json.loads(line) for line in f]
        logger.info(f"Loaded {len(self.cases)} cases")

    def _deduplicate(self) -> None:
        """Remove near-duplicate cases."""
        # TODO: Implement minhash LSH or embedding-based deduplication
        logger.info("Deduplication...")

        # Placeholder: exact hash deduplication
        seen = set()
        deduplicated = []

        for case in self.cases:
            # Hash based on messages content
            messages_str = json.dumps(case["messages"], sort_keys=True)
            case_hash = hashlib.md5(messages_str.encode()).hexdigest()

            if case_hash not in seen:
                seen.add(case_hash)
                deduplicated.append(case)

        removed = len(self.cases) - len(deduplicated)
        if removed > 0:
            logger.info(f"Removed {removed} duplicate cases")

        self.cases = deduplicated

    def _apply_quality_filters(self) -> None:
        """Apply quality filters."""
        filters = self.config["quality_filters"]
        filtered = []

        for case in self.cases:
            if self._passes_quality_filters(case, filters):
                filtered.append(case)

        removed = len(self.cases) - len(filtered)
        if removed > 0:
            logger.info(f"Filtered out {removed} cases")

        self.cases = filtered

    def _passes_quality_filters(self, case: Dict[str, Any], filters: Dict) -> bool:
        """Check if case passes quality filters."""
        messages = case["messages"]

        # Message count
        if len(messages) < filters["min_message_count"]:
            return False
        if len(messages) > filters["max_message_count"]:
            return False

        # Require assistant response
        if filters["require_assistant_response"]:
            if not any(msg["role"] == "assistant" for msg in messages):
                return False

        # TODO: Add token count checks

        return True

    def _split_train_test(self) -> None:
        """Deterministic train/test splitting using hash."""
        splitting_config = self.config["splitting"]
        test_ratio = splitting_config["test_ratio"]
        seed = splitting_config["random_seed"]

        for case in self.cases:
            # Deterministic hash
            hash_str = f"{case.get('id', '')}_{seed}"
            hash_val = int(hashlib.md5(hash_str.encode()).hexdigest(), 16)
            split_val = (hash_val % 100) / 100.0

            if split_val < test_ratio:
                case["split"] = "test"
                self.test_cases.append(case)
            else:
                case["split"] = "train"
                self.train_cases.append(case)

        logger.info(f"Split: {len(self.train_cases)} train, {len(self.test_cases)} test")

    def _balance_splits(self) -> None:
        """Balance train/test splits to match target mixture."""
        # TODO: Implement stratified balancing
        logger.info("Balancing splits...")
        pass

    def _save_splits(self) -> None:
        """Save train and test splits."""
        train_file = Path(self.config["output_files"]["train"])
        test_file = Path(self.config["output_files"]["test"])

        with open(train_file, "w") as f:
            for case in self.train_cases:
                f.write(json.dumps(case) + "\n")

        with open(test_file, "w") as f:
            for case in self.test_cases:
                f.write(json.dumps(case) + "\n")

        logger.info(f"Saved train to {train_file}")
        logger.info(f"Saved test to {test_file}")

    def _compute_statistics(self) -> None:
        """Compute and save statistics."""
        stats = {
            "train": self._compute_split_stats(self.train_cases),
            "test": self._compute_split_stats(self.test_cases),
        }

        stats_file = Path(self.config["output_files"]["stats"])
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Saved statistics to {stats_file}")

    def _compute_split_stats(self, cases: List[Dict]) -> Dict[str, Any]:
        """Compute statistics for a split."""
        stats = {
            "total": len(cases),
            "by_scenario": defaultdict(int),
            "by_alignment": defaultdict(int),
        }

        for case in cases:
            stats["by_scenario"][case["scenario"]] += 1
            stats["by_alignment"][case["alignment"]] += 1

        return dict(stats)


def main():
    """CLI entry point for Stage D."""
    import argparse

    parser = argparse.ArgumentParser(description="Stage D: Quality Control")
    parser.add_argument(
        "--config",
        default="config/pipeline_config.yaml",
        help="Path to pipeline config",
    )

    args = parser.parse_args()

    # Load pipeline config
    with open(args.config) as f:
        pipeline_config = yaml.safe_load(f)

    stage_config = pipeline_config["pipeline"]["stage_d_quality"]

    # Run quality control
    qc = QualityControl(stage_config)
    stats = qc.run()

    print(f"\nStage D Summary:")
    print(f"  Input cases: {stats['input_cases']}")
    print(f"  After dedup: {stats['after_dedup']}")
    print(f"  After quality filters: {stats['after_quality']}")
    print(f"  Train: {stats['train_count']}")
    print(f"  Test: {stats['test_count']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
