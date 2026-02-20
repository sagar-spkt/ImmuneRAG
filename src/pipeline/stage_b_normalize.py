"""
Stage B: Raw Normalization to Seed Examples

Converts diverse dataset formats to canonical seed format:
{seed_id, source_dataset, prompt, response, metadata}
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml
from datasets import load_from_disk, load_dataset

logger = logging.getLogger(__name__)


class SeedNormalizer:
    """Normalizes raw datasets to canonical seed format."""

    def __init__(self, manifest_path: str, config: Dict[str, Any]):
        """
        Initialize seed normalizer.

        Args:
            manifest_path: Path to datasets_manifest.yaml
            config: Configuration from pipeline_config.yaml (stage_b_normalize)
        """
        self.manifest_path = Path(manifest_path)
        self.config = config
        self.input_dir = Path(config["input_dir"])
        self.output_file = Path(config["output_file"])

        # Create output directory
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        # Load manifest
        with open(self.manifest_path) as f:
            self.manifest = yaml.safe_load(f)

        self.normalization_rules = config["normalization_rules"]
        self.seeds = []

    def run(self) -> Dict[str, Any]:
        """
        Execute Stage B: Normalize datasets to seed format.

        Returns:
            Dictionary containing normalization statistics
        """
        logger.info("Starting Stage B: Normalization")

        stats = {
            "datasets_processed": 0,
            "total_seeds": 0,
            "filtered_out": 0,
            "by_dataset": {},
        }

        for dataset_name, dataset_config in self.manifest["datasets"].items():
            if not dataset_config.get("enabled", True):
                continue

            try:
                logger.info(f"Normalizing dataset: {dataset_name}")
                dataset_stats = self._normalize_dataset(dataset_name, dataset_config)
                stats["by_dataset"][dataset_name] = dataset_stats
                stats["datasets_processed"] += 1

            except Exception as e:
                logger.error(f"Failed to normalize {dataset_name}: {e}")

        # Apply deduplication
        if self.normalization_rules.get("deduplicate_prompts", True):
            self._deduplicate()

        # Save seeds
        self._save_seeds()

        stats["total_seeds"] = len(self.seeds)

        logger.info(f"Stage B complete: {stats['total_seeds']} seeds from "
                    f"{stats['datasets_processed']} datasets")

        return stats

    def _normalize_dataset(self, name: str, config: Dict[str, Any]) -> Dict[str, int]:
        """Normalize a single dataset."""
        identifier = config["identifier"]
        source_type = config.get("source_type", "huggingface")

        stats = {"extracted": 0, "filtered": 0}

        if source_type == "huggingface":
            stats = self._normalize_huggingface(name, config)
        elif source_type == "github":
            logger.warning(f"GitHub normalization not yet implemented for {name}")

        return stats

    def _normalize_huggingface(self, name: str, config: Dict[str, Any]) -> Dict[str, int]:
        """Normalize HuggingFace dataset."""
        identifier = config["identifier"]
        fields = config.get("fields", [])
        splits = config.get("splits", None)

        stats = {"extracted": 0, "filtered": 0}

        # Load dataset
        cache_dir = self.input_dir / ".cache"
        try:
            if splits:
                for split in splits:
                    dataset = load_dataset(identifier, split=split, cache_dir=str(cache_dir))
                    extracted, filtered = self._extract_seeds(name, dataset, config)
                    stats["extracted"] += extracted
                    stats["filtered"] += filtered
            else:
                dataset = load_dataset(identifier, cache_dir=str(cache_dir))
                for split_name in dataset.keys():
                    extracted, filtered = self._extract_seeds(name, dataset[split_name], config)
                    stats["extracted"] += extracted
                    stats["filtered"] += filtered

        except Exception as e:
            logger.error(f"Error loading {identifier}: {e}")
            raise

        return stats

    def _extract_seeds(self, dataset_name: str, dataset: Any, config: Dict[str, Any]) -> tuple:
        """Extract seed examples from dataset."""
        extracted = 0
        filtered = 0

        # Dataset-specific extraction logic
        # TODO: Implement extraction for each dataset type

        for idx, example in enumerate(dataset):
            seed = self._extract_seed_from_example(dataset_name, idx, example, config)

            if seed and self._passes_filters(seed):
                self.seeds.append(seed)
                extracted += 1
            else:
                filtered += 1

        return extracted, filtered

    def _extract_seed_from_example(
        self, dataset_name: str, idx: int, example: Dict, config: Dict
    ) -> Optional[Dict[str, Any]]:
        """Extract a seed from a single example."""
        # Placeholder: Implement dataset-specific extraction
        # This is where you'd parse the specific format of each dataset

        seed = {
            "seed_id": f"{dataset_name}_{idx}",
            "source_dataset": dataset_name,
            "prompt": "",
            "response": "",
            "metadata": {
                "language": "en",
                "safety_flag": "ok",
                "task_hint": "",
                "source_example_id": str(idx),
            },
        }

        # TODO: Implement actual extraction based on dataset format
        # For now, return None to skip
        return None

    def _passes_filters(self, seed: Dict[str, Any]) -> bool:
        """Check if seed passes quality filters."""
        rules = self.normalization_rules

        prompt = seed["prompt"]
        response = seed["response"]

        # Length filters
        if len(prompt) < rules["min_prompt_length"]:
            return False
        if len(prompt) > rules["max_prompt_length"]:
            return False
        if len(response) < rules["min_response_length"]:
            return False
        if len(response) > rules["max_response_length"]:
            return False

        # Safety flag check
        if seed["metadata"]["safety_flag"] != "ok":
            return False

        return True

    def _deduplicate(self) -> None:
        """Remove duplicate prompts."""
        seen = set()
        deduplicated = []

        for seed in self.seeds:
            prompt_hash = hash(seed["prompt"])
            if prompt_hash not in seen:
                seen.add(prompt_hash)
                deduplicated.append(seed)

        removed = len(self.seeds) - len(deduplicated)
        if removed > 0:
            logger.info(f"Removed {removed} duplicate seeds")
        self.seeds = deduplicated

    def _save_seeds(self) -> None:
        """Save seeds to JSONL file."""
        with open(self.output_file, "w") as f:
            for seed in self.seeds:
                f.write(json.dumps(seed) + "\n")
        logger.info(f"Saved {len(self.seeds)} seeds to {self.output_file}")


def main():
    """CLI entry point for Stage B."""
    import argparse

    parser = argparse.ArgumentParser(description="Stage B: Seed Normalization")
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

    args = parser.parse_args()

    # Load pipeline config
    with open(args.config) as f:
        pipeline_config = yaml.safe_load(f)

    stage_config = pipeline_config["pipeline"]["stage_b_normalize"]

    # Run normalizer
    normalizer = SeedNormalizer(args.manifest, stage_config)
    stats = normalizer.run()

    print(f"\nStage B Summary:")
    print(f"  Datasets processed: {stats['datasets_processed']}")
    print(f"  Total seeds: {stats['total_seeds']}")
    print(f"\nBy dataset:")
    for name, dataset_stats in stats["by_dataset"].items():
        print(f"  {name}: {dataset_stats['extracted']} extracted, "
              f"{dataset_stats['filtered']} filtered")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
