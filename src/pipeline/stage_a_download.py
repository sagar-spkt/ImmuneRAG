"""
Stage A: Dataset Download and Caching

Downloads datasets from HuggingFace and GitHub, caches them locally,
and creates an index mapping (source_dataset, source_id) -> local_pointer.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datasets import load_dataset
import yaml

logger = logging.getLogger(__name__)


class DatasetDownloader:
    """Downloads and caches datasets according to manifest."""

    def __init__(self, manifest_path: str, config: Dict[str, Any]):
        """
        Initialize dataset downloader.

        Args:
            manifest_path: Path to datasets_manifest.yaml
            config: Configuration from pipeline_config.yaml (stage_a_download)
        """
        self.manifest_path = Path(manifest_path)
        self.config = config
        self.output_dir = Path(config["output_dir"])
        self.cache_dir = Path(config["cache_dir"])
        self.index_file = Path(config["index_file"])

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load manifest
        with open(self.manifest_path) as f:
            self.manifest = yaml.safe_load(f)

        self.index = {}

    def run(self) -> Dict[str, Any]:
        """
        Execute Stage A: Download all datasets.

        Returns:
            Dictionary containing download statistics and index path
        """
        logger.info("Starting Stage A: Dataset Download")

        stats = {
            "datasets_downloaded": 0,
            "total_examples": 0,
            "failed": [],
        }

        for dataset_name, dataset_config in self.manifest["datasets"].items():
            # Skip disabled datasets
            if not dataset_config.get("enabled", True):
                logger.info(f"Skipping disabled dataset: {dataset_name}")
                continue

            try:
                logger.info(f"Downloading dataset: {dataset_name}")
                self._download_dataset(dataset_name, dataset_config)
                stats["datasets_downloaded"] += 1

            except Exception as e:
                logger.error(f"Failed to download {dataset_name}: {e}")
                stats["failed"].append({"dataset": dataset_name, "error": str(e)})

        # Save index
        self._save_index()

        stats["total_examples"] = len(self.index)
        stats["index_path"] = str(self.index_file)

        logger.info(f"Stage A complete: {stats['datasets_downloaded']} datasets, "
                    f"{stats['total_examples']} examples indexed")

        return stats

    def _download_dataset(self, name: str, config: Dict[str, Any]) -> None:
        """Download a single dataset."""
        source_type = config.get("source_type", "huggingface")

        if source_type == "huggingface":
            self._download_huggingface(name, config)
        elif source_type == "github":
            self._download_github(name, config)
        else:
            raise ValueError(f"Unknown source type: {source_type}")

    def _download_huggingface(self, name: str, config: Dict[str, Any]) -> None:
        """Download dataset from HuggingFace."""
        identifier = config["identifier"]
        splits = config.get("splits", None)

        # Load dataset
        try:
            if splits:
                for split in splits:
                    dataset = load_dataset(
                        identifier,
                        split=split,
                        cache_dir=str(self.cache_dir),
                        trust_remote_code=False,
                    )
                    self._index_dataset(name, dataset, split)
            else:
                dataset = load_dataset(
                    identifier,
                    cache_dir=str(self.cache_dir),
                    trust_remote_code=False,
                )
                # Index all available splits
                for split_name in dataset.keys():
                    self._index_dataset(name, dataset[split_name], split_name)

            logger.info(f"Successfully downloaded: {identifier}")

        except Exception as e:
            logger.error(f"Error downloading {identifier}: {e}")
            raise

    def _download_github(self, name: str, config: Dict[str, Any]) -> None:
        """Download dataset from GitHub."""
        # TODO: Implement GitHub dataset downloading
        logger.warning(f"GitHub download not yet implemented for {name}")
        pass

    def _index_dataset(self, dataset_name: str, dataset: Any, split: str) -> None:
        """Create index entries for dataset examples."""
        for idx, example in enumerate(dataset):
            key = f"{dataset_name}_{split}_{idx}"
            self.index[key] = {
                "source_dataset": dataset_name,
                "split": split,
                "index": idx,
                "id": example.get("id", f"{dataset_name}_{idx}"),
            }

    def _save_index(self) -> None:
        """Save the index to JSON file."""
        with open(self.index_file, "w") as f:
            json.dump(self.index, f, indent=2)
        logger.info(f"Saved index to {self.index_file}")


def main():
    """CLI entry point for Stage A."""
    import argparse

    parser = argparse.ArgumentParser(description="Stage A: Dataset Download")
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

    stage_config = pipeline_config["pipeline"]["stage_a_download"]

    # Run downloader
    downloader = DatasetDownloader(args.manifest, stage_config)
    stats = downloader.run()

    print(f"\nStage A Summary:")
    print(f"  Datasets downloaded: {stats['datasets_downloaded']}")
    print(f"  Total examples: {stats['total_examples']}")
    print(f"  Index saved to: {stats['index_path']}")

    if stats["failed"]:
        print(f"\nFailed downloads ({len(stats['failed'])}):")
        for failure in stats["failed"]:
            print(f"  - {failure['dataset']}: {failure['error']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
