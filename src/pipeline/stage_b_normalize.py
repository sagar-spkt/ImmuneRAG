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

        # Special handling for OASST2 (requires tree reconstruction)
        if name == "oasst2":
            return self._normalize_oasst2_with_tree(config)

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

    def _normalize_oasst2_with_tree(self, config: Dict[str, Any]) -> Dict[str, int]:
        """
        Special handler for OASST2 with conversation tree reconstruction.
        Loads full dataset, builds message map, extracts (prompter -> assistant) pairs.
        """
        identifier = config["identifier"]
        cache_dir = self.input_dir / ".cache"
        stats = {"extracted": 0, "filtered": 0}

        logger.info("Loading OASST2 dataset for tree reconstruction...")

        try:
            # Load full dataset (all splits)
            dataset = load_dataset(identifier, cache_dir=str(cache_dir))

            # Process train split (or all available splits)
            for split_name in dataset.keys():
                split_data = dataset[split_name]
                logger.info(f"Processing OASST2 split: {split_name} ({len(split_data)} messages)")

                # Build message lookup map
                message_map = {}
                for msg in split_data:
                    message_id = msg.get("message_id")
                    if message_id:
                        message_map[message_id] = msg

                # Extract (prompter -> assistant) pairs
                for msg in split_data:
                    # Only process assistant messages
                    if msg.get("role") != "assistant":
                        continue

                    # Language filter
                    if msg.get("lang") != "en":
                        continue

                    # Quality checks
                    if msg.get("deleted", False) or not msg.get("review_result", False):
                        continue

                    # Find parent (should be prompter)
                    parent_id = msg.get("parent_id")
                    if not parent_id:
                        continue

                    parent = message_map.get(parent_id)
                    if not parent or parent.get("role") != "prompter":
                        continue

                    # Also check parent quality
                    if parent.get("lang") != "en":
                        continue
                    if parent.get("deleted", False):
                        continue

                    # Create seed
                    seed = {
                        "seed_id": f"oasst2_{msg['message_id']}",
                        "source_dataset": "oasst2",
                        "prompt": parent.get("text", ""),
                        "response": msg.get("text", ""),
                        "metadata": {
                            "language": msg.get("lang", "en"),
                            "safety_flag": "ok",
                            "task_hint": "conversation",
                            "source_example_id": msg.get("message_id"),
                            "parent_id": parent_id,
                            "message_tree_id": msg.get("message_tree_id"),
                        },
                    }

                    if self._passes_filters(seed):
                        self.seeds.append(seed)
                        stats["extracted"] += 1
                    else:
                        stats["filtered"] += 1

                logger.info(f"OASST2 {split_name}: {stats['extracted']} pairs extracted")

        except Exception as e:
            logger.error(f"Error processing OASST2: {e}")
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
        """Extract a seed from a single example - dispatches to dataset-specific handler."""
        # Dispatch to dataset-specific handler
        handlers = {
            "oasst2": self._extract_oasst2,
            "ultrachat_200k": self._extract_ultrachat,
            "hackaprompt": self._extract_hackaprompt,
            "gandalf_ignore_instructions": self._extract_gandalf_ignore,
            "gandalf_summarization": self._extract_gandalf_sum,
            "llmail_inject": self._extract_llmail,
            "prompt_injections": self._extract_prompt_inject,
        }

        handler = handlers.get(dataset_name)
        if not handler:
            logger.warning(f"No handler for dataset: {dataset_name}")
            return None

        return handler(idx, example, config)

    # === Dataset-Specific Extractors ===

    def _extract_ultrachat(self, idx: int, ex: Dict, config: Dict) -> Optional[Dict]:
        """Extract from UltraChat messages list."""
        messages = ex.get("messages", [])

        if len(messages) < 2:
            return None

        # Extract last user-assistant turn
        if messages[-1].get("role") != "assistant":
            return None
        if messages[-2].get("role") != "user":
            return None

        return {
            "seed_id": f"ultrachat_{ex.get('prompt_id', idx)}",
            "source_dataset": "ultrachat_200k",
            "prompt": messages[-2].get("content", ""),
            "response": messages[-1].get("content", ""),
            "metadata": {
                "language": "en",
                "safety_flag": "ok",
                "task_hint": "instruction_following",
                "source_example_id": ex.get("prompt_id", str(idx)),
            },
        }

    def _extract_hackaprompt(self, idx: int, ex: Dict, config: Dict) -> Optional[Dict]:
        """Extract hackaprompt as payload for attack library."""
        # Filter errors
        if ex.get("error", True):
            return None

        # Optional: only successful attacks (can be disabled via config)
        filter_correct = config.get("sampling", {}).get("filter_correct", False)
        if filter_correct and not ex.get("correct", False):
            return None

        return {
            "seed_id": f"hackaprompt_{ex.get('level', 0)}_{idx}",
            "source_dataset": "hackaprompt",
            "prompt": ex.get("user_input", ex.get("prompt", "")),  # Fallback to prompt if user_input missing
            "response": "",  # No response - this is attack payload
            "metadata": {
                "language": "en",
                "safety_flag": "adversarial",
                "task_hint": f"level_{ex.get('level', 0)}_attack",
                "source_example_id": str(idx),
                "attack_family": "override",
                "is_payload": True,
                "level": ex.get("level", 0),
            },
        }

    def _extract_gandalf_ignore(self, idx: int, ex: Dict, config: Dict) -> Optional[Dict]:
        """Extract gandalf_ignore_instructions as direct injection payload."""
        text = ex.get("text", "")
        if not text:
            return None

        return {
            "seed_id": f"gandalf_ignore_{idx}",
            "source_dataset": "gandalf_ignore_instructions",
            "prompt": text,
            "response": "",
            "metadata": {
                "language": "en",
                "safety_flag": "adversarial",
                "task_hint": "direct_injection",
                "source_example_id": str(idx),
                "attack_family": "override",
                "is_payload": True,
            },
        }

    def _extract_gandalf_sum(self, idx: int, ex: Dict, config: Dict) -> Optional[Dict]:
        """Extract gandalf_summarization as indirect injection payload."""
        text = ex.get("text", "")
        if not text:
            return None

        # Don't use gandalf_answer - it shows compromised behavior
        return {
            "seed_id": f"gandalf_sum_{idx}",
            "source_dataset": "gandalf_summarization",
            "prompt": text,
            "response": "",
            "metadata": {
                "language": "en",
                "safety_flag": "adversarial",
                "task_hint": "indirect_injection_summarization",
                "source_example_id": str(idx),
                "attack_family": "indirect",
                "is_payload": True,
            },
        }

    def _extract_llmail(self, idx: int, ex: Dict, config: Dict) -> Optional[Dict]:
        """Extract llmail_inject as email injection payload."""
        subject = ex.get("subject", "")
        body = ex.get("body", "")
        scenario = ex.get("scenario", "unknown")

        if not (subject or body):
            return None

        return {
            "seed_id": f"llmail_{scenario}_{ex.get('RowKey', idx)[:8] if isinstance(ex.get('RowKey'), str) else idx}",
            "source_dataset": "llmail_inject",
            "prompt": f"Subject: {subject}\n\nBody: {body}",
            "response": "",
            "metadata": {
                "language": "en",
                "safety_flag": "adversarial",
                "task_hint": f"email_injection_{scenario}",
                "source_example_id": ex.get("RowKey", str(idx)),
                "attack_family": "tool_exfil",
                "is_payload": True,
                "scenario": scenario,
            },
        }

    def _extract_prompt_inject(self, idx: int, ex: Dict, config: Dict) -> Optional[Dict]:
        """Extract prompt_injections - filter to only injections (label=1)."""
        # Only extract injections
        if ex.get("label", 0) != 1:
            return None

        text = ex.get("text", "")
        if not text:
            return None

        return {
            "seed_id": f"prompt_inject_{idx}",
            "source_dataset": "prompt_injections",
            "prompt": text,
            "response": "",
            "metadata": {
                "language": "en",
                "safety_flag": "adversarial",
                "task_hint": "injection",
                "source_example_id": str(idx),
                "attack_family": "override",
                "is_payload": True,
            },
        }

    def _extract_oasst2(self, idx: int, ex: Dict, config: Dict) -> Optional[Dict]:
        """
        Extract OASST2 - requires tree reconstruction.
        This version stores all messages for later tree processing.
        """
        # Language filter
        if ex.get("lang") != "en":
            return None

        # Quality checks
        if ex.get("deleted", False) or not ex.get("review_result", False):
            return None

        # Store message for tree reconstruction (both prompter and assistant)
        # We'll pair them up in a special processing step
        return {
            "seed_id": f"oasst2_{ex.get('message_id', idx)}",
            "source_dataset": "oasst2",
            "prompt": "",  # Will be filled during tree reconstruction
            "response": ex.get("text", "") if ex.get("role") == "assistant" else "",
            "metadata": {
                "language": ex.get("lang", "en"),
                "safety_flag": "ok",
                "task_hint": "conversation",
                "source_example_id": ex.get("message_id", str(idx)),
                # Store for tree reconstruction
                "role": ex.get("role"),
                "parent_id": ex.get("parent_id"),
                "message_id": ex.get("message_id"),
                "message_tree_id": ex.get("message_tree_id"),
                "text": ex.get("text", ""),
            },
        }

    def _passes_filters(self, seed: Dict[str, Any]) -> bool:
        """Check if seed passes quality filters."""
        rules = self.normalization_rules

        prompt = seed["prompt"]
        response = seed["response"]
        is_payload = seed["metadata"].get("is_payload", False)

        # Payload seeds (adversarial data) have relaxed filters
        if is_payload:
            # Only check prompt length for payloads
            if len(prompt) < rules["min_prompt_length"]:
                return False
            if len(prompt) > rules["max_prompt_length"]:
                return False
            # Allow empty responses for payloads
            # Allow adversarial safety_flag
            return True

        # Normal seeds require both prompt and response
        # Length filters
        if len(prompt) < rules["min_prompt_length"]:
            return False
        if len(prompt) > rules["max_prompt_length"]:
            return False
        if len(response) < rules["min_response_length"]:
            return False
        if len(response) > rules["max_response_length"]:
            return False

        # Safety flag check (must be "ok" for non-payloads)
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
