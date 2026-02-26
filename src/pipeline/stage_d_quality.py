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
from typing import Dict, Any, List, Set, Tuple
from collections import defaultdict
import yaml

try:
    from datasketch import MinHash, MinHashLSH
    MINHASH_AVAILABLE = True
except ImportError:
    MINHASH_AVAILABLE = False
    logging.warning("datasketch not available. MinHash LSH deduplication will fall back to exact hash.")

try:
    from transformers import AutoTokenizer
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False
    logging.warning("transformers not available. Token count validation will be skipped.")

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

        # Statistics tracking
        self.dedup_report = {
            "total_input": 0,
            "duplicates_removed": 0,
            "by_scenario": {},
        }

        # Load tokenizer for token count validation
        self.tokenizer = None
        if TOKENIZER_AVAILABLE and self.config["quality_filters"].get("min_total_tokens") is not None:
            try:
                # Use Llama-3 tokenizer for consistency with training
                self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
                logger.info("Loaded tokenizer for token count validation")
            except Exception as e:
                logger.warning(f"Failed to load tokenizer: {e}. Token validation will be skipped.")

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

    def _create_shingles(self, text: str, k: int = 3) -> Set[str]:
        """
        Create character-level k-shingles from text.

        Args:
            text: Input text
            k: Shingle size (default: 3)

        Returns:
            Set of k-character shingles
        """
        shingles = set()
        for i in range(len(text) - k + 1):
            shingles.add(text[i:i+k])
        return shingles

    def _get_messages_text(self, case: Dict[str, Any]) -> str:
        """Extract concatenated message text from case."""
        messages = case.get("messages", [])
        # Concatenate all message content (role + content)
        text_parts = []
        for msg in messages:
            text_parts.append(f"{msg.get('role', '')}:{msg.get('content', '')}")
        return " ".join(text_parts)

    def _deduplicate(self) -> None:
        """Remove near-duplicate cases using MinHash LSH or exact hash."""
        dedup_config = self.config["deduplication"]
        method = dedup_config.get("method", "exact_hash")
        threshold = dedup_config.get("similarity_threshold", 0.85)

        self.dedup_report["total_input"] = len(self.cases)

        logger.info(f"Deduplication using method: {method}")

        if method == "minhash_lsh" and MINHASH_AVAILABLE:
            self._deduplicate_minhash_lsh(threshold)
        else:
            if method == "minhash_lsh" and not MINHASH_AVAILABLE:
                logger.warning("MinHash LSH requested but datasketch not available. Falling back to exact hash.")
            self._deduplicate_exact_hash()

        removed = self.dedup_report["total_input"] - len(self.cases)
        self.dedup_report["duplicates_removed"] = removed

        if removed > 0:
            logger.info(f"Removed {removed} duplicate cases")

    def _deduplicate_exact_hash(self) -> None:
        """Exact hash deduplication (fallback method)."""
        seen = set()
        deduplicated = []

        for case in self.cases:
            # Hash based on messages content
            messages_str = json.dumps(case["messages"], sort_keys=True)
            case_hash = hashlib.md5(messages_str.encode()).hexdigest()

            if case_hash not in seen:
                seen.add(case_hash)
                deduplicated.append(case)
            else:
                # Track by scenario
                scenario = case.get("scenario", "unknown")
                self.dedup_report["by_scenario"][scenario] = \
                    self.dedup_report["by_scenario"].get(scenario, 0) + 1

        self.cases = deduplicated

    def _deduplicate_minhash_lsh(self, threshold: float) -> None:
        """
        Near-duplicate detection using MinHash LSH.

        Args:
            threshold: Jaccard similarity threshold (default: 0.85)
        """
        # Create LSH index
        lsh = MinHashLSH(threshold=threshold, num_perm=128)
        minhashes = {}

        # Build MinHash signatures
        for idx, case in enumerate(self.cases):
            # Get text representation
            text = self._get_messages_text(case)

            # Create shingles (3-grams)
            shingles = self._create_shingles(text, k=3)

            if len(shingles) == 0:
                # Skip empty cases
                continue

            # Create MinHash signature
            minhash = MinHash(num_perm=128)
            for shingle in shingles:
                minhash.update(shingle.encode('utf-8'))

            minhashes[idx] = minhash

        # Find duplicates using LSH
        duplicates = set()

        for idx, minhash in minhashes.items():
            if idx in duplicates:
                continue

            # Query LSH for similar items
            # Insert into LSH and query
            lsh.insert(str(idx), minhash)

        # Second pass: find duplicates
        for idx, minhash in minhashes.items():
            if idx in duplicates:
                continue

            # Query for similar items
            similar = lsh.query(minhash)
            similar_indices = [int(s) for s in similar if s != str(idx)]

            # Mark later indices as duplicates (keep first)
            for sim_idx in similar_indices:
                if sim_idx > idx and sim_idx not in duplicates:
                    duplicates.add(sim_idx)

                    # Track by scenario
                    scenario = self.cases[sim_idx].get("scenario", "unknown")
                    self.dedup_report["by_scenario"][scenario] = \
                        self.dedup_report["by_scenario"].get(scenario, 0) + 1

        # Remove duplicates
        self.cases = [case for idx, case in enumerate(self.cases) if idx not in duplicates]

        logger.info(f"MinHash LSH deduplication: found {len(duplicates)} near-duplicates "
                   f"(threshold={threshold})")

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

    def _validate_message_ordering(self, messages: List[Dict[str, str]]) -> bool:
        """
        Validate message role sequence is valid.

        Valid patterns:
        - [system, user, assistant]
        - [user, assistant]
        - [system, user, assistant, user, assistant, ...]

        Invalid:
        - Consecutive same roles
        - assistant before user
        - user after assistant without another user

        Args:
            messages: List of message dictionaries

        Returns:
            True if valid, False otherwise
        """
        if len(messages) == 0:
            return False

        roles = [msg.get("role") for msg in messages]

        # Check for consecutive same roles
        for i in range(len(roles) - 1):
            if roles[i] == roles[i+1]:
                return False

        # Check valid role transitions
        # First message must be system or user
        if roles[0] not in ["system", "user"]:
            return False

        # If starts with system, second must be user
        if roles[0] == "system" and len(roles) > 1:
            if roles[1] != "user":
                return False

        # After first user, must alternate user <-> assistant
        # Find first user
        first_user_idx = 0
        for i, role in enumerate(roles):
            if role == "user":
                first_user_idx = i
                break

        # Check alternation after first user
        expected_role = "assistant"
        for i in range(first_user_idx + 1, len(roles)):
            if roles[i] != expected_role:
                return False
            # Alternate between assistant and user
            expected_role = "user" if expected_role == "assistant" else "assistant"

        return True

    def _count_tokens(self, case: Dict[str, Any]) -> int:
        """
        Count total tokens in case using tokenizer.

        Args:
            case: Case dictionary

        Returns:
            Total token count (or 0 if tokenizer unavailable)
        """
        if not self.tokenizer:
            return 0

        # Concatenate all message content
        text = self._get_messages_text(case)

        # Tokenize and count
        tokens = self.tokenizer.encode(text)
        return len(tokens)

    def _passes_quality_filters(self, case: Dict[str, Any], filters: Dict) -> bool:
        """Check if case passes quality filters."""
        messages = case.get("messages", [])

        # Message count
        if len(messages) < filters["min_message_count"]:
            return False
        if len(messages) > filters["max_message_count"]:
            return False

        # Require assistant response
        if filters["require_assistant_response"]:
            if not any(msg.get("role") == "assistant" for msg in messages):
                return False

        # Check empty content
        for msg in messages:
            content = msg.get("content", "")

            # Filter out non-string content (lists, dicts, etc.)
            # This can happen if LLM returns structured data for extraction tasks
            if not isinstance(content, str):
                logger.warning(
                    f"Case {case.get('id', 'unknown')} has non-string content "
                    f"in {msg.get('role', 'unknown')} message (type: {type(content).__name__}), filtering out"
                )
                return False

            if not content.strip():
                return False

        # Message ordering validation
        if filters.get("check_message_ordering", False):
            if not self._validate_message_ordering(messages):
                return False

        # Token count checks
        if self.tokenizer:
            token_count = self._count_tokens(case)

            min_tokens = filters.get("min_total_tokens", 0)
            max_tokens = filters.get("max_total_tokens", float('inf'))

            if token_count < min_tokens or token_count > max_tokens:
                return False

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

        logger.info(f"Initial split: {len(self.train_cases)} train, {len(self.test_cases)} test")

        # Prevent seed leakage if enabled
        if splitting_config.get("prevent_seed_leakage", False):
            self._prevent_seed_leakage()

    def _prevent_seed_leakage(self) -> None:
        """
        Prevent same source seed from appearing in both train and test.

        If a seed appears in both splits, move all test instances to train
        (to preserve the deterministic test split size as much as possible).
        """
        logger.info("Checking for seed leakage...")

        # Extract source seed IDs
        train_seed_ids = set()
        test_seed_ids = set()

        for case in self.train_cases:
            seed_id = case.get("notes", {}).get("source_seed_id")
            if seed_id:
                train_seed_ids.add(seed_id)

        for case in self.test_cases:
            seed_id = case.get("notes", {}).get("source_seed_id")
            if seed_id:
                test_seed_ids.add(seed_id)

        # Find leaked seeds (appearing in both)
        leaked_seeds = train_seed_ids & test_seed_ids

        if len(leaked_seeds) == 0:
            logger.info("No seed leakage detected.")
            return

        logger.warning(f"Found {len(leaked_seeds)} seeds appearing in both train and test")

        # Move all test cases with leaked seeds to train
        leaked_test_cases = []
        kept_test_cases = []

        for case in self.test_cases:
            seed_id = case.get("notes", {}).get("source_seed_id")
            if seed_id in leaked_seeds:
                case["split"] = "train"
                leaked_test_cases.append(case)
            else:
                kept_test_cases.append(case)

        # Update splits
        self.train_cases.extend(leaked_test_cases)
        self.test_cases = kept_test_cases

        logger.info(f"Moved {len(leaked_test_cases)} test cases to train to prevent leakage")
        logger.info(f"After leakage prevention: {len(self.train_cases)} train, {len(self.test_cases)} test")

    def _map_scenario_to_category(self, scenario: str, alignment: str) -> str:
        """
        Map scenario + alignment to balancing category.

        Args:
            scenario: Scenario name
            alignment: "aligned" or "misaligned"

        Returns:
            Category name matching config targets
        """
        mapping = {
            ("open_aligned", "aligned"): "aligned_open_domain",
            ("sys_probe_aligned", "aligned"): "aligned_system_probing",
            ("open_misaligned", "misaligned"): "misaligned_open_domain",
            ("closed_domain_misaligned", "misaligned"): "misaligned_closed_domain",
            ("tool_output_misaligned", "misaligned"): "misaligned_tool_output",
            ("sys_extract_misaligned", "misaligned"): "misaligned_system_extraction",
        }

        key = (scenario, alignment)
        return mapping.get(key, f"unknown_{scenario}_{alignment}")

    def _balance_splits(self) -> None:
        """
        Balance train/test splits to match target mixture.

        Applies stratified sampling to match exact target counts per category.
        """
        logger.info("Balancing splits...")

        balancing_config = self.config.get("balancing", {})
        train_targets = balancing_config.get("train_targets", {})
        test_targets = balancing_config.get("test_targets", {})

        if not train_targets or not test_targets:
            logger.warning("No balancing targets specified. Skipping balancing.")
            return

        # Group cases by category
        train_by_category = defaultdict(list)
        test_by_category = defaultdict(list)

        for case in self.train_cases:
            category = self._map_scenario_to_category(
                case.get("scenario", "unknown"),
                case.get("alignment", "unknown")
            )
            train_by_category[category].append(case)

        for case in self.test_cases:
            category = self._map_scenario_to_category(
                case.get("scenario", "unknown"),
                case.get("alignment", "unknown")
            )
            test_by_category[category].append(case)

        # Log current distribution
        logger.info("Current distribution before balancing:")
        logger.info("  Train:")
        for cat, cases in sorted(train_by_category.items()):
            target = train_targets.get(cat, "N/A")
            logger.info(f"    {cat}: {len(cases)} (target: {target})")

        logger.info("  Test:")
        for cat, cases in sorted(test_by_category.items()):
            target = test_targets.get(cat, "N/A")
            logger.info(f"    {cat}: {len(cases)} (target: {target})")

        # Balance train split
        balanced_train = []
        import random
        random.seed(42)

        for category, target_count in train_targets.items():
            available = train_by_category.get(category, [])

            if len(available) == 0:
                logger.warning(f"Train: No cases available for category '{category}' (target: {target_count})")
                continue

            if len(available) < target_count:
                logger.warning(f"Train: {category} has only {len(available)} cases (target: {target_count}). Using all available.")
                balanced_train.extend(available)
            else:
                # Random sample to match target
                sampled = random.sample(available, target_count)
                balanced_train.extend(sampled)

        # Balance test split
        balanced_test = []

        for category, target_count in test_targets.items():
            available = test_by_category.get(category, [])

            if len(available) == 0:
                logger.warning(f"Test: No cases available for category '{category}' (target: {target_count})")
                continue

            if len(available) < target_count:
                logger.warning(f"Test: {category} has only {len(available)} cases (target: {target_count}). Using all available.")
                balanced_test.extend(available)
            else:
                # Random sample to match target
                sampled = random.sample(available, target_count)
                balanced_test.extend(sampled)

        self.train_cases = balanced_train
        self.test_cases = balanced_test

        logger.info(f"Balanced: {len(self.train_cases)} train, {len(self.test_cases)} test cases")

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
        stats_config = self.config.get("statistics", {})

        stats = {
            "summary": {
                "total_input": self.dedup_report["total_input"],
                "duplicates_removed": self.dedup_report["duplicates_removed"],
                "train_count": len(self.train_cases),
                "test_count": len(self.test_cases),
            },
            "train": self._compute_split_stats(self.train_cases, stats_config),
            "test": self._compute_split_stats(self.test_cases, stats_config),
        }

        # Save main statistics
        stats_file = Path(self.config["output_files"]["stats"])
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Saved statistics to {stats_file}")

        # Save deduplication report
        if self.dedup_report["duplicates_removed"] > 0:
            dedup_file = Path(self.config["output_files"].get("dedup_report", "data/final/dedup_report.json"))
            with open(dedup_file, "w") as f:
                json.dump(self.dedup_report, f, indent=2)
            logger.info(f"Saved deduplication report to {dedup_file}")

    def _compute_split_stats(self, cases: List[Dict], stats_config: Dict) -> Dict[str, Any]:
        """Compute statistics for a split."""
        stats = {
            "total": len(cases),
            "by_scenario": defaultdict(int),
            "by_alignment": defaultdict(int),
        }

        # Basic counts
        for case in cases:
            stats["by_scenario"][case.get("scenario", "unknown")] += 1
            stats["by_alignment"][case.get("alignment", "unknown")] += 1

        # Convert defaultdicts to regular dicts
        stats["by_scenario"] = dict(stats["by_scenario"])
        stats["by_alignment"] = dict(stats["by_alignment"])

        # Count by category (for balancing verification)
        by_category = defaultdict(int)
        for case in cases:
            category = self._map_scenario_to_category(
                case.get("scenario", "unknown"),
                case.get("alignment", "unknown")
            )
            by_category[category] += 1
        stats["by_category"] = dict(by_category)

        # Attack family distribution (for misaligned cases)
        attack_families = defaultdict(int)
        for case in cases:
            if case.get("alignment") == "misaligned":
                family = case.get("notes", {}).get("attack_family")
                if family:
                    attack_families[family] += 1
        if attack_families:
            stats["attack_families"] = dict(attack_families)

        # Token length histogram
        if stats_config.get("token_length_histogram", False) and self.tokenizer:
            bins = stats_config.get("bins", [0, 256, 512, 1024, 2048, 4096])
            token_lengths = []

            for case in cases:
                token_count = self._count_tokens(case)
                if token_count > 0:
                    token_lengths.append(token_count)

            # Create histogram
            histogram = {f"{bins[i]}-{bins[i+1]}": 0 for i in range(len(bins)-1)}
            histogram[f"{bins[-1]}+"] = 0

            for length in token_lengths:
                placed = False
                for i in range(len(bins)-1):
                    if bins[i] <= length < bins[i+1]:
                        histogram[f"{bins[i]}-{bins[i+1]}"] += 1
                        placed = True
                        break
                if not placed:
                    histogram[f"{bins[-1]}+"] += 1

            stats["token_length_histogram"] = histogram

            if token_lengths:
                stats["token_stats"] = {
                    "min": min(token_lengths),
                    "max": max(token_lengths),
                    "mean": sum(token_lengths) / len(token_lengths),
                }

        # Message count distribution
        if stats_config.get("message_count_distribution", False):
            msg_counts = defaultdict(int)
            for case in cases:
                count = len(case.get("messages", []))
                msg_counts[count] += 1
            stats["message_count_distribution"] = dict(msg_counts)

        return stats


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
