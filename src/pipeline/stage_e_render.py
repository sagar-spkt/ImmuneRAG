"""
Stage E: Render to Model-Specific Training Text

Converts canonical message format to Llama-3 chat template format.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict
import yaml

try:
    from transformers import AutoTokenizer
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False
    logging.warning("transformers not available. Using manual template only.")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    logging.warning("tqdm not available. Progress bars disabled.")

logger = logging.getLogger(__name__)


class ModelRenderer:
    """Renders canonical format to model-specific chat format."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize model renderer.

        Args:
            config: Configuration from pipeline_config.yaml (stage_e_render)
        """
        self.config = config
        self.input_dir = Path(config["input_dir"])
        self.output_files = config["output_files"]

        self.model_family = config["model_family"]
        self.chat_template_config = config["chat_template"]

        # Statistics tracking
        self.render_stats = {
            "validation_errors": defaultdict(int),
            "token_lengths": [],
            "examples_rendered": [],
        }

        # Load tokenizer for chat template
        if TOKENIZER_AVAILABLE and self.chat_template_config.get("use_official_template", True):
            tokenizer_name = self.chat_template_config["tokenizer_name"]
            logger.info(f"Loading tokenizer: {tokenizer_name}")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                logger.info("Tokenizer loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load tokenizer: {e}. Will use manual template.")
                self.tokenizer = None
        else:
            logger.info("Using manual Llama-3 template (tokenizer disabled in config)")
            self.tokenizer = None

    def run(self) -> Dict[str, Any]:
        """
        Execute Stage E: Render to model format.

        Returns:
            Dictionary containing rendering statistics
        """
        logger.info("Starting Stage E: Model Rendering")

        stats = {
            "train_rendered": 0,
            "test_rendered": 0,
            "validation_errors": 0,
        }

        # Render train split
        train_input = Path(self.config["input_files"]["train"])
        train_output = Path(self.output_files["train"])
        stats["train_rendered"] = self._render_split(train_input, train_output, split_name="train")

        # Render test split
        test_input = Path(self.config["input_files"]["test"])
        test_output = Path(self.output_files["test"])
        stats["test_rendered"] = self._render_split(test_input, test_output, split_name="test")

        # Compute and save statistics
        self._save_statistics(stats)

        logger.info(f"Stage E complete: {stats['train_rendered']} train, "
                    f"{stats['test_rendered']} test examples rendered")

        return stats

    def _validate_role_sequence(self, messages: List[Dict[str, str]]) -> Tuple[bool, str]:
        """
        Validate message role sequence.

        Args:
            messages: List of messages

        Returns:
            (is_valid, error_message)
        """
        if len(messages) == 0:
            return False, "Empty message list"

        roles = [msg.get("role") for msg in messages]

        # Check for None roles
        if None in roles:
            return False, "Message with missing role"

        # Check for consecutive same roles
        for i in range(len(roles) - 1):
            if roles[i] == roles[i+1]:
                return False, f"Consecutive {roles[i]} roles"

        # First message must be system or user
        if roles[0] not in ["system", "user"]:
            return False, f"Invalid first role: {roles[0]}"

        # If starts with system, second must be user
        if roles[0] == "system" and len(roles) > 1:
            if roles[1] != "user":
                return False, "System not followed by user"

        # After first user, must alternate user <-> assistant
        first_user_idx = 0
        for i, role in enumerate(roles):
            if role == "user":
                first_user_idx = i
                break

        expected_role = "assistant"
        for i in range(first_user_idx + 1, len(roles)):
            if roles[i] != expected_role:
                return False, f"Expected {expected_role}, got {roles[i]}"
            expected_role = "user" if expected_role == "assistant" else "assistant"

        return True, ""

    def _render_split(self, input_file: Path, output_file: Path, split_name: str = "unknown") -> int:
        """
        Render a single split with progress tracking.

        Args:
            input_file: Input JSONL file
            output_file: Output JSONL file
            split_name: Name of split for logging

        Returns:
            Number of successfully rendered cases
        """
        # Count total lines first for progress bar
        with open(input_file) as f:
            total_cases = sum(1 for _ in f)

        logger.info(f"Rendering {split_name} split: {total_cases} cases")

        rendered_count = 0
        skipped_count = 0
        example_count = 0

        # Create output directory
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Setup progress bar
        if TQDM_AVAILABLE:
            pbar = tqdm(total=total_cases, desc=f"Rendering {split_name}", unit="cases")
        else:
            pbar = None

        with open(input_file) as f_in, open(output_file, "w") as f_out:
            for line in f_in:
                case = json.loads(line)

                # Validate role sequence if enabled
                validation_config = self.config.get("validation", {})
                if validation_config.get("validate_role_sequence", False):
                    is_valid, error = self._validate_role_sequence(case.get("messages", []))
                    if not is_valid:
                        logger.warning(f"Case {case.get('id')} failed role validation: {error}")
                        self.render_stats["validation_errors"]["role_sequence"] += 1
                        skipped_count += 1
                        if pbar:
                            pbar.update(1)
                        continue

                # Render to text
                try:
                    rendered_text = self._render_case(case)
                except Exception as e:
                    logger.error(f"Failed to render case {case.get('id')}: {e}")
                    self.render_stats["validation_errors"]["render_failed"] += 1
                    skipped_count += 1
                    if pbar:
                        pbar.update(1)
                    continue

                if rendered_text and self._validate_rendered(rendered_text, case):
                    output_record = {
                        "id": case["id"],
                        "text": rendered_text,
                    }

                    # Optionally include metadata
                    if self.config["output_format"].get("include_metadata", False):
                        output_record["metadata"] = case.get("notes", {})

                    f_out.write(json.dumps(output_record) + "\n")
                    rendered_count += 1

                    # Track token length
                    if self.tokenizer:
                        tokens = self.tokenizer.encode(rendered_text)
                        self.render_stats["token_lengths"].append(len(tokens))

                    # Save first 3 examples for logging
                    if example_count < 3:
                        self.render_stats["examples_rendered"].append({
                            "split": split_name,
                            "id": case["id"],
                            "scenario": case.get("scenario", "unknown"),
                            "text_preview": rendered_text[:300] + "..." if len(rendered_text) > 300 else rendered_text,
                        })
                        example_count += 1
                else:
                    skipped_count += 1

                if pbar:
                    pbar.update(1)

        if pbar:
            pbar.close()

        logger.info(f"Rendered {rendered_count} examples to {output_file}")
        if skipped_count > 0:
            logger.warning(f"Skipped {skipped_count} cases due to validation failures")

        return rendered_count

    def _render_case(self, case: Dict[str, Any]) -> str:
        """Render a single case to Llama-3 format."""
        messages = case["messages"]

        # Use official tokenizer chat template if available
        if self.tokenizer and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                # Apply chat template
                rendered = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                return rendered
            except Exception as e:
                logger.warning(f"Failed to apply chat template: {e}. Using manual template.")

        # Fallback: Manual Llama-3 template
        return self._manual_llama3_template(messages)

    def _manual_llama3_template(self, messages: List[Dict[str, str]]) -> str:
        """Manual Llama-3 chat template."""
        special_tokens = self.chat_template_config["special_tokens"]

        bos = special_tokens["bos"]
        start_header = special_tokens["start_header"]
        end_header = special_tokens["end_header"]
        eot = special_tokens["eot"]

        # Build formatted string
        formatted = bos

        for message in messages:
            role = message["role"]
            content = message["content"]

            formatted += f"{start_header}{role}{end_header}\n\n{content}{eot}"

        return formatted

    def _validate_rendered(self, text: str, case: Dict[str, Any]) -> bool:
        """Validate rendered text."""
        validation_config = self.config.get("validation", {})

        # Check token count
        if validation_config.get("check_token_count", True):
            if self.tokenizer:
                tokens = self.tokenizer.encode(text)
                max_tokens = validation_config.get("max_tokens", 4096)

                if len(tokens) > max_tokens:
                    logger.warning(f"Case {case.get('id')} exceeds max tokens: "
                                   f"{len(tokens)} > {max_tokens}")
                    self.render_stats["validation_errors"]["token_count_exceeded"] += 1
                    return False

        # Check for special tokens
        if validation_config.get("validate_special_tokens", True):
            special_tokens = self.chat_template_config["special_tokens"]
            if special_tokens["bos"] not in text:
                logger.warning(f"Case {case.get('id')} missing BOS token")
                self.render_stats["validation_errors"]["missing_bos"] += 1
                return False

        return True

    def _save_statistics(self, stats: Dict[str, Any]) -> None:
        """
        Save rendering statistics to file.

        Args:
            stats: Statistics dictionary
        """
        # Compute token length statistics
        token_stats = {}
        if self.render_stats["token_lengths"]:
            lengths = self.render_stats["token_lengths"]
            token_stats = {
                "min": min(lengths),
                "max": max(lengths),
                "mean": sum(lengths) / len(lengths),
                "p50": sorted(lengths)[len(lengths)//2],
                "p95": sorted(lengths)[int(len(lengths)*0.95)],
                "p99": sorted(lengths)[int(len(lengths)*0.99)],
            }

        # Build final statistics
        final_stats = {
            "summary": {
                "train_rendered": stats["train_rendered"],
                "test_rendered": stats["test_rendered"],
                "total_rendered": stats["train_rendered"] + stats["test_rendered"],
            },
            "validation_errors": dict(self.render_stats["validation_errors"]),
            "token_statistics": token_stats,
            "examples": self.render_stats["examples_rendered"],
        }

        # Save to file
        stats_file = Path(self.config["output_files"].get("stats", "data/final/render_stats.json"))
        stats_file.parent.mkdir(parents=True, exist_ok=True)

        with open(stats_file, "w") as f:
            json.dump(final_stats, f, indent=2)

        logger.info(f"Saved rendering statistics to {stats_file}")

        # Log example outputs
        if self.render_stats["examples_rendered"]:
            logger.info("\nExample rendered outputs:")
            for ex in self.render_stats["examples_rendered"][:3]:
                logger.info(f"\n  [{ex['split']}] {ex['id']} ({ex['scenario']}):")
                logger.info(f"  {ex['text_preview']}")


def main():
    """CLI entry point for Stage E."""
    import argparse

    parser = argparse.ArgumentParser(description="Stage E: Model Rendering")
    parser.add_argument(
        "--config",
        default="config/pipeline_config.yaml",
        help="Path to pipeline config",
    )

    args = parser.parse_args()

    # Load pipeline config
    with open(args.config) as f:
        pipeline_config = yaml.safe_load(f)

    stage_config = pipeline_config["pipeline"]["stage_e_render"]

    # Run renderer
    renderer = ModelRenderer(stage_config)
    stats = renderer.run()

    print(f"\nStage E Summary:")
    print(f"  Train rendered: {stats['train_rendered']}")
    print(f"  Test rendered: {stats['test_rendered']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
