"""
Stage E: Render to Model-Specific Training Text

Converts canonical message format to Llama-3 chat template format.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import yaml
from transformers import AutoTokenizer

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

        # Load tokenizer for chat template
        if self.chat_template_config.get("use_official_template", True):
            tokenizer_name = self.chat_template_config["tokenizer_name"]
            logger.info(f"Loading tokenizer: {tokenizer_name}")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            except Exception as e:
                logger.warning(f"Failed to load tokenizer: {e}. Will use manual template.")
                self.tokenizer = None
        else:
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
        stats["train_rendered"] = self._render_split(train_input, train_output)

        # Render test split
        test_input = Path(self.config["input_files"]["test"])
        test_output = Path(self.output_files["test"])
        stats["test_rendered"] = self._render_split(test_input, test_output)

        logger.info(f"Stage E complete: {stats['train_rendered']} train, "
                    f"{stats['test_rendered']} test examples rendered")

        return stats

    def _render_split(self, input_file: Path, output_file: Path) -> int:
        """Render a single split."""
        rendered_count = 0

        with open(input_file) as f_in, open(output_file, "w") as f_out:
            for line in f_in:
                case = json.loads(line)

                # Render to text
                rendered_text = self._render_case(case)

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

        logger.info(f"Rendered {rendered_count} examples to {output_file}")
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
                    logger.warning(f"Case {case['id']} exceeds max tokens: "
                                   f"{len(tokens)} > {max_tokens}")
                    return False

        # Check for special tokens
        if validation_config.get("validate_special_tokens", True):
            special_tokens = self.chat_template_config["special_tokens"]
            if special_tokens["bos"] not in text:
                logger.warning(f"Case {case['id']} missing BOS token")
                return False

        return True


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
