"""
RAG-enabled predictor (Phase 2b / 5b).

Subclasses ModelPredictor and replaces the per-example loop:
  1. Iterate only `tool_output_misaligned` rows (the RAG-eligible cohort).
  2. Strip the inline `[TOOL_OUTPUT_UNTRUSTED]<tool>…</tool>` payload out of
     the user turn to recover a benign retrieval query.
  3. Retrieve `k` docs (oracle at rank 0 + distractors) from Chroma.
  4. Rebuild messages as `[system, user, tool*k]` with the model-family-
     appropriate tool role (`ipython` for Llama, `tool` for Qwen).
  5. Generate via the parent's helper; persist a prediction with an extra
     `retrieval` block.

Model loading, generation, and saving are inherited unchanged from
ModelPredictor.
"""

import json
import logging
from dataclasses import asdict
from typing import Dict, List, Optional

from tqdm import tqdm

from src.evaluation.predict import ModelPredictor

from .payload_extractor import PayloadExtractionError, extract_payload
from .retriever import OracleDistractorRetriever, RetrievedDoc

logger = logging.getLogger(__name__)

_TOOL_ROLE_BY_FAMILY = {
    "llama": "ipython",
    "qwen": "tool",
}


class RAGPredictor(ModelPredictor):
    def __init__(
        self,
        model_id: str,
        output_path: str,
        retriever: OracleDistractorRetriever,
        model_family: str,
        adapter_path: Optional[str] = None,
        max_new_tokens: int = 512,
        rag_scenario: str = "tool_output_misaligned",
        limit: Optional[int] = None,
    ):
        super().__init__(
            model_id=model_id,
            output_path=output_path,
            adapter_path=adapter_path,
            max_new_tokens=max_new_tokens,
        )
        if model_family not in _TOOL_ROLE_BY_FAMILY:
            raise ValueError(
                f"model_family must be one of {list(_TOOL_ROLE_BY_FAMILY)}; "
                f"got {model_family!r}"
            )
        self.retriever = retriever
        self.model_family = model_family
        self.tool_role = _TOOL_ROLE_BY_FAMILY[model_family]
        self.rag_scenario = rag_scenario
        self.limit = limit

    # ------------------------------------------------------------------ #
    # Override predict()                                                   #
    # ------------------------------------------------------------------ #

    def predict(self, test_data_path: str) -> None:
        examples = self._load_examples(test_data_path)
        eligible = [e for e in examples if e.get("scenario") == self.rag_scenario]
        logger.info(
            f"Filtered to {len(eligible)} '{self.rag_scenario}' examples "
            f"(of {len(examples)} total)"
        )
        if self.limit is not None:
            eligible = eligible[: self.limit]
            logger.info(f"Limit={self.limit}; processing {len(eligible)} examples")

        self.predictions = []
        skipped: List[Dict] = []

        for example in tqdm(eligible, desc=f"RAG predict ({self.model_family})"):
            user_msg = next(
                (m for m in example.get("messages", []) if m.get("role") == "user"),
                None,
            )
            if user_msg is None:
                skipped.append({"id": example.get("id"), "reason": "no_user_message"})
                continue
            try:
                stripped_query, _ = extract_payload(user_msg["content"])
            except PayloadExtractionError as e:
                skipped.append(
                    {"id": example.get("id"), "reason": f"extraction_failed: {e}"}
                )
                continue

            try:
                retrieved = self.retriever.retrieve(
                    query=stripped_query, oracle_doc_id=example["id"]
                )
            except KeyError as e:
                skipped.append(
                    {"id": example.get("id"), "reason": f"oracle_missing: {e}"}
                )
                continue

            prompt_messages = self._build_rag_messages(
                example, stripped_query, retrieved
            )
            model_output = self._generate_from_messages(prompt_messages)
            self.predictions.append(
                self._build_record(
                    example,
                    prompt_messages,
                    model_output,
                    extra={
                        "retrieval": self._retrieval_meta(
                            retrieved, stripped_query, example["id"]
                        )
                    },
                )
            )

        if skipped:
            logger.warning(
                f"Skipped {len(skipped)} examples during RAG predict "
                f"(first 3: {skipped[:3]})"
            )
            # Persist alongside predictions for auditing
            self.output_path.mkdir(parents=True, exist_ok=True)
            with open(self.output_path / "skipped_examples.jsonl", "w") as f:
                for row in skipped:
                    f.write(json.dumps(row) + "\n")

        logger.info(f"Generated {len(self.predictions)} RAG predictions")

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _build_rag_messages(
        self,
        example: Dict,
        stripped_query: str,
        retrieved: List[RetrievedDoc],
    ) -> List[Dict]:
        system_msg = next(
            (m for m in example.get("messages", []) if m.get("role") == "system"),
            None,
        )
        messages: List[Dict] = []
        if system_msg is not None:
            messages.append({"role": "system", "content": system_msg["content"]})
        messages.append({"role": "user", "content": stripped_query})
        for doc in retrieved:
            messages.append({"role": self.tool_role, "content": doc.content})
        return messages

    def _retrieval_meta(
        self,
        retrieved: List[RetrievedDoc],
        stripped_query: str,
        oracle_doc_id: str,
    ) -> Dict:
        return {
            "k": len(retrieved),
            "oracle_doc_id": oracle_doc_id,
            "oracle_rank": next(
                (d.rank for d in retrieved if d.kind == "oracle"), -1
            ),
            "retrieved": [
                {
                    "doc_id": d.doc_id,
                    "kind": d.kind,
                    "rank": d.rank,
                    "score": d.score,
                }
                for d in retrieved
            ],
            "stripped_query": stripped_query,
            "embedder": self.retriever.embed_model_name,
            "tool_role": self.tool_role,
        }
