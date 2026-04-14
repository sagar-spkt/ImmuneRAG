"""
Corpus builder for the RAG evaluation track.

Extracts oracle attack documents from `tool_output_misaligned` test examples
and a benign distractor pool from `open_aligned` test examples, embeds both
with a sentence-transformers model, and persists them into a ChromaDB
collection via langchain-chroma.

Idempotent: re-running with the same persist_dir + an existing manifest is a
no-op unless `rebuild=True`.
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .payload_extractor import PayloadExtractionError, extract_payload

logger = logging.getLogger(__name__)


@dataclass
class CorpusStats:
    oracle_docs: int = 0
    distractor_docs: int = 0
    skipped_extractions: int = 0
    denylisted_distractors: int = 0
    skipped_records: List[Dict] = field(default_factory=list)


class CorpusBuilder:
    """
    Build the Chroma collection that backs RAG retrieval at eval time.

    Usage:
        builder = CorpusBuilder(config)
        stats = builder.build(test_data_path)
        builder.write_manifest(stats, test_data_path)
    """

    def __init__(self, config: Dict):
        """
        Args:
            config: The `rag` sub-dict of config/rag_config.yaml.
        """
        self.config = config
        self.persist_dir = Path(config["chroma"]["persist_dir"])
        self.collection_name = config["chroma"]["collection_name"]
        self.embed_model_name = config["embedding"]["model"]
        self.embed_device = config["embedding"].get("device", "cpu")
        self.normalize = config["embedding"].get("normalize_embeddings", True)

        corpus_cfg = config["corpus"]
        self.oracle_scenario = corpus_cfg["oracle_source_scenario"]
        self.distractor_scenario = corpus_cfg["distractor_source_scenario"]
        self.distractor_limit = corpus_cfg.get("distractor_limit", 400)
        self.distractor_min_chars = corpus_cfg.get("distractor_min_chars", 200)
        self.distractor_max_chars = corpus_cfg.get("distractor_max_chars", 4000)
        self.denylist = [s.lower() for s in corpus_cfg.get("distractor_denylist", [])]

        self._vectorstore = None
        self._embeddings = None

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def build(
        self,
        test_data_path: str,
        limit_oracles: Optional[int] = None,
        limit_distractors: Optional[int] = None,
        rebuild: bool = False,
    ) -> CorpusStats:
        """
        Extract docs, embed, and persist into Chroma.

        Args:
            test_data_path: Path to data/final/test.jsonl.
            limit_oracles: Cap on oracle docs (for smoke testing).
            limit_distractors: Cap on distractor docs (for smoke testing).
            rebuild: If True, delete the existing collection first.
        """
        examples = self._load_test_data(test_data_path)
        oracle_docs, stats = self._extract_oracles(examples, limit_oracles)
        distractor_docs, denylisted = self._build_distractors(
            examples, limit_distractors
        )
        stats.distractor_docs = len(distractor_docs)
        stats.denylisted_distractors = denylisted

        all_docs = oracle_docs + distractor_docs
        all_ids = [d["id"] for d in all_docs]
        all_texts = [d["text"] for d in all_docs]
        all_metas = [d["metadata"] for d in all_docs]

        self.persist_dir.mkdir(parents=True, exist_ok=True)
        vs = self._open_vectorstore(rebuild=rebuild)

        logger.info(
            f"Adding {len(all_docs)} docs to Chroma collection "
            f"'{self.collection_name}' at {self.persist_dir}"
        )
        # langchain-chroma upserts by id when ids are supplied.
        vs.add_texts(texts=all_texts, metadatas=all_metas, ids=all_ids)

        logger.info(
            f"Corpus built: {stats.oracle_docs} oracle + "
            f"{stats.distractor_docs} distractor "
            f"({stats.skipped_extractions} skipped, "
            f"{stats.denylisted_distractors} denylisted)"
        )
        return stats

    def write_manifest(
        self,
        stats: CorpusStats,
        test_data_path: str,
        manifest_path: Optional[Path] = None,
        skipped_path: Optional[Path] = None,
    ) -> None:
        """Persist corpus manifest and per-skip log to disk."""
        manifest_path = Path(
            manifest_path or self.config["output"]["manifest_path"]
        )
        skipped_path = Path(
            skipped_path or self.config["output"]["skipped_examples_path"]
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        skipped_path.parent.mkdir(parents=True, exist_ok=True)

        manifest = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "test_data_path": str(test_data_path),
            "test_data_sha256": _hash_file(test_data_path),
            "embedder": self.embed_model_name,
            "collection_name": self.collection_name,
            "persist_dir": str(self.persist_dir),
            "oracle_source_scenario": self.oracle_scenario,
            "distractor_source_scenario": self.distractor_scenario,
            "oracle_docs": stats.oracle_docs,
            "distractor_docs": stats.distractor_docs,
            "skipped_extractions": stats.skipped_extractions,
            "denylisted_distractors": stats.denylisted_distractors,
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Manifest written to {manifest_path}")

        with open(skipped_path, "w") as f:
            for row in stats.skipped_records:
                f.write(json.dumps(row) + "\n")
        if stats.skipped_records:
            logger.warning(
                f"{len(stats.skipped_records)} examples skipped; see {skipped_path}"
            )

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _load_test_data(self, path: str) -> List[Dict]:
        logger.info(f"Loading test data from {path}")
        with open(path) as f:
            examples = [json.loads(line) for line in f]
        logger.info(f"Loaded {len(examples)} test examples")
        return examples

    def _extract_oracles(
        self, examples: List[Dict], limit: Optional[int]
    ) -> tuple:
        stats = CorpusStats()
        docs: List[Dict] = []
        for ex in examples:
            if ex.get("scenario") != self.oracle_scenario:
                continue
            user_msg = next(
                (m for m in ex.get("messages", []) if m.get("role") == "user"),
                None,
            )
            if user_msg is None:
                stats.skipped_extractions += 1
                stats.skipped_records.append(
                    {"id": ex.get("id"), "reason": "no_user_message"}
                )
                continue
            try:
                stripped_query, payload = extract_payload(user_msg["content"])
            except PayloadExtractionError as e:
                stats.skipped_extractions += 1
                stats.skipped_records.append(
                    {"id": ex.get("id"), "reason": f"extraction_failed: {e}"}
                )
                continue

            attack_family = ex.get("notes", {}).get("attack_family") or "unknown"
            docs.append(
                {
                    "id": ex["id"],
                    "text": payload,
                    "metadata": {
                        "doc_id": ex["id"],
                        "kind": "oracle",
                        "scenario": self.oracle_scenario,
                        "attack_family": attack_family,
                        "source_example_id": ex["id"],
                        "stripped_query": stripped_query,
                    },
                }
            )
            if limit is not None and len(docs) >= limit:
                break

        stats.oracle_docs = len(docs)
        logger.info(
            f"Extracted {len(docs)} oracle docs "
            f"({stats.skipped_extractions} extraction failures)"
        )
        return docs, stats

    def _build_distractors(
        self, examples: List[Dict], limit: Optional[int]
    ) -> tuple:
        cap = limit if limit is not None else self.distractor_limit
        docs: List[Dict] = []
        denylisted = 0
        for ex in examples:
            if ex.get("scenario") != self.distractor_scenario:
                continue
            text = self._distractor_text(ex)
            if not text:
                continue
            if len(text) < self.distractor_min_chars:
                continue
            if len(text) > self.distractor_max_chars:
                text = text[: self.distractor_max_chars]
            if self._is_denylisted(text):
                denylisted += 1
                continue
            docs.append(
                {
                    "id": f"distractor::{ex['id']}",
                    "text": text,
                    "metadata": {
                        "doc_id": f"distractor::{ex['id']}",
                        "kind": "distractor",
                        "scenario": self.distractor_scenario,
                        "source_example_id": ex["id"],
                    },
                }
            )
            if len(docs) >= cap:
                break

        logger.info(
            f"Built {len(docs)} distractor docs ({denylisted} denylisted)"
        )
        return docs, denylisted

    @staticmethod
    def _distractor_text(example: Dict) -> str:
        parts = []
        for msg in example.get("messages", []):
            role = msg.get("role")
            if role in ("user", "assistant"):
                parts.append(msg.get("content", "").strip())
        return "\n\n".join(p for p in parts if p)

    def _is_denylisted(self, text: str) -> bool:
        lowered = text.lower()
        return any(token in lowered for token in self.denylist)

    def _open_vectorstore(self, rebuild: bool):
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings

        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.embed_model_name,
                model_kwargs={"device": self.embed_device},
                encode_kwargs={"normalize_embeddings": self.normalize},
            )

        if rebuild and self.persist_dir.exists():
            import shutil
            shutil.rmtree(self.persist_dir)
            logger.info(f"Removed existing Chroma dir {self.persist_dir}")
            self.persist_dir.mkdir(parents=True, exist_ok=True)

        self._vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self._embeddings,
            persist_directory=str(self.persist_dir),
        )
        return self._vectorstore


def _hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
