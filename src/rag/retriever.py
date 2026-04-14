"""
Oracle + distractor retriever over the persisted Chroma collection.

Guarantees the known attack document for a given example surfaces at rank 0,
and fills positions 1..k-1 with semantically-retrieved benign distractors.
This decouples retrieval quality from defense behaviour so ASR measures only
the model's response to a marked tool channel.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RetrievedDoc:
    doc_id: str
    kind: str          # "oracle" or "distractor"
    content: str
    rank: int
    score: Optional[float] = None  # cosine distance from Chroma (None for oracle)


class OracleDistractorRetriever:
    """
    Wraps a persisted langchain-chroma Chroma collection.

    The retriever does NOT manage embedding model loading directly — it
    receives a pre-built `Chroma` instance, which in turn owns the
    HuggingFaceEmbeddings used to embed queries on the fly.
    """

    def __init__(
        self,
        config: Dict,
    ):
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
        self.k = config["retrieval"].get("k", 4)
        self.over_fetch = config["retrieval"].get("search_k_multiplier", 4)
        self.distractor_filter = config["retrieval"].get(
            "distractor_filter", {"kind": "distractor"}
        )

        self._vectorstore = None

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def load(self) -> None:
        """Open the persisted Chroma collection."""
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(
            model_name=self.embed_model_name,
            model_kwargs={"device": self.embed_device},
            encode_kwargs={"normalize_embeddings": self.normalize},
        )
        self._vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=embeddings,
            persist_directory=str(self.persist_dir),
        )
        logger.info(
            f"Loaded Chroma collection '{self.collection_name}' from "
            f"{self.persist_dir}"
        )

    def retrieve(self, query: str, oracle_doc_id: str) -> List[RetrievedDoc]:
        """
        Return `k` docs: oracle at rank 0, top `k-1` distractors after.
        """
        if self._vectorstore is None:
            raise RuntimeError("Call load() before retrieve()")

        n_distractors = max(self.k - 1, 0)
        fetch_n = max(n_distractors * self.over_fetch, n_distractors, 1)

        results: List[RetrievedDoc] = []

        oracle_content = self._lookup_doc(oracle_doc_id)
        if oracle_content is None:
            raise KeyError(
                f"Oracle doc '{oracle_doc_id}' not found in collection "
                f"'{self.collection_name}'"
            )
        results.append(
            RetrievedDoc(
                doc_id=oracle_doc_id,
                kind="oracle",
                content=oracle_content,
                rank=0,
                score=None,
            )
        )

        if n_distractors == 0:
            return results

        hits = self._vectorstore.similarity_search_with_score(
            query,
            k=fetch_n,
            filter=self.distractor_filter,
        )
        added = 0
        for doc, score in hits:
            if added >= n_distractors:
                break
            doc_id = doc.metadata.get("doc_id") or doc.metadata.get(
                "source_example_id", "unknown"
            )
            if doc_id == oracle_doc_id:
                continue
            results.append(
                RetrievedDoc(
                    doc_id=doc_id,
                    kind=doc.metadata.get("kind", "distractor"),
                    content=doc.page_content,
                    rank=len(results),
                    score=float(score),
                )
            )
            added += 1

        if added < n_distractors:
            logger.warning(
                f"Only {added}/{n_distractors} distractors retrieved for "
                f"oracle '{oracle_doc_id}' (collection may be too small)"
            )
        return results

    # ------------------------------------------------------------------ #
    # Internals                                                            #
    # ------------------------------------------------------------------ #

    def _lookup_doc(self, doc_id: str) -> Optional[str]:
        """Fetch a document by its primary id from the Chroma collection."""
        # langchain-chroma exposes the underlying Chroma client at `._collection`.
        # `.get(ids=[...])` is a stable API in chromadb >=0.4 that returns a
        # dict with parallel "documents" / "metadatas" / "ids" lists.
        try:
            res = self._vectorstore._collection.get(ids=[doc_id])
        except Exception as e:
            logger.error(f"Chroma .get(ids=[{doc_id!r}]) failed: {e}")
            return None
        docs = res.get("documents") or []
        if not docs:
            return None
        return docs[0]
