import json
import logging
import time
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.schemas import RetrievedChunk

logger = logging.getLogger(__name__)

INDEX_BASE_DIR = Path(__file__).parent.parent / "data" / "index"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"


class RetrievalEngine:
    """Warm-loaded FAISS retrieval engine.

    Loads the embedding model and all strategy indices ONCE at startup.
    Each strategy's FAISS index and chunk list are kept in memory so that
    a search hit (FAISS row position) maps directly to the chunk via
    simple list indexing — no per-query disk reads.
    """

    def __init__(self, index_dir: Path | str | None = None):
        self.index_dir = Path(index_dir) if index_dir else INDEX_BASE_DIR

        logger.info("Loading embedding model '%s' ...", EMBEDDING_MODEL_NAME)
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        # {strategy_name: {"index": faiss.Index, "chunks": list[dict]}}
        self.strategies: dict[str, dict] = {}

        self._load_all_strategies()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_all_strategies(self) -> None:
        """Discover and load every strategy folder under the index dir."""
        if not self.index_dir.exists():
            raise FileNotFoundError(
                f"Index directory does not exist: {self.index_dir}"
            )

        for strategy_dir in sorted(self.index_dir.iterdir()):
            if not strategy_dir.is_dir():
                continue

            faiss_path = strategy_dir / "faiss.index"
            chunks_path = strategy_dir / "chunks.jsonl"
            meta_path = strategy_dir / "meta.json"

            if not faiss_path.exists() or not chunks_path.exists():
                logger.warning(
                    "Skipping incomplete strategy dir: %s", strategy_dir.name
                )
                continue

            # Load FAISS index
            index = faiss.read_index(str(faiss_path))

            # Load chunks (row-order matches FAISS index positions)
            chunks: list[dict] = []
            with open(chunks_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        chunks.append(json.loads(line))

            # Load metadata
            meta = {}
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)

            assert index.ntotal == len(chunks), (
                f"Strategy '{strategy_dir.name}': FAISS index has {index.ntotal} "
                f"vectors but chunks.jsonl has {len(chunks)} rows"
            )

            self.strategies[strategy_dir.name] = {
                "index": index,
                "chunks": chunks,
                "meta": meta,
            }
            logger.info(
                "Loaded strategy '%s': %d vectors (dim=%d)",
                strategy_dir.name,
                index.ntotal,
                meta.get("dimension", index.d),
            )

        if not self.strategies:
            raise RuntimeError(
                f"No valid strategy indices found under {self.index_dir}"
            )

        logger.info(
            "RetrievalEngine ready — %d strategies loaded: %s",
            len(self.strategies),
            list(self.strategies.keys()),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def available_strategies(self) -> list[str]:
        return list(self.strategies.keys())

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a query string using the same model and normalization as
        build_index.py.  E5 models require the 'query: ' prefix."""
        vec = self.model.encode(
            [f"query: {text}"],
            normalize_embeddings=True,
        )
        return np.array(vec, dtype=np.float32)

    def search(
        self,
        query: str,
        strategy: str = "sentence_semantic",
        top_k: int = 5,
        language: Optional[str] = None,
    ) -> list[RetrievedChunk]:
        """Single-strategy FAISS search with optional language filter.

        Over-fetches ``top_k * 3`` candidates before filtering so the
        result set stays full even when a language filter removes many
        candidates.
        """
        if strategy not in self.strategies:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Available: {self.available_strategies}"
            )

        strat = self.strategies[strategy]
        index: faiss.Index = strat["index"]
        chunks: list[dict] = strat["chunks"]

        query_vec = self.embed_query(query)
        fetch_k = min(top_k * 3, index.ntotal)
        scores, indices = index.search(query_vec, fetch_k)

        results: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = chunks[idx]
            meta = chunk.get("metadata", {})

            # Optional language filter
            if language:
                chunk_lang = meta.get("language", "")
                if language.lower() not in chunk_lang.lower():
                    continue

            results.append(
                RetrievedChunk(
                    chunk_id=chunk["chunk_id"],
                    text=chunk["text"],
                    score=float(score),
                    metadata={
                        **meta,
                        "strategy": chunk.get("strategy", strategy),
                    },
                )
            )
            if len(results) >= top_k:
                break

        return results

    def search_ensemble(
        self,
        query: str,
        top_k: int = 5,
        language: Optional[str] = None,
    ) -> list[RetrievedChunk]:
        """Run search against ALL strategies, deduplicate by text overlap,
        re-rank merged pool by score, return top_k.

        Deduplication uses (query_id, passage text prefix) rather than
        exact chunk_id match, because the same passage appears with
        different chunk_ids across strategies.
        """
        all_candidates: list[RetrievedChunk] = []

        for strat_name in self.strategies:
            hits = self.search(
                query=query,
                strategy=strat_name,
                top_k=top_k,
                language=language,
            )
            all_candidates.extend(hits)

        # Deduplicate: keep highest-scoring version of each passage.
        # Use doc_id from metadata as primary dedup key (uniquely identifies
        # a passage across strategies).  Fall back to text prefix when
        # doc_id is missing.
        seen: dict[str, RetrievedChunk] = {}
        for chunk in all_candidates:
            meta = chunk.metadata or {}
            doc_id = str(meta.get("doc_id", ""))
            if doc_id:
                # doc_id like "1102432_0" uniquely identifies a passage;
                # strip the chunk sub-index added by chunking strategies
                base_doc_id = "_".join(doc_id.split("_")[:2])
                dedup_key = base_doc_id
            else:
                dedup_key = chunk.text[:100].strip()

            existing = seen.get(dedup_key)
            if existing is None or chunk.score > existing.score:
                seen[dedup_key] = chunk

        merged = sorted(seen.values(), key=lambda c: c.score, reverse=True)
        return merged[:top_k]
