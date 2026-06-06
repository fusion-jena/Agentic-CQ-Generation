"""
FAISS-backed per-paper vector store.

Each paper gets its own index file (`<paper_id>.faiss`) and a sidecar JSON
that maps row-id -> chunk text. Inner-product on L2-normalised vectors = cosine.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

from utils.embeddings import Embedder


@dataclass
class RetrievedChunk:
    text: str
    score: float
    chunk_id: int


class PaperVectorStore:
    """FAISS IndexFlatIP over normalised embeddings. One instance == one paper."""

    def __init__(self, embedder: Embedder):
        try:
            import faiss  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "faiss-cpu is required for v2 pipeline retrieval. "
                "Install with: pip install faiss-cpu"
            ) from e
        self.embedder = embedder
        self._faiss = __import__("faiss")
        self._index = None
        self._chunks: List[str] = []

    def build(self, chunks: List[str]) -> None:
        """Embed and index a list of chunk texts."""
        if not chunks:
            raise ValueError("Cannot build a vector store from an empty chunk list.")
        vecs = self.embedder.embed_documents(chunks)
        index = self._faiss.IndexFlatIP(self.embedder.dim)
        index.add(vecs)
        self._index = index
        self._chunks = list(chunks)
        logging.info("[VectorStore] Indexed %d chunks (dim=%d)", len(chunks), self.embedder.dim)

    def search(self, query: str, k: int = 5) -> List[RetrievedChunk]:
        if self._index is None:
            raise RuntimeError("Vector store has not been built or loaded.")
        if k <= 0 or len(self._chunks) == 0:
            return []
        k = min(k, len(self._chunks))
        qvec = self.embedder.embed_query(query).reshape(1, -1)
        scores, ids = self._index.search(qvec, k)
        out: List[RetrievedChunk] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            out.append(RetrievedChunk(text=self._chunks[int(idx)], score=float(score), chunk_id=int(idx)))
        return out

    def save(self, index_path: Path, meta_path: Path) -> None:
        if self._index is None:
            raise RuntimeError("Nothing to save - vector store is empty.")
        index_path.parent.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self._index, str(index_path))
        meta = {
            "model_name": self.embedder.model_name,
            "dim":        self.embedder.dim,
            "chunks":     self._chunks,
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        logging.info("[VectorStore] Saved index -> %s (%d chunks)", index_path, len(self._chunks))

    def load(self, index_path: Path, meta_path: Path) -> None:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("model_name") != self.embedder.model_name:
            logging.warning(
                "[VectorStore] Loaded index was built with '%s' but current embedder is '%s'. "
                "Search results may be inconsistent.",
                meta.get("model_name"), self.embedder.model_name,
            )
        self._index = self._faiss.read_index(str(index_path))
        self._chunks = list(meta["chunks"])

    @property
    def n_chunks(self) -> int:
        return len(self._chunks)
