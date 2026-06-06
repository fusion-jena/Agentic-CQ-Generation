"""
Stage 1b: build (or load) a FAISS vector store for a single paper's chunks.

Public surface is intentionally small:
  - build_or_load(paper_id, chunks, index_dir) -> PaperVectorStore
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

from utils.embeddings import Embedder
from utils.vectorstore import PaperVectorStore


class PaperIndexer:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder

    def _paths(self, index_dir: Path, paper_id: str) -> tuple[Path, Path]:
        return (
            index_dir / f"{paper_id}.faiss",
            index_dir / f"{paper_id}_meta.json",
        )

    def build_or_load(self, paper_id: str, chunks: List[str], index_dir: Path) -> PaperVectorStore:
        """Build a fresh index from `chunks`, or load a cached one only if both
        the embedder model and chunk count match. A model mismatch forces a
        rebuild — searching a stored index with a different embedder than the
        one that built it produces meaningless cosine scores (different vector
        spaces)."""
        index_path, meta_path = self._paths(index_dir, paper_id)
        store = PaperVectorStore(self.embedder)
        if index_path.exists() and meta_path.exists():
            cached_model = None
            try:
                cached_model = json.loads(meta_path.read_text(encoding="utf-8")).get("model_name")
            except Exception as e:
                logging.warning("[Indexer] Could not read meta for %s: %s - rebuilding", paper_id, e)
            if cached_model == self.embedder.model_name:
                try:
                    store.load(index_path, meta_path)
                    if store.n_chunks == len(chunks):
                        logging.info("[Indexer] Loaded cached index for %s (%d chunks, model=%s)",
                                     paper_id, store.n_chunks, self.embedder.model_name)
                        return store
                    logging.info("[Indexer] Cached index for %s has %d chunks but expected %d - rebuilding",
                                 paper_id, store.n_chunks, len(chunks))
                except Exception as e:
                    logging.warning("[Indexer] Failed to load cached index for %s: %s - rebuilding", paper_id, e)
            elif cached_model is not None:
                logging.info("[Indexer] Cached index for %s was built with '%s' but current embedder is '%s' - rebuilding",
                             paper_id, cached_model, self.embedder.model_name)

        store.build(chunks)
        store.save(index_path, meta_path)
        return store
