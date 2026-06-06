"""
Local sentence-transformer embedder.

Default model: BAAI/bge-small-en-v1.5 (~130 MB, 384-dim, strong on technical
English). The model is lazy-loaded on first call and cached at process level.

Why local (not Ollama):
  We tried bge-m3 via Ollama (1024-dim, retrieval-tuned). It works in
  isolation (curl, fresh Python interpreter) but the long-running pipeline
  process gets 500s during stages 5 and 6 even with: exponential backoff
  (5/15/30/60/120s), 1s process-wide throttling, and Connection: close to
  force fresh TCP connections per call. The most likely cause is that the
  shared Ollama upstream evicts bge-m3 from memory while heavy LLM models
  (gemma4:31b, deepseek-r1:32b) are loaded for stages 4-5, and the embed
  endpoint then returns 500 for ~minutes at a time. Standalone probes from
  the same machine succeed because they don't carry the same client-side
  state or trigger the same eviction window. Local sentence-transformers
  has no network dependency, so the pipeline always completes.

A process-level cache (dict keyed by sha1(text)) avoids re-embedding the
same string within a single paper run.
"""
from __future__ import annotations

import hashlib
import logging
import threading
from functools import lru_cache
from typing import List

import numpy as np

_DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
_model_lock = threading.Lock()

_CACHE: dict[tuple[str, str], np.ndarray] = {}
_CACHE_LOCK = threading.Lock()


def _text_key(model: str, text: str) -> tuple[str, str]:
    return (model, hashlib.sha1(text.encode("utf-8")).hexdigest())


@lru_cache(maxsize=4)
def _load_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is required for the v2 pipeline embedder. "
            "Install with: pip install sentence-transformers"
        ) from e
    logging.info("[Embedder] Loading local model '%s' (first call may download weights)", model_name)
    return SentenceTransformer(model_name)


class Embedder:
    """Local sentence-transformer wrapper. Returns L2-normalised float32 vectors
    so dot product = cosine. Same surface as the previous Ollama-backed embedder:
      - embed_documents(list[str]) -> ndarray(n, dim)
      - embed_query(str) -> ndarray(dim)
      - cache_stats() -> dict
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL):
        self.model_name = model_name
        with _model_lock:
            self.model = _load_model(model_name)
        get_dim = getattr(self.model, "get_embedding_dimension", None) \
            or self.model.get_sentence_embedding_dimension
        self.dim = int(get_dim())

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        slots: List[np.ndarray | None] = [None] * len(texts)
        missing_idx: List[int] = []
        for i, t in enumerate(texts):
            k = _text_key(self.model_name, t)
            with _CACHE_LOCK:
                hit = _CACHE.get(k)
            if hit is not None:
                slots[i] = hit
            else:
                missing_idx.append(i)
        if missing_idx:
            missing_texts = [texts[i] for i in missing_idx]
            new_vecs = self.model.encode(
                missing_texts,
                batch_size=32,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).astype(np.float32)
            # Guard: NaN/Inf can appear when the model's pooling produces a
            # near-zero vector and normalize_embeddings divides by ~0.
            new_vecs = np.nan_to_num(new_vecs, nan=0.0, posinf=0.0, neginf=0.0)
            norms = np.linalg.norm(new_vecs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            new_vecs = new_vecs / norms
            for idx, vec in zip(missing_idx, new_vecs):
                k = _text_key(self.model_name, texts[idx])
                with _CACHE_LOCK:
                    _CACHE[k] = vec
                slots[idx] = vec
        return np.vstack(slots).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        text = _BGE_QUERY_INSTRUCTION + query if "bge" in self.model_name.lower() else query
        k = _text_key(self.model_name, text)
        with _CACHE_LOCK:
            hit = _CACHE.get(k)
        if hit is not None:
            return hit
        vec = self.model.encode(
            [text],
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)[0]
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        with _CACHE_LOCK:
            _CACHE[k] = vec
        return vec

    @staticmethod
    def cache_stats() -> dict:
        with _CACHE_LOCK:
            return {"cache_size": len(_CACHE)}
