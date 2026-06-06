"""
Stage 3: retrieval-augmented grounding.

For each of the top-K concepts in the final rolling state, pull verbatim
chunk passages from the FAISS vector store. The returned mapping is later
injected into the CQ-generation prompt so questions are grounded in actual
paper text (not just the structured state).
"""
from __future__ import annotations

from typing import Dict, List

from utils.rolling_state import top_concept_names
from utils.vectorstore import PaperVectorStore


class PaperRetriever:
    def __init__(self, store: PaperVectorStore, top_k_concepts: int = 20, passages_per_concept: int = 2):
        self.store = store
        self.top_k_concepts = top_k_concepts
        self.passages_per_concept = passages_per_concept

    def retrieve(self, rolling_state: Dict[str, list]) -> Dict[str, List[str]]:
        """Return {concept_name: [passage_text, ...]} for the top-K concepts."""
        names = top_concept_names(rolling_state, k=self.top_k_concepts)
        out: Dict[str, List[str]] = {}
        for name in names:
            hits = self.store.search(name, k=self.passages_per_concept)
            if hits:
                out[name] = [h.text for h in hits]
        return out

    def retrieve_for_persona(self, rolling_state: Dict[str, list], persona_hint: str, n_passages: int = 6) -> List[str]:
        """
        Retrieve passages biased toward a persona's interest area (using their
        question_style or perspective as the query). Returns deduplicated passages.
        """
        if not persona_hint or self.store.n_chunks == 0:
            return []
        hits = self.store.search(persona_hint, k=n_passages)
        seen: set[str] = set()
        out: List[str] = []
        for h in hits:
            if h.text in seen:
                continue
            seen.add(h.text)
            out.append(h.text)
        return out
