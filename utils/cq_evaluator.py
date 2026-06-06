"""
Stage 6: reference-free CQ evaluation (v2.1).

The deliverable of this pipeline is the SET OF QUESTIONS, not the expected
answers. v2 evaluated answer-groundedness; v2.1 reframes the primary metric
to QUESTION FAITHFULNESS (does the question text refer to content actually
in the paper?). Answer verifiability is retained as a diagnostic but does
not contribute to the composite.

Four scored families (down from five — specificity removed from composite):
  1. Coverage              (0.38) - % of extracted state items referenced by CQs
  2. Question Faithfulness (0.44) - cos(question_text, retrieved passages)
  3. Diversity             (0.12) - 1 - mean pairwise cosine across CQ texts
  4. Triplifiability       (0.06) - % of CQs that map to a subject-predicate-object form

Diagnostics (recorded but NOT scored):
  - specificity:           pct_concept + pct_numeric + pct_unit + length proxy.
                           Moved to diagnostics because: (a) pct_concept is already
                           captured by Coverage and the validator's concept_usage check;
                           (b) pct_num/pct_unit systematically penalise valid non-numeric
                           archetypes (GAP_OR_LIMITATION, MOTIVATION, ONTOLOGY_TRIPLE);
                           (c) the 110-char length target is an arbitrary heuristic with
                           no grounding in CQ-evaluation literature.
  - answer_verifiability:  cos(expected_answer, passages) - sanity check on answers
  - per-paper run metadata (model versions, embedder, weights) for reproducibility
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

_CONFIG_DIR = Path(__file__).parent.parent / "config"

# ---------------------------------------------------------------------------
# Weights — 4-family composite.
# Dropped: persona_archetype, distribution_balance (v2→v2.1)
# Dropped: specificity (v2.1 — moved to diagnostics; see module docstring)
# Redistribution: each remaining weight scaled by 1/0.80 so they sum to 1.0.
# ---------------------------------------------------------------------------
_WEIGHTS = {
    "coverage":              0.38,   # was 0.30 → ×(1/0.80)
    "question_faithfulness": 0.44,   # was 0.35 → ×(1/0.80)
    "diversity":             0.12,   # was 0.10 → ×(1/0.80)
    "triplifiability":       0.06,   # was 0.05 → ×(1/0.80)
}

# Thresholds shared with the validator's per-question score
_FAITHFUL_OK_THRESHOLD   = 0.55
_FAITHFUL_GOOD_THRESHOLD = 0.70
_DUP_THRESHOLD           = 0.92

_NUMERIC_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_UNIT_RE    = re.compile(
    r"\b(?:g/mol|mol%|wt%|vol%|°C|deg\s*C|K|MPa|Pa\b|kPa|GPa|nm|µm|um|mm|cm|"
    r"min|hours?|h|s\b|rpm|kJ/mol|kcal/mol|Hz|kHz|MHz|Da|kDa)\b",
    re.IGNORECASE,
)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


# ---------------------------------------------------------------------------
# Metric 1: Coverage of extracted knowledge
# ---------------------------------------------------------------------------

def _coverage(questions: List[dict], rolling_state: Dict[str, list]) -> Dict[str, Any]:
    """
    What fraction of the extracted state items is referenced by >= 1 CQ?
    Categories: concepts, properties, relations. Findings excluded (they are
    paper-specific observations, not entities to be referenced by CQs).
    """
    concept_names = [_norm(c.get("name", "")) for c in rolling_state.get("concepts", []) if c.get("name")]
    concept_names = [c for c in concept_names if len(c) >= 3]

    property_keys = []
    for p in rolling_state.get("properties", []):
        material = _norm(p.get("material", ""))
        prop     = _norm(p.get("property_name", ""))
        if material and prop:
            property_keys.append((material, prop))

    relation_keys = []
    for r in rolling_state.get("relations", []):
        s = _norm(r.get("subject", "")); o = _norm(r.get("object", ""))
        if s and o: relation_keys.append((s, o))

    blobs = [_norm(f"{q.get('question','')} {q.get('expected_answer','')}") for q in questions]

    concepts_hit   = {c for c in concept_names if any(c in b for b in blobs)}
    properties_hit = {
        (m, p) for (m, p) in property_keys
        if any(m in b and (p in b or p.split()[0] in b) for b in blobs)
    }
    relations_hit  = {(s, o) for (s, o) in relation_keys if any(s in b and o in b for b in blobs)}

    def _ratio(hit, total):
        return round(len(hit) / total, 3) if total else 1.0

    score = round(
        0.5 * _ratio(concepts_hit,   len(set(concept_names))) +
        0.3 * _ratio(properties_hit, len(set(property_keys))) +
        0.2 * _ratio(relations_hit,  len(set(relation_keys))),
        3,
    )
    return {
        "score": score,
        "concepts":   {"total": len(set(concept_names)),   "referenced": len(concepts_hit),
                       "ratio": _ratio(concepts_hit, len(set(concept_names)))},
        "properties": {"total": len(set(property_keys)),   "referenced": len(properties_hit),
                       "ratio": _ratio(properties_hit, len(set(property_keys)))},
        "relations":  {"total": len(set(relation_keys)),   "referenced": len(relations_hit),
                       "ratio": _ratio(relations_hit, len(set(relation_keys)))},
        "uncovered_concept_examples": sorted(set(concept_names) - concepts_hit)[:10],
    }


def gap_concepts(questions: List[dict], rolling_state: Dict[str, list], top_k: int = 10) -> List[str]:
    """
    Public helper: return up to top_k concept names (by mentions) that no
    question currently references. Used by the coverage-gap pass in Stage 4.
    """
    blobs = [_norm(f"{q.get('question','')} {q.get('expected_answer','')}") for q in questions]
    concepts_sorted = sorted(
        rolling_state.get("concepts", []),
        key=lambda c: c.get("mentions", 1),
        reverse=True,
    )
    out: List[str] = []
    for c in concepts_sorted:
        name = (c.get("name") or "").strip()
        if not name or len(name) < 3:
            continue
        if not any(_norm(name) in b for b in blobs):
            out.append(name)
        if len(out) >= top_k:
            break
    return out


# ---------------------------------------------------------------------------
# Metric 2: Question Faithfulness (REPLACES v2's answer-groundedness)
# ---------------------------------------------------------------------------

def _question_faithfulness(questions: List[dict], retrieved_passages: Dict[str, List[str]], embedder) -> Dict[str, Any]:
    """
    Primary metric. The DELIVERABLE is the question; check that the question
    text aligns semantically with content actually in the paper.

    score = mean over questions of:
        max_cosine(question_text, passages)  bucketed
    """
    qtexts = [q.get("question", "").strip() for q in questions]
    if not qtexts or embedder is None:
        return {"score": None, "skipped": True}
    flat: List[str] = []
    for ps in (retrieved_passages or {}).values():
        for p in ps:
            if p not in flat: flat.append(p)
    if not flat:
        return {"score": None, "skipped": True, "reason": "no_passages"}

    try:
        q_vecs = embedder.embed_documents(qtexts)
        p_vecs = embedder.embed_documents(flat)
    except Exception as e:
        return {"score": None, "skipped": True, "reason": f"embedder unavailable: {e}"}
    with np.errstate(all="ignore"):
        sims = q_vecs @ p_vecs.T
    sims = np.nan_to_num(sims, nan=0.0, posinf=1.0, neginf=-1.0)
    max_sim_per_q = sims.max(axis=1)

    n_good      = int((max_sim_per_q >= _FAITHFUL_GOOD_THRESHOLD).sum())
    n_ok        = int(((max_sim_per_q >= _FAITHFUL_OK_THRESHOLD) & (max_sim_per_q < _FAITHFUL_GOOD_THRESHOLD)).sum())
    n_off       = int((max_sim_per_q < _FAITHFUL_OK_THRESHOLD).sum())

    n = len(qtexts)
    score = round((n_good * 1.0 + n_ok * 0.5) / n, 3)

    weakest_idx = int(max_sim_per_q.argmin())
    weakest = {
        "question_id": questions[weakest_idx].get("id", ""),
        "question":    questions[weakest_idx].get("question", "")[:200],
        "max_cosine":  float(max_sim_per_q[weakest_idx]),
    }
    return {
        "score":             score,
        "mean_cosine":       round(float(max_sim_per_q.mean()), 3),
        "min_cosine":        round(float(max_sim_per_q.min()), 3),
        "n_well_faithful":   n_good,
        "n_weak":            n_ok,
        "n_off_paper":       n_off,
        "thresholds":        {"ok": _FAITHFUL_OK_THRESHOLD, "good": _FAITHFUL_GOOD_THRESHOLD},
        "weakest_example":   weakest,
    }


def _answer_verifiability(questions: List[dict], retrieved_passages: Dict[str, List[str]], embedder) -> Dict[str, Any]:
    """
    Diagnostic only - NOT in the composite. Same shape as v2's groundedness
    but kept as a side signal so reviewers can still see if any answers
    appear hallucinated.
    """
    answers = [q.get("expected_answer", "").strip() for q in questions]
    if not answers or embedder is None:
        return {"skipped": True}
    flat: List[str] = []
    for ps in (retrieved_passages or {}).values():
        for p in ps:
            if p not in flat: flat.append(p)
    if not flat:
        return {"skipped": True, "reason": "no_passages"}
    try:
        a_vecs = embedder.embed_documents(answers)
        p_vecs = embedder.embed_documents(flat)
    except Exception as e:
        return {"skipped": True, "reason": f"embedder unavailable: {e}"}
    with np.errstate(all="ignore"):
        sims = a_vecs @ p_vecs.T
    sims = np.nan_to_num(sims, nan=0.0, posinf=1.0, neginf=-1.0)
    max_sim_per_a = sims.max(axis=1)
    return {
        "mean_cosine": round(float(max_sim_per_a.mean()), 3),
        "min_cosine":  round(float(max_sim_per_a.min()), 3),
        "thresholds":  {"ok": _FAITHFUL_OK_THRESHOLD, "good": _FAITHFUL_GOOD_THRESHOLD},
    }


# ---------------------------------------------------------------------------
# Metric 3: Specificity
# ---------------------------------------------------------------------------

def _specificity(questions: List[dict], rolling_state: Dict[str, list]) -> Dict[str, Any]:
    if not questions:
        return {"score": None, "skipped": True}
    concept_names = [_norm(c.get("name", "")) for c in rolling_state.get("concepts", []) if c.get("name")]
    concept_names = [c for c in concept_names if len(c) >= 3]
    lens          = [len(q.get("question", "")) for q in questions]
    n_with_num    = sum(1 for q in questions if _NUMERIC_RE.search(q.get("question", "") + " " + q.get("expected_answer", "")))
    n_with_unit   = sum(1 for q in questions if _UNIT_RE.search(q.get("question", "") + " " + q.get("expected_answer", "")))
    n_with_concept = 0
    for q in questions:
        blob = _norm(f"{q.get('question','')} {q.get('expected_answer','')}")
        if any(c in blob for c in concept_names):
            n_with_concept += 1
    n = len(questions)
    pct_num     = n_with_num     / n
    pct_unit    = n_with_unit    / n
    pct_concept = n_with_concept / n
    avg_len     = sum(lens) / n
    length_score = max(0.0, 1.0 - abs(avg_len - 110) / 110)
    score = round(
        0.5 * pct_concept +
        0.2 * pct_num +
        0.1 * pct_unit +
        0.2 * length_score,
        3,
    )
    return {
        "score":                     score,
        "avg_question_chars":        round(avg_len, 1),
        "pct_with_numeric_value":    round(pct_num, 3),
        "pct_with_unit":             round(pct_unit, 3),
        "pct_referencing_concept":   round(pct_concept, 3),
    }


# ---------------------------------------------------------------------------
# Metric 4: Diversity
# ---------------------------------------------------------------------------

def _diversity(questions: List[dict], embedder) -> Dict[str, Any]:
    texts = [q.get("question", "") for q in questions]
    if len(texts) < 2 or embedder is None:
        return {"score": None, "skipped": True}
    try:
        vecs = embedder.embed_documents(texts)
    except Exception as e:
        return {"score": None, "skipped": True, "reason": f"embedder unavailable: {e}"}
    with np.errstate(all="ignore"):
        sims = vecs @ vecs.T
    sims = np.nan_to_num(sims, nan=0.0, posinf=1.0, neginf=-1.0)
    n = len(texts)
    iu = np.triu_indices(n, k=1)
    pair_sims = sims[iu]
    mean_sim = float(pair_sims.mean())
    max_sim  = float(pair_sims.max())
    n_dups   = int((pair_sims >= _DUP_THRESHOLD).sum())
    score = round(max(0.0, 1.0 - mean_sim), 3)
    return {
        "score":             score,
        "mean_pairwise_cos": round(mean_sim, 3),
        "max_pairwise_cos":  round(max_sim, 3),
        "n_near_duplicates": n_dups,
        "dup_threshold":     _DUP_THRESHOLD,
    }


# ---------------------------------------------------------------------------
# Metric 5: Triplifiability  (NEW in v2.1)
# ---------------------------------------------------------------------------

_TRIPLE_PATTERNS = [
    # Object-of-subject patterns: "what is the X of Y", "which X did Y use"
    re.compile(r"\bwhat\s+(?:is|was|are|were)\s+the\s+\w[\w\s\-]{0,30}\s+of\s+\w", re.IGNORECASE),
    re.compile(r"\bwhich\s+\w[\w\s\-]{0,30}\s+(?:is|are|was|were|did|do|does)\s+", re.IGNORECASE),
    re.compile(r"\bby\s+which\s+\w[\w\s\-]{0,30}\b",                                 re.IGNORECASE),
    re.compile(r"\bhow\s+does\s+\w[\w\s\-]{0,40}\s+(?:affect|influence|relate)\b",   re.IGNORECASE),
    re.compile(r"\b(?:produced|measured|characterised|prepared|reported)\s+by\b",    re.IGNORECASE),
    # Direct triple request
    re.compile(r"\btriple\b|\bsubject[- ]predicate[- ]object\b",                     re.IGNORECASE),
]


def _triplifiability(questions: List[dict], rolling_state: Dict[str, list]) -> Dict[str, Any]:
    """
    Heuristic: does each question look like it asks about a subject-predicate-object
    relationship between extracted entities?

    A question scores 1.0 if BOTH (a) its phrasing matches one of the triple
    patterns AND (b) at least one extracted concept is named. 0.5 if only one
    is true. 0.0 if neither.
    """
    if not questions:
        return {"score": None, "skipped": True}
    concept_names = [_norm(c.get("name", "")) for c in rolling_state.get("concepts", []) if c.get("name")]
    concept_names = [c for c in concept_names if len(c) >= 3]

    per_q = []
    for q in questions:
        text = q.get("question", "")
        blob = _norm(f"{text} {q.get('expected_answer','')}")
        has_pattern = any(p.search(text) for p in _TRIPLE_PATTERNS)
        has_concept = any(c in blob for c in concept_names) if concept_names else False
        s = 1.0 if (has_pattern and has_concept) else (0.5 if (has_pattern or has_concept) else 0.0)
        per_q.append(s)
    n = len(per_q)
    pct_full    = sum(1 for s in per_q if s == 1.0) / n
    pct_partial = sum(1 for s in per_q if s == 0.5) / n
    score = round(sum(per_q) / n, 3)
    return {
        "score":             score,
        "pct_full_triples":  round(pct_full, 3),
        "pct_partial":       round(pct_partial, 3),
    }


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def evaluate(
    paper_id: str,
    generated_questions: List[dict],
    rolling_state: Dict[str, list] | None = None,
    retrieved_passages: Dict[str, List[str]] | None = None,
    embedder=None,
    reproducibility: dict | None = None,
) -> dict:
    """
    Reference-free 4-family evaluation. Returns a dict with overall_score
    plus per-family breakdown.

    Specificity is computed but stored in diagnostics only — it does not
    contribute to overall_score. See module docstring for rationale.

    `reproducibility` is an optional dict of metadata (model versions,
    embedder, prompt hashes) that gets attached to the result for
    publication-quality traceability.
    """
    rolling_state      = rolling_state or {}
    retrieved_passages = retrieved_passages or {}

    # ── Scored families (contribute to overall_score) ─────────────────────
    families: Dict[str, dict] = {}
    families["coverage"]              = _coverage(generated_questions, rolling_state)
    families["question_faithfulness"] = _question_faithfulness(generated_questions, retrieved_passages, embedder)
    families["diversity"]             = _diversity(generated_questions, embedder)
    families["triplifiability"]       = _triplifiability(generated_questions, rolling_state)

    # ── Diagnostics (recorded, NOT scored) ───────────────────────────────
    diagnostics = {
        "answer_verifiability": _answer_verifiability(generated_questions, retrieved_passages, embedder),
        "specificity":          _specificity(generated_questions, rolling_state),
    }

    # ── Composite over non-skipped families ──────────────────────────────
    parts: List[tuple] = []
    for name, w in _WEIGHTS.items():
        s = families[name].get("score")
        if s is not None:
            parts.append((s, w))
    overall = round(sum(s * w for s, w in parts) / sum(w for _, w in parts), 3) if parts else None

    return {
        "status":          "reference_free",
        "schema_version":  "v2.1",
        "paper_id":        paper_id,
        "n_questions":     len(generated_questions),
        "overall_score":   overall,
        "weights":         _WEIGHTS,
        "families":        families,   # 4 scored families
        "diagnostics":     diagnostics, # specificity + answer_verifiability
        "reproducibility": reproducibility or {},
    }
