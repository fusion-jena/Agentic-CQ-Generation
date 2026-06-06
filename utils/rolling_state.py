"""
Rolling-state accumulator for the iterative-extraction loop (v2.1 schema).

Maintains a deduplicated, capped knowledge structure across chunks for four
categories:
    concepts, relations, properties, findings

(v2 had a fifth category 'use_cases' and named 'findings' as 'axioms'. Both
were renamed/dropped in v2.1 - papers do not state OWL-style axioms, and
applications are already concepts/relations.)

Evidence is captured as evidence_sentence + evidence_context (1-2 sentences
before and after) so downstream CQ generation has real anchors. A confidence
field per item (high|medium|low) lets the validator weight low-confidence
items less. Dedup keys are case-insensitive normalisations of the identifying
field(s) per category. When a duplicate is seen we increment `mentions` and
merge auxiliary fields (aliases, evidence list). Hard caps keep prompt size
bounded.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

EMPTY_STATE: Dict[str, List[dict]] = {
    "concepts":   [],
    "relations":  [],
    "properties": [],
    "findings":   [],
}

_CONFIDENCE_RANK = {"high": 3, "medium": 2, "low": 1}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _key_concept(item: dict) -> str:
    return _norm(item.get("name", ""))


def _key_relation(item: dict) -> str:
    return "|".join([_norm(item.get("subject", "")),
                     _norm(item.get("predicate", "")),
                     _norm(item.get("object", ""))])


def _key_property(item: dict) -> str:
    return "|".join([_norm(item.get("material", "")),
                     _norm(item.get("property_name", "")),
                     _norm(str(item.get("numeric_value", "")))])


def _key_finding(item: dict) -> str:
    return _norm(item.get("statement", ""))


_KEY_FNS = {
    "concepts":   _key_concept,
    "relations":  _key_relation,
    "properties": _key_property,
    "findings":   _key_finding,
}


def new_state() -> Dict[str, List[dict]]:
    return {k: [] for k in EMPTY_STATE.keys()}


def _merge_aliases(existing: dict, incoming: dict) -> None:
    a = list(existing.get("aliases") or [])
    b = list(incoming.get("aliases") or [])
    seen = {_norm(x) for x in a}
    for x in b:
        if x and _norm(x) not in seen:
            a.append(x)
            seen.add(_norm(x))
    if a:
        existing["aliases"] = a


def _merge_evidence(existing: dict, incoming: dict) -> None:
    """
    Append incoming evidence_sentence + evidence_context to a small history.
    Keeps up to 3 distinct sentence/context pairs so downstream retrieval has
    multiple anchors for the same entity.
    """
    history = existing.get("evidence_history")
    if history is None:
        history = []
        if existing.get("evidence_sentence"):
            history.append({
                "evidence_sentence": existing.get("evidence_sentence", ""),
                "evidence_context":  existing.get("evidence_context", ""),
            })

    incoming_sentence = (incoming.get("evidence_sentence") or "").strip()
    if incoming_sentence:
        already = any(h.get("evidence_sentence", "").strip() == incoming_sentence for h in history)
        if not already:
            history.append({
                "evidence_sentence": incoming_sentence,
                "evidence_context":  incoming.get("evidence_context", ""),
            })

    existing["evidence_history"] = history[:3]


def _promote_confidence(existing: dict, incoming: dict) -> None:
    """Keep the HIGHEST confidence seen across mentions."""
    e = _CONFIDENCE_RANK.get((existing.get("confidence") or "").lower(), 1)
    i = _CONFIDENCE_RANK.get((incoming.get("confidence") or "").lower(), 1)
    if i > e:
        existing["confidence"] = incoming["confidence"]


def merge(
    state: Dict[str, List[dict]],
    new_data: Dict[str, List[dict]],
    caps: Dict[str, int] | None = None,
) -> Dict[str, List[dict]]:
    """
    Merge a per-chunk extraction result into the rolling state.

    - Dedup keyed by category-specific identity fields
    - `mentions` counter bumped on collision
    - Evidence sentences accumulated into evidence_history (max 3)
    - Confidence promoted to the highest seen
    - Per-category caps (default 80/60/60/40) keep state size bounded; when a
      category overflows we sort by (mentions desc, confidence rank desc) and
      drop the tail.

    Returns the same `state` dict (mutated in place) for convenience.
    """
    caps = caps or {}
    for category in EMPTY_STATE.keys():
        incoming = new_data.get(category) or []
        if not isinstance(incoming, list):
            continue
        key_fn = _KEY_FNS[category]
        existing_by_key: Dict[str, dict] = {key_fn(it): it for it in state[category]}

        for item in incoming:
            if not isinstance(item, dict):
                continue
            k = key_fn(item)
            if not k or k.replace("|", "").strip() == "":
                continue
            if k in existing_by_key:
                existing = existing_by_key[k]
                existing["mentions"] = existing.get("mentions", 1) + 1
                _merge_aliases(existing, item)
                _merge_evidence(existing, item)
                _promote_confidence(existing, item)
            else:
                item = dict(item)
                item.setdefault("mentions", 1)
                item.setdefault("confidence", "medium")
                # Seed evidence_history from the initial evidence_sentence
                if item.get("evidence_sentence"):
                    item["evidence_history"] = [{
                        "evidence_sentence": item.get("evidence_sentence", ""),
                        "evidence_context":  item.get("evidence_context", ""),
                    }]
                state[category].append(item)
                existing_by_key[k] = item

        cap = caps.get(category)
        if cap and len(state[category]) > cap:
            state[category].sort(
                key=lambda x: (
                    x.get("mentions", 1),
                    _CONFIDENCE_RANK.get((x.get("confidence") or "").lower(), 1),
                ),
                reverse=True,
            )
            state[category] = state[category][:cap]

    return state


def summary(state: Dict[str, List[dict]]) -> Dict[str, int]:
    return {k: len(v) for k, v in state.items()}


def top_concept_names(state: Dict[str, List[dict]], k: int = 20) -> List[str]:
    """
    Return the top-k concept names by (mentions, confidence). Used as
    retrieval queries against the FAISS index in Stage 3.
    """
    concepts = sorted(
        state.get("concepts", []),
        key=lambda x: (
            x.get("mentions", 1),
            _CONFIDENCE_RANK.get((x.get("confidence") or "").lower(), 1),
        ),
        reverse=True,
    )
    names: List[str] = []
    seen: set[str] = set()
    for c in concepts:
        name = (c.get("name") or "").strip()
        if name and _norm(name) not in seen:
            names.append(name)
            seen.add(_norm(name))
        if len(names) >= k:
            break
    return names
