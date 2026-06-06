"""
Token-budget-aware compaction of the rolling state for prompt injection (v2.1).

The full rolling state can hold tens of items per category - too large to
paste verbatim into every chunk's extraction prompt. `compact()` produces
a terse JSON-like text summary capped at a target character budget.

Strategy:
  - Sort each category by (mentions desc, confidence rank desc).
  - Drop the bulky evidence_history, evidence_context, and aliases fields
    from the compact view (we keep evidence_sentence as the most useful
    short anchor).
  - Truncate to char budget with explicit "...truncated" markers.
"""
from __future__ import annotations

import json
from typing import Dict, List

_DEFAULT_PER_CATEGORY_KEEP = {
    "concepts":   30,
    "relations":  25,
    "properties": 20,
    "findings":   15,
}

_CONF_RANK = {"high": 3, "medium": 2, "low": 1}

# Fields to drop from the compact view (kept in the full state on disk).
_DROP_FIELDS = ("evidence_history", "evidence_context", "aliases")


def _strip(item: dict) -> dict:
    return {k: v for k, v in item.items() if k not in _DROP_FIELDS}


def compact(
    state: Dict[str, List[dict]],
    char_budget: int = 4000,
    per_category_keep: Dict[str, int] | None = None,
) -> str:
    """
    Return a compact JSON text summary of `state` capped at ~char_budget characters.
    """
    keep = per_category_keep or _DEFAULT_PER_CATEGORY_KEEP
    view: Dict[str, list] = {}
    for category, items in state.items():
        n = keep.get(category, 20)
        sorted_items = sorted(
            items,
            key=lambda x: (
                x.get("mentions", 1),
                _CONF_RANK.get((x.get("confidence") or "").lower(), 1),
            ),
            reverse=True,
        )[:n]
        view[category] = [_strip(it) for it in sorted_items]

    text = json.dumps(view, ensure_ascii=False, indent=1)
    if len(text) <= char_budget:
        return text

    # Progressively shrink per-category counts until it fits
    for shrink in [0.75, 0.5, 0.3, 0.15]:
        shrunk: Dict[str, list] = {}
        for category, items in view.items():
            n = max(1, int(len(items) * shrink))
            shrunk[category] = items[:n]
        text = json.dumps(shrunk, ensure_ascii=False, indent=1)
        if len(text) <= char_budget:
            return text + "\n[... rolling state truncated for prompt budget ...]"

    return text[:char_budget] + "\n[... rolling state hard-truncated ...]"
