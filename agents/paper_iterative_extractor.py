"""
Stage 2: iterative knowledge extractor (v2.1).

For each chunk N:
    prompt = static_priors + compact(rolling_state) + chunk
    new_data = LLM.extract(prompt)
    rolling_state = merge(rolling_state, new_data)

v2.1 changes vs v2:
  - Schema reduced to 4 categories (concepts, relations, properties, findings).
    Use_cases dropped (already covered by concepts+relations). Axioms renamed
    to findings because papers report empirical observations, not general axioms.
  - Evidence is now evidence_sentence (full sentence) + evidence_context
    (1-2 sentences before/after).
  - Each item carries confidence in {high, medium, low}.
  - Prompt rewritten with anti-sycophancy guardrails:
      * explicit "omit rather than fabricate" rule
      * neutral observation phrasing instead of leading questions
      * explicit empty-list-is-fine rule
      * confidence is mandatory and must be calibrated
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from utils.llm_client import CustomOllamaClient, LLMNetworkError
from utils.rolling_state import new_state, merge, summary
from utils.state_compactor import compact

_CONFIG_DIR  = Path(__file__).parent.parent / "config"
_PRIORS_PATH = _CONFIG_DIR / "domain_priors.json"
_SCHEMA_PATH = _CONFIG_DIR / "extraction_schema.json"

_OUTPUT_SKELETON: Dict[str, List[dict]] = {
    "concepts":   [],
    "relations":  [],
    "properties": [],
    "findings":   [],
}


class PaperIterativeExtractor:
    """Runs the per-chunk extraction loop and returns the final rolling state."""

    def __init__(self, model_name: str, state_caps: Dict[str, int] | None = None):
        self.model_name = model_name
        self.llm        = CustomOllamaClient(model=model_name)
        self.priors     = json.loads(_PRIORS_PATH.read_text(encoding="utf-8"))
        self.schema     = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
        self.state_caps = state_caps or {}
        self._static_block = self._build_static_block()

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_static_block(self) -> str:
        categories = self.priors.get("concept_categories", {})
        cat_lines = []
        for cat, examples in categories.items():
            head = ", ".join(examples[:8])
            cat_lines.append(f"  - {cat}: {head}{' ...' if len(examples) > 8 else ''}")

        relations = ", ".join(self.priors.get("relation_types", []))

        schema_view = {
            k: {"fields": v.get("fields"), "example": v.get("example")}
            for k, v in self.schema.items() if not k.startswith("_")
        }

        return f"""DOMAIN: {self.priors.get('domain', 'Copolymer Science')}

CONCEPT CATEGORIES (label every concept with exactly one of these):
{chr(10).join(cat_lines)}

ALLOWED RELATION PREDICATES (prefer these; you may invent one only if essential):
  {relations}

OUTPUT SCHEMA (the JSON object you must produce):
{json.dumps(schema_view, ensure_ascii=False, indent=2)}
"""

    def _build_prompt(self, chunk: str, compact_state: str, paper_metadata: Dict[str, Any]) -> str:
        meta_view = {k: paper_metadata.get(k) for k in ("title", "authors", "journal", "year", "domain")
                     if paper_metadata.get(k)}
        return f"""You are a copolymer-science information extractor. Read ONE CHUNK from a research paper and emit a JSON object capturing every concept, relation, property, and finding the chunk introduces - but ONLY those grounded in the chunk text.

{self._static_block}

PAPER METADATA (for context only - do NOT use to invent facts):
{json.dumps(meta_view, ensure_ascii=False)}

ROLLING STATE SO FAR (already extracted from earlier chunks):
{compact_state}

CURRENT CHUNK:
\"\"\"
{chunk}
\"\"\"

OBJECTIVE OBSERVATIONS, NOT INFERENCES:
- Extract only what the CURRENT CHUNK explicitly states. Do not extrapolate from your background knowledge of polymer chemistry.
- If the chunk lacks enough surrounding context to confidently assert a fact, OMIT the item. An empty array is the correct answer when there is nothing to extract.
- Do NOT fabricate items to appear thorough. Do NOT repeat items already in the rolling state unless the chunk provides genuinely new evidence (new value, new evidence sentence, new method linkage).

EVIDENCE REQUIREMENTS (mandatory - if you cannot provide both, OMIT the item):
- evidence_sentence: the FULL sentence containing the mention, verbatim from the chunk. Not a fragment. Not paraphrased.
- evidence_context:  1-2 sentences before AND 1-2 sentences after evidence_sentence, verbatim (~200-400 chars total). If the sentence is at the start/end of the chunk and one side is unavailable, include only the available side.

CONFIDENCE CALIBRATION (mandatory):
- high   = the chunk explicitly states the fact with the entity/value clearly named in the same sentence
- medium = the entity and the fact are in adjacent sentences and the link is unambiguous from the context
- low    = the chunk implies the fact but the linkage is not explicit; consider omitting low-confidence items

FINDINGS vs PROPERTIES (do not duplicate):
- properties = a measured quantity with a numeric value and unit (e.g., Mn = 24,500 g/mol)
- findings   = a non-trivial empirical observation or cause-effect statement that is NOT just a single measurement (e.g., "increasing T from 110 to 140 deg C decreased cell size from 85 to 35 um")
- Do not create a findings entry for what is already captured as a property entry.

OUTPUT FORMAT (strict JSON object with exactly these 4 keys, no markdown, no commentary):
{{
  "concepts":   [ ... ],
  "relations":  [ ... ],
  "properties": [ ... ],
  "findings":   [ ... ]
}}

Each array may be empty. Empty arrays are valid output when the chunk has nothing of that kind.
"""

    # ------------------------------------------------------------------
    # Driver
    # ------------------------------------------------------------------

    def extract_chunk(
        self,
        chunk: str,
        rolling_state: Dict[str, List[dict]],
        paper_metadata: Dict[str, Any],
    ) -> Dict[str, List[dict]]:
        compact_state = compact(rolling_state, char_budget=4000)
        prompt = self._build_prompt(chunk, compact_state, paper_metadata)
        response = self.llm.invoke(prompt, format="json")
        return self._parse(response)

    # Maximum consecutive network failures tolerated before aborting the whole
    # extraction.  Single-chunk transient errors are fine; a run of 3+ means the
    # server is genuinely down and there is no point waiting through retries for
    # the remaining chunks.
    _MAX_CONSECUTIVE_NETWORK_FAILURES = 3

    def run(self, chunks: List[str], paper_metadata: Dict[str, Any]) -> Dict[str, List[dict]]:
        state = new_state()
        total = len(chunks)
        consecutive_net_failures = 0

        for i, chunk in enumerate(chunks, start=1):
            print(f"    [Extract] chunk {i}/{total} (chars={len(chunk)})")
            try:
                new_data = self.extract_chunk(chunk, state, paper_metadata)
                consecutive_net_failures = 0          # server responded — reset counter
            except LLMNetworkError as e:
                consecutive_net_failures += 1
                logging.warning(
                    "[Extract] Network failure on chunk %d/%d "
                    "(consecutive: %d/%d): %s",
                    i, total,
                    consecutive_net_failures,
                    self._MAX_CONSECUTIVE_NETWORK_FAILURES,
                    e,
                )
                if consecutive_net_failures >= self._MAX_CONSECUTIVE_NETWORK_FAILURES:
                    raise LLMNetworkError(
                        f"Aborting extraction after {consecutive_net_failures} consecutive "
                        f"network failures (chunk {i - consecutive_net_failures + 1}–{i} "
                        f"of {total}). Server appears to be down."
                    ) from e
                continue                               # skip this chunk, keep partial state
            except Exception as e:
                # Non-network error (e.g. JSON parse failure in a single chunk):
                # log and skip without touching the consecutive-failure counter.
                logging.warning("[Extract] chunk %d failed: %s", i, e)
                continue

            merge(state, new_data, caps=self.state_caps)
            counts = summary(state)
            print(f"      state -> concepts={counts['concepts']} relations={counts['relations']} "
                  f"properties={counts['properties']} findings={counts['findings']}")
        return state

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse(response: str) -> Dict[str, List[dict]]:
        if not response:
            return {k: [] for k in _OUTPUT_SKELETON}
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```", 2)[-1]
            if cleaned.lstrip().lower().startswith("json"):
                cleaned = cleaned.split("\n", 1)[-1] if "\n" in cleaned else cleaned[4:]
            cleaned = cleaned.rsplit("```", 1)[0]
        start = cleaned.find("{")
        end   = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            logging.warning("[Extract] No JSON object in response - returning empty extraction.")
            return {k: [] for k in _OUTPUT_SKELETON}
        try:
            parsed = json.loads(cleaned[start:end + 1])
        except Exception as e:
            logging.warning("[Extract] JSON parse failed (%s) - returning empty extraction.", e)
            return {k: [] for k in _OUTPUT_SKELETON}
        out = {k: [] for k in _OUTPUT_SKELETON}
        for k in out.keys():
            v = parsed.get(k, [])
            if isinstance(v, list):
                out[k] = v
        return out
