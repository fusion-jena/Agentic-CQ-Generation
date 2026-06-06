"""
Stage 4: persona-based CQ generation from structured knowledge (v2.1).

Replaces full-paper-text-stuffing (v1) with a much smaller, focused prompt:
    [persona role + style]
  + [STRUCTURED KNOWLEDGE]    = compact rolling state
  + [EVIDENCE PASSAGES]       = persona-biased retrieved chunks
  + [FEW-SHOT EXAMPLES]       = persona-specific few-shots
  + [ARCHETYPE TEMPLATES]     = CQ shape guidance

No raw paper text is fed in - so the prompt stays well under any model's
context window regardless of paper length.

v2.1 additions:
  - generate_all() optionally takes `gap_concepts`: a list of extracted-state
    concepts that the first pass did not reference. When provided, each
    persona generates an extra round of questions explicitly targeting
    those concepts (coverage-gap pass). This directly attacks the low
    coverage scores observed in v2 (0.17-0.35) by forcing the generator to
    revisit untouched entities.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from utils.llm_client import CustomOllamaClient
from utils.state_compactor import compact

_CONFIG_DIR      = Path(__file__).parent.parent / "config"
_PERSONAS_PATH   = _CONFIG_DIR / "personas.json"
_ARCHETYPES_PATH = _CONFIG_DIR / "cq_archetypes.json"
_FEW_SHOTS_PATH  = _CONFIG_DIR / "cq_few_shots.json"

# ---------------------------------------------------------------------------
# Valid archetype IDs — built once at import time for _parse normalisation.
# Maps lowercase / snake_case variants → canonical uppercase ID so the model
# can return "entity lookup" or "Entity_Lookup" and still be accepted.
# ---------------------------------------------------------------------------

def _load_valid_archetype_map() -> dict[str, str]:
    """Return {normalised_key: canonical_id} for fuzzy matching in _parse."""
    try:
        cfg = json.loads(_ARCHETYPES_PATH.read_text(encoding="utf-8"))
        mapping: dict[str, str] = {}
        for a in cfg.get("archetypes", []):
            cid = a["id"]                              # e.g. "ENTITY_LOOKUP"
            mapping[cid.lower()] = cid                 # "entity_lookup"
            mapping[cid.lower().replace("_", " ")] = cid  # "entity lookup"
            mapping[cid.lower().replace("_", "")] = cid   # "entitylookup"
        return mapping
    except Exception:
        return {}

_VALID_ARCHETYPE_MAP: dict[str, str] = _load_valid_archetype_map()


class PaperPersonaCQGeneratorV2:
    """Generates N CQs per persona using state + retrieved passages + few-shots."""

    def __init__(self, model_name: str):
        self.model_name  = model_name
        self.llm         = CustomOllamaClient(model=model_name)
        persona_cfg      = json.loads(_PERSONAS_PATH.read_text(encoding="utf-8"))
        self.personas    = persona_cfg["personas"]
        self.n_questions = persona_cfg.get("n_questions_per_persona", 5)
        self.archetypes  = json.loads(_ARCHETYPES_PATH.read_text(encoding="utf-8"))
        self.few_shots   = json.loads(_FEW_SHOTS_PATH.read_text(encoding="utf-8"))["examples"]

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------

    def generate_all(
        self,
        paper_id: str,
        paper_metadata: dict,
        rolling_state: Dict[str, list],
        retriever,
        gap_concepts: List[str] | None = None,
    ) -> tuple[Dict[str, list], Dict[str, List[str]]]:
        """
        Generate CQs for every persona.

        Normal pass (gap_concepts=None):
            Generates `n_questions_per_persona` questions per persona using
            persona-biased retrieved passages. Returns (by_persona, passages).

        Gap pass (gap_concepts=<list>):
            Generates ONLY gap-targeted questions (1-2 per persona) — does NOT
            re-run the full normal generation. Questions are immediately re-IDed
            with a `_GAP###` suffix so they can be merged without ID collisions.
            Returns (gap_questions_by_persona, gap_passages).

        Returns (questions_by_persona, passages_by_persona).
        passages_by_persona is merged into state.retrieved_passages by the workflow
        so validator + evaluator see the same evidence the generator used.
        """
        compact_state = compact(rolling_state, char_budget=5500)
        out: Dict[str, list]               = {}
        all_passages: Dict[str, List[str]] = {}

        if gap_concepts:
            # ── GAP PASS: targeted generation only ───────────────────────
            # Does NOT re-run the full normal pass — only generates
            # n_override (1-2) gap questions per persona. Avoids wasting
            # N*len(personas) LLM calls on duplicates that will be discarded.
            n_gap = max(1, min(2, len(gap_concepts)))
            print(f"    [GAP] Generating {n_gap} gap question(s)/persona for "
                  f"{len(gap_concepts)} untouched concept(s): "
                  f"{gap_concepts[:5]}{'...' if len(gap_concepts) > 5 else ''}")
            for persona in self.personas:
                gap_passages: List[str] = []
                for concept in gap_concepts[:8]:
                    hits = retriever.retrieve_for_persona(rolling_state, concept, n_passages=2)
                    gap_passages.extend(hits)
                # Dedup while preserving order
                seen_p: set[str] = set()
                gap_passages = [p for p in gap_passages if not (p in seen_p or seen_p.add(p))]
                all_passages[f"_persona_{persona['code']}_GAP"] = gap_passages
                extra = self._generate_for_persona(
                    paper_id, paper_metadata, persona, compact_state, gap_passages,
                    gap_concepts=gap_concepts[:8],
                    n_override=n_gap,
                )
                # Re-ID immediately with _GAP suffix (index from 1, not from
                # existing question count — caller handles merge)
                for i, q in enumerate(extra):
                    q["id"] = f"{paper_id}_{persona['code']}_GAP{i+1:03d}"
                out[persona["code"]] = extra
        else:
            # ── NORMAL PASS: full generation ──────────────────────────────
            for persona in self.personas:
                print(f"    [{persona['code']}] {persona['name']} - generating {self.n_questions} questions...")
                persona_hint = f"{persona['perspective']} {persona['question_style']}"
                passages = retriever.retrieve_for_persona(rolling_state, persona_hint, n_passages=6)
                all_passages[f"_persona_{persona['code']}"] = passages
                questions = self._generate_for_persona(
                    paper_id, paper_metadata, persona, compact_state, passages,
                )
                out[persona["code"]] = questions

        return out, all_passages

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _generate_for_persona(
        self,
        paper_id: str,
        paper_metadata: dict,
        persona: dict,
        compact_state: str,
        passages: List[str],
        gap_concepts: List[str] | None = None,
        n_override: int | None = None,
    ) -> List[dict]:
        n = n_override or self.n_questions
        prompt = self._build_prompt(paper_id, paper_metadata, persona, compact_state, passages,
                                    gap_concepts=gap_concepts, n=n)
        response = self.llm.invoke(prompt, format="json")
        return self._parse(response, paper_id, persona, n=n)

    def _build_prompt(
        self,
        paper_id: str,
        paper_metadata: dict,
        persona: dict,
        compact_state: str,
        passages: List[str],
        gap_concepts: List[str] | None = None,
        n: int = 5,
    ) -> str:
        meta_view = {k: paper_metadata.get(k) for k in ("title", "authors", "journal", "year", "domain")
                     if paper_metadata.get(k)}

        evidence_block = "\n\n".join(f"--- passage {i+1} ---\n{p}" for i, p in enumerate(passages)) \
            if passages else "(no passages retrieved)"

        archetype_codes = self.archetypes.get("persona_archetype_affinity", {}).get(persona["code"], [])
        archetype_codes = [c for c in archetype_codes if not c.startswith("_")]
        archetypes_view = [a for a in self.archetypes["archetypes"] if a["id"] in archetype_codes] \
            or self.archetypes["archetypes"][:4]
        archetype_block = "\n".join(f"  - {a['id']}: {a['pattern']}  // {a['purpose']}" for a in archetypes_view)

        few_shot_examples = self.few_shots.get(persona["code"], [])
        few_shot_block = "\n".join(
            f"  Example {i+1} ({ex.get('archetype','?')}):\n"
            f"    Q: {ex['question']}\n"
            f"    A: {ex['expected_answer']}"
            for i, ex in enumerate(few_shot_examples)
        ) if few_shot_examples else "(no examples available for this persona)"

        gap_block = ""
        if gap_concepts:
            gap_block = (
                "\nCOVERAGE-GAP TARGETS - these specific concepts were extracted from the paper "
                "but no earlier question referenced them. Aim your questions at these explicitly:\n"
                + "\n".join(f"  - {c}" for c in gap_concepts)
                + "\n"
            )

        template = self._build_template(paper_id, persona, n=n)

        return f"""You are a {persona['name']} working in the polymer-science domain.

ROLE & PERSPECTIVE:
{persona['perspective']}

YOUR QUESTION STYLE:
{persona['question_style']}

YOU ARE NOT READING THE FULL PAPER. You are given:
  (1) a STRUCTURED KNOWLEDGE summary already extracted from the paper, AND
  (2) verbatim EVIDENCE PASSAGES retrieved from the paper that are relevant to your persona.

Generate up to {n} competency questions strictly from YOUR perspective as a {persona['name']}.
Fewer is acceptable — do NOT pad with low-quality questions to reach the quota.

PAPER METADATA:
{json.dumps(meta_view, ensure_ascii=False)}

STRUCTURED KNOWLEDGE (extracted from the paper):
{compact_state}

EVIDENCE PASSAGES (verbatim from the paper, retrieved for your persona):
{evidence_block}
{gap_block}
CQ ARCHETYPE PATTERNS (aim to cover at least 2 different archetypes across your {n} questions — do not produce {n} variants of the same shape):
{archetype_block}

⚠ Use the EXACT archetype ID string as shown above (e.g. PROCESS_FOR_MATERIAL, ENTITY_LOOKUP).
  Do NOT use descriptions, lowercase, or paraphrases — the ID must match one of those names verbatim.

FEW-SHOT EXAMPLES (use these for STYLE only - do NOT copy their content; your questions must be about THIS paper):
{few_shot_block}

ANTI-SYCOPHANCY GUARDRAILS:
- Use only the STRUCTURED KNOWLEDGE and EVIDENCE PASSAGES above as your source of truth. Do not invent materials, methods, values, or section names.
- If you cannot generate a high-quality question for a particular archetype because the inputs lack the necessary anchors, generate a different archetype rather than fabricate content.
- Do NOT copy the few-shot examples verbatim. The examples show form, not content.
- Do NOT emit paper-summary CQs: "What did the authors conclude about X?" cannot test an ontology — replace with a parametric query about the entities and relationships involved.
- Do NOT emit hedge-answer CQs: if the expected_answer would be vague text like "it depends" or "several factors", the question is invalid — rewrite it with specific anchors or drop it.
- Do NOT emit glossary CQs: "What is X?" is a definition, not a competency question. Emit at most ONE per full output, and only for the single most central domain term.
- Do NOT cite specific figures, tables, or equations: "According to Table 2..." — the figure does not exist in an ontology. Ask for the underlying datum directly instead.
- A useful CQ must be re-askable against a different paper in the same domain. If the question only makes sense with this paper as context, it is NOT a valid CQ.

RULES:
- Every question MUST be answerable using ONLY the inputs above.
- Each question covers a DIFFERENT aspect or archetype.
- "source_section": the section heading the evidence appears to come from ("Abstract", "Experimental Section", "Results and Discussion"). If you cannot tell, use "Body".
- "ontology_hints": list 1–3 objects identifying the ontology fragments this CQ implies. Each object must have "role" (one of: class, property, individual) and "term" (the domain term). These are the entities and relationships that would need to exist in the ontology to answer the CQ.
- Stay strictly in role — questions must reflect the perspective of a {persona['name']}.

SELF-CHECK before emitting:
- Does every expected_answer name specific values, materials, methods, or named entities — not "varies" or "it depends"?
- Does every question survive re-asking against a different paper in the same polymer-science domain?
- Does every ontology_hints array contain at least one entry?
If any question fails these checks, drop it rather than emit it.

OUTPUT FORMAT — return a JSON OBJECT with a single key "questions" whose value is the array below.
Rules:
  - Replace EVERY "..." placeholder with your actual content — never return "..." literally.
  - Replace EVERY "<...>" placeholder with a real value — never return angle-bracket tokens.
  - No markdown fences, no commentary, no keys other than "questions".
{{
  "questions": {template}
}}
"""

    def _build_template(self, paper_id: str, persona: dict, n: int = 5) -> str:
        entries = []
        for i in range(1, n + 1):
            entries.append(
                f"""  {{
    "id": "{paper_id}_{persona['code']}_Q{i:03d}",
    "persona": "{persona['name']}",
    "persona_code": "{persona['code']}",
    "bloom_level": "{persona['bloom_level']}",
    "archetype": "<exact archetype ID e.g. ENTITY_LOOKUP>",
    "question": "<your question text here>",
    "expected_answer": "<your expected answer here>",
    "source_section": "<heading e.g. Abstract / Experimental Section / Results and Discussion / Body>",
    "ontology_hints": [
      {{"role": "<class|property|individual>", "term": "<domain term>"}}
    ]
  }}"""
            )
        return "[\n" + ",\n".join(entries) + "\n]"

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse(self, response: str, paper_id: str, persona: dict, n: int = 5) -> List[dict]:
        if not response:
            return self._parse_fallback(paper_id, persona, n=n)
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```", 2)[-1]
            if cleaned.lstrip().lower().startswith("json"):
                cleaned = cleaned.split("\n", 1)[-1] if "\n" in cleaned else cleaned[4:]
            cleaned = cleaned.rsplit("```", 1)[0]

        parsed_list: List[dict] | None = None

        # ── Try object-wrapped array {"questions": [...]} ─────────────────
        # Look for the "questions" key FIRST before falling back to any list value.
        try:
            start = cleaned.find("{")
            end   = cleaned.rfind("}") + 1
            if start != -1 and end > start:
                obj = json.loads(cleaned[start:end])
                if isinstance(obj, dict):
                    # 1. Prefer the explicit "questions" key
                    qs = obj.get("questions") or obj.get("cqs") or obj.get("competency_questions")
                    if isinstance(qs, list) and qs and isinstance(qs[0], dict):
                        parsed_list = qs
                    # 2. Fallback: first list-of-dicts value (catches other wrappers)
                    if parsed_list is None:
                        for v in obj.values():
                            if isinstance(v, list) and v and isinstance(v[0], dict):
                                parsed_list = v
                                break
        except Exception:
            pass

        # ── Fallback: bare JSON array ─────────────────────────────────────
        if parsed_list is None:
            try:
                start = cleaned.find("[")
                end   = cleaned.rfind("]") + 1
                if start != -1 and end > start:
                    arr = json.loads(cleaned[start:end])
                    if isinstance(arr, list):
                        parsed_list = arr
            except Exception:
                pass

        if parsed_list is None:
            return self._parse_fallback(paper_id, persona, n=n)

        # ── Post-parse normalisation ──────────────────────────────────────
        return [self._normalise_question(q) for q in parsed_list if isinstance(q, dict)]

    @staticmethod
    def _normalise_question(q: dict) -> dict:
        """
        Normalise a single question dict after parsing:
          - archetype: fuzzy-map returned string to a valid canonical ID.
          - source_section: strip any angle-bracket placeholder tokens.
          - question / expected_answer: strip "..." sentinels.
          - ontology_hints: ensure list; strip placeholder entries.
        """
        # Archetype normalisation
        arch_raw = (q.get("archetype") or "").strip()
        # Try exact match first, then fuzzy (lowercase + strip underscores/spaces)
        arch_key = arch_raw.lower().strip()
        canonical = (
            _VALID_ARCHETYPE_MAP.get(arch_raw)           # exact
            or _VALID_ARCHETYPE_MAP.get(arch_key)        # lowercase
            or _VALID_ARCHETYPE_MAP.get(arch_key.replace("_", " "))   # underscores → spaces
            or _VALID_ARCHETYPE_MAP.get(arch_key.replace(" ", "_"))   # spaces → underscores
            or _VALID_ARCHETYPE_MAP.get(arch_key.replace("_", ""))    # no separators
            or _VALID_ARCHETYPE_MAP.get(arch_key.replace(" ", ""))
        )
        if canonical:
            q["archetype"] = canonical
        # If still not canonical, leave the raw value in — validator will catch it.

        # Strip placeholder sentinels from text fields so they trigger the right validator check
        for field in ("question", "expected_answer"):
            val = (q.get(field) or "").strip()
            if val in ("...", "<your question text here>",
                       "<your expected answer here>", ""):
                q[field] = ""

        sec = (q.get("source_section") or "").strip()
        if sec.startswith("<") and sec.endswith(">"):
            q["source_section"] = "Body"   # safe fallback; validator accepts "Body"
        elif sec in ("...", ""):
            q["source_section"] = "Body"

        # ontology_hints: ensure it's a list and strip placeholder entries
        hints = q.get("ontology_hints")
        if not isinstance(hints, list):
            q["ontology_hints"] = []
        else:
            _PLACEHOLDER_TERMS = {"<domain term>", "...", "", "<term>"}
            _VALID_ROLES = {"class", "property", "individual", "axiom"}
            cleaned = [
                h for h in hints
                if isinstance(h, dict)
                and h.get("term", "").strip() not in _PLACEHOLDER_TERMS
                and h.get("role", "").strip().lower() in _VALID_ROLES
            ]
            q["ontology_hints"] = cleaned

        return q

    @staticmethod
    def _parse_fallback(paper_id: str, persona: dict, n: int = 1) -> List[dict]:
        """
        Return `n` placeholder questions so question_count validation does not
        cascade on top of the real issue (model parse failure).
        All placeholders have archetype=PARSE_ERROR so the validator flags them
        specifically as parse failures, not as wrong-count issues.
        """
        return [
            {
                "id":              f"{paper_id}_{persona['code']}_Q{i:03d}",
                "persona":         persona["name"],
                "persona_code":    persona["code"],
                "bloom_level":     persona["bloom_level"],
                "archetype":       "PARSE_ERROR",
                "question":        f"CQ parse error for persona [{persona['code']}] — model response could not be parsed.",
                "expected_answer": "",
                "source_section":  "Body",
                "ontology_hints":  [],
            }
            for i in range(n)
        ]
