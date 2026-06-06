"""
Stage 5b: persona CQ refiner for the v2.1 pipeline.

Public API
----------
refine_batch(persona_code, failing_questions, passages_flat, per_question)
    - `persona_code`      : the persona being refined (e.g. "SYNTH")
    - `failing_questions` : only the questions that failed validation
                            (already filtered by the workflow — do NOT pass all questions)
    - `passages_flat`     : deduplicated flat list of ALL retrieved passages for this paper
                            (workflow flattens state.retrieved_passages before calling us)
    - `per_question`      : {qid: {"score", "verdict", "reasons", "evidence_used"}}
                            from the validator — used to give the model its specific diagnosis

Returns a list of question dicts (same length as `failing_questions`), each
extended with two new fields:

    refiner_action : "revise" | "keep" | "unrepairable"
        revise       — model rewrote the question; new text is in the dict
        keep         — model judged the question acceptable despite the validator flag
        unrepairable — the evidence does not contain enough information to fix the question
                       (question is kept verbatim but tagged for human review)

    revise_reason  : str
        Always populated for "revise". Optional for "keep"/"unrepairable".
        Explains in one sentence WHY the action was taken.

Design notes
------------
- Prompt explicitly lists each failing question with its validator score + reasons,
  so the model knows the exact diagnosis before deciding.
- Evidence is semantically selected per-question (not a flat char-budget truncation):
  we embed each question, rank passages by cosine similarity, and feed the top K
  to the model — same approach as PaperPersonaCQValidatorV2._select_evidence_for_batch.
- The model is asked to return the FULL question dict (all fields) with refiner_action
  and revise_reason. The _parse method validates the action field and falls back
  gracefully on parse failure.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List

from utils.llm_client import CustomOllamaClient

_CONFIG_DIR    = Path(__file__).parent.parent / "config"
_PERSONAS_PATH = _CONFIG_DIR / "personas.json"

_VALID_ACTIONS = {"revise", "keep", "unrepairable"}


class PaperPersonaCQRefinerV2:
    """
    Per-question refiner for the v2.1 pipeline.
    Called by the workflow's refine_step with only the failing questions,
    the flat passage pool, and the validator's per-question verdicts.
    """

    def __init__(self, model_name: str, embedder=None):
        self.model_name = model_name
        self.llm        = CustomOllamaClient(model=model_name)
        self.embedder   = embedder                            # used for evidence selection
        cfg             = json.loads(_PERSONAS_PATH.read_text(encoding="utf-8"))
        self._personas  = {p["code"]: p for p in cfg["personas"]}

    # ------------------------------------------------------------------
    # Public entry point (matches workflow call signature)
    # ------------------------------------------------------------------

    def refine_batch(
        self,
        persona_code:      str,
        failing_questions: List[dict],
        passages_flat:     List[str],
        per_question:      Dict[str, dict],
    ) -> List[dict]:
        """
        Re-prompt the LLM for one persona's failing questions.

        Returns the same questions, each annotated with:
          - refiner_action : "revise" | "keep" | "unrepairable"
          - revise_reason  : str
        If the LLM response cannot be parsed, every question gets
        refiner_action="keep" so the merge step leaves them untouched
        (safe fallback — better than dropping them).
        """
        if not failing_questions:
            return []

        persona = self._personas.get(persona_code, {
            "code": persona_code, "name": persona_code,
            "perspective": "", "bloom_level": "understand",
        })

        # Select the most relevant passages for THIS batch of questions
        evidence_text = self._select_evidence(failing_questions, passages_flat, top_k=6)

        prompt = self._build_prompt(persona, failing_questions, per_question, evidence_text)
        try:
            response = self.llm.invoke(prompt, format="json")
        except Exception as e:
            logging.warning("[Refiner] LLM call failed for %s: %s", persona_code, e)
            return self._safe_fallback(failing_questions)

        refined = self._parse(response, failing_questions, persona_code)
        n_revise = sum(1 for r in refined if r.get("refiner_action") == "revise")
        n_keep   = sum(1 for r in refined if r.get("refiner_action") == "keep")
        n_unr    = sum(1 for r in refined if r.get("refiner_action") == "unrepairable")
        print(f"      [{persona_code}] revised={n_revise}  kept={n_keep}  unrepairable={n_unr}")
        return refined

    # ------------------------------------------------------------------
    # Evidence selection — semantic top-K per question
    # ------------------------------------------------------------------

    def _select_evidence(
        self,
        questions:  List[dict],
        passages:   List[str],
        top_k:      int = 6,
        max_chars:  int = 20_000,
    ) -> str:
        """
        Return a formatted evidence block with the passages most relevant
        to the failing questions.  Falls back to first-N if embedder absent.
        """
        if not passages:
            return "(no passages available)"

        selected_idx: List[int]
        try:
            if self.embedder is None:
                raise RuntimeError("no embedder")
            queries = [
                (q.get("question") or "") + " " + (q.get("expected_answer") or "")
                for q in questions
            ]
            import numpy as np
            q_vecs = self.embedder.embed_documents(queries)
            p_vecs = self.embedder.embed_documents(passages)
            with np.errstate(all="ignore"):
                sims = q_vecs @ p_vecs.T          # (n_questions, n_passages)
            sims = np.nan_to_num(sims, nan=0.0, posinf=1.0, neginf=-1.0)
            seen: set[int] = set()
            selected_idx = []
            for row in sims:
                for idx in reversed(row.argsort().tolist()):
                    if idx not in seen:
                        seen.add(idx)
                        selected_idx.append(idx)
                    if len(selected_idx) >= top_k * len(questions):
                        break
                if len(selected_idx) >= top_k * len(questions):
                    break
        except Exception as e:
            logging.warning("[Refiner] evidence selection fell back to first-N (%s)", e)
            selected_idx = list(range(min(top_k * len(questions), len(passages))))

        parts: List[str] = []
        total = 0
        for idx in selected_idx:
            block = f"--- passage {len(parts)+1} ---\n{passages[idx]}"
            if total + len(block) + 2 > max_chars and parts:
                break
            parts.append(block)
            total += len(block) + 2
        return "\n\n".join(parts) or "(no passages available)"

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        persona:       dict,
        questions:     List[dict],
        per_question:  Dict[str, dict],
        evidence_text: str,
    ) -> str:
        """
        Prompt that:
        1. Shows the persona role so the model stays in perspective.
        2. Lists each failing question WITH its validator diagnosis
           (score + full reasons list) so the model knows exactly what to fix.
        3. Shows the evidence passages the question must be answerable from.
        4. Asks for one JSON entry per question with refiner_action + revise_reason.
        """
        # Build per-question diagnosis view (includes validator data)
        diagnosis_entries = []
        for q in questions:
            qid     = q.get("id", "?")
            pq      = per_question.get(qid, {})
            score   = pq.get("score", "?")
            verdict = pq.get("verdict", "unknown")
            reasons = pq.get("reasons") or ["(no specific reason given)"]
            diagnosis_entries.append({
                "id":                  qid,
                "archetype":           q.get("archetype", ""),
                "question":            q.get("question", ""),
                "expected_answer":     q.get("expected_answer", ""),
                "source_section":      q.get("source_section", ""),
                "_validator_score":    score,
                "_validator_verdict":  verdict,
                "_validator_reasons":  reasons,
            })

        # Build the output template so the model knows the exact schema
        output_template = []
        for q in questions:
            qid = q.get("id", "?")
            output_template.append({
                "id":              qid,
                "persona":         q.get("persona", persona.get("name", "")),
                "persona_code":    q.get("persona_code", persona.get("code", "")),
                "bloom_level":     q.get("bloom_level", persona.get("bloom_level", "")),
                "archetype":       q.get("archetype", ""),
                "question":        "...",
                "expected_answer": "...",
                "source_section":  "...",
                "refiner_action":  "<revise|keep|unrepairable>",
                "revise_reason":   "<one sentence explaining the action>",
            })

        return f"""You are a rigorous editor improving competency questions (CQs) for a polymer-science knowledge graph.

PERSONA WHO WROTE THESE QUESTIONS: {persona.get('name', persona.get('code', ''))} ({persona.get('code', '')})
Their perspective: {persona.get('perspective', '')}

YOU ARE NOT READING THE FULL PAPER. Your only sources of truth are:
  (1) EVIDENCE PASSAGES below — verbatim excerpts from the paper
  (2) The question itself and its expected_answer

=== EVIDENCE PASSAGES ===
{evidence_text}

=== FAILING QUESTIONS WITH VALIDATOR DIAGNOSIS ===
Each entry includes the validator score (0.0–1.0), verdict, and the specific reasons
the question failed. Use these reasons as your repair instructions.
{json.dumps(diagnosis_entries, indent=2, ensure_ascii=False)}

=== YOUR TASK ===
For EACH question above, choose EXACTLY ONE action:

  "revise"
    When: the question can be fixed using the EVIDENCE PASSAGES above.
    How:
      - Rewrite question and/or expected_answer so the answer is explicitly
        in one of the EVIDENCE PASSAGES.
      - Add a specific material name, method, property, or value if the issue
        was TOO VAGUE.
      - Set source_section to the heading the evidence appears under
        ("Abstract", "Experimental Section", "Results and Discussion", "Body").
      - Do NOT invent facts not present in the EVIDENCE PASSAGES.
      - Fill revise_reason with one sentence: what you changed and why.

  "keep"
    When: you believe the question is acceptable despite the validator flag
    (e.g. the validator lacked the right passage but the answer IS in the evidence
    above, or the flag was a false positive).
    How:
      - Copy question / expected_answer / source_section verbatim from the input.
      - Fill revise_reason with a brief justification.

  "unrepairable"
    When: the EVIDENCE PASSAGES genuinely do not contain enough information
    to produce a valid answerable question on this topic.
    Use sparingly — only when revision is truly impossible, not just difficult.
    How:
      - Copy question / expected_answer / source_section verbatim from the input.
      - Fill revise_reason explaining what information is missing.

=== FIXING RULES (apply when action == "revise") ===
- NOT ANSWERABLE  → rewrite so the answer is explicitly in an EVIDENCE PASSAGE
- TOO VAGUE       → add a specific material, method, property value, or condition
- WRONG SECTION   → set source_section to the heading from the relevant passage
- HALLUCINATION   → replace the invented concept with one actually in the passages
- Keep id, persona, persona_code, bloom_level unchanged
- Change archetype only if the archetype-mismatch reason explicitly requires it

=== OUTPUT FORMAT ===
Return a JSON OBJECT with a single key "questions" — one entry per input question,
in the same order. No markdown fences, no commentary:
{{
  "questions": {json.dumps(output_template, indent=2, ensure_ascii=False)}
}}
"""

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse(
        self,
        response:     str,
        originals:    List[dict],
        persona_code: str,
    ) -> List[dict]:
        """
        Parse the LLM response and merge with originals.

        Validation rules:
        - If response is unparseable → safe_fallback (all "keep")
        - If a question id is absent from the response → "keep" original
        - If refiner_action is not in {"revise","keep","unrepairable"} → "keep"
        - If action == "revise" but question text is unchanged or empty → downgrade to "keep"
        """
        if not response:
            logging.warning("[Refiner] empty response for %s", persona_code)
            return self._safe_fallback(originals)

        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```", 2)[-1]
            if cleaned.lstrip().lower().startswith("json"):
                cleaned = cleaned.split("\n", 1)[-1] if "\n" in cleaned else cleaned[4:]
            cleaned = cleaned.rsplit("```", 1)[0]

        parsed_list: List[dict] | None = None

        # Try {"questions": [...]} wrapper first
        try:
            start = cleaned.find("{")
            end   = cleaned.rfind("}") + 1
            if start != -1 and end > start:
                obj = json.loads(cleaned[start:end])
                if isinstance(obj, dict):
                    for v in obj.values():
                        if isinstance(v, list) and v and isinstance(v[0], dict):
                            parsed_list = v
                            break
        except Exception:
            pass

        # Fallback: bare JSON array
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
            logging.warning("[Refiner] JSON parse failed for %s — keeping originals", persona_code)
            return self._safe_fallback(originals)

        # Index parsed output by id
        by_id: Dict[str, dict] = {}
        for entry in parsed_list:
            if isinstance(entry, dict) and entry.get("id"):
                by_id[str(entry["id"])] = entry

        result: List[dict] = []
        for original in originals:
            qid       = str(original.get("id", ""))
            candidate = by_id.get(qid)

            if candidate is None:
                # Model omitted this question — safe keep
                merged = dict(original)
                merged["refiner_action"] = "keep"
                merged["revise_reason"]  = "question not returned by refiner — original kept"
                result.append(merged)
                continue

            action        = (candidate.get("refiner_action") or "").strip().lower()
            revise_reason = (candidate.get("revise_reason") or "").strip() or "no reason given"

            if action not in _VALID_ACTIONS:
                action        = "keep"
                revise_reason = f"invalid refiner_action '{action}' — defaulted to keep"

            if action == "revise":
                new_q = (candidate.get("question") or "").strip()
                old_q = (original.get("question") or "").strip()
                if not new_q or new_q == old_q:
                    action        = "keep"
                    revise_reason = "refiner returned unchanged or empty question — treated as keep"

            # Merge: start from the original, overlay revised content when action == "revise"
            merged = dict(original)
            if action == "revise":
                for field in ("question", "expected_answer", "source_section", "archetype"):
                    val = candidate.get(field)
                    if val and str(val).strip() not in ("", "..."):
                        merged[field] = val
            merged["refiner_action"] = action
            merged["revise_reason"]  = revise_reason
            result.append(merged)

        return result

    # ------------------------------------------------------------------
    # Safe fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_fallback(questions: List[dict]) -> List[dict]:
        """Return all questions unchanged with refiner_action='keep'."""
        out = []
        for q in questions:
            m = dict(q)
            m.setdefault("refiner_action", "keep")
            m.setdefault("revise_reason",  "refiner did not run (parse error or empty response)")
            out.append(m)
        return out
