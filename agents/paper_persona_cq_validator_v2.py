"""
Stage 5: validation for v2.1 persona CQs.

Two layers:
  1. Rule-based deterministic checks (cheap, used as fast diagnostics).
  2. Per-question LLM judge (batched per persona) - replaces v2's
     single sampled-question judge.

v2.1 changes:
  - LLM judge runs ALL questions, batched per persona (7 calls total
    instead of 1 sampled call). Each question gets a per-question score
    in [0, 1] and a pass/fail/borderline label so the refiner knows
    exactly which questions to fix.
  - Anti-sycophancy guardrails baked into the judge prompt: explicit
    neutral observation phrasing, "no issues = empty list" instruction,
    calibrated rubric, mandatory verbatim quote citations.
  - archetype_balance check removed (per scientific-paper-eval
    convention; was an engineering choice, not a CQ quality dimension).
  - Records per-question validation provenance (which passages were
    visible to the judge) so the refiner can use the same evidence.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

from utils.llm_client import CustomOllamaClient
from utils.embeddings import Embedder

_CONFIG_DIR    = Path(__file__).parent.parent / "config"
_PERSONAS_PATH = _CONFIG_DIR / "personas.json"
_CHECKS_PATH   = _CONFIG_DIR / "paper_persona_cq_v2_checks.json"

_INVALID_PLACEHOLDERS = {"", "unknown", "...", "n/a", "none"}

# Per-question pass thresholds. score >= PASS_THRESHOLD is left alone.
PASS_THRESHOLD       = 0.75
BORDERLINE_THRESHOLD = 0.40   # score in [0.40, 0.75) -> refine; < 0.40 -> refine (worse)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


class PaperPersonaCQValidatorV2:
    """Per-question LLM judge + rule checks. Returns per-question verdicts the refiner can act on."""

    def __init__(self, model_name: str, embedder: Embedder | None = None):
        self.model_name = model_name
        self.llm        = CustomOllamaClient(model=model_name)
        self.embedder   = embedder

        persona_cfg = json.loads(_PERSONAS_PATH.read_text(encoding="utf-8"))
        self.personas    = persona_cfg["personas"]
        self.n_questions = persona_cfg.get("n_questions_per_persona", 5)
        self.required_codes = {p["code"] for p in self.personas}
        self._persona_by_code = {p["code"]: p for p in self.personas}

        raw_checks = json.loads(_CHECKS_PATH.read_text(encoding="utf-8"))
        self.checks = {}
        for phase in ("phase_1", "phase_2", "phase_3"):
            self.checks.update(raw_checks.get(phase, {}))
        # archetype_balance removed in v2.1 - drop from checks even if config still has it
        self.checks.pop("archetype_balance", None)

        self.thresholds = raw_checks.get("thresholds", {})
        self.pass_threshold = float(self.thresholds.get("pass_score_per_question", PASS_THRESHOLD))

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------

    def validate(
        self,
        questions: List[dict],
        rolling_state: Dict[str, list],
        retrieved_passages: Dict[str, List[str]],
    ) -> dict:
        """
        Returns:
            {
              "quality_score":        composite of rule checks + judge scores,
              "pass_score_threshold": per-question pass threshold,
              "active_checks":        [...],
              "issues":               flat list of issues (for backwards compat),
              "issues_by_persona":    {persona_code: [issue, ...]},
              "per_question":         {question_id: {score, verdict, reasons, evidence_used}},
              "questions_to_refine":  [list of question ids with verdict != 'pass'],
            }
        """
        issues:            List[str] = []
        scores:            List[float] = []
        by_persona:        Dict[str, List[str]] = {}

        if self.checks.get("persona_coverage"):
            r = self._check_persona_coverage(questions)
            issues.extend(r["issues"]); scores.append(r["score"])
            if r["issues"]:
                by_persona.setdefault("_coverage", []).extend(r["issues"])

        if self.checks.get("question_count"):
            r = self._check_question_count(questions)
            issues.extend(r["issues"]); scores.append(r["score"])
            for code, msgs in r.get("by_persona", {}).items():
                by_persona.setdefault(code, []).extend(msgs)

        if self.checks.get("section_validity"):
            r = self._check_section_validity(questions)
            issues.extend(r["issues"]); scores.append(r["score"])
            for code, msgs in r.get("by_persona", {}).items():
                by_persona.setdefault(code, []).extend(msgs)

        if self.checks.get("concept_usage"):
            r = self._check_concept_usage(questions, rolling_state)
            issues.extend(r["issues"]); scores.append(r["score"])
            for code, msgs in r.get("by_persona", {}).items():
                by_persona.setdefault(code, []).extend(msgs)

        if self.checks.get("redundancy") and self.embedder is not None:
            r = self._check_redundancy(questions)
            issues.extend(r["issues"]); scores.append(r["score"])

        # ----- Per-question LLM judge (the new core) -----
        per_question, judge_issues, judge_avg = self._llm_judge_per_question(
            questions, retrieved_passages,
        )
        issues.extend(judge_issues)
        scores.append(judge_avg)
        for issue in judge_issues:
            code = self._extract_code(issue)
            if code:
                by_persona.setdefault(code, []).append(issue)

        questions_to_refine = [
            qid for qid, v in per_question.items() if v["verdict"] != "pass"
        ]

        quality_score = round(sum(scores) / len(scores), 3) if scores else 1.0
        return {
            "quality_score":        quality_score,
            "pass_score_threshold": self.pass_threshold,
            "active_checks":        [k for k, v in self.checks.items() if v],
            "issues":               issues,
            "issues_by_persona":    by_persona,
            "per_question":         per_question,
            "questions_to_refine":  questions_to_refine,
            "n_total":              len(questions),
            "n_pass":               sum(1 for v in per_question.values() if v["verdict"] == "pass"),
            "n_borderline":         sum(1 for v in per_question.values() if v["verdict"] == "borderline"),
            "n_fail":               sum(1 for v in per_question.values() if v["verdict"] == "fail"),
        }

    # ------------------------------------------------------------------
    # Rule-based checks (cheap)
    # ------------------------------------------------------------------

    def _check_persona_coverage(self, questions: List[dict]) -> dict:
        present = {q.get("persona_code", "") for q in questions}
        missing = self.required_codes - present
        if missing:
            return {
                "score":  round(len(present) / len(self.required_codes), 3),
                "issues": [f"Missing persona codes: {', '.join(sorted(missing))}"],
            }
        return {"score": 1.0, "issues": []}

    def _check_question_count(self, questions: List[dict]) -> dict:
        by_code: Dict[str, list] = {}
        for q in questions:
            by_code.setdefault(q.get("persona_code", "?"), []).append(q)
        issues: List[str] = []
        by_persona: Dict[str, list] = {}
        penalty = 0
        for code in self.required_codes:
            count = len(by_code.get(code, []))
            # The coverage-gap pass may add extras, so >= n_questions is fine
            if count < self.n_questions:
                msg = f"[{code}] expected at least {self.n_questions} questions, got {count}"
                issues.append(msg)
                by_persona.setdefault(code, []).append(msg)
                penalty += max(0, self.n_questions - count)
        total = self.n_questions * len(self.required_codes)
        score = round(max(0.0, 1.0 - penalty / total), 3)
        return {"score": score, "issues": issues, "by_persona": by_persona}

    def _check_section_validity(self, questions: List[dict]) -> dict:
        issues: List[str] = []
        by_persona: Dict[str, list] = {}
        for q in questions:
            sec  = (q.get("source_section") or "").strip().lower()
            code = q.get("persona_code", "?")
            if sec in _INVALID_PLACEHOLDERS:
                msg = f"[{code}] {q.get('id', '?')}: source_section is missing or a placeholder"
                issues.append(msg)
                by_persona.setdefault(code, []).append(msg)
        score = round(1.0 - len(issues) / max(len(questions), 1), 3)
        return {"score": score, "issues": issues, "by_persona": by_persona}

    def _check_concept_usage(self, questions: List[dict], rolling_state: Dict[str, list]) -> dict:
        concept_names = [
            _norm(c.get("name", "")) for c in rolling_state.get("concepts", [])
            if c.get("name")
        ]
        for p in rolling_state.get("properties", []):
            m = _norm(p.get("material", ""))
            if m: concept_names.append(m)
        concept_names = [c for c in concept_names if len(c) >= 3]
        if not concept_names:
            return {"score": 1.0, "issues": [], "by_persona": {}}
        issues: List[str] = []
        by_persona: Dict[str, list] = {}
        for q in questions:
            blob = _norm(f"{q.get('question', '')} {q.get('expected_answer', '')}")
            if not any(c in blob for c in concept_names):
                code = q.get("persona_code", "?")
                msg = (f"[{code}] {q.get('id', '?')}: question does not reference any extracted "
                       "concept (possible hallucination)")
                issues.append(msg)
                by_persona.setdefault(code, []).append(msg)
        score = round(1.0 - len(issues) / max(len(questions), 1), 3)
        return {"score": score, "issues": issues, "by_persona": by_persona}

    def _check_redundancy(self, questions: List[dict], threshold: float = 0.92) -> dict:
        texts = [q.get("question", "") for q in questions]
        if len(texts) < 2 or self.embedder is None:
            return {"score": 1.0, "issues": []}
        try:
            vecs = self.embedder.embed_documents(texts)
        except Exception as e:
            logging.warning("[Validator] redundancy check skipped - embedder unavailable: %s", e)
            return {"score": 1.0, "issues": [], "skipped": True, "reason": str(e)}
        import numpy as np
        with np.errstate(all="ignore"):
            sims = vecs @ vecs.T
        sims = np.nan_to_num(sims, nan=0.0, posinf=1.0, neginf=-1.0)
        n = len(texts)
        dup_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                if sims[i, j] >= threshold:
                    dup_pairs.append((i, j, float(sims[i, j])))
        issues = [
            f"Near-duplicate questions: {questions[i].get('id','?')} ~ {questions[j].get('id','?')} (cos={s:.2f})"
            for i, j, s in dup_pairs[:10]
        ]
        score = round(max(0.0, 1.0 - len(dup_pairs) / max(n, 1)), 3)
        return {"score": score, "issues": issues}

    # ------------------------------------------------------------------
    # Per-question LLM judge (the main change in v2.1)
    # ------------------------------------------------------------------

    def _llm_judge_per_question(
        self,
        questions: List[dict],
        retrieved_passages: Dict[str, List[str]],
    ) -> tuple[Dict[str, dict], List[str], float]:
        """
        Judge ALL questions, batched per persona (one LLM call per persona).
        For each persona batch we select the most relevant passages to that
        batch using cosine similarity (NOT a blind character-budget truncation),
        so the judge sees the evidence actually needed to verify those questions.

        Returns:
          - per_question: {question_id: {"score","verdict","reasons","evidence_used"}}
          - issues:       human-readable flat list
          - avg_score:    mean per-question score (for the composite)
        """
        # Flatten + dedupe the passage pool; skip empty/None entries
        flat: List[str] = []
        for ps in (retrieved_passages or {}).values():
            for p in ps:
                if p and p.strip() and p not in flat:
                    flat.append(p)
        n_total_passages = len(flat)

        # Group questions by persona
        by_code: Dict[str, List[dict]] = {}
        for q in questions:
            by_code.setdefault(q.get("persona_code", "?"), []).append(q)

        per_question: Dict[str, dict] = {}
        all_issues:   List[str]       = []
        all_scores:   List[float]     = []

        for code, persona_qs in by_code.items():
            persona = self._persona_by_code.get(code, {"name": code, "perspective": ""})

            # Per-batch evidence selection: embed the persona's questions and pull
            # the top-K passages from the full pool that are MOST relevant to them.
            # K is chosen so that K * ~3500-char chunks ~= 25k char budget,
            # comfortably within the qwen3/deepseek-r1 32k-token context window.
            selected_passages, evidence_text = self._select_evidence_for_batch(persona_qs, flat, top_k=8)
            n_passages_used = len(selected_passages)

            judged = self._judge_batch(persona, persona_qs, evidence_text)
            for q in persona_qs:
                qid = q.get("id", "")
                entry = judged.get(qid)
                if entry is None:
                    entry = {
                        "score":   0.5,
                        "verdict": "borderline",
                        "reasons": ["judge did not return a verdict for this question"],
                    }
                entry["evidence_used"] = {
                    "n_passages_selected": n_passages_used,
                    "n_passages_in_pool":  n_total_passages,
                    "evidence_chars":      len(evidence_text),
                }
                s = float(entry.get("score", 0.5))
                if s >= self.pass_threshold:
                    entry["verdict"] = "pass"
                elif s >= BORDERLINE_THRESHOLD:
                    entry["verdict"] = "borderline"
                else:
                    entry["verdict"] = "fail"
                per_question[qid] = entry
                all_scores.append(s)
                if entry["verdict"] != "pass":
                    reasons = entry.get("reasons") or ["no reason given"]
                    all_issues.append(f"[{code}] {qid}: {entry['verdict']} (score={s:.2f}) - {reasons[0]}")

        avg = sum(all_scores) / max(len(all_scores), 1)
        return per_question, all_issues, round(avg, 3)

    def _select_evidence_for_batch(
        self,
        persona_qs: List[dict],
        passage_pool: List[str],
        top_k: int = 8,
        max_chars: int = 25000,
    ) -> tuple[List[str], str]:
        """
        Return (selected_passages, formatted_text). Uses semantic similarity:
        for each question in the batch we score every passage, then take the
        union of each question's top-K (so every question is represented).

        Falls back to top-K-by-pool-order if the embedder is unavailable.
        """
        if not passage_pool:
            return [], ""

        # Compose query texts (question + expected_answer carries useful tokens)
        queries = [(q.get("question") or "") + " " + (q.get("expected_answer") or "") for q in persona_qs]
        try:
            if self.embedder is None:
                raise RuntimeError("no embedder")
            q_vecs = self.embedder.embed_documents(queries)
            p_vecs = self.embedder.embed_documents(passage_pool)
            import numpy as np
            with np.errstate(all="ignore"):
                sims = q_vecs @ p_vecs.T  # (n_questions, n_passages); vectors are L2-normalised
            sims = np.nan_to_num(sims, nan=0.0, posinf=1.0, neginf=-1.0)
            selected_idx: List[int] = []
            for row in sims:
                top = list(reversed(row.argsort()))[: max(1, top_k)]
                for idx in top:
                    if idx not in selected_idx:
                        selected_idx.append(int(idx))
        except Exception as e:
            logging.warning("[Validator] evidence selection fell back to first-N (embedder unavailable: %s)", e)
            selected_idx = list(range(min(top_k * len(persona_qs), len(passage_pool))))

        # Materialise + enforce char budget
        selected: List[str] = []
        parts: List[str] = []
        total = 0
        for idx in selected_idx:
            p = passage_pool[idx]
            block = f"--- passage {len(selected)+1} ---\n{p}"
            if total + len(block) + 2 > max_chars and selected:
                break
            selected.append(p)
            parts.append(block)
            total += len(block) + 2
        return selected, "\n\n".join(parts)

    def _judge_batch(self, persona: dict, persona_qs: List[dict], evidence_text: str) -> Dict[str, dict]:
        """One LLM call to judge all questions for one persona."""
        # Trim each question's payload to keep prompt size sane
        questions_view = [
            {
                "id":              q.get("id", ""),
                "archetype":       q.get("archetype", ""),
                "question":        q.get("question", ""),
                "expected_answer": q.get("expected_answer", ""),
                "source_section":  q.get("source_section", ""),
            }
            for q in persona_qs
        ]

        prompt = f"""You are an independent, skeptical reviewer. You evaluate competency questions (CQs) generated from a polymer-chemistry research paper. You are NOT the author and you are NOT allowed to assume facts that are not in the evidence below.

PERSONA WHO WROTE THESE QUESTIONS: {persona.get('name','')} ({persona.get('code', '')})
Their perspective: {persona.get('perspective','')}

EVIDENCE PASSAGES (the only source of truth available - verbatim excerpts from the paper):
{evidence_text}

QUESTIONS TO JUDGE:
{json.dumps(questions_view, ensure_ascii=False, indent=2)}

NEUTRAL OBSERVATION (read carefully):
Your task is to assess each question on TWO independent dimensions. Do not anchor on any expected pass rate. Some questions may be excellent, some may be flawed. Score each ON ITS OWN merits using the rubric below.

DIMENSIONS:
  1. ANSWERABILITY - can the question be answered using ONLY the EVIDENCE PASSAGES above?
     - 1.0 = the answer is explicitly stated in the passages
     - 0.5 = the answer can be derived from the passages with a short inference
     - 0.0 = the passages do not contain the answer (or require outside knowledge)
  2. SPECIFICITY - is the question concrete enough to be useful for ontology construction?
     - 1.0 = names a specific material/method/property/value
     - 0.5 = somewhat specific but uses generic terms ("the material", "the method")
     - 0.0 = vague catch-all question with no anchor

PER-QUESTION SCORE = mean of the two dimensions (0.0 to 1.0).

CALIBRATED RUBRIC FOR THE FINAL SCORE:
  0.0  = clear failure (unanswerable AND vague, or fabricated entity)
  0.3  = serious flaw (one dimension failed)
  0.5  = borderline (both dimensions partial)
  0.75 = solid pass (specific AND grounded)
  1.0  = exemplary (no possible improvement)

ANTI-SYCOPHANCY RULES (read each before scoring):
  - You are NOT required to find issues. If a question is genuinely good, score it accordingly.
  - You are NOT required to be lenient. If a question is unanswerable from the evidence, do not score it as a pass to avoid conflict.
  - Do NOT fabricate flaws to seem thorough.
  - Do NOT manufacture evidence that isn't in the passages.
  - For every flaw you cite, quote the verbatim text in the question that demonstrates the flaw. If you cannot quote, do not raise the flaw.
  - If a question references a material/method/value that does NOT appear anywhere in the EVIDENCE PASSAGES, that is a hard failure on ANSWERABILITY regardless of how plausible the question sounds.

OUTPUT FORMAT (strict JSON object, no markdown, no commentary):
{{
  "verdicts": [
    {{
      "id":      "<question id>",
      "score":   0.0,
      "reasons": ["short specific reason 1", "short specific reason 2"]
    }}
    ...one entry per question, in any order...
  ]
}}

If a question is fine, return an empty `reasons` array. Empty `reasons` is the expected output for passing questions.
"""
        try:
            response = self.llm.invoke(prompt, format="json")
            parsed = self._parse_judge(response)
            return {entry["id"]: entry for entry in parsed if "id" in entry}
        except Exception as e:
            logging.warning("[Validator] judge call for persona %s failed: %s", persona.get("code"), e)
            return {}

    @staticmethod
    def _parse_judge(response: str) -> List[dict]:
        if not response:
            return []
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```", 2)[-1]
            if cleaned.lstrip().lower().startswith("json"):
                cleaned = cleaned.split("\n", 1)[-1] if "\n" in cleaned else cleaned[4:]
            cleaned = cleaned.rsplit("```", 1)[0]
        start = cleaned.find("{")
        end   = cleaned.rfind("}") + 1
        if start == -1 or end <= start:
            return []
        try:
            obj = json.loads(cleaned[start:end])
            if isinstance(obj, dict):
                verdicts = obj.get("verdicts", [])
                if isinstance(verdicts, list):
                    return [
                        {
                            "id":      str(v.get("id", "")),
                            "score":   max(0.0, min(1.0, float(v.get("score", 0.5)))),
                            "reasons": [str(r) for r in (v.get("reasons") or []) if r],
                        }
                        for v in verdicts if isinstance(v, dict)
                    ]
        except Exception:
            return []
        return []

    @staticmethod
    def _extract_code(issue: str) -> str:
        m = re.match(r"\[([A-Z]+)\]", issue.strip())
        return m.group(1) if m else ""
