"""
Step 1 of the CQ narrowing pipeline: intra-paper deduplication.

Reads a v2 CQ output file, flattens all persona questions into one pool
(irrespective of which persona generated them), asks the LLM to identify
semantic duplicates, and writes a structured dedup output with full
traceability.

Design principles
─────────────────
• Every original v2 CQ ID is preserved — either as a representative's
  source_id or in a discarded_questions entry with absorbed_into set.
• paper_id is embedded in every representative question so Step 2 can
  safely flatten across all 14 dedup files without losing provenance.
• expected_answer is preserved in merged_from entries so Step 3 can
  validate against ALL merged expected answers, not just the kept one.
• deepseek-r1 thinking trace (think=True) provides the merge_reasoning
  — we get it for free without asking for it separately in the JSON.
• If the LLM response cannot be parsed, a safe fallback treats every
  question as a singleton (no data loss, pipeline continues).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from utils.llm_client import CustomOllamaClient, fmt_duration

# ---------------------------------------------------------------------------
# Bloom level ordering — used to explain selection priority in prompt
# ---------------------------------------------------------------------------
_BLOOM_ORDER: Dict[str, int] = {
    "recall":        1,
    "comprehension": 2,
    "application":   3,
    "analysis":      4,
    "synthesis":     5,
    "evaluation":    6,
}


class CQDedupAgent:
    """
    Intra-paper CQ deduplication agent.

    One LLM call per paper. Uses deepseek-r1 with think=True so the
    chain-of-thought reasoning is captured as merge_reasoning in the output.
    """

    STEP_VERSION:   str = "step1_v1.0"
    PROMPT_VERSION: str = "dedup_prompt_v1.0"

    def __init__(self, model_name: str, think: bool = True):
        self.model_name = model_name
        self.think      = think
        self.llm        = CustomOllamaClient(model=model_name)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def deduplicate(self, source_path: Path) -> dict:
        """
        Run intra-paper deduplication on a single v2 CQ output file.

        Args:
            source_path: absolute path to a *_cq_v2.json file.

        Returns:
            Fully structured dedup output dict (Step 1 schema).
        """
        source_data  = json.loads(source_path.read_text(encoding="utf-8"))
        paper_id     = source_data["paper_id"]
        paper_meta   = source_data.get("paper_metadata", {})

        all_questions = self._flatten_questions(source_data)
        n_input       = len(all_questions)

        print(f"  [Dedup] {paper_id}: {n_input} input CQs "
              f"({source_data.get('n_personas', '?')} personas)")

        if not all_questions:
            print(f"  [Dedup] WARNING: no questions found in {source_path.name}")
            return self._empty_output(paper_id, paper_meta, source_path, source_data)

        # ── Vague-answer pre-screen (syntactic, zero LLM cost) ────────
        vague_warnings = self._check_vague_answers(all_questions)
        if vague_warnings:
            print(f"  [Dedup] {len(vague_warnings)} vague-answer CQ(s) flagged "
                  f"(still included in dedup — see dedup_stats.vague_answer_warnings)")

        # ── LLM call ─────────────────────────────────────────────────
        prompt = self._build_prompt(paper_id, paper_meta, all_questions)

        t0 = time.time()
        if self.think:
            response_text, thinking_text = self.llm.invoke_with_thinking(
                prompt, format="json"
            )
        else:
            response_text = self.llm.invoke(prompt, format="json")
            thinking_text = ""
        elapsed = time.time() - t0

        print(f"  [Dedup] LLM call done in {fmt_duration(elapsed)}")

        # ── Parse + assemble output ───────────────────────────────────
        questions_by_id = {q["id"]: q for q in all_questions}
        dedup_result    = self._parse(
            response_text, thinking_text,
            paper_id, all_questions, questions_by_id,
        )

        llm_stats = self.llm.drain_stats()
        return self._build_output(
            paper_id        = paper_id,
            paper_meta      = paper_meta,
            source_path     = source_path,
            source_data     = source_data,
            dedup_result    = dedup_result,
            all_questions   = all_questions,
            llm_stats       = llm_stats,
            total_elapsed   = elapsed,
            vague_warnings  = vague_warnings,
        )

    # ------------------------------------------------------------------
    # Flatten helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten_questions(source_data: dict) -> List[dict]:
        """Merge persona_questions dict into a single flat list."""
        flat: List[dict] = []
        for questions in source_data.get("persona_questions", {}).values():
            flat.extend(questions)
        return flat

    @staticmethod
    def _check_vague_answers(questions: List[dict]) -> List[dict]:
        """
        Syntactic pre-screen for hedge-answer CQs (zero LLM cost).
        Returns a list of warning dicts — questions are NOT removed from the
        pipeline, only flagged so the reviewer can see them in dedup_stats.
        """
        _HEDGE_PHRASES = {
            "depends on", "various factors", "multiple factors",
            "several factors", "context-dependent", "it depends",
            "many factors", "complex relationship", "not straightforward",
        }
        warnings: List[dict] = []
        for q in questions:
            answer = (q.get("expected_answer") or "").strip().lower()
            flags: List[str] = []
            if len(answer.split()) < 8:
                flags.append("answer_too_short")
            for phrase in _HEDGE_PHRASES:
                if phrase in answer:
                    flags.append(f"hedge_phrase: '{phrase}'")
                    break
            if flags:
                warnings.append({
                    "id":              q.get("id", "?"),
                    "persona_code":    q.get("persona_code", "?"),
                    "expected_answer": q.get("expected_answer", ""),
                    "flags":           flags,
                })
        return warnings

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        paper_id:     str,
        paper_meta:   dict,
        all_questions: List[dict],
    ) -> str:
        meta_view = {
            k: paper_meta.get(k)
            for k in ("title", "journal", "year", "polymer_systems", "keywords")
            if paper_meta.get(k)
        }
        n_personas  = len({q.get("persona_code", "?") for q in all_questions})
        bloom_guide = " < ".join(
            k for k, _ in sorted(_BLOOM_ORDER.items(), key=lambda x: x[1])
        )

        questions_block = self._format_questions_block(all_questions)
        output_template = self._output_template()

        return f"""You are an expert ontology engineer working on a copolymer knowledge graph.

You are given {len(all_questions)} competency questions (CQs) generated from a SINGLE
paper by {n_personas} different personas (SYNTH, CHAR, PROC, COMP, ONTO, STU, PHD).

PAPER CONTEXT:
{json.dumps(meta_view, ensure_ascii=False, indent=2)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TASK — INTRA-PAPER DEDUPLICATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Find groups of semantically equivalent questions — questions asking the same
underlying thing, just phrased differently or viewed from a different persona.

CRITICAL RULES:
1. NEVER merge questions with DIFFERENT archetypes.
   Each archetype serves a distinct ontology purpose even when two questions
   discuss the same topic.
   Example: SYNTH "What synthesis method was used?" (PROCESS_FOR_MATERIAL) and
   ONTO "Which triple represents producedBy?" (ONTOLOGY_TRIPLE) are NOT duplicates.

2. For each duplicate group, select ONE representative to keep.
   Selection priority (apply in order):
     a. Highest Bloom level  →  {bloom_guide}
     b. Tie on Bloom: richer expected_answer (more specific, more quantitative)
     c. ONTO persona present with ONTOLOGY_TRIPLE archetype → prefer ONTO

3. ONTO persona questions are usually already abstract. Be conservative — only
   merge them if a question from another persona is word-for-word equivalent.

4. A singleton = a question with NO semantic duplicate in this paper.
   Do NOT force singletons into groups.

5. Every input ID must appear exactly once: either as keep_id, in discard_ids,
   or in singletons.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPETENCY QUESTIONS  ({len(all_questions)} total)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{questions_block}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SELF-CHECK — complete this before emitting output
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For EVERY CQ you mark as a singleton, identify the single other CQ in
the input that comes closest to being its duplicate, and confirm you
have a clear reason it does NOT qualify for a merge. If you cannot
give a clear reason, reconsider whether it should be merged instead.

For every merge group with 3 or more members, identify the member that
fits the group least well and confirm it genuinely belongs.

If your output contains MORE singletons than merge groups, re-read the
input looking for questions that ask the same underlying thing from
different personas or with different entity names. Finding zero merges
on a set of {len(all_questions)} questions is highly suspicious.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return ONLY a JSON object — no markdown fences, no commentary.
{output_template}
"""

    @staticmethod
    def _format_questions_block(questions: List[dict]) -> str:
        lines: List[str] = []
        for q in questions:
            lines.append(
                f"ID          : {q['id']}\n"
                f"Persona     : {q.get('persona_code','?')}  |  "
                f"Archetype: {q.get('archetype','?')}  |  "
                f"Bloom: {q.get('bloom_level','?')}\n"
                f"Question    : {q.get('question','')}\n"
                f"Exp. Answer : {q.get('expected_answer','')}\n"
            )
        return "\n".join(lines)

    @staticmethod
    def _output_template() -> str:
        return """{
  "duplicate_groups": [
    {
      "keep_id"         : "<ID of question to keep as representative>",
      "keep_reason"     : "<why this one over the others — Bloom level, richer answer, etc.>",
      "discard_ids"     : ["<ID1_to_discard>", "<ID2_to_discard>"],
      "merge_reasoning" : "<one sentence: why these questions are semantic duplicates>"
    }
  ],
  "singletons": ["<ID_with_no_duplicate>", "..."]
}"""

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse(
        self,
        response_text:  str,
        thinking_text:  str,
        paper_id:       str,
        all_questions:  List[dict],
        questions_by_id: Dict[str, dict],
    ) -> dict:
        """Parse the LLM JSON response into a structured dedup result."""

        parsed = self._extract_json(response_text)
        if parsed is None:
            print(f"  [Dedup] WARNING: JSON parse failed for {paper_id} — using fallback")
            return self._fallback_result(paper_id, all_questions)

        duplicate_groups: List[dict] = parsed.get("duplicate_groups", [])
        singleton_ids:    List[str]  = parsed.get("singletons", [])

        # ── Build lookup sets ─────────────────────────────────────────
        discarded_id_set: set[str] = set()
        for grp in duplicate_groups:
            for did in grp.get("discard_ids", []):
                discarded_id_set.add(did)

        kept_ids: set[str] = (
            {grp["keep_id"] for grp in duplicate_groups if "keep_id" in grp}
            | set(singleton_ids)
        )

        # Guard: any ID the LLM didn't mention → treat as singleton
        all_ids     = {q["id"] for q in all_questions}
        unaccounted = all_ids - kept_ids - discarded_id_set
        if unaccounted:
            print(f"  [Dedup] {len(unaccounted)} unaccounted ID(s) treated as singletons")
            singleton_ids = list(singleton_ids) + sorted(unaccounted)
            kept_ids     |= unaccounted

        # ── Reverse lookup: discard_id → keep_id ─────────────────────
        keeper_of: Dict[str, str] = {}
        group_of:  Dict[str, dict] = {}
        for grp in duplicate_groups:
            keep_id = grp.get("keep_id", "")
            group_of[keep_id] = grp
            for did in grp.get("discard_ids", []):
                keeper_of[did] = keep_id

        singleton_set = set(singleton_ids)

        # ── Build representative questions (in original order) ────────
        representative_questions: List[dict] = []
        discarded_questions:      List[dict] = []
        dedup_counter = 1

        # Maps original keep_id → assigned DEDUP id (needed to resolve absorbed_into)
        source_to_dedup: Dict[str, str] = {}

        for q in all_questions:
            qid = q["id"]

            if qid in discarded_id_set:
                # Will be filled in below once DEDUP IDs are known
                discarded_questions.append({
                    "id":            qid,
                    "paper_id":      paper_id,
                    "question":      q.get("question", ""),
                    "expected_answer": q.get("expected_answer", ""),
                    "persona_code":  q.get("persona_code", ""),
                    "archetype":     q.get("archetype", ""),
                    "bloom_level":   q.get("bloom_level", ""),
                    "source_section": q.get("source_section", "Body"),
                    "_keeper_source_id": keeper_of.get(qid, "UNKNOWN"),  # temp
                })
                continue

            if qid not in kept_ids:
                continue  # shouldn't happen after unaccounted guard, but be safe

            # ── Assign DEDUP id ───────────────────────────────────────
            dedup_id = f"{paper_id}_DEDUP_{dedup_counter:03d}"
            source_to_dedup[qid] = dedup_id
            dedup_counter += 1

            is_singleton = qid in singleton_set
            grp          = group_of.get(qid)

            # Build merged_from list — include expected_answer for Step 3
            merged_from: List[dict] = []
            if grp:
                for did in grp.get("discard_ids", []):
                    dq = questions_by_id.get(did, {})
                    merged_from.append({
                        "id":             did,
                        "paper_id":       paper_id,
                        "persona_code":   dq.get("persona_code", ""),
                        "question":       dq.get("question", ""),
                        "expected_answer": dq.get("expected_answer", ""),
                        "reason_discarded": grp.get("keep_reason", ""),
                    })

            # merge_reasoning: prefer explicit field from LLM JSON;
            # fall back to first 400 chars of thinking trace if available
            merge_reasoning: Optional[str] = None
            if grp:
                merge_reasoning = (
                    grp.get("merge_reasoning")
                    or (thinking_text[:400].strip() if thinking_text else None)
                )

            representative_questions.append({
                "id":            dedup_id,
                "paper_id":      paper_id,
                "source_id":     qid,
                "question":      q.get("question", ""),
                "expected_answer": q.get("expected_answer", ""),
                "persona_code":  q.get("persona_code", ""),
                "persona_full":  q.get("persona", ""),
                "archetype":     q.get("archetype", ""),
                "bloom_level":   q.get("bloom_level", ""),
                "source_section": q.get("source_section", "Body"),
                "is_singleton":  is_singleton,
                "why_kept":      (
                    grp.get("keep_reason", "Only question of this type; no merge")
                    if grp else "Only question of this type; no merge"
                ),
                "merged_from":      merged_from,
                "merge_reasoning":  merge_reasoning,
            })

        # ── Resolve absorbed_into now that DEDUP IDs are assigned ─────
        for dq in discarded_questions:
            keeper_source = dq.pop("_keeper_source_id", "UNKNOWN")
            dq["absorbed_into"] = source_to_dedup.get(keeper_source, f"(source: {keeper_source})")

        return {
            "representative_questions": representative_questions,
            "discarded_questions":      discarded_questions,
            "parse_fallback_used":      False,
        }

    def _fallback_result(self, paper_id: str, all_questions: List[dict]) -> dict:
        """
        Safe fallback when the LLM response cannot be parsed.
        Every question is treated as a singleton — no data loss,
        pipeline continues, output is flagged with parse_fallback_used=True.
        """
        representative_questions = []
        for i, q in enumerate(all_questions, start=1):
            representative_questions.append({
                "id":            f"{paper_id}_DEDUP_{i:03d}",
                "paper_id":      paper_id,
                "source_id":     q["id"],
                "question":      q.get("question", ""),
                "expected_answer": q.get("expected_answer", ""),
                "persona_code":  q.get("persona_code", ""),
                "persona_full":  q.get("persona", ""),
                "archetype":     q.get("archetype", ""),
                "bloom_level":   q.get("bloom_level", ""),
                "source_section": q.get("source_section", "Body"),
                "is_singleton":  True,
                "why_kept":      "PARSE_FALLBACK — all questions treated as singletons",
                "merged_from":   [],
                "merge_reasoning": None,
            })
        return {
            "representative_questions": representative_questions,
            "discarded_questions":      [],
            "parse_fallback_used":      True,
        }

    # ------------------------------------------------------------------
    # JSON extraction helper
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_json(text: str) -> Optional[dict]:
        """Robustly extract the first JSON object from model output."""
        if not text:
            return None
        cleaned = text.strip()

        # Strip markdown fences if present
        if cleaned.startswith("```"):
            parts = cleaned.split("```")
            for part in parts:
                part = part.strip()
                if part.lower().startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    cleaned = part
                    break

        try:
            start = cleaned.find("{")
            end   = cleaned.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(cleaned[start:end])
        except json.JSONDecodeError:
            pass

        return None

    # ------------------------------------------------------------------
    # Final output assembly
    # ------------------------------------------------------------------

    def _build_output(
        self,
        paper_id:       str,
        paper_meta:     dict,
        source_path:    Path,
        source_data:    dict,
        dedup_result:   dict,
        all_questions:  List[dict],
        llm_stats:      list,
        total_elapsed:  float,
        vague_warnings: Optional[List[dict]] = None,
    ) -> dict:

        rqs         = dedup_result["representative_questions"]
        dqs         = dedup_result["discarded_questions"]
        n_input     = len(all_questions)
        n_output    = len(rqs)
        n_discarded = len(dqs)
        n_merged    = sum(1 for r in rqs if not r["is_singleton"] and r["merged_from"])
        n_singletons = sum(1 for r in rqs if r["is_singleton"])

        # ── Accounting audit ──────────────────────────────────────────
        accounting_ok = (n_input == n_output + n_discarded)
        grouping_ok   = (n_merged + n_singletons == n_output)
        if not accounting_ok:
            print(
                f"  [Dedup] WARNING accounting drift — "
                f"input={n_input}, reps={n_output}, discarded={n_discarded}, "
                f"delta={n_input - n_output - n_discarded}"
            )
        if not grouping_ok:
            print(
                f"  [Dedup] WARNING grouping drift — "
                f"merged_groups={n_merged}, singletons={n_singletons}, "
                f"sum={n_merged + n_singletons}, expected={n_output}"
            )

        # Per-archetype breakdown
        arch_input:  Dict[str, int] = {}
        arch_output: Dict[str, int] = {}
        for q in all_questions:
            a = q.get("archetype", "UNKNOWN")
            arch_input[a] = arch_input.get(a, 0) + 1
        for r in rqs:
            a = r.get("archetype", "UNKNOWN")
            arch_output[a] = arch_output.get(a, 0) + 1
        archetype_breakdown = {
            a: {"n_input": arch_input.get(a, 0), "n_output": arch_output.get(a, 0)}
            for a in sorted(set(arch_input) | set(arch_output))
        }

        # Relative source path for portability
        try:
            rel_source = str(source_path.relative_to(source_path.parents[3]))
        except ValueError:
            rel_source = source_path.name

        return {
            # ── Identity ──────────────────────────────────────────────
            "paper_id":        paper_id,
            "pipeline_step":   "intra_paper_dedup",
            "pipeline_version": "v2",
            "step_version":    self.STEP_VERSION,

            # ── What produced this ────────────────────────────────────
            "dedup_config": {
                "llm_model":              self.model_name,
                "think":                  self.think,
                "format":                 "json",
                "prompt_version":         self.PROMPT_VERSION,
                "source_file":            rel_source,
                "source_pipeline_version": source_data.get("pipeline_version", "v2"),
            },

            # ── Paper context (carried forward, not re-extracted) ─────
            "paper_metadata": paper_meta,
            "paper_stats":    source_data.get("paper_stats", {}),

            # ── Dedup statistics ──────────────────────────────────────
            "dedup_stats": {
                "n_input_personas":   source_data.get("n_personas", 0),
                "n_input_cqs":        n_input,
                "n_representative_cqs": n_output,
                "n_merged_groups":    n_merged,
                "n_singletons":       n_singletons,
                "n_discarded":        n_discarded,
                "reduction_rate":     round((n_input - n_output) / n_input, 3) if n_input else 0.0,
                "archetype_breakdown": archetype_breakdown,
                "parse_fallback_used": dedup_result.get("parse_fallback_used", False),
                "accounting_ok":      accounting_ok,
                "grouping_ok":        grouping_ok,
                "vague_answer_warnings": vague_warnings or [],
            },

            # ── Execution stats ───────────────────────────────────────
            "execution_stats": {
                "total_wall_time": fmt_duration(total_elapsed),
                "llm_calls":       llm_stats,
            },

            # ── Step 1 outputs ────────────────────────────────────────
            "representative_questions": rqs,

            # ── Audit trail: every discarded CQ is preserved here ─────
            "discarded_questions": dqs,
        }

    @staticmethod
    def _empty_output(
        paper_id:    str,
        paper_meta:  dict,
        source_path: Path,
        source_data: dict,
    ) -> dict:
        return {
            "paper_id":        paper_id,
            "pipeline_step":   "intra_paper_dedup",
            "pipeline_version": "v2",
            "step_version":    CQDedupAgent.STEP_VERSION,
            "dedup_config":    {"source_file": str(source_path)},
            "paper_metadata":  paper_meta,
            "paper_stats":     source_data.get("paper_stats", {}),
            "dedup_stats": {
                "n_input_cqs": 0,
                "n_representative_cqs": 0,
                "parse_fallback_used": True,
            },
            "execution_stats":          {},
            "representative_questions": [],
            "discarded_questions":       [],
        }
