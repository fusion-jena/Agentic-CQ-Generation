"""
Step 2 of the CQ narrowing pipeline: cross-paper generalization.

Reads all Step 1 dedup files, groups representative questions by archetype,
and for each archetype makes ONE LLM call that:

  1. Groups CQs asking the SAME conceptual thing across different papers
     (even when paper-specific entities differ — e.g. CO2 vs N2 are both
     blowing agents → same generalised CQ).

  2. Writes a domain-level generalised CQ for each group.

  3. Also generalises singletons (CQs unique to one paper) by removing
     paper-specific entity names — they still become domain-level CQs.

Design principles
─────────────────
• Slicing by archetype before the LLM call keeps each prompt focused
  and prevents cross-archetype false merges (CAUSAL ≠ ENTITY_LOOKUP
  even if both mention CO2).
• Every Step 1 DEDUP ID appears in the reverse_index so Step 3 can
  look up which GenCQ to validate for any original CQ.
• source_id (original v2 ID) is carried through so the full provenance
  chain stays unbroken: GenCQ → DEDUP ID → v2 CQ ID.
• Per-archetype output files are saved alongside the combined file for
  easy debugging and re-running of a single archetype.
• deepseek-r1 thinking trace (think=True) is captured and stored as a
  truncated abstraction_note alongside each group's abstraction_reasoning.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from utils.llm_client import CustomOllamaClient, fmt_duration

# ── Coverage breadth thresholds ───────────────────────────────────────────────
_BREADTH_NARROW = 2   # 1-2 papers  → possibly still paper-specific
_BREADTH_MEDIUM = 6   # 3-6 papers  → good domain generalisation
# 7+             → broad / core domain CQ


def _coverage_breadth(n_papers: int) -> str:
    if n_papers <= _BREADTH_NARROW:
        return "narrow"
    if n_papers <= _BREADTH_MEDIUM:
        return "medium"
    return "broad"


class CQGeneralizerAgent:
    """
    Cross-paper CQ generalisation agent.

    One LLM call per archetype (10 calls total for a full run).
    Uses deepseek-r1 with think=True so the chain-of-thought reasoning
    is captured as abstraction_reasoning in the output.
    """

    STEP_VERSION:   str = "step2_v1.0"
    PROMPT_VERSION: str = "generalize_prompt_v1.0"

    def __init__(self, model_name: str, think: bool = True):
        self.model_name = model_name
        self.think      = think
        self.llm        = CustomOllamaClient(model=model_name)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def generalize_all(self, dedup_dir: Path) -> dict:
        """
        Run cross-paper generalisation over all Step 1 dedup files.

        Args:
            dedup_dir: path to data/papers/v2/dedup/

        Returns:
            Full Step 2 output dict (combined across all archetypes).
        """
        # ── Load + flatten all representative questions ────────────────
        all_rqs, paper_ids = self._load_all_rqs(dedup_dir)
        print(f"  [Gen] Loaded {len(all_rqs)} representative CQs "
              f"from {len(paper_ids)} paper(s)")

        if not all_rqs:
            raise ValueError(f"No representative questions found in {dedup_dir}")

        # ── Group by archetype ─────────────────────────────────────────
        by_archetype: Dict[str, List[dict]] = {}
        for rq in all_rqs:
            arch = rq.get("archetype", "UNKNOWN")
            by_archetype.setdefault(arch, []).append(rq)

        print(f"  [Gen] Archetypes found: "
              f"{', '.join(f'{a}({len(v)})' for a, v in sorted(by_archetype.items()))}\n")

        # ── One LLM call per archetype ─────────────────────────────────
        all_gen_cqs:     List[dict] = []
        archetype_stats: Dict[str, dict] = {}
        gen_counter:     Dict[str, int]  = {}   # per-archetype counter
        llm_calls_all:   List[dict] = []

        t_total = time.time()

        for archetype in sorted(by_archetype.keys()):
            questions  = by_archetype[archetype]
            n_arch_in  = len(questions)
            print(f"  [Gen] Archetype {archetype}: {n_arch_in} CQs from "
                  f"{len({q['paper_id'] for q in questions})} paper(s)...")

            t_arch = time.time()
            gen_cqs = self._generalize_archetype(archetype, questions, gen_counter)
            arch_elapsed = time.time() - t_arch

            llm_calls_arch = self.llm.drain_stats()
            llm_calls_all.extend(llm_calls_arch)

            all_gen_cqs.extend(gen_cqs)
            archetype_stats[archetype] = {
                "n_input_cqs":    n_arch_in,
                "n_gen_cqs":      len(gen_cqs),
                "n_papers":       len({q["paper_id"] for q in questions}),
                "wall_time":      fmt_duration(arch_elapsed),
                "llm_calls":      llm_calls_arch,
            }
            print(f"           → {len(gen_cqs)} GenCQs in {fmt_duration(arch_elapsed)}")

        total_elapsed = time.time() - t_total

        return self._build_output(
            all_gen_cqs      = all_gen_cqs,
            archetype_stats  = archetype_stats,
            all_rqs          = all_rqs,
            paper_ids        = paper_ids,
            llm_calls_all    = llm_calls_all,
            total_elapsed    = total_elapsed,
        )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_all_rqs(dedup_dir: Path) -> tuple[List[dict], List[str]]:
        """
        Load every *_dedup.json and return (flat_rq_list, sorted_paper_ids).
        Each RQ already carries paper_id, source_id, archetype from Step 1.
        """
        flat:      List[dict] = []
        paper_ids: List[str]  = []

        for path in sorted(dedup_dir.glob("*_dedup.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            pid  = data.get("paper_id", path.stem.replace("_dedup", ""))
            paper_ids.append(pid)
            flat.extend(data.get("representative_questions", []))

        return flat, sorted(set(paper_ids))

    # ------------------------------------------------------------------
    # Per-archetype generalisation
    # ------------------------------------------------------------------

    def _generalize_archetype(
        self,
        archetype:   str,
        questions:   List[dict],
        gen_counter: Dict[str, int],
    ) -> List[dict]:
        """Run one LLM call for a single archetype and return GenCQ list."""
        prompt = self._build_prompt(archetype, questions)

        if self.think:
            response_text, thinking_text = self.llm.invoke_with_thinking(
                prompt, format="json"
            )
        else:
            response_text = self.llm.invoke(prompt, format="json")
            thinking_text = ""

        return self._parse(
            response_text, thinking_text,
            archetype, questions, gen_counter,
        )

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(self, archetype: str, questions: List[dict]) -> str:
        n_papers = len({q["paper_id"] for q in questions})
        questions_block = self._format_questions_block(questions)
        output_template = self._output_template()
        valid_ids_line  = ", ".join(f'"{q["id"]}"' for q in questions)

        return f"""You are an expert polymer-science ontology engineer building a copolymer knowledge graph.

You are given {len(questions)} competency questions (CQs) — all belonging to archetype
"{archetype}" — drawn from {n_papers} different research papers on copolymers.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VALID INPUT IDs  ← YOU MAY ONLY USE THESE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{valid_ids_line}

HARD CONSTRAINT: every value in every "member_ids" array MUST be one of the
{len(questions)} IDs listed above — verbatim, character-for-character.
Inventing, abbreviating, or paraphrasing an ID is an error.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TASK — CROSS-PAPER GROUPING + GENERALISATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step A — GROUP
  Find groups of CQs that ask about the SAME conceptual thing, even when
  paper-specific entities differ.

  KEY INSIGHT: recognise functional equivalence, not surface similarity.
  Example:
    "What effect does CO2 concentration have on cell size?"   (paper A)
    "How does N2 pressure affect foam density?"               (paper B)
  → Both ask: how does a BLOWING AGENT PARAMETER affect FOAM MORPHOLOGY.
  → They belong to the same group.

  More examples of entity → domain-role substitution:
    poly(styrene-co-MMA), PLA-co-PHBV, PMMA-co-BMA  →  "the copolymer"
    CO2, N2, scCO2                                    →  "the blowing agent"
    RAFT, ATRP, FRP, NMP                              →  "the controlled / radical polymerization method"
    DSC, TGA, DMA                                     →  "the thermal characterisation technique"
    GPC/SEC, DLS, MALLS                               →  "the molecular weight characterisation technique"
    SEM, TEM, AFM                                     →  "the morphological characterisation technique"
    Tg, Tm, Tc                                        →  "the thermal transition property"
    Mn, Mw, Đ                                         →  "the molecular weight parameter"

Step B — GENERALISE
  For each group (including singletons), write ONE domain-level CQ that:
    • Replaces paper-specific entity names with their domain role/category.
    • Is answerable from ANY copolymer paper on the same topic.
    • Preserves the original question's intent and archetype pattern.
    • Is a complete, well-formed question (not a template with placeholders).

  For singletons: still generalise — remove the specific polymer/gas/method
  name even though no other paper asks the same question.

RULES:
  • Do NOT merge questions that ask about genuinely different phenomena
    (e.g. "effect of temperature on Tg" ≠ "effect of pressure on cell size").
  • Do NOT merge if the generalised question would be too vague to be useful.
  • Every input ID must appear in exactly one group or as a singleton.
  • member_ids must contain ONLY IDs from the VALID INPUT IDs list above.
  • Return ONLY the JSON object below — no markdown fences, no commentary.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ANTI-SYCOPHANCY GUARDRAILS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FAILURE MODE A — blanket narrow.
If your output contains ZERO groups and ONLY singletons, you have
failed unless every input is genuinely unique. Before declaring all
singletons, list at least 3 candidate pattern templates you considered
and explain why each was rejected. If you cannot generate 3 candidates
to reject, you have not searched hard enough.

FAILURE MODE B — all narrow coverage.
If every CQ you emit has coverage_breadth "narrow" (1–2 papers), you
have failed. Either find patterns that span 3+ papers and mark them
"medium" or "broad", OR emit a single diagnostic entry:
  {{"gen_question": "No cross-paper patterns detected — corpus review recommended.",
    "abstraction_reasoning": "DIAGNOSTIC: all {len(questions)} inputs are genuinely distinct.",
    "member_ids": []}}
Silent all-narrow output is the worst possible result.

FAILURE MODE C — padded reasoning.
Do NOT write abstraction_reasoning as a restatement of the question.
"Both questions ask about how X affects Y" is a restatement, not
reasoning. Follow the structured format shown in the output template.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INPUT CQs  (archetype: {archetype}, {len(questions)} total)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{questions_block}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{output_template}
"""

    @staticmethod
    def _format_questions_block(questions: List[dict]) -> str:
        lines: List[str] = []
        for q in questions:
            lines.append(
                f"ID      : {q['id']}\n"
                f"Paper   : {q.get('paper_id','?')}  |  "
                f"Persona : {q.get('persona_code','?')}  |  "
                f"Bloom   : {q.get('bloom_level','?')}\n"
                f"Question: {q.get('question','')}\n"
                f"Answer  : {q.get('expected_answer','')}\n"
            )
        return "\n".join(lines)

    @staticmethod
    def _output_template() -> str:
        return """{
  "groups": [
    {
      "gen_question"         : "<domain-level generalised CQ covering all members>",
      "abstraction_reasoning": "PATTERN: <the relational template shared by these CQs> | ENTITY MAPPING: <input entity 1> -> <domain role>, <input entity 2> -> <domain role> | WHY MEDIUM/BROAD: <one specific reason citing which inputs support this breadth>",
      "member_ids"           : ["<DEDUP_ID_1>", "<DEDUP_ID_2>"]
    }
  ],
  "singletons": [
    {
      "gen_question"         : "<domain-level generalised CQ (paper-specific entities removed)>",
      "abstraction_reasoning": "PATTERN: <the relational template> | ENTITY MAPPING: <input entity> -> <domain role> | WHY NARROW: <reason no other paper covers this pattern>",
      "member_ids"           : ["<DEDUP_ID>"]
    }
  ]
}"""

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse(
        self,
        response_text: str,
        thinking_text: str,
        archetype:     str,
        questions:     List[dict],
        gen_counter:   Dict[str, int],
    ) -> List[dict]:
        """Parse LLM output into a list of GenCQ dicts."""

        parsed = self._extract_json(response_text)
        if parsed is None:
            print(f"  [Gen] WARNING: parse failed for {archetype} — using fallback (all singletons)")
            return self._fallback_gen_cqs(archetype, questions, gen_counter)

        # Merge groups + singletons into one unified list
        all_entries: List[dict] = (
            [{**e, "_type": "group"}    for e in parsed.get("groups",    [])]
          + [{**e, "_type": "singleton"} for e in parsed.get("singletons", [])]
        )

        # Build lookup: DEDUP_ID → original question dict
        rq_by_id  = {q["id"]: q for q in questions}
        valid_ids = set(rq_by_id.keys())

        # ── Option B: strip hallucinated IDs from every entry ─────────
        n_hallucinated = 0
        for entry in all_entries:
            raw_ids      = entry.get("member_ids", [])
            clean_ids    = [m for m in raw_ids if m in valid_ids]
            n_hallucinated += len(raw_ids) - len(clean_ids)
            entry["member_ids"] = clean_ids

        if n_hallucinated:
            print(f"  [Gen] WARNING: {n_hallucinated} hallucinated ID(s) stripped "
                  f"from LLM output for archetype {archetype}")

        # Drop entries that have no valid members left after stripping
        all_entries = [e for e in all_entries if e.get("member_ids")]

        # Guard: IDs the LLM didn't mention → add as singletons
        mentioned_ids: set[str] = set()
        for entry in all_entries:
            mentioned_ids.update(entry.get("member_ids", []))

        unmentioned = {q["id"] for q in questions} - mentioned_ids
        if unmentioned:
            print(f"  [Gen] {len(unmentioned)} unmentioned ID(s) added as singletons")
            for uid in sorted(unmentioned):
                uq = rq_by_id.get(uid, {})
                all_entries.append({
                    "_type":               "singleton",
                    "gen_question":        uq.get("question", ""),
                    "abstraction_reasoning": "FALLBACK: not mentioned by LLM; original question used",
                    "member_ids":          [uid],
                })

        # Build GenCQ objects
        gen_cqs: List[dict] = []

        for entry in all_entries:
            member_ids = entry.get("member_ids", [])
            if not member_ids:
                continue

            # Assign GenCQ ID
            gen_counter[archetype] = gen_counter.get(archetype, 0) + 1
            gen_id = f"GEN_{archetype}_{gen_counter[archetype]:03d}"

            # Collect original CQ details for every member
            original_cqs: List[dict] = []
            papers_covered: List[str] = []
            bloom_levels: List[str]   = []

            for mid in member_ids:
                rq = rq_by_id.get(mid, {})
                pid = rq.get("paper_id", "UNKNOWN")
                if pid not in papers_covered:
                    papers_covered.append(pid)
                bloom_levels.append(rq.get("bloom_level", ""))
                original_cqs.append({
                    "step1_id":       mid,
                    "paper_id":       pid,
                    "source_id":      rq.get("source_id", ""),    # original v2 ID
                    "question":       rq.get("question", ""),
                    "expected_answer": rq.get("expected_answer", ""),
                    "persona_code":   rq.get("persona_code", ""),
                    "bloom_level":    rq.get("bloom_level", ""),
                    "source_section": rq.get("source_section", "Body"),
                })

            n_papers = len(papers_covered)

            # Resolve bloom level for the GenCQ (highest among members)
            _bloom_order = {
                "recall": 1, "comprehension": 2, "application": 3,
                "analysis": 4, "synthesis": 5, "evaluation": 6,
            }
            gen_bloom = max(
                bloom_levels,
                key=lambda b: _bloom_order.get(b, 0),
                default="comprehension",
            )

            gen_cqs.append({
                "gen_id":              gen_id,
                "archetype":           archetype,
                "bloom_level":         gen_bloom,
                "generalized_question": entry.get("gen_question", ""),
                "abstraction_reasoning": entry.get("abstraction_reasoning", ""),
                "is_singleton_group":  entry["_type"] == "singleton",
                "coverage_breadth":    _coverage_breadth(n_papers),
                "n_papers_covered":    n_papers,
                "papers_covered":      sorted(papers_covered),
                "original_cqs":        original_cqs,
            })

        return gen_cqs

    def _fallback_gen_cqs(
        self,
        archetype:   str,
        questions:   List[dict],
        gen_counter: Dict[str, int],
    ) -> List[dict]:
        """When parsing fails, treat every question as its own singleton GenCQ."""
        gen_cqs: List[dict] = []
        for q in questions:
            gen_counter[archetype] = gen_counter.get(archetype, 0) + 1
            gen_id = f"GEN_{archetype}_{gen_counter[archetype]:03d}"
            gen_cqs.append({
                "gen_id":               gen_id,
                "archetype":            archetype,
                "bloom_level":          q.get("bloom_level", "comprehension"),
                "generalized_question": q.get("question", ""),
                "abstraction_reasoning": "PARSE_FALLBACK — original question used unchanged",
                "is_singleton_group":   True,
                "coverage_breadth":     "narrow",
                "n_papers_covered":     1,
                "papers_covered":       [q.get("paper_id", "UNKNOWN")],
                "original_cqs": [{
                    "step1_id":       q["id"],
                    "paper_id":       q.get("paper_id", "UNKNOWN"),
                    "source_id":      q.get("source_id", ""),
                    "question":       q.get("question", ""),
                    "expected_answer": q.get("expected_answer", ""),
                    "persona_code":   q.get("persona_code", ""),
                    "bloom_level":    q.get("bloom_level", ""),
                    "source_section": q.get("source_section", "Body"),
                }],
            })
        return gen_cqs

    # ------------------------------------------------------------------
    # JSON extraction helper
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_json(text: str) -> Optional[dict]:
        if not text:
            return None
        cleaned = text.strip()
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
    # Output assembly
    # ------------------------------------------------------------------

    def _build_output(
        self,
        all_gen_cqs:     List[dict],
        archetype_stats: Dict[str, dict],
        all_rqs:         List[dict],
        paper_ids:       List[str],
        llm_calls_all:   List[dict],
        total_elapsed:   float,
    ) -> dict:

        n_input  = len(all_rqs)
        n_output = len(all_gen_cqs)

        # Archetype-level summary
        arch_summary = {
            arch: {
                "n_input_cqs": s["n_input_cqs"],
                "n_gen_cqs":   s["n_gen_cqs"],
                "n_papers":    s["n_papers"],
                "wall_time":   s["wall_time"],
            }
            for arch, s in archetype_stats.items()
        }

        # Coverage breadth counts
        breadth_counts = {"narrow": 0, "medium": 0, "broad": 0}
        for g in all_gen_cqs:
            breadth_counts[g["coverage_breadth"]] = (
                breadth_counts.get(g["coverage_breadth"], 0) + 1
            )

        # Reverse index: step1_id → gen_id (for Step 3 lookup)
        reverse_index: Dict[str, str] = {}
        for g in all_gen_cqs:
            for oq in g["original_cqs"]:
                reverse_index[oq["step1_id"]] = g["gen_id"]

        return {
            # ── Identity ──────────────────────────────────────────────
            "pipeline_step":   "cross_paper_generalization",
            "pipeline_version": "v2",
            "step_version":    self.STEP_VERSION,

            # ── Config ────────────────────────────────────────────────
            "generalize_config": {
                "llm_model":      self.model_name,
                "think":          self.think,
                "format":         "json",
                "prompt_version": self.PROMPT_VERSION,
                "source_dir":     "data/papers/v2/dedup/",
            },

            # ── Run summary ───────────────────────────────────────────
            "run_stats": {
                "n_input_papers":  len(paper_ids),
                "papers":          paper_ids,
                "n_input_cqs":     n_input,
                "n_gen_cqs":       n_output,
                "reduction_rate":  round((n_input - n_output) / n_input, 3) if n_input else 0.0,
                "coverage_breadth_counts": breadth_counts,
                "archetype_breakdown":     arch_summary,
            },

            # ── Execution stats ───────────────────────────────────────
            "execution_stats": {
                "total_wall_time": fmt_duration(total_elapsed),
                "n_llm_calls":     len(llm_calls_all),
                "llm_calls":       [
                    {k: v for k, v in c.items()
                     if k not in ("total_duration", "load_duration",
                                  "prompt_eval_duration", "eval_duration")}
                    for c in llm_calls_all
                ],
            },

            # ── Step 2 outputs ────────────────────────────────────────
            "generalized_cqs": all_gen_cqs,

            # ── Reverse index for Step 3 ──────────────────────────────
            # step1_id (DEDUP) → gen_id
            "reverse_index": reverse_index,
        }
