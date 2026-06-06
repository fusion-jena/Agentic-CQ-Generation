"""
Step 3 of the CQ narrowing pipeline: answer retrieval + validation.

For every GenCQ produced by Step 2, and for every original CQ it maps to:

  1. RAG — retrieve the top-K most relevant chunks from the paper's existing
     FAISS index using the *generalised* question as the query.

  2. Generate + Judge (one LLM call) — ask the LLM to:
       a. Answer the generalised question using the retrieved evidence.
       b. Compare its answer to the original expected_answer.
       c. Return a structured fidelity score + verdict.

  3. Aggregate — roll per-original-CQ verdicts up to a GenCQ-level verdict
     and assign a quality_flag used for downstream ontology decisions.

Design principles
─────────────────
• One LLM call per (GenCQ, original_cq) pair — answer generation and
  fidelity judging are combined to halve the call count.
• NOT_FOUND is a distinct verdict from FAIL. NOT_FOUND means the FAISS
  index returned no relevant chunks (score < threshold) — the GenCQ may
  be over-generalised or the paper may not cover this aspect. FAIL means
  the LLM could find relevant text but the answer didn't match.
• All original expected_answers from merged_from (Step 1) are also stored
  in original_cqs (Step 2 carries them through), so the judge sees the
  full ground truth for each paper.
• Missing FAISS index → verdict = INDEX_MISSING (not a pipeline error,
  just logged and recorded — pipeline continues).
• think=False for Step 3: there are potentially 80+ LLM calls; speed
  matters more than chain-of-thought here. Fidelity reasoning is captured
  in the structured JSON output instead.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from utils.embeddings import Embedder
from utils.llm_client import CustomOllamaClient, fmt_duration
from utils.vectorstore import PaperVectorStore, RetrievedChunk

# ── Thresholds ────────────────────────────────────────────────────────────────
_RELEVANCE_THRESHOLD = 0.40   # min cosine score to consider a chunk relevant
_PASS_THRESHOLD      = 0.75   # fidelity score → PASS
_PARTIAL_THRESHOLD   = 0.40   # fidelity score → PARTIAL (below = FAIL)
_TOP_K_CHUNKS        = 5      # chunks retrieved per GenCQ per paper


def _overall_verdict(
    n_passed: int,
    n_partial: int,
    n_failed: int,
    n_not_found: int,
    n_missing: int,
) -> str:
    n_total = n_passed + n_partial + n_failed + n_not_found + n_missing
    if n_total == 0:
        return "NO_DATA"
    if n_not_found + n_missing == n_total:
        return "NOT_FOUND"
    answerable = n_passed + n_partial + n_failed
    if answerable == 0:
        return "NOT_FOUND"
    pass_rate = n_passed / answerable
    if pass_rate >= 0.75:
        return "PASS"
    if pass_rate >= 0.40:
        return "PARTIAL"
    return "FAIL"


def _quality_flag(
    overall: str,
    n_papers: int,
    n_not_found: int,
    n_missing: int,
    mean_score: float,
) -> str:
    if overall == "NOT_FOUND" or (n_not_found + n_missing) > 0 and overall != "PASS":
        return "OVER_GENERALISED"
    if overall == "PASS" and n_papers == 1:
        return "PAPER_SPECIFIC"
    if overall == "PASS" and mean_score >= 0.75:
        return "STRONG"
    if overall in ("PARTIAL", "FAIL") and mean_score < _PARTIAL_THRESHOLD:
        return "UNDER_ANSWERED"
    if overall == "PARTIAL":
        return "PARTIAL"
    return "STRONG"


class CQValidatorAgent:
    """
    RAG-based answer generation + fidelity validation agent for Step 3.

    Reuses the existing FAISS indexes built by the v2 pipeline — no
    re-indexing needed.
    """

    STEP_VERSION:   str = "step3_v1.0"
    PROMPT_VERSION: str = "validate_prompt_v1.0"

    def __init__(
        self,
        model_name:    str,
        embed_model:   str = "BAAI/bge-small-en-v1.5",
        index_dir:     Optional[Path] = None,
        top_k:         int = _TOP_K_CHUNKS,
    ):
        self.model_name  = model_name
        self.embed_model = embed_model
        self.top_k       = top_k
        self.llm         = CustomOllamaClient(model=model_name)
        self.embedder    = Embedder(model_name=embed_model)

        _root = Path(__file__).parent.parent
        self.index_dir = index_dir or (_root / "data" / "papers" / "v2" / "index")

        # Cache loaded vector stores — avoid reloading the same paper index
        # across multiple GenCQs
        self._store_cache: Dict[str, PaperVectorStore] = {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def validate_all(self, gen_path: Path) -> dict:
        """
        Validate all GenCQs from a Step 2 output file.

        Args:
            gen_path: path to cross_paper_gen_v2.json

        Returns:
            Full Step 3 output dict.
        """
        gen_data    = json.loads(gen_path.read_text(encoding="utf-8"))
        gen_cqs     = gen_data.get("generalized_cqs", [])
        paper_ids   = gen_data.get("run_stats", {}).get("papers", [])

        print(f"  [Val] {len(gen_cqs)} GenCQs across {len(paper_ids)} paper(s)")

        results:       List[dict] = []
        all_llm_calls: List[dict] = []

        verdict_counts = {"PASS": 0, "PARTIAL": 0, "FAIL": 0,
                          "NOT_FOUND": 0, "INDEX_MISSING": 0}
        flag_counts: Dict[str, int] = {}

        t_total = time.time()

        for i, gen_cq in enumerate(gen_cqs, 1):
            print(f"  [Val] [{i}/{len(gen_cqs)}] {gen_cq['gen_id']} — "
                  f"{gen_cq['generalized_question'][:70]}...")

            result, llm_calls = self._validate_gen_cq(gen_cq)
            results.append(result)
            all_llm_calls.extend(llm_calls)

            ov = result["aggregate"]["overall_verdict"]
            verdict_counts[ov] = verdict_counts.get(ov, 0) + 1
            qf = result["aggregate"]["quality_flag"]
            flag_counts[qf] = flag_counts.get(qf, 0) + 1

        total_elapsed = time.time() - t_total

        return self._build_output(
            results        = results,
            gen_data       = gen_data,
            gen_path       = gen_path,
            all_llm_calls  = all_llm_calls,
            verdict_counts = verdict_counts,
            flag_counts    = flag_counts,
            total_elapsed  = total_elapsed,
        )

    # ------------------------------------------------------------------
    # Per-GenCQ validation
    # ------------------------------------------------------------------

    def _validate_gen_cq(
        self,
        gen_cq: dict,
    ) -> Tuple[dict, List[dict]]:
        """
        Validate one GenCQ against all its original CQs.
        Returns (result_dict, llm_calls_list).
        """
        gen_id       = gen_cq["gen_id"]
        gen_question = gen_cq["generalized_question"]
        archetype    = gen_cq.get("archetype", "")
        original_cqs = gen_cq.get("original_cqs", [])

        per_paper_results: List[dict] = []
        llm_calls_here:    List[dict] = []

        for oq in original_cqs:
            paper_id = oq["paper_id"]
            step1_id = oq["step1_id"]

            # ── Load vector store ──────────────────────────────────────
            store, missing = self._get_store(paper_id)
            if missing:
                per_paper_results.append({
                    "paper_id":               paper_id,
                    "step1_id":               step1_id,
                    "source_id":              oq.get("source_id", ""),
                    "original_question":      oq.get("question", ""),
                    "original_expected_answer": oq.get("expected_answer", ""),
                    "retrieved_chunks":       [],
                    "generated_answer":       None,
                    "fidelity": {
                        "score":    0.0,
                        "verdict":  "INDEX_MISSING",
                        "reasoning": f"No FAISS index found for paper '{paper_id}' "
                                     f"in {self.index_dir}",
                    },
                })
                continue

            # ── RAG: retrieve chunks ───────────────────────────────────
            chunks = store.search(gen_question, k=self.top_k)
            max_score = max((c.score for c in chunks), default=0.0)

            if max_score < _RELEVANCE_THRESHOLD:
                per_paper_results.append({
                    "paper_id":               paper_id,
                    "step1_id":               step1_id,
                    "source_id":              oq.get("source_id", ""),
                    "original_question":      oq.get("question", ""),
                    "original_expected_answer": oq.get("expected_answer", ""),
                    "retrieved_chunks":       [
                        {"chunk_id": c.chunk_id, "score": round(c.score, 4),
                         "text": c.text[:200]}
                        for c in chunks
                    ],
                    "generated_answer": None,
                    "fidelity": {
                        "score":     0.0,
                        "verdict":   "NOT_FOUND",
                        "reasoning": f"Best chunk score {max_score:.3f} below threshold "
                                     f"{_RELEVANCE_THRESHOLD} — paper may not cover this aspect",
                    },
                })
                continue

            # ── LLM: generate answer + judge fidelity ─────────────────
            t_llm = time.time()
            prompt = self._build_prompt(
                gen_question = gen_question,
                chunks       = chunks,
                expected_answer = oq.get("expected_answer", ""),
                paper_id     = paper_id,
            )
            response = self.llm.invoke(prompt, format="json")
            llm_elapsed = time.time() - t_llm

            parsed = self._parse_validation(response)
            llm_stats = self.llm.drain_stats()
            llm_calls_here.extend(llm_stats)

            per_paper_results.append({
                "paper_id":               paper_id,
                "step1_id":               step1_id,
                "source_id":              oq.get("source_id", ""),
                "original_question":      oq.get("question", ""),
                "original_expected_answer": oq.get("expected_answer", ""),
                "retrieved_chunks": [
                    {"chunk_id": c.chunk_id, "score": round(c.score, 4),
                     "text": c.text[:300]}
                    for c in chunks
                ],
                "generated_answer": parsed.get("generated_answer", ""),
                "fidelity": {
                    "score":    parsed.get("fidelity_score", 0.0),
                    "verdict":  parsed.get("verdict", "FAIL"),
                    "reasoning": parsed.get("fidelity_reasoning", ""),
                },
            })

        # ── Aggregate ──────────────────────────────────────────────────
        aggregate = self._aggregate(per_paper_results, gen_cq)

        return (
            {
                "gen_id":               gen_id,
                "generalized_question": gen_question,
                "archetype":            archetype,
                "coverage_breadth":     gen_cq.get("coverage_breadth", ""),
                "n_papers_covered":     gen_cq.get("n_papers_covered", 0),
                "per_paper_results":    per_paper_results,
                "aggregate":            aggregate,
            },
            llm_calls_here,
        )

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        gen_question:    str,
        chunks:          List[RetrievedChunk],
        expected_answer: str,
        paper_id:        str,
    ) -> str:

        evidence_block = "\n\n".join(
            f"[Chunk {i+1}  score={c.score:.3f}]\n{c.text}"
            for i, c in enumerate(chunks)
        )

        return f"""You are validating a competency question for a copolymer knowledge graph.

GENERALISED QUESTION:
{gen_question}

EVIDENCE RETRIEVED FROM PAPER "{paper_id}":
{evidence_block}

ORIGINAL EXPECTED ANSWER (ground truth for this paper):
{expected_answer}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR TASKS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. ANSWER the generalised question using ONLY the retrieved evidence above.
   - If the evidence does not contain enough information, say so explicitly.
   - Do NOT use prior knowledge — only what appears in the evidence chunks.

2. COMPARE your answer to the original expected answer.
   - Focus on factual content, not wording.
   - A good match means your answer captures the same key facts/relationships.

3. SCORE the fidelity (0.0 to 1.0):
   - 1.0 = your answer fully matches the expected answer
   - 0.75-0.99 = most key facts match, minor gaps
   - 0.40-0.74 = some facts match but important content is missing
   - 0.0-0.39 = does not match or evidence insufficient

4. VERDICT:
   - "PASS"    if score ≥ 0.75
   - "PARTIAL" if score 0.40–0.74
   - "FAIL"    if score < 0.40

Return ONLY a JSON object — no markdown fences:
{{
  "generated_answer"  : "<your answer based on the evidence>",
  "fidelity_score"    : <float 0.0-1.0>,
  "verdict"           : "<PASS|PARTIAL|FAIL>",
  "fidelity_reasoning": "<one or two sentences: what matched and what was missing>"
}}"""

    # ------------------------------------------------------------------
    # Parse validation response
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_validation(response: str) -> dict:
        """Extract and normalise the validation JSON from LLM output."""
        fallback = {
            "generated_answer":   "",
            "fidelity_score":     0.0,
            "verdict":            "FAIL",
            "fidelity_reasoning": "Parse error — LLM response could not be parsed",
        }
        if not response:
            return fallback

        cleaned = response.strip()
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
                parsed = json.loads(cleaned[start:end])
                # Normalise score to float in [0, 1]
                score = float(parsed.get("fidelity_score", 0.0))
                score = max(0.0, min(1.0, score))
                parsed["fidelity_score"] = score

                # Normalise verdict
                raw_verdict = str(parsed.get("verdict", "")).upper()
                if raw_verdict not in ("PASS", "PARTIAL", "FAIL"):
                    raw_verdict = (
                        "PASS"    if score >= _PASS_THRESHOLD else
                        "PARTIAL" if score >= _PARTIAL_THRESHOLD else
                        "FAIL"
                    )
                parsed["verdict"] = raw_verdict
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass

        return fallback

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate(per_paper_results: List[dict], gen_cq: dict) -> dict:
        n_passed   = sum(1 for r in per_paper_results if r["fidelity"]["verdict"] == "PASS")
        n_partial  = sum(1 for r in per_paper_results if r["fidelity"]["verdict"] == "PARTIAL")
        n_failed   = sum(1 for r in per_paper_results if r["fidelity"]["verdict"] == "FAIL")
        n_not_found = sum(1 for r in per_paper_results if r["fidelity"]["verdict"] == "NOT_FOUND")
        n_missing  = sum(1 for r in per_paper_results if r["fidelity"]["verdict"] == "INDEX_MISSING")
        n_tested   = len(per_paper_results)

        scores = [
            r["fidelity"]["score"]
            for r in per_paper_results
            if r["fidelity"]["verdict"] not in ("NOT_FOUND", "INDEX_MISSING")
        ]
        mean_score = round(sum(scores) / len(scores), 3) if scores else 0.0

        overall = _overall_verdict(n_passed, n_partial, n_failed, n_not_found, n_missing)
        qflag   = _quality_flag(
            overall,
            n_papers  = gen_cq.get("n_papers_covered", 1),
            n_not_found = n_not_found,
            n_missing   = n_missing,
            mean_score  = mean_score,
        )

        return {
            "n_tested":          n_tested,
            "n_passed":          n_passed,
            "n_partial":         n_partial,
            "n_failed":          n_failed,
            "n_not_found":       n_not_found,
            "n_index_missing":   n_missing,
            "mean_fidelity_score": mean_score,
            "overall_verdict":   overall,
            "quality_flag":      qflag,
        }

    # ------------------------------------------------------------------
    # Vector store loading (with cache)
    # ------------------------------------------------------------------

    def _get_store(self, paper_id: str) -> Tuple[Optional[PaperVectorStore], bool]:
        """
        Load the FAISS index for paper_id. Returns (store, is_missing).
        Caches loaded stores to avoid re-loading across multiple GenCQs.
        """
        if paper_id in self._store_cache:
            return self._store_cache[paper_id], False

        index_path = self.index_dir / f"{paper_id}.faiss"
        meta_path  = self.index_dir / f"{paper_id}_meta.json"

        if not index_path.exists() or not meta_path.exists():
            print(f"  [Val] WARNING: No FAISS index for '{paper_id}' — marking as INDEX_MISSING")
            return None, True

        store = PaperVectorStore(embedder=self.embedder)
        store.load(index_path, meta_path)
        self._store_cache[paper_id] = store
        return store, False

    # ------------------------------------------------------------------
    # Output assembly
    # ------------------------------------------------------------------

    def _build_output(
        self,
        results:        List[dict],
        gen_data:       dict,
        gen_path:       Path,
        all_llm_calls:  List[dict],
        verdict_counts: Dict[str, int],
        flag_counts:    Dict[str, int],
        total_elapsed:  float,
    ) -> dict:

        n_tested = sum(r["aggregate"]["n_tested"] for r in results)
        all_scores = [
            r["aggregate"]["mean_fidelity_score"]
            for r in results
            if r["aggregate"]["mean_fidelity_score"] > 0
        ]
        overall_mean = round(sum(all_scores) / len(all_scores), 3) if all_scores else 0.0

        _root = Path(__file__).parent.parent
        try:
            rel_gen = str(gen_path.relative_to(_root))
        except ValueError:
            rel_gen = gen_path.name

        return {
            # ── Identity ──────────────────────────────────────────────
            "pipeline_step":    "answer_validation",
            "pipeline_version": "v2",
            "step_version":     self.STEP_VERSION,

            # ── Config ────────────────────────────────────────────────
            "validation_config": {
                "llm_model":          self.model_name,
                "embed_model":        self.embed_model,
                "think":              False,
                "format":             "json",
                "prompt_version":     self.PROMPT_VERSION,
                "top_k_chunks":       self.top_k,
                "relevance_threshold": _RELEVANCE_THRESHOLD,
                "pass_threshold":     _PASS_THRESHOLD,
                "partial_threshold":  _PARTIAL_THRESHOLD,
                "source_file":        rel_gen,
            },

            # ── Run summary ───────────────────────────────────────────
            "run_stats": {
                "n_gen_cqs":             len(results),
                "n_original_cqs_tested": n_tested,
                "overall_mean_fidelity": overall_mean,
                "verdict_counts":        verdict_counts,
                "quality_flag_counts":   flag_counts,
            },

            # ── Execution stats ───────────────────────────────────────
            "execution_stats": {
                "total_wall_time": fmt_duration(total_elapsed),
                "n_llm_calls":     len(all_llm_calls),
            },

            # ── Per-GenCQ results ─────────────────────────────────────
            "results": results,
        }
