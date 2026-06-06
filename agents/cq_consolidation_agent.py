"""
Step 4 of the CQ narrowing pipeline: cross-archetype consolidation +
ontology schema extraction.

Two-stage batched approach (v2):

  Stage A — per-archetype intra-merge (one LLM call per archetype, ~20-96 CQs each).
            ONTOLOGY_TRIPLE is purely meta and bypasses Stage A entirely.
  Stage B — cross-archetype merge + schema extraction (one call over Stage A results).

Replaces the original single-pass approach which fed all 490 CQs in one
call, causing the model to drop ~91% of them due to context overload.

Requires: pip install "pydantic>=2"
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Literal, Optional

try:
    from pydantic import BaseModel, ConfigDict, Field, ValidationError
except ImportError:
    raise ImportError(
        "This agent requires Pydantic v2.  Install it with:\n"
        "    pip install \"pydantic>=2\""
    )

from utils.llm_client import CustomOllamaClient, fmt_duration


# =========================================================================
# Pydantic v2 output schemas
# =========================================================================

class Binding(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    instance: str
    paper: str


class Variable(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    name: str
    class_: str = Field(alias="class")
    bindings: List[Binding] = Field(default_factory=list)


class Triple(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    subject: str
    predicate: str
    object_: str = Field(alias="object")


class ConsolidatedCQ(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    cq_id: str
    competency_question: str
    variables: List[Variable] = Field(default_factory=list)
    triple_pattern: List[Triple] = Field(min_length=1)
    sparql_sketch: Optional[str] = None
    source_archetypes: List[str] = Field(default_factory=list)
    merged_from: List[str] = Field(min_length=1)
    papers_covered: List[str] = Field(default_factory=list)
    n_papers: int
    coverage_breadth: Literal["narrow", "medium", "broad"]
    consolidation_reasoning: Optional[str] = None


class ConsolidationRun(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    n_input_cqs: int
    n_consolidated_cqs: int
    reduction_rate: float


class ConsolidationOutput(BaseModel):
    """Top-level contract returned by Stage B."""
    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    consolidation_run: ConsolidationRun
    consolidated_cqs: List[ConsolidatedCQ] = Field(min_length=1)


class StageAOutput(BaseModel):
    """Simpler contract for Stage A (no schema, no folded CQs needed)."""
    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    consolidated_cqs: List[ConsolidatedCQ] = Field(min_length=1)


# =========================================================================
# Stage A prompt — intra-archetype merge only
# =========================================================================
_STAGE_A_PROMPT = (
    "# CQ INTRA-ARCHETYPE CONSOLIDATION — Stage A\n"
    "You are a senior ontology engineer. You are given BATCH_N generalised\n"
    "competency questions (CQs), all from the archetype: BATCH_ARCHETYPE.\n"
    "\n"
    "Your task: merge CQs that share the same triple signature\n"
    "(SubjectClass, predicate, ObjectClass). Different filler values (specific\n"
    "polymer, property name) become BOUND VARIABLES — not separate CQs.\n"
    "\n"
    "## INPUTS\n"
    "BATCH_INPUT\n"
    "Each item: gen_id, archetype, generalized_question, papers_covered.\n"
    "\n"
    "## CONTROLLED PREDICATE VOCABULARY (reuse; do not coin synonyms)\n"
    "producedBy (Material->Process)  hasParameter (Process->ProcessParameter)\n"
    "performedUnder (Process->ProcessCondition)  affects (Entity->Property)\n"
    "characterisedBy (Material->Property)  hasValue (Property->xsd literal)\n"
    "measuredBy (Property->Method)  suitableFor (Material->Application)\n"
    "hasLimitation (Entity->Limitation)  hasComponent (Material->Material)\n"
    "rdf:type  rdfs:subClassOf\n"
    "\n"
    "## RULES\n"
    "R1  Same signature, different filler -> ONE CQ. Record each filler + paper\n"
    "    in variables[].bindings.\n"
    "R2  Different predicate OR different subject/object class -> keep separate.\n"
    "    Do NOT over-merge.\n"
    "R3  coverage_breadth: broad >=5 papers  medium 2-4  narrow 1.\n"
    "R4  PROVENANCE IS MANDATORY — merged_from must never be empty:\n"
    "    Every input gen_id appears in exactly one merged_from list.\n"
    "    If a CQ has no merge partner, put its own gen_id alone:\n"
    "      \"merged_from\": [\"GEN_CAUSAL_007\"]\n"
    "    A CQ with merged_from=[] is invalid and will be rejected.\n"
    "R5  No duplicate competency_question text.\n"
    "R6  Write a concise consolidation_reasoning (1-2 sentences) per output CQ.\n"
    "\n"
    "## OUTPUT\n"
    "Return ONE JSON object with key \"consolidated_cqs\" containing a list.\n"
    "Each element fields: cq_id (use SA_BATCH_ABBREV_001, _002, ...),\n"
    "competency_question, variables (list), triple_pattern (list of\n"
    "subject/predicate/object), sparql_sketch (SELECT...WHERE{...}),\n"
    "source_archetypes, merged_from (list of gen_ids from this batch),\n"
    "papers_covered (union of all merged), n_papers (int),\n"
    "coverage_breadth, consolidation_reasoning.\n"
    "No prose, no markdown fences.\n"
    "\n"
    "Consolidate the BATCH_N BATCH_ARCHETYPE CQs now.\n"
)

# =========================================================================
# Stage B prompt — cross-archetype merge + schema extraction
# =========================================================================
_STAGE_B_PROMPT = r"""# CQ CROSS-ARCHETYPE CONSOLIDATION — Stage B
You are a senior ontology engineer. You are given a list of already
intra-archetype-merged CQs from Stage A. Different archetypes may still
capture the SAME triple signature — merge them now.

## INPUTS
```json
{INPUT_JSON}
```
Each item: gen_id (Stage A id), archetype, generalized_question, papers_covered.

## CONTROLLED PREDICATE VOCABULARY (reuse; do not coin synonyms)
producedBy (Material->Process); hasParameter (Process->ProcessParameter);
performedUnder (Process->ProcessCondition); affects (Entity->Property);
characterisedBy (Material->Property); hasValue (Property->xsd literal, data property);
measuredBy (Property->Method); suitableFor (Material/Process->Application);
hasLimitation (Entity->Limitation); hasComponent (Material->Material/Monomer);
rdf:type / rdfs:subClassOf.

## RULES
R1 Bound variables, not abstraction: shared signature differing only on a
   filler -> ONE CQ with a typed variable and bindings.
R2 Cross-archetype merging is REQUIRED. List all contributing archetypes in
   source_archetypes (length > 1 expected and good).
R3 Do NOT over-merge: different predicate OR different subject/object class
   = different signature = keep separate.
R4 Collapse literal duplicates to one. Never emit the same
   competency_question text twice.
R5 coverage_breadth: broad if >=5 papers, medium if 2-4, narrow if 1.
R6 PROVENANCE IS MANDATORY — empty merged_from is a hard error:
   - Every input gen_id appears in exactly one merged_from list.
     Nothing dropped, nothing skipped.
   - merged_from must contain AT LEAST ONE gen_id. A CQ with
     merged_from=[] is invalid and will be rejected by the validator.
   - If a CQ was not merged with anything, put its own gen_id alone:
     "merged_from": ["SA_CAUSAL_003"]
   - If three CQs were merged: "merged_from": ["SA_EL_001","SA_CAUSAL_007","SA_COMP_002"]
R7 Write a concise consolidation_reasoning (1-2 sentences) for every CQ.

## OUTPUT
Return ONE JSON object with keys: consolidation_run, consolidated_cqs.
No prose, no markdown fences.

## PROCEDURE (internal; output only the final JSON)
1 Derive each CQ's triple signature.
2 Bucket CQs by signature (ignore archetype while bucketing).
3 Collapse literal duplicates (R4).
4 Emit one CQ per signature bucket. In merged_from list ALL input
  gen_ids that contributed — every single one, no exceptions.
5 SELF-CHECK: count input gen_ids vs sum of all merged_from lengths —
  the totals must match. No merged_from is empty. Fix before output.

Now consolidate the Stage A results.
"""


# =========================================================================
# Archetype abbreviation map for Stage A cq_id prefixes
# =========================================================================
_ARCHETYPE_ABBREV = {
    "CAUSAL":                "CAUSAL",
    "COMPARATIVE":           "COMP",
    "CONDITION_FOR_PROCESS": "CFP",
    "ENTITY_LOOKUP":         "EL",
    "ENUMERATION":           "ENUM",
    "GAP_OR_LIMITATION":     "GAP",
    "METHOD_OF_MEASUREMENT": "MOM",
    "MOTIVATION":            "MOTIV",
    "PROCESS_FOR_MATERIAL":  "PFM",
}

class CQConsolidationAgent:
    """
    Step 4: two-stage cross-archetype CQ consolidation.

    Stage A: per-archetype intra-merge (one call per archetype, ~20-96 CQs).
    Stage B: cross-archetype merge (one call over Stage A results).
    """

    STEP_VERSION:   str = "step4_v2.4"
    PROMPT_VERSION: str = "consolidate_prompt_v5.0"

    def __init__(
        self,
        model_name:       str,
        no_schema_format: bool  = False,
        timeout:          int   = 1800,
        temperature:      float = 0.2,
        num_ctx:          int   = 65536,
    ):
        self.model_name       = model_name
        self.no_schema_format = no_schema_format
        self.timeout          = timeout
        self.temperature      = temperature
        self.num_ctx          = num_ctx
        self.llm = CustomOllamaClient(model=model_name)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def consolidate(self, gen_path: Path) -> dict:
        """
        Run two-stage consolidation + schema extraction.

        Stage A: one call per archetype (intra-archetype merge).
        Stage B: one call across all Stage A results (cross-archetype + schema).
        """
        original = json.loads(gen_path.read_text(encoding="utf-8"))
        src = original["generalized_cqs"]

        # Group by archetype, keep only fields the model needs
        by_archetype: dict[str, list] = defaultdict(list)
        for g in src:
            by_archetype[g.get("archetype", "UNKNOWN")].append({
                "gen_id":               g["gen_id"],
                "archetype":            g.get("archetype", ""),
                "generalized_question": g["generalized_question"],
                "papers_covered":       g.get("papers_covered", []),
            })

        # ── Stage A: per-archetype intra-merge ────────────────────────
        print(f"\n[Consolidate] Stage A — {len(by_archetype)} archetype batches:")

        stage_a_cqs: list[ConsolidatedCQ] = []
        stage_a_map: dict[str, list[str]] = {}  # sa_id -> [orig gen_ids]
        total_wall      = 0.0
        n_stage_a_calls = 0

        for archetype, batch in sorted(by_archetype.items()):
            print(f"  [{archetype}] {len(batch)} CQs ...", end=" ", flush=True)
            partial_cqs, wall = self._stage_a_call(batch, archetype)
            total_wall += wall
            n_stage_a_calls += 1

            for cq in partial_cqs:
                stage_a_cqs.append(cq)
                stage_a_map[cq.cq_id] = cq.merged_from

            print(f"-> {len(partial_cqs)} merged ({fmt_duration(wall)})")

        n_domain = sum(len(b) for b in by_archetype.values())
        print(f"  Stage A total: {n_domain} CQs -> {len(stage_a_cqs)} merged")

        # ── Stage B: cross-archetype merge ─────────────────────────────
        stage_b_input: list[dict] = [
            {
                "gen_id":               cq.cq_id,
                "archetype":            (cq.source_archetypes[0] if cq.source_archetypes else "UNKNOWN"),
                "generalized_question": cq.competency_question,
                "papers_covered":       cq.papers_covered,
            }
            for cq in stage_a_cqs
        ]

        print(f"\n[Consolidate] Stage B — {len(stage_b_input)} CQs -> cross-archetype + schema ...",
              flush=True)
        model_out, wall_b = self._stage_b_call(stage_b_input)
        total_wall += wall_b
        print(f"  Stage B: {len(stage_b_input)} -> {len(model_out.consolidated_cqs)} CQs "
              f"({fmt_duration(wall_b)})")

        # ── Collect Stage A IDs referenced by Stage B (before expansion) ──
        referenced_sa_ids: set[str] = set()
        for cq in model_out.consolidated_cqs:
            referenced_sa_ids.update(sid for sid in cq.merged_from if sid in stage_a_map)

        # ── Fallback: Stage A CQs that Stage B ignored ────────────────────
        fallback_sa_cqs = [cq for cq in stage_a_cqs if cq.cq_id not in referenced_sa_ids]

        # ── Expand merged_from: Stage A IDs -> original gen IDs ───────────
        # IDs absent from stage_a_map are already original gen IDs (passthrough).
        for cq in model_out.consolidated_cqs:
            cq.merged_from = _expand_ids(cq.merged_from, stage_a_map)
        for i, sa_cq in enumerate(fallback_sa_cqs, 1):
            sa_cq.merged_from = _expand_ids(sa_cq.merged_from, stage_a_map)
            sa_cq.cq_id = f"FB_{i:03d}"

        model_out.consolidated_cqs.extend(fallback_sa_cqs)
        if fallback_sa_cqs:
            print(f"  [Fallback] Added {len(fallback_sa_cqs)} Stage A CQs ignored by Stage B")

        return self._build_output(original, model_out, total_wall, n_stage_a_calls)

    # ------------------------------------------------------------------
    # Stage A: single-archetype intra-merge
    # ------------------------------------------------------------------

    def _stage_a_call(
        self, batch: list[dict], archetype: str
    ) -> tuple[list[ConsolidatedCQ], float]:
        abbrev = _ARCHETYPE_ABBREV.get(archetype, archetype[:6].upper())
        prompt = (
            _STAGE_A_PROMPT
            .replace("BATCH_N",        str(len(batch)))
            .replace("BATCH_ARCHETYPE", archetype)
            .replace("BATCH_ABBREV",   abbrev)
            .replace("BATCH_INPUT",    json.dumps(batch, ensure_ascii=False))
        )
        validated, wall = self._call_llm(prompt, StageAOutput, label=f"StageA/{archetype}")
        return validated.consolidated_cqs, wall

    # ------------------------------------------------------------------
    # Stage B: cross-archetype merge + schema
    # ------------------------------------------------------------------

    def _stage_b_call(
        self, stage_b_input: list[dict]
    ) -> tuple[ConsolidationOutput, float]:
        prompt = _STAGE_B_PROMPT.replace(
            "{INPUT_JSON}", json.dumps(stage_b_input, ensure_ascii=False)
        )
        validated, wall = self._call_llm(prompt, ConsolidationOutput, label="StageB")
        return validated, wall

    # ------------------------------------------------------------------
    # Shared LLM call + Pydantic validation
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str, schema_class: type, *, label: str = ""):
        import requests as _requests

        fmt = (
            schema_class.model_json_schema()
            if not self.no_schema_format else "json"
        )
        data = {
            "model":   self.model_name,
            "prompt":  prompt,
            "stream":  False,
            "think":   False,
            "format":  fmt,
            "options": {
                "temperature": self.temperature,
                "num_ctx":     self.num_ctx,
            },
        }
        t0 = time.time()
        try:
            r = _requests.post(self.llm.base_url, json=data, timeout=self.timeout)
            r.raise_for_status()
        except _requests.exceptions.RequestException as e:
            raise RuntimeError(f"[{label}] LLM call failed: {e}") from e

        raw  = r.json().get("response", "")
        wall = time.time() - t0

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            slug = label.replace("/", "_")
            Path(f"consolidate_raw_{slug}.txt").write_text(raw)
            raise RuntimeError(
                f"[{label}] Model returned non-JSON ({e}). "
                f"Saved to consolidate_raw_{slug}.txt. "
                f"First 400 chars:\n{raw[:400]}"
            )

        try:
            validated = schema_class.model_validate(parsed)
        except ValidationError as e:
            slug = label.replace("/", "_")
            Path(f"consolidate_raw_{slug}.json").write_text(json.dumps(parsed, indent=2))
            errors = "\n".join(
                f"  - {'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
                for err in e.errors()
            )
            raise RuntimeError(
                f"[{label}] Pydantic validation failed. Saved to consolidate_raw_{slug}.json.\n"
                f"Errors:\n{errors}"
            )

        return validated, wall

    # ------------------------------------------------------------------
    # Output assembly
    # ------------------------------------------------------------------

    def _build_output(
        self,
        original:        dict,
        model_out:       ConsolidationOutput,
        total_wall:      float,
        n_stage_a_calls: int,
    ) -> dict:
        ccqs    = model_out.consolidated_cqs
        src_gen = original["generalized_cqs"]
        all_ids = {g["gen_id"] for g in src_gen}

        # Lookup table: original gen_id -> gen CQ dict (for deterministic field repair)
        orig_lookup: dict = {g["gen_id"]: g for g in src_gen}

        new_gen_cqs:   list = []
        reverse_index: dict = {}

        for c in ccqs:
            # ── Deterministic field repair ─────────────────────────────
            # The LLM often leaves source_archetypes, papers_covered,
            # n_papers_covered, and variables empty even when merged_from
            # is correctly filled.  Derive them from the source gen CQs
            # so the output is always correct regardless of model behaviour.

            papers: set  = set()
            archs:  list = []
            for gid in c.merged_from:
                src = orig_lookup.get(gid)
                if src:
                    papers.update(src.get("papers_covered", []))
                    a = src.get("archetype", "")
                    if a and a not in archs:
                        archs.append(a)

            papers_covered   = sorted(papers)
            n_papers_covered = len(papers_covered)
            source_archetypes = archs if archs else c.source_archetypes
            archetype = source_archetypes[0] if len(source_archetypes) == 1 else "MULTI"

            # Extract SPARQL-style variable names from the question text
            variables = re.findall(r"\?([a-zA-Z]\w*)", c.competency_question)

            # ── Assemble output record ─────────────────────────────────
            new_gen_cqs.append({
                "gen_id":                  c.cq_id,
                "archetype":               archetype,
                "source_archetypes":       source_archetypes,
                "generalized_question":    c.competency_question,
                "variables":               variables,
                "triple_pattern":          [t.model_dump(by_alias=True) for t in c.triple_pattern],
                "sparql_sketch":           c.sparql_sketch or "",
                "is_singleton_group":      n_papers_covered <= 1,
                "coverage_breadth":        c.coverage_breadth,
                "n_papers_covered":        n_papers_covered,
                "papers_covered":          papers_covered,
                "consolidation_reasoning": c.consolidation_reasoning or "",
                "merged_from":             c.merged_from,
            })
            for old_id in c.merged_from:
                reverse_index[old_id] = c.cq_id

        breadth = Counter(g["coverage_breadth"] for g in new_gen_cqs)

        out = dict(original)
        out["pipeline_step"] = "cq_consolidation_schema"
        out["step_version"]  = self.STEP_VERSION
        out["generalize_config"] = {
            **original.get("generalize_config", {}),
            "consolidation_model": self.model_name,
            "prompt_version":      self.PROMPT_VERSION,
            "output_validation":   "pydantic-v2",
            "fed_field":           "generalized_question_only",
            "pipeline_mode":       "two_stage_batched",
            "n_stage_a_batches":   n_stage_a_calls,
        }
        out["run_stats"] = {
            **original.get("run_stats", {}),
            "n_input_generalized_cqs": len(src_gen),
            "n_consolidated_cqs":      len(new_gen_cqs),
            "reduction_rate": round(
                1 - len(new_gen_cqs) / len(src_gen), 3
            ) if src_gen else 0,
            "coverage_breadth_counts": dict(breadth),
        }
        out["execution_stats"] = {
            "total_wall_time_s": round(total_wall, 1),
            "n_llm_calls":       n_stage_a_calls + 1,
            "stage_a_calls":     n_stage_a_calls,
            "stage_b_calls":     1,
            "model":             self.model_name,
        }
        out["generalized_cqs"] = new_gen_cqs
        out["reverse_index"]   = reverse_index

        # ── Semantic audit ─────────────────────────────────────────────
        accounted = set(reverse_index)
        missing   = sorted(all_ids - accounted)
        extra     = sorted(accounted - all_ids)
        qtexts    = Counter(
            re.sub(r"\s+", " ", g["generalized_question"].strip().lower())
            for g in new_gen_cqs
        )
        dup_q = [q for q, n in qtexts.items() if n > 1]

        out["_audit"] = {
            "unaccounted_input_gen_ids":        missing,
            "hallucinated_gen_ids":             extra,
            "duplicate_consolidated_questions": dup_q,
        }

        print(f"\n  [Consolidate] {len(src_gen)} -> {len(new_gen_cqs)} CQs "
              f"({out['run_stats']['reduction_rate'] * 100:.0f}% reduction)")
        print(f"  [Consolidate] Coverage: {dict(breadth)}")
        print(f"  [Consolidate] LLM calls: {n_stage_a_calls} Stage A + 1 Stage B = {n_stage_a_calls + 1} total")

        flags = []
        if missing: flags.append(f"{len(missing)} unaccounted gen_ids")
        if extra:   flags.append(f"{len(extra)} hallucinated gen_ids")
        if dup_q:   flags.append(f"{len(dup_q)} duplicate questions")
        print("  [Consolidate] Audit: " + ("OK" if not flags else "; ".join(flags)))

        return out

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def list_available_models(self) -> list:
        import requests as _requests
        tags_url = self.llm.base_url.replace("/api/generate", "/api/tags")
        r = _requests.get(tags_url, timeout=60)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]

    @staticmethod
    def json_schema() -> dict:
        return ConsolidationOutput.model_json_schema()


# =========================================================================
# Module-level helpers
# =========================================================================

def _expand_ids(id_list: list[str], stage_a_map: dict[str, list[str]]) -> list[str]:
    """Expand Stage A CQ IDs back to original gen IDs using the provenance map.
    IDs not in the map are already original gen IDs (passthrough) — kept as-is."""
    expanded = []
    for sa_id in id_list:
        expanded.extend(stage_a_map.get(sa_id, [sa_id]))
    return expanded
