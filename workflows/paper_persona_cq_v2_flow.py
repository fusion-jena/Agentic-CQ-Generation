"""
LangGraph workflow for the v2.1 (chunked + iterative-extraction + RAG +
refiner) persona CQ pipeline.

Graph:
    START
      v
    chunk                          Stage 1a
      v
    index                          Stage 1b
      v
    extract                        Stage 2  (rolling state on disk)
      v
    retrieve                       Stage 3  (evidence passages on disk)
      v
    generate_cqs                   Stage 4  (one LLM call per persona)
      v
    coverage_gap_pass              Stage 4b (extra round per persona, only if gaps exist)
      v
    validate                       Stage 5  (per-question LLM judge + rule checks)
      v
   [router]
      |--> refine ---------------- Stage 5b (NEW - only failing questions)
      |     v
      |   revalidate               re-judge ONLY the refined questions
      |     v
      |   merge_refined            replace failing originals with refined ones
      |     v
      v
    evaluate                       Stage 6  (5-family reference-free eval)
      v
    END
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, TypedDict

from langgraph.graph import StateGraph, START, END

from agents.paper_chunker             import chunk_paper
from agents.paper_indexer             import PaperIndexer
from agents.paper_iterative_extractor import PaperIterativeExtractor
from agents.paper_retriever           import PaperRetriever
from agents.paper_persona_cq_generator_v2 import PaperPersonaCQGeneratorV2
from agents.paper_persona_cq_validator_v2 import PaperPersonaCQValidatorV2
from agents.paper_persona_cq_refiner_v2   import PaperPersonaCQRefinerV2
from utils.cq_evaluator import evaluate as evaluate_cqs, gap_concepts
from utils.embeddings   import Embedder
from utils.model_config import get_model
from utils.vectorstore  import PaperVectorStore

_CONFIG_DIR  = Path(__file__).parent.parent / "config"
_CHECKS_PATH = _CONFIG_DIR / "paper_persona_cq_v2_checks.json"
_MODELS_PATH = _CONFIG_DIR / "models.json"

_PIPELINE = "paper_persona_v2_pipeline"


class PaperPersonaCQV2State(TypedDict, total=False):
    paper_id:           str
    paper_text:         str
    paper_metadata:     Dict[str, Any]

    chunks:             List[str]
    index_dir:          str

    rolling_state:      Dict[str, list]
    retrieved_passages: Dict[str, List[str]]

    persona_questions:  Dict[str, list]
    final_questions:    List[dict]

    cq_validation:      dict
    refined_questions:  Dict[str, list]   # {persona_code: [refined_q, ...]}
    retry_count:        int

    evaluation:         dict
    run_log:            dict


# -------------------------------------------------------------------------
# Shared singletons (loaded once per process)
# -------------------------------------------------------------------------

_checks         = json.loads(_CHECKS_PATH.read_text(encoding="utf-8"))
_models_cfg     = json.loads(_MODELS_PATH.read_text(encoding="utf-8"))
_pipeline_models = _models_cfg.get(_PIPELINE, {})

_chunking  = _checks.get("chunking", {})
_retrieval = _checks.get("retrieval", {})
_caps      = _checks.get("rolling_state_caps", {})
_gap_cfg   = _checks.get("coverage_gap_pass", {})

_embedder  = Embedder(
    model_name=_pipeline_models.get("_embedding_model", "BAAI/bge-small-en-v1.5"),
)
_indexer   = PaperIndexer(embedder=_embedder)
_extractor = PaperIterativeExtractor(
    model_name=get_model(_PIPELINE, "iterative_extractor"),
    state_caps=_caps,
)
_generator = PaperPersonaCQGeneratorV2(model_name=get_model(_PIPELINE, "cq_generator"))
_validator = PaperPersonaCQValidatorV2(
    model_name=get_model(_PIPELINE, "cq_validator"),
    embedder=_embedder,
)
_refiner   = PaperPersonaCQRefinerV2(
    model_name=get_model(_PIPELINE, "cq_refiner"),
    embedder=_embedder,
)


# -------------------------------------------------------------------------
# Per-paper run log (latency, call counts)
# -------------------------------------------------------------------------

def _log_step(state: PaperPersonaCQV2State, name: str, t0: float) -> dict:
    rl = state.get("run_log") or {"stage_seconds": {}, "started_at": time.time()}
    rl.setdefault("stage_seconds", {})[name] = round(time.time() - t0, 2)
    return rl


# -------------------------------------------------------------------------
# Stage nodes
# -------------------------------------------------------------------------

def chunk_step(state: PaperPersonaCQV2State):
    t0 = time.time()
    print(f"  [Stage 1a] Chunking {state['paper_id']}...")
    chunks = chunk_paper(
        state["paper_text"],
        chunk_chars=_chunking.get("chunk_chars", 3500),
        chunk_overlap=_chunking.get("chunk_overlap", 200),
    )
    print(f"    -> {len(chunks)} chunks")
    return {"chunks": chunks, "run_log": _log_step(state, "chunk", t0)}


def index_step(state: PaperPersonaCQV2State):
    t0 = time.time()
    print(f"  [Stage 1b] Indexing chunks for {state['paper_id']}...")
    index_dir = Path(state["index_dir"])
    index_dir.mkdir(parents=True, exist_ok=True)
    store = _indexer.build_or_load(state["paper_id"], state["chunks"], index_dir)
    print(f"    -> index ready ({store.n_chunks} chunks)")
    return {"run_log": _log_step(state, "index", t0)}


def extract_step(state: PaperPersonaCQV2State):
    t0 = time.time()
    print(f"  [Stage 2] Iterative extraction over {len(state['chunks'])} chunks...")
    final_state = _extractor.run(state["chunks"], state["paper_metadata"])
    return {"rolling_state": final_state, "run_log": _log_step(state, "extract", t0)}


def retrieve_step(state: PaperPersonaCQV2State):
    t0 = time.time()
    print(f"  [Stage 3] Retrieving evidence passages...")
    index_dir = Path(state["index_dir"])
    store = PaperVectorStore(_embedder)
    store.load(index_dir / f"{state['paper_id']}.faiss",
               index_dir / f"{state['paper_id']}_meta.json")
    retriever = PaperRetriever(
        store,
        top_k_concepts=_retrieval.get("top_k_concepts", 20),
        passages_per_concept=_retrieval.get("passages_per_concept", 2),
    )
    passages = retriever.retrieve(state["rolling_state"])
    return {"retrieved_passages": passages, "run_log": _log_step(state, "retrieve", t0)}


def generate_step(state: PaperPersonaCQV2State):
    t0 = time.time()
    print(f"  [Stage 4] Generating persona CQs (no full paper text in prompt)...")
    index_dir = Path(state["index_dir"])
    store = PaperVectorStore(_embedder)
    store.load(index_dir / f"{state['paper_id']}.faiss",
               index_dir / f"{state['paper_id']}_meta.json")
    retriever = PaperRetriever(
        store,
        top_k_concepts=_retrieval.get("top_k_concepts", 20),
        passages_per_concept=_retrieval.get("passages_per_concept", 2),
    )
    by_persona, persona_passages = _generator.generate_all(
        state["paper_id"], state["paper_metadata"], state["rolling_state"], retriever,
    )

    # Coverage-gap pass: identify untouched concepts and run a 2nd round per persona
    if _gap_cfg.get("enabled", True):
        flat_so_far: List[dict] = []
        for qs in by_persona.values():
            flat_so_far.extend(qs)
        gaps = gap_concepts(
            flat_so_far, state["rolling_state"],
            top_k=_gap_cfg.get("max_gap_concepts", 8),
        )
        if gaps:
            print(f"  [Stage 4b] Coverage-gap pass for {len(gaps)} untouched concept(s)...")
            extra, gap_passages = _generator.generate_all(
                state["paper_id"], state["paper_metadata"], state["rolling_state"], retriever,
                gap_concepts=gaps,
            )
            # Merge extras (the generator already prefixed GAP ids) - drop the
            # first-round questions extra() returns and only keep the gap-specific
            # additions, which were appended with `_GAP###` suffixes.
            for code, qs in extra.items():
                gap_only = [q for q in qs if "_GAP" in (q.get("id") or "")]
                by_persona.setdefault(code, []).extend(gap_only)
            persona_passages.update(gap_passages)

    # Merge the persona-biased passages back into state.retrieved_passages.
    # Stage 3 populated it with concept-biased passages; now we union both,
    # so the validator + evaluator see the same evidence the generator used.
    merged_passages = dict(state.get("retrieved_passages") or {})
    for k, ps in persona_passages.items():
        merged_passages.setdefault(k, []).extend(ps)

    flat: List[dict] = []
    for qs in by_persona.values():
        flat.extend(qs)
    return {
        "persona_questions":  by_persona,
        "final_questions":    flat,
        "retrieved_passages": merged_passages,
        "run_log":            _log_step(state, "generate", t0),
    }


def validate_step(state: PaperPersonaCQV2State):
    t0 = time.time()
    print(f"  [Stage 5] Per-question validation (rule checks + LLM judge)...")
    result = _validator.validate(
        state["final_questions"],
        state["rolling_state"],
        state.get("retrieved_passages", {}),
    )
    pq = result.get("per_question", {})
    print(f"    -> n_pass={result.get('n_pass')} n_borderline={result.get('n_borderline')} "
          f"n_fail={result.get('n_fail')} (threshold={result.get('pass_score_threshold')})")
    return {"cq_validation": result, "run_log": _log_step(state, "validate", t0)}


def refine_step(state: PaperPersonaCQV2State):
    t0 = time.time()
    print(f"  [Stage 5b] Refiner working on failing/borderline questions...")
    val = state.get("cq_validation", {})
    failing_ids = set(val.get("questions_to_refine", []))
    if not failing_ids:
        print(f"    -> nothing to refine; skipping")
        return {"refined_questions": {}, "retry_count": state.get("retry_count", 0) + 1,
                "run_log": _log_step(state, "refine", t0)}

    per_question = val.get("per_question", {})
    # Flatten retrieved passages once
    passages_flat: List[str] = []
    for ps in (state.get("retrieved_passages") or {}).values():
        for p in ps:
            if p not in passages_flat: passages_flat.append(p)

    refined_by_persona: Dict[str, List[dict]] = {}
    for code, qs in (state.get("persona_questions") or {}).items():
        failing_qs = [q for q in qs if q.get("id") in failing_ids]
        if not failing_qs:
            continue
        refined = _refiner.refine_batch(code, failing_qs, passages_flat, per_question)
        refined_by_persona[code] = refined

    n_refined = sum(len(v) for v in refined_by_persona.values())
    print(f"    -> refined {n_refined} question(s) across {len(refined_by_persona)} persona(s)")
    return {
        "refined_questions": refined_by_persona,
        "retry_count":       state.get("retry_count", 0) + 1,
        "run_log":           _log_step(state, "refine", t0),
    }


def merge_refined_step(state: PaperPersonaCQV2State):
    """Replace failing originals with the refined versions (where refiner chose to revise)."""
    refined = state.get("refined_questions") or {}
    by_persona = dict(state.get("persona_questions") or {})

    n_revised = n_kept = n_dropped = 0
    for code, refined_list in refined.items():
        by_id_new = {r["id"]: r for r in refined_list}
        new_list = []
        for q in by_persona.get(code, []):
            qid = q.get("id", "")
            replacement = by_id_new.get(qid)
            if replacement is None:
                new_list.append(q); continue
            action = replacement.get("refiner_action", "keep")
            if action == "revise":
                new_list.append(replacement); n_revised += 1
            elif action == "keep":
                new_list.append(replacement); n_kept += 1
            elif action == "unrepairable":
                # Tag but do NOT drop - keep the question with a flag so reviewers can see it
                replacement["unrepairable"] = True
                new_list.append(replacement); n_dropped += 1
            else:
                new_list.append(q)
        by_persona[code] = new_list

    flat: List[dict] = []
    for qs in by_persona.values():
        flat.extend(qs)

    print(f"  [Stage 5c] Merge: revised={n_revised} kept={n_kept} unrepairable={n_dropped}")
    return {"persona_questions": by_persona, "final_questions": flat}


def evaluate_step(state: PaperPersonaCQV2State):
    t0 = time.time()
    print(f"  [Stage 6] Reference-free evaluation (5 families)...")
    repro = {
        "pipeline_version":  "v2.1",
        "models":            _pipeline_models,
        "validator_pass_threshold_per_question": _checks.get("thresholds", {}).get("pass_score_per_question"),
        "coverage_gap_pass": _gap_cfg,
        "rolling_state_caps": _caps,
        "embedder_dim":      _embedder.dim,
    }
    result = evaluate_cqs(
        paper_id=state["paper_id"],
        generated_questions=state.get("final_questions", []),
        rolling_state=state.get("rolling_state"),
        retrieved_passages=state.get("retrieved_passages"),
        embedder=_embedder,
        reproducibility=repro,
    )
    overall = result.get("overall_score")
    print(f"    -> overall_score={overall}")

    rl = _log_step(state, "evaluate", t0)
    rl["finished_at"] = time.time()
    rl["total_seconds"] = round(rl["finished_at"] - rl.get("started_at", t0), 2)
    return {"evaluation": result, "run_log": rl}


# -------------------------------------------------------------------------
# Router: refine if anyone failed/borderline
# -------------------------------------------------------------------------

def router(state: PaperPersonaCQV2State):
    val = state.get("cq_validation", {})
    n_to_refine = len(val.get("questions_to_refine", []))
    retries = state.get("retry_count", 0)
    if n_to_refine > 0 and retries < 1:
        return "refine"
    return "finalize"


# -------------------------------------------------------------------------
# Build graph
# -------------------------------------------------------------------------

_workflow = StateGraph(PaperPersonaCQV2State)
_workflow.add_node("chunk",         chunk_step)
_workflow.add_node("index",         index_step)
_workflow.add_node("extract",       extract_step)
_workflow.add_node("retrieve",      retrieve_step)
_workflow.add_node("generate",      generate_step)
_workflow.add_node("validate",      validate_step)
_workflow.add_node("refine",        refine_step)
_workflow.add_node("merge_refined", merge_refined_step)
_workflow.add_node("evaluate",      evaluate_step)

_workflow.add_edge(START,         "chunk")
_workflow.add_edge("chunk",       "index")
_workflow.add_edge("index",       "extract")
_workflow.add_edge("extract",     "retrieve")
_workflow.add_edge("retrieve",    "generate")
_workflow.add_edge("generate",    "validate")
_workflow.add_conditional_edges(
    "validate", router,
    {"refine": "refine", "finalize": "evaluate"},
)
_workflow.add_edge("refine",        "merge_refined")
_workflow.add_edge("merge_refined", "evaluate")
_workflow.add_edge("evaluate",      END)

paper_persona_cq_v2_app = _workflow.compile()
