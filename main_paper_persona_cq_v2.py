"""Entry point for Phase A: per-paper persona CQ generation pipeline (v2.1)."""
# ── original content below ──────────────────────────────────────────────────
from __future__ import annotations

import json
import time
from pathlib import Path
from tqdm import tqdm

from utils.pdf_extractor          import extract_text
from utils.rich_pdf_extractor     import extract_paper_stats
from utils.model_config           import get_model, model_slug
from utils.llm_client             import CustomOllamaClient, LLMNetworkError
from agents.paper_metadata_extractor import PaperMetadataExtractor
from workflows.paper_persona_cq_v2_flow import paper_persona_cq_v2_app

PAPERS_DIR    = Path("data/papers")
RAW_DIR       = PAPERS_DIR / "raw"
TEXT_DIR      = PAPERS_DIR / "text"
METADATA_DIR  = PAPERS_DIR / "metadata"

V2_DIR             = PAPERS_DIR / "v2"
V2_CHUNKS_DIR      = V2_DIR / "chunks"
V2_INDEX_DIR       = V2_DIR / "index"
V2_STATE_DIR       = V2_DIR / "state"
V2_RETRIEVED_DIR   = V2_DIR / "retrieved"
V2_CQ_DIR          = V2_DIR / "cq_output"
V2_VAL_DIR         = V2_DIR / "validation_logs"
V2_EVAL_DIR        = V2_DIR / "evaluation"
V2_RUNLOG_DIR      = V2_DIR / "run_logs"

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}
_PIPELINE = "paper_persona_v2_pipeline"


def _v2_slug() -> str:
    ext = model_slug(get_model(_PIPELINE, "iterative_extractor"))
    g   = model_slug(get_model(_PIPELINE, "cq_generator"))
    v   = model_slug(get_model(_PIPELINE, "cq_validator"))
    r   = model_slug(get_model(_PIPELINE, "cq_refiner"))
    return f"ext-{ext}_g-{g}_v-{v}_r-{r}"


metadata_extractor = PaperMetadataExtractor(
    model_name=get_model(_PIPELINE, "metadata_extractor"),
)


def _check_server_connectivity() -> bool:
    required_models = {
        "metadata_extractor":  get_model(_PIPELINE, "metadata_extractor"),
        "iterative_extractor": get_model(_PIPELINE, "iterative_extractor"),
        "cq_generator":        get_model(_PIPELINE, "cq_generator"),
        "cq_validator":        get_model(_PIPELINE, "cq_validator"),
        "cq_refiner":          get_model(_PIPELINE, "cq_refiner"),
    }
    probe    = CustomOllamaClient(model=next(iter(required_models.values())))
    tags_url = probe.base_url.replace("/api/generate", "/api/tags")
    print("  [Preflight] Checking Ollama server connectivity...", end=" ", flush=True)
    try:
        import requests as _req
        resp = _req.get(tags_url, timeout=8)
        resp.raise_for_status()
    except Exception as e:
        print("FAILED")
        print(f"  [ERROR] Cannot reach the Ollama server ({e}).")
        return False
    print("OK")
    available = {m["name"] for m in resp.json().get("models", [])}
    missing   = {role: name for role, name in required_models.items()
                 if name not in available}
    if missing:
        print("  [WARNING] Missing models:")
        for role, name in missing.items():
            print(f"    {role:25s} → {name}  ✗")
        return False
    print("  [Preflight] All required models confirmed.")
    return True


def _ensure_dirs() -> None:
    for d in (TEXT_DIR, METADATA_DIR,
              V2_CHUNKS_DIR, V2_INDEX_DIR, V2_STATE_DIR,
              V2_RETRIEVED_DIR, V2_CQ_DIR, V2_VAL_DIR, V2_EVAL_DIR, V2_RUNLOG_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _resolve_paper_id(paper_id: str) -> str:
    if not (METADATA_DIR / f"{paper_id}.json").exists():
        return paper_id
    for suffix in "abcdefghij":
        candidate = f"{paper_id}{suffix}"
        if not (METADATA_DIR / f"{candidate}.json").exists():
            return candidate
    return paper_id


def _load_or_extract(raw_path: Path):
    for meta_file in METADATA_DIR.glob("*.json"):
        try:
            m = json.loads(meta_file.read_text(encoding="utf-8"))
            if m.get("source_filename") == raw_path.name:
                paper_id   = m["paper_id"]
                paper_text = (TEXT_DIR / f"{paper_id}.txt").read_text(encoding="utf-8")
                print(f"  [Cache] Reusing existing text + metadata -> paper_id: {paper_id}")
                return paper_id, paper_text, m
        except Exception:
            continue
    paper_text = extract_text(raw_path)
    if not paper_text.strip():
        return None, None, None
    metadata = metadata_extractor.extract(paper_text, source_filename=raw_path.name)
    paper_id = _resolve_paper_id(metadata["paper_id"])
    metadata["paper_id"] = paper_id
    (TEXT_DIR / f"{paper_id}.txt").write_text(paper_text, encoding="utf-8")
    (METADATA_DIR / f"{paper_id}.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    return paper_id, paper_text, metadata


def _build_paper_stats(raw_path: Path, paper_text: str) -> dict:
    stats = extract_paper_stats(raw_path)
    stats["char_count"]       = len(paper_text)
    stats["estimated_tokens"] = len(paper_text) // 4
    stats["n_sections"]       = sum(1 for ln in paper_text.splitlines() if ln.startswith("## "))
    return stats


def _build_pipeline_config() -> dict:
    return {
        "pipeline_version":    "v2.1",
        "extraction_method":   "plain",
        "metadata_extractor":  get_model(_PIPELINE, "metadata_extractor"),
        "iterative_extractor": get_model(_PIPELINE, "iterative_extractor"),
        "cq_generator":        get_model(_PIPELINE, "cq_generator"),
        "cq_validator":        get_model(_PIPELINE, "cq_validator"),
        "cq_refiner":          get_model(_PIPELINE, "cq_refiner"),
        "embedding_model":     "BAAI/bge-small-en-v1.5",
    }


def _cached_paper_id(raw_path: Path) -> str | None:
    for meta_file in METADATA_DIR.glob("*.json"):
        try:
            m = json.loads(meta_file.read_text(encoding="utf-8"))
            if m.get("source_filename") == raw_path.name:
                return m["paper_id"]
        except Exception:
            continue
    return None


def process_paper(raw_path: Path, target_paper_id: str | None = None) -> None:
    if target_paper_id:
        cached_id = _cached_paper_id(raw_path)
        if cached_id is not None and cached_id != target_paper_id:
            return
        if cached_id is None:
            stem = raw_path.stem
            if not (target_paper_id in stem or stem in target_paper_id):
                return
    print(f"\n[Paper] {raw_path.name}")
    t_process_start = time.time()
    t0 = time.time()
    paper_id, paper_text, metadata = _load_or_extract(raw_path)
    text_extraction_secs = round(time.time() - t0, 2)
    if not paper_id:
        print(f"  [Skip] Could not extract text from {raw_path.name}")
        return
    if target_paper_id and paper_id != target_paper_id:
        print(f"  [Skip] paper_id={paper_id} != target={target_paper_id}")
        return
    slug   = _v2_slug()
    cq_out = V2_CQ_DIR / f"{paper_id}_{slug}_cq_v2.json"
    if cq_out.exists():
        print(f"  [Skip] {paper_id} already processed by v2 ({slug}).")
        return
    paper_stats     = _build_paper_stats(raw_path, paper_text)
    pipeline_config = _build_pipeline_config()
    t_start = time.time()
    initial_state = {
        "paper_id":       paper_id,
        "paper_text":     paper_text,
        "paper_metadata": metadata,
        "index_dir":      str(V2_INDEX_DIR),
        "retry_count":    0,
        "run_log":        {"stage_seconds": {}, "started_at": t_start},
    }
    final_state = paper_persona_cq_v2_app.invoke(initial_state)
    run_log         = final_state.get("run_log", {})
    total_wall_secs = round(time.time() - t_process_start, 2)
    execution_stats = {
        "text_extraction_secs": text_extraction_secs,
        "total_wall_secs":      total_wall_secs,
        "stage_seconds":        run_log.get("stage_seconds", {}),
    }
    final_questions   = final_state.get("final_questions", [])
    persona_questions = final_state.get("persona_questions", {})
    n_personas        = len({q.get("persona_code") for q in final_questions})
    if final_questions:
        n_parse_errors   = sum(1 for q in final_questions if q.get("archetype") == "PARSE_ERROR")
        parse_error_rate = n_parse_errors / len(final_questions)
        if parse_error_rate > 0.5:
            print(f"\n  [CORRUPT OUTPUT DETECTED] {paper_id}: "
                  f"{n_parse_errors}/{len(final_questions)} PARSE_ERROR. No output written.")
            return
    cq_out.write_text(
        json.dumps({
            "paper_id":          paper_id,
            "paper_metadata":    metadata,
            "paper_stats":       paper_stats,
            "pipeline_config":   pipeline_config,
            "execution_stats":   execution_stats,
            "pipeline_version":  "v2.1",
            "n_personas":        n_personas,
            "n_questions":       len(final_questions),
            "was_refined":       final_state.get("retry_count", 0) > 0,
            "persona_questions": persona_questions,
            "final_questions":   final_questions,
            "validation":        final_state.get("cq_validation", {}),
            "refined_questions": final_state.get("refined_questions", {}),
            "evaluation":        final_state.get("evaluation", {}),
        }, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"  [Done] {len(final_questions)} questions ({n_personas} personas) "
          f"in {total_wall_secs}s -> {cq_out}")


def main(only_paper_id: str | None = None) -> None:
    print("Starting Paper Persona CQ Generation Pipeline (v2.1)...")
    _ensure_dirs()
    if not _check_server_connectivity():
        return
    raw_files = sorted(
        f for f in RAW_DIR.iterdir()
        if f.suffix.lower() in SUPPORTED_SUFFIXES and not f.name.startswith(".")
    )
    if not raw_files:
        print(f"No papers found in {RAW_DIR}.")
        return
    print(f"  Found {len(raw_files)} paper(s) in {RAW_DIR}\n")
    for raw_path in tqdm(raw_files):
        try:
            process_paper(raw_path, target_paper_id=only_paper_id)
        except LLMNetworkError as e:
            print(f"\n  [NETWORK FAILURE] {raw_path.name}: {e}\n  Stopping batch.")
            break
        except Exception as e:
            print(f"  [Error] {raw_path.name}: {e}")
            import traceback; traceback.print_exc()
    print("\nPaper Persona CQ Generation v2.1 Complete.")


if __name__ == "__main__":
    import sys
    only = sys.argv[1] if len(sys.argv) > 1 else None
    main(only_paper_id=only)
