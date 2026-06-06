"""
Workflow for Step 1: intra-paper CQ deduplication.

Deliberately kept as plain Python functions (no LangGraph) because
Step 1 has no conditional branching — it is a one-shot operation:
    load source file → run dedup agent → save output.

LangGraph would add boilerplate without benefit here. The workflow
pattern matches the rest of the codebase by using a single orchestration
function (run_dedup_pipeline) that callers import.

Output directory: data/papers/v2/dedup/
Output filename:  {paper_id}_dedup.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional

from agents.cq_dedup_agent import CQDedupAgent
from utils.llm_client import fmt_duration
from utils.model_config import get_model

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT       = Path(__file__).parent.parent
_V2_CQ_DIR  = _ROOT / "data" / "papers" / "v2" / "cq_output"
_DEDUP_DIR  = _ROOT / "data" / "papers" / "v2" / "dedup"


# ── Single-paper runner ────────────────────────────────────────────────────────

def run_dedup_for_paper(
    source_path: Path,
    agent:       CQDedupAgent,
    output_dir:  Path,
    overwrite:   bool = False,
) -> Optional[Path]:
    """
    Run dedup for a single v2 CQ file and save the result.

    Args:
        source_path: path to the *_cq_v2.json source file.
        agent:       initialised CQDedupAgent.
        output_dir:  directory where the output JSON is written.
        overwrite:   if False, skip files that already have a dedup output.

    Returns:
        Path to the written output file, or None if skipped.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Derive a clean paper_id from the filename
    # Handles both "Jacobs2007_cq_v2.json" and
    # "Jacobs2007_ext-qw-35b_g-ge4-31b_..._cq_v2.json"
    stem     = source_path.stem                   # strip .json
    paper_id = stem.split("_cq_v2")[0].split("_ext-")[0]
    out_path = output_dir / f"{paper_id}_dedup.json"

    if out_path.exists() and not overwrite:
        print(f"  [Dedup] Skipping {paper_id} — output exists (use --overwrite to rerun)")
        return None

    print(f"\n[Dedup] ─── {source_path.name} ───")
    t_paper = time.time()

    try:
        result = agent.deduplicate(source_path)
    except Exception as exc:
        print(f"  [Dedup] ERROR: {exc}")
        raise

    out_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    stats  = result.get("dedup_stats", {})
    n_in   = stats.get("n_input_cqs", "?")
    n_out  = stats.get("n_representative_cqs", "?")
    rate   = stats.get("reduction_rate", 0)
    elapsed = time.time() - t_paper

    print(
        f"  [Dedup] {n_in} → {n_out} CQs  "
        f"({rate * 100:.0f}% reduction)  "
        f"total: {fmt_duration(elapsed)}"
    )
    print(f"  [Dedup] Saved → {out_path.relative_to(_ROOT)}")

    return out_path


# ── Pipeline orchestrator ──────────────────────────────────────────────────────

def run_dedup_pipeline(
    paper_id:  Optional[str] = None,
    overwrite: bool = False,
) -> List[Path]:
    """
    Run the intra-paper dedup pipeline over one or all v2 CQ files.

    Args:
        paper_id:  if given, only the file matching this paper ID is processed.
                   if None, every *.json file in v2/cq_output/ is processed.
        overwrite: if True, re-run and overwrite existing dedup outputs.

    Returns:
        List of paths to successfully written output files.
    """
    model_name = get_model("cq_dedup_pipeline", "dedup_agent", default="deepseek-r1:32b")
    agent      = CQDedupAgent(model_name=model_name, think=True)

    print(f"[Dedup] Model: {model_name}  |  think=True")
    print(f"[Dedup] Source dir: {_V2_CQ_DIR.relative_to(_ROOT)}")
    print(f"[Dedup] Output dir: {_DEDUP_DIR.relative_to(_ROOT)}")

    # ── Resolve source files ───────────────────────────────────────────
    if paper_id:
        candidates = sorted(_V2_CQ_DIR.glob(f"{paper_id}*.json"))
        if not candidates:
            raise FileNotFoundError(
                f"No v2 CQ file found for paper_id='{paper_id}' in {_V2_CQ_DIR}.\n"
                f"Available files: {[f.name for f in _V2_CQ_DIR.glob('*.json')]}"
            )
        # Prefer the plain "*_cq_v2.json" over model-tagged variants if both exist
        plain = [c for c in candidates if "_ext-" not in c.name]
        source_files = plain[:1] if plain else candidates[:1]
    else:
        source_files = sorted(_V2_CQ_DIR.glob("*.json"))

    print(f"[Dedup] Processing {len(source_files)} file(s)\n")

    output_paths: List[Path] = []
    t_total = time.time()

    for source_path in source_files:
        try:
            out = run_dedup_for_paper(source_path, agent, _DEDUP_DIR, overwrite=overwrite)
            if out is not None:
                output_paths.append(out)
        except Exception as exc:
            print(f"  [Dedup] SKIPPING {source_path.name} after error: {exc}\n")

    print(
        f"\n[Dedup] ══ Pipeline complete ══  "
        f"{len(output_paths)} file(s) written  |  "
        f"total: {fmt_duration(time.time() - t_total)}"
    )

    return output_paths
