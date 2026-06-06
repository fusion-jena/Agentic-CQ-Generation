"""
Workflow for Step 2: cross-paper CQ generalisation.

Reads all Step 1 dedup files, groups by archetype, runs one LLM call per
archetype, and writes two sets of output files:

  Per-archetype (for debugging / partial re-runs):
    data/papers/v2/generalized/by_archetype/GEN_{ARCHETYPE}.json

  Combined (used by Step 3):
    data/papers/v2/generalized/cross_paper_gen_v2.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional

from agents.cq_generalizer_agent import CQGeneralizerAgent
from utils.llm_client import fmt_duration
from utils.model_config import get_model

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT          = Path(__file__).parent.parent
_DEDUP_DIR     = _ROOT / "data" / "papers" / "v2" / "dedup"
_GEN_DIR       = _ROOT / "data" / "papers" / "v2" / "generalized"
_GEN_ARCH_DIR  = _GEN_DIR / "by_archetype"
_GEN_COMBINED  = _GEN_DIR / "cross_paper_gen_v2.json"


# ── Per-archetype saver ────────────────────────────────────────────────────────

def _save_per_archetype(output: dict, arch_dir: Path) -> None:
    """
    Split the combined output into per-archetype files under by_archetype/.
    Each file contains only the GenCQs for that archetype + shared metadata.
    """
    arch_dir.mkdir(parents=True, exist_ok=True)

    # Group gen CQs by archetype
    by_arch: dict[str, list] = {}
    for g in output.get("generalized_cqs", []):
        arch = g["archetype"]
        by_arch.setdefault(arch, []).append(g)

    # Build reverse index per archetype
    rev_full = output.get("reverse_index", {})

    for arch, gen_cqs in sorted(by_arch.items()):
        # Only include reverse_index entries relevant to this archetype
        arch_step1_ids = {oq["step1_id"] for g in gen_cqs for oq in g["original_cqs"]}
        arch_rev = {sid: gid for sid, gid in rev_full.items() if sid in arch_step1_ids}

        arch_output = {
            "archetype":        arch,
            "pipeline_step":    output["pipeline_step"],
            "pipeline_version": output["pipeline_version"],
            "step_version":     output["step_version"],
            "generalize_config": output["generalize_config"],
            "arch_stats": output["run_stats"]["archetype_breakdown"].get(arch, {}),
            "generalized_cqs":  gen_cqs,
            "reverse_index":    arch_rev,
        }

        out_path = arch_dir / f"GEN_{arch}.json"
        out_path.write_text(
            json.dumps(arch_output, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"    [Gen] Archetype file → {out_path.relative_to(_ROOT)}")


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_generalize_pipeline(
    dedup_dir: Optional[Path] = None,
    overwrite: bool = False,
) -> Path:
    """
    Run the cross-paper generalisation pipeline.

    Args:
        dedup_dir: path to Step 1 dedup outputs (defaults to v2/dedup/).
        overwrite: if False and combined output already exists, skip.

    Returns:
        Path to the combined output file.
    """
    dedup_dir = dedup_dir or _DEDUP_DIR

    # ── Check source ───────────────────────────────────────────────────
    dedup_files = sorted(dedup_dir.glob("*_dedup.json"))
    if not dedup_files:
        raise FileNotFoundError(
            f"No Step 1 dedup files found in {dedup_dir}.\n"
            "Run Step 1 first: python main_cq_dedup.py"
        )

    # ── Check if already done ──────────────────────────────────────────
    if _GEN_COMBINED.exists() and not overwrite:
        print(f"[Gen] Combined output already exists — skipping "
              f"(use --overwrite to rerun):\n  {_GEN_COMBINED.relative_to(_ROOT)}")
        return _GEN_COMBINED

    # ── Initialise agent ───────────────────────────────────────────────
    model_name = get_model("cq_generalize_pipeline", "generalizer_agent",
                           default="deepseek-r1:32b")
    agent = CQGeneralizerAgent(model_name=model_name, think=True)

    print(f"[Gen] Model       : {model_name}  |  think=True")
    print(f"[Gen] Source dir  : {dedup_dir.relative_to(_ROOT)}")
    print(f"[Gen] Output dir  : {_GEN_DIR.relative_to(_ROOT)}")
    print(f"[Gen] Source files: {len(dedup_files)}\n")

    # ── Run ────────────────────────────────────────────────────────────
    t0     = time.time()
    output = agent.generalize_all(dedup_dir)
    elapsed = time.time() - t0

    # ── Save combined output ───────────────────────────────────────────
    _GEN_DIR.mkdir(parents=True, exist_ok=True)
    _GEN_COMBINED.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # ── Save per-archetype files ───────────────────────────────────────
    print("\n[Gen] Writing per-archetype files...")
    _save_per_archetype(output, _GEN_ARCH_DIR)

    # ── Summary ───────────────────────────────────────────────────────
    rs = output.get("run_stats", {})
    print(f"\n[Gen] ══ Done ══")
    print(f"  {rs.get('n_input_cqs','?')} input CQs → "
          f"{rs.get('n_gen_cqs','?')} GenCQs  "
          f"({rs.get('reduction_rate',0)*100:.0f}% reduction)")
    bc = rs.get("coverage_breadth_counts", {})
    print(f"  Coverage: broad={bc.get('broad',0)}  "
          f"medium={bc.get('medium',0)}  narrow={bc.get('narrow',0)}")
    print(f"  Total time: {fmt_duration(elapsed)}")
    print(f"\n  Combined  → {_GEN_COMBINED.relative_to(_ROOT)}")
    print(f"  Archetype → {_GEN_ARCH_DIR.relative_to(_ROOT)}/GEN_*.json")

    return _GEN_COMBINED
