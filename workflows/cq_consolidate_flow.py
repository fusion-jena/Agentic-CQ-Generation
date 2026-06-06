"""
Workflow for Step 4: CQ consolidation + ontology schema extraction.

Reads the Step 2 generalised CQs, merges those that share a triple
signature across archetypes, and derives the ontology schema.

  Input:   data/papers/v2/generalized/cross_paper_gen_v2.json
  Output:  data/papers/v2/consolidated/cross_paper_consolidated.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from agents.cq_consolidation_agent import CQConsolidationAgent
from utils.llm_client import fmt_duration
from utils.model_config import get_model

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT     = Path(__file__).parent.parent
_GEN_FILE = _ROOT / "data" / "papers" / "v2" / "generalized" / "cross_paper_gen_v2.json"
_OUT_DIR  = _ROOT / "data" / "papers" / "v2" / "consolidated"
_OUT_FILE = _OUT_DIR / "cross_paper_consolidated.json"

_DEFAULT_MODEL = "llama4:128x17b"


def run_consolidate_pipeline(
    gen_path:         Optional[Path] = None,
    overwrite:        bool           = False,
    no_schema_format: bool           = False,
) -> Path:
    """
    Run the CQ consolidation + schema extraction pipeline.

    Args:
        gen_path:         path to Step 2 output (defaults to cross_paper_gen_v2.json).
        overwrite:        if False and output already exists, skip.
        no_schema_format: send format='json' instead of the Pydantic schema
                          (use if the Ollama server is too old for constrained output).

    Returns:
        Path to the consolidated output file.
    """
    gen_path = gen_path or _GEN_FILE

    if not gen_path.exists():
        raise FileNotFoundError(
            f"Step 2 output not found: {gen_path}\n"
            "Run Step 2 first: python main_cq_generalize.py"
        )

    if _OUT_FILE.exists() and not overwrite:
        print(f"[Consolidate] Output already exists — skipping "
              f"(use --overwrite to rerun):\n  {_OUT_FILE.relative_to(_ROOT)}")
        return _OUT_FILE

    model_name = get_model(
        "cq_consolidation_pipeline", "model", default=_DEFAULT_MODEL
    )

    agent = CQConsolidationAgent(
        model_name=model_name,
        no_schema_format=no_schema_format,
    )

    print(f"[Consolidate] Model  : {model_name}")
    print(f"[Consolidate] Input  : {gen_path.relative_to(_ROOT)}")
    print(f"[Consolidate] Output : {_OUT_FILE.relative_to(_ROOT)}")
    print()

    t0      = time.time()
    output  = agent.consolidate(gen_path)
    elapsed = time.time() - t0

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    _OUT_FILE.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    rs = output.get("run_stats", {})
    print(f"\n[Consolidate] ══ Done ══")
    print(f"  {rs.get('n_input_generalized_cqs', '?')} GenCQs → "
          f"{rs.get('n_consolidated_cqs', '?')} consolidated  "
          f"({rs.get('reduction_rate', 0) * 100:.0f}% reduction)")
    bc = rs.get("coverage_breadth_counts", {})
    print(f"  Coverage: broad={bc.get('broad', 0)}  "
          f"medium={bc.get('medium', 0)}  narrow={bc.get('narrow', 0)}")
    print(f"  Total time: {fmt_duration(elapsed)}")
    print(f"  Output → {_OUT_FILE.relative_to(_ROOT)}")

    return _OUT_FILE
