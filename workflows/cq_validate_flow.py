"""
Workflow for Step 3: answer retrieval + validation.

Reads the Step 2 combined output, runs the CQValidatorAgent over all
GenCQs, and saves the results to:

  data/papers/v2/cq_validation/answer_validation_v2.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from agents.cq_validator_agent import CQValidatorAgent
from utils.llm_client import fmt_duration
from utils.model_config import get_model

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT         = Path(__file__).parent.parent
_GEN_COMBINED = _ROOT / "data" / "papers" / "v2" / "generalized" / "cross_paper_gen_v2.json"
_VAL_DIR      = _ROOT / "data" / "papers" / "v2" / "cq_validation"
_VAL_OUTPUT   = _VAL_DIR / "answer_validation_v2.json"


def run_validate_pipeline(
    gen_path:  Optional[Path] = None,
    overwrite: bool = False,
) -> Path:
    """
    Run the answer validation pipeline.

    Args:
        gen_path:  path to the Step 2 combined JSON (defaults to cross_paper_gen_v2.json).
        overwrite: if False and output already exists, skip.

    Returns:
        Path to the written validation output file.
    """
    gen_path = gen_path or _GEN_COMBINED

    # ── Pre-flight checks ─────────────────────────────────────────────
    if not gen_path.exists():
        raise FileNotFoundError(
            f"Step 2 output not found: {gen_path}\n"
            "Run Step 2 first: python main_cq_generalize.py"
        )

    if _VAL_OUTPUT.exists() and not overwrite:
        print(f"[Val] Validation output already exists — skipping "
              f"(use --overwrite to rerun):\n  {_VAL_OUTPUT.relative_to(_ROOT)}")
        return _VAL_OUTPUT

    # ── Initialise agent ──────────────────────────────────────────────
    model_name  = get_model("cq_validate_pipeline", "validator_agent",
                            default="deepseek-r1:32b")
    embed_model = get_model("cq_validate_pipeline", "embed_model",
                            default="BAAI/bge-small-en-v1.5")

    agent = CQValidatorAgent(
        model_name  = model_name,
        embed_model = embed_model,
    )

    print(f"[Val] Model       : {model_name}  |  think=False")
    print(f"[Val] Embed model : {embed_model}")
    print(f"[Val] Source      : {gen_path.relative_to(_ROOT)}")
    print(f"[Val] Output      : {_VAL_DIR.relative_to(_ROOT)}\n")

    # ── Run ───────────────────────────────────────────────────────────
    t0     = time.time()
    output = agent.validate_all(gen_path)
    elapsed = time.time() - t0

    # ── Save ──────────────────────────────────────────────────────────
    _VAL_DIR.mkdir(parents=True, exist_ok=True)
    _VAL_OUTPUT.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # ── Summary ───────────────────────────────────────────────────────
    rs = output.get("run_stats", {})
    vc = rs.get("verdict_counts", {})
    qc = rs.get("quality_flag_counts", {})

    print(f"\n[Val] ══ Done ══")
    print(f"  {rs.get('n_gen_cqs','?')} GenCQs  |  "
          f"{rs.get('n_original_cqs_tested','?')} original CQ validations")
    print(f"  Overall mean fidelity : {rs.get('overall_mean_fidelity', 0):.3f}")
    print(f"  Verdicts  : PASS={vc.get('PASS',0)}  PARTIAL={vc.get('PARTIAL',0)}  "
          f"FAIL={vc.get('FAIL',0)}  NOT_FOUND={vc.get('NOT_FOUND',0)}  "
          f"INDEX_MISSING={vc.get('INDEX_MISSING',0)}")
    print(f"  Quality   : {qc}")
    print(f"  Time      : {fmt_duration(elapsed)}")
    print(f"\n  Output → {_VAL_OUTPUT.relative_to(_ROOT)}")

    return _VAL_OUTPUT
