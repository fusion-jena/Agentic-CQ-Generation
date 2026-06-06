#!/usr/bin/env python3
"""
aggregate_eval_v2.py
====================================================================
Aggregate the per-paper reference-free evaluation results produced by
the v2.1 CQ-generation pipeline into corpus-wide summary statistics.

WHY THIS SCRIPT EXISTS
----------------------
The pipeline writes one evaluation file per paper to:

    data/papers/v2/evaluation/<paper_id>_..._eval_v2.json

Each file contains a reference-free quality score for that single paper.
For the paper write-up we don't want 14 separate numbers -- we want the
CORPUS MEAN (and spread) of each metric, e.g.

    "Across the 14-paper corpus, mean question faithfulness was 0.97
     (SD 0.02), while mean coverage was 0.45 (SD 0.06)."

This script reads every *_eval_v2.json file in the evaluation folder,
pulls out the four scored metric families plus a few useful diagnostics,
and prints:
    1. A per-paper table (so you can spot outliers).
    2. The corpus mean, standard deviation, min and max for each metric.
    3. A handful of derived totals (e.g. total questions, total off-paper).

It is intentionally dependency-free (standard library only) so you can
run it in any Python 3.8+ environment without installing anything.

HOW THE SCORE IS STRUCTURED (for reference)
-------------------------------------------
Inside each file, the numbers we care about live under "evaluation":

    evaluation.overall_score                         -> weighted composite
    evaluation.n_questions                           -> # CQs for this paper
    evaluation.families.coverage.score               -> 0..1
    evaluation.families.question_faithfulness.score  -> 0..1
    evaluation.families.diversity.score              -> 0..1
    evaluation.families.triplifiability.score        -> 0..1

    # extra detail we also surface because it is useful for the paper:
    evaluation.families.coverage.concepts.ratio              -> concept coverage
    evaluation.families.question_faithfulness.mean_cosine    -> avg Q->evidence sim
    evaluation.families.question_faithfulness.n_off_paper    -> # ungrounded CQs
    evaluation.families.diversity.n_near_duplicates          -> # near-dup CQs

The composite weights (coverage 0.38, faithfulness 0.44, diversity 0.12,
triplifiability 0.06) are already baked into overall_score by the
pipeline, so we just read overall_score directly rather than recomputing.

USAGE
-----
    cd Copolymer_Kg_development_V
    python aggregate_eval_v2.py

    # or point it at a different folder:
    python aggregate_eval_v2.py --eval-dir data/papers/v2/evaluation

    # or also dump the aggregated result to a JSON file:
    python aggregate_eval_v2.py --out data/papers/v2/evaluation_summary.json
====================================================================
"""

import argparse
import glob
import json
import os
import statistics
from typing import Any, Dict, List, Optional


# ----------------------------------------------------------------------
# Helper: safely walk a nested dict via a list of keys.
# Returns `default` if any key along the path is missing, so a single
# malformed file never crashes the whole aggregation.
# ----------------------------------------------------------------------
def dig(d: Dict[str, Any], path: List[str], default: Optional[Any] = None) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


# ----------------------------------------------------------------------
# Helper: compute mean / sd / min / max for a list of numbers.
# Guards against empty lists and single-element lists (population stdev
# of one value is 0, sample stdev is undefined -> we report 0.0).
# ----------------------------------------------------------------------
def summarize(values: List[float]) -> Dict[str, float]:
    clean = [v for v in values if isinstance(v, (int, float))]
    if not clean:
        return {"mean": 0.0, "sd": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    return {
        "mean": statistics.mean(clean),
        "sd": statistics.pstdev(clean) if len(clean) > 1 else 0.0,
        "min": min(clean),
        "max": max(clean),
        "n": len(clean),
    }


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Parse command-line arguments (all optional, sensible defaults).
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Aggregate per-paper reference-free eval JSONs into corpus stats."
    )
    parser.add_argument(
        "--eval-dir",
        default="data/papers/v2/evaluation",
        help="Folder containing *_eval_v2.json files "
             "(default: data/papers/v2/evaluation)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional path to write the aggregated summary as JSON.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 2. Find every evaluation file in the folder.
    #    The pipeline names them "<paper>..._eval_v2.json", so we glob on
    #    that suffix to avoid accidentally picking up other JSON files.
    # ------------------------------------------------------------------
    pattern = os.path.join(args.eval_dir, "*_eval_v2.json")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"[!] No files matched: {pattern}")
        print("    Check the --eval-dir path and try again.")
        return

    print(f"[i] Found {len(files)} evaluation file(s) in {args.eval_dir}\n")

    # ------------------------------------------------------------------
    # 3. Read each file and extract the metrics we care about.
    #    We collect them into parallel lists (one entry per paper) so we
    #    can both print a per-paper table and compute aggregates after.
    # ------------------------------------------------------------------
    rows: List[Dict[str, Any]] = []   # per-paper extracted values

    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            # A broken file shouldn't kill the run -- warn and skip it.
            print(f"[!] Skipping unreadable file {os.path.basename(fp)}: {exc}")
            continue

        ev = data.get("evaluation", {})

        rows.append({
            "paper_id":      ev.get("paper_id", os.path.basename(fp)),
            "n_questions":   ev.get("n_questions", 0),
            "overall":       ev.get("overall_score"),
            "coverage":      dig(ev, ["families", "coverage", "score"]),
            "faithfulness":  dig(ev, ["families", "question_faithfulness", "score"]),
            "diversity":     dig(ev, ["families", "diversity", "score"]),
            "triplifiability": dig(ev, ["families", "triplifiability", "score"]),
            # extra diagnostics worth reporting in the paper:
            "concept_cov":   dig(ev, ["families", "coverage", "concepts", "ratio"]),
            "faith_cosine":  dig(ev, ["families", "question_faithfulness", "mean_cosine"]),
            "n_off_paper":   dig(ev, ["families", "question_faithfulness", "n_off_paper"], 0),
            "n_near_dupes":  dig(ev, ["families", "diversity", "n_near_duplicates"], 0),
        })

    # ------------------------------------------------------------------
    # 4. Print a per-paper table.
    #    Seeing every paper lets you eyeball outliers (e.g. one paper
    #    dragging coverage down) before trusting the mean.
    # ------------------------------------------------------------------
    header = (
        f"{'paper_id':<16} {'n_q':>4} {'overall':>8} {'cover':>7} "
        f"{'faith':>7} {'divers':>7} {'triple':>7} {'off':>4}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        def fmt(x: Optional[float]) -> str:
            return f"{x:.3f}" if isinstance(x, (int, float)) else "  -  "
        print(
            f"{str(r['paper_id'])[:16]:<16} "
            f"{r['n_questions']:>4} "
            f"{fmt(r['overall']):>8} "
            f"{fmt(r['coverage']):>7} "
            f"{fmt(r['faithfulness']):>7} "
            f"{fmt(r['diversity']):>7} "
            f"{fmt(r['triplifiability']):>7} "
            f"{r['n_off_paper']:>4}"
        )

    # ------------------------------------------------------------------
    # 5. Compute corpus-wide aggregates for each scored metric.
    # ------------------------------------------------------------------
    metrics = {
        "overall_score":   [r["overall"] for r in rows],
        "coverage":        [r["coverage"] for r in rows],
        "question_faithfulness": [r["faithfulness"] for r in rows],
        "diversity":       [r["diversity"] for r in rows],
        "triplifiability": [r["triplifiability"] for r in rows],
        "concept_coverage_ratio": [r["concept_cov"] for r in rows],
        "faithfulness_mean_cosine": [r["faith_cosine"] for r in rows],
    }

    summary = {name: summarize(vals) for name, vals in metrics.items()}

    # A few simple corpus totals (sums, not means).
    totals = {
        "n_papers":           len(rows),
        "total_questions":    sum(r["n_questions"] for r in rows),
        "total_off_paper":    sum(r["n_off_paper"] for r in rows),
        "total_near_dupes":   sum(r["n_near_dupes"] for r in rows),
    }

    # ------------------------------------------------------------------
    # 6. Print the aggregates in a paper-ready "mean (SD) [min-max]" form.
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"CORPUS SUMMARY  (n = {totals['n_papers']} papers)")
    print("=" * 60)
    for name, s in summary.items():
        if s["n"] == 0:
            print(f"{name:<26} (no values found)")
            continue
        print(
            f"{name:<26} mean={s['mean']:.3f}  sd={s['sd']:.3f}  "
            f"min={s['min']:.3f}  max={s['max']:.3f}"
        )

    print("-" * 60)
    print(f"{'total CQs evaluated':<26} {totals['total_questions']}")
    print(f"{'total off-paper CQs':<26} {totals['total_off_paper']}")
    print(f"{'total near-duplicate CQs':<26} {totals['total_near_dupes']}")
    print("=" * 60)

    # A ready-to-paste sentence for the paper, using the two headline
    # metrics (faithfulness = strength, coverage = limitation).
    f = summary["question_faithfulness"]
    c = summary["coverage"]
    print("\nDraft sentence for the paper:")
    print(
        f'  "Across the {totals["n_papers"]}-paper corpus, mean question '
        f'faithfulness was {f["mean"]:.2f} (SD {f["sd"]:.2f}), with '
        f'{totals["total_off_paper"]} off-paper questions in total, while '
        f'mean coverage was {c["mean"]:.2f} (SD {c["sd"]:.2f})."'
    )

    # ------------------------------------------------------------------
    # 7. Optionally write the full aggregate to a JSON file so it can be
    #    version-controlled / re-used in a table-generation script.
    # ------------------------------------------------------------------
    if args.out:
        out_obj = {
            "n_papers": totals["n_papers"],
            "per_metric": summary,
            "totals": totals,
            "per_paper": rows,
        }
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out_obj, fh, indent=2)
        print(f"\n[i] Wrote aggregated summary to {args.out}")


if __name__ == "__main__":
    main()
