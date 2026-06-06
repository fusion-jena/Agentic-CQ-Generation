"""
Entry point for Step 2 of the CQ narrowing pipeline:
cross-paper generalisation.

Reads all Step 1 dedup outputs, groups CQs by archetype, and makes one
LLM call per archetype to produce domain-level generalised CQs.

Usage
─────
  # Run on all dedup files in v2/dedup/
  python main_cq_generalize.py

  # Re-run and overwrite existing output
  python main_cq_generalize.py --overwrite

Prerequisites
─────────────
  Step 1 must have been run first:
    python main_cq_dedup.py          (single paper test)
    python main_cq_dedup.py          (all papers)

Output
──────
  data/papers/v2/generalized/cross_paper_gen_v2.json        ← combined
  data/papers/v2/generalized/by_archetype/GEN_{ARCH}.json   ← per archetype
"""

import argparse
import sys

from workflows.cq_generalize_flow import run_generalize_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CQ Cross-paper Generalisation — Step 2 of the CQ narrowing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing generalisation output. Default: skip if output exists.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  CQ Narrowing Pipeline — Step 2: Cross-paper Generalisation")
    print("=" * 60)
    print(f"  Overwrite: {args.overwrite}")
    print()

    try:
        out_path = run_generalize_pipeline(overwrite=args.overwrite)
    except FileNotFoundError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"\nUnexpected error: {exc}", file=sys.stderr)
        raise

    print(f"\n{'=' * 60}")
    print(f"  Done — output written to:")
    print(f"    {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
