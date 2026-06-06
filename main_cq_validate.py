"""
Entry point for Step 3 of the CQ narrowing pipeline:
answer retrieval + validation.

For every GenCQ from Step 2, retrieves relevant chunks from the paper's
FAISS index, generates an answer, and scores fidelity against the
original expected answers from Step 1/2.

Usage
─────
  # Run validation
  python main_cq_validate.py

  # Re-run and overwrite existing output
  python main_cq_validate.py --overwrite

Prerequisites
─────────────
  Steps 1 and 2 must have been run first:
    python main_cq_dedup.py
    python main_cq_generalize.py

Output
──────
  data/papers/v2/cq_validation/answer_validation_v2.json
"""

import argparse
import sys

from workflows.cq_validate_flow import run_validate_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CQ Answer Validation — Step 3 of the CQ narrowing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing validation output. Default: skip if output exists.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  CQ Narrowing Pipeline — Step 3: Answer Validation")
    print("=" * 60)
    print(f"  Overwrite: {args.overwrite}")
    print()

    try:
        out_path = run_validate_pipeline(overwrite=args.overwrite)
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
