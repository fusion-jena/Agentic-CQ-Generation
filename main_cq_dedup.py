"""
Entry point for Step 1 of the CQ narrowing pipeline:
intra-paper deduplication.

Usage
─────
  # Single paper (recommended for testing first)
  python main_cq_dedup.py --paper Jacobs2007

  # All papers
  python main_cq_dedup.py

  # Re-run and overwrite existing outputs
  python main_cq_dedup.py --paper Jacobs2007 --overwrite
  python main_cq_dedup.py --overwrite

Output
──────
  data/papers/v2/dedup/{paper_id}_dedup.json  (one file per paper)
"""

import argparse
import sys

from workflows.cq_dedup_flow import run_dedup_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CQ Intra-paper Deduplication — Step 1 of the CQ narrowing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--paper",
        type=str,
        default=None,
        metavar="PAPER_ID",
        help=(
            "Paper ID to process, e.g. 'Jacobs2007'. "
            "If omitted, all papers in v2/cq_output/ are processed."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing dedup outputs. Default: skip already-processed papers.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  CQ Narrowing Pipeline — Step 1: Intra-paper Deduplication")
    print("=" * 60)

    if args.paper:
        print(f"  Mode    : single paper — {args.paper}")
    else:
        print("  Mode    : all papers")
    print(f"  Overwrite: {args.overwrite}")
    print()

    try:
        output_paths = run_dedup_pipeline(
            paper_id=args.paper,
            overwrite=args.overwrite,
        )
    except FileNotFoundError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"\nUnexpected error: {exc}", file=sys.stderr)
        raise

    print("\n" + "=" * 60)
    print(f"  Done — {len(output_paths)} output(s) written:")
    for p in output_paths:
        print(f"    {p}")
    print("=" * 60)


if __name__ == "__main__":
    main()
