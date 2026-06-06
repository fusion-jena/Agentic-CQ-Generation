"""
Entry point for Step 4 of the CQ narrowing pipeline:
cross-archetype consolidation + ontology schema extraction.

Reads the Step 2 generalised CQs, merges those that share a triple
signature across archetypes, and emits a compact set of consolidated CQs
together with an ontology schema (classes + properties).

Usage
─────
  # Run with defaults
  python main_cq_consolidate.py

  # Re-run and overwrite existing output
  python main_cq_consolidate.py --overwrite

  # Fall back to format='json' if Ollama server is too old for schema-constrained output
  python main_cq_consolidate.py --no-schema-format

  # List models available at the Jena endpoint
  python main_cq_consolidate.py --list-models

  # Inspect the Pydantic JSON Schema the model must conform to
  python main_cq_consolidate.py --dump-schema

Prerequisites
─────────────
  Step 2 must have been run first:
    python main_cq_generalize.py

Output
──────
  data/papers/v2/consolidated/cross_paper_consolidated.json
"""

import argparse
import json
import sys

from agents.cq_consolidation_agent import CQConsolidationAgent
from utils.model_config import get_model
from workflows.cq_consolidate_flow import run_consolidate_pipeline

_DEFAULT_MODEL = "llama4:128x17b"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CQ Consolidation + Schema — Step 4 of the CQ narrowing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing consolidated output. Default: skip if output exists.",
    )
    parser.add_argument(
        "--no-schema-format",
        action="store_true",
        default=False,
        dest="no_schema_format",
        help="Send format='json' instead of the Pydantic JSON Schema "
             "(use if the Ollama server is too old for constrained decoding).",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        default=False,
        dest="list_models",
        help="List models available at the Jena endpoint and exit.",
    )
    parser.add_argument(
        "--dump-schema",
        action="store_true",
        default=False,
        dest="dump_schema",
        help="Print the Pydantic JSON Schema for the model output contract and exit.",
    )
    args = parser.parse_args()

    if args.dump_schema:
        print(json.dumps(CQConsolidationAgent.json_schema(), indent=2))
        return

    model_name = get_model(
        "cq_consolidation_pipeline", "model", default=_DEFAULT_MODEL
    )

    if args.list_models:
        agent = CQConsolidationAgent(model_name=model_name)
        print("Models at Jena Ollama endpoint:")
        try:
            for m in agent.list_available_models():
                print(f"  {m}")
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            sys.exit(1)
        return

    print("=" * 60)
    print("  CQ Narrowing Pipeline — Step 4: Consolidation + Schema")
    print("=" * 60)
    print(f"  Model             : {model_name}")
    print(f"  Overwrite         : {args.overwrite}")
    print(f"  Schema-constrained: {not args.no_schema_format}")
    print()

    try:
        out_path = run_consolidate_pipeline(
            overwrite=args.overwrite,
            no_schema_format=args.no_schema_format,
        )
    except FileNotFoundError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as exc:
        print(f"\nLLM / validation error: {exc}", file=sys.stderr)
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
