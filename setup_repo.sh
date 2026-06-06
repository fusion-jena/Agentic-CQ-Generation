#!/bin/bash
# =============================================================================
# setup_repo.sh
# Copies all v2-pipeline files from the original experiment folder into this
# clean repository. Run this ONCE from inside the copolymer-cq-pipeline folder.
#
# Usage:
#   cd /Users/vishvapalsinhji/Documents/PhD/Experimets/copolymer-cq-pipeline
#   bash setup_repo.sh
# =============================================================================

SRC="/Users/vishvapalsinhji/Documents/PhD/Experimets/Copolymer_Kg_development_V"
DEST="$(cd "$(dirname "$0")" && pwd)"

echo "=================================================="
echo "  copolymer-cq-pipeline repo setup"
echo "  SRC  : $SRC"
echo "  DEST : $DEST"
echo "=================================================="

# ── helpers ──────────────────────────────────────────────────────────────────
ok()   { echo "  ✓  $1"; }
skip() { echo "  –  $1  (already exists, skipped)"; }
fail() { echo "  ✗  MISSING: $1"; }

copy_if_exists() {
    local src="$SRC/$1"
    local dst="$DEST/$1"
    if [ ! -f "$src" ]; then
        fail "$1"
        return
    fi
    if [ -f "$dst" ]; then
        skip "$1"
        return
    fi
    cp "$src" "$dst" && ok "$1" || fail "$1"
}

# ── entry points ─────────────────────────────────────────────────────────────
echo ""
echo "── Entry points ──"
# main_paper_persona_cq_v2.py already copied by Claude — skip
copy_if_exists "main_cq_dedup.py"
copy_if_exists "main_cq_generalize.py"
copy_if_exists "main_cq_validate.py"
copy_if_exists "main_cq_consolidate.py"
copy_if_exists "aggregate_eval_v2.py"

# ── root config files ─────────────────────────────────────────────────────────
echo ""
echo "── Root files ──"
copy_if_exists "requirements.txt"
copy_if_exists "pyproject.toml"
copy_if_exists ".python-version"

# ── workflows ─────────────────────────────────────────────────────────────────
echo ""
echo "── Workflows ──"
copy_if_exists "workflows/paper_persona_cq_v2_flow.py"
copy_if_exists "workflows/cq_dedup_flow.py"
copy_if_exists "workflows/cq_generalize_flow.py"
copy_if_exists "workflows/cq_validate_flow.py"
copy_if_exists "workflows/cq_consolidate_flow.py"

# ── agents ────────────────────────────────────────────────────────────────────
echo ""
echo "── Agents ──"
copy_if_exists "agents/paper_chunker.py"
copy_if_exists "agents/paper_indexer.py"
copy_if_exists "agents/paper_iterative_extractor.py"
copy_if_exists "agents/paper_retriever.py"
copy_if_exists "agents/paper_persona_cq_generator_v2.py"
copy_if_exists "agents/paper_persona_cq_validator_v2.py"
copy_if_exists "agents/paper_persona_cq_refiner_v2.py"
copy_if_exists "agents/paper_metadata_extractor.py"
copy_if_exists "agents/cq_dedup_agent.py"
copy_if_exists "agents/cq_generalizer_agent.py"
copy_if_exists "agents/cq_validator_agent.py"
copy_if_exists "agents/cq_consolidation_agent.py"

# ── utils ─────────────────────────────────────────────────────────────────────
echo ""
echo "── Utils ──"
copy_if_exists "utils/rolling_state.py"
copy_if_exists "utils/cq_evaluator.py"
copy_if_exists "utils/llm_client.py"
copy_if_exists "utils/embeddings.py"
copy_if_exists "utils/state_compactor.py"
copy_if_exists "utils/vectorstore.py"
copy_if_exists "utils/pdf_extractor.py"
copy_if_exists "utils/rich_pdf_extractor.py"
copy_if_exists "utils/model_config.py"

# ── config ────────────────────────────────────────────────────────────────────
echo ""
echo "── Config ──"
copy_if_exists "config/models.json"
copy_if_exists "config/domain_priors.json"
copy_if_exists "config/extraction_schema.json"
copy_if_exists "config/paper_persona_cq_v2_checks.json"
copy_if_exists "config/personas.json"
copy_if_exists "config/cq_archetypes.json"
copy_if_exists "config/cq_few_shots.json"

# ── raw PDFs (flagged in .gitignore — local use only) ─────────────────────────
echo ""
echo "── Raw PDFs (local only — listed in .gitignore) ──"
mkdir -p "$DEST/data/papers/raw"
for pdf in "$SRC/data/papers/raw/"*.pdf; do
    fname="$(basename "$pdf")"
    dst="$DEST/data/papers/raw/$fname"
    if [ -f "$dst" ]; then
        skip "data/papers/raw/$fname"
    else
        cp "$pdf" "$dst" && ok "data/papers/raw/$fname" || fail "data/papers/raw/$fname"
    fi
done

# ── v2 pipeline outputs ───────────────────────────────────────────────────────
echo ""
echo "── v2 CQ outputs ──"
cp -n "$SRC/data/papers/v2/cq_output/"*.json \
      "$DEST/data/papers/v2/cq_output/" 2>/dev/null \
  && echo "  ✓  cq_output/ ($(ls "$DEST/data/papers/v2/cq_output/" | wc -l | tr -d ' ') files)" \
  || echo "  –  cq_output/ (already present)"

echo ""
echo "── v2 dedup outputs ──"
cp -n "$SRC/data/papers/v2/dedup/"*.json \
      "$DEST/data/papers/v2/dedup/" 2>/dev/null \
  && echo "  ✓  dedup/ ($(ls "$DEST/data/papers/v2/dedup/" | wc -l | tr -d ' ') files)" \
  || echo "  –  dedup/ (already present)"

echo ""
echo "── v2 generalized outputs ──"
cp -n "$SRC/data/papers/v2/generalized/"*.json \
      "$DEST/data/papers/v2/generalized/" 2>/dev/null \
  && echo "  ✓  generalized/ json" || echo "  –  generalized/ json (already present)"
cp -n "$SRC/data/papers/v2/generalized/by_archetype/"*.json \
      "$DEST/data/papers/v2/generalized/by_archetype/" 2>/dev/null \
  && echo "  ✓  generalized/by_archetype/ json" || echo "  –  by_archetype/ (already present)"

echo ""
echo "── v2 validation outputs ──"
cp -n "$SRC/data/papers/v2/cq_validation/"*.json \
      "$DEST/data/papers/v2/cq_validation/" 2>/dev/null \
  && echo "  ✓  cq_validation/" || echo "  –  cq_validation/ (already present)"

echo ""
echo "── v2 consolidated outputs ──"
cp -n "$SRC/data/papers/v2/consolidated/"*.json \
      "$DEST/data/papers/v2/consolidated/" 2>/dev/null \
  && echo "  ✓  consolidated/" || echo "  –  consolidated/ (already present)"

echo ""
echo "── v2 evaluation outputs ──"
cp -n "$SRC/data/papers/v2/evaluation/"*.json \
      "$DEST/data/papers/v2/evaluation/" 2>/dev/null \
  && echo "  ✓  evaluation/ ($(ls "$DEST/data/papers/v2/evaluation/" | wc -l | tr -d ' ') files)" \
  || echo "  –  evaluation/ (already present)"

# evaluation summary if it exists
[ -f "$SRC/data/papers/v2/evaluation_summary.json" ] && \
    cp -n "$SRC/data/papers/v2/evaluation_summary.json" \
          "$DEST/data/papers/v2/evaluation_summary.json" && \
    ok "data/papers/v2/evaluation_summary.json"

echo ""
echo "=================================================="
echo "  Setup complete."
echo "  Next steps:"
echo "    1. Review the repo structure with: find . -not -path '*/__pycache__/*' | sort"
echo "    2. Confirm PDFs are in data/papers/raw/"
echo "    3. When ready to push to GitHub: git init && git add . && git commit"
echo "       (PDFs are in .gitignore so they won't be committed)"
echo "=================================================="
