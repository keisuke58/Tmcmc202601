#!/bin/bash
# Phase 2 → Phase 3: HMP 組成 → GNN prior → TMCMC
# Usage:
#   ./run_phase2_tmcmc.sh                    # デモ prior で TMCMC
#   ./run_phase2_tmcmc.sh data/hmp_oral/species_abundance.csv  # HMP 実データ

set -e
cd "$(dirname "$0")"
PROJECT_ROOT="$(cd .. && pwd)"

INPUT="${1:-data/hmp_oral/species_abundance_demo.csv}"
PRIOR_JSON="data/hmp_oral/gnn_prior.json"

# 1. GNN で a_ij 予測 → prior JSON
echo "=== Step 1: GNN predict → prior JSON ==="
python predict_hmp.py --input "$INPUT" --output-prior "$PRIOR_JSON"

# 2. TMCMC with GNN prior
echo ""
echo "=== Step 2: TMCMC with --use-gnn-prior --gnn-prior-json ==="
cd "$PROJECT_ROOT/data_5species/main"
python estimate_reduced_nishioka.py \
  --condition Dysbiotic \
  --cultivation HOBIC \
  --use-gnn-prior \
  --gnn-prior-json "$PROJECT_ROOT/gnn/$PRIOR_JSON" \
  --gnn-sigma 1.0 \
  --n-particles 100 \
  --n-stages 15

echo ""
echo "Done. Check output in data_5species/_runs/"
