#!/bin/bash
# Importance-weighted DeepONet retraining pipeline for a32, a35 overlap improvement.
#
# Prerequisites:
#   - 50k importance-weighted data: data/train_Dysbiotic_HOBIC_N50000_importance.npz
#     (generate with: python generate_training_data.py --condition Dysbiotic_HOBIC \
#       --n-samples 50000 --posterior-dir ../data_5species/_runs/dh_baseline --posterior-frac 0.5)
#     Note: _importance suffix preserves uniform data (train_*_N50000.npz)
#
# Usage:
#   ./run_importance_weighted_pipeline.sh
#
# Output:
#   - checkpoints_DH_50k_importance/  (new DeepONet)
#   - data_5species/_runs/deeponet_DH_50k_importance/  (TMCMC posterior)
#   - FEM/figures/paper_final/Fig22_deeponet_posterior_comparison.*  (overlap comparison)

set -e
cd "$(dirname "$0")"
PROJECT_ROOT="$(cd .. && pwd)"
DATA_5SPECIES="${PROJECT_ROOT}/data_5species"
PYTHON="${HOME}/.pyenv/versions/miniconda3-latest/envs/klempt_fem2/bin/python"

# Prefer _importance (posterior-weighted); fallback to base (e.g. if renamed)
DATA="data/train_Dysbiotic_HOBIC_N50000_importance.npz"
[ ! -f "$DATA" ] && DATA="data/train_Dysbiotic_HOBIC_N50000.npz"
CKPT="checkpoints_DH_50k_importance"
OUT_RUN="deeponet_DH_50k_importance"

echo "=============================================="
echo " Importance-Weighted DeepONet Pipeline"
echo " $(date)"
echo "=============================================="

# Phase 1: Verify data exists
if [ ! -f "$DATA" ]; then
    echo "ERROR: $DATA not found."
    echo "Run first: python generate_training_data.py --condition Dysbiotic_HOBIC \\"
    echo "  --n-samples 50000 --posterior-dir ../data_5species/_runs/dh_baseline --posterior-frac 0.5"
    exit 1
fi
echo "[1/4] Data OK: $DATA"

# Phase 2: Train DeepONet
echo ""
echo "[2/4] Training DeepONet (50k importance-weighted)..."
$PYTHON deeponet_hamilton.py train \
    --data "$DATA" \
    --epochs 800 \
    --batch-size 512 \
    --lr 1e-3 \
    --p 128 \
    --hidden 256 \
    --n-layers 4 \
    --checkpoint-dir "$CKPT"

# Phase 3: Run TMCMC with new DeepONet (DH only)
echo ""
echo "[3/4] Running TMCMC with new DeepONet (Dysbiotic HOBIC)..."
mkdir -p "${DATA_5SPECIES}/_runs"
(cd "${DATA_5SPECIES}" && $PYTHON main/estimate_reduced_nishioka.py \
    --condition Dysbiotic \
    --cultivation HOBIC \
    --use-deeponet \
    --deeponet-checkpoint "${PROJECT_ROOT}/deeponet/${CKPT}" \
    --output-dir "${DATA_5SPECIES}/_runs/${OUT_RUN}" \
    --n-particles 1000 \
    --n-stages 30 \
    --n-chains 2 \
    --n-jobs 1 \
    --use-threads \
    --lambda-pg 2.0 \
    --lambda-late 1.5 \
    --seed 42)

# Phase 4: Evaluate overlap
echo ""
echo "[4/4] Generating Fig22 overlap comparison..."
$PYTHON generate_fig22_posterior_comparison.py --don-dir "${DATA_5SPECIES}/_runs/${OUT_RUN}"

echo ""
echo "=============================================="
echo " Pipeline complete."
echo " Overlap results in: FEM/figures/paper_final/Fig22_*"
echo "=============================================="
