#!/bin/bash
# Full pipeline: retrain CS/CH DeepONet + NUTS 4-condition production run
# Usage: bash run_full_pipeline.sh
set -euo pipefail

PYTHON="$HOME/.pyenv/versions/miniconda3-latest/envs/klempt_fem/bin/python"
DEEPONET_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DEEPONET_DIR"

echo "=============================================="
echo " DeepONet + NUTS Full Pipeline"
echo " $(date)"
echo "=============================================="

# ─── Phase 1: Generate 50k training data for CS and CH ───
echo ""
echo "=== Phase 1: Generate 50k training data ==="

for COND in Commensal_Static Commensal_HOBIC; do
    DATAFILE="data/train_${COND}_N50000.npz"
    if [ -f "$DATAFILE" ]; then
        echo "  [SKIP] $DATAFILE already exists"
    else
        echo "  [GEN] $COND 50k samples..."
        $PYTHON generate_training_data.py \
            --condition "$COND" \
            --n-samples 50000 \
            --seed 42 \
            --n-time 100 \
            --maxtimestep 500
    fi
done

# ─── Phase 2: Train v2 models for CS and CH ───
echo ""
echo "=== Phase 2: Train v2 DeepONet models ==="

for COND in Commensal_Static Commensal_HOBIC; do
    case "$COND" in
        Commensal_Static) CKPT_DIR="checkpoints_CS_v2" ;;
        Commensal_HOBIC)  CKPT_DIR="checkpoints_CH_v2" ;;
    esac
    DATAFILE="data/train_${COND}_N50000.npz"

    # If 50k data doesn't exist, fall back to 10k
    if [ ! -f "$DATAFILE" ]; then
        DATAFILE="data/train_${COND}_N10000.npz"
    fi

    echo "  [TRAIN] $COND → $CKPT_DIR (data: $DATAFILE)"
    $PYTHON deeponet_hamilton.py train \
        --data "$DATAFILE" \
        --epochs 800 \
        --batch-size 256 \
        --lr 1e-3 \
        --p 64 --hidden 128 --n-layers 3 \
        --checkpoint-dir "$CKPT_DIR"
done

# ─── Phase 3: Evaluate MAP accuracy ───
echo ""
echo "=== Phase 3: Evaluate MAP accuracy ==="
$PYTHON eval_map_accuracy.py 2>&1 | tee eval_map_results.txt

# ─── Phase 4: NUTS 4-condition production run ───
echo ""
echo "=== Phase 4: NUTS 4-condition production run (--compare-all) ==="

# Run each condition with RW/HMC/NUTS comparison
for COND in Commensal_Static Commensal_HOBIC Dysbiotic_Static Dysbiotic_HOBIC; do
    echo ""
    echo "--- $COND: RW vs HMC vs NUTS ---"
    $PYTHON gradient_tmcmc_nuts.py \
        --real \
        --condition "$COND" \
        --compare-all \
        --n-particles 200 \
        --seed 42 \
        --paper-fig \
        --hmc-step-size 0.005 \
        --hmc-n-leapfrog 5 \
        --nuts-max-depth 6
done

# Also generate the 4-condition summary figure
echo ""
echo "--- 4-condition NUTS summary ---"
$PYTHON gradient_tmcmc_nuts.py \
    --real \
    --all-conditions \
    --n-particles 200 \
    --seed 42 \
    --paper-fig \
    --nuts-max-depth 6

echo ""
echo "=============================================="
echo " Pipeline complete: $(date)"
echo "=============================================="
echo " Results:"
echo "   - Checkpoints: checkpoints_CS_v2/, checkpoints_CH_v2/"
echo "   - MAP accuracy: eval_map_results.txt"
echo "   - NUTS figures: ../FEM/figures/paper_final/nuts_comparison_*.png"
echo "   - NUTS results: nuts_tmcmc_results.json"
echo "=============================================="
