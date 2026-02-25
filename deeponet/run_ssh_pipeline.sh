#!/bin/bash
# =============================================================================
# SSH Background Pipeline: DeepONet training + TMCMC production run
# =============================================================================
# Usage:  nohup bash run_ssh_pipeline.sh > ssh_pipeline.log 2>&1 &
# Monitor: tail -f ssh_pipeline.log
# =============================================================================
set -euo pipefail

# Python environments
PYTHON_NUMBA="/home/nishioka/.pyenv/shims/python3"         # has numba (for data gen)
PYTHON_JAX="/home/nishioka/.pyenv/versions/miniconda3-latest/envs/klempt_fem/bin/python"  # has JAX

DEEPONET_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$DEEPONET_DIR/.." && pwd)"
cd "$DEEPONET_DIR"

# Force unbuffered output
export PYTHONUNBUFFERED=1

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

echo "=============================================="
echo " DeepONet SSH Pipeline"
echo " Started: $(timestamp)"
echo " DEEPONET_DIR: $DEEPONET_DIR"
echo "=============================================="

# ═══════════════════════════════════════════════════════════════
# Phase A: Generate 50k training data for CS and CH (numba env)
# ═══════════════════════════════════════════════════════════════
echo ""
echo "=== Phase A: Generate 50k training data (numba env) ==="

for COND in Commensal_Static Commensal_HOBIC; do
    DATAFILE="data/train_${COND}_N50000.npz"
    if [ -f "$DATAFILE" ]; then
        echo "  [SKIP] $DATAFILE already exists ($(ls -lh "$DATAFILE" | awk '{print $5}'))"
    else
        echo "  [$(timestamp)] Generating $COND 50k samples..."
        $PYTHON_NUMBA generate_training_data.py \
            --condition "$COND" \
            --n-samples 50000 \
            --seed 42 \
            --n-time 100 \
            --maxtimestep 500 \
            --map-frac 0.5 \
            --map-std 0.15
        echo "  [$(timestamp)] Done: $COND"
    fi
done

echo "  Phase A complete: $(timestamp)"

# ═══════════════════════════════════════════════════════════════
# Phase B: Train v2 DeepONet models for CS and CH (JAX env)
# ═══════════════════════════════════════════════════════════════
echo ""
echo "=== Phase B: Train v2 DeepONet models (JAX env) ==="

for COND in Commensal_Static Commensal_HOBIC; do
    case "$COND" in
        Commensal_Static) CKPT_DIR="checkpoints_CS_50k" ;;
        Commensal_HOBIC)  CKPT_DIR="checkpoints_CH_50k" ;;
    esac

    # Prefer 50k data, fall back to 10k
    DATAFILE="data/train_${COND}_N50000.npz"
    if [ ! -f "$DATAFILE" ]; then
        DATAFILE="data/train_${COND}_N10000.npz"
        echo "  [WARN] 50k data not found, using 10k: $DATAFILE"
    fi

    echo "  [$(timestamp)] Training $COND → $CKPT_DIR (data: $DATAFILE)"
    $PYTHON_JAX deeponet_hamilton.py train \
        --data "$DATAFILE" \
        --epochs 800 \
        --batch-size 256 \
        --lr 1e-3 \
        --p 64 --hidden 128 --n-layers 3 \
        --checkpoint-dir "$CKPT_DIR"
    echo "  [$(timestamp)] Done: $COND → $CKPT_DIR"
done

echo "  Phase B complete: $(timestamp)"

# ═══════════════════════════════════════════════════════════════
# Phase C: Evaluate MAP accuracy with new checkpoints
# ═══════════════════════════════════════════════════════════════
echo ""
echo "=== Phase C: Evaluate MAP accuracy ==="
$PYTHON_JAX eval_map_accuracy.py 2>&1
echo "  Phase C complete: $(timestamp)"

# ═══════════════════════════════════════════════════════════════
# Phase D: DeepONet TMCMC 4-condition production run
# ═══════════════════════════════════════════════════════════════
echo ""
echo "=== Phase D: DeepONet TMCMC 4-condition production run ==="

MAIN_SCRIPT="$PROJECT_ROOT/data_5species/main/estimate_reduced_nishioka.py"
DATA_DIR="$PROJECT_ROOT/data_5species/data"

declare -A COND_CULT=(
    ["Commensal"]="Static HOBIC"
    ["Dysbiotic"]="Static HOBIC"
)

for CONDITION in Commensal Dysbiotic; do
    for CULTIVATION in ${COND_CULT[$CONDITION]}; do
        COND_KEY="${CONDITION}_${CULTIVATION}"
        OUT_DIR="$PROJECT_ROOT/data_5species/_runs/deeponet_${COND_KEY}"

        echo ""
        echo "  [$(timestamp)] Running DeepONet TMCMC: $COND_KEY"
        echo "  Output: $OUT_DIR"

        $PYTHON_JAX "$MAIN_SCRIPT" \
            --condition "$CONDITION" \
            --cultivation "$CULTIVATION" \
            --data-dir "$DATA_DIR" \
            --use-deeponet \
            --use-threads \
            --n-particles 500 \
            --n-stages 30 \
            --n-chains 2 \
            --n-jobs 8 \
            --seed 42 \
            --output-dir "$OUT_DIR" \
            --debug-level INFO \
            2>&1 | tee "${OUT_DIR}_tmcmc.log" || {
                echo "  [ERROR] TMCMC failed for $COND_KEY (exit=$?)"
                continue
            }

        echo "  [$(timestamp)] Done: $COND_KEY"
    done
done

echo ""
echo "  Phase D complete: $(timestamp)"

# ═══════════════════════════════════════════════════════════════
# Phase E: DH-only production with higher particles (if time)
# ═══════════════════════════════════════════════════════════════
echo ""
echo "=== Phase E: DH (Dysbiotic_HOBIC) high-particle run ==="

DH_HI_DIR="$PROJECT_ROOT/data_5species/_runs/deeponet_DH_1000p"
echo "  [$(timestamp)] Running DH 1000-particle DeepONet TMCMC..."

$PYTHON_JAX "$MAIN_SCRIPT" \
    --condition Dysbiotic \
    --cultivation HOBIC \
    --data-dir "$DATA_DIR" \
    --use-deeponet \
    --use-threads \
    --n-particles 1000 \
    --n-stages 30 \
    --n-chains 2 \
    --n-jobs 8 \
    --seed 42 \
    --output-dir "$DH_HI_DIR" \
    --debug-level INFO \
    2>&1 | tee "${DH_HI_DIR}_tmcmc.log" || {
        echo "  [ERROR] DH high-particle run failed (exit=$?)"
    }

echo "  [$(timestamp)] Phase E complete"

# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════
echo ""
echo "=============================================="
echo " Pipeline complete: $(timestamp)"
echo "=============================================="
echo " Phase A: 50k training data → data/"
echo " Phase B: CS/CH v2 checkpoints → checkpoints_CS_50k/, checkpoints_CH_50k/"
echo " Phase C: MAP accuracy → map_accuracy_results.json"
echo " Phase D: TMCMC results → data_5species/_runs/deeponet_*"
echo " Phase E: DH 1000p → data_5species/_runs/deeponet_DH_1000p"
echo "=============================================="
