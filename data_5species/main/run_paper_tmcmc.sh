#!/bin/bash
# run_paper_tmcmc.sh
# ===================
# Run TMCMC for all 4 conditions with paper-consistent settings:
#   - N=1000 particles
#   - start-from-day=3 (Day 1 as IC, fit from Day 3)
#   - K-hill=0.05, n-hill=4
#   - No lambda weighting
#
# Usage:
#   nohup bash run_paper_tmcmc.sh > run_paper_tmcmc.log 2>&1 &

set -e
cd "$(dirname "$0")"

PYTHON=python3
ESTIMATOR=estimate_reduced_nishioka.py
COMMON="--n-particles 1000 --n-stages 30 --n-chains 2 --start-from-day 3 --K-hill 0.05 --n-hill 4 --n-jobs 12 --checkpoint-every 5"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "============================================="
echo "Paper TMCMC runs — $(date)"
echo "Settings: N=1000, start-from-day=3, K=0.05, n=4"
echo "============================================="

for COND_CULT in "Commensal Static" "Commensal HOBIC" "Dysbiotic Static" "Dysbiotic HOBIC"; do
    COND=$(echo $COND_CULT | awk '{print $1}')
    CULT=$(echo $COND_CULT | awk '{print $2}')
    TAG="${COND}_${CULT}_paper_${TIMESTAMP}"
    OUTDIR="_runs/${TAG}"

    echo ""
    echo "========================================="
    echo "[$COND $CULT] Starting — $(date)"
    echo "Output: $OUTDIR"
    echo "========================================="

    $PYTHON $ESTIMATOR \
        --condition "$COND" --cultivation "$CULT" \
        --output-dir "$OUTDIR" \
        --seed 42 \
        $COMMON \
        2>&1 | tee "${OUTDIR}.log"

    echo "[$COND $CULT] Done — $(date)"
    echo ""
done

echo ""
echo "============================================="
echo "All 4 conditions complete — $(date)"
echo "============================================="
