#!/bin/bash
# Run TMCMC estimation on Commensal HOBIC data
# Compare different cultivation condition to see if model fits better

set -e

cd /home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="_runs/Commensal_HOBIC_${TIMESTAMP}"

echo "=========================================="
echo "TMCMC Estimation: Commensal HOBIC"
echo "Timestamp: ${TIMESTAMP}"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="
echo "Settings: 1000 particles, 50 stages, 2 chains"
echo "Starting from Day 3, normalized data"
echo "=========================================="

python main/estimate_commensal_static.py \
    --condition Commensal \
    --cultivation HOBIC \
    --n-particles 1000 \
    --n-stages 50 \
    --n-chains 2 \
    --use-exp-init \
    --start-from-day 3 \
    --normalize-data \
    --n-jobs 10 \
    --seed 42 \
    --output-dir "${OUTPUT_DIR}" \
    --debug-level INFO

echo "=========================================="
echo "Estimation complete!"
echo "=========================================="

# Generate figures
echo "Generating all figures..."
python main/generate_all_figures.py --run-dir "${OUTPUT_DIR}" --n-samples 100

echo "All done! Check ${OUTPUT_DIR}/figures/ for plots."
