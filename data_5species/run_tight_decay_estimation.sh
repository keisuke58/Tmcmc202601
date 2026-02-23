#!/bin/bash
# Run TMCMC estimation with tighter decay priors
# Purpose: Address Species 1 over-decay issue by limiting b1-b5 to [0, 1]
# Expected runtime: ~6-8 hours

set -e

cd /home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="_runs/tight_decay_v1_${TIMESTAMP}"

echo "=========================================="
echo "TMCMC with Tighter Decay Priors"
echo "Timestamp: ${TIMESTAMP}"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="
echo "Key change: b1-b5 bounds [0, 1] instead of [0, 3]"
echo "Particles: 1000, Stages: 50, Chains: 2"
echo "=========================================="

python main/estimate_commensal_static.py \
    --condition Commensal \
    --cultivation Static \
    --n-particles 1000 \
    --n-stages 50 \
    --n-chains 2 \
    --use-exp-init \
    --start-from-day 3 \
    --normalize-data \
    --prior-decay-max 1.0 \
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

# Compare with baseline
echo "Comparing with baseline run..."
BASELINE="_runs/Commensal_Static_20260204_062733"
if [ -d "${BASELINE}" ]; then
    python main/compare_runs.py --run1 "${BASELINE}" --run2 "${OUTPUT_DIR}"
fi

echo "All done! Check ${OUTPUT_DIR}/figures/ for plots."
