#!/bin/bash
# Run improved TMCMC estimation with more particles
# Server: fifawc (12 cores)
# Expected runtime: ~6-8 hours

set -e

# Activate conda environment if needed
source ~/.bashrc 2>/dev/null || true

cd /home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species

# Create output directory name with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="_runs/improved_v1_${TIMESTAMP}"

echo "=========================================="
echo "Starting improved TMCMC estimation"
echo "Timestamp: ${TIMESTAMP}"
echo "Output: ${OUTPUT_DIR}"
echo "Particles: 1000, Stages: 50, Chains: 2"
echo "=========================================="

# Run with increased particles and stages
python main/estimate_commensal_static.py \
    --condition Commensal \
    --cultivation Static \
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
echo "Results saved to: ${OUTPUT_DIR}"
echo "=========================================="

# Generate all figures
echo "Generating all figures..."
python main/generate_all_figures.py --run-dir "${OUTPUT_DIR}" --n-samples 100

echo "All done! Check ${OUTPUT_DIR}/figures/ for plots."
