#!/bin/bash
# Run all 4 TMCMC conditions with DeepONet surrogate (klempt_fem env).
# Requires: JAX, Equinox, DeepONet checkpoints for each condition.
#
# Usage:
#   ./run_all_4conditions_deeponet.sh [quick|full]
#   quick: 200 particles, 8 stages (validation)
#   full:  1000 particles, 30 stages (production)

set -e

BASEDIR="/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species"
PYTHON="${HOME}/.pyenv/versions/miniconda3-latest/envs/klempt_fem/bin/python"
SCRIPT="main/estimate_reduced_nishioka.py"

MODE="${1:-full}"
if [ "$MODE" = "quick" ]; then
    N_PARTICLES=200
    N_STAGES=8
    echo "Quick mode: ${N_PARTICLES} particles, ${N_STAGES} stages"
else
    N_PARTICLES=1000
    N_STAGES=30
    echo "Full mode: ${N_PARTICLES} particles, ${N_STAGES} stages"
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="${BASEDIR}/_runs/batch_deeponet_${TIMESTAMP}"
mkdir -p "${LOGDIR}"

echo "=========================================="
echo "TMCMC + DeepONet: 4 conditions"
echo "Timestamp: ${TIMESTAMP}"
echo "Log dir: ${LOGDIR}"
echo "Python: ${PYTHON}"
echo "=========================================="

cd "${BASEDIR}"

# Run batch with --use-deeponet (klempt_fem has JAX/Equinox)
# NOTE: --n-jobs 1 required (JAX/DeepONet not pickleable for multiprocessing)
echo "[1/1] Running batch: Commensal_Static, Commensal_HOBIC, Dysbiotic_Static, Dysbiotic_HOBIC"
${PYTHON} "${SCRIPT}" \
    --batch all \
    --use-deeponet \
    --n-particles ${N_PARTICLES} \
    --n-stages ${N_STAGES} \
    --n-chains 2 \
    --n-jobs 1 \
    --use-exp-init \
    --start-from-day 1 \
    --normalize-data \
    --lambda-pg 2.0 \
    --lambda-late 1.5 \
    --seed 42 \
    --no-notify-slack \
    --debug-level INFO \
    2>&1 | tee "${LOGDIR}/batch.log"

echo ""
echo "=========================================="
echo "Batch complete. Results in data_5species/_runs/"
echo "=========================================="
