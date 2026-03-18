#!/bin/bash
#PBS -N tmcmc-joint
#PBS -l nodes=1:ppn=24
#PBS -l walltime=168:00:00
#PBS -q default
#PBS -j oe
#PBS -o ${PBS_JOBNAME}_${PBS_JOBID}.log
#PBS -m ae
#PBS -M nishioka@ikm.uni-hannover.de

# ============================================================
# Joint 4-Condition TMCMC PBS Job Script
# ============================================================
#
# Usage:
#   qsub tmcmc_joint_job.sh
#   qsub -v NPART=500 tmcmc_joint_job.sh
#   qsub -v NPART=200,MULTICHANNEL=1 tmcmc_joint_job.sh
#   qsub -l nodes=frontale04:ppn=12 tmcmc_joint_job.sh
# ============================================================

set -euo pipefail

# --- Defaults (override via -v) ---
NPART="${NPART:-200}"
NSTAGES="${NSTAGES:-30}"
NCHAINS="${NCHAINS:-4}"
NJOBS="${NJOBS:-24}"
SEED="${SEED:-42}"
MULTICHANNEL="${MULTICHANNEL:-0}"
N_MUTATION_STEPS="${N_MUTATION_STEPS:-10}"
USE_DEEPONET="${USE_DEEPONET:-0}"

# Condition weights (default: uniform)
WEIGHT_CS="${WEIGHT_CS:-1.0}"
WEIGHT_CH="${WEIGHT_CH:-1.0}"
WEIGHT_DS="${WEIGHT_DS:-1.0}"
WEIGHT_DH="${WEIGHT_DH:-1.0}"

# --- Environment ---
cd /home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/main
PYTHON=python3

# --- Output directory (override via OUTDIR env var) ---
if [ -z "${OUTDIR:-}" ]; then
    TS=$(date +%Y%m%d_%H%M%S)
    OUTDIR="_runs/joint_4cond_${NPART}p_${TS}"
fi

echo "=============================================="
echo "Joint 4-Condition TMCMC"
echo "  Particles: ${NPART}, Chains: ${NCHAINS}, Stages: ${NSTAGES}"
echo "  Node: $(hostname), PPNs: ${NJOBS}"
echo "  Output: ${OUTDIR}"
echo "  Start: $(date)"
echo "  PBS Job ID: ${PBS_JOBID:-local}"
echo "=============================================="

EXTRA_ARGS=""
if [ "${MULTICHANNEL}" = "1" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --multichannel"
    echo "  Multi-channel likelihood: ON"
fi
if [ "${USE_DEEPONET}" = "1" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --use-deeponet"
    echo "  DeepONet surrogate: ON"
fi

$PYTHON estimate_joint_nishioka.py \
    --n-particles "${NPART}" \
    --n-stages "${NSTAGES}" \
    --n-chains "${NCHAINS}" \
    --n-jobs "${NJOBS}" \
    --n-mutation-steps "${N_MUTATION_STEPS}" \
    --seed "${SEED}" \
    --weight-cs "${WEIGHT_CS}" \
    --weight-ch "${WEIGHT_CH}" \
    --weight-ds "${WEIGHT_DS}" \
    --weight-dh "${WEIGHT_DH}" \
    --output-dir "${OUTDIR}" \
    ${EXTRA_ARGS}

echo "=============================================="
echo "Joint TMCMC finished: $(date)"
echo "Results: ${OUTDIR}"
echo "=============================================="
