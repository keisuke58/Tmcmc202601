#!/bin/bash
#PBS -N tmcmc
#PBS -l nodes=1:ppn=12
#PBS -l walltime=168:00:00
#PBS -q default
#PBS -j oe
#PBS -o ${PBS_JOBNAME}_${PBS_JOBID}.log
#PBS -m ae
#PBS -M nishioka@ikm.uni-hannover.de

# ============================================================
# TMCMC PBS Job Script (PBS/Torque on IKM cluster)
# ============================================================
#
# Usage:
#   # Single condition:
#   qsub -v CONDITION=Commensal,CULTIVATION=Static,NPART=1000 tmcmc_job.sh
#
#   # With custom options:
#   qsub -v CONDITION=Dysbiotic,CULTIVATION=HOBIC,NPART=500,NCHAINS=1,SEED=123 tmcmc_job.sh
#
#   # On specific node:
#   qsub -l nodes=frontale04:ppn=12 -v CONDITION=Commensal,CULTIVATION=Static,NPART=1000 tmcmc_job.sh
#
#   # With expIC + heteroscedastic sigma:
#   qsub -v CONDITION=Dysbiotic,CULTIVATION=HOBIC,NPART=1000,USE_EXP_INIT=1,REPLICATE_SIGMA=1 tmcmc_job.sh
#
#   # All 4 conditions at once:
#   for c in "Commensal,Static" "Commensal,HOBIC" "Dysbiotic,Static" "Dysbiotic,HOBIC"; do
#     IFS=',' read -r COND CULT <<< "$c"
#     qsub -v CONDITION=$COND,CULTIVATION=$CULT,NPART=1000,USE_EXP_INIT=1,REPLICATE_SIGMA=1 tmcmc_job.sh
#   done
# ============================================================

set -euo pipefail

# --- Defaults (override via -v) ---
CONDITION="${CONDITION:-Dysbiotic}"
CULTIVATION="${CULTIVATION:-HOBIC}"
NPART="${NPART:-1000}"
NSTAGES="${NSTAGES:-30}"
NCHAINS="${NCHAINS:-2}"
NJOBS="${NJOBS:-12}"
SEED="${SEED:-42}"
START_DAY="${START_DAY:-1}"
K_HILL="${K_HILL:-0.05}"
N_HILL="${N_HILL:-4}"
CHECKPOINT="${CHECKPOINT:-5}"
USE_EXP_INIT="${USE_EXP_INIT:-0}"
REPLICATE_SIGMA="${REPLICATE_SIGMA:-0}"
N_MUTATION_STEPS="${N_MUTATION_STEPS:-}"

# --- Environment ---
cd /home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/main
PYTHON=python3

# --- Output directory ---
TS=$(date +%Y%m%d_%H%M%S)
SHORT="${CONDITION:0:1}${CULTIVATION:0:1}"
if [ "${USE_EXP_INIT}" = "1" ] && [ "${REPLICATE_SIGMA}" = "1" ]; then
    OUTDIR="_runs/${SHORT}_${NPART}p_expIC_repSigma_${TS}"
elif [ "${USE_EXP_INIT}" = "1" ]; then
    OUTDIR="_runs/${SHORT}_${NPART}p_expIC_${TS}"
elif [ "${REPLICATE_SIGMA}" = "1" ]; then
    OUTDIR="_runs/${SHORT}_${NPART}p_repSigma_${TS}"
else
    OUTDIR="_runs/${SHORT}_${NPART}p_${NCHAINS}ch_${TS}"
fi

echo "=============================================="
echo "TMCMC Job: ${CONDITION} ${CULTIVATION}"
echo "  Particles: ${NPART}, Chains: ${NCHAINS}, Stages: ${NSTAGES}"
echo "  Node: $(hostname), PPNs: ${NJOBS}"
echo "  Output: ${OUTDIR}"
echo "  Start: $(date)"
echo "  PBS Job ID: ${PBS_JOBID:-local}"
echo "=============================================="

EXTRA_ARGS=""
if [ "${USE_EXP_INIT}" = "1" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --use-exp-init"
    echo "  Using experimental Day 1 IC (--use-exp-init)"
fi
if [ "${REPLICATE_SIGMA}" = "1" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --replicate-sigma"
    echo "  Using heteroscedastic sigma from replicates (--replicate-sigma)"
fi
if [ -n "${N_MUTATION_STEPS}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --n-mutation-steps ${N_MUTATION_STEPS}"
    echo "  Mutation steps: ${N_MUTATION_STEPS}"
fi

$PYTHON estimate_reduced_nishioka.py \
    --condition "${CONDITION}" \
    --cultivation "${CULTIVATION}" \
    --n-particles "${NPART}" \
    --n-stages "${NSTAGES}" \
    --n-chains "${NCHAINS}" \
    --n-jobs "${NJOBS}" \
    --start-from-day "${START_DAY}" \
    --K-hill "${K_HILL}" \
    --n-hill "${N_HILL}" \
    --checkpoint-every "${CHECKPOINT}" \
    --seed "${SEED}" \
    --output-dir "${OUTDIR}" \
    ${EXTRA_ARGS}

echo "=============================================="
echo "TMCMC Job finished: $(date)"
echo "Results: ${OUTDIR}"
echo "=============================================="
