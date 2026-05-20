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
#   # With expIC (recommended):
#   qsub -v CONDITION=Dysbiotic,CULTIVATION=HOBIC,NPART=1000,USE_EXP_INIT=1 tmcmc_job.sh
#
#   # All 4 conditions at once (REPLICATE_SIGMA=0 is default and correct):
#   for c in "Commensal,Static" "Commensal,HOBIC" "Dysbiotic,Static" "Dysbiotic,HOBIC"; do
#     IFS=',' read -r COND CULT <<< "$c"
#     qsub -v CONDITION=$COND,CULTIVATION=$CULT,NPART=500,USE_EXP_INIT=1 tmcmc_job.sh
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
MULTICHANNEL="${MULTICHANNEL:-0}"
N_MUTATION_STEPS="${N_MUTATION_STEPS:-}"
SIGN_PRIOR="${SIGN_PRIOR:-0}"
SIGN_LAMBDA="${SIGN_LAMBDA:-0.1}"
BC_LAMBDA="${BC_LAMBDA:-0}"

# --- Environment ---
cd /home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/main
PYTHON=python3

# --- Output directory ---
TS=$(date +%Y%m%d_%H%M%S)
SHORT="${CONDITION:0:1}${CULTIVATION:0:1}"
SUFFIX="${NPART}p"
[ "${USE_EXP_INIT}" = "1" ] && SUFFIX="${SUFFIX}_expIC"
[ "${REPLICATE_SIGMA}" = "1" ] && SUFFIX="${SUFFIX}_repSigma"
[ "${MULTICHANNEL}" = "1" ] && SUFFIX="${SUFFIX}_mc"
[ "${SIGN_PRIOR}" = "1" ] && SUFFIX="${SUFFIX}_sp${SIGN_LAMBDA}"
OUTDIR="_runs/${SHORT}_${SUFFIX}_${NCHAINS}ch_${TS}"

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
else
    EXTRA_ARGS="${EXTRA_ARGS} --no-replicate-sigma"
    echo "  Using uniform sigma (--no-replicate-sigma)"
fi
if [ "${MULTICHANNEL}" = "1" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --multichannel"
    echo "  Multi-channel likelihood enabled"
fi
if [ -n "${N_MUTATION_STEPS}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --n-mutation-steps ${N_MUTATION_STEPS}"
    echo "  Mutation steps: ${N_MUTATION_STEPS}"
fi
if [ "${SIGN_PRIOR}" = "1" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --sign-prior --sign-lambda ${SIGN_LAMBDA}"
    echo "  Sign prior enabled (lam=${SIGN_LAMBDA})"
fi
if [ "$(echo "${BC_LAMBDA} > 0" | bc -l 2>/dev/null)" = "1" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --bc-lambda ${BC_LAMBDA}"
    echo "  Bray-Curtis penalty enabled (lam=${BC_LAMBDA})"
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
