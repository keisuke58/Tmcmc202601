#!/bin/bash
#PBS -N det_de
#PBS -l nodes=1:ppn=12
#PBS -l walltime=10:00:00
#PBS -q default
#PBS -j oe
#PBS -o ${PBS_JOBNAME}_${PBS_JOBID}.log
#PBS -m ae
#PBS -M nishioka@ikm.uni-hannover.de

# ============================================================
# Deterministic MAP — DE (global) overnight run
# ============================================================
# DE = Differential Evolution: population-based global optimizer
# - popsize=15*n_params (scipy default), workers=-1 (all cores)
# - 20 multi-start LHS for robustness
# - Expected runtime: 2-8 hours per condition
#
# Usage:
#   qsub -l nodes=frontale02:ppn=12 -v CONDITION=Commensal,CULTIVATION=Static det_map_de_job.sh
# ============================================================

set -euo pipefail

CONDITION="${CONDITION:-Dysbiotic}"
CULTIVATION="${CULTIVATION:-HOBIC}"
OPTIMIZER="${OPTIMIZER:-DE}"
NUM_STARTS="${NUM_STARTS:-1}"
MAXITER="${MAXITER:-2000}"
SEED="${SEED:-42}"
START_DAY="${START_DAY:-1}"
K_HILL="${K_HILL:-0.05}"
N_HILL="${N_HILL:-4}"
MAXTIMESTEP="${MAXTIMESTEP:-2500}"
RELIN_THRESHOLD="${RELIN_THRESHOLD:-0.3}"
RELIN_INTERVAL="${RELIN_INTERVAL:-30}"
NJOBS="${NJOBS:-12}"
DE_POPSIZE="${DE_POPSIZE:-30}"
HYBRID="${HYBRID:-1}"
USE_EXP_INIT="${USE_EXP_INIT:-1}"
REPLICATE_SIGMA="${REPLICATE_SIGMA:-0}"

cd /home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/main
PYTHON=python3

TS=$(date +%Y%m%d_%H%M%S)
SHORT="${CONDITION:0:1}${CULTIVATION:0:1}"
OUTDIR="deterministic_results/${SHORT}_${OPTIMIZER}_${NUM_STARTS}starts_${TS}"

echo "=============================================="
echo "Deterministic MAP (DE overnight): ${CONDITION} ${CULTIVATION}"
echo "  Optimizer:    ${OPTIMIZER}"
echo "  Multi-start:  ${NUM_STARTS} (LHS)"
echo "  Max iter:     ${MAXITER}"
echo "  Parallel:     ${NJOBS} workers"
echo "  Node:         $(hostname)"
echo "  Output:       ${OUTDIR}"
echo "  Start:        $(date)"
echo "  PBS Job ID:   ${PBS_JOBID:-local}"

EXTRA_ARGS=""
if [ "${USE_EXP_INIT}" = "1" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --use-exp-init"
    echo "  ExpIC:        ON"
fi
if [ "${REPLICATE_SIGMA}" = "1" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --replicate-sigma"
    echo "  Sigma:        heteroscedastic"
else
    echo "  Sigma:        default (scalar)"
fi
if [ "${HYBRID}" = "1" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --hybrid"
    echo "  Hybrid:       DE→L-BFGS-B (top-5 polish)"
fi
echo "  DE popsize:   ${DE_POPSIZE}×n_params"
echo "=============================================="

$PYTHON estimate_deterministic.py \
    --condition "${CONDITION}" \
    --cultivation "${CULTIVATION}" \
    --optimizer "${OPTIMIZER}" \
    --num-starts "${NUM_STARTS}" \
    --maxiter "${MAXITER}" \
    --seed "${SEED}" \
    --start-from-day "${START_DAY}" \
    --maxtimestep "${MAXTIMESTEP}" \
    --K-hill "${K_HILL}" \
    --n-hill "${N_HILL}" \
    --cov-rel 0.005 \
    --de-popsize "${DE_POPSIZE}" \
    --relinearization-threshold "${RELIN_THRESHOLD}" \
    --min-relinearization-interval "${RELIN_INTERVAL}" \
    --n-jobs "${NJOBS}" \
    --output-dir "${OUTDIR}" \
    ${EXTRA_ARGS}

echo "=============================================="
echo "Deterministic MAP (DE) finished: $(date)"
echo "Results: ${OUTDIR}"
echo "=============================================="
