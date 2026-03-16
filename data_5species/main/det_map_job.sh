#!/bin/bash
#PBS -N det_map
#PBS -l nodes=1:ppn=12
#PBS -l walltime=01:00:00
#PBS -q default
#PBS -j oe
#PBS -o ${PBS_JOBNAME}_${PBS_JOBID}.log
#PBS -m ae
#PBS -M nishioka@ikm.uni-hannover.de

# ============================================================
# Deterministic MAP Estimation — "最強設定"
# ============================================================
# Strategy: DE (global) × 20 LHS multi-start × adaptive relinearization
#
# DE is the best choice because:
#   - Basin sensitivity confirmed multiple attractors (CS: 49/51 jump)
#   - 20D parameter space needs global exploration
#   - polish=True refines with L-BFGS-B at the end
#   - 20 multi-start LHS covers prior volume uniformly
#
# Usage:
#   # Single condition:
#   qsub -l nodes=frontale03:ppn=12 -v CONDITION=Commensal,CULTIVATION=Static det_map_job.sh
#
#   # All 4 conditions (parallel on 4 nodes):
#   qsub -l nodes=frontale03:ppn=12 -v CONDITION=Commensal,CULTIVATION=Static  det_map_job.sh
#   qsub -l nodes=frontale04:ppn=12 -v CONDITION=Commensal,CULTIVATION=HOBIC   det_map_job.sh
#   qsub -l nodes=marinos01:ppn=12  -v CONDITION=Dysbiotic,CULTIVATION=Static  det_map_job.sh
#   qsub -l nodes=marinos03:ppn=12  -v CONDITION=Dysbiotic,CULTIVATION=HOBIC   det_map_job.sh
# ============================================================

set -euo pipefail

# --- Defaults (override via -v) ---
CONDITION="${CONDITION:-Dysbiotic}"
CULTIVATION="${CULTIVATION:-HOBIC}"
# Strategy: L-BFGS-B × 10 LHS multi-start (~5 min total)
#   - Each L-BFGS-B converges in ~500 evals (~30s)
#   - 10 starts from LHS covers prior volume
#   - Best of 10 = robust MAP without DE's 300× overhead
# For extra safety: OPTIMIZER=basinhopping, NUM_STARTS=1, MAXITER=100
OPTIMIZER="${OPTIMIZER:-L-BFGS-B}"
NUM_STARTS="${NUM_STARTS:-10}"
MAXITER="${MAXITER:-2000}"
SEED="${SEED:-42}"
START_DAY="${START_DAY:-1}"
K_HILL="${K_HILL:-0.05}"
N_HILL="${N_HILL:-4}"
MAXTIMESTEP="${MAXTIMESTEP:-2500}"
RELIN_THRESHOLD="${RELIN_THRESHOLD:-0.3}"
RELIN_INTERVAL="${RELIN_INTERVAL:-30}"
NJOBS="${NJOBS:-12}"

# --- Environment ---
cd /home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/main
PYTHON=python3

# --- Output directory ---
TS=$(date +%Y%m%d_%H%M%S)
SHORT="${CONDITION:0:1}${CULTIVATION:0:1}"
OUTDIR="deterministic_results/${SHORT}_${OPTIMIZER}_${NUM_STARTS}starts_${TS}"

echo "=============================================="
echo "Deterministic MAP: ${CONDITION} ${CULTIVATION}"
echo "  Optimizer:    ${OPTIMIZER}"
echo "  Multi-start:  ${NUM_STARTS} (LHS)"
echo "  Max iter:     ${MAXITER}"
echo "  Relin thresh: ${RELIN_THRESHOLD}"
echo "  Parallel:     ${NJOBS} workers"
echo "  Node:         $(hostname)"
echo "  Output:       ${OUTDIR}"
echo "  Start:        $(date)"
echo "  PBS Job ID:   ${PBS_JOBID:-local}"
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
    --relinearization-threshold "${RELIN_THRESHOLD}" \
    --min-relinearization-interval "${RELIN_INTERVAL}" \
    --n-jobs "${NJOBS}" \
    --output-dir "${OUTDIR}"

echo "=============================================="
echo "Deterministic MAP finished: $(date)"
echo "Results: ${OUTDIR}"
echo "=============================================="
