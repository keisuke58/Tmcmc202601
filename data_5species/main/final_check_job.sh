#!/bin/bash
#PBS -N final_chk
#PBS -l nodes=1:ppn=12
#PBS -l walltime=02:00:00
#PBS -q default
#PBS -j oe
#PBS -o ${PBS_JOBNAME}_${PBS_JOBID}.log
#PBS -m ae
#PBS -M nishioka@ikm.uni-hannover.de

# ============================================================
# Final Validation Check: LOO-CV + Sensitivity + θ Comparison
# ============================================================
# Runs after det_MAP results are available.
#
# Usage:
#   # All 4 conditions (auto-detect latest results):
#   qsub -l nodes=frontale02:ppn=12 final_check_job.sh
#
#   # Single condition:
#   qsub -l nodes=frontale02:ppn=12 -v CONDITION=Commensal,CULTIVATION=Static final_check_job.sh
# ============================================================

set -euo pipefail

cd /home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/main
PYTHON=python3

CONDITION="${CONDITION:-}"
CULTIVATION="${CULTIVATION:-}"

echo "=============================================="
echo "Final Validation Check"
echo "  Node:         $(hostname)"
echo "  Start:        $(date)"
echo "  PBS Job ID:   ${PBS_JOBID:-local}"
echo "=============================================="

if [ -z "${CONDITION}" ]; then
    echo "Running all 4 conditions..."
    $PYTHON final_check.py --all
else
    echo "Running ${CONDITION} ${CULTIVATION}..."
    $PYTHON final_check.py --condition "${CONDITION}" --cultivation "${CULTIVATION}"
fi

echo "=============================================="
echo "Final Check finished: $(date)"
echo "=============================================="
