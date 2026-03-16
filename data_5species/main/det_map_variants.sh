#!/bin/bash
#PBS -N det_var
#PBS -l nodes=1:ppn=1
#PBS -l walltime=02:00:00
#PBS -q default
#PBS -j oe
#PBS -o ${PBS_JOBNAME}_${PBS_JOBID}.log
#PBS -m ae
#PBS -M nishioka@ikm.uni-hannover.de

# ============================================================
# Deterministic MAP — Multiple optimizer variants
# ============================================================
# Usage:
#   qsub -l nodes=frontale02:ppn=1 -v CONDITION=Commensal,CULTIVATION=Static,VARIANT=basinhopping det_map_variants.sh
# ============================================================

set -euo pipefail

CONDITION="${CONDITION:-Dysbiotic}"
CULTIVATION="${CULTIVATION:-HOBIC}"
VARIANT="${VARIANT:-basinhopping}"
SEED="${SEED:-42}"

cd /home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/main
PYTHON=python3

TS=$(date +%Y%m%d_%H%M%S)
SHORT="${CONDITION:0:1}${CULTIVATION:0:1}"
OUTDIR="deterministic_results/${SHORT}_${VARIANT}_${TS}"

echo "=============================================="
echo "Det MAP Variant: ${CONDITION} ${CULTIVATION} [${VARIANT}]"
echo "  Node: $(hostname)"
echo "  Output: ${OUTDIR}"
echo "  Start: $(date)"
echo "=============================================="

case "${VARIANT}" in
    basinhopping)
        # Basin-Hopping: random perturbation + L-BFGS-B local search × 50 hops
        # Good at escaping local minima without DE's population overhead
        $PYTHON estimate_deterministic.py \
            --condition "${CONDITION}" --cultivation "${CULTIVATION}" \
            --optimizer basinhopping --num-starts 1 --maxiter 500 \
            --seed "${SEED}" --K-hill 0.05 --n-hill 4.0 --cov-rel 0.005 \
            --relinearization-threshold 0.3 --min-relinearization-interval 30 \
            --output-dir "${OUTDIR}"
        ;;
    powell)
        # Powell: gradient-free conjugate direction × 10 LHS
        # Validates that L-BFGS-B gradient approximation is not misleading
        $PYTHON estimate_deterministic.py \
            --condition "${CONDITION}" --cultivation "${CULTIVATION}" \
            --optimizer Powell --num-starts 10 --maxiter 2000 \
            --seed "${SEED}" --K-hill 0.05 --n-hill 4.0 --cov-rel 0.005 \
            --relinearization-threshold 0.3 --min-relinearization-interval 30 \
            --output-dir "${OUTDIR}"
        ;;
    dual_annealing)
        # Dual Annealing: simulated annealing + local refinement
        # Best theoretical guarantee for global optimum in bounded domain
        $PYTHON estimate_deterministic.py \
            --condition "${CONDITION}" --cultivation "${CULTIVATION}" \
            --optimizer dual_annealing --num-starts 1 --maxiter 1000 \
            --seed "${SEED}" --K-hill 0.05 --n-hill 4.0 --cov-rel 0.005 \
            --relinearization-threshold 0.3 --min-relinearization-interval 30 \
            --output-dir "${OUTDIR}"
        ;;
    *)
        echo "Unknown variant: ${VARIANT}"
        exit 1
        ;;
esac

echo "=============================================="
echo "Det MAP Variant finished: $(date)"
echo "Results: ${OUTDIR}"
echo "=============================================="
