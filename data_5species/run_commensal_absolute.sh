#!/bin/bash
# Launch Commensal_Static + Commensal_HOBIC WITHOUT --normalize-data (absolute volume fit)
# Run this when a server frees up (frontale02/marinos03 after normalized runs finish)
#
# Usage: bash run_commensal_absolute.sh <server_name>
# Example: bash run_commensal_absolute.sh frontale02

set -e

SERVER="${1:?Usage: bash run_commensal_absolute.sh <server_name>}"
BASEDIR="/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species"
TIMESTAMP="20260208_005500"
LOGDIR="${BASEDIR}/_runs/batch_absolute_${TIMESTAMP}"

echo "Launching Commensal absolute-volume runs on ${SERVER}..."

echo "[1/2] Commensal_Static (absolute) on ${SERVER}..."
ssh ${SERVER} "cd ${BASEDIR} && nohup python main/estimate_commensal_static.py \
    --condition Commensal \
    --cultivation Static \
    --n-particles 1000 \
    --n-stages 30 \
    --n-chains 2 \
    --use-exp-init \
    --start-from-day 3 \
    --use-absolute-volume \
    --n-jobs 12 \
    --seed 42 \
    --output-dir '_runs/Commensal_Static_abs_${TIMESTAMP}' \
    --debug-level INFO \
    > '${LOGDIR}/Commensal_Static_abs.log' 2>&1 &"
echo "  -> launched"

sleep 2

echo "[2/2] Commensal_HOBIC (absolute) on ${SERVER}..."
ssh ${SERVER} "cd ${BASEDIR} && nohup python main/estimate_commensal_static.py \
    --condition Commensal \
    --cultivation HOBIC \
    --n-particles 1000 \
    --n-stages 30 \
    --n-chains 2 \
    --use-exp-init \
    --start-from-day 3 \
    --use-absolute-volume \
    --n-jobs 12 \
    --seed 42 \
    --output-dir '_runs/Commensal_HOBIC_abs_${TIMESTAMP}' \
    --debug-level INFO \
    > '${LOGDIR}/Commensal_HOBIC_abs.log' 2>&1 &"
echo "  -> launched"

echo ""
echo "Monitor:"
echo "  tail -f ${LOGDIR}/Commensal_Static_abs.log"
echo "  tail -f ${LOGDIR}/Commensal_HOBIC_abs.log"
