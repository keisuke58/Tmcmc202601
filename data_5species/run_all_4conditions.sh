#!/bin/bash
# Run all 4 TMCMC conditions in parallel across 2 servers
# frontale02: Dysbiotic_Static + Dysbiotic_HOBIC  (priority conditions)
# marinos03:  Commensal_Static + Commensal_HOBIC  (re-run with 1000 particles)
#
# Settings: 1000 particles, 30 stages, 2 chains, 12 workers each
# Each server runs 2 conditions (12+12 = 24 cores)

set -e

BASEDIR="/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="${BASEDIR}/_runs/batch_${TIMESTAMP}"
mkdir -p "${LOGDIR}"

echo "=========================================="
echo "Launching all 4 conditions"
echo "Timestamp: ${TIMESTAMP}"
echo "Log dir: ${LOGDIR}"
echo "=========================================="

# ── frontale02: Dysbiotic conditions (highest priority) ──
echo "[1/4] Dysbiotic_Static on frontale02..."
ssh frontale02 "cd ${BASEDIR} && nohup python main/estimate_commensal_static.py \
    --condition Dysbiotic \
    --cultivation Static \
    --n-particles 1000 \
    --n-stages 30 \
    --n-chains 2 \
    --use-exp-init \
    --start-from-day 1 \
    --normalize-data \
    --n-jobs 12 \
    --seed 42 \
    --output-dir '_runs/Dysbiotic_Static_${TIMESTAMP}' \
    --debug-level INFO \
    > '${LOGDIR}/Dysbiotic_Static.log' 2>&1 &"
echo "  -> PID launched on frontale02"

sleep 2

echo "[2/4] Dysbiotic_HOBIC on frontale02..."
ssh frontale02 "cd ${BASEDIR} && nohup python main/estimate_commensal_static.py \
    --condition Dysbiotic \
    --cultivation HOBIC \
    --n-particles 1000 \
    --n-stages 30 \
    --n-chains 2 \
    --use-exp-init \
    --start-from-day 1 \
    --normalize-data \
    --n-jobs 12 \
    --seed 42 \
    --output-dir '_runs/Dysbiotic_HOBIC_${TIMESTAMP}' \
    --debug-level INFO \
    > '${LOGDIR}/Dysbiotic_HOBIC.log' 2>&1 &"
echo "  -> PID launched on frontale02"

sleep 2

# ── marinos03: Commensal conditions ──
echo "[3/4] Commensal_Static on marinos03..."
ssh marinos03 "cd ${BASEDIR} && nohup python main/estimate_commensal_static.py \
    --condition Commensal \
    --cultivation Static \
    --n-particles 1000 \
    --n-stages 30 \
    --n-chains 2 \
    --use-exp-init \
    --start-from-day 3 \
    --normalize-data \
    --n-jobs 12 \
    --seed 42 \
    --output-dir '_runs/Commensal_Static_${TIMESTAMP}' \
    --debug-level INFO \
    > '${LOGDIR}/Commensal_Static.log' 2>&1 &"
echo "  -> PID launched on marinos03"

sleep 2

echo "[4/4] Commensal_HOBIC on marinos03..."
ssh marinos03 "cd ${BASEDIR} && nohup python main/estimate_commensal_static.py \
    --condition Commensal \
    --cultivation HOBIC \
    --n-particles 1000 \
    --n-stages 30 \
    --n-chains 2 \
    --use-exp-init \
    --start-from-day 3 \
    --normalize-data \
    --n-jobs 12 \
    --seed 42 \
    --output-dir '_runs/Commensal_HOBIC_${TIMESTAMP}' \
    --debug-level INFO \
    > '${LOGDIR}/Commensal_HOBIC.log' 2>&1 &"
echo "  -> PID launched on marinos03"

echo ""
echo "=========================================="
echo "All 4 conditions launched!"
echo "=========================================="
echo ""
echo "Monitor progress:"
echo "  tail -f ${LOGDIR}/Dysbiotic_Static.log"
echo "  tail -f ${LOGDIR}/Dysbiotic_HOBIC.log"
echo "  tail -f ${LOGDIR}/Commensal_Static.log"
echo "  tail -f ${LOGDIR}/Commensal_HOBIC.log"
echo ""
echo "Check server load:"
echo "  ssh frontale02 uptime"
echo "  ssh marinos03 uptime"
