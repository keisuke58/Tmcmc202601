#!/bin/bash
# Run TMCMC for 3 remaining conditions (CS, CH, DS) in parallel
# Mild-weight settings: 150 particles, 8 stages, K=0.05, n=4
# Expected runtime: ~2-4 hours each

PROJ="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJ/data_5species/main"

PYTHON=python3
SCRIPT=estimate_reduced_nishioka.py
OUTBASE="$PROJ/data_5species/_runs"
COMMON_ARGS="--n-particles 150 --n-stages 8 --K-hill 0.05 --n-hill 4.0 --n-chains 2 --n-jobs 4 --checkpoint-every 2"

echo "Starting 3 TMCMC runs at $(date)"
echo "================================="

# Commensal Static
echo "[CS] Starting Commensal Static..."
$PYTHON $SCRIPT \
    --condition Commensal --cultivation Static \
    --output-dir "$OUTBASE/commensal_static_posterior" \
    $COMMON_ARGS --seed 42 \
    > "$OUTBASE/commensal_static_posterior.log" 2>&1 &
PID_CS=$!
echo "[CS] PID=$PID_CS"

# Commensal HOBIC
echo "[CH] Starting Commensal HOBIC..."
$PYTHON $SCRIPT \
    --condition Commensal --cultivation HOBIC \
    --output-dir "$OUTBASE/commensal_hobic_posterior" \
    $COMMON_ARGS --seed 43 \
    > "$OUTBASE/commensal_hobic_posterior.log" 2>&1 &
PID_CH=$!
echo "[CH] PID=$PID_CH"

# Dysbiotic Static
echo "[DS] Starting Dysbiotic Static..."
$PYTHON $SCRIPT \
    --condition Dysbiotic --cultivation Static \
    --output-dir "$OUTBASE/dysbiotic_static_posterior" \
    $COMMON_ARGS --seed 44 \
    > "$OUTBASE/dysbiotic_static_posterior.log" 2>&1 &
PID_DS=$!
echo "[DS] PID=$PID_DS"

echo ""
echo "All 3 running in background:"
echo "  CS: PID=$PID_CS"
echo "  CH: PID=$PID_CH"
echo "  DS: PID=$PID_DS"
echo ""
echo "Monitor: tail -f $OUTBASE/*_posterior.log"
echo "Check:   ps aux | grep estimate_reduced"

wait $PID_CS $PID_CH $PID_DS
echo ""
echo "All 3 conditions completed at $(date)"
