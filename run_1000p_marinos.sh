#!/bin/bash
# 1000 particles TMCMC on marinos01: Commensal HOBIC + Dysbiotic Static
# High-precision run with 12 stages, 2 chains, 12 parallel jobs per chain
# Expected runtime: ~6-12 hours each

PROJ="$HOME/IKM_Hiwi/Tmcmc202601"
cd "$PROJ/data_5species/main"

PYTHON=python3
SCRIPT=estimate_reduced_nishioka.py
OUTBASE="$PROJ/data_5species/_runs"
COMMON_ARGS="--n-particles 1000 --n-stages 12 --K-hill 0.05 --n-hill 4.0 --n-chains 2 --n-jobs 12 --checkpoint-every 2"

echo "Starting 1000p TMCMC on $(hostname) at $(date)"
echo "================================================="

# Commensal HOBIC
echo "[CH] Starting Commensal HOBIC (1000p)..."
mkdir -p "$OUTBASE/commensal_hobic_1000p"
$PYTHON $SCRIPT \
    --condition Commensal --cultivation HOBIC \
    --output-dir "$OUTBASE/commensal_hobic_1000p" \
    $COMMON_ARGS --seed 200 \
    > "$OUTBASE/commensal_hobic_1000p.log" 2>&1 &
PID_CH=$!
echo "[CH] PID=$PID_CH"

# Dysbiotic Static
echo "[DS] Starting Dysbiotic Static (1000p)..."
mkdir -p "$OUTBASE/dysbiotic_static_1000p"
$PYTHON $SCRIPT \
    --condition Dysbiotic --cultivation Static \
    --output-dir "$OUTBASE/dysbiotic_static_1000p" \
    $COMMON_ARGS --seed 201 \
    > "$OUTBASE/dysbiotic_static_1000p.log" 2>&1 &
PID_DS=$!
echo "[DS] PID=$PID_DS"

echo ""
echo "Running in background:"
echo "  CH: PID=$PID_CH"
echo "  DS: PID=$PID_DS"
echo ""
echo "Monitor: tail -f $OUTBASE/*_1000p.log"
echo "Check:   ps aux | grep estimate_reduced"

wait $PID_CH $PID_DS
echo ""
echo "Both conditions completed on $(hostname) at $(date)"
