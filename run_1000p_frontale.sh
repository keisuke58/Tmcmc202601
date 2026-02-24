#!/bin/bash
# 1000 particles TMCMC on frontale01: Commensal Static + Dysbiotic HOBIC
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

# Commensal Static
echo "[CS] Starting Commensal Static (1000p)..."
mkdir -p "$OUTBASE/commensal_static_1000p"
$PYTHON $SCRIPT \
    --condition Commensal --cultivation Static \
    --output-dir "$OUTBASE/commensal_static_1000p" \
    $COMMON_ARGS --seed 100 \
    > "$OUTBASE/commensal_static_1000p.log" 2>&1 &
PID_CS=$!
echo "[CS] PID=$PID_CS"

# Dysbiotic HOBIC
echo "[DH] Starting Dysbiotic HOBIC (1000p)..."
mkdir -p "$OUTBASE/dysbiotic_hobic_1000p"
$PYTHON $SCRIPT \
    --condition Dysbiotic --cultivation HOBIC \
    --output-dir "$OUTBASE/dysbiotic_hobic_1000p" \
    $COMMON_ARGS --seed 101 \
    > "$OUTBASE/dysbiotic_hobic_1000p.log" 2>&1 &
PID_DH=$!
echo "[DH] PID=$PID_DH"

echo ""
echo "Running in background:"
echo "  CS: PID=$PID_CS"
echo "  DH: PID=$PID_DH"
echo ""
echo "Monitor: tail -f $OUTBASE/*_1000p.log"
echo "Check:   ps aux | grep estimate_reduced"

wait $PID_CS $PID_DH
echo ""
echo "Both conditions completed on $(hostname) at $(date)"
