#!/bin/bash
# Parameter variants on marinos03: V5 (No gate) + V6 (Wide bounds baseline)
# Dysbiotic HOBIC only, 1 chain, 1000 particles

PROJ="$HOME/IKM_Hiwi/Tmcmc202601"
cd "$PROJ/data_5species/main"

PYTHON=python3
SCRIPT=estimate_reduced_nishioka.py
OUTBASE="$PROJ/data_5species/_runs"

echo "Starting parameter variants on $(hostname) at $(date)"

# V5: No Hill gate (K=0.0 disables gate)
echo "[V5] No Hill gate: K=0.0"
mkdir -p "$OUTBASE/dh_v5_no_gate"
$PYTHON $SCRIPT \
    --condition Dysbiotic --cultivation HOBIC \
    --output-dir "$OUTBASE/dh_v5_no_gate" \
    --n-particles 1000 --n-stages 12 \
    --K-hill 0.0 --n-hill 2.0 \
    --n-chains 1 --n-jobs 12 --checkpoint-every 2 --seed 305 \
    > "$OUTBASE/dh_v5_no_gate.log" 2>&1 &
PID1=$!

# V6: Wide bounds baseline (original bounds, no lambda weight)
echo "[V6] Wide bounds baseline: a35[0,30], a45[0,20], lambda=1.0"
mkdir -p "$OUTBASE/dh_v6_wide_baseline"
$PYTHON $SCRIPT \
    --condition Dysbiotic --cultivation HOBIC \
    --output-dir "$OUTBASE/dh_v6_wide_baseline" \
    --n-particles 1000 --n-stages 12 \
    --K-hill 0.05 --n-hill 4.0 \
    --override-bounds "18:0:30,19:0:20" \
    --lambda-pg 1.0 --lambda-late 1.0 \
    --n-chains 1 --n-jobs 12 --checkpoint-every 2 --seed 306 \
    > "$OUTBASE/dh_v6_wide_baseline.log" 2>&1 &
PID2=$!

echo "PIDs: V5=$PID1, V6=$PID2"
wait $PID1 $PID2
echo "Done at $(date)"
