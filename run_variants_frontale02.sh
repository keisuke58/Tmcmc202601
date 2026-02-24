#!/bin/bash
# Parameter variants on frontale02: V1 (Sharp gate) + V2 (Soft gate)
# Dysbiotic HOBIC only, 1 chain, 1000 particles

PROJ="$HOME/IKM_Hiwi/Tmcmc202601"
cd "$PROJ/data_5species/main"

PYTHON=python3
SCRIPT=estimate_reduced_nishioka.py
OUTBASE="$PROJ/data_5species/_runs"

echo "Starting parameter variants on $(hostname) at $(date)"

# V1: Sharp gate (K=0.01, n=6)
echo "[V1] Sharp gate: K=0.01, n=6"
mkdir -p "$OUTBASE/dh_v1_sharp_gate"
$PYTHON $SCRIPT \
    --condition Dysbiotic --cultivation HOBIC \
    --output-dir "$OUTBASE/dh_v1_sharp_gate" \
    --n-particles 1000 --n-stages 12 \
    --K-hill 0.01 --n-hill 6.0 \
    --n-chains 1 --n-jobs 12 --checkpoint-every 2 --seed 301 \
    > "$OUTBASE/dh_v1_sharp_gate.log" 2>&1 &
PID1=$!

# V2: Soft gate (K=0.10, n=2)
echo "[V2] Soft gate: K=0.10, n=2"
mkdir -p "$OUTBASE/dh_v2_soft_gate"
$PYTHON $SCRIPT \
    --condition Dysbiotic --cultivation HOBIC \
    --output-dir "$OUTBASE/dh_v2_soft_gate" \
    --n-particles 1000 --n-stages 12 \
    --K-hill 0.10 --n-hill 2.0 \
    --n-chains 1 --n-jobs 12 --checkpoint-every 2 --seed 302 \
    > "$OUTBASE/dh_v2_soft_gate.log" 2>&1 &
PID2=$!

echo "PIDs: V1=$PID1, V2=$PID2"
wait $PID1 $PID2
echo "Done at $(date)"
