#!/bin/bash
# Parameter variants on frontale04: V3 (Tight bounds) + V4 (Strong Pg weight)
# Dysbiotic HOBIC only, 1 chain, 1000 particles

PROJ="$HOME/IKM_Hiwi/Tmcmc202601"
cd "$PROJ/data_5species/main"

PYTHON=python3
SCRIPT=estimate_reduced_nishioka.py
OUTBASE="$PROJ/data_5species/_runs"

echo "Starting parameter variants on $(hostname) at $(date)"

# V3: Tight bounds (a35 [0,3], a45 [0,3])
echo "[V3] Tight bounds: a35/a45 [0,3]"
mkdir -p "$OUTBASE/dh_v3_tight_bounds"
$PYTHON $SCRIPT \
    --condition Dysbiotic --cultivation HOBIC \
    --output-dir "$OUTBASE/dh_v3_tight_bounds" \
    --n-particles 1000 --n-stages 12 \
    --K-hill 0.05 --n-hill 4.0 \
    --override-bounds "18:0:3,19:0:3" \
    --n-chains 1 --n-jobs 12 --checkpoint-every 2 --seed 303 \
    > "$OUTBASE/dh_v3_tight_bounds.log" 2>&1 &
PID1=$!

# V4: Strong Pg weight (lambda_pg=3.0)
echo "[V4] Strong Pg weight: lambda_pg=3.0"
mkdir -p "$OUTBASE/dh_v4_strong_pg"
$PYTHON $SCRIPT \
    --condition Dysbiotic --cultivation HOBIC \
    --output-dir "$OUTBASE/dh_v4_strong_pg" \
    --n-particles 1000 --n-stages 12 \
    --K-hill 0.05 --n-hill 4.0 \
    --lambda-pg 3.0 --lambda-late 2.0 \
    --n-chains 1 --n-jobs 12 --checkpoint-every 2 --seed 304 \
    > "$OUTBASE/dh_v4_strong_pg.log" 2>&1 &
PID2=$!

echo "PIDs: V3=$PID1, V4=$PID2"
wait $PID1 $PID2
echo "Done at $(date)"
