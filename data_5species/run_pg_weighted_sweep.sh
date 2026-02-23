#!/bin/bash
# =============================================================================
# P. gingivalis weighted likelihood + Hill gating sweep
# SSH-safe: all runs use nohup, logs saved per-run
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MAIN="$SCRIPT_DIR/main/estimate_reduced_nishioka.py"
SWEEP_DIR="$SCRIPT_DIR/_runs/sweep_pg_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SWEEP_DIR"

export PYTHONPATH="/home/nishioka/IKM_Hiwi/Tmcmc202601/tmcmc/program2602:/home/nishioka/IKM_Hiwi/Tmcmc202601/tmcmc:/home/nishioka/IKM_Hiwi/Tmcmc202601:${PYTHONPATH:-}"

# Common args
COMMON="--condition Dysbiotic --cultivation HOBIC --n-particles 500 --n-stages 30 --n-chains 2 --seed 42 --compute-evidence"

echo "=========================================="
echo " P.g. Weighted Likelihood + Hill Sweep"
echo " Started: $(date)"
echo " Sweep dir: $SWEEP_DIR"
echo "=========================================="

PIDS=""

# -------------------------------------------------------
# RUN 1: Baseline — no Hill, no weighting
# -------------------------------------------------------
TAG="dh_baseline"
OUT="$SWEEP_DIR/$TAG"
echo "[$(date +%H:%M:%S)] Launching $TAG ..."
nohup python3 "$MAIN" $COMMON \
    --K-hill 0.0 --n-hill 2 \
    --output-dir "$OUT" \
    > "$SWEEP_DIR/${TAG}.log" 2>&1 &
PID=$!; PIDS="$PIDS $PID"
echo "  PID=$PID"

# -------------------------------------------------------
# RUN 2: Hill only — K_hill=0.05, no weighting
# -------------------------------------------------------
TAG="dh_hill005"
OUT="$SWEEP_DIR/$TAG"
echo "[$(date +%H:%M:%S)] Launching $TAG ..."
nohup python3 "$MAIN" $COMMON \
    --K-hill 0.05 --n-hill 2 \
    --output-dir "$OUT" \
    > "$SWEEP_DIR/${TAG}.log" 2>&1 &
PID=$!; PIDS="$PIDS $PID"
echo "  PID=$PID"

# -------------------------------------------------------
# RUN 3: Hill + Weighted — K_hill=0.05, lambda_pg=5, lambda_late=3
# -------------------------------------------------------
TAG="dh_hill005_wpg5_wlate3"
OUT="$SWEEP_DIR/$TAG"
echo "[$(date +%H:%M:%S)] Launching $TAG ..."
nohup python3 "$MAIN" $COMMON \
    --K-hill 0.05 --n-hill 2 \
    --lambda-pg 5 --lambda-late 3 --n-late 2 \
    --output-dir "$OUT" \
    > "$SWEEP_DIR/${TAG}.log" 2>&1 &
PID=$!; PIDS="$PIDS $PID"
echo "  PID=$PID"

# -------------------------------------------------------
# RUN 4: Hill scan — K_hill=0.02, weighted
# -------------------------------------------------------
TAG="dh_hill002_wpg5_wlate3"
OUT="$SWEEP_DIR/$TAG"
echo "[$(date +%H:%M:%S)] Launching $TAG ..."
nohup python3 "$MAIN" $COMMON \
    --K-hill 0.02 --n-hill 2 \
    --lambda-pg 5 --lambda-late 3 --n-late 2 \
    --output-dir "$OUT" \
    > "$SWEEP_DIR/${TAG}.log" 2>&1 &
PID=$!; PIDS="$PIDS $PID"
echo "  PID=$PID"

# -------------------------------------------------------
# RUN 5: Hill scan — K_hill=0.10, weighted
# -------------------------------------------------------
TAG="dh_hill010_wpg5_wlate3"
OUT="$SWEEP_DIR/$TAG"
echo "[$(date +%H:%M:%S)] Launching $TAG ..."
nohup python3 "$MAIN" $COMMON \
    --K-hill 0.10 --n-hill 2 \
    --lambda-pg 5 --lambda-late 3 --n-late 2 \
    --output-dir "$OUT" \
    > "$SWEEP_DIR/${TAG}.log" 2>&1 &
PID=$!; PIDS="$PIDS $PID"
echo "  PID=$PID"

# -------------------------------------------------------
# Save PIDs
# -------------------------------------------------------
echo "$PIDS" > "$SWEEP_DIR/pids.txt"

echo ""
echo "All 5 runs launched. PIDs:$PIDS"
echo ""
echo "Monitor:  tail -f $SWEEP_DIR/*.log"
echo "Status:   ps -p $(echo $PIDS | tr ' ' ',') -o pid,stat,etime,cmd"
echo ""

# Wait for all
for P in $PIDS; do
    wait $P 2>/dev/null || true
done

echo "=========================================="
echo " All runs finished at $(date)"
echo "=========================================="

# Quick summary
echo ""
echo "=== Results Summary ==="
for LOG in "$SWEEP_DIR"/*.log; do
    NAME=$(basename "$LOG" .log)
    EVIDENCE=$(grep -oP "log\(evidence\) = [-0-9.]+" "$LOG" 2>/dev/null | tail -1 || echo "N/A")
    RMSE=$(grep -oP "rmse_total.*?[0-9.]+" "$LOG" 2>/dev/null | tail -1 || echo "N/A")
    PG_RMSE=$(grep -oP "rmse_pg_last2.*?[0-9.]+" "$LOG" 2>/dev/null | tail -1 || echo "N/A")
    echo "  $NAME:"
    echo "    Evidence: $EVIDENCE"
    echo "    RMSE:     $RMSE"
    echo "    P.g.last2: $PG_RMSE"
done
