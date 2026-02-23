#!/bin/bash
# Quick status check for the P.g. weighted likelihood sweep
SWEEP=$(ls -td /home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/_runs/sweep_pg_*/ 2>/dev/null | head -1)

if [ -z "$SWEEP" ]; then
    echo "No sweep directory found"
    exit 1
fi

echo "=== Sweep: $SWEEP ==="
echo ""

# Check running processes
N_RUNNING=$(ps aux | grep estimate_reduced | grep -v grep | grep -c "sweep_pg" 2>/dev/null || echo 0)
echo "Running processes: $N_RUNNING"
echo ""

echo "=== Per-run status ==="
for LOG in "$SWEEP"/*.log; do
    [ -f "$LOG" ] || continue
    NAME=$(basename "$LOG" .log)
    STAGE=$(grep -oP "Stage \d+/\d+" "$LOG" 2>/dev/null | tail -1)
    BETA=$(grep -oP "β=[0-9.]+" "$LOG" 2>/dev/null | tail -1)
    DONE=$(grep -c "Estimation complete" "$LOG" 2>/dev/null)

    if [ "$DONE" -gt 0 ]; then
        STATUS="DONE"
        EVIDENCE=$(grep -oP "log\(evidence\) = [-0-9.]+" "$LOG" 2>/dev/null | tail -1 || echo "N/A")
        RMSE=$(grep -oP '"rmse_total": [0-9.]+' "$SWEEP/$NAME/fit_metrics.json" 2>/dev/null | head -1 || echo "N/A")
        echo "  $NAME: $STATUS | Evidence: $EVIDENCE | $RMSE"
    elif [ -n "$STAGE" ]; then
        echo "  $NAME: $STAGE ($BETA)"
    else
        echo "  $NAME: initializing..."
    fi
done

echo ""
echo "To monitor live: tail -f ${SWEEP}*.log"
