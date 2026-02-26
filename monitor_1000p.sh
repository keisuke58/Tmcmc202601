#!/bin/bash
# Monitor all TMCMC runs (baseline 1000p + V1-V6 variants)
# Usage: bash monitor_1000p.sh

LOGDIR="$HOME/IKM_Hiwi/Tmcmc202601/data_5species/_runs"

echo "=== TMCMC Monitor: $(date) ==="
echo ""

echo "--- Baseline 1000p (2 chains) ---"
for run in commensal_static_1000p commensal_hobic_1000p dysbiotic_static_1000p dysbiotic_hobic_1000p; do
    last=$(grep "Stage.*β=" "$LOGDIR/${run}.log" 2>/dev/null | tail -1)
    if [ -z "$last" ]; then
        echo "  [$run] initializing..."
    else
        echo "  [$run] $last"
    fi
done
echo ""

echo "--- Variants V1-V6 (1 chain, DH only) ---"
for run in dh_v1_sharp_gate dh_v2_soft_gate dh_v3_tight_bounds dh_v4_strong_pg dh_v5_no_gate dh_v6_wide_baseline; do
    last=$(grep "Stage.*β=" "$LOGDIR/${run}.log" 2>/dev/null | tail -1)
    if [ -z "$last" ]; then
        echo "  [$run] initializing..."
    else
        echo "  [$run] $last"
    fi
done
echo ""

echo "--- Process counts per server ---"
for srv in frontale01 frontale02 frontale04 marinos01 marinos03; do
    cnt=$(ssh $srv 'ps aux | grep estimate_reduced | grep -v grep | wc -l' 2>/dev/null)
    echo "  $srv: $cnt processes"
done
echo "=== Done ==="
