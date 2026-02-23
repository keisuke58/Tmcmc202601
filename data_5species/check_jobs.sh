#!/bin/bash
# Check status of all running TMCMC jobs

echo "=========================================="
echo "TMCMC Job Status - $(date)"
echo "=========================================="

echo ""
echo "=== frontale01 (Improved 1000p) ==="
PROCS=$(ssh -o ConnectTimeout=3 frontale01 "ps aux | grep estimate_commensal | grep -v grep | wc -l" 2>/dev/null)
echo "Processes: $PROCS"
if [ "$PROCS" -gt 0 ]; then
    ssh frontale01 "tail -3 /home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/frontale_improved.log 2>/dev/null | grep -E 'Stage|Chain|complete'"
fi

echo ""
echo "=== frontale02 (Tight Decay) ==="
PROCS=$(ssh -o ConnectTimeout=3 frontale02 "ps aux | grep estimate_commensal | grep -v grep | wc -l" 2>/dev/null)
echo "Processes: $PROCS"
if [ "$PROCS" -gt 0 ]; then
    ssh frontale02 "tail -3 /home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/frontale02_tight_decay.log 2>/dev/null | grep -E 'Stage|Chain|complete'"
fi

echo ""
echo "=== frontale03 (HOBIC) ==="
PROCS=$(ssh -o ConnectTimeout=3 frontale03 "ps aux | grep estimate_commensal | grep -v grep | wc -l" 2>/dev/null)
echo "Processes: $PROCS"
if [ "$PROCS" -gt 0 ]; then
    ssh frontale03 "tail -3 /home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/frontale03_hobic.log 2>/dev/null | grep -E 'Stage|Chain|complete'"
fi

echo ""
echo "=========================================="
