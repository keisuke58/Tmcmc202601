#!/bin/bash
# Stuttgart (GPU) サーバの接続・空き確認
# Usage: bash check_stuttgart.sh
SERVERS="stuttgart01 stuttgart02 stuttgart03"

echo "Stuttgart servers (GPU: RTX 3090)"
echo "=========================================="
for srv in $SERVERS; do
    echo "[$srv]"
    ssh -o ConnectTimeout=3 "$srv" "hostname; nvidia-smi -L 2>/dev/null | head -2; ps aux | grep -E 'gradient_tmcmc|estimate_reduced' | grep -v grep | wc -l" 2>/dev/null || echo "  (connection failed)"
    echo "------------------------------------------"
done
