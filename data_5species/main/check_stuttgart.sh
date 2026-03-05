#!/bin/bash
# Stuttgart (GPU) サーバの接続・空き確認
# Usage: bash check_stuttgart.sh
#   conda deactivate してから実行すると SSH が安定（OpenSSL 競合回避）
SERVERS="${SERVERS:-stuttgart01 stuttgart02 stuttgart03}"
SSH_CMD="${SSH_CMD:-/usr/bin/ssh}"
[[ ! -x "$SSH_CMD" ]] && SSH_CMD="ssh"

echo "Stuttgart servers (GPU: RTX 3090)"
echo "=========================================="
for srv in $SERVERS; do
    echo "[$srv]"
    $SSH_CMD -o ConnectTimeout=3 "$srv" "hostname; nvidia-smi -L 2>/dev/null | head -2; ps aux | grep -E 'gradient_tmcmc|estimate_reduced' | grep -v grep | wc -l" 2>/dev/null || echo "  (connection failed)"
    echo "------------------------------------------"
done
