#!/bin/bash
# 調査対象のサーバ
SERVERS="marinos01 marinos02 marinos03 frontale01 frontale02 frontale03 frontale04"
OUTPUT_FILE="/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/server.md"

# ヘッダー書き込み
echo "# Server Status Report" > "$OUTPUT_FILE"
echo "Generated on: $(date)" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "| Server | Status | Load (1m/5m/15m) | Memory (Used/Total) | Note |" >> "$OUTPUT_FILE"
echo "|---|---|---|---|---|" >> "$OUTPUT_FILE"

echo "Checking servers..."

for server in $SERVERS; do
    echo -n "  $server ... "
    # SSHで接続確認 (タイムアウト2秒)
    info=$(ssh -o BatchMode=yes -o ConnectTimeout=2 "$server" "uptime && echo '---' && free -h | grep Mem" 2>&1)

    if [ $? -eq 0 ]; then
        echo "OK"
        # Load Averageの抽出
        load=$(echo "$info" | grep "load average" | sed 's/.*load average: //')
        # Memoryの抽出
        mem_line=$(echo "$info" | grep "Mem:")
        mem_used=$(echo "$mem_line" | awk '{print $3}')
        mem_total=$(echo "$mem_line" | awk '{print $2}')

        # 簡易的な混雑判定 (Load > 10 なら Busy と表示)
        note="Idle"
        load1=$(echo "$load" | awk -F',' '{print $1}')
        if (( $(echo "$load1 > 10.0" | bc -l 2>/dev/null) )); then
            note="**Busy**"
        fi

        echo "| $server | 🟢 Online | $load | ${mem_used}/${mem_total} | $note |" >> "$OUTPUT_FILE"
    else
        echo "Failed"
        echo "| $server | 🔴 Offline | - | - | Connection Failed |" >> "$OUTPUT_FILE"
    fi
done

echo ""
echo "---------------------------------------------------"
echo "Report saved to: $OUTPUT_FILE"
echo "---------------------------------------------------"
