#!/bin/bash
SERVERS="marinos01 marinos02 marinos03"

echo "Checking active users on Marinos servers..."
echo "---------------------------------------------------"

for server in $SERVERS; do
    echo "[$server]"
    # Get top 5 CPU consuming processes with user info
    ssh -o BatchMode=yes -o ConnectTimeout=3 "$server" "ps -eo user,%cpu,comm --sort=-%cpu | head -n 6" 2>/dev/null

    if [ $? -ne 0 ]; then
        echo "  (Connection failed or timeout)"
    fi
    echo "---------------------------------------------------"
done
