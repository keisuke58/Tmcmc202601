#!/bin/bash
SERVERS="marinos01 marinos02 marinos03"

echo "Checking detailed process info on Marinos servers..."
echo "---------------------------------------------------"

for server in $SERVERS; do
    echo "[$server]"
    # Get detailed process info: PID, Start Time, Command Line
    # Filtering for python processes owned by nishioka
    ssh -o BatchMode=yes -o ConnectTimeout=3 "$server" "ps -eo user,pid,lstart,cmd | grep python | grep nishioka | grep -v grep | head -n 5" 2>/dev/null
    
    if [ $? -ne 0 ]; then
        echo "  (Connection failed or timeout)"
    fi
    echo "---------------------------------------------------"
done
