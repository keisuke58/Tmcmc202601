#!/bin/bash
SERVERS="marinos01 marinos02 marinos03"

echo "Cleaning up python processes on Marinos servers..."
echo "---------------------------------------------------"

for server in $SERVERS; do
    echo "[$server]"
    # Kill all python processes owned by nishioka
    ssh -o BatchMode=yes -o ConnectTimeout=3 "$server" "pkill -u nishioka python" 2>/dev/null

    if [ $? -eq 0 ]; then
        echo "  >> Successfully sent kill signal to python processes."
    else
        # pkill returns 1 if no processes matched, which is also fine (already clean)
        echo "  >> No python processes found or connection failed."
    fi

    # Double check
    count=$(ssh -o BatchMode=yes "$server" "pgrep -u nishioka python | wc -l")
    echo "  >> Remaining python processes: $count"
    echo "---------------------------------------------------"
done
