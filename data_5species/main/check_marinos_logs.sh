#!/bin/bash
SERVERS="marinos01 marinos02 marinos03"

echo "Checking log files on Marinos servers..."
echo "---------------------------------------------------"

for server in $SERVERS; do
    echo "[$server]"
    # Find log files modified in the last 3 days
    ssh -o BatchMode=yes -o ConnectTimeout=3 "$server" "find /home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species -name '*.log' -mtime -3 -ls" 2>/dev/null

    # Also check tail of specific logs seen in process list
    if [ "$server" == "marinos02" ]; then
        echo "  >> Tail of dysbiotic_static_1000.log:"
        ssh -o BatchMode=yes "$server" "tail -n 5 /home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/dysbiotic_static_1000.log" 2>/dev/null
    elif [ "$server" == "marinos03" ]; then
        echo "  >> Tail of commensal_static_run.log:"
        ssh -o BatchMode=yes "$server" "tail -n 5 /home/nishioka/IKM_Hiwi/Tmcmc202601/commensal_static_run.log" 2>/dev/null
    fi

    if [ $? -ne 0 ]; then
        echo "  (Connection failed or timeout)"
    fi
    echo "---------------------------------------------------"
done
