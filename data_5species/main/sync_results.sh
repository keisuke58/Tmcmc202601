#!/bin/bash
# Sync results from remote servers to local _runs directory

echo "Syncing Commensal Static from marinos03..."
rsync -avz marinos03:/home/nishioka/IKM_Hiwi/Tmcmc202601/_runs/ _runs/

echo "Syncing Dysbiotic HOBIC from marinos02..."
rsync -avz marinos02:/home/nishioka/IKM_Hiwi/Tmcmc202601/_runs/ _runs/

echo "Syncing Commensal HOBIC from frontale02..."
rsync -avz frontale02:/home/nishioka/IKM_Hiwi/Tmcmc202601/_runs/ _runs/

echo "Syncing Dysbiotic Static from frontale04..."
rsync -avz frontale04:/home/nishioka/IKM_Hiwi/Tmcmc202601/_runs/ _runs/

echo "Sync complete!"
