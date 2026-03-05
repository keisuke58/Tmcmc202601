#!/bin/bash
# Run DH and CH TMCMC on remote server (e.g. frontale04)
# Usage: ssh frontale04 'bash -s' < run_dh_ch_remote.sh
#   or:  scp run_dh_ch_remote.sh frontale04:~/IKM_Hiwi/Tmcmc202601/data_5species/main/ && ssh frontale04 "cd ~/IKM_Hiwi/Tmcmc202601/data_5species/main && bash run_dh_ch_remote.sh"

set -e
PROJECT_ROOT="${PROJECT_ROOT:-/home/nishioka/IKM_Hiwi/Tmcmc202601}"
cd "$PROJECT_ROOT/data_5species/main"

TS=$(date +%Y%m%d_%H%M%S)

# DH
nohup python estimate_reduced_nishioka.py \
  --condition Dysbiotic --cultivation HOBIC \
  --n-particles 500 --n-stages 30 \
  --use-exp-init --checkpoint-every 5 \
  --out-dir "_runs/dh_500p30_${TS}" \
  > "dh_500p30_${TS}.log" 2>&1 &
DH_PID=$!
echo "DH started PID=$DH_PID, log=dh_500p30_${TS}.log"

# CH
nohup python estimate_reduced_nishioka.py \
  --condition Commensal --cultivation HOBIC \
  --n-particles 500 --n-stages 30 \
  --use-exp-init --checkpoint-every 5 \
  --out-dir "_runs/ch_500p30_${TS}" \
  > "ch_500p30_${TS}.log" 2>&1 &
CH_PID=$!
echo "CH started PID=$CH_PID, log=ch_500p30_${TS}.log"

echo ""
echo "Both jobs running. Monitor with:"
echo "  tail -f dh_500p30_${TS}.log"
echo "  tail -f ch_500p30_${TS}.log"
echo "  ps -p $DH_PID,$CH_PID"
