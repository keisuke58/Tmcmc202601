#!/bin/bash
# fix-psi + narrow prior: 4 conditions, 1000p, GPU
# Requires: prior_bounds.json already swapped to narrow version

PYTHON="$HOME/miniconda3/envs/klempt_fem/bin/python"
SCRIPT="$HOME/IKM_Hiwi/Tmcmc202601/data_5species/main/estimate_reduced_nishioka_jax.py"
LOGDIR="$HOME/Tmcmc202601/data_5species/main/_runs"
TS=$(date +%Y%m%d_%H%M%S)

COMMON="--mutation rw --n-particles 1000 --n-mutation-steps 20 \
  --multichannel --replicate-sigma --use-exp-init --use-de-mc \
  --fix-psi --lambda-ch5 0"

# Condition  Cultivation  Node       GPU
JOBS=(
  "Dysbiotic  HOBIC       vancouver01  1"
  "Dysbiotic  Static      vancouver01  3"
  "Commensal  Static      vancouver02  2"
  "Commensal  HOBIC       vancouver02  3"
)

for job in "${JOBS[@]}"; do
  read -r COND CULT NODE GPU <<< "$job"
  TAG="${COND:0:1}${CULT:0:1}"  # DH, DS, CS, CH
  LOGFILE="${LOGDIR}/${TAG,,}_1000p_fixpsi_narrow_${TS}.log"

  echo "Launching $TAG on $NODE GPU:$GPU -> $LOGFILE"

  ssh "$NODE" "nohup bash -c 
