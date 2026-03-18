#!/bin/bash
# GPU TMCMC long run — runs ON vancouver01 (launched via nohup)
# 5000p × 20 mutation steps × 4 conditions sequential
# Expected: ~6-12h total on RTX 4090
#
# This script is meant to run directly on vancouver01, NOT via ssh per-condition.

set -e

# --- Config ---
LOGDIR="/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/main/_runs/gpu_long_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"
MASTERLOG="$LOGDIR/master.log"

WORKDIR="/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/main"
SCRIPT="estimate_reduced_nishioka_jax.py"

COMMON="--mutation rw --n-particles 5000 --n-mutation-steps 20 \
  --use-exp-init --n-steps 2500 --max-stages 50 \
  --n-hill 4.0 --K-hill 0.05 \
  --replicate-sigma \
  --multichannel \
  --use-de-mc \
  --use-student-t --student-t-nu 5.0 \
  --lambda-rare 0.1 \
  --max-info-ratio 10.0 \
  --resample-method systematic \
  --device gpu --seed 42"

# --- Conda ---
source /home/nishioka/miniforge3/etc/profile.d/conda.sh
conda activate /home/nishioka/miniconda3/envs/klempt_fem

cd "$WORKDIR"

echo "=== GPU TMCMC Long Run ===" | tee "$MASTERLOG"
echo "Settings: 5000p × 20mut × max50stages × n_hill=4 × multichannel × DE-MC × Student-t × replicate-σ" | tee -a "$MASTERLOG"
echo "Log dir: $LOGDIR" | tee -a "$MASTERLOG"
echo "Start: $(date)" | tee -a "$MASTERLOG"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')" | tee -a "$MASTERLOG"
echo "" | tee -a "$MASTERLOG"

CONDITIONS=(
  "Commensal Static CS"
  "Commensal HOBIC CH"
  "Dysbiotic Static DS"
  "Dysbiotic HOBIC DH"
)

TOTAL=${#CONDITIONS[@]}
DONE=0
FAILED=0

for PAIR in "${CONDITIONS[@]}"; do
  COND=$(echo "$PAIR" | cut -d' ' -f1)
  CULT=$(echo "$PAIR" | cut -d' ' -f2)
  TAG=$(echo "$PAIR" | cut -d' ' -f3)
  CONDLOG="$LOGDIR/${TAG}.log"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$((DONE+1))/$TOTAL] Starting $TAG ($COND $CULT)..." | tee -a "$MASTERLOG"

  if python "$SCRIPT" --condition "$COND" --cultivation "$CULT" \
      --output-dir "$LOGDIR/$TAG" \
      $COMMON 2>&1 | tee "$CONDLOG"; then
    DONE=$((DONE+1))
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $TAG DONE ($DONE/$TOTAL)" | tee -a "$MASTERLOG"
  else
    FAILED=$((FAILED+1))
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $TAG FAILED (exit=$?)" | tee -a "$MASTERLOG"
  fi
  echo "" | tee -a "$MASTERLOG"
done

echo "=== Complete: $DONE/$TOTAL succeeded, $FAILED failed ===" | tee -a "$MASTERLOG"
echo "End: $(date)" | tee -a "$MASTERLOG"
echo "Results in: $LOGDIR" | tee -a "$MASTERLOG"
