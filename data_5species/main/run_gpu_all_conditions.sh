#!/bin/bash
# GPU TMCMC: 4 conditions sequential on vancouver01 (最強設定)
# 2000p, 10 mutation steps, RW, n_hill=4, GPU vmap
#
# Usage: bash run_gpu_all_conditions.sh
# Requires: rsync code to vancouver01 first

set -e

CONDA_SETUP="source /home/nishioka/miniforge3/etc/profile.d/conda.sh && conda activate /home/nishioka/miniconda3/envs/klempt_fem"
WORKDIR="/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/main"
SCRIPT="estimate_reduced_nishioka_jax.py"

COMMON="--mutation rw --n-particles 2000 --n-mutation-steps 10 \
  --use-exp-init --n-steps 2500 --max-stages 30 \
  --n-hill 4.0 --K-hill 0.05 \
  --replicate-sigma \
  --multichannel \
  --use-de-mc \
  --use-student-t --student-t-nu 5.0 \
  --lambda-rare 0.1 \
  --max-info-ratio 10.0 \
  --resample-method systematic \
  --device gpu --seed 42"

echo "=== GPU TMCMC: 4 conditions × 2000p × 10mut (n_hill=4) ==="
echo "Estimated: ~10min per condition = ~40min total"
echo ""

# Run sequentially (avoid GPU OOM from parallel runs)
for PAIR in "Commensal Static CS" "Commensal HOBIC CH" "Dysbiotic Static DS" "Dysbiotic HOBIC DH"; do
  COND=$(echo $PAIR | cut -d' ' -f1)
  CULT=$(echo $PAIR | cut -d' ' -f2)
  TAG=$(echo $PAIR | cut -d' ' -f3)

  echo "[$(date +%H:%M:%S)] Starting $TAG ($COND $CULT)..."
  ssh vancouver01 "$CONDA_SETUP && cd $WORKDIR && python $SCRIPT \
    --condition $COND --cultivation $CULT $COMMON 2>&1" | tee /tmp/gpu_${TAG}.log
  echo "[$(date +%H:%M:%S)] $TAG done."
  echo ""
done

echo "=== All 4 conditions complete ==="
