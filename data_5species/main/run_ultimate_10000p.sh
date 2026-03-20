#!/bin/bash
# ============================================================
# Ultimate TMCMC: 10000p × 50mut × free-ψ × narrow A prior
#
# Final production run for paper-quality posterior:
#   - 10000 particles → ~5000-9500 ESS for smooth posterior
#   - free-ψ with viability soft constraint (ch3)
#   - Warm-start from Phase 2 (free-ψ 2000p) posterior
#   - 2σ narrowing — very tight A prior from Phase 2
#   - Corner plots, Bayes factors, posterior predictive all benefit
#
# Estimated time: 3-5 hours per condition (overnight run)
# Memory: ~5GB/GPU (10000 × ODE vmap fits in 24GB)
#
# Usage:
#   bash run_ultimate_10000p.sh              # all 4 on stuttgart01
#   bash run_ultimate_10000p.sh DS DH        # specific conditions
#   NODE=stuttgart02 bash run_ultimate_10000p.sh  # different server
# ============================================================

set -euo pipefail

SCRIPT="$HOME/Tmcmc202601/data_5species/main/estimate_reduced_nishioka_jax.py"
LOGDIR="$HOME/Tmcmc202601/data_5species/main/_runs"
TS=$(date +%Y%m%d_%H%M%S)
SEED="${SEED:-42}"
NODE="${NODE:-stuttgart01}"

# Phase 2 result directories (free-ψ 2000p) — will be set after Phase 2 completes
# These are placeholders — update with actual Phase 2 output dirs before running
P2_DH="${P2_DH:-PENDING}"
P2_DS="${P2_DS:-PENDING}"
P2_CS="${P2_CS:-PENDING}"
P2_CH="${P2_CH:-PENDING}"

declare -A P2_DIRS=([DH]="$P2_DH" [DS]="$P2_DS" [CS]="$P2_CS" [CH]="$P2_CH")
declare -A COND_MAP=([DH]="Dysbiotic" [DS]="Dysbiotic" [CS]="Commensal" [CH]="Commensal")
declare -A CULT_MAP=([DH]="HOBIC" [DS]="Static" [CS]="Static" [CH]="HOBIC")

# Parse targets
TARGETS=("$@")
if [ "${#TARGETS[@]}" -eq 0 ]; then
  TARGETS=(DH DS CS CH)
fi

# Validate Phase 2 dirs exist
for TAG in "${TARGETS[@]}"; do
  TAG=$(echo "$TAG" | tr '[:lower:]' '[:upper:]')
  DIR="${P2_DIRS[$TAG]}"
  if [ "$DIR" = "PENDING" ] || [ ! -d "$DIR" ]; then
    echo "ERROR: Phase 2 dir for $TAG not found: $DIR"
    echo "Set P2_${TAG}=/path/to/phase2/result before running"
    exit 1
  fi
done

echo "=============================================="
echo "Ultimate TMCMC: 10000p × 50mut × free-ψ"
echo "  Targets: ${TARGETS[*]}"
echo "  Server: $NODE"
echo "  Seed: $SEED"
echo "  Time: $(date)"
echo "=============================================="

GPU=0
for TAG in "${TARGETS[@]}"; do
  TAG=$(echo "$TAG" | tr '[:lower:]' '[:upper:]')
  COND=${COND_MAP[$TAG]}
  CULT=${CULT_MAP[$TAG]}
  P2_DIR=${P2_DIRS[$TAG]}
  LOGFILE="${LOGDIR}/${TAG,,}_ultimate_10000p_${TS}.log"

  echo "[$TAG] $COND $CULT -> $NODE GPU:$GPU"
  echo "  Phase 2 warm-start: $(basename $P2_DIR)"

  # DH HOBIC gets pH channel (ch5) with low weight
  LAMBDA_CH5=0.0
  if [ "$CULT" = "HOBIC" ]; then
    LAMBDA_CH5=0.3
  fi

  ssh "$NODE" "nohup bash -c '\
    source ~/miniconda3/etc/profile.d/conda.sh && \
    conda activate klempt_fem && \
    cd ~/Tmcmc202601/data_5species/main && \
    CUDA_VISIBLE_DEVICES=$GPU JAX_PLATFORMS=cuda \
    python3 -u $SCRIPT \
      --condition $COND --cultivation $CULT \
      --init-from-dir $P2_DIR \
      --mutation rw --n-particles 10000 --n-mutation-steps 50 \
      --use-exp-init --use-de-mc \
      --multichannel \
      --lambda-ch1 1.0 --lambda-ch2 0.0 --lambda-ch3 2.0 --lambda-ch5 $LAMBDA_CH5 \
      --posterior-prior-nsigma 2 \
      --seed $SEED \
  ' > $LOGFILE 2>&1 &"

  GPU=$((GPU + 1))
  sleep 2
done

echo ""
echo "=== Ultimate 10000p launched ==="
echo "Logs: ${LOGDIR}/*_ultimate_10000p_${TS}.log"
echo "Monitor: tail -f ${LOGDIR}/*_ultimate_10000p_*.log"
echo ""
echo "Expected: 3-5 hours per condition (overnight)"
echo "ESS target: ~5000-9500 (paper-quality posterior)"
