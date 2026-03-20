#!/bin/bash
# ============================================================
# Phase 2: free-ψ + narrow A prior (from Phase 1 fix-ψ posterior)
#
# Strategy:
#   Phase 1 (done): fix-ψ → A posterior (tight, well-identified)
#   Phase 2 (this): free ψ + narrow A prior + viability as soft constraint
#
#   With A already constrained from Phase 1, ψ multimodality is reduced.
#   Viability enters as likelihood channel (ch3) with σ_ψ, not as hard fix.
#
# Channel setup:
#   ch1 (species fractions): weight 1.0 — primary
#   ch2 (total species): weight 0.0 — OFF (conflicted in Phase 1 tests)
#   ch3 (viability): weight 2.0 — soft ψ constraint
#   ch5 (pH): weight 0.0 — OFF for Static, can enable for HOBIC
#
# Usage:
#   bash run_phase2_free_psi.sh DS          # DS only
#   bash run_phase2_free_psi.sh DS DH       # DS and DH
#   bash run_phase2_free_psi.sh ALL         # all 4 conditions
# ============================================================

set -euo pipefail

SCRIPT="$HOME/Tmcmc202601/data_5species/main/estimate_reduced_nishioka_jax.py"
LOGDIR="$HOME/Tmcmc202601/data_5species/main/_runs"
TS=$(date +%Y%m%d_%H%M%S)
SEED="${SEED:-42}"

# Phase 1 production results (2000p fix-ψ) — used as warm-start
declare -A P1_DIRS=(
  [DH]="$HOME/Tmcmc202601/data_5species/main/_runs/jax_ode_nuts_Dysbiotic_HOBIC_20260320_005944"
  [DS]="$HOME/Tmcmc202601/data_5species/main/_runs/jax_ode_nuts_Dysbiotic_Static_20260320_004825"
  [CS]="$HOME/Tmcmc202601/data_5species/main/_runs/jax_ode_nuts_Commensal_Static_20260320_003350"
  [CH]="$HOME/Tmcmc202601/data_5species/main/_runs/jax_ode_nuts_Commensal_HOBIC_20260320_003519"
)
declare -A COND_MAP=([DH]="Dysbiotic" [DS]="Dysbiotic" [CS]="Commensal" [CH]="Commensal")
declare -A CULT_MAP=([DH]="HOBIC" [DS]="Static" [CS]="Static" [CH]="HOBIC")

# GPU assignment
declare -A GPU_NODE=([DH]="vancouver02" [DS]="vancouver01" [CS]="vancouver02" [CH]="vancouver02")
declare -A GPU_ID=([DH]="0" [DS]="3" [CS]="0" [CH]="3")

# Parse targets
TARGETS=("$@")
if [ "${#TARGETS[@]}" -eq 0 ] || [ "${TARGETS[0]}" = "ALL" ]; then
  TARGETS=(DS DH CS CH)
fi

echo "=============================================="
echo "Phase 2: free-ψ + narrow A prior"
echo "  Targets: ${TARGETS[*]}"
echo "  Seed: $SEED"
echo "  Time: $(date)"
echo "=============================================="

for TAG in "${TARGETS[@]}"; do
  TAG=$(echo "$TAG" | tr '[:lower:]' '[:upper:]')
  COND=${COND_MAP[$TAG]}
  CULT=${CULT_MAP[$TAG]}
  NODE=${GPU_NODE[$TAG]}
  GPU=${GPU_ID[$TAG]}
  P1_DIR=${P1_DIRS[$TAG]}
  LOGFILE="${LOGDIR}/${TAG,,}_phase2_freepsi_${TS}.log"

  # Channel weights: ch1=species, ch3=viability (soft ψ), ch2=OFF, ch5=OFF
  LAMBDA_CH2=0.0
  LAMBDA_CH3=2.0
  LAMBDA_CH5=0.0

  echo "[$TAG] $COND $CULT -> $NODE GPU:$GPU"
  echo "  Warm-start: $(basename $P1_DIR)"
  echo "  Channels: ch1=1.0, ch2=$LAMBDA_CH2, ch3=$LAMBDA_CH3, ch5=$LAMBDA_CH5"

  ssh "$NODE" "nohup bash -c '\
    source ~/miniconda3/etc/profile.d/conda.sh && \
    conda activate klempt_fem && \
    cd ~/Tmcmc202601/data_5species/main && \
    CUDA_VISIBLE_DEVICES=$GPU JAX_PLATFORMS=cuda \
    python3 -u $SCRIPT \
      --condition $COND --cultivation $CULT \
      --init-from-dir $P1_DIR \
      --mutation rw --n-particles 2000 --n-mutation-steps 40 \
      --use-exp-init --use-de-mc \
      --multichannel \
      --lambda-ch1 1.0 --lambda-ch2 $LAMBDA_CH2 --lambda-ch3 $LAMBDA_CH3 --lambda-ch5 $LAMBDA_CH5 \
      --posterior-prior-nsigma 3 \
      --seed $SEED \
  ' > $LOGFILE 2>&1 &"

  sleep 2
done

echo ""
echo "=== Phase 2 launched ==="
echo "Logs: ${LOGDIR}/*_phase2_freepsi_${TS}.log"
echo "Monitor: tail -f ${LOGDIR}/*_phase2_freepsi_*.log"
