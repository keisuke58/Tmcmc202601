#!/bin/bash
# ============================================================
# Production TMCMC: 2000p × 40mut × fix-psi × single-channel
#   + warm-start from best pilot runs
#   + posterior-informed prior narrowing (MAP ± 4σ)
#
# Strategy:
#   1. Single-channel (no multichannel) — 10-90× better logL
#   2. Fix-psi (experimental viability) — reduces model complexity
#   3. MAP warm-start — 50% particles from previous posterior
#   4. Posterior prior — bounds narrowed to MAP ± 4σ
#   5. 40 mutation steps — target ~23% acceptance (Roberts 1997)
#   6. DE-MCMC — alternating proposals for mode exploration
#
# Usage:
#   bash run_production_2000p.sh          # all 4 conditions, seed 42
#   bash run_production_2000p.sh 2        # round 2 (seed 142)
#   bash run_production_2000p.sh 3        # round 3 (seed 242)
# ============================================================

set -euo pipefail

ROUND="${1:-1}"
case $ROUND in
  1) SEED=42  ;;
  2) SEED=142 ;;
  3) SEED=242 ;;
  *) SEED=$ROUND ;;
esac

SCRIPT="$HOME/Tmcmc202601/data_5species/main/estimate_reduced_nishioka_jax.py"
LOGDIR="$HOME/Tmcmc202601/data_5species/main/_runs"
TS=$(date +%Y%m%d_%H%M%S)

# Warm-start directories
# Round 1: use pilot runs (200-500p single-channel)
# Round 2+: CS uses Round 1 production (2000p) for tighter posterior
BEST_DH="$HOME/Tmcmc202601/data_5species/main/_runs/jax_ode_nuts_Dysbiotic_HOBIC_20260319_170643"
BEST_DS="$HOME/Tmcmc202601/data_5species/main/_runs/jax_ode_nuts_Dysbiotic_Static_20260319_184034"
BEST_CH="$HOME/Tmcmc202601/data_5species/main/_runs/jax_ode_nuts_Commensal_HOBIC_20260319_161245"
if [ "$ROUND" -ge 2 ]; then
  # CS: use R1 production (2000p) for much tighter posterior
  BEST_CS="$HOME/Tmcmc202601/data_5species/main/_runs/jax_ode_nuts_Commensal_Static_20260320_003350"
  # Also upgrade DH/DS/CH to R1 production results
  BEST_DH="$HOME/Tmcmc202601/data_5species/main/_runs/jax_ode_nuts_Dysbiotic_HOBIC_20260320_005944"
  BEST_DS="$HOME/Tmcmc202601/data_5species/main/_runs/jax_ode_nuts_Dysbiotic_Static_20260320_004825"
  BEST_CH="$HOME/Tmcmc202601/data_5species/main/_runs/jax_ode_nuts_Commensal_HOBIC_20260320_003519"
else
  BEST_CS="$HOME/Tmcmc202601/data_5species/main/_runs/jax_ode_nuts_Commensal_Static_20260319_161257"
fi

# Per-condition nsigma: CS gets tighter (3σ) since R1 posterior is better
declare -a NSIGMAS=("4" "4" "3" "4")  # DH DS CS CH
if [ "$ROUND" -ge 2 ]; then
  # All conditions get tighter priors from R1 production posterior
  NSIGMAS=("3" "3" "3" "3")
fi

# GPU assignment: vancouver01 GPU 0,1,3; vancouver02 GPU 1,2,3
declare -a CONDS=("Dysbiotic"  "Dysbiotic"  "Commensal"  "Commensal")
declare -a CULTS=("HOBIC"      "Static"     "Static"     "HOBIC")
declare -a NODES=("vancouver01" "vancouver01" "vancouver02" "vancouver02")
declare -a GPUS=("0"           "1"          "1"          "2")
declare -a TAGS=("dh"          "ds"         "cs"         "ch")
declare -a BEST_DIRS=("$BEST_DH" "$BEST_DS" "$BEST_CS" "$BEST_CH")

echo "=============================================="
echo "Production TMCMC: 2000p × 40mut × fix-psi"
echo "  Round: $ROUND, Seed: $SEED"
echo "  Time: $(date)"
echo "=============================================="

for i in 0 1 2 3; do
  COND=${CONDS[$i]}
  CULT=${CULTS[$i]}
  NODE=${NODES[$i]}
  GPU=${GPUS[$i]}
  TAG=${TAGS[$i]}
  BEST=${BEST_DIRS[$i]}
  NSIG=${NSIGMAS[$i]}
  LOGFILE="${LOGDIR}/${TAG}_2000p_prod_r${ROUND}_${TS}.log"

  echo "[$i] $TAG ($COND $CULT) -> $NODE GPU:$GPU nsigma=$NSIG (warm-start: $(basename $BEST))"

  ssh "$NODE" "nohup bash -c '\
    source ~/miniconda3/etc/profile.d/conda.sh && \
    conda activate klempt_fem && \
    cd ~/Tmcmc202601/data_5species/main && \
    CUDA_VISIBLE_DEVICES=$GPU JAX_PLATFORMS=cuda \
    python3 -u $SCRIPT \
      --condition $COND --cultivation $CULT \
      --init-from-dir $BEST \
      --mutation rw --n-particles 2000 --n-mutation-steps 40 \
      --use-exp-init --use-de-mc --fix-psi \
      --posterior-prior-nsigma $NSIG \
      --seed $SEED \
  ' > $LOGFILE 2>&1 &"

  sleep 2
done

echo ""
echo "=== All 4 production jobs launched (round $ROUND) ==="
echo "Logs: ${LOGDIR}/*_2000p_prod_r${ROUND}_${TS}.log"
echo ""
echo "Monitor:"
echo "  tail -f ${LOGDIR}/*_2000p_prod_r${ROUND}_*.log"
echo ""
echo "Expected time: ~50-60 min per condition"
echo ""
echo "After completion, run rounds 2 and 3 for multi-chain:"
echo "  bash run_production_2000p.sh 2"
echo "  bash run_production_2000p.sh 3"
