#!/bin/bash
# run_v5_multistart.sh — v5: 条件別最適設定 + MAP polish + 3 seed 並列
#
# 最良設定:
#   CS: sign_prior=True,  lam=0.5, bc=0 (sign prior が正則化として有効)
#   CH: sign_prior=True,  lam=0.5, bc=0
#   DH: sign_prior=False, —,      bc=0 (制約なしが最良)
#   DS: sign_prior=True,  lam=0.5, bc=0
#
# 使い方:
#   bash run_v5_multistart.sh              # 4 GPU に 4 条件を1 seed ずつ
#   bash run_v5_multistart.sh --3seeds     # 各条件 3 seed (GPU0-2 で並列)
#   bash run_v5_multistart.sh --condition CS  # CS だけ

set -euo pipefail

PYTHON=~/miniforge3/envs/klempt_fem2/bin/python
SCRIPT=~/IKM_Hiwi/Tmcmc202601/data_5species/main/estimate_reduced_nishioka_jax.py
LOG_DIR=~/IKM_Hiwi/Tmcmc202601/data_5species/main
NPART=1000
NSTAGES=30
NHILL=4
KHILL=0.05

MODE="single"
ONLY_CONDITION=""

for arg in "$@"; do
    case $arg in
        --3seeds) MODE="3seeds" ;;
        --condition) ONLY_CONDITION="${2:-}" ;;
        CS|CH|DH|DS) ONLY_CONDITION="$arg" ;;
    esac
done

run_one() {
    local COND=$1 CULT=$2 SEED=$3 GPU=$4
    local SIGN_ARGS=$5
    local TAG="v5_${COND:0:1}${CULT:0:1}_s${SEED}"
    local LOG="${LOG_DIR}/${TAG}.log"
    echo "  GPU${GPU}: ${COND} ${CULT} seed=${SEED} ${SIGN_ARGS} → ${TAG}.log"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON $SCRIPT \
        --condition "$COND" --cultivation "$CULT" \
        --n-particles "$NPART" --max-stages "$NSTAGES" \
        --n-hill "$NHILL" --K-hill "$KHILL" \
        --use-exp-init --seed "$SEED" \
        --mutation rw \
        $SIGN_ARGS \
        > "$LOG" 2>&1 &
}

echo "=============================================="
echo "v5 multi-start TMCMC (MAP polish ON)"
echo "  Particles: $NPART, Stages: $NSTAGES, n_hill=$NHILL"
echo "  Mode: $MODE"
echo "=============================================="

# Condition-specific sign settings (best from experiments)
CS_ARGS="--sign-prior --sign-lambda 0.5"
CH_ARGS="--sign-prior --sign-lambda 0.5"
DH_ARGS=""                              # unconstrained: best for DH
DS_ARGS="--sign-prior --sign-lambda 0.5"

# CS uniform-weight variant: equal weight for all species (no Pg bias)
# Optimizes for overall RMSE rather than Pg-focused RMSE
CS_UW_ARGS="--sign-prior --sign-lambda 0.5 --lambda-pg 1.0 --lambda-late 1.0 --sigma-scale 0.85"

if [ "$MODE" = "3seeds" ]; then
    echo "3-seed mode: 3 seeds per condition sequentially on 4 GPUs"
    echo "CS: standard×2 seeds + uniform-weight×1 seed"
    for SEED in 42 123 456; do
        echo "--- Seed $SEED ---"
        if [[ -z "$ONLY_CONDITION" || "$ONLY_CONDITION" == "CS" ]]; then
            if [ "$SEED" -eq 456 ]; then
                run_one Commensal Static $SEED 0 "$CS_UW_ARGS"  # uniform-weight variant
            else
                run_one Commensal Static $SEED 0 "$CS_ARGS"
            fi
        fi
        [[ -z "$ONLY_CONDITION" || "$ONLY_CONDITION" == "CH" ]] && run_one Commensal HOBIC  $SEED 1 "$CH_ARGS"
        [[ -z "$ONLY_CONDITION" || "$ONLY_CONDITION" == "DH" ]] && run_one Dysbiotic HOBIC  $SEED 2 "$DH_ARGS"
        [[ -z "$ONLY_CONDITION" || "$ONLY_CONDITION" == "DS" ]] && run_one Dysbiotic Static $SEED 3 "$DS_ARGS"
        wait
        echo "Seed $SEED done."
    done
else
    echo "Single seed (42), 4 GPUs parallel"
    [[ -z "$ONLY_CONDITION" || "$ONLY_CONDITION" == "CS" ]] && run_one Commensal Static 42 0 "$CS_ARGS"
    [[ -z "$ONLY_CONDITION" || "$ONLY_CONDITION" == "CH" ]] && run_one Commensal HOBIC  42 1 "$CH_ARGS"
    [[ -z "$ONLY_CONDITION" || "$ONLY_CONDITION" == "DH" ]] && run_one Dysbiotic HOBIC  42 2 "$DH_ARGS"
    [[ -z "$ONLY_CONDITION" || "$ONLY_CONDITION" == "DS" ]] && run_one Dysbiotic Static 42 3 "$DS_ARGS"
    wait
    echo "All done."
fi

echo "=============================================="
echo "v5 runs finished. Run:"
echo "  python generate_supplementary_jax.py"
echo "to generate figures."
echo "=============================================="
