#!/bin/bash
# =============================================================================
# DH_TMCMC_IMPROVEMENT_PLAN Step 1: JAX ODE + NUTS で DH 精度改善
# 500p, 30 stages, λ_Pg=8, λ_late=4, use_exp_init
# =============================================================================
# Usage:
#   conda deactivate
#   cd Tmcmc202601/data_5species/main
#   bash run_dh_improvement_plan.sh                    # vancouver01 でバックグラウンド
#   SERVER=stuttgart02 bash run_dh_improvement_plan.sh
# =============================================================================
set -euo pipefail

SSH_CMD="${SSH_CMD:-/usr/bin/ssh}"
[[ ! -x "$SSH_CMD" ]] && SSH_CMD="ssh"

PROJECT_ROOT="${PROJECT_ROOT:-/home/nishioka/IKM_Hiwi/Tmcmc202601}"
MAIN_DIR="${PROJECT_ROOT}/data_5species/main"
REMOTE_PYTHON="${REMOTE_PYTHON:-/home/nishioka/miniforge3/envs/klempt_fem2/bin/python3}"
SERVER="${SERVER:-vancouver01}"

TS=$(date +%Y%m%d_%H%M%S)
OUT_DIR="${MAIN_DIR}/_runs/dh_improvement_${TS}"
LOG="${MAIN_DIR}/_runs/dh_improvement_${TS}.log"

echo "=============================================="
echo " DH 改善プラン Step 1 (JAX ODE + NUTS)"
echo " $(date)"
echo " Server: $SERVER"
echo " 500p, 30st, λ_Pg=8, λ_late=4, use_exp_init"
echo "=============================================="

# rsync
echo "rsync to $SERVER ..."
$SSH_CMD "$SERVER" "mkdir -p $(dirname $PROJECT_ROOT)" 2>/dev/null || true
rsync -e "${SSH_CMD}" -avz --exclude '_runs' --exclude '__pycache__' --exclude '.git' --exclude '*.odb' \
    "$PROJECT_ROOT/" "$SERVER:$PROJECT_ROOT/" || { echo "rsync failed"; exit 1; }

echo ""
echo "Launching on $SERVER in background..."
$SSH_CMD "$SERVER" "cd $MAIN_DIR && mkdir -p $(dirname $OUT_DIR) && \
    nohup env LD_LIBRARY_PATH= JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0 $REMOTE_PYTHON estimate_reduced_nishioka_jax.py \
    --condition Dysbiotic --cultivation HOBIC \
    --n-particles 500 --max-stages 30 \
    --lambda-pg 8 --lambda-late 4 --use-exp-init \
    --output-dir $OUT_DIR \
    > $LOG 2>&1 &"

echo ""
echo "=============================================="
echo " 投入完了"
echo " ログ: ssh $SERVER \"tail -f $LOG\""
echo " 結果: $SERVER:$OUT_DIR"
echo "=============================================="
