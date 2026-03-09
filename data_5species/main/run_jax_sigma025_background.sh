#!/bin/bash
# =============================================================================
# JAX ODE + NUTS: sigma_scale=0.25 で GPU マシンにバックグラウンド投入
# vancouver01 または stuttgart で実行
#
# vancouver01 で cuSPARSE エラーが出る場合:
#   ssh vancouver01
#   bash setup_vancouver01_jax_gpu.sh   # nvidia-cusparse-cu12 インストール
#   LD_LIBRARY_PATH= で実行（スクリプトに組み込み済み）
# または: SERVER=stuttgart02 bash run_jax_sigma025_background.sh
# =============================================================================
# Usage:
#   conda deactivate  # OpenSSL 競合回避
#   cd Tmcmc202601/data_5species/main
#   bash run_jax_sigma025_background.sh                    # vancouver01, DH のみ
#   SERVER=vancouver01 bash run_jax_sigma025_background.sh
#   SERVER=stuttgart02 bash run_jax_sigma025_background.sh
#   bash run_jax_sigma025_background.sh --all-conditions  # 4条件すべて
# =============================================================================
set -euo pipefail

SSH_CMD="${SSH_CMD:-/usr/bin/ssh}"
[[ ! -x "$SSH_CMD" ]] && SSH_CMD="ssh"

PROJECT_ROOT="${PROJECT_ROOT:-/home/nishioka/IKM_Hiwi/Tmcmc202601}"
MAIN_DIR="${PROJECT_ROOT}/data_5species/main"
REMOTE_PYTHON="${REMOTE_PYTHON:-/home/nishioka/miniforge3/envs/klempt_fem2/bin/python3}"
SERVER="${SERVER:-vancouver01}"
ALL_CONDITIONS=false

while [[ "${1:-}" == --* ]]; do
    case "$1" in
        --all-conditions) ALL_CONDITIONS=true; shift ;;
        *) shift ;;
    esac
done

TS=$(date +%Y%m%d_%H%M%S)
LOG_BASE="${MAIN_DIR}/_runs/jax_sigma025_${TS}"
mkdir -p "$LOG_BASE"

echo "=============================================="
echo " JAX ODE + NUTS (sigma_scale=0.25) バックグラウンド"
echo " $(date)"
echo " Server: $SERVER"
echo "=============================================="

# rsync
echo "rsync to $SERVER ..."
$SSH_CMD "$SERVER" "mkdir -p $(dirname $PROJECT_ROOT)" 2>/dev/null || true
rsync -e "${SSH_CMD}" -avz --exclude '_runs' --exclude '__pycache__' --exclude '.git' --exclude '*.odb' \
    "$PROJECT_ROOT/" "$SERVER:$PROJECT_ROOT/" || { echo "rsync failed"; exit 1; }

if $ALL_CONDITIONS; then
    CONDITIONS=("Commensal:Static" "Commensal:HOBIC" "Dysbiotic:Static" "Dysbiotic:HOBIC")
else
    CONDITIONS=("Dysbiotic:HOBIC")
fi

for cond_cult in "${CONDITIONS[@]}"; do
    cond="${cond_cult%%:*}"
    cult="${cond_cult##*:}"
    out_dir="${MAIN_DIR}/_runs/jax_sigma025_${cond}_${cult}_${TS}"
    log="${LOG_BASE}/${cond}_${cult}.log"
    echo ""
    echo "Launching ${cond}_${cult} (sigma=0.25, 500p, 30st) on $SERVER in background..."
    $SSH_CMD "$SERVER" "cd $MAIN_DIR && mkdir -p $out_dir $LOG_BASE && \
        nohup env LD_LIBRARY_PATH= JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0 $REMOTE_PYTHON estimate_reduced_nishioka_jax.py \
        --condition $cond --cultivation $cult \
        --n-particles 500 --max-stages 30 \
        --sigma-scale 0.25 --use-exp-init \
        --output-dir $out_dir \
        > $log 2>&1 &"
    echo "  Log: $log"
    sleep 2
done

echo ""
echo "=============================================="
echo " 投入完了。ログ確認: ssh $SERVER \"tail -f $LOG_BASE/*.log\""
echo " 結果: $SERVER:${MAIN_DIR}/_runs/jax_sigma025_*_${TS}/"
echo "=============================================="
