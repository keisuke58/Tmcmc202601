#!/bin/bash
# =============================================================================
# JAX ODE + NUTS を 4 条件 × 4 サーバ並列で stuttgart に投入
# fifawc（ローカル）から実行。conda deactivate してから実行すると SSH が安定
# =============================================================================
# Prerequisites (stuttgart01-03 各サーバ):
#   conda create -n klempt_fem python=3.10 -y
#   conda activate klempt_fem && pip install jax[cuda12] equinox matplotlib numpy
#
# Usage:
#   cd Tmcmc202601/data_5species/main
#   bash run_jax_ode_4parallel_stuttgart.sh
#   bash run_jax_ode_4parallel_stuttgart.sh --sync-only
#   bash run_jax_ode_4parallel_stuttgart.sh --quick
#   SERVERS="stuttgart01 stuttgart02 stuttgart03" bash run_jax_ode_4parallel_stuttgart.sh
# =============================================================================
set -euo pipefail

# OpenSSL 競合回避: conda 環境では /usr/bin/ssh を明示使用
SSH_CMD="${SSH_CMD:-/usr/bin/ssh}"
[[ ! -x "$SSH_CMD" ]] && SSH_CMD="ssh"

PROJECT_ROOT="${PROJECT_ROOT:-/home/nishioka/IKM_Hiwi/Tmcmc202601}"
MAIN_DIR="${PROJECT_ROOT}/data_5species/main"
# stuttgart は miniforge3 を使用（conda create で作成した場合）
# パスはリモート（stuttgart）のものを想定
REMOTE_PYTHON="${REMOTE_PYTHON:-/home/nishioka/miniforge3/envs/klempt_fem/bin/python}"
SERVERS="${SERVERS:-stuttgart01 stuttgart02 stuttgart03}"
SYNC_ONLY=false
N_PARTICLES=200
QUICK=""

while [[ "${1:-}" == --* ]]; do
    case "$1" in
        --sync-only)   SYNC_ONLY=true; shift ;;
        --quick)       QUICK="--quick"; shift ;;
        --n-particles) shift; N_PARTICLES="$1"; shift ;;
        --servers)     shift; SERVERS="${1:-$SERVERS}"; shift ;;
        *) shift ;;
    esac
done

# 4 条件: (condition, cultivation)
CONDITIONS=(
    "Commensal:Static"
    "Commensal:HOBIC"
    "Dysbiotic:Static"
    "Dysbiotic:HOBIC"
)
read -ra SRV_ARRAY <<< "$SERVERS"
N_SRV=${#SRV_ARRAY[@]}

echo "=============================================="
echo " JAX ODE + NUTS × 4 並列 on Stuttgart"
echo " $(date)"
echo " Servers: $SERVERS"
echo " N_PARTICLES: $N_PARTICLES"
echo "=============================================="

# Phase 0: rsync 全サーバに同期
echo ""
echo "=== Phase 0: Syncing to Stuttgart ==="
REMOTE_DIR="$(dirname "$PROJECT_ROOT")"
for srv in $SERVERS; do
    echo "  $srv: rsync ..."
    $SSH_CMD "$srv" "mkdir -p $REMOTE_DIR" 2>/dev/null || true
    rsync -avz --exclude '_runs' --exclude '__pycache__' --exclude '.git' --exclude '*.odb' \
        "$PROJECT_ROOT/" "$srv:$PROJECT_ROOT/" || echo "  [WARN] rsync to $srv failed"
done

$SYNC_ONLY && { echo "Sync only. Done."; exit 0; }

TS=$(date +%Y%m%d_%H%M%S)
LOG_BASE="${MAIN_DIR}/_runs/jax_ode_4parallel_stuttgart_${TS}"
mkdir -p "$LOG_BASE"

# Phase 1: 4 条件を並列投入（1条件1サーバ、サーバ数不足時は循環）
echo ""
echo "=== Phase 1: Launching 4 conditions in parallel ==="
PIDS=()
for i in "${!CONDITIONS[@]}"; do
    cond_cult="${CONDITIONS[$i]}"
    cond="${cond_cult%%:*}"
    cult="${cond_cult##*:}"
    srv="${SRV_ARRAY[$((i % N_SRV))]}"
    out_dir="${MAIN_DIR}/_runs/jax_ode_${cond}_${cult}_${TS}"
    log="${LOG_BASE}/${cond}_${cult}.log"

    echo "  [$((i+1))/4] ${cond}_${cult} -> $srv"
    $SSH_CMD "$srv" "cd $MAIN_DIR && mkdir -p $out_dir && \
        CUDA_VISIBLE_DEVICES=0 $REMOTE_PYTHON estimate_reduced_nishioka_jax.py \
        --condition $cond --cultivation $cult \
        --n-particles $N_PARTICLES --use-exp-init \
        --output-dir $out_dir \
        $QUICK \
        2>&1 | tee $log" &
    PIDS+=($!)
done

echo ""
echo "Waiting for 4 jobs..."
for pid in "${PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
done

# Phase 2: 結果をローカルに同期
echo ""
echo "=== Phase 2: Syncing results back ==="
for i in "${!CONDITIONS[@]}"; do
    cond_cult="${CONDITIONS[$i]}"
    cond="${cond_cult%%:*}"
    cult="${cond_cult##*:}"
    srv="${SRV_ARRAY[$((i % N_SRV))]}"
    remote_dir="${MAIN_DIR}/_runs/jax_ode_${cond}_${cult}_${TS}"
    local_dir="${MAIN_DIR}/_runs/jax_ode_${cond}_${cult}_${TS}"
    mkdir -p "$local_dir"
    rsync -avz "$srv:$remote_dir/" "$local_dir/" 2>/dev/null || echo "  [WARN] sync ${cond}_${cult} failed"
done
for srv in $SERVERS; do
    rsync -avz "$srv:$LOG_BASE/" "$LOG_BASE/" 2>/dev/null || true
done

echo ""
echo "=============================================="
echo " Done: $(date)"
echo " Logs: $LOG_BASE"
echo " Results: ${MAIN_DIR}/_runs/jax_ode_*_${TS}/"
echo "=============================================="
