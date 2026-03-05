#!/bin/bash
# =============================================================================
# DeepONet + NUTS を stuttgart01-03 (GPU) で分散実行
# =============================================================================
# Prerequisites (stuttgart 側):
#   - klempt_fem2 環境 (JAX + CUDA) をセットアップ
#   - 例: conda create -n klempt_fem2 python=3.10; pip install jax[cuda12] equinox
#
# Usage:
#   bash run_nuts_stuttgart.sh              # 4条件を3台に振り分け
#   bash run_nuts_stuttgart.sh --sync-only  # プロジェクト同期のみ
#   bash run_nuts_stuttgart.sh --servers "stuttgart01 stuttgart02"
#   PYTHON=/path/to/python bash run_nuts_stuttgart.sh  # stuttgart 用 Python 指定
# =============================================================================
# fifawc から実行時: conda deactivate してから実行すると SSH が安定（OpenSSL 競合回避）
# =============================================================================
set -euo pipefail

SSH_CMD="${SSH_CMD:-/usr/bin/ssh}"
[[ ! -x "$SSH_CMD" ]] && SSH_CMD="ssh"

PROJECT_ROOT="${PROJECT_ROOT:-/home/nishioka/IKM_Hiwi/Tmcmc202601}"
DEEPONET_DIR="${PROJECT_ROOT}/deeponet"
# stuttgart 用 Python（リモートパス）。miniforge3 で作成した場合
PYTHON="${PYTHON:-/home/nishioka/miniforge3/envs/klempt_fem2/bin/python}"

SERVERS="${SERVERS:-stuttgart01 stuttgart02 stuttgart03}"
SYNC_ONLY=false
N_PARTICLES="${N_PARTICLES:-500}"

while [[ "${1:-}" == --* ]]; do
    case "$1" in
        --sync-only)   SYNC_ONLY=true; shift ;;
        --servers)     shift; SERVERS="${1:-$SERVERS}"; shift ;;
        --n-particles) shift; N_PARTICLES="${1:-500}"; shift ;;
        *) shift ;;
    esac
done

echo "=============================================="
echo " DeepONet + NUTS on Stuttgart (GPU)"
echo " $(date)"
echo " Servers: $SERVERS"
echo " N_PARTICLES: $N_PARTICLES"
echo "=============================================="

# Phase 0: rsync プロジェクトを stuttgart に同期
echo ""
echo "=== Phase 0: Syncing project to Stuttgart ==="
REMOTE_DIR="$(dirname "$PROJECT_ROOT")"
for srv in $SERVERS; do
    echo "  $srv: mkdir + rsync ..."
    $SSH_CMD "$srv" "mkdir -p $REMOTE_DIR" 2>/dev/null || true
    rsync -avz --exclude '_runs' --exclude '__pycache__' --exclude '*.pyc' \
        --exclude '.git' --exclude '*.odb' --exclude '*.odb_f' \
        "$PROJECT_ROOT/" "$srv:$PROJECT_ROOT/" || echo "  [WARN] rsync to $srv failed"
done

$SYNC_ONLY && { echo "Sync only. Exiting."; exit 0; }

read -ra SRV_ARRAY <<< "$SERVERS"

# Pre-check: Python/JAX on first server
echo ""
echo "=== Pre-check: Python env on ${SRV_ARRAY[0]} ==="
if ! $SSH_CMD "${SRV_ARRAY[0]}" "test -x $PYTHON" 2>/dev/null; then
    echo "  [WARN] $PYTHON not found on remote. Set PYTHON=... for stuttgart."
    echo "  Example: PYTHON=/path/to/klempt_fem2/bin/python bash run_nuts_stuttgart.sh"
    echo "  Skipping pre-check (will fail at run time if env missing)."
fi

# Phase 1: 4条件をサーバに振り分け（1条件1サーバ、結果は deeponet/ に上書きされるため）
# stuttgart01: Commensal_Static
# stuttgart02: Commensal_HOBIC
# stuttgart03: Dysbiotic_Static
# stuttgart01: Dysbiotic_HOBIC（1が終わってから）
CONDITIONS=("Commensal_Static" "Commensal_HOBIC" "Dysbiotic_Static" "Dysbiotic_HOBIC")
N_SRV=${#SRV_ARRAY[@]}

TS=$(date +%Y%m%d_%H%M%S)
LOG_BASE="${PROJECT_ROOT}/data_5species/_runs/nuts_stuttgart_${TS}"
mkdir -p "$LOG_BASE"

echo ""
echo "=== Phase 1: Launching NUTS on Stuttgart (GPU) ==="
PIDS=()
for i in "${!CONDITIONS[@]}"; do
    cond="${CONDITIONS[$i]}"
    srv="${SRV_ARRAY[$((i % N_SRV))]}"
    log="${LOG_BASE}/${cond}.log"

    echo "  [$((i+1))/4] $cond -> $srv"
    $SSH_CMD "$srv" "cd $DEEPONET_DIR && mkdir -p $LOG_BASE && \
        CUDA_VISIBLE_DEVICES=0 $PYTHON gradient_tmcmc_nuts.py \
        --real --condition $cond --compare-all \
        --n-particles $N_PARTICLES \
        --seed 42 --nuts-max-depth 6 --paper-fig \
        2>&1 | tee $log" &
    PIDS+=($!)
done

echo ""
echo "Waiting for jobs..."
for pid in "${PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
done

echo ""
echo "=== Phase 2: Syncing results back ==="
for srv in $SERVERS; do
    rsync -avz "$srv:$DEEPONET_DIR/nuts_tmcmc_results.json" "$DEEPONET_DIR/nuts_tmcmc_results_${srv}.json" 2>/dev/null || true
    rsync -avz "$srv:$DEEPONET_DIR/nuts_comparison_"*.png "$DEEPONET_DIR/" 2>/dev/null || true
    rsync -avz "$srv:$LOG_BASE/" "$LOG_BASE/" 2>/dev/null || true
done

echo ""
echo "=============================================="
echo " Done: $(date)"
echo " Logs: $LOG_BASE"
echo " Figures: deeponet/nuts_comparison_*.png"
echo "=============================================="
