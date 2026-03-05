#!/bin/bash
# =============================================================================
# JAX ODE + NUTS × 4 GPU を stuttgart で実行
# =============================================================================
# Prerequisites (stuttgart):
#   - klempt_fem 環境: conda create -n klempt_fem python=3.10
#     pip install jax[cuda12] equinox matplotlib numpy
#
# Usage:
#   bash run_jax_ode_4gpu_stuttgart.sh
#   bash run_jax_ode_4gpu_stuttgart.sh --sync-only
#   SERVER=stuttgart02 bash run_jax_ode_4gpu_stuttgart.sh
# =============================================================================
# fifawc から実行時: conda deactivate してから実行すると SSH が安定
# =============================================================================
set -euo pipefail

SSH_CMD="${SSH_CMD:-/usr/bin/ssh}"
[[ ! -x "$SSH_CMD" ]] && SSH_CMD="ssh"

PROJECT_ROOT="${PROJECT_ROOT:-/home/nishioka/IKM_Hiwi/Tmcmc202601}"
MAIN_DIR="${PROJECT_ROOT}/data_5species/main"
SERVER="${SERVER:-stuttgart01}"
PYTHON="${PYTHON:-$HOME/.pyenv/versions/miniconda3-latest/envs/klempt_fem/bin/python}"
SYNC_ONLY=false
CONDITION="Dysbiotic"
CULTIVATION="HOBIC"
N_PARTICLES=200
QUICK=""

while [[ "${1:-}" == --* ]]; do
    case "$1" in
        --sync-only)   SYNC_ONLY=true; shift ;;
        --condition)   shift; CONDITION="$1"; shift ;;
        --cultivation) shift; CULTIVATION="$1"; shift ;;
        --n-particles) shift; N_PARTICLES="$1"; shift ;;
        --quick)       QUICK="--quick"; shift ;;
        *) shift ;;
    esac
done

echo "=============================================="
echo " JAX ODE × 4 GPU on Stuttgart"
echo " $(date)"
echo " Server: $SERVER"
echo " Condition: $CONDITION $CULTIVATION"
echo "=============================================="

# Phase 0: rsync
echo ""
echo "=== Phase 0: Syncing to $SERVER ==="
$SSH_CMD "$SERVER" "mkdir -p $(dirname "$PROJECT_ROOT")" 2>/dev/null || true
rsync -avz --exclude '_runs' --exclude '__pycache__' --exclude '.git' --exclude '*.odb' \
    "$PROJECT_ROOT/" "$SERVER:$PROJECT_ROOT/" || echo "  [WARN] rsync failed"

$SYNC_ONLY && { echo "Sync only. Done."; exit 0; }

# Phase 1: stuttgart で 4 GPU 実行
echo ""
echo "=== Phase 1: Running 4-GPU TMCMC on $SERVER ==="
# REMOTE_PYTHON で stuttgart 用 Python を指定可能（例: /home/nishioka/miniconda3/envs/klempt_fem/bin/python）
REMOTE_PY="${REMOTE_PYTHON:-/home/nishioka/miniforge3/envs/klempt_fem/bin/python}"
$SSH_CMD "$SERVER" "cd $MAIN_DIR && PYTHON=$REMOTE_PY bash run_jax_ode_4gpu.sh \
    --condition $CONDITION --cultivation $CULTIVATION \
    --n-particles $N_PARTICLES $QUICK" 2>&1 | tee /tmp/jax_ode_4gpu_${SERVER}.log

# Phase 2: 結果をローカルに同期
echo ""
echo "=== Phase 2: Syncing results back ==="
LATEST_RUN=$($SSH_CMD "$SERVER" "ls -td $MAIN_DIR/_runs/jax_ode_4gpu_${CONDITION}_${CULTIVATION}_* 2>/dev/null | head -1")
if [[ -n "$LATEST_RUN" ]]; then
    rsync -avz "$SERVER:$LATEST_RUN/" "$MAIN_DIR/_runs/$(basename "$LATEST_RUN")/" 2>/dev/null || true
    echo "  Synced: $(basename "$LATEST_RUN")"
fi

echo ""
echo "=============================================="
echo " Done: $(date)"
echo "=============================================="
