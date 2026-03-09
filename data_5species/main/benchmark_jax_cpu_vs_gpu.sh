#!/bin/bash
# =============================================================================
# JAX ODE + NUTS: CPU vs GPU 実行時間比較
# ローカルまたはリモート（vancouver 等）で実行可能
# =============================================================================
# Usage:
#   bash benchmark_jax_cpu_vs_gpu.sh                    # ローカル実行
#   bash benchmark_jax_cpu_vs_gpu.sh vancouver           # vancouver で実行
#   SERVER=vancouver bash benchmark_jax_cpu_vs_gpu.sh   # 同上
#   SERVER=stuttgart02 bash benchmark_jax_cpu_vs_gpu.sh  # stuttgart02 で実行
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MAIN_DIR="$SCRIPT_DIR"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REMOTE_PYTHON="${REMOTE_PYTHON:-/home/nishioka/miniforge3/envs/klempt_fem2/bin/python3}"
SSH_CMD="${SSH_CMD:-/usr/bin/ssh}"
[[ ! -x "$SSH_CMD" ]] && SSH_CMD="ssh"

# 第1引数または SERVER 環境変数
SERVER="${1:-${SERVER:-}}"
RUN_REMOTE=false
if [[ -n "$SERVER" ]]; then
    RUN_REMOTE=true
fi

run_cmd() {
    if $RUN_REMOTE; then
        $SSH_CMD -o ConnectTimeout=5 "$SERVER" "cd $MAIN_DIR && PYTHONUNBUFFERED=1 $1"
    else
        (cd "$MAIN_DIR" && PYTHONUNBUFFERED=1 eval "$1")
    fi
}

check_server() {
    if ! $SSH_CMD -o ConnectTimeout=5 "$SERVER" "hostname" &>/dev/null; then
        echo "ERROR: $SERVER に接続できません。SSH 設定を確認してください。"
        exit 1
    fi
    echo "  $SERVER: $(run_cmd 'hostname')"
    run_cmd "nvidia-smi -L 2>/dev/null | head -2" || true
}

sync_to_remote() {
    REMOTE_DIR="$(dirname "$PROJECT_ROOT")"
    echo "  rsync to $SERVER ..."
    $SSH_CMD "$SERVER" "mkdir -p $REMOTE_DIR" 2>/dev/null || true
    rsync -avz --exclude '_runs' --exclude '__pycache__' --exclude '.git' --exclude '*.odb' \
        "$PROJECT_ROOT/" "$SERVER:$PROJECT_ROOT/" || { echo "  [WARN] rsync failed"; exit 1; }
}

echo "=============================================="
echo " JAX ODE + NUTS: CPU vs GPU ベンチマーク"
echo " $(date)"
if $RUN_REMOTE; then
    echo " Server: $SERVER"
    check_server
    sync_to_remote
else
    echo " ローカル実行"
    PYTHON="${PYTHON:-python3}"
    echo " Python: $(which $PYTHON 2>/dev/null || echo $PYTHON)"
    nvidia-smi -L 2>/dev/null | head -2 || true
fi
BENCH_OPTS="${BENCH_OPTS:---benchmark}"
echo " Mode: $BENCH_OPTS (20p/500steps/3stages or --quick for 50p)"
echo "=============================================="

TS=$(date +%Y%m%d_%H%M%S)
OUT_CPU="${MAIN_DIR}/_runs/bench_cpu_${TS}"
OUT_GPU="${MAIN_DIR}/_runs/bench_gpu_${TS}"

if $RUN_REMOTE; then
    run_cmd "mkdir -p $OUT_CPU $OUT_GPU"
fi

# CPU 実行
echo ""
echo "=== [1/2] CPU 実行 ==="
if $RUN_REMOTE; then
    run_cmd "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= XLA_PYTHON_CLIENT_PREALLOCATE=false $REMOTE_PYTHON estimate_reduced_nishioka_jax.py \
        --condition Dysbiotic --cultivation HOBIC $BENCH_OPTS \
        --device cpu --output-dir $OUT_CPU 2>&1" | tee /tmp/bench_cpu_${TS}.log
else
    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= XLA_PYTHON_CLIENT_PREALLOCATE=false python3 estimate_reduced_nishioka_jax.py \
        --condition Dysbiotic --cultivation HOBIC $BENCH_OPTS \
        --device cpu --output-dir "$OUT_CPU" 2>&1 | tee /tmp/bench_cpu_${TS}.log
fi
CPU_TIME=$(grep 'total_time=' /tmp/bench_cpu_${TS}.log 2>/dev/null | sed -n 's/.*total_time=\([0-9.]*\).*/\1/p' | tail -1 || echo "")
echo "  CPU total_time: ${CPU_TIME:-N/A}s"

# GPU 実行
echo ""
echo "=== [2/2] GPU 実行 ==="
# GPU 0 を使用（GPU 1 が使用中でも 0 は空いている想定）
if $RUN_REMOTE; then
    run_cmd "JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0 $REMOTE_PYTHON estimate_reduced_nishioka_jax.py \
        --condition Dysbiotic --cultivation HOBIC $BENCH_OPTS \
        --device gpu --output-dir $OUT_GPU 2>&1" | tee /tmp/bench_gpu_${TS}.log
else
    JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0 python3 estimate_reduced_nishioka_jax.py \
        --condition Dysbiotic --cultivation HOBIC $BENCH_OPTS \
        --device gpu --output-dir "$OUT_GPU" 2>&1 | tee /tmp/bench_gpu_${TS}.log
fi
GPU_TIME=$(grep 'total_time=' /tmp/bench_gpu_${TS}.log 2>/dev/null | sed -n 's/.*total_time=\([0-9.]*\).*/\1/p' | tail -1 || echo "")
echo "  GPU total_time: ${GPU_TIME:-N/A}s"

# 比較
echo ""
echo "=============================================="
echo " 結果"
echo "=============================================="
if [[ -n "$CPU_TIME" && -n "$GPU_TIME" ]]; then
    RATIO=$(echo "scale=2; $CPU_TIME / $GPU_TIME" | bc 2>/dev/null || echo "N/A")
    echo "  CPU: ${CPU_TIME}s"
    echo "  GPU: ${GPU_TIME}s"
    echo "  Speedup (CPU/GPU): ${RATIO}x"
else
    echo "  CPU: ${CPU_TIME:-parse failed}s"
    echo "  GPU: ${GPU_TIME:-parse failed}s"
fi
echo "=============================================="
