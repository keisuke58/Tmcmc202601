#!/bin/bash
# =============================================================================
# JAX ODE + NUTS を 4 GPU 並列（4 チェーン）で実行
# =============================================================================
# Usage:
#   bash run_jax_ode_4gpu.sh                    # ローカル 4 GPU
#   bash run_jax_ode_4gpu.sh --condition Dysbiotic --cultivation HOBIC
#   bash run_jax_ode_4gpu.sh --n-particles 200 --quick
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MAIN_DIR="$SCRIPT_DIR"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${PYTHON:-$HOME/.pyenv/versions/miniconda3-latest/envs/klempt_fem2/bin/python}"

CONDITION="Dysbiotic"
CULTIVATION="HOBIC"
N_PARTICLES=200
QUICK=""
EXTRA_ARGS=()

while [[ "${1:-}" == --* ]] || [[ "${1:-}" =~ ^[A-Za-z] ]]; do
    case "$1" in
        --condition)     shift; CONDITION="$1"; shift ;;
        --cultivation)   shift; CULTIVATION="$1"; shift ;;
        --n-particles)   shift; N_PARTICLES="$1"; shift ;;
        --quick)         QUICK="--quick"; shift ;;
        --)              shift; break ;;
        *)               EXTRA_ARGS+=("$1"); shift ;;
    esac
done

TS=$(date +%Y%m%d_%H%M%S)
OUT_BASE="${MAIN_DIR}/_runs/jax_ode_4gpu_${CONDITION}_${CULTIVATION}_${TS}"
mkdir -p "$OUT_BASE"

echo "=============================================="
echo " JAX ODE + NUTS × 4 GPU (4 chains)"
echo " $(date)"
echo " Condition: $CONDITION $CULTIVATION"
echo " N_PARTICLES: $N_PARTICLES"
echo " Output: $OUT_BASE"
echo "=============================================="

# GPU 0/1 が OOM やハードウェアエラーの場合: GPU_IDS="2 3" で 2 チェーンに縮小
# XLA_PYTHON_CLIENT_PREALLOCATE=false でメモリ節約
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
GPU_IDS=(${GPU_IDS:-0 1 2 3})
N_CHAINS=${#GPU_IDS[@]}

# 4 チェーンを GPU に振り分け
PIDS=()
for idx in "${!GPU_IDS[@]}"; do
    i="${GPU_IDS[$idx]}"
    CHAIN_DIR="${OUT_BASE}/chain_${idx}"
    mkdir -p "$CHAIN_DIR"
    SEED=$((42 + i * 1000))
    LOG="${OUT_BASE}/chain_${idx}.log"

    echo ""
    echo "  Chain $idx -> GPU $i (seed=$SEED)"
    JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=$i $PYTHON "$MAIN_DIR/estimate_reduced_nishioka_jax.py" \
        --device gpu \
        --condition "$CONDITION" \
        --cultivation "$CULTIVATION" \
        --n-particles "$N_PARTICLES" \
        --seed "$SEED" \
        --use-exp-init \
        --output-dir "$CHAIN_DIR" \
        $QUICK \
        "${EXTRA_ARGS[@]}" \
        > "$LOG" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "Waiting for $N_CHAINS chains..."
for pid in "${PIDS[@]}"; do
    wait "$pid" || true
done

# マージ: 全チェーンのサンプルを結合
echo ""
echo "=== Merging chains ==="
COMBINED_SAMPLES="${OUT_BASE}/samples_combined.npy"
COMBINED_LOGL="${OUT_BASE}/logL_combined.npy"

$PYTHON - << EOF
import numpy as np
from pathlib import Path

out_base = Path("$OUT_BASE")
n_chains = $N_CHAINS
samples_list, logL_list = [], []
for i in range(n_chains):
    chain_dir = out_base / f"chain_{i}"
    s = np.load(chain_dir / "samples.npy")
    l = np.load(chain_dir / "logL.npy")
    samples_list.append(s)
    logL_list.append(l)

samples = np.concatenate(samples_list, axis=0)
logL = np.concatenate(logL_list, axis=0)
np.save(out_base / "samples_combined.npy", samples)
np.save(out_base / "logL_combined.npy", logL)

# R-hat 簡易計算（パラメータごと）
n_chains, n_per = len(samples_list), len(samples_list[0])
chain_samples = np.array(samples_list)  # (4, n_per, 20)
n_params = chain_samples.shape[2]
rhat = []
for p in range(n_params):
    x = chain_samples[:, :, p]  # (4, n_per)
    m = x.mean(axis=1)
    M = m.mean()
    B = n_per * ((m - M)**2).sum() / max(n_chains - 1, 1)
    s2 = ((x - m[:, None])**2).sum(axis=1) / (n_per - 1)
    W = s2.mean()
    var_hat = (n_per - 1)/n_per * W + B/n_per
    rhat.append(np.sqrt(var_hat / W) if W > 1e-12 else 1.0)

rhat_max = max(rhat)
print(f"  Combined: {samples.shape[0]} samples")
print(f"  R-hat max: {rhat_max:.4f}")
np.save(out_base / "rhat.npy", np.array(rhat))
EOF

echo ""
echo "=============================================="
echo " Done: $(date)"
echo " Results: $OUT_BASE"
echo "  chain_0/ .. chain_3/  : per-chain results"
echo "  samples_combined.npy  : merged samples"
echo "  rhat.npy              : R-hat per parameter"
echo "=============================================="
