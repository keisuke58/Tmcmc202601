#!/bin/bash
# =============================================================================
# JAX ODE + NUTS のテスト／本番実行
# DeepONet 不要、サロゲート誤差ゼロで NUTS が使える
# =============================================================================
# Usage:
#   bash run_jax_ode_nuts.sh --test              # テスト (50p, 500 steps)
#   bash run_jax_ode_nuts.sh --production         # 本番 (200p, 2500 steps)
#   bash run_jax_ode_nuts.sh --test --condition Commensal --cultivation HOBIC
# =============================================================================
# Requires: JAX (e.g. klempt_fem conda env)
# GPU: jax[cuda12] 入りなら自動で GPU 使用（高速化）
# =============================================================================
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/nishioka/IKM_Hiwi/Tmcmc202601}"
cd "$PROJECT_ROOT/data_5species/main"

# JAX 環境（なければ python3 で試す）
PYTHON="${PYTHON:-$HOME/.pyenv/versions/miniconda3-latest/envs/klempt_fem/bin/python}"
if ! command -v "$PYTHON" &>/dev/null; then
    PYTHON="python3"
fi

MODE="${1:-}"
shift || true

# デフォルト条件（--condition/--cultivation で上書き可）
CONDITION="${CONDITION:-Dysbiotic}"
CULTIVATION="${CULTIVATION:-HOBIC}"

if [ "$MODE" = "--test" ]; then
    echo "=========================================="
    echo " JAX ODE + NUTS — テスト (50p, 500 steps)"
    echo " Condition: $CONDITION $CULTIVATION"
    echo "=========================================="
    $PYTHON estimate_reduced_nishioka_jax.py \
        --condition "$CONDITION" --cultivation "$CULTIVATION" \
        --quick --use-exp-init \
        "$@"

elif [ "$MODE" = "--production" ]; then
    TS=$(date +%Y%m%d_%H%M%S)
    OUT="_runs/jax_ode_nuts_${CONDITION}_${CULTIVATION}_${TS}"
    echo "=========================================="
    echo " JAX ODE + NUTS — 本番 (200p, 2500 steps)"
    echo " Condition: $CONDITION $CULTIVATION"
    echo " Output: $OUT"
    echo "=========================================="
    $PYTHON estimate_reduced_nishioka_jax.py \
        --condition "$CONDITION" --cultivation "$CULTIVATION" \
        --n-particles 200 --use-exp-init \
        --output-dir "$OUT" \
        "$@"
    echo ""
    echo "Results: $OUT"

else
    echo "Usage: bash run_jax_ode_nuts.sh --test | --production [options]"
    echo ""
    echo "  --test       50 particles, 500 ODE steps (数分)"
    echo "  --production 200 particles, 2500 steps (30分〜1時間)"
    echo ""
    echo "Options: --condition, --cultivation, --lambda-pg, --sigma-scale, etc."
    echo ""
    echo "Example:"
    echo "  bash run_jax_ode_nuts.sh --test"
    echo "  bash run_jax_ode_nuts.sh --production --condition Dysbiotic --cultivation HOBIC"
    exit 1
fi
