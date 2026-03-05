#!/bin/bash
# =============================================================================
# JAX GPU 環境セットアップ
# =============================================================================
# Usage:
#   bash setup_jax_gpu.sh              # 新規 conda 環境作成
#   bash setup_jax_gpu.sh --existing  # 既存 klempt_fem に追加
# =============================================================================
set -euo pipefail

EXISTING="${1:-}"

if [ "$EXISTING" = "--existing" ]; then
    PYTHON="${PYTHON:-$HOME/.pyenv/versions/miniconda3-latest/envs/klempt_fem/bin/python}"
    echo "Installing JAX GPU into existing env: $PYTHON"
    "$PYTHON" -m pip install --upgrade pip
    "$PYTHON" -m pip install -r "$(dirname "$0")/requirements-jax-gpu.txt"
else
    ENV_NAME="${ENV_NAME:-klempt_fem_gpu}"
    echo "Creating conda env: $ENV_NAME"
    conda create -n "$ENV_NAME" python=3.11 -y
    PYTHON="$(conda run -n $ENV_NAME which python)"
    "$PYTHON" -m pip install --upgrade pip
    "$PYTHON" -m pip install -r "$(dirname "$0")/requirements-jax-gpu.txt"
    "$PYTHON" -m pip install numpy  # 他依存も必要なら追加
    echo ""
    echo "Activate: conda activate $ENV_NAME"
    echo "Run: PYTHON=$PYTHON bash run_jax_ode_nuts.sh --production"
fi

echo ""
echo "Verify GPU: $PYTHON -c \"import jax; print(jax.devices())\""
