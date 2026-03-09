#!/bin/bash
# =============================================================================
# Stuttgart 各サーバで klempt_fem2 環境をセットアップ（GPU 用）
# =============================================================================
# Usage: 各 stuttgart サーバに SSH して実行
#   ssh stuttgart01
#   bash setup_stuttgart_klempt_fem2.sh
#
# またはリモート実行:
#   ssh stuttgart01 'bash -s' < setup_stuttgart_klempt_fem2.sh
# =============================================================================
set -euo pipefail

ENV_NAME="klempt_fem2"
CONDA_BASE="${CONDA_BASE:-$HOME/miniforge3}"

echo "=== Stuttgart klempt_fem2 セットアップ ==="
echo "CONDA_BASE: $CONDA_BASE"

# 壊れた環境を削除
conda deactivate 2>/dev/null || true
conda env remove -n $ENV_NAME -y 2>/dev/null || true
rm -rf "$CONDA_BASE/envs/$ENV_NAME" 2>/dev/null || true

# 環境作成
conda create -n $ENV_NAME python=3.10 -y
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# pip と依存
conda install pip -y
pip install jax[cuda12] equinox matplotlib numpy pandas requests numba

# 確認
echo ""
echo "=== 確認 ==="
$CONDA_BASE/envs/$ENV_NAME/bin/python3 -c "
import jax
print('JAX:', jax.__version__)
print('Devices:', jax.devices())
d = str(jax.devices())
print('GPU OK' if 'gpu' in d.lower() or 'cuda' in d.lower() else 'CPU only (GPU 未検出)')
"

echo ""
echo "Done. Python: $CONDA_BASE/envs/$ENV_NAME/bin/python3"
