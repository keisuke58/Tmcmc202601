#!/bin/bash
# =============================================================================
# vancouver01 で JAX GPU (cuSPARSE) を有効にするセットアップ
# cuSPARSE が見つからない場合に実行
# =============================================================================
# Usage: vancouver01 に SSH して実行
#   ssh vancouver01
#   cd ~/IKM_Hiwi/Tmcmc202601/data_5species/main
#   bash setup_vancouver01_jax_gpu.sh
# =============================================================================
set -euo pipefail

echo "=== vancouver01 JAX GPU セットアップ ==="

# 1. nvidia-cusparse-cu12 を pip でインストール
echo ""
echo "[1] pip install nvidia-cusparse-cu12 ..."
pip install nvidia-cusparse-cu12 nvidia-cublas-cu12

# 2. LD_LIBRARY_PATH を空にして JAX が pip の CUDA を使うようにする
echo ""
echo "[2] 確認: JAX が GPU を認識するか"
env LD_LIBRARY_PATH= JAX_PLATFORMS=cuda python3 -c "
import jax
print('JAX devices:', jax.devices())
assert any('cuda' in str(d).lower() for d in jax.devices()), 'GPU not found!'
print('OK: GPU 利用可能')
"

echo ""
echo "=== セットアップ完了 ==="
echo "実行時は LD_LIBRARY_PATH= を付けるか、~/.bashrc に export LD_LIBRARY_PATH= を追加"
