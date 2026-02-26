"""
M3モデル構造の確認

M3は4つのspeciesをモデル化しているが、パラメータは4つだけ（結合項のみ）
M1とM2のパラメータが固定されている可能性を確認
"""

import numpy as np
import sys
import io
from pathlib import Path

# Fix encoding for Windows console
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import MODEL_CONFIGS
from improved1207_paper_jit import get_theta_true

print("=" * 80)
print("M3モデル構造の確認")
print("=" * 80)

# Model configurations
config_M1 = MODEL_CONFIGS["M1"]
config_M2 = MODEL_CONFIGS["M2"]
config_M3 = MODEL_CONFIGS["M3"]

theta_true = get_theta_true()

print("\n1. モデル構造の比較")
print("-" * 80)

print("\nM1:")
print(f"  Active species: {config_M1['active_species']}")
print(f"  Active indices: {config_M1['active_indices']}")
print(f"  Param names: {config_M1['param_names']}")
print(f"  Number of parameters: {len(config_M1['param_names'])}")
print(f"  theta_true values: {theta_true[config_M1['active_indices']]}")

print("\nM2:")
print(f"  Active species: {config_M2['active_species']}")
print(f"  Active indices: {config_M2['active_indices']}")
print(f"  Param names: {config_M2['param_names']}")
print(f"  Number of parameters: {len(config_M2['param_names'])}")
print(f"  theta_true values: {theta_true[config_M2['active_indices']]}")

print("\nM3:")
print(f"  Active species: {config_M3['active_species']}")
print(f"  Active indices: {config_M3['active_indices']}")
print(f"  Param names: {config_M3['param_names']}")
print(f"  Number of parameters: {len(config_M3['param_names'])}")
print(f"  theta_true values: {theta_true[config_M3['active_indices']]}")

print("\n2. パラメータの意味")
print("-" * 80)

print(
    """
M1パラメータ（species 0, 1）:
  a11: species 0 の自己相互作用
  a12: species 0-1 の相互作用
  a22: species 1 の自己相互作用
  b1: species 0 の成長率
  b2: species 1 の成長率

M2パラメータ（species 2, 3）:
  a33: species 2 の自己相互作用
  a34: species 2-3 の相互作用
  a44: species 3 の自己相互作用
  b3: species 2 の成長率
  b4: species 3 の成長率

M3パラメータ（species 0,1,2,3 の結合項）:
  a13: species 0-2 の相互作用
  a14: species 0-3 の相互作用
  a23: species 1-2 の相互作用
  a24: species 1-3 の相互作用
"""
)

print("\n3. 問題点の分析")
print("-" * 80)

print(
    """
M3は4つのspeciesをモデル化しているが、パラメータは4つだけ（結合項のみ）

完全なモデルに必要なパラメータ:
  - 自己相互作用: a11, a22, a33, a44 (4つ)
  - 種内相互作用: a12, a34 (2つ)
  - 種間相互作用: a13, a14, a23, a24 (4つ)
  - 成長率: b1, b2, b3, b4 (4つ)
  合計: 14パラメータ

M3で推定しているパラメータ:
  - 種間相互作用のみ: a13, a14, a23, a24 (4つ)

M1とM2のパラメータは固定されている可能性:
  - M1のパラメータ（a11, a12, a22, b1, b2）はM1の推定結果を使用
  - M2のパラメータ（a33, a34, a44, b3, b4）はM2の推定結果を使用
  - M3では種間相互作用（a13, a14, a23, a24）のみを推定

これは階層的ベイズアプローチで、M1とM2の結果をM3に統合している。
"""
)

print("\n4. 構造ミスマッチの可能性")
print("-" * 80)

print(
    """
問題の可能性:

1. M1とM2の推定誤差がM3に伝播
   - M1とM2のパラメータが真値からずれている場合、M3の適合性が低下

2. 種間相互作用だけでは不十分
   - 4つのspeciesのダイナミクスを4つのパラメータだけで説明するのは困難
   - 特に、M1とM2のパラメータが固定されているため、柔軟性が低い

3. データ生成と推定の不整合
   - データ生成時: theta_trueの全パラメータを使用
   - M3推定時: M1とM2の推定結果 + M3のパラメータのみ
   - この不整合が残差の原因の可能性
"""
)

print("\n5. 確認すべき点")
print("-" * 80)

print(
    """
1. M1とM2の推定結果がtheta_trueにどれだけ近いか
   - M1のMAP/Meanがtheta_true[0:5]に近いか
   - M2のMAP/Meanがtheta_true[5:10]に近いか

2. M3の推定時に使用されるM1/M2の値
   - theta_base_M3の構成を確認
   - M1とM2の推定結果が正しく統合されているか

3. データ生成時のtheta_trueの値
   - M3データ生成時に全14パラメータを使用しているか
   - M3推定時はM1/M2推定結果 + M3パラメータのみか
"""
)

print(f"\n{'='*80}")
print("確認完了")
print(f"{'='*80}")
