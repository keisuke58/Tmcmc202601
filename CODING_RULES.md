# IKM_Hiwi プロジェクト - コーディングルール詳細

このドキュメントは、IKM_Hiwiプロジェクト（TMCMC + TSM階層ベイズ推定）のコーディング標準を定義します。

## 📋 目次

1. [基本方針](#基本方針)
2. [コードスタイル](#コードスタイル)
3. [ドキュメント](#ドキュメント)
4. [エラーハンドリング](#エラーハンドリング)
5. [ロギング](#ロギング)
6. [型ヒント](#型ヒント)
7. [Numba最適化](#numba最適化)
8. [NumPy使用規則](#numpy使用規則)
9. [設定管理](#設定管理)
10. [テスト](#テスト)
11. [Git管理](#git管理)

---

## 基本方針

### 原則
- **可読性優先**: 科学計算コードは将来の自分と他者が理解できることが重要
- **保守性**: モジュール化と明確な責任分離
- **パフォーマンス**: Numba最適化を適切に使用
- **検証可能性**: テスト可能なコード構造

---

## コードスタイル

### Python バージョン
- **最小バージョン**: Python 3.9+
- **推奨**: Python 3.10 または 3.11

### インデント・フォーマット
- **インデント**: スペース4つ（タブ禁止）
- **行の長さ**: 最大100文字
  - 科学計算の長い式は例外可
  - 複数行に分割する場合は適切にインデント

### 命名規則

```python
# 関数・変数: snake_case
def compute_effective_sample_size():
    max_iterations = 100

# クラス: PascalCase
class TMCMCSampler:
    pass

# 定数: UPPER_SNAKE_CASE
MAX_ITERATIONS = 1000
TARGET_ESS_RATIO = 0.5

# プライベート: _leading_underscore
def _internal_helper():
    pass
```

### インポート順序

```python
# 1. 標準ライブラリ
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict

# 2. サードパーティ（アルファベット順）
import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from scipy import optimize

# 3. ローカルモジュール（相対パス可）
from src.config import get_config
from src.tsm import TSM
from .utils import validate_theta
```

---

## ドキュメント

### Docstring形式
**NumPyスタイル**を標準とする。

```python
def solve_tsm(
    self,
    theta: np.ndarray,
    t_span: Optional[Tuple[float, float]] = None
) -> TSMResult:
    """
    Solve Time-Separated Mechanics problem.

    Computes mean trajectory and variance propagation via 1st-order Taylor
    expansion of the PDE solution with respect to parameter uncertainty.

    Parameters
    ----------
    theta : np.ndarray, shape (14,)
        Parameter vector: [a11, a12, a22, b1, b2, a33, a34, a44, b3, b4,
                           a13, a14, a23, a24]
    t_span : tuple of float, optional
        Time span (t_start, t_end). If None, uses config defaults.

    Returns
    -------
    TSMResult
        Result object containing:
        - t_array : np.ndarray
            Time points
        - mu : np.ndarray, shape (n_times, n_states)
            Mean trajectory (deterministic solution)
        - sigma2 : np.ndarray, shape (n_times, n_states)
            Variance at each time/state

    Raises
    ------
    RuntimeError
        If Newton solver encounters NaN values or fails to converge
    ValueError
        If theta has incorrect shape or contains invalid values

    Notes
    -----
    Uses analytical sensitivity computation when Numba is available,
    falls back to numerical differentiation otherwise.

    The variance propagation follows:
        σ²(t) = Σₖ (∂g/∂θₖ)² Var(θₖ)

    Examples
    --------
    >>> tsm = TSM(config)
    >>> theta = np.array([0.8, 2.0, 1.0, 0.1, 0.2, 1.5, 1.0, 2.0, 0.3, 0.4,
    ...                   2.0, 1.0, 2.0, 1.0])
    >>> result = tsm.solve_tsm(theta)
    >>> print(f"Final variance: {result.sigma2[-1, 0]}")
    """
```

### モジュールレベルDocstring

```python
"""
Time-Separated Mechanics (TSM) implementation.

This module implements the TSM method for efficient uncertainty quantification
in biofilm formation simulations. It uses analytical sensitivity computation
to propagate parameter uncertainty through the PDE system.

References
----------
.. [1] Fritsch et al. (2025), "Hierarchical Bayesian Inference for Biofilm..."
"""
```

---

## エラーハンドリング

### 禁止事項
- ❌ 裸の `except:` ブロック
- ❌ エラーを無視する（`pass` のみ）

### 推奨パターン

```python
# ❌ 悪い例
try:
    result = solver.run(theta)
except:
    return -1e20

# ✅ 良い例
try:
    result = solver.run(theta)
except (RuntimeError, np.linalg.LinAlgError) as e:
    logger.warning(
        f"Solver failed for theta={theta[:3]}... "
        f"Error: {type(e).__name__}: {e}"
    )
    return -1e20
except ValueError as e:
    logger.error(f"Invalid input: {e}")
    raise  # 再発生させる
```

### 入力検証

```python
def validate_theta(
    theta: np.ndarray,
    bounds: Optional[List[Tuple[float, float]]] = None
) -> np.ndarray:
    """
    Validate parameter vector.

    Parameters
    ----------
    theta : np.ndarray
        Parameter vector
    bounds : list of tuple, optional
        (min, max) bounds for each parameter

    Returns
    -------
    np.ndarray
        Validated theta (as float64)

    Raises
    ------
    ValueError
        If theta has incorrect shape or is outside bounds
    """
    theta = np.asarray(theta, dtype=np.float64)
    
    if theta.shape != (14,):
        raise ValueError(
            f"theta must have shape (14,), got {theta.shape}"
        )
    
    if np.any(~np.isfinite(theta)):
        raise ValueError("theta contains non-finite values")
    
    if bounds is not None:
        for i, (low, high) in enumerate(bounds):
            if not (low <= theta[i] <= high):
                raise ValueError(
                    f"theta[{i}]={theta[i]:.4f} outside bounds [{low}, {high}]"
                )
    
    return theta
```

---

## ロギング

### 基本設定

```python
import logging

# モジュールレベルでloggerを取得
logger = logging.getLogger(__name__)

# 使用例
logger.debug("Detailed debugging information")
logger.info("Starting TMCMC simulation with N0=%d", n_particles)
logger.warning("Low ESS detected: %.2f (target: %.2f)", ess, target_ess)
logger.error("Convergence failed after %d iterations", max_iter)
```

### ログレベル
- **DEBUG**: 詳細なデバッグ情報
- **INFO**: 一般的な情報（実行開始、完了など）
- **WARNING**: 警告（低ESS、収束が遅いなど）
- **ERROR**: エラー（失敗、例外など）

### 禁止事項
- ❌ 本番コードでの `print()` 使用（デバッグ時のみ可）

---

## 型ヒント

### 基本使用

```python
from typing import Tuple, Optional, List, Dict
import numpy.typing as npt

def compute_ess(
    log_weights: npt.NDArray[np.float64],
    delta_beta: float
) -> float:
    """Compute effective sample size."""
    ...

def run_tmcmc(
    n_particles: int,
    n_stages: int,
    log_likelihood: callable,
    prior: Optional[Dict] = None
) -> Tuple[npt.NDArray[np.float64], Dict]:
    """
    Run TMCMC algorithm.
    
    Returns
    -------
    samples : np.ndarray
        Posterior samples
    diagnostics : dict
        Convergence diagnostics
    """
    ...
```

### Numba関数
Numba関数には型ヒントを追加できないが、コメントで型を明記：

```python
@njit(cache=True, fastmath=True)
def compute_Q_vector_numba(
    phi,      # np.ndarray, shape (4,)
    psi,      # np.ndarray, shape (4,)
    c_val,    # float
    alpha_val # float
) -> np.ndarray:  # shape (10,)
    """Numba-accelerated Q-vector computation."""
    ...
```

---

## Numba最適化

### 使用指針
- **パフォーマンスクリティカルな関数**: `@njit` を使用
- **数値計算カーネル**: ループが多い関数を優先

### デコレータ設定

```python
from numba import njit

@njit(cache=True, fastmath=True)
def compute_jacobian_numba(phi, psi, c_val, alpha_val, Eta_vec):
    """
    Numba-accelerated Jacobian computation.
    
    Notes
    -----
    - cache=True: コンパイル結果をキャッシュ
    - fastmath=True: 浮動小数点演算の最適化（精度に注意）
    """
    ...
```

### 制約
- Pythonオブジェクト（リスト、辞書など）は使用不可
- NumPy配列のみ
- 例外処理は限定的

### フォールバック

```python
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

if HAS_NUMBA:
    @njit(cache=True)
    def compute_Q_vector(phi, psi, ...):
        ...
else:
    def compute_Q_vector(phi, psi, ...):
        """NumPy fallback when Numba unavailable."""
        ...
```

---

## NumPy使用規則

### 配列作成
```python
# ✅ 既存配列の再利用（メモリ効率）
theta = np.asarray(theta, dtype=np.float64)

# ❌ 不要なコピー
theta = np.array(theta)  # 既に配列の場合はコピーされる
```

### 数値安定性

```python
# ✅ オーバーフロー防止
logL_eff = logL * logL_scale
x = delta_beta * (logL_eff - np.max(logL_eff))
w_unnorm = np.exp(x)

# ❌ 危険（オーバーフロー）
w_unnorm = np.exp(delta_beta * logL_eff)
```

### 非有限値チェック

```python
# ✅ チェック
if np.any(~np.isfinite(result)):
    raise RuntimeError("Non-finite values detected")

# NaN/Infの検出
if np.isnan(result).any():
    logger.warning("NaN detected in result")
```

---

## 設定管理

### 集中管理
- すべての設定は `config.py` に集約
- マジックナンバーを避ける

```python
# config.py
@dataclass(frozen=True)
class TMCMCDefaults:
    n_particles: int = 2000
    n_stages: int = 30
    target_ess_ratio: float = 0.5
    min_delta_beta: float = 0.02
    max_iterations: int = 100  # マジックナンバーを定数化
```

### デバッグモード

```python
def get_config(debug: bool = False):
    """
    Get configuration for simulation.
    
    Parameters
    ----------
    debug : bool
        If True, use fast settings for testing
    """
    if debug:
        return {
            "dt": 1e-4,
            "maxtimestep": 80,
            "n_particles": 10,
            ...
        }
    else:
        return {
            "dt": 1e-5,
            "maxtimestep": 2500,
            "n_particles": 500,
            ...
        }
```

---

## テスト

### ファイル構造
```
tests/
├── __init__.py
├── test_solver.py
├── test_tsm.py
├── test_tmcmc.py
├── fixtures/
│   ├── true_params.json
│   └── reference_solutions.npz
└── conftest.py
```

### テスト例

```python
import pytest
import numpy as np
from src.solver_newton import BiofilmNewtonSolver

def test_solver_mass_conservation():
    """Verify Σφᵢ = 1 at all times."""
    solver = BiofilmNewtonSolver(dt=1e-4, maxtimestep=100)
    theta = np.array([0.8, 2.0, 1.0, 0.1, 0.2,
                      1.5, 1.0, 2.0, 0.3, 0.4,
                      2.0, 1.0, 2.0, 1.0])
    t, g = solver.run_deterministic(theta)
    
    # Check mass conservation
    phi_sum = g[:, 0:4].sum(axis=1) + g[:, 4]
    np.testing.assert_allclose(phi_sum, 1.0, atol=1e-6)

def test_tsm_analytical_vs_numerical():
    """Compare analytical and numerical sensitivities."""
    # Implementation...
    pass
```

---

## Git管理

### コミットメッセージ

```
feat: add adaptive ESS targeting to TMCMC
fix: resolve NaN issue in Newton solver
docs: update README with usage examples
refactor: split large file into modules
test: add unit tests for TSM sensitivity
chore: update dependencies
```

### ブランチ戦略
- **master**: 安定版
- **feature/xxx**: 新機能開発
- **fix/xxx**: バグ修正

---

## チェックリスト

新しいコードを書く際のチェックリスト：

- [ ] 関数にdocstringを追加したか
- [ ] 型ヒントを追加したか
- [ ] エラーハンドリングが適切か（裸のexceptなし）
- [ ] ロギングを使用しているか（printなし）
- [ ] マジックナンバーを定数化したか
- [ ] テストを追加したか（可能な場合）
- [ ] コードが100行以内か（長い場合は分割を検討）

---

## 参考資料

- [NumPy Style Guide](https://numpydoc.readthedocs.io/)
- [PEP 8](https://pep8.org/)
- [Numba Documentation](https://numba.readthedocs.io/)

---

**最終更新**: 2026-01-24
