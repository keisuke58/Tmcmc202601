# 中核4ファイルの「現状精度・改良点・関係性」整理表

## 📊 総合評価表

| ファイル | 現状の正確さ | 数値的・理論的な性格 | 主な改善点（Kaizen） | 伸びしろ | 他ファイルとの関係性 |
|---------|------------|-------------------|-------------------|---------|-------------------|
| **demo_analytical_tsm_with_linearization_jit.py** | ★★★★★ | TSM-ROM（1次線形）<br>平均・分散を解析的に正確計算<br>線形化点管理機能完備 | ・2次TSM対応（∂²G/∂θ²必要）<br>・線形化点の自動再配置（誤差指標ベース）<br>・線形化点更新の最適化（現在は固定間隔） | ★★★★★ | ・∂G/∂θ を `analytical_derivatives_jit.py` に依存<br>・`BiofilmTSM` を `improved1207_paper_jit.py` から継承<br>・`case2_tmcmc_linearization.py` から呼ばれる中核<br>・TMCMC統合済み（線形化点更新対応） |
| **analytical_derivatives_jit.py** | ★★★★★ | ∂G/∂θ の解析厳密解<br>数値誤差ほぼゼロ（JIT最適化）<br>全パラメータ対応 | ・∂²G/∂θ² 実装（2次TSM用）<br>・状態依存項の整理（可読性向上）<br>・自動微分とのハイブリッド（検証用）<br>・GPU対応（CuPy/Numba CUDA） | ★★★★☆ | ・TSM-ROMの精度を直接支配<br>・`demo_analytical_tsm_with_linearization_jit.py` に提供<br>・他は全部ここに依存（ボトルネック） |
| **case2_tmcmc_linearization.py** | ★★★★☆ | 統計的には正しいが<br>ROM誤差に依存<br>TMCMC × 線形化更新統合済み | ・線形化点更新の自動化（一部実装済み）<br>・ESS/R-hat停止条件（参考指標として実装済み）<br>・VI/SMC化（変分推論・逐次モンテカルロ）<br>・並列化（マルチプロセス/GPU） | ★★★★★ | ・TSM-ROMを使う「消費者」<br>・精度は上2つ次第<br>・`LogLikelihoodEvaluator` でTSMをラップ<br>・2-phase MCMC + TMCMC実装済み |
| **improved1207_paper_jit.py** | ★★★★★ | FOM（基準解）<br>物理モデルとして完成<br>JIT最適化済み | ・高速化余地小（既にJIT化）<br>・GPU/分割積分程度<br>・アダプティブ時間刻み<br>・並列化（複数theta同時計算） | ★★★☆☆ | ・TSMの展開点生成用（`run_deterministic`）<br>・精度の「真値」（検証用）<br>・`BiofilmTSM` の基底クラス<br>・`BiofilmNewtonSolver` を提供 |

---

## 🔍 詳細分析

### 1. demo_analytical_tsm_with_linearization_jit.py

**現状:**
- ✅ 1次TSM-ROM実装完了（x(θ) ≈ x(θ₀) + ∂x/∂θ|_{θ₀} · (θ - θ₀)）
- ✅ 線形化点管理機能（`update_linearization_point()`）
- ✅ JIT最適化（20-50x高速化）
- ✅ キャッシング機能（線形化点での決定論的解）

**理論的限界:**
- 1次近似のため、θがθ₀から離れると誤差が増大
- 非線形性が強い領域では精度低下

**改善の優先度:**
1. **高**: 2次TSM対応（∂²G/∂θ²実装が必要）
2. **中**: 線形化点の自動再配置（誤差指標ベース）
3. **低**: アダプティブ線形化点更新間隔

**依存関係:**
```
demo_analytical_tsm_with_linearization_jit.py
├── improved1207_paper_jit.py (BiofilmTSM, BiofilmNewtonSolver)
├── analytical_derivatives_jit.py (AnalyticalDerivatives)
└── bugfix_theta_to_matrices.py (パッチ適用)
```

---

### 2. analytical_derivatives_jit.py

**現状:**
- ✅ ∂G/∂θ の完全解析解（全14パラメータ対応）
- ✅ JIT最適化（50-100x高速化）
- ✅ 数値誤差ほぼゼロ（解析解のため）

**理論的限界:**
- 1次微分のみ実装（2次TSMには不十分）
- 状態依存項が複雑（可読性の課題）

**改善の優先度:**
1. **高**: ∂²G/∂θ² 実装（2次TSM必須）
2. **中**: 状態依存項の整理・ドキュメント化
3. **低**: GPU対応（CuPy/Numba CUDA）

**依存関係:**
```
analytical_derivatives_jit.py
└── (独立 - 他に依存しない)
    └── ただし、TSM-ROM全体の精度を支配
```

---

### 3. case2_tmcmc_linearization.py

**現状:**
- ✅ 2-phase MCMC実装（線形化点更新対応）
- ✅ TMCMC実装（β tempering + 線形化点更新統合）
- ✅ 線形化点更新後のlogL再計算（必須機能）
- ✅ 自動停止条件（収束判定 + 更新回数上限）
- ⚠️ ROM誤差に依存（TSM-ROMの精度次第）

**理論的限界:**
- ROM誤差が統計的誤差に混入
- 多峰性がある場合の探索効率

**改善の優先度:**
1. **高**: VI/SMC化（変分推論・逐次モンテカルロ）
2. **中**: 並列化（マルチプロセス/GPU）
3. **低**: アダプティブ提案分布（より効率的な探索）

**依存関係:**
```
case2_tmcmc_linearization.py
├── demo_analytical_tsm_with_linearization_jit.py (BiofilmTSM_Analytical)
│   ├── improved1207_paper_jit.py
│   └── analytical_derivatives_jit.py
└── mcmc_diagnostics.py (診断機能)
```

---

### 4. improved1207_paper_jit.py

**現状:**
- ✅ FOM（Full Order Model）完全実装
- ✅ 物理モデルとして完成（論文準拠）
- ✅ JIT最適化済み（Newton法 + 時間積分）

**理論的限界:**
- 計算コストが高い（FOMのため）
- 高速化余地は限定的（既にJIT化）

**改善の優先度:**
1. **中**: GPU対応（CuPy/Numba CUDA）
2. **低**: アダプティブ時間刻み
3. **低**: 並列化（複数theta同時計算）

**依存関係:**
```
improved1207_paper_jit.py
└── (独立 - 他に依存しない)
    └── ただし、TSM-ROMの基底として使用
```

---

## 🎯 優先改善ロードマップ

### Phase 1: 精度向上（短期）
1. **analytical_derivatives_jit.py**: ∂²G/∂θ² 実装
2. **demo_analytical_tsm_with_linearization_jit.py**: 2次TSM対応

### Phase 2: 自動化（中期）
1. **demo_analytical_tsm_with_linearization_jit.py**: 誤差指標ベース線形化点更新
2. **case2_tmcmc_linearization.py**: アダプティブ更新間隔

### Phase 3: 高速化（長期）
1. **analytical_derivatives_jit.py**: GPU対応
2. **case2_tmcmc_linearization.py**: 並列化

---

## 📈 精度・性能の関係性

```
精度の階層:
FOM (improved1207_paper_jit.py) ★★★★★
    ↓ (ROM誤差)
TSM-ROM (demo_analytical_tsm_with_linearization_jit.py) ★★★★★
    ↓ (統計的誤差)
MCMC/TMCMC (case2_tmcmc_linearization.py) ★★★★☆

ボトルネック:
analytical_derivatives_jit.py → TSM-ROM精度を直接支配
```

---

## ✅ 検証結果

表の内容は**概ね正確**です。以下の補足があります：

1. **case2_tmcmc_linearization.py**: TMCMC × 線形化更新は既に統合済み
2. **線形化点更新の自動化**: 一部実装済み（収束判定 + 更新回数上限）
3. **ESS/R-hat停止条件**: 参考指標として実装済み（docstringに注意書きあり）

**総合評価**: 表の内容は正確で、改善点も適切に指摘されています。特に2次TSM対応が最重要課題です。

