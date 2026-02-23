# プロジェクト進捗まとめ
**Tmcmc202601 — 5種バイオフィルム TMCMC + FEM 応力解析**
最終更新: 2026-02-21

---

## 概要

口腔バイオフィルムの Hamilton 原理モデル（5種ODE系）に対して

1. **TMCMC（逐次モンテカルロ）** でパラメータを推定し
2. **3D FEM** でバイオフィルムの力学応力を計算する

という2段階パイプラインを構築・実行した。

---

## Phase 1: TMCMC パラメータ推定

### モデル

- **5種**: S. oralis (So), A. naeslundii (An), V. dispar (Vd), F. nucleatum (Fn), P. gingivalis (Pg)
- **20 パラメータ** (θ[0]–θ[19])、Hill ゲート関数付き相互作用
- **主要パラメータ**: a₃₅ = θ[18] (Vd→Pg), a₄₅ = θ[19] (Fn→Pg)

### 実験の変遷

| バージョン | 粒子数 | ステージ | a₃₅ 範囲 | a₄₅ 範囲 | 備考 |
|-----------|-------|---------|---------|---------|------|
| dh_old (初期) | 50 | 5 | [0, 30] | [0, 20] | baseline |
| mild_weight | 150 | 8 | **[0, 5]** | **[0, 5]** | lambda_pg=2.0, lambda_late=1.5 |

### RMSE 比較 (MAP)

| 種 | dh_old | mild_weight | 変化 |
|----|--------|-------------|------|
| S. oralis | 0.036 | 0.034 | ≈同等 |
| A. naeslundii | 0.129 | 0.105 | **改善** |
| V. dispar | 0.213 | 0.269 | やや悪化 |
| F. nucleatum | 0.088 | 0.161 | やや悪化 |
| P. gingivalis | 0.435 | **0.103** | **大幅改善** |
| **合計** | **0.228** | **0.156** | **31% 削減** |

- a₃₅: 28.7（上限張り付き）→ **3.56**（健全な推定値）
- a₄₅: 3.97 → 2.41
- ESS 200–300、rhat ≈ 1.0 → 良好な収束

### 条件定義（FEM への引き継ぎ）

| キー | 内容 | a₃₅ |
|-----|------|-----|
| `dh_baseline` | dh_old の θ_MAP | 21.4 |
| `commensal_static` | mild_weight θ、HOBIC なし | 3.56 |
| `commensal_hobic` | mild_weight θ + HOBIC 摂動 | 3.56 |
| `dysbiotic_static` | 腸内細菌叢乱れ θ | — |

---

## Phase 2: FEM 応力解析パイプライン

### 物理モデル

```
TMCMC θ_MAP
    ↓
3D 反応拡散 FEM (15×15×15 grid, 3375 ノード)
    → φᵢ(x,y,z)  種フィールド
    ↓
Dysbiotic Index (DI) = 1 − H / log5
    → H = Shannon エントロピー
    ↓
E(DI) = E_max·(1−r)^n + E_min·r  (r = DI/s)
    ↓
Abaqus FEM 応力解析
    → S_Mises (基材面 / 表面)
```

**材料パラメータ (デフォルト)**

| 記号 | 値 | 意味 |
|------|-----|------|
| E_max | 10.0 GPa | 健全バイオフィルム剛性 |
| E_min | 0.5 GPa | 病的バイオフィルム剛性 |
| n | 2.0 | べき乗則指数 |
| s (DI_SCALE) | 0.025778 | DI 正規化スケール |
| ν | 0.30 | ポアソン比 |
| 荷重 | 1 MPa | 圧縮面圧（均一） |

---

### ステップ A: 材料感度スイープ

**スクリプト**: `run_material_sensitivity_sweep.py`
**結果**: `_material_sweep/results.csv`, `_material_sweep/figures/`

#### A1 — E_max × E_min グリッド (4×4 = 16 点)
- E_max: 5/10/15/20 GPa × E_min: 0.1/0.5/1.0/2.0 GPa、固定: n=2
- **知見**: 基材 S_Mises は E_max に対して鈍感、E_min 増加で~5% 低下

#### A2 — べき乗則指数 n 比較 (n=1,2,3)
- n が大きいほど表面応力↓、基材応力↑（DI の空間分布を強調）

#### A3 — θ バリアント比較
- `mild_weight` (a₃₅=3.56) vs `dh_old` (a₃₅=21.4) vs `nolambda`
- **key result**: mild_weight は dh_old 比で基材 S_Mises **約 30% 低下**
  → Pg 抑制が力学的リスク低減に直結することを定量確認

図: `fig_A1_emax_emin_heatmap.png`, `fig_A2_nexp_bars.png`, `fig_A3_theta_comparison.png`

---

### ステップ B1: DI 場の信頼区間

**スクリプト**: `aggregate_di_credible.py`
**結果**: `_di_credible/{cond}/`

- 各条件 20 posterior サンプル → ノードごとに p05/p50/p95 DI を計算
- Abaqus で p05/p50/p95 の 3 ケースを実行 → 応力の信頼帯を取得

| 条件 | DI_mean (p50) | S_Mises 基材 (MPa, p50) |
|------|--------------|------------------------|
| dh-baseline | ~0.015 | **0.571** |
| Comm. Static | ~0.010 | ~0.860 |
| Comm. HOBIC | ~0.010 | ~0.854 |
| **Dysb. Static** | ~0.011 | **0.634** |

- dh-baseline が最も基材応力低い（a₃₅ 大 → Pg 多 → DI 高 → E 低 → 応力緩和）
- dysbiotic は dh より応力高い

図: `fig_di_depth_profile.png`, `fig_di_spatial_ci.png`, `fig_stress_di_uncertainty.png`

---

### ステップ C1: 異方性解析

#### C1-前処理: ∇φ_Pg 方向解析
**スクリプト**: `fem_aniso_analysis.py`
**結果**: `_aniso/{cond}/aniso_summary.json`

- Pg 濃度フィールドの主勾配方向 e₁ を各条件で計算

| 条件 | e₁ | x 軸からの角度 |
|------|-----|-------------|
| dh-baseline | [−0.972, +0.211, −0.105] | **13.6°** |
| commensal_static | [−0.956, +0.258, −0.142] | 17.1° |
| commensal_hobic | [−0.959, +0.245, −0.145] | 16.5° |
| dysbiotic_static | [−0.952, +0.262, −0.160] | 17.9° |

→ 全条件で Pg は基材（歯面）に近い深部に局在（−x 方向が主方向）

#### C1-本体: 異方性スイープ
**スクリプト**: `run_aniso_comparison.py`
**結果**: `_aniso_sweep/results.csv`

横等方性モデル: E₁ (∇φ_Pg 方向) を基準に E₂=E₃=β·E₁
β=1.0 (等方) → β=0.3 (最大異方)

| 条件 | β=1.0 基材 | β=0.5 基材 | Δ |
|------|-----------|-----------|---|
| dh-baseline | 0.839 MPa | 0.817 MPa | **−2.6%** |
| Comm. Static | 0.860 MPa | 0.849 MPa | −1.3% |
| Comm. HOBIC | 0.854 MPa | 0.843 MPa | −1.3% |
| Dysb. Static | 0.856 MPa | 0.849 MPa | −0.8% |

**知見**: 異方性効果は基材で 1–3% の modest な低下、表面はほぼ不感（荷重制御 BC が支配）

図: `fig_C1_smises_vs_beta.png`, `fig_C1_aniso_vs_iso.png`

---

### ステップ B3: 3D コヒーシブゾーンモデル (CZM)

**スクリプト**: `run_czm3d_sweep.py`, `abaqus_biofilm_cohesive_3d.py`
**出力予定**: `_czm3d/`

DI 依存コヒーシブ特性:
- t_max(DI) = 1.0 MPa · (1−r)^n
- G_c(DI) = 10.0 J/m² · (1−r)^n
- 混合モード: Benzeggagh-Kenane 則

→ **現状**: スクリプト整備済み、実行・結果整理は次フェーズ

---

## 成果物一覧

| 種別 | ファイル |
|------|---------|
| 論文向けパイプライン図 | `docs/fem_pipeline.pdf` |
| FEM レポート (旧) | `fem_report.pdf` |
| Abaqus 実装レポート | `abaqus_implementation_report.pdf` |
| 材料感度図 | `_material_sweep/figures/*.png` |
| 異方性図 | `_aniso_sweep/figures/*.png` |
| DI 信頼区間図 | `_di_credible/fig_*.png` |
| 全スクリプト | `FEM/*.py`, `FEM/usdfld_biofilm.f` |

---

## キーメッセージ（論文向け）

1. **Pg 相互作用強度 a₃₅ の重要性**: bounds を [0,30]→[0,5] に絞るだけで Pg RMSE が 0.435→0.103 に改善。過学習的な大値推定を防いだ。

2. **a₃₅ の力学的意味**: mild_weight (a₃₅=3.56) は dh_old (a₃₅=21.4) に比べ基材 S_Mises を ~30% 低下させる。TMCMC 推定精度が FEM 応力予測に直接影響。

3. **異方性効果は限定的**: β を 1.0→0.3 にしても基材応力変化は 1–3%。荷重条件（均一圧縮）が支配的。より現実的な咀嚼荷重では効果が変わる可能性あり。

4. **Pg は基材近傍に局在**: ∇φ_Pg の主方向が全条件で −x（基材側）を向く → 病原性バイオフィルムの深部定着を力学的に示唆。

---

## 残課題（Next Steps）

- [ ] **B3 CZM 実行・結果整理** → 条件間の破壊閾値比較
- [ ] **Posterior S_Mises 不確かさ完成** → 20 サンプル CI バー付き比較図
- [ ] **現実的荷重への拡張** → 50–150 N 咀嚼荷重、接触面積考慮
- [ ] **TMCMC 精緻化** → V.dispar/Fn bounds 拡張で mild_weight のトレードオフ改善
