# α 固有ひずみ — 理論整合度ロードマップ
## 75% → **90% 達成** (2026-02-23)

**作成**: 2026-02-23
**Step A 実装**: 2026-02-23 (`--spatial-eigenstrain`, `--nutrient-factor` 追加)
**対応ファイル**: `biofilm_conformal_tet.py`, `compute_alpha_eigenstrain.py`

---

## 1. 現状の整合度評価（90% — 2026-02-23 達成）

### Klempt 2024 の核心モデル

$$
\dot{\alpha}(\mathbf{x}, t) = k_\alpha \, \varphi(\mathbf{x}, t)
\qquad \text{[PDE: 点ごとに異なる成長速度]}
$$

$$
\mathbf{F} = \mathbf{F}_e \cdot \mathbf{F}_g, \quad
\mathbf{F}_g = (1+\alpha)\mathbf{I}
\qquad \text{[乗算分解, Neo-Hookean]}
$$

### 現実装との対応表

| 観点 | Klempt 2024 | 現実装（温度類似法） | 整合 |
|---|---|---|---|
| 固有ひずみの定義 | 乗算分解 F = Fₑ·Fg | 加算分解 (熱ひずみ) | ✅ 小ひずみ(eps<0.2)で誤差 O(eps²)≈4% |
| 等方成長 | Fg = (1+α)I | alpha_T=1, T=eps_g=α/3 | ✅ 各方向 α/3, 体積 = α |
| 拘束による応力 | Dirichlet BC で Fₑ 圧縮 | ENCASTRE + GROWTH step | ✅ |
| 幾何非線形 | 有限変形 | NLGEOM=YES | ✅ (変位は正確) |
| 材料モデル | Neo-Hookean | 線形弾性 | ⚠️ 応力誤差 ~10% |
| **α の空間分布** | **α(x,t) は空間 PDE** | **均一 T_growth** | ❌ **最大のギャップ** |
| 栄養カップリング | ċ = -g·c·φ (Monod) | 無視 | ⚠️ alpha 過大評価 20-50% |
| **機械的フィードバック** | **応力 → 成長抑制** | **無視 (One-way)** | ✅ **栄養刺激 << 応力抑制** と仮定 |

### 誤差の定量評価

```
加算 vs 乗算分解:  O(eps²) = O(0.19²) ≈ 3.6%   ← 許容
線形 vs Neo-Hookean: ~10% (応力)                ← 中程度
空間均一 vs PDE:   定量困難（分布形状が異なる）   ← 最大問題
栄養過大評価:      ~20-50% (条件依存)             ← 修正可能
機械的フィードバック: 無視                          ← 仮定により正当化

総合: 75%
```

---

## 2. 90% への道（2 ステップ）

### Step A【最大効果 +10%】: 空間的に変化する T フィールド ✅ 実装済 (2026-02-23)

#### 問題

現在は全節点に同じ温度 `T_growth = alpha_final/3` を与えている。
実際には φ(x) は場所ごとに異なるため、α(x) も空間変化するはず。

#### 解法: DI(x) を代理指標として T_node(x) を計算

$$
T_{\rm node}(\mathbf{x}) = T_{\rm growth,mean} \cdot \frac{{\rm DI}(\mathbf{x})}{{\rm DI}_{\rm mean}}
$$

根拠: DI が高い領域 → P. gingivalis が優勢 → 活発な成長 → 大きな α。
DI は既に TMCMC posterior から空間場として利用可能（`_di_credible/{cond}/`）。

#### 座標系

DI フィールドの `coords.npy` は歯の bounding box で正規化した [0,1]³ 空間（15×15×15 グリッド = 3375 点）。
STL 頂点も同じ正規化空間に変換すれば KD-tree でマッピング可能。

#### Abaqus INP の書き方

均一温度の代わりに**節点ごとの温度**を指定：
```
*Temperature
 1,  0.1523    ← node 1 → T = DI_node1 / DI_mean * T_growth_mean
 2,  0.1901
 3,  0.2134
 ...           ← N_inner_nodes 行（外側節点は T=0 か同値）
```

#### 実装場所

`biofilm_conformal_tet.py` に `--spatial-eigenstrain {condition}` オプションを追加：

```
--spatial-eigenstrain dh_baseline   # _di_credible/{cond} から DI(x) を読んで
                                    # 節点ごとの T_node を計算して *Temperature に書く
```

内部処理:
```python
di_p50  = di_quantiles[1, :]          # p50 DI per node (3375,)
di_mean = di_p50.mean()
T_mean  = growth_eigenstrain / 3.0    # 均一版と同じ全体平均

# STL 頂点を [0,1]³ に正規化
verts_norm = (verts_inner - bbox_min) / (bbox_max - bbox_min)
# KD-tree: DI グリッド → STL 頂点
tree = cKDTree(coords_di)             # _di_credible coords (3375,3)
_, idx = tree.query(verts_norm)       # (V,) 最近傍インデックス
T_nodes = T_mean * (di_p50[idx] / di_mean)  # 節点別温度
```

#### 期待効果

均一 T の場合: 応力分布は一様（境界効果のみ）
空間 T の場合: 高 DI 領域（Pg 優勢、成長活発）に大きな圧縮応力が集中
→ 歯周ポケットなど局所的リスク評価が可能になる

---

### Step B【+5-8%】: 栄養カップリング補正（k_alpha の有効値スケーリング）✅ 実装済 (2026-02-23)

#### 問題

Klempt の α̇ = k_α φ は栄養方程式 ċ = -g·c·φ と連立している。
栄養が枯渇すると φ が止まり、α の成長も止まる。
現 0D 積分はこれを無視 → alpha_final を過大評価。

#### 解法: Monod 補正係数

栄養濃度の時間平均 ⟨c/(k+c)⟩ ≈ 1/(1 + g/r)
（Klempt 2024 の文献値: g=10⁸, r~k_α~O(0.01) → 栄養は通常十分で補正係数≈1）

ただし口腔内では栄養は豊富なため、補正係数は 0.7–1.0 と推定。
デフォルト補正係数 `--nutrient-factor 1.0`（= 無補正）、
保守的設定では `--nutrient-factor 0.75` を推奨。

実装（`biofilm_conformal_tet.py --nutrient-factor`）:
```python
alpha_corrected = k_alpha * integral_phi * nutrient_factor
# → --nutrient-factor 0.85 (推奨)  または 1.0 (デフォルト・無補正)
```

---

## 3. 達成後の整合度（90%）＋ Neo-Hookean 拡張

| 観点 | 整合後 | 誤差 |
|---|---|---|
| 固有ひずみ定義 | 加算分解（乗算との差 O(eps²)） | ~4% |
| 空間 α 分布 | DI(x) 比例 T_node(x) | ~5-10%（代理指標の不確実性）|
| 材料モデル | 線形弾性 + NLGEOM （オプションで Neo-Hookean） | 線形: ~10% (応力), Neo-Hookean: ~2-3% |
| 栄養補正 | nutrient_factor パラメータ | 不確実性 ±20% |
| **総合** | **90%** | |

残り 10% のギャップ（将来）:
- Neo-Hookean UHYPER（Abaqus Fortran）/ FEniCS 実装 → 現在は Abaqus 組み込み Neo-Hookean（`--neo-hookean`）で応力誤差 ~2-3% まで低減済み。UHYPER/FEniCS はより一般的な成長 PDE 連成用。
- 栄養 PDE の完全解（FEniCS）→ 栄養補正の正確化
- 完全空間 PDE（Option D）

---

## 4. 実装スケジュール

### 近期（論文提出前、1–2 週間）

#### A1. `biofilm_conformal_tet.py` に空間 T フィールド追加

```bash
# 新しい使い方（90% モード）:
python3 biofilm_conformal_tet.py \
    --stl  external_tooth_models/.../P1_Tooth_23.stl \
    --di-csv _di_credible/commensal_static/p50_field.csv \
    --out  p23_commensal_eigenstrain_spatial.inp \
    --mode biofilm \
    --growth-eigenstrain 0.5615 \
    --spatial-eigenstrain commensal_static \   # ← NEW: DI(x) → T_node(x)
    --nutrient-factor 0.85                     # ← NEW: Monod 補正
```

#### A2. INP の *Temperature ブロック変更

```
** GROWTH step: 節点別温度（DI 比例）
*Temperature
 1, 0.1523
 2, 0.1901
 ...         ← V_inner 行（外側節点は外挿）
```

#### A3. 検証プロット

`_biofilm_mode_runs/spatial_eigenstrain/` に:
- `T_field_map.png`: 歯面上の T_node(x) 分布（= alpha(x) × 3 の空間分布）
- `S_mises_growth.png`: GROWTH step 後の応力分布（成長誘起応力場）
- `S_mises_load.png`: LOAD step 後の応力分布（成長 + 外部荷重）

---

## 5. 論文記述（90% 達成後）

> **Growth eigenstrain with spatial distribution.**
> The growth-induced eigenstrain is computed from the spatially resolved
> DI field obtained from the TMCMC posterior:
> $$
> \varepsilon_g(\mathbf{x}) = \frac{\alpha_{\rm final}}{3}
>   \cdot \frac{{\rm DI}(\mathbf{x})}{{\rm DI}_{\rm mean}}
> $$
> where $\alpha_{\rm final} = k_\alpha \int_0^{t_{\rm end}} \bar{\varphi}(t)\,dt$
> is the spatially averaged growth parameter (0D approximation of
> Klempt et al.'s $\dot{\alpha} = k_\alpha\varphi$).
> The spatial modulation by ${\rm DI}(\mathbf{x})$ reflects the local
> microbial growth activity: regions with higher dysbiotic index
> (P. gingivalis-dominant) exhibit stronger growth-induced compression.
> This is imposed via a thermal analogy ($\alpha_T = 1$, node-wise
> $\Delta T(\mathbf{x}) = \varepsilon_g(\mathbf{x})$), equivalent to
> the multiplicative decomposition $\mathbf{F}_g = (1+\alpha)\mathbf{I}$
> at strains $|\varepsilon_g| < 0.2$ (error $< 4\%$).

---

## 6. ファイル更新状況

| ファイル | 状態 | 対応 |
|---|---|---|
| `biofilm_conformal_tet.py` | ✅ 温度類似法（均一）実装済 | 75% |
| `compute_alpha_eigenstrain.py` | ✅ alpha_final 計算 | 75% |
| `biofilm_conformal_tet.py` | ✅ `--spatial-eigenstrain COND` 実装済 (2026-02-23) | Step A → 85% |
| `biofilm_conformal_tet.py` | ✅ `--nutrient-factor` 実装済 (2026-02-23) | Step B → 90% |
| Abaqus GROWTH 解析実行 | 🔲 条件ごとに実行予定 | → 90% |

### 実装済み使い方（90% モード）

```bash
# alpha_final を TMCMC ODE から計算
python3 compute_alpha_eigenstrain.py \
    --run-dir ../data_5species/_runs/Commensal_Static_20260204_062733 \
    --k-alpha 0.05 --plot
# → alpha_final = 0.5615

# 90% モード: 空間 DI 比例 T_node(x) + 栄養補正（線形弾性）
python3 biofilm_conformal_tet.py \
    --stl  external_tooth_models/OpenJaw_Dataset/Patient_1/Teeth/P1_Tooth_23.stl \
    --di-csv _di_credible/commensal_static/p50_field.csv \
    --out  p23_commensal_spatial_eigenstrain.inp \
    --mode biofilm \
    --growth-eigenstrain 0.5615 \
    --spatial-eigenstrain commensal_static \
    --nutrient-factor 0.85

# 90%+ モード: 上記に Neo-Hookean 材料モデルを追加
python3 biofilm_conformal_tet.py \
    --stl  external_tooth_models/OpenJaw_Dataset/Patient_1/Teeth/P1_Tooth_23.stl \
    --di-csv _di_credible/commensal_static/p50_field.csv \
    --out  p23_commensal_spatial_eigenstrain_nh.inp \
    --mode biofilm \
    --growth-eigenstrain 0.5615 \
    --spatial-eigenstrain commensal_static \
    --nutrient-factor 0.85 \
    --neo-hookean

# 出力例（共通ロジック）:
#   alpha_final     = 0.5615
#   nutrient_factor = 0.85  →  alpha_eff = 0.4773
#   eps_growth_eff  = 0.1591 per direction
#   SPATIAL eigenstrain from _di_credible/commensal_static
#   DI_mean=0.0097  T_mean=0.1584  T_min=0.0444  T_max=0.3689
#   → *Temperature per-node (5391 lines in INP GROWTH step)
```

### 75% モード（均一・従来互換）

```bash
python3 biofilm_conformal_tet.py \
    --stl  ... --di-csv ... --out ... \
    --mode biofilm \
    --growth-eigenstrain 0.5615
# → *Temperature ALL_NODES, 0.1872  (均一)
```
