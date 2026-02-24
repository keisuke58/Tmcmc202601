# Methods Supplement — FEM Biofilm Analysis
## (Klempt 2024 Gap Analysis: Option A 記述 + TMCMC→Monod 変換)

**作成**: 2026-02-23
**対応タスク**: klempt2024_gap_analysis.md § 5 (P1, 近期)

---

## 1. E(DI) の物理的位置づけ（論文 Methods への挿入文）

### 挿入候補箇所
論文 Methods 節「Finite Element Model」サブセクション末尾、
または「DI-Dependent Stiffness Mapping」の後。

### 挿入文（英語）

> **Clarification on stiffness scale.**
> The DI-dependent stiffness $E(\text{DI})$ in our model represents the
> *effective stiffness of the biofilm-covered periodontal substrate surface*,
> not the intrinsic mechanical property of the biofilm extracellular
> polymeric substance (EPS) matrix.
> The latter has been reported as $E \approx 10\,\text{Pa}$ by
> Billings et al. (2015), and is adopted in the Klempt et al. (2024)
> Hamilton-principle model of biofilm growth mechanics.
> In contrast, our mapping $E(\text{DI}) \in [E_\text{min}, E_\text{max}]$
> with $E_\text{max} = 10\,\text{GPa}$ and $E_\text{min} = 0.5\,\text{GPa}$
> (substrate mode) reflects how microbial community composition alters the
> *mechanical integrity of the periodontal attachment apparatus* — including
> enamel, cementum, and the biofilm–substrate interface — at the millimeter
> scale relevant to clinical periodontal risk assessment.
> This is consistent with the effective stiffness formulation used in
> Soleimani et al. (2016, 2019) for similar biofilm–substrate systems.
> A separate biofilm-scale analysis at $E \in [10, 1000]\,\text{Pa}$
> (biofilm mode, Klempt et al. 2024) was performed to characterize
> deformation of the EPS matrix itself under gingival crevicular fluid
> pressure (see Section~\ref{sec:biofilm_mode}).

### 挿入文（日本語訳）

> **剛性スケールの定義について。**
> 本モデルの DI 依存剛性 $E(\text{DI})$ は，バイオフィルム外多糖体（EPS）
> マトリクス固有の力学特性ではなく，「バイオフィルムに覆われた歯周組織
> 表面の有効（合成）剛性」を表す。
> EPS 固有剛性は Billings et al. (2015) により $E \approx 10\,\text{Pa}$
> と報告されており，Klempt et al. (2024) の Hamilton 原理バイオフィルム
> 成長力学モデルでも採用されている。
> これに対し，本研究の $E(\text{DI})$ マッピングは，菌叢組成の変化が
> エナメル質・セメント質・バイオフィルム‐基質界面を含む歯周付着装置の
> 力学的完全性をどのように変化させるかを，臨床的に意義のある mm スケールで
> 記述するものである。

---

## 2. Biofilm Mode 解析の Methods 節追記（Section ref: biofilm_mode）

### 英語

> **Biofilm-scale deformation analysis.**
> In addition to the substrate-scale analysis, we performed a separate
> finite element analysis of the biofilm EPS layer itself (biofilm mode)
> using the conformal tetrahedral mesh with EPS-scale material parameters:
> $E_\text{max} = 1000\,\text{Pa}$, $E_\text{min} = 10\,\text{Pa}$,
> applied pressure $= 100\,\text{Pa}$ (gingival crevicular fluid, GCF),
> and biofilm thickness $= 0.2\,\text{mm}$.
> Geometric nonlinearity (NLGEOM) was activated to account for
> finite-deformation effects at the expected strain level of $O(0.1)$.
> Since the load is pressure-controlled (Neumann BC), the Von Mises stress
> field is determined primarily by the load balance and varies by less than
> 0.1\% across microbial community conditions.
> The maximum nodal displacement $U_\text{max}$, which scales as
> $U \propto 1/E_\text{mean}$, shows a condition-dependent variation of
> approximately 10\%, reflecting the DI-driven stiffness contrast:
> the dysbiotic baseline ($U_\text{max} = 26.7\,\mu\text{m}$) is stiffer
> than the commensal HOBIC community ($U_\text{max} = 29.4\,\mu\text{m}$).

---

## 3. TMCMC 成長率パラメータ → Monod r 変換式

### 背景

Klempt 2024 の FEM では Monod 型成長モデルのパラメータ
$(r, k, g, k_\alpha, \beta)$ を使用する。
本プロジェクトの TMCMC 推定パラメータ $\theta$ とは直接対応しないが、
近似的な変換式を以下に示す。

### パラメータ対応表

| Klempt FEM パラメータ | 物理意味 | TMCMC $\theta$ 対応 | 推奨値 |
|---|---|---|---|
| $r$ [T*⁻¹] | Monod 成長速度定数 | $\theta_i$ の自己成長項 $a_{ii}$ | $r \approx \max_i(a_{ii}) / t_\text{scale}$ |
| $k$ [-] | 半飽和定数（栄養感受性） | ODE に非明示 | $k = 1$（文献値） |
| $g$ [T*⁻¹] | 栄養消費係数 | ODE に非明示 | $g = 10^8$（文献値; Klempt 2024） |
| $k_\alpha$ [T*⁻¹] | 成長誘起膨張係数 | **対応なし** | $k_\alpha = 0.05$（推奨初期値） |
| $\beta$ [μm²·T*⁻¹] | 界面拡散係数 | ODE に非明示 | $\beta = 2$（文献値; Klempt 2024） |

### 変換式

**Monod 成長率 $r$** の TMCMC からの近似:
$$
r \approx \frac{\max_i a_{ii}}{\Delta t_\text{day} \cdot N_\text{steps}}
$$
ただし $\Delta t_\text{day}$ は 1 日あたりの ODE ステップ数、$N_\text{steps}$ は 1 T* に相当するステップ数。

TMCMC 推定値（Commensal Static, MAP）の例:
- $a_{11}$ (S. oralis 自己成長) $\approx 1.85$ → 対数スケール成長率
- $a_{22}$ (A. naeslundii 自己成長) $\approx 2.11$
- ODE timestep $= 0.01$ T*、総時間 $\approx 25$ T*
- $r \approx 2.1 / (25 \times 0.01^{-1}) \approx 0.00084$ T*⁻¹

**α 膨張係数 $k_\alpha$** の感度解析:

| $k_\alpha$ | $\alpha_\text{final}$ (est.) | $\varepsilon_\text{growth}$ | $\sigma_0$ (E=100 Pa) |
|---|---|---|---|
| 0.01 | 0.11 | 0.037 | -3.7 Pa |
| 0.05 | 0.56 | 0.187 | -18.7 Pa |
| 0.10 | 1.12 | 0.373 | -37.3 Pa |

$k_\alpha = 0.05$ は $E=100\,\text{Pa}$ 程度の剛性で $\sim20\%$ の
圧縮プレストレスを与え、成長誘起応力が外部 GCF 荷重（100 Pa）と
同オーダーになる合理的な値。

### 0D 積分による $\alpha_\text{final}$ の計算

```bash
# TMCMC ODE トラジェクトリから alpha_final を計算
python3 compute_alpha_eigenstrain.py \
    --run-dir ../data_5species/_runs/Commensal_Static_20260204_062733 \
    --k-alpha 0.05 \
    --plot

# 出力例:
#   alpha_final : 0.5615
#   eps_growth  : 0.1872  (= alpha/3, per direction)
#
# → biofilm_conformal_tet.py に渡す（GROWTH + LOAD の 2 step INP を生成）:
python3 biofilm_conformal_tet.py \
    --stl  external_tooth_models/.../P1_Tooth_23.stl \
    --di-csv _di_credible/commensal_static/p50_field.csv \
    --out  p23_commensal_eigenstrain.inp \
    --mode biofilm \
    --growth-eigenstrain 0.5615
# → 生成 INP の構造:
#   *Material ... *Elastic ... *Expansion, alpha_T=1.0   (全 DI ビン)
#   *Step, name=GROWTH  →  *Temperature ALL_NODES, 0.1872  (成長固有ひずみ)
#   *End Step
#   *Step, name=LOAD    →  *Cload (GCF 100 Pa 外部荷重)
#   *End Step
```

### 実装の理論的整合性

| 観点 | 旧実装 (`*INITIAL CONDITIONS, TYPE=STRESS`) | 新実装（温度類似法） |
|---|---|---|
| 固有ひずみの性質 | ❌ 初期残留応力（Abaqus が平衡化しようとして消える） | ✅ 真の応力ゼロ参照配置変更 |
| Klempt F_g = (1+α)I との対応 | 近似的 | ✅ alpha_T=1, T=eps_growth で完全対応 |
| 実装複雑度 | 低 | 低（2 step + *Expansion のみ） |
| Abaqus ソルバー動作 | 平衡ステップで初期応力が変形として解放 | GROWTH step で拘束された成長を正確に解く |

---

## 4. 関連文献

- **Klempt, Soleimani, Wriggers, Junker (2024)**: Hamilton FEM, E≈10 Pa, α̇ = k_α φ
  DOI: 10.1007/s10237-024-01883-x
- **Billings et al. (2015)**: EPS matrix stiffness E≈10 Pa 実測
- **Soleimani et al. (2016, 2019)**: GPa スケール有効剛性と FEM バイオフィルム解析

---

## 5. 実装済みファイル一覧

| ファイル | 内容 | 状態 |
|---|---|---|
| `biofilm_conformal_tet.py` | `--growth-eigenstrain ALPHA` 追加 | ✅ 実装済 |
| `compute_alpha_eigenstrain.py` | TMCMC ODE → alpha_final 0D 計算 | ✅ 実装済 |
| `plot_biofilm_nlgeom_enhanced.py` | 4条件比較プロット強化版 | ✅ 実装済 |
| `_biofilm_mode_runs/biofilm_nlgeom_enhanced_combined.png` | 2×2 複合比較図 | ✅ 生成済 |
| Option C (二層 Tie モデル) | biofilm_3tooth_assembly.py 参考に実装 | 将来 |

---

## 6. DI の時間・スナップショットの扱い（論文 Methods への挿入文）

### 背景

FEM Stage 2 で用いる DI 場は、3D 反応拡散シミュレーションの**ある時点の状態量**である。
成長固有ひずみ $\varepsilon$ は時間積分量であるのに対し、DI は**最終組成状態を反映**する。

### 挿入文（英語）

> **Temporal specification of the DI field.**
> The Dysbiotic Index field $\mathrm{DI}(\mathbf{x})$ used in the FEM stress analysis
> is computed from the species volume fractions $\varphi_i(\mathbf{x}, t)$ at a
> *single time snapshot* of the 3D reaction-diffusion simulation.
> The default snapshot index is 20 (corresponding to $t \approx 0.2\,T^*$ in the
> dimensionless time scale, or approximately 20% of the total simulation duration).
> This choice reflects the quasi-steady community composition reached after the
> initial transient; sensitivity to snapshot selection (e.g., snapshot 10 vs 50)
> was verified to yield less than 5% variation in mean DI across conditions.
> In contrast, the growth eigenstrain $\varepsilon(\mathbf{x})$ (when enabled via
> `--growth-eigenstrain`) is a *time-integrated quantity*:
> $\alpha_{\mathrm{Monod}}(\mathbf{x}) = k_\alpha \int_0^T \varphi_{\mathrm{total}}
> \cdot c/(k+c)\,\mathrm{d}t$, reflecting the cumulative growth history.

### 挿入文（日本語訳）

> **DI 場の時間的指定。**
> FEM 応力解析に用いる Dysbiotic Index 場 $\mathrm{DI}(\mathbf{x})$ は、
> 3D 反応拡散シミュレーションの**単一時点**における種体積分率
> $\varphi_i(\mathbf{x}, t)$ から計算される。
> デフォルトのスナップショットは 20（無次元時間 $t \approx 0.2\,T^*$、
> 総シミュレーション時間の約 20% に相当）。
> この選択は初期過渡後の準定常的な群集組成を反映する。
> スナップショット選択（例: 10 vs 50）への感度は、条件間の平均 DI の
> 変動が 5% 未満であることを確認済み。
> 一方、成長固有ひずみ $\varepsilon(\mathbf{x})$（`--growth-eigenstrain` で有効化）
> は**時間積分量**であり、累積成長履歴を反映する。
