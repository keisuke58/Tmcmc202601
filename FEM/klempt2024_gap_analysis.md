# Klempt et al. (2024) — ギャップ分析と対応方針

**対象論文**: Klempt, Soleimani, Wriggers, Junker (2024)
"A Hamilton principle-based model for diffusion-driven biofilm growth"
*Biomechanics and Modeling in Mechanobiology* 23:2091–2113
DOI: 10.1007/s10237-024-01883-x

---

## 1. 現パイプラインと論文のモデル対象の違い

| 観点 | Klempt 2024 | 現プロジェクト |
|---|---|---|
| **FEM ドメイン** | バイオフィルム**そのもの** (μm スケール) | 歯・PDL・骨の基板 (mm スケール) |
| **材料剛性** | E ≈ 10 Pa（バイオフィルム EPS の固有剛性） | E_max = 10 GPa / E_min = 0.5 GPa（歯質スケール） |
| **荷重起源** | α 膨張による**成長誘起応力** | 外部咬合圧縮荷重（1 MPa） |
| **状態変数** | u, φ, c, α（4種, Hamilton 由来） | DI → E(DI)（現象論マッピング） |
| **菌種数** | 単一種 | 5 菌種 |
| **パラメータ同定** | 現象論的・手動 | TMCMC ベイズ推定 |

→ **両者は異なるスケール・異なる問いを対象にしており、直接の比較・置換関係にはない。**
　現プロジェクトは「菌叢組成が基板の力学リスクをどう変えるか」、Klempt は「バイオフィルム成長が自身にどんな応力を生むか」という問い。

---

## 2. 剛性スケールの不整合と対処法

### 問題の本質

Klempt 2024 の E ≈ 10 Pa はバイオフィルム EPS マトリクス自体の計測値（Billings et al. 2015 引用）。
現プロジェクトの E_min = 0.5 GPa は dysbiotic 状態の歯・PDL 表面の有効剛性であり、**別の物理量**。

### 対処オプション

#### Option A（推奨・論文対策）: 位置づけを明確化するだけ

E(DI) は「バイオフィルムで覆われた表面の**有効（合成）剛性**」と定義し直す。
バイオフィルム固有の E ≈ 10 Pa と E_min = 0.5 GPa は異なる物理量として明示する。

- メリット: 現在の実装を変更不要、論文で説明するだけで済む
- 根拠: Soleimani et al. (2016, 2019) も GPa スケールの有効剛性を使用している
- Methods 節に追記する文言例:
  > "The DI-dependent stiffness E(DI) represents the effective stiffness of the biofilm-covered substrate surface, not the intrinsic mechanical property of the biofilm EPS matrix (E ~ 10 Pa, Billings et al. 2015). The mapping reflects how microbial community composition alters the mechanical integrity of the periodontal surface."

#### Option B ✅ 実装済み (2026-02-23): `--mode biofilm` フラグ

既存の `biofilm_conformal_tet.py` は既に「歯 STL の外側に薄層を張る」構造になっていた
（Inner face = 歯面固定、Outer face = 荷重）。問題は材料定数のスケールのみだった。

`--mode biofilm` を追加して biofilm-scale E を使えるようにした：

```bash
python3 biofilm_conformal_tet.py \
    --stl external_tooth_models/.../P1_Tooth_23.stl \
    --di-csv abaqus_field_dh_3d.csv \
    --out p23_biofilm_mode.inp \
    --mode biofilm
# → E_max=1000 Pa, E_min=10 Pa, pressure=100 Pa, thickness=0.2 mm
```

| パラメータ | substrate（既存） | biofilm（新モード） | 根拠 |
|---|---|---|---|
| E_max | 10 GPa | 1000 Pa | Billings et al. 2015（EPS-rich） |
| E_min | 0.5 GPa | 10 Pa | Klempt et al. 2024 |
| pressure | 1 MPa | 100 Pa | 歯肉溝液圧（GCF）|
| thickness | 0.5 mm | 0.2 mm | 口腔バイオフィルム実測値 |

**注意**: biofilm mode での線形弾性ひずみは O(0.1) 程度になる可能性あり。
結果は定性的（条件間の相対比較）用途で解釈する。大変形が必要なら `*NLGEOM` または FEniCS 移行。

#### NLGEOM 追加 ✅ 実装済み (2026-02-23)

`biofilm_conformal_tet.py` の `write_abaqus_inp()` に `nlgeom` 引数を追加。
`--mode biofilm` 指定時に自動で `*Step, name=LOAD, nlgeom=YES` を出力する。

```python
# 呼び出し側
write_abaqus_inp(..., nlgeom=(args.mode == "biofilm"))

# INP 生成側
nlgeom_str = "YES" if nlgeom else "NO"
f.write("*Step, name=LOAD, nlgeom=%s\n" % nlgeom_str)
if nlgeom:
    f.write(" 0.01, 1.0, 1e-6, 0.1\n")   # 初期増分を小さく設定
else:
    f.write(" 0.1, 1.0, 1e-5, 1.0\n")
```

#### Option C（今後・将来）: 二層モデル（歯＋バイオフィルム）

```
[外部荷重（咬合力）]
    ↓
[バイオフィルム層: E ~ Pa, --mode biofilm]   ← 実装済み（独立解析）
    ↓ 界面 Tie 拘束（cohesive zone も可）
[歯・エナメル: E ~ GPa, --mode substrate]    ← 将来統合
    ↓
[PDL → 骨]
```

実装ポイント：
- biofilm INP と tooth INP を別アセンブリとして Abaqus に読み込み
- Inner face（歯面）と Tooth outer face を `*TIE` 拘束で結合
- 歯側に実際の咬合荷重を適用、バイオフィルム側は荷重なし（変位従動）

#### Option D（将来・マルチスケール）: Hamilton FEM との二スケール結合

- ミクロ (μm): Hamilton FEM（Klempt 2024 方程式）でバイオフィルム成長・応力
- マクロ (mm): 現 Abaqus モデルでバイオフィルム → 基板への力伝達
- 実装コスト大、現論文のスコープ外

### 結論

**Option A**: 論文 Methods に一文追記（完了次第）。
**Option B**: 実装済み → `--mode biofilm` で biofilm-scale 解析が可能。
**Option C**: 将来の二層 Tie モデル（今後やりたい）。

---

## 3. 現パイプラインに欠けている物理（優先度順）

### [高] 成長誘起応力（α 膨張）の欠如

論文の核心: バイオフィルムが成長するだけで内部に圧縮・縁辺に引張が生じる（Fg = αI 由来）。
現モデルは外部荷重のみ→ 成長プロセス自体の力学的効果なし。

**※機械的フィードバックに関する仮定**:
Klempt らのモデルでは「成長 → 応力 → 成長抑制」という双方向フィードバックが含まれるが、本プロジェクトでは**この機械的フィードバックは栄養刺激による成長促進に比べて小さい**と仮定し、無視する（One-way coupling: Growth → Stress のみ）。
この仮定により、成長計算（TMCMC/ODE）と応力解析（Abaqus）を分離して実行することが正当化される。

**近似的対処**:
α 進化方程式 `α̇ = k_α φ` を 0D で積分 → 最終 α 値を eigenstrain として Abaqus に入力

```python
# 疑似コード
alpha_final = k_alpha * integrate(phi_avg, t_end)  # 0D近似
# Abaqus *INITIAL CONDITIONS, TYPE=STRESS で等方圧縮固有ひずみとして設定
eps_growth = alpha_final / 3.0  # 等方膨張 → 体積ひずみ
```

### [高] TMCMC パラメータ → FEM Monod パラメータの変換が未整備

| FEM パラメータ | 物理意味 | TMCMC 対応物 | 現状 |
|---|---|---|---|
| r（成長速度定数） | Monod 成長率 [T*⁻¹] | θ の各種 growth rate | **未マッピング** |
| k（半飽和定数） | 栄養感受性 [-] | ODE に非明示 | 文献値固定（k=1）で可 |
| g（消費因子） | 栄養消費 [T*⁻¹] | ODE に非明示 | 文献値固定（g=10⁸）で可 |
| k_α（膨張係数） | 局所体積成長 [T*⁻¹] | **対応なし** | 新規パラメータ |
| β（phase-field 拡散） | 界面拡散 [μm²·T*⁻¹] | ODE に非明示 | 文献値固定（β=2）で可 |

### [中] 熱力学的一貫性の担保がない

現 E(DI) 冪乗則は現象論的であり、自由エネルギーから導出されていない。
審査員から問われた際の回答: 「E(DI) は機械的リスク評価のための操作的定義であり、Klempt (2024) の Ψ = φ² μ/2 (I:Cₑ−3) ... とは異なるスケール・モデル階層での記述」と説明する。

### [低・将来] 多菌種 Hamilton 定式化

5 菌種に拡張するには各 φ_i の進化方程式を追加し、TMCMC で推定した a_ij 行列を結合項として組み込む。
Soleimani et al. (2023, Sci. Rep.) の 2 種版が直接の先行実装。
実装は FEniCS 等が現実的（Abaqus USDFLD では連立 PDE が困難）。

---

## 4. Biofilm Mode NLGEOM — 4条件解析結果 (2026-02-23)

### 解析設定

| パラメータ | 値 |
|---|---|
| メッシュ | P1_Tooth_23.stl 外側 conformal tet（厚み 0.2 mm, 4層, 43080 要素） |
| 材料 | E_max=1000 Pa, E_min=10 Pa, E(DI) 冪乗則 (n=2, DI_scale=0.025778) |
| 荷重 | 外表面圧力 100 Pa（GCF 歯肉溝液圧） |
| 境界条件 | 内表面（歯面）固定 |
| 幾何非線形 | `nlgeom=YES`（自動増分 0.01〜0.1） |
| DI 入力 | 各条件の p50 DI フィールド（3375 点, `_di_credible/{cond}/p50_field.csv`） |

### 結果: S_Mises（Von Mises 応力）

| 条件 | DI_mean | E_mean概算 (Pa) | Min (Pa) | Mean (Pa) | p95 (Pa) | Max (Pa) |
|------|---------|----------------|----------|-----------|----------|----------|
| dh_baseline | 0.00852 | 451 | 40.9 | 56.6 | 61.0 | 89.0 |
| dysbiotic_static | 0.00950 | 403 | 42.1 | 56.6 | 61.1 | 89.2 |
| commensal_static | 0.00971 | 392 | 42.1 | 56.6 | 61.1 | 89.3 |
| commensal_hobic | 0.00990 | 383 | 42.2 | 56.6 | 61.1 | 89.3 |

**条件間の S_Mises 差: < 0.1%** → 圧力制御荷重（Neumann BC）では応力は釣り合い条件で決まり、材料剛性に鈍感。

### 結果: 最大変位 U_max

| 条件 | U_max (mm) | 備考 |
|------|------------|------|
| dh_baseline | **0.0267** | 最剛（E_mean=451 Pa） → 最小変形 |
| dysbiotic_static | 0.0286 | |
| commensal_static | 0.0290 | |
| commensal_hobic | **0.0294** | 最柔（E_mean=383 Pa） → 最大変形（+10%） |

**条件間の変位差: 約 10%** → DI 由来の剛性差は応力より変位に現れる。

### 力学的解釈

| 荷重タイプ | 応力の剛性依存性 | 変位の剛性依存性 |
|-----------|----------------|----------------|
| **圧力制御**（biofilm mode: GCF 100 Pa）| **鈍感**（釣り合いで決まる）| **敏感**（ε = σ/E）|
| **変位制御**（substrate mode: 咬合変位） | **敏感** | **鈍感** |

→ biofilm mode の条件比較指標としては **U_max（変位）** が有意義。substrate mode は **S_Mises（応力）** が有意義。

### NLGEOM vs 線形弾性の比較（dh_baseline）

| 指標 | 線形弾性 | NLGEOM=YES | 差 |
|------|---------|------------|-----|
| S_Mises mean | 55.8 Pa | 56.6 Pa | +1.4% |
| S_Mises max | 86.5 Pa | 89.0 Pa | +2.9% |
| U_max | — | 0.0267 mm（13% 圧縮）| — |

線形弾性と NLGEOM の差は < 3%。このひずみレベル（約 13%）では両者が整合しており、
定性比較目的には線形弾性も実用的。ただし物理的正確性のために以降は NLGEOM を維持する。

### 出力ファイル

```
_biofilm_mode_runs/
├── dh_baseline_nlgeom/       BiofilmNL_dh.odb
├── commensal_static_nlgeom/  BiofilmNL_commensal_static.odb
├── commensal_hobic_nlgeom/   BiofilmNL_commensal_hobic.odb
├── dysbiotic_static_nlgeom/  BiofilmNL_dysbiotic_static.odb
└── biofilm_nlgeom_comparison.png   4条件比較図（S_Mises / U_max / DI-E散布）
```

---

## 5. アクションリスト

| 優先度 | タスク | 状態 |
|---|---|---|
| ~~今すぐ~~ | ~~`--mode biofilm` を `biofilm_conformal_tet.py` に追加~~ | ✅ 完了 (2026-02-23) |
| ~~今すぐ~~ | ~~NLGEOM (`*Step, nlgeom=YES`) を biofilm mode に追加~~ | ✅ 完了 (2026-02-23) |
| ~~今すぐ~~ | ~~4条件 biofilm+NLGEOM Abaqus 実行・比較~~ | ✅ 完了 (2026-02-23) |
| ~~近期~~ | ~~E(DI) の物理的位置づけを論文 Methods に明記（Option A 記述）~~ | ✅ 完了 (2026-02-23) → `methods_supplement_fem.md` § 1 |
| ~~近期~~ | ~~α 固有ひずみの 0D 近似 → Abaqus `*INITIAL CONDITIONS` eigenstrain 追加~~ | ✅ 完了 (2026-02-23) → `biofilm_conformal_tet.py --growth-eigenstrain` + `compute_alpha_eigenstrain.py` |
| ~~近期~~ | ~~TMCMC 成長率 → Monod r の変換式を文書化~~ | ✅ 完了 (2026-02-23) → `methods_supplement_fem.md` § 3 |
| ~~近期~~ | ~~U_max 差分の条件間比較プロット強化（誤差棒 + DI-E 散布 + 複合図）~~ | ✅ 完了 (2026-02-23) → `plot_biofilm_nlgeom_enhanced.py` (5図) |
| **近期** | Option C: 二層 Tie モデル（biofilm 層 + tooth 層 統合） | 未着手 |
| **将来** | Neo-Hookean 材料モデル（UHYPER or FEniCS）→ 大変形で物理的に正確（現状は Abaqus 組み込み Neo-Hookean `--neo-hookean` により biofilm モードのみ対応済み） | 将来 |
| **将来** | Option D: 5 菌種 Hamilton PDE 定式化・FEniCS 実装 | 将来 |

---

## 6. 今後のプラン（優先度付き）

### 近期（論文提出前）

#### P1. 論文 Methods への Option A 記述追加（1日）
E(DI) が「基板有効剛性」であり Klempt 2024 の EPS 固有剛性（E~10 Pa）と異なることを明記。
下記文章を Methods に挿入：
> "The DI-dependent stiffness E(DI) represents the effective stiffness of the biofilm-covered substrate surface, not the intrinsic mechanical property of the biofilm EPS matrix (E ~ 10 Pa, Billings et al. 2015). The mapping reflects how microbial community composition alters the mechanical integrity of the periodontal surface."

#### P2. α 固有ひずみ（成長誘起応力）の 0D 近似実装（2–3日）
Klempt 2024 の核心「成長だけで内部応力が生じる」を近似的に取り込む。

手順：
1. `alpha_final = k_alpha * int(phi_avg(t), 0, t_end)` を 0D で計算（TMCMC ODE 出力から）
2. `eps_growth = alpha_final / 3.0`（等方膨張体積ひずみ）
3. Abaqus INP に `*INITIAL CONDITIONS, TYPE=STRESS` として等方圧縮固有ひずみを追加
4. 外部荷重なしで固有ひずみのみのジョブを実行 → 成長誘起 S_Mises を確認

対象ファイル：`biofilm_conformal_tet.py` に `--growth-eigenstrain ALPHA` オプション追加。

#### P3. U_max 差分の条件間比較プロット強化
現在の比較図 (`biofilm_nlgeom_comparison.png`) に
- 変位場のカラーマップ（条件ごと）
- U_max / U_mean の棒グラフ（誤差棒付き、posterior 複数サンプル）
を追加する。

### 中期（論文 revision 後）

#### M1. Option C: 二層 Tie モデル
biofilm 層（E~Pa）と tooth 層（E~GPa）を Abaqus `*TIE` 拘束で結合する。

```
[外部咬合荷重 1 MPa]
    ↓  tooth_layer.inp (E~GPa, substrate mode)
    |  *TIE: tooth outer face ↔ biofilm inner face
    ↓  biofilm_layer.inp (E~Pa, biofilm mode)
```

実装ポイント：
- `biofilm_3tooth_assembly.py` を参考に 2-part assembly INP を書くスクリプトを作成
- Node matching tolerance が重要（tet メッシュの node 位置ずれを許容する `*TIE, ADJUST=YES`）
- 歯肉溝液圧（100 Pa）は biofilm 外表面に、咬合力（1 MPa）は tooth 上面に分けて印加

期待される知見：咬合力が biofilm 層にどの程度伝達されるか（歯→バイオフィルムへの応力伝達）。

#### M2. Neo-Hookean 材料モデル
大変形（ε > 0.3）では線形弾性（NLGEOM であっても）は材料則として不適切。
Klempt 2024 が採用している超弾性（μ ≈ 1.7 Pa, K ≈ 10 Pa 程度）に移行する。

```fortran
! Abaqus UHYPER subroutine
! W = mu/2 * (I1_bar - 3) + K/2 * (J-1)^2
```

現状のステータス:
- biofilm_conformal_tet.py に `--neo-hookean` オプションを実装済み。
- Abaqus 組み込み `*Hyperelastic, Neo Hooke` を用い、E(DI), ν から μ, K（→ C10, D1）を計算して各 DI ビンに割当て。
- eigenstrain（GROWTH step, thermal analogy）は線形版と同一ロジックで適用。

将来の拡張:
- 上記 Neo-Hookean を UHYPER サブルーチンとして書き下ろし、より一般的な内部変数や成長 PDE 連成に対応。
- または FEniCS（Dolfin-X）で Neo-Hookean モデルを実装し、biofilm 層を FEniCS、tooth 層を Abaqus で連成させる（高コスト）。

### 将来（マルチスケール・論文次版）

#### L1. 5菌種 Hamilton PDE（FEniCS）
Soleimani et al. (2023, Sci. Rep.) の 2 種版を 5 種に拡張。
TMCMC 推定の a_ij 行列を結合項として組み込む。FEniCS-dolfinx + MPI 並列。

#### L2. Option D: ミクロ（μm）↔ マクロ（mm）二スケール連成
- ミクロ: Hamilton FEM でバイオフィルム成長・固有ひずみ場 α(x) を生成
- マクロ: Abaqus で α(x) を材料点入力として応力解析
- スケール変換: homogenization（均質化理論）または concurrent multiscale

---

## 7. 論文での位置づけ（Related Work との差分ストーリー）

```
Klempt 2024         : バイオフィルム固有の成長力学（μm, E~10 Pa, 単種, 現象論パラメータ）
                         ↕ 本研究との補完関係
本研究              : 菌叢組成→基板力学リスク（mm, E~GPa, 5種, TMCMC不確実性定量化）
```

「Klempt らのアプローチはバイオフィルム内部の力学・形態進化を解明する一方、
 本研究は菌叢構成の変化が顎・歯列スケールの力学リスクをどう変えるかを、
 ベイズ推定による不確実性定量化とともに評価する。両者は相補的スケールを扱い、
 将来的なマルチスケール結合（本論文 §Future Work）へとつながる。」
