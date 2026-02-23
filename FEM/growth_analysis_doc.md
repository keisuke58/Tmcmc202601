# Growth Load 解析ドキュメント — P1 Tooth 23 / 30 / 31

**最終更新**: 2026-02-23
**対象歯**: P1_Tooth_23, P1_Tooth_30, P1_Tooth_31 (Patient 1, 下顎)
**解析条件数**: 4条件 × 3歯 = 12 ジョブ

---

## 1. 解析フロー概要

```
TMCMC ODE 軌跡 (data_5species/_runs/)
  └─ compute_alpha_eigenstrain.py  ─→  alpha_final (条件別, k_alpha=0.05)
       │
       ├─ _di_credible/{COND}/p50_field.csv   (1D DI 中央値場 / 深さ軸)
       ├─ _di_credible/{COND}/coords.npy      (3D 座標グリッド [0,1]³)
       └─ _di_credible/{COND}/di_quantiles.npy (p05/p50/p95 DI 場)
              │
              ▼
       biofilm_conformal_tet.py   ─→  p{23|30|31}_growth_{COND}.inp
              │
              ▼
       abaqus job=P{23|30|31}Growth_{COND}   (cpus=4, interactive)
              │
              ▼
       P{23|30|31}Growth_{COND}.odb
              │
              ▼
       extract_growth_stress.py  ─→  stress_{COND}.csv  +  growth_summary.csv
              │
              ▼
       plot_growth_stress.py      ─→  fig_growth_*.png  (T23 単体)
       plot_3tooth_comparison.py  ─→  fig_3tooth_*.png  (T23/T30/T31 比較)
```

---

## 2. INP 生成パラメータ

### 共通設定（全歯・全条件）

| パラメータ | 値 | 備考 |
|-----------|-----|------|
| モード | `biofilm` | Pa スケール EPS 材料 (Klempt 2024) |
| E_max / E_min | 1000 / 10 Pa | バイオフィルム剛性範囲 |
| ν | 0.30 | |
| 厚さ | 0.2 mm | |
| 層数 | 4 | C3D4 四面体, 3 tet/prism |
| DI ビン数 | 20 | DI → 剛性の離散化 |
| 外面圧力 | 100 Pa | 歯肉溝液 (GCF) 圧力 |
| BC | `inner_fixed` (ENCASTRE) | 歯面固着 |
| nlgeom | YES | Biofilm mode 固定 |
| nutrient_factor | 0.85 | Monod 栄養補正 (Klempt: ċ = −g·c·φ) |
| Laplacian smooth | 3 回 | オフセット自己交差抑制 |
| Abaqus cpus | 4 | |

### 歯別メッシュ情報

| 歯 | STL 面数 | STL 頂点数 | Nodes | C3D4 Tets | 負体積 |
|----|---------|-----------|-------|-----------|-------|
| T23 | 3,590 | 1,797 | 8,985 | 43,080 | 0 |
| T30 | 7,928 | 3,966 | 19,830 | 95,136 | 0 |
| T31 | 6,710 | 3,357 | 16,785 | 80,520 | 0 |

> T23 のアスペクト比: min=0.075, mean=0.101 (理想=1.0)。薄層 C3D4 では必然的に低くなる。

---

## 3. 固有ひずみパラメータ（条件別）

### alpha_final 計算コマンド

```bash
cd Tmcmc202601/FEM
python3 compute_alpha_eigenstrain.py --runs-base ../data_5species/_runs --k-alpha 0.05
```

### 条件別パラメータ

| 条件 | 代表 run | alpha_final | ×0.85 = alpha_eff | eps_growth = α/3 |
|------|---------|------------|------------------|-----------------|
| commensal_static | Commensal_Static_20260204_062733 | 0.0202 | **0.01717** | 0.005723 |
| commensal_hobic | Commensal_HOBIC_20260205_003113 | 0.0169 | **0.01436** | 0.004788 |
| dh_baseline | Dysbiotic_HOBIC_20260205_013530 | 0.0083 | **0.007055** | 0.002352 |
| dysbiotic_static | Dysbiotic_Static_20260207_022019 | 0.0207 | **0.01759** | 0.005865 |

### 空間固有ひずみ場

GROWTH ステップで per-node 温度により DI 空間分布を付与:

```
T_node(x) = T_mean × DI_p50(x) / DI_mean,   T_mean = alpha_eff / 3
```

| 条件 | DI_mean | T_mean | T_min | T_max |
|------|---------|--------|-------|-------|
| commensal_static | 0.00971 | 0.005693 | 0.001599 | 0.01327 |
| commensal_hobic  | 0.00990 | 0.004763 | 0.001332 | 0.01107 |
| dh_baseline      | 0.00852 | 0.002327 | 0.000746 | 0.005643 |
| dysbiotic_static | 0.00950 | 0.005831 | 0.001594 | 0.01372 |

---

## 4. Abaqus ジョブ実行（再現コマンド）

```bash
# ── Tooth 23 ──────────────────────────────────────────────────
cd Tmcmc202601/FEM/_growth_job_runs
for COND in commensal_static commensal_hobic dh_baseline dysbiotic_static; do
    abaqus job=P23Growth_${COND} inp=../p23_growth_${COND}.inp cpus=4 interactive
done

# ── Tooth 30 ──────────────────────────────────────────────────
cd Tmcmc202601/FEM/_growth_job_runs_t30
for COND in commensal_static commensal_hobic dh_baseline dysbiotic_static; do
    abaqus job=P30Growth_${COND} inp=../p30_growth_${COND}.inp cpus=4 interactive
done

# ── Tooth 31 ──────────────────────────────────────────────────
cd Tmcmc202601/FEM/_growth_job_runs_t31
for COND in commensal_static commensal_hobic dh_baseline dysbiotic_static; do
    abaqus job=P31Growth_${COND} inp=../p31_growth_${COND}.inp cpus=4 interactive
done
```

### ステップ構成

| Step | 名前 | 内容 | 出力 |
|------|------|------|------|
| 1 | GROWTH | 熱アナロジー固有ひずみ (alpha_T=1.0, per-node *Temperature) | S, E, MISES |
| 2 | LOAD | GCF 外面圧力 100 Pa (集中力) | S, E, MISES, U, RF |

### 実行時間（実測）

| 歯 | Tets | 所要時間/ジョブ |
|----|------|--------------|
| T23 | 43,080 | ~15 s |
| T30 | 95,136 | ~30 s |
| T31 | 80,520 | ~25 s |

ライセンス: 76 tokens/ジョブ (220 total)

---

## 5. 応力抽出

```bash
# 各ジョブディレクトリで実行
abaqus python extract_growth_stress.py
```

- インスタンス名: `PART-1-1` (フラット INP の Abaqus 自動命名)
- GROWTH / LOAD 各最終フレームから S (応力) · U (変位) を抽出
- 出力: `stress_{COND}.csv` (要素別) + `growth_summary.csv` (条件別サマリ)

---

## 6. 結果サマリ

> Abaqus 内部単位 MPa → Pa 換算 (×10⁶)

### Tooth 23

| 条件 | σG_max [Pa] | σG_mean [Pa] | σL_max [Pa] | σL_mean [Pa] | U_max [mm] |
|------|------------|-------------|------------|-------------|----------|
| commensal_static | 10.3 | 4.26 | 83.9 | 52.3 | 0.0274 |
| commensal_hobic  | 8.52 | 3.52 | 84.9 | 53.0 | 0.0280 |
| dh_baseline      | 4.08 | 1.87 | 86.7 | 54.7 | 0.0259 |
| dysbiotic_static | 10.6 | 4.41 | 83.7 | 52.1 | 0.0270 |

### Tooth 30

| 条件 | σG_max [Pa] | σL_max [Pa] | U_max [mm] |
|------|------------|------------|----------|
| commensal_static | 10.2 | 78.9 | 0.0275 |
| commensal_hobic  | 8.47 | 80.0 | 0.0280 |
| dh_baseline      | 3.97 | 82.2 | 0.0258 |
| dysbiotic_static | 10.5 | 78.6 | 0.0271 |

### Tooth 31

| 条件 | σG_max [Pa] | σL_max [Pa] | U_max [mm] |
|------|------------|------------|----------|
| commensal_static | 11.3 | 81.8 | 0.0278 |
| commensal_hobic  | 9.37 | 82.8 | 0.0283 |
| dh_baseline      | 4.54 | 87.5 | 0.0261 |
| dysbiotic_static | 11.7 | 81.6 | 0.0274 |

### 3歯横断的考察

**GROWTH ステップ (固有ひずみ起源)**
- sigma ∝ alpha_eff: dysbiotic_static ≈ commensal_static > commensal_hobic ≫ dh_baseline (全歯共通)
- T31 の σG がわずかに高い → 歯形が細長く拘束が強い
- E_eff × eps_growth ≈ 300 Pa × 0.006 ≈ 2–3 Pa (mean と整合)

**LOAD ステップ (GCF 圧力支配)**
- T30 < T31 < T23 の順で σL が高い → 歯の有効面積・剛性依存
- dh_baseline は全歯で最高応力 (E_eff 低下 → 同圧力で変形大)
- GROWTH / LOAD ≈ 10–12%: 固有ひずみは補助的寄与

**条件間の分離度**
- LOAD step: 条件間差 ≈ 3–9 Pa (全圧力 100 Pa の 3–9%)
- GROWTH step: 条件間差は alpha_eff 比に直接対応 (dh_baseline は他の約 40%)

---

## 7. 可視化

### Tooth 23 単体 (`_growth_job_runs/`)

| ファイル | 内容 |
|---------|------|
| [fig_growth_1_condition_bar.png](_growth_job_runs/fig_growth_1_condition_bar.png) | 条件別 max/mean σ_Mises (GROWTH・LOAD) |
| [fig_growth_2_depth_profile.png](_growth_job_runs/fig_growth_2_depth_profile.png) | 深さ方向 σ プロファイル (±1σ バンド) |
| [fig_growth_3_spatial_load.png](_growth_job_runs/fig_growth_3_spatial_load.png) | LOAD 応力場 空間分布 (x-z 投影, 4 条件) |
| [fig_growth_4_spatial_growth.png](_growth_job_runs/fig_growth_4_spatial_growth.png) | GROWTH 応力場 空間分布 |
| [fig_growth_5_alpha_vs_stress.png](_growth_job_runs/fig_growth_5_alpha_vs_stress.png) | alpha_eff vs σ の線形関係 |

### 3歯比較 (`_growth_job_runs/`)

| ファイル | 内容 |
|---------|------|
| [fig_3tooth_comparison.png](_growth_job_runs/fig_3tooth_comparison.png) | σL_max + U_max: T23/T30/T31 × 4条件 バーチャート |
| [fig_3tooth_growth_stress.png](_growth_job_runs/fig_3tooth_growth_stress.png) | σG_max: T23/T30/T31 × 4条件 バーチャート |

### 再描画コマンド

```bash
cd Tmcmc202601/FEM/_growth_job_runs
python3 plot_growth_stress.py        # T23 図 (fig_growth_1〜5)
python3 plot_3tooth_comparison.py    # 3歯比較図
```

---

## 8. ファイル一覧

### INP / Validation（FEM/ 直下）

```
p{23|30|31}_growth_{COND}.inp          # Abaqus 入力ファイル (12 個)
p{23|30|31}_growth_{COND}_validation.txt  # メッシュ品質レポート
```

### ジョブ出力ディレクトリ

```
FEM/_growth_job_runs/          # T23 (4 ODB + CSV + 7 PNG)
FEM/_growth_job_runs_t30/      # T30 (4 ODB + CSV)
FEM/_growth_job_runs_t31/      # T31 (4 ODB + CSV)
```

各ディレクトリの共通構成:

```
P{23|30|31}Growth_{COND}.odb          # Abaqus ODB (CAE 可視化)
P{23|30|31}Growth_{COND}.sta          # 収束ログ
stress_{COND}.csv                     # 要素別: cx/cy/cz, mises_growth/load, s11/s22/s33
growth_summary.csv                    # 条件サマリ (1 行/条件)
extract_growth_stress.py              # ODB 抽出スクリプト (abaqus python)
```

---

## 9. 既知の課題・残作業

| 優先度 | 課題 | 対処方針 |
|-------|------|---------|
| 中 | **DI ビン分布** | 4 ビンのみ実効的 (n_bins=20 中 bins 3,5,6,8)。n_layers=8 化 or di_scale 縮小で改善。 |
| 低 | **アスペクト比** | mean≈0.10。C3D4 は一次要素で感度低いが、精度向上には C3D10 二次要素化を要検討。 |
| 低 | **k_alpha 感度** | 現在 k_alpha=0.05 固定。文献値不明のため k_alpha ∈ [0.01, 0.1] スウィープが有用。 |
| 低 | **他患者** | Patient 2 以降の歯 STL にも同パイプラインを適用可能。 |
| 低 | **CAE 可視化** | ODB を Abaqus CAE で開き contour plot → PNG として保存する工程を自動化検討。 |
