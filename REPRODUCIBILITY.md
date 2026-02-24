# Reproducibility Guide

本ドキュメントは、本研究のすべての結果を再現するための手順を記載する。

## 1. 環境構築

### 1.1 TMCMC (Parameter Estimation)

```bash
# Python 3.11+ with scientific stack
pip install numpy scipy matplotlib
```

### 1.2 JAX-FEM (Multiscale Coupling)

```bash
conda create -n klempt_fem python=3.11
conda activate klempt_fem
pip install jax[cpu]==0.9.0.1 jax-fem==0.0.11 basix==0.10.0 matplotlib

# jax-fem 0.0.11 patch (petsc4py optional)
# See README.md "JAX-FEM Setup" section for solver.py patch details
```

### 1.3 Abaqus (3D FEM)

- Abaqus 2023 (institutional license)
- fTetWild for conformal tet meshing

## 2. データ

### 2.1 実験データ (Heine et al. 2025)

- 5 種口腔バイオフィルム in vitro CFU/mL データ
- 4 条件 × 5 種 × 5 時間点 = 100 data points
- 配置: `data_5species/model_config/`

### 2.2 歯モデル (Open-Full-Jaw)

- Gholamalizadeh et al. (2022) より Patient 1 下顎歯
- 配置: `FEM/external_tooth_models/Open-Full-Jaw-main/`
- ダウンロード: https://doi.org/10.1016/j.cmpb.2022.107009 の supplementary data

## 3. TMCMC ベイズ推定の再現

### 3.1 単一条件 (Dysbiotic HOBIC)

```bash
python data_5species/main/estimate_reduced_nishioka.py \
    --n-particles 150 --n-stages 8 \
    --lambda-pg 2.0 --lambda-late 1.5
```

- 期待出力: `data_5species/main/_runs/<timestamp>/`
- 所要時間: 約 2–4 h (single CPU)
- 期待 MAP RMSE: < 0.08

### 3.2 全 4 条件 (Production Run)

```bash
python data_5species/main/estimate_reduced_nishioka.py \
    --n-particles 1000 --n-stages 8 \
    --lambda-pg 2.0 --lambda-late 1.5 \
    --conditions all
```

- 所要時間: 約 90 h (single node)
- Best runs (2026-02-08): `data_5species/main/_runs/` 内の各条件ディレクトリ

### 3.3 Prior Bounds 比較実験

```bash
# Mild-weight bounds (recommended)
# prior_bounds.json: a35 [0, 5], a45 [0, 5]
python data_5species/main/estimate_reduced_nishioka.py \
    --n-particles 150 --n-stages 8

# Original bounds (baseline comparison)
# prior_bounds_original.json: a35 [0, 30], a45 [0, 20]
cp data_5species/model_config/prior_bounds_original.json \
   data_5species/model_config/prior_bounds.json
python data_5species/main/estimate_reduced_nishioka.py \
    --n-particles 150 --n-stages 8
```

## 4. マルチスケール連成の再現

### 4.1 0D+1D Pipeline

```bash
~/.pyenv/versions/miniconda3-latest/envs/klempt_fem/bin/python \
    FEM/multiscale_coupling_1d.py
```

- 出力: `FEM/_multiscale_results/macro_eigenstrain_{condition}.csv`
- 所要時間: < 2 min

### 4.2 Hybrid CSV 生成

```bash
~/.pyenv/versions/miniconda3-latest/envs/klempt_fem/bin/python \
    FEM/generate_hybrid_macro_csv.py
```

- 出力: `FEM/_multiscale_results/macro_eigenstrain_{condition}_hybrid.csv`

### 4.3 Abaqus INP 生成

```bash
~/.pyenv/versions/miniconda3-latest/envs/klempt_fem/bin/python \
    FEM/generate_abaqus_eigenstrain.py
```

- 出力: 3 models × 4 conditions = 12 INP files

## 5. FEM 応力解析の再現

### 5.1 Biofilm Mode (Pa-scale)

```bash
cd FEM
python biofilm_conformal_tet.py \
    --stl external_tooth_models/.../P1_Tooth_23.stl \
    --di-csv abaqus_field_dh_3d.csv \
    --out p23_biofilm.inp --mode biofilm
```

### 5.2 Posterior Uncertainty Propagation

```bash
python FEM/posterior_uncertainty_propagation.py
```

- 出力: `FEM/_posterior_uncertainty/Fig1_stress_violin.png`

## 6. JAX-FEM Klempt 2024 Benchmark

```bash
~/.pyenv/versions/miniconda3-latest/envs/klempt_fem/bin/python \
    FEM/jax_fem_reaction_diffusion_demo.py
```

- 期待結果: c_min ≈ 0.31 (Klempt 2024 Fig. 1 参照値 ~0.3)
- Newton iterations: 4

## 7. 図の再生成

### 主要図一覧

| Figure | Script | Output |
|--------|--------|--------|
| Pipeline overview | `docs/` (manual) | `docs/overview_figs/fig1_pipeline.png` |
| MAP fit | TMCMC run output | `_runs/.../TSM_simulation_*_MAP_Fit_with_data.png` |
| Posterior band | TMCMC run output | `_runs/.../posterior_predictive_*_PosteriorBand.png` |
| 3 model comparison | `FEM/generate_hybrid_macro_csv.py` | `FEM/_multiscale_results/hybrid_3model_comparison.png` |
| Species competition | `FEM/plot_species_competition.py` | `FEM/_multiscale_results/species_competition_analysis.png` |
| Stress violin | `FEM/posterior_uncertainty_propagation.py` | `FEM/_posterior_uncertainty/Fig1_stress_violin.png` |
| a35 sweep | `FEM/run_condition_sweep.py` | `FEM/klempt2024_results/a35_sweep/` |

## 8. 期待される主要数値

| 指標 | 期待値 | 許容範囲 |
|------|--------|---------|
| MAP RMSE (全条件) | < 0.075 | ±0.01 |
| Pg RMSE (mild-weight) | ~0.103 | ±0.02 |
| DI_0D commensal | ~0.05 | ±0.02 |
| DI_0D dysbiotic | ~0.84 | ±0.05 |
| E_eff commensal (Pa) | ~909 | ±50 |
| E_eff dysbiotic (Pa) | ~33 | ±10 |
| c_min (Klempt benchmark) | ~0.31 | ±0.02 |

## 注意事項

- TMCMC は確率的アルゴリズムのため、実行ごとに結果が微小に変動する（seed 固定で再現可能）
- Abaqus ライセンスが必要な解析は、INP ファイル生成までは Abaqus なしで実行可能
- Git LFS が必要なファイル: `FEM/external_tooth_models/Open-Full-Jaw-main/dataset/Patient_1.zip`
