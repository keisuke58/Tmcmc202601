# Architecture

本ドキュメントは、コードベースの設計思想と主要モジュール間の依存関係を記述する。

## 全体構成

```
┌─────────────────────────────────────────────────────────────┐
│                    Tmcmc202601                               │
│                                                              │
│  ┌──────────────────┐    ┌──────────────────────────────┐   │
│  │  data_5species/   │    │  FEM/                         │   │
│  │  (Stage 1: TMCMC) │───▶│  (Stage 2: FEM + Multiscale) │   │
│  └──────────────────┘    └──────────────────────────────┘   │
│           │                          │                       │
│           ▼                          ▼                       │
│  ┌──────────────────┐    ┌──────────────────────────────┐   │
│  │  _runs/ (output)  │    │  _results_3d/ _stress_2d/    │   │
│  │  theta_MAP, post. │    │  _multiscale_results/        │   │
│  └──────────────────┘    └──────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Stage 1: TMCMC Bayesian Estimation

### モジュール構成

```
data_5species/
├── core/
│   ├── tmcmc.py           # TMCMC エンジン: run_TMCMC(), sequential tempering
│   ├── evaluator.py       # 尤度計算: build_likelihood_weights(), log_likelihood()
│   ├── nishioka_model.py  # パラメータ index mapping, interaction graph
│   └── mcmc.py            # MH-MCMC ステップ
├── main/
│   └── estimate_reduced_nishioka.py  # エントリーポイント
├── model_config/
│   ├── prior_bounds.json             # 事前分布 bounds (現在: mild-weight)
│   └── prior_bounds_original.json    # バックアップ (original wide bounds)
└── _runs/                            # 出力 (条件別ディレクトリ)
```

### データフロー

```
prior_bounds.json
       │
       ▼
estimate_reduced_nishioka.py
       │
       ├──▶ nishioka_model.py    (ODE定義, パラメータ mapping)
       │         │
       │         ▼
       ├──▶ evaluator.py         (尤度計算, 重み付け)
       │         │
       │         ▼
       └──▶ tmcmc.py             (TMCMC sampling loop)
                 │
                 ▼
         _runs/{condition}/
         ├── theta_MAP.json
         ├── theta_MEAN.json
         ├── posterior_samples.csv
         └── figures/
```

### 主要パラメータフロー

- **入力**: prior_bounds.json (20 params: r_i, d_i, a_ij)
- **ODE**: Hamilton variational (improved_5species_jit.py)
- **Hill gate**: H(φ) = φⁿ/(Kⁿ+φⁿ), K=0.05, n=4 (固定)
- **出力**: θ_MAP, θ_MEAN, N=1000 posterior samples

## Stage 2: FEM Stress Analysis + Multiscale

### モジュール構成

```
FEM/
├── 3D FEM Pipeline
│   ├── biofilm_conformal_tet.py      # STL → conformal C3D4 mesh + INP
│   ├── run_posterior_abaqus_ensemble.py  # Posterior ensemble → Abaqus batch
│   └── posterior_uncertainty_propagation.py  # σ_Mises 90% CI
│
├── Multiscale Coupling
│   ├── multiscale_coupling_1d.py     # 0D+1D pipeline → α_Monod(x) CSV
│   ├── multiscale_coupling_2d.py     # 2D reaction-diffusion extension
│   ├── generate_hybrid_macro_csv.py  # 0D DI × 1D spatial α → Hybrid CSV
│   └── generate_abaqus_eigenstrain.py  # Hybrid CSV → Abaqus INP (thermal)
│
├── JAX-FEM Demos
│   ├── jax_fem_reaction_diffusion_demo.py  # Klempt 2024 benchmark
│   └── jaxfem_adjoint_poc.py               # Adjoint inverse problem PoC
│
├── Analysis & Visualization
│   ├── plot_species_competition.py   # 6-panel competition analysis
│   ├── generate_pipeline_summary.py  # 9-panel summary
│   ├── plot_stress_comparison.py     # von Mises comparison
│   └── run_condition_sweep.py        # K_hill × n_hill sweep
│
└── JAXFEM/                           # JAX-based FEM modules
    └── core_hamilton_2d_nutrient.py   # 2D Hamilton + nutrient PDE
```

### DI → E(x) マッピングパイプライン

```
θ_MAP (from Stage 1)
    │
    ▼
Hamilton ODE integration → φᵢ(x)
    │
    ▼
Shannon entropy: H(x) = −Σ φᵢ ln φᵢ
    │
    ▼
DI(x) = 1 − H(x)/ln5
    │
    ▼
E(x) = E_max(1−r)ⁿ + E_min·r,  r = clamp(DI/s, 0, 1)
    │
    ▼
Abaqus 3D FEM → σ_Mises(x), U(x)
```

### 3 つの E モデル

| Model | 入力 | 特性 |
|-------|------|------|
| DI (entropy) | Shannon entropy → DI | 条件間差 28x (推奨) |
| φ_Pg (Hill) | Pg volume fraction | 全条件 ≈ 1000 Pa (区別不可) |
| Virulence | Pg+Fn weighted | 中間的 |

## 設計判断

### なぜ DI (entropy) が φ_Pg より優れるか

Hamilton ODE の構造的制約:
- Pg は locked edges (a₁₅=a₂₅=0) + Hill gate (K=0.05)
- Vd→Fn→Pg カスケード依存
- 結果: 全条件で φ_Pg < 0.10 → 条件間区別不可

DI は多様性の喪失を検出 → ecosystem-level dysbiosis indicator として適切

### Hybrid アプローチの理由

- 1D/2D 拡散は種組成を均質化 → DI の空間変化消失
- 条件間差異は 0D ODE の DI_0D が保持
- 空間構造は nutrient field c(x) が支配 → α_Monod(x)
- 解決策: 0D DI_0D (条件別) × 1D α_Monod(x) (空間) の積

## 依存関係

```
numpy, scipy, matplotlib  ─── TMCMC (Stage 1)
jax, jax-fem, basix       ─── JAX-FEM demos, multiscale
Abaqus 2023               ─── 3D FEM stress analysis
fTetWild                   ─── Conformal tet meshing
```
