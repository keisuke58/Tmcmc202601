# Tmcmc202601 — 5-Species Oral Biofilm: TMCMC + FEM Pipeline

**5種口腔バイオフィルムのベイズパラメータ推定 (TMCMC) と 3D FEM 応力解析の統合パイプライン**

---

## Overview

This project couples two computational stages for periodontal biofilm mechanics:

```
Hamilton ODE model (5 species)
         ↓
   TMCMC Bayesian estimation
         ↓  θ_MAP, posterior samples
   3D FEM reaction–diffusion
         ↓  φᵢ(x,y,z) species fields
   Dysbiotic Index (DI) field
         ↓  DI(x,y,z)
   E(DI) power-law mapping
         ↓  E(x,y,z)
   Abaqus FEM stress analysis
         ↓  S_Mises, U (substrate / surface)
```

**5 species**: *S. oralis* (So), *A. naeslundii* (An), *V. dispar* (Vd), *F. nucleatum* (Fn), *P. gingivalis* (Pg)

---

## Repository Structure

```
Tmcmc202601/
├── data_5species/          # TMCMC estimation pipeline
│   ├── core/               # TMCMC engine, evaluator, model ODE
│   ├── main/               # Entry points (estimate_reduced_nishioka.py)
│   ├── model_config/       # Prior bounds, model configuration JSON
│   └── _runs/              # Run outputs (theta_MAP, posterior samples)
│
├── FEM/                    # FEM stress analysis pipeline
│   ├── abaqus_biofilm_*.py # Abaqus INP generators (isotropic/aniso/CZM)
│   ├── biofilm_conformal_tet.py   # Conformal tet mesh + --mode biofilm/substrate
│   ├── jax_fem_reaction_diffusion_demo.py  # JAX-FEM Klempt 2024 demo
│   ├── jax_pure_reaction_diffusion_demo.py # Pure JAX PDE demo (egg morphology)
│   ├── jax_hamilton_*.py   # Hamilton PDE demos (0D / 1D)
│   ├── JAXFEM/             # JAX-based FEM modules
│   ├── klempt2024_gap_analysis.md  # Gap analysis vs Klempt (2024)
│   ├── FEM_README.md       # Detailed FEM pipeline documentation
│   └── _*/                 # Output directories (results, sweeps, plots)
│
├── tmcmc/                  # Core TMCMC library
├── docs/                   # LaTeX reports and slides
├── run_bridge_sweep.py     # Parameter sweep runner
├── analyze_sweep_results.py
└── PROJECT_SUMMARY.md      # Full progress summary (JP)
```

---

## TMCMC: Bayesian Parameter Estimation

### Model

- **ODE system**: 5-species Hamilton principle biofilm (20 parameters)
- **Inference**: Transitional Markov Chain Monte Carlo (TMCMC)
- **Prior**: Uniform with physiologically motivated bounds
- **Hill gate**: interaction nonlinearity (K_hill, n_hill fixed)

### Key Parameters

| Index | Symbol | Meaning |
|-------|--------|---------|
| θ[18] | a₃₅ | *V. dispar* → *P. gingivalis* facilitation |
| θ[19] | a₄₅ | *F. nucleatum* → *P. gingivalis* facilitation |
| θ[12] | a₂₃ | *S. oralis* → *A. naeslundii* cross-feeding |

### Best Run (mild_weight, 2026-02-18)

| Species | RMSE prev | RMSE (mild) |
|---------|-----------|-------------|
| S. oralis | 0.036 | **0.034** |
| A. naeslundii | 0.129 | **0.105** |
| V. dispar | 0.213 | 0.269 |
| F. nucleatum | 0.088 | 0.161 |
| P. gingivalis | 0.435 | **0.103** |
| **Total** | **0.228** | **0.156** (−31%) |

Settings: 150 particles, 8 stages, λ_Pg=2.0, λ_late=1.5, a₃₅/a₄₅ bounds [0, 5]

---

## FEM: Stress Analysis Pipeline

### Dysbiotic Index → Stiffness Mapping

```
DI(x) = 1 − H(x) / log(5)         H = Shannon entropy of species mix
r(x)  = clamp(DI(x) / s, 0, 1)    s = 0.025778
E(x)  = E_max · (1−r)ⁿ + E_min · r
```

Default: E_max = 10 GPa (commensal), E_min = 0.5 GPa (dysbiotic), n = 2

### Analysis Modes

| Mode | Scale | Purpose |
|------|-------|---------|
| `--mode substrate` | GPa | Dental substrate effective stiffness; S_Mises is primary metric |
| `--mode biofilm` | Pa | EPS matrix (Klempt 2024); U_max is primary metric |
| `--neo-hookean` | Pa | Neo-Hookean hyperelastic for large strains (biofilm mode) |

### Biofilm Mode Results (4 conditions, NLGEOM, 2026-02-23)

| Condition | DI_mean | E_mean (Pa) | U_max (mm) |
|-----------|---------|------------|------------|
| dh_baseline | 0.00852 | 451 | 0.0267 |
| dysbiotic_static | 0.00950 | 403 | 0.0286 |
| commensal_static | 0.00971 | 392 | 0.0290 |
| commensal_hobic | 0.00990 | 383 | **0.0294** (+10%) |

→ Displacement (not stress) discriminates conditions under pressure-controlled BC.

### JAX-FEM Demos (Klempt 2024 benchmark)

`FEM/jax_fem_reaction_diffusion_demo.py` implements the steady-state nutrient transport PDE from Klempt et al. (2024):

```
−D_c Δc + g φ₀(x) c/(k+c) = 0    in Ω = [0,1]²
c = 1                               on ∂Ω
```

- Egg-shaped biofilm morphology φ₀ (Klempt 2024 Fig. 1): ax=0.35, ay=0.25, skew=0.3
- Thiele modulus ~4 (diffusion-limited regime)
- Result: c_min ≈ 0.31 inside biofilm; Newton converges in 4 iterations
- Full JAX autodiff: ∂(loss)/∂D_c demonstrated

---

## Quick Start

### TMCMC Estimation

```bash
cd Tmcmc202601
python data_5species/main/estimate_reduced_nishioka.py \
    --n-particles 150 --n-stages 8 \
    --lambda-pg 2.0 --lambda-late 1.5
```

### FEM Stress Analysis

```bash
cd Tmcmc202601/FEM

# Posterior ensemble → DI fields → Abaqus stress
python run_posterior_abaqus_ensemble.py
python aggregate_di_credible.py
python run_material_sensitivity_sweep.py

# Biofilm mode (Pa-scale EPS, NLGEOM)
python biofilm_conformal_tet.py \
    --stl external_tooth_models/.../P1_Tooth_23.stl \
    --di-csv abaqus_field_dh_3d.csv \
    --out p23_biofilm.inp --mode biofilm
```

### JAX-FEM Klempt 2024 Demo

```bash
# klempt_fem conda env (Python 3.11, jax-fem 0.0.11)
~/.pyenv/versions/miniconda3-latest/envs/klempt_fem/bin/python \
    FEM/jax_fem_reaction_diffusion_demo.py
# → FEM/klempt2024_results/klempt2024_nutrient.vtu + .png
```

---

## Environment

| Component | Version / Path |
|-----------|---------------|
| Python (TMCMC) | system Python 3.x |
| Python (JAX-FEM) | klempt_fem conda env (Python 3.11) |
| JAX | 0.9.0.1 (x64 enabled) |
| jax-fem | 0.0.11 |
| Abaqus | 2023 (HPC cluster) |

---

## Key References

- **Klempt, Soleimani, Wriggers, Junker (2024)**: *A Hamilton principle-based model for diffusion-driven biofilm growth*, Biomech Model Mechanobiol 23:2091–2113. [DOI](https://doi.org/10.1007/s10237-024-01883-x)
- **Soleimani et al. (2016, 2019)**: Periodontal ligament FEM with GPa-scale effective stiffness
- **Billings et al. (2015)**: EPS matrix stiffness E ≈ 10 Pa

---

## Author

Nishioka — Keio University / IKM Leibniz Universität Hannover, 2026
