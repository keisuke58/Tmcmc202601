---
layout: default
title: Tmcmc202601 — 5-Species Oral Biofilm Pipeline
description: Bayesian parameter estimation (TMCMC) and 3D FEM stress analysis for oral biofilm dysbiosis
---

# Tmcmc202601

**5-Species Oral Biofilm: TMCMC Bayesian Inference + 3D FEM Stress Analysis**

> Keio University / IKM Leibniz Universität Hannover · 2026

---

## What is this?

This pipeline quantifies **how bridge organisms drive periodontal dysbiosis** by:

1. Inferring species interaction strengths (*a*ᵢⱼ) from in vitro longitudinal data using **Transitional MCMC (TMCMC)**
2. Propagating the posterior uncertainty into **3D Abaqus FEM stress analysis** of periodontal tissue

The 5 species are: *S. oralis*, *A. naeslundii*, *V. dispar* (bridge), *F. nucleatum* (bridge), *P. gingivalis* (pathogen).

---

## Key Results

### TMCMC MAP Fit — Dysbiotic HOBIC

![MAP fit](../data_5species/_runs/Dysbiotic_HOBIC_20260208_002100/figures/TSM_simulation_Dysbiotic_HOBIC_MAP_Fit_with_data.png)

### 3D Composition Fields — P. gingivalis (All 4 Conditions)

![Pg 3D overview](../FEM/_results_3d/panel_pg_overview_4conditions.png)

### Posterior Stress Uncertainty

![Stress violin](../FEM/_posterior_uncertainty/Fig1_stress_violin.png)

---

## MAP RMSE Summary (1000 particles, 2026-02-08)

| Species | Comm. Static | Comm. HOBIC | Dysb. Static | Dysb. HOBIC |
|---------|:-----------:|:-----------:|:-----------:|:-----------:|
| *S. oralis* | 0.0935 | 0.1044 | 0.0256 | 0.0416 |
| *A. naeslundii* | 0.0422 | 0.0807 | 0.0566 | 0.0706 |
| *V. dispar* | 0.0604 | 0.0458 | 0.0748 | 0.1069 |
| *F. nucleatum* | 0.0210 | 0.0137 | 0.0291 | 0.0807 |
| *P. gingivalis* | 0.0191 | 0.0169 | 0.0645 | 0.0562 |
| **Total** | **0.0547** | **0.0632** | **0.0538** | **0.0746** |

---

## Links

- [GitHub Repository](https://github.com/keisuke58/Tmcmc202601)
- [Full README](https://github.com/keisuke58/Tmcmc202601#readme)
- [Issues & Feature Requests](https://github.com/keisuke58/Tmcmc202601/issues)
- [CITATION.cff](https://github.com/keisuke58/Tmcmc202601/blob/master/CITATION.cff)

---

## Quick Start

```bash
# TMCMC estimation
python data_5species/main/estimate_reduced_nishioka.py \
    --n-particles 150 --n-stages 8 \
    --lambda-pg 2.0 --lambda-late 1.5

# JAX-FEM nutrient transport demo
~/.pyenv/versions/miniconda3-latest/envs/klempt_fem/bin/python \
    FEM/jax_fem_reaction_diffusion_demo.py
```

---

*Nishioka Keisuke · Keio University / IKM Leibniz Universität Hannover · 2026*
