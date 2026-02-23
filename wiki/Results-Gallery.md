# Results Gallery

## Best Runs: 2026-02-08 (1000 particles, ~90 h)

### MAP RMSE Summary

| Species | Comm. Static | Comm. HOBIC | Dysb. Static | Dysb. HOBIC |
|---------|:-----------:|:-----------:|:-----------:|:-----------:|
| *S. oralis* | 0.0935 | 0.1044 | 0.0256 | 0.0416 |
| *A. naeslundii* | 0.0422 | 0.0807 | 0.0566 | 0.0706 |
| *V. dispar* | 0.0604 | 0.0458 | 0.0748 | 0.1069 |
| *F. nucleatum* | 0.0210 | 0.0137 | 0.0291 | 0.0807 |
| *P. gingivalis* | 0.0191 | 0.0169 | 0.0645 | 0.0562 |
| **Total** | **0.0547** | **0.0632** | **0.0538** | **0.0746** |

---

## TMCMC Figures

### Species Interaction Network

![Interaction network](../data_5species/interaction_network.png)

Inferred 5-species interaction network (Dysbiotic HOBIC). Positive weights (blue) = facilitation, negative (red) = inhibition. The large a₃₅ (Vd→Pg) and a₄₅ (Fn→Pg) confirm bridge-mediated dysbiosis.

---

### MAP Posterior Fit — Dysbiotic HOBIC

![MAP fit DH](../data_5species/_runs/Dysbiotic_HOBIC_20260208_002100/figures/TSM_simulation_Dysbiotic_HOBIC_MAP_Fit_with_data.png)

MAP estimate vs. measured data (Dysbiotic HOBIC). The Pg "surge" driven by bridge organisms is well-captured.

---

### Posterior Predictive Band — Dysbiotic HOBIC

![Posterior band DH](../data_5species/_runs/Dysbiotic_HOBIC_20260208_002100/figures/posterior_predictive_Dysbiotic_HOBIC_PosteriorBand.png)

90% credible interval from 1000 posterior samples. Uncertainty is tightest for dominant commensal species and widest for bridge organisms.

---

### Interaction Heatmap — Dysbiotic HOBIC

![Interaction heatmap DH](../data_5species/_runs/Dysbiotic_HOBIC_20260208_002100/figures/pub_interaction_heatmap_Dysbiotic_HOBIC.png)

Inferred interaction matrix. Row = influenced species, column = influencing species.

---

### MAP Fit — Commensal Static (Negative Control)

![MAP fit CS](../data_5species/_runs/Commensal_Static_20260208_002100/figures/TSM_simulation_Commensal_Static_MAP_Fit_with_data.png)

Commensal Static: Pg is suppressed. a₃₅ and a₄₅ are near-zero, confirming absence of bridge-mediated facilitation.

---

### TMCMC β Schedule (Convergence)

![Beta schedule DH](../data_5species/_runs/Dysbiotic_HOBIC_20260208_002100/figures/Fig01_TMCMC_beta_schedule_Dysbiotic_HOBIC.png)

Tempering schedule β₀→β_J=1. Smooth progression indicates good annealing without particle collapse.

---

## FEM Figures

### 3D P. gingivalis Overview — All 4 Conditions

![Pg 3D panel](../FEM/_results_3d/panel_pg_overview_4conditions.png)

Spatial distribution of *P. gingivalis* (φ_Pg) across all 4 conditions. Dysbiotic HOBIC (bottom-right) shows highest Pg penetration depth.

---

### Dysbiotic Index — Cross-Condition Depth Profiles

![DI cross condition](../FEM/_di_credible/fig_di_cross_condition.png)

DI depth profiles with 90% credible intervals. Higher DI = more dysbiotic community composition near the tooth surface.

---

### Posterior Stress Uncertainty

![Stress violin](../FEM/_posterior_uncertainty/Fig1_stress_violin.png)

Von Mises stress distribution across 4 conditions, with uncertainty propagated from TMCMC posterior through the DI→E→FEM chain.

---

### DI Spatial Field — Dysbiotic HOBIC (3D)

![DI 3D DH](../FEM/_results_3d/Dysbiotic_HOBIC/fig4_dysbiotic_3d.png)

Spatial Dysbiotic Index field on the 3D tooth model (Dysbiotic HOBIC).

---

### Material Sensitivity Sweep

![Material sweep](../FEM/_material_sweep/figures/fig_A_combined_overview.png)

Effect of E_max / E_min / n on von Mises stress. S_Mises increases approximately linearly with E_max for substrate-mode loading.

---

## Run Directories

| Condition | Directory |
|-----------|-----------|
| Commensal Static | `data_5species/_runs/Commensal_Static_20260208_002100/` |
| Commensal HOBIC | `data_5species/_runs/Commensal_HOBIC_20260208_002100/` |
| Dysbiotic Static | `data_5species/_runs/Dysbiotic_Static_20260207_203752/` |
| Dysbiotic HOBIC | `data_5species/_runs/Dysbiotic_HOBIC_20260208_002100/` |
