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

### Posterior Stress Uncertainty (updated 2026-02-24)

![Stress violin](../FEM/_posterior_uncertainty/Fig1_stress_violin.png)

Von Mises stress distribution across 4 conditions (20 TMCMC posterior samples each). Median [5th–95th CI] annotated per condition. dh-baseline shows widest spread (p95/p05 = 1.58×) due to unconstrained a₃₅; commensal/dysbiotic conditions are tight (1.05–1.17×).

![Stress summary](../FEM/_posterior_uncertainty/Fig5_stress_summary_panel.png)

6-panel summary: box plots (substrate+surface), parameter sensitivity (Spearman ρ), CI bars, and relative change vs. commensal static reference.

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

---

## Multiscale Coupling Results (2026-02-24)

### DI_0D — Condition Discrimination

| Condition | DI_0D | E_eff mean (Pa) |
|-----------|:-----:|:--------------:|
| Commensal | 0.047 | ~909 |
| Dysbiotic | 0.845 | ~33 |
| **Ratio** | **18×** | **28×** |

### α_Monod Spatial Profile

| Depth | α_Monod | ε_growth |
|-------|:-------:|:--------:|
| x=0 (tooth surface) | 0.004 | ~0% (nutrient-starved) |
| x=1 (saliva side) | 0.420 | 14% |
| **Gradient** | **101×** | **101×** |

### Pipeline Summary Figure

![Pipeline summary](../FEM/_pipeline_summary/pipeline_summary.png)

9-panel overview: pipeline schematic, DI_0D comparison, 1D nutrient field c(x), species profiles φᵢ(x), α_Monod(x), ε_growth(x), and effective stiffness E(x) for both conditions.

### Multiscale Comparison Figure

![Multiscale comparison](../FEM/_multiscale_results/multiscale_comparison.png)

Side-by-side: 0D ODE trajectories, 1D nutrient + species profiles, and eigenstrain fields for commensal vs dysbiotic.

---

→ See [Multiscale Coupling](Multiscale-Coupling) for the full technical guide.

---

## 2D Nutrient PDE Extension (2026-02-24, Issue #3)

Hamilton 0D ODE → 2D depth profile → steady-state Klempt nutrient PDE on 40×40 grid.

### 2D Heatmap Comparison

![2D comparison](../FEM/klempt2024_results/hamilton_rd_2d_pipeline/hamilton_rd_2d_comparison.png)

Left: total biofilm volume fraction φ_total(x,y). Center: nutrient field c(x,y) at g_eff=50. Right: *P. gingivalis* distribution. Top=Commensal, Bottom=Dysbiotic. Pg max ratio = **5.7×** between conditions.

### Cross-Section Comparison (y = 0.5)

![2D cross-section](../FEM/klempt2024_results/hamilton_rd_2d_pipeline/hamilton_rd_2d_cross_section.png)

1D cross-sections from the 2D simulation, directly comparable to the 1D pipeline. Panels: 0D ODE trajectories, final composition bar chart, depth profiles, nutrient profiles at g_eff=50, and nutrient sensitivity sweep (g_eff = 5, 20, 50).

### Condition Difference Maps

![2D difference](../FEM/klempt2024_results/hamilton_rd_2d_pipeline/hamilton_rd_2d_difference.png)

Δ(Dysbiotic − Commensal) for φ_total, nutrient c, and Pg. The Pg difference is concentrated near the tooth surface (x ≈ 0) with max Δφ_Pg ≈ 0.04.

### Summary Table (g_eff = 50)

| Metric | Commensal | Dysbiotic | Ratio |
|--------|:---------:|:---------:|:-----:|
| φ_total mean | 0.368 | 0.354 | 0.96 |
| c(tooth, y=0.5) | 0.028 | 0.031 | 1.09 |
| Thiele mod | 4.29 | 4.21 | 0.98 |
| **Pg max** | **0.0085** | **0.0484** | **5.7×** |
