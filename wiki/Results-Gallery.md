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

![Interaction network](images/interaction_network.png)

Inferred 5-species interaction network (Dysbiotic HOBIC). Positive weights (blue) = facilitation, negative (red) = inhibition. The large a₃₅ (Vd→Pg) and a₄₅ (Fn→Pg) confirm bridge-mediated dysbiosis.

---

### MAP Posterior Fit — Dysbiotic HOBIC

![MAP fit DH](images/map_fit_dysbiotic_hobic.png)

MAP estimate vs. measured data (Dysbiotic HOBIC). The Pg "surge" driven by bridge organisms is well-captured.

---

### Posterior Predictive Band — Dysbiotic HOBIC

![Posterior band DH](images/posterior_band_dysbiotic_hobic.png)

90% credible interval from 1000 posterior samples. Uncertainty is tightest for dominant commensal species and widest for bridge organisms.

---

### Interaction Heatmap — Dysbiotic HOBIC

![Interaction heatmap DH](images/interaction_heatmap_dysbiotic_hobic.png)

Inferred interaction matrix. Row = influenced species, column = influencing species.

---

### MAP Fit — Commensal Static (Negative Control)

![MAP fit CS](images/map_fit_commensal_static.png)

Commensal Static: Pg is suppressed. a₃₅ and a₄₅ are near-zero, confirming absence of bridge-mediated facilitation.

---

### TMCMC β Schedule (Convergence)

![Beta schedule DH](images/beta_schedule_dysbiotic_hobic.png)

Tempering schedule β₀→β_J=1. Smooth progression indicates good annealing without particle collapse.

---

## FEM Figures

### 3D P. gingivalis Overview — All 4 Conditions

![Pg 3D panel](images/panel_pg_overview_4conditions.png)

Spatial distribution of *P. gingivalis* (φ_Pg) across all 4 conditions. Dysbiotic HOBIC (bottom-right) shows highest Pg penetration depth.

---

### Dysbiotic Index — Cross-Condition Depth Profiles

![DI cross condition](images/di_cross_condition.png)

DI depth profiles with 90% credible intervals. Higher DI = more dysbiotic community composition near the tooth surface.

---

### Posterior Stress Uncertainty (updated 2026-02-24)

![Stress violin](images/stress_violin.png)

Von Mises stress distribution across 4 conditions (20 TMCMC posterior samples each). Median [5th–95th CI] annotated per condition. dh-baseline shows widest spread (p95/p05 = 1.58×) due to unconstrained a₃₅; commensal/dysbiotic conditions are tight (1.05–1.17×).

![Stress summary](images/stress_summary_panel.png)

6-panel summary: box plots (substrate+surface), parameter sensitivity (Spearman ρ), CI bars, and relative change vs. commensal static reference.

---

### DI Spatial Field — Dysbiotic HOBIC (3D)

![DI 3D DH](images/di_3d_dysbiotic_hobic.png)

Spatial Dysbiotic Index field on the 3D tooth model (Dysbiotic HOBIC).

---

### Material Sensitivity Sweep

![Material sweep](images/material_sweep_overview.png)

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

![Pipeline summary](images/pipeline_summary.png)

9-panel overview: pipeline schematic, DI_0D comparison, 1D nutrient field c(x), species profiles φᵢ(x), α_Monod(x), ε_growth(x), and effective stiffness E(x) for both conditions.

### Multiscale Comparison Figure

![Multiscale comparison](images/multiscale_comparison.png)

Side-by-side: 0D ODE trajectories, 1D nutrient + species profiles, and eigenstrain fields for commensal vs dysbiotic.

---

> See [Multiscale Coupling](Multiscale-Coupling) for the full technical guide.

---

## 2D Nutrient PDE Extension (2026-02-24, Issue #3)

Hamilton 0D ODE → 2D depth profile → steady-state Klempt nutrient PDE on 40×40 grid.

### 2D Heatmap Comparison

![2D comparison](images/hamilton_rd_2d_comparison.png)

Left: total biofilm volume fraction φ_total(x,y). Center: nutrient field c(x,y) at g_eff=50. Right: *P. gingivalis* distribution. Top=Commensal, Bottom=Dysbiotic. Pg max ratio = **5.7×** between conditions.

### Cross-Section Comparison (y = 0.5)

![2D cross-section](images/hamilton_rd_2d_cross_section.png)

1D cross-sections from the 2D simulation, directly comparable to the 1D pipeline. Panels: 0D ODE trajectories, final composition bar chart, depth profiles, nutrient profiles at g_eff=50, and nutrient sensitivity sweep (g_eff = 5, 20, 50).

### Condition Difference Maps

![2D difference](images/hamilton_rd_2d_difference.png)

Δ(Dysbiotic − Commensal) for φ_total, nutrient c, and Pg. The Pg difference is concentrated near the tooth surface (x ≈ 0) with max Δφ_Pg ≈ 0.04.

### Summary Table (g_eff = 50)

| Metric | Commensal | Dysbiotic | Ratio |
|--------|:---------:|:---------:|:-----:|
| φ_total mean | 0.368 | 0.354 | 0.96 |
| c(tooth, y=0.5) | 0.028 | 0.031 | 1.09 |
| Thiele mod | 4.29 | 4.21 | 0.98 |
| **Pg max** | **0.0085** | **0.0484** | **5.7×** |

---

## Posterior → Nutrient Uncertainty (2026-02-24, Issue #7)

50 posterior samples per condition → Hamilton 0D (T*=25) → 2D biofilm profile → steady-state nutrient PDE → c_min distribution.

### c_min Distribution

![c_min distribution](images/fig1_cmin_distribution.png)

Bimodal posterior structure: most samples converge to V.dispar-dominated equilibrium (c_min ≈ 0.028), with a secondary mode at c_min ≈ 0.034 corresponding to Pg-enriched states. Both conditions show overlapping distributions — c_min is driven by total biomass, not species composition.

### Spatial Credible Bands

![Spatial credible bands](images/fig3_overlay_credible_bands.png)

Cross-section at y = Ly/2 with 90% CI. The two conditions' medians are nearly indistinguishable. Uncertainty is tightest near the tooth surface (x=0) and widens towards saliva (x=1).

### Parameter Sensitivity: θ vs c_min

![theta vs cmin](images/fig4_theta_vs_cmin.png)

Spearman rank correlations reveal **a₃₃ (V. dispar self-interaction) as the strongest driver** of c_min uncertainty (ρ ≈ −0.5 for both conditions). Pg coupling parameters a₃₅/a₄₅ have weaker influence (ρ ≈ 0.2–0.3).

### 2D Uncertainty Map

![2D uncertainty map](images/fig5_2d_uncertainty_map.png)

Top: Median nutrient c(x,y). Bottom: 90% CI width. Maximum uncertainty (~0.015) occurs at mid-depth (x ≈ 0.6–0.8), where the exponential biofilm profile amplifies parameter sensitivity.

### Summary

| Metric | Commensal | Dysbiotic |
|--------|:---------:|:---------:|
| c_min MAP | 0.028 | 0.028 |
| c_min median | 0.028 | 0.028 |
| c_min 90% CI | [0.028, 0.035] | [0.028, 0.035] |
| c_min std | 0.0027 | 0.0028 |
| Top sensitivity | a₃₃ (ρ=−0.46) | a₃₃ (ρ=−0.50) |

**Key insight**: In the 2D steady-state model, nutrient depletion is controlled by total biomass (dominated by V. dispar), not by dysbiotic species composition. Condition-specific differences appear in DI and Pg abundance, but not in c_min — validating that the multiscale DI→eigenstrain pathway (not c_min alone) is necessary for mechanical discrimination.

---

## a₃₅ Sweep Sensitivity (2026-02-24, Issue #6)

Sweep a₃₅ (Vd→Pg coupling) from 0 to 25 at the Dysbiotic Static MAP, keeping all other 19 parameters fixed. 51 points, T*=25, g_eff=50.

### 4-Panel Overview

![a35 sweep overview](images/fig1_a35_sweep_overview.png)

Top-left: Pg peaks at moderate a₃₅ ≈ 1 then slowly decreases (saturation). Top-right: c_min responds weakly (0.028–0.035 range). Bottom-left: DI shows a sharp transition zone at a₃₅ ≈ 1–4. Bottom-right: c_min vs Pg scatter colored by a₃₅ — reveals the full nonlinear relationship.

### Species Composition Phase Diagram

![Species stacked area](images/fig2_species_composition_stacked.png)

The most revealing figure. Three distinct regimes:
- **a₃₅ = 0–1**: V. dispar dominated (~80%), minimal Pg. Stable mono-culture.
- **a₃₅ ≈ 1–4**: **Critical transition zone** — dramatic species reorganization with oscillatory behavior (multiple equilibria/bifurcation). Vd drops, Fn/Pg surge.
- **a₃₅ > 5**: New stable equilibrium — F. nucleatum dominant (~50%), Pg moderate (~7–10%), gradual smooth evolution.

### Dual-Axis: Pg & c_min

![Dual axis](images/fig3_dual_axis_pg_cmin.png)

Pg and c_min track together (both controlled by total biomass change during the transition). The a₃₅ ≈ 1–4 bifurcation zone is clearly visible as oscillations in both quantities.

### Key Observations

| a₃₅ range | Regime | Pg | DI | c_min |
|:---------:|--------|:--:|:--:|:-----:|
| 0–1 | Vd-dominated | 0.01 | 0.85 | 0.028 |
| 1–4 | **Bifurcation** | 0.05–0.10 | 0.02–0.85 | 0.028–0.035 |
| 5–25 | Fn/Pg stable | 0.07–0.10 | 0.02–0.30 | 0.032–0.035 |

**Key insights**:
1. **a₃₅ does NOT monotonically increase Pg** — instead, it triggers a *community phase transition* at a₃₅ ≈ 1–4 where the entire species equilibrium reorganizes
2. **The bifurcation zone** (a₃₅ ≈ 1–4) shows multiple coexisting equilibria — a hallmark of ecological regime shifts
3. **c_min is robust** to a₃₅: only 20% relative variation across the full 0–25 range, confirming that nutrient depletion is not the primary mechanical pathway for dysbiosis
