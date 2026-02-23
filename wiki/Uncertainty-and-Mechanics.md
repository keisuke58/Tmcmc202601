# Uncertainty Quantification & Mechanics

Propagation of TMCMC posterior uncertainty through FEM stress analysis, material models, and fracture mechanics.

---

## Posterior Stress Uncertainty

Von Mises stress computed for 20 TMCMC posterior samples per condition, propagated through the full FEM pipeline.

### Stress Violin Distributions

![Stress violin](images/stress_violin.png)

Violin plots showing stress distributions across conditions. Median and 5th–95th CI annotated. dh-baseline shows the widest spread (1.58x) due to unconstrained bridge parameters.

### Stress CI Bars

![Stress CI bars](images/stress_ci_bars.png)

95% credible intervals for substrate and surface von Mises stress per condition. Dysbiotic conditions show higher median stress with wider uncertainty.

### Sensitivity Heatmap

![Sensitivity heatmap](images/sensitivity_heatmap.png)

Spearman rank correlation (|ρ|) between each model parameter and stress output. Bridge parameters (a₃₅, a₄₅) dominate the sensitivity for dysbiotic conditions.

### Top Parameter Scatter

![Top params scatter](images/top_params_scatter.png)

Scatter plots of the most influential parameters vs. stress output, colored by condition. Clear monotonic trends for bridge interaction strengths.

### Summary Panel

![Stress summary](images/stress_summary_panel.png)

6-panel summary: box plots, sensitivity, CI bars, and relative change vs. commensal static reference.

---

## Anisotropic Material Fields

Spatially varying elastic anisotropy derived from biofilm orientation heterogeneity.

### Cross-Condition Anisotropy Comparison

![Aniso cross condition](images/aniso_cross_condition.png)

Anisotropy ratio depth profiles across all 4 conditions. Higher anisotropy near the tooth surface where species gradients are steepest.

### Anisotropy Field — Dysbiotic Baseline

![Aniso field DH](images/aniso_field_dh_baseline.png)

Spatial anisotropy eigenstrain field for the dysbiotic baseline condition.

---

## Cohesive Zone Model (CZM) — Debonding Fracture

Traction-separation laws for biofilm-substrate interface, parameterized by DI and Pg concentration.

### CZM 4-Panel Comparison

![CZM 4panel](images/czm_4panel.png)

Traction-separation response under all 4 conditions. Critical traction and fracture energy decrease with increasing DI.

### CZM Summary

![CZM summary](images/czm_summary.png)

Enhanced summary showing debonding energy, peak traction, and critical separation across conditions.

### CZM Posterior Bands

![CZM posterior](images/czm_posterior_bands.png)

Traction-separation curves with posterior credible intervals. Dysbiotic conditions show wide uncertainty in fracture properties due to bridge parameter variability.

---

## Nonlinear Geometry Effects

Large deformation analysis (NLGEOM) vs. linear assumption for biofilm eigenstrain loading.

### Linear vs. Nonlinear Comparison

![NLGEOM comparison](images/nlgeom_comparison.png)

Side-by-side displacement and stress fields. Nonlinear geometry effects become significant when eigenstrain > 5%.

### Enhanced Combined Analysis

![NLGEOM enhanced](images/nlgeom_enhanced_combined.png)

Multi-panel comparison of displacement, stress, and strain fields across conditions. DI–strain scatter shows the coupling between microbial state and deformation.

---

## FEM Benchmarks

Solver validation and discretization studies.

### Split Scheme Performance

![Benchmark split](images/benchmark_split_scheme.png)

Operator splitting vs. monolithic solution accuracy. Split scheme reduces cost ~3x with < 1% error for typical biofilm stiffness contrast.

### Anisotropic Material Benchmark

![Benchmark aniso](images/benchmark_anisotropic.png)

Convergence study for anisotropic elastic tensor implementation. Quadratic elements achieve engineering accuracy at ~5000 DOFs.

---

## Nutrient PDE — Klempt 2024 Validation

Reaction-diffusion steady state for nutrient consumption within biofilm geometry.

### Nutrient Field (Egg-Shaped Biofilm)

![Nutrient field](images/klempt_nutrient_field.png)

Steady-state nutrient concentration c(x,y) with egg-shaped biofilm geometry (Klempt 2024 Fig. 1). c = 1 on boundary, c_min = 0.31 inside biofilm at g_eff = 50.

### Thiele Modulus Curve

![Thiele curve](images/klempt_thiele_curve.png)

Effectiveness factor vs. Thiele modulus. Transition from reaction-limited (Φ < 1) to diffusion-limited (Φ > 3) regimes.

### Nutrient c_min Distribution (Posterior)

![c_min distribution](images/nutrient_cmin_distribution.png)

Distribution of minimum nutrient concentration from TMCMC posterior samples. Reflects uncertainty in biofilm consumption rate.

### Spatial Credible Bands

![Nutrient spatial CI](images/nutrient_spatial_credible.png)

Nutrient cross-section with 90% credible interval from posterior propagation.

### 2D Uncertainty Map

![Nutrient 2D uncertainty](images/nutrient_2d_uncertainty.png)

Spatial map of nutrient uncertainty (std/mean). Highest uncertainty inside the biofilm where consumption is strongest.

### Hamilton-RD 2D: Pg Comparison

![2D Pg comparison](images/hamilton_rd_2d_pg_comparison.png)

*P. gingivalis* 2D field comparison between commensal and dysbiotic conditions from the coupled Hamilton-RD pipeline.

---

## Multiscale Coupling

### Hybrid DI Comparison

![Hybrid DI](images/hybrid_di_comparison.png)

0D DI vs. Hybrid (0D x 1D) DI scaling. The hybrid approach preserves condition discrimination while adding spatial resolution.

---

## Overview Diagrams

### Full Analysis Pipeline

![Pipeline](images/overview_pipeline.png)

End-to-end pipeline: experimental data → TMCMC → Hamilton ODE → 1D/2D spatial → 3D FEM.

### Interaction Network

![Network](images/overview_network.png)

5-species interaction topology.

### TMCMC Algorithm

![TMCMC](images/overview_tmcmc.png)

Transitional Markov Chain Monte Carlo schematic: prior → tempering stages → posterior.

### Data & Model

![Data model](images/overview_data_model.png)

Data sources, ODE model structure, and parameter mapping.

### Results Summary

![Results](images/overview_results.png)

Graphical abstract summarizing key findings across all analyses.

---

## Model Verification

### 5-Species ODE Dynamics

![5species dynamics](images/5species_dynamics.png)

Hamilton 5-species ODE time series at reference parameters. All species reach steady state; Pg growth depends on bridge interactions.

### Phase Plane Projections

![Phase planes](images/verify_phase_planes.png)

Pairwise phase portraits showing species trajectories. The Pg–Fn and Pg–Vd planes reveal the bridge-mediated coupling.

---

> See [Results Gallery](Results-Gallery) for the overview.
> See [TMCMC Per-Condition](TMCMC-Per-Condition) for detailed posterior results.
> See [FEM Spatial Analysis](FEM-Spatial-Analysis) for spatial dynamics.
