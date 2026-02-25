# Paper Outline: Multiscale Bayesian-Mechanical Framework for Oral Biofilm Dysbiosis

## Working Title
"Bridging microbial ecology and continuum mechanics: A TMCMC-based multiscale framework for oral biofilm stress analysis"

---

## Fig 1–7: 先行研究・既存図（Nishioka/Hamilton/Klempt等で既出）

---

## Fig 8–15: 本研究の新規 Figure

| Fig | Title | Script | Status |
|-----|-------|--------|--------|
| **Fig 8** | TMCMC posterior: 20 param distributions (CS vs DH) | `generate_paper_figures.py` | DONE |
| **Fig 9** | 2D Hamilton+Nutrient: species + nutrient field | `generate_paper_figures.py` | DONE |
| **Fig 10** | DI spatial field: commensal vs dysbiotic | `generate_paper_figures.py` | DONE |
| **Fig 11** | E(DI) material model + literature data overlay | `generate_paper_figures.py` | DONE → **UPDATE: add Pattem 2018/2021 data** |
| **Fig 12** | 3D conformal mesh + tooth assembly | `generate_paper_figures.py` | DONE |
| **Fig 13** | von Mises / displacement comparison | `generate_paper_figures.py` | DONE |
| **Fig 14** | Klempt 2024 nutrient benchmark | `generate_paper_figures.py` | DONE |
| **Fig 15** | Posterior → DI → E uncertainty propagation | `generate_paper_figures.py` | DONE |
| **Fig 16** | **NEW: Basin sensitivity / multi-attractor** | `plot_basin_sensitivity.py` | DONE |
| **Fig 17** | **NEW: 3-model comparison (DI vs φ_Pg vs Vir)** | existing `3model_comparison_3d.png` | EXISTS |
| **Fig 18** | 3-model Bayes discrimination | `compute_3model_bayes_factor.py` | DONE |
| **Fig 19** | Corner plot dh_baseline key params | `generate_corner_plot_paper.py` | DONE |
| **Fig 20** | **E2E differentiable pipeline: 4-cond results** | `e2e_differentiable_pipeline.py` | DONE |
| **Fig 21** | **NUTS vs HMC vs RW TMCMC comparison** | `generate_fig21_paper.py` | DONE |
| **Fig 22** | **DeepONet vs ODE posterior comparison** | `generate_fig22_posterior_comparison.py` | DONE |

---

## Section Structure

### 1. Introduction
- Oral biofilm = structured multispecies community
- Dysbiosis → periodontitis → mechanical consequences (detachment, stress)
- Gap: no quantitative link from microbial ecology → continuum mechanics
- Contribution: TMCMC + Hamilton ODE + DI material model + 3D FEM

### 2. Methods

#### 2.1 Hamilton 5-species ODE
- 5 species (So, An, Vd, Fn, Pg), 20 params, Hill gate for Pg
- 4 experimental conditions (CS, CH, DS, DH)
- Reference: Hamilton et al.

#### 2.2 TMCMC Bayesian Inference
- Transitional MCMC, uniform prior, 1000 particles
- Mild-weight settings: K=0.05, n=4, λ_Pg=2.0, λ_late=1.5
- a35∈[0,5], a45∈[0,5] (biologically constrained)
- **Fig 8**: Posterior distributions

#### 2.3 Multiscale Coupling Pipeline
- **0D → DI_0D**: condition-level diversity index (18× range)
- **1D → α_Monod(x)**: nutrient-dependent growth eigenstrain
- **2D → DI_2D(x,y)**: spatial heterogeneity via Hamilton+nutrient PDE
- **Hybrid**: DI(x,y) = DI_0D × [DI_2D(x,y) / mean(DI_2D)]
- **Fig 9**: Species + nutrient field
- **Fig 10**: DI spatial distribution

#### 2.4 DI-Based Material Model
- E(DI) = E_max(1-r)² + E_min·r, r = DI/DI_scale
- **Literature justification**: Pattem 2018 (10-80× range), Pattem 2021, Gloag 2019
  - E_commensal ≈ 900 Pa (diverse, structured ECM)
  - E_dysbiotic ≈ 30 Pa (So-dominated, weak ECM)
  - 30× ratio consistent with experiments
- Comparison: φ_Pg model, Virulence model → fail to discriminate (φ_Pg < 0.03 all conditions)
- **Fig 11**: E(DI) curve + experimental overlay
- **Fig 17**: 3-model comparison

#### 2.5 E2E Differentiable Pipeline (DeepONet + DEM)
- θ → DeepONet → φ(T;θ) → DI → E(DI) → DEM → u(x,y,z)
- All steps JAX-differentiable → ∂u/∂θ via single backward pass
- DeepONet surrogate: 160K params, trained per-condition (10K samples)
- DEM (Deep Energy Method): 58K params, variational elasticity solver
- Forward: 0.03 ms, Gradient (20 params): 0.04 ms → **3.8M× speedup vs Abaqus**
- **Fig 20**: 4-condition pipeline results (φ, DI, E, u_y profile, ∂u_y/∂θ)

#### 2.6 Gradient-Based TMCMC (HMC / NUTS)
- Standard TMCMC uses random-walk Metropolis → 50% accept in 20D
- Differentiable pipeline enables HMC mutation: leapfrog along ∂logL/∂θ
- **NUTS extension**: automatic trajectory length via U-turn criterion
- Dual averaging for step-size adaptation (Hoffman & Gelman 2014)
- **Fig 21**: RW vs HMC vs NUTS comparison

#### 2.7 3D FEM Setup
- Patient T23 tooth, dentin E=18.6 GPa
- Conformal C3D4 mesh, 17,970 nodes
- Tie constraint (tooth-biofilm interface)
- Eigenstrain loading (thermal analogy)
- **Fig 12**: Mesh assembly

### 3. Results

#### 3.1 TMCMC Posterior
- MAP RMSE: 0.054–0.075 across conditions
- Bridge params: a35=3.56, a45=2.41 (mild-weight) vs a35=17.3 (original)
- 30% total RMSE improvement vs baseline
- **Table 1**: Per-species RMSE comparison

#### 3.2 Multiscale DI Fields
- 0D DI: CS=0.42, CH=0.84, DH=0.16, DS=0.85
- α_Monod(x): 101× spatial gradient (tooth→saliva)
- 2D DI: nutrient-dependent heterogeneity inside egg-shaped biofilm
- **Fig 10**: DI comparison

#### 3.3 3D Stress Analysis
- E range: 31–908 Pa (DI model)
- U_max range: 0.44–12.9 mm (29.4× ratio)
- von Mises: dentin-dominated (identical across conditions)
- **Fig 13**: Stress comparison
- **Fig 14**: Klempt benchmark

#### 3.4 E2E Pipeline & Sensitivity Analysis
- Forward pass: 0.03 ms (Abaqus: ~120 s) → 3.8M× speedup
- Exact ∂u_y/∂θ identifies most sensitive parameters:
  - θ[16] (a₂₅, Vd→Pg): strongest mechanical sensitivity
  - θ[15] (a₁₅, So→Pg): cross-feeding to pathogen
- These control pathogen cross-species feeding → direct mechanical consequence
- **Fig 20**: Pipeline results

#### 3.5 Gradient-Based TMCMC Results (4-condition real data, 200 particles)
- Acceptance improvement scales with dimensionality:
  - DH (20 free): RW 0.45 → HMC 0.99 → NUTS 0.97 (**2.2× vs RW**)
  - DS (15 free): RW 0.49 → HMC 0.97 → NUTS 0.97 (**2.0× vs RW**)
  - CH (13 free): RW 0.54 → HMC 0.92 → NUTS 0.94 (**1.7× vs RW**)
  - CS (9 free): RW 0.65 → HMC 0.70 → NUTS 0.84 (**1.3× vs RW**)
- Average across conditions: RW 0.53 → NUTS 0.93 (**1.75×**)
- NUTS auto-adapts step size via dual averaging (ε: 0.015→0.10)
- Same tempering schedule for all 3 methods — beta schedule is data-driven
- NUTS finds better max logL: DH -7.1 (NUTS) vs -7.9 (RW)
- **Fig 21**: Per-condition RW/HMC/NUTS comparison
- **Table**: Summary metrics (acceptance, logL, wall time)

#### 3.6 DeepONet Surrogate TMCMC (4-condition, 500 particles)
- DeepONet replaces ODE solver inside TMCMC log-likelihood
- JAX+fork issue solved via ThreadPoolExecutor (`--use-threads`)
- **~100× wall-time speedup**: ODE ~1800s → DeepONet 12-18s per condition
- All 4 conditions converged: R-hat max 1.010, ESS min 692
- DH posterior comparison (300 ODE vs 1000 DeepONet samples):
  - 17/20 params: Bhattacharyya overlap > 0.95 (excellent)
  - 2 params low overlap: a₃₂ (0.42), a₃₅ (0.23) — Pg cross-feeding with wide ODE posterior
  - Mean overlap: 0.903
- Limitation: DeepONet MAP error (CS 62%, CH 44%) → Pg-related params poorly resolved
- **Fig 22**: Posterior comparison (marginals + speedup + MAP comparison)

#### 3.7 Posterior Uncertainty Propagation
- dh_baseline (real posterior): DI CI = [0.22, 0.89], E CI = [21, 609] Pa
- commensal_static: basin fragility (49/51 samples jump to DI≈0.85)
- Multi-attractor sensitivity = key UQ finding
- **Fig 15**: CI bands
- **Fig 16**: Basin sensitivity

### 4. Discussion

#### 4.1 DI as Mechanical Indicator
- Shannon entropy captures ecosystem-level dysbiosis
- φ_Pg fails: Pg fraction too low in Hamilton model (Hill gate + locked edges)
- Dysbiotic ≠ Pg-dominated; Dysbiotic = diversity loss (So-dominated)
- Clinical: DI links to biofilm detachability

#### 4.2 Multi-Attractor Sensitivity
- ODE has multiple stable equilibria
- Small parameter perturbation → basin switching
- commensal_static is at basin boundary → fragile prediction
- Implication: uncertainty quantification must include attractor landscape

#### 4.3 Literature Comparison
- E_bio values consistent with Pattem 2018/2021 AFM data
- Thiele modulus consistent with Klempt 2024
- No prior work linking DI → mechanical properties quantitatively

#### 4.4 Differentiable Surrogate & Gradient Sampling
- DeepONet+DEM surrogate enables gradient-based sampling impossible with Abaqus
- NUTS automatically adapts to posterior geometry (no tuning needed)
- 3.8M× speedup makes probabilistic analysis feasible
- Limitation: surrogate accuracy bounded by training data coverage

#### 4.5 Limitations
- E(DI) mapping assumed (not experimentally calibrated for this system)
- ODE TMCMC: 150 particles × 2 chains → 300 samples (1000p runs pending)
- Linear elastic FEM (nonlinear needed for ε > 5%)
- 2D PDE homogenizes species (mitigated by Hybrid approach)
- DeepONet MAP accuracy varies: DH 11%, DS 52%, CS 62%, CH 44% → Pg params poorly resolved

### 5. Conclusion
- First quantitative pipeline: experimental data → TMCMC → DI → 3D FEM stress
- DI is the correct mechanical indicator (not φ_Pg)
- 30× stiffness range → dysbiotic biofilm mechanically vulnerable
- Basin sensitivity = important UQ consideration for ODE-based models

---

## Key Numbers for Abstract
- 5 species, 20 parameters, 4 conditions
- 30% RMSE improvement (mild-weight vs original)
- 28× stiffness range (commensal 900 Pa vs dysbiotic 30 Pa)
- 29.4× displacement ratio
- Multi-attractor sensitivity: DI CI spans [0.22, 0.89] for dh_baseline
- 3.8M× speedup via DeepONet+DEM differentiable pipeline
- Gradient TMCMC acceptance 1.75× vs random-walk (avg 0.93 vs 0.53, 4-condition real data)
- Improvement scales with dimensionality: 1.3× (d=9) → 2.2× (d=20)
- NUTS dual averaging auto-tunes step size (no manual tuning)
- DeepONet TMCMC: ~100× speedup (12-18s vs ~1800s), 17/20 params overlap > 0.95
- Full posterior recovery for growth parameters; Pg cross-feeding needs higher accuracy

---

## Literature for E(DI) Justification

1. **Pattem et al. (2018)** Sci Rep. PMC5890245. Oral biofilm AFM: low-sucrose 14 kPa vs high-sucrose 0.55 kPa (10-80× ratio)
2. **Pattem et al. (2021)** Sci Rep. PMC8355335. Hydrated: LC 10.4 kPa vs HC 2.8 kPa (3.7×)
3. **Gloag et al. (2019)** J Bacteriol. PMC6707914. Dual-species G'=160 Pa
4. **Gloag et al. (2020)** PMC7798440. Biofilm mechanics review: Pa–kPa consensus
5. **npj Biofilms (2018)** doi:10.1038/s41522-018-0062-5. Standardization review
6. **Southampton thesis** eprints.soton.ac.uk/359755/. S. mutans 380 Pa
7. **Aravas & Picu (2008)** Biotech Bioeng. PubMed 18383138. FEM constitutive model
