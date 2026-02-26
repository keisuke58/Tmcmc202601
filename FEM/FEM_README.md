# Biofilm FEM Pipeline — Reference Documentation

**Updated**: 2026-02-21
**Author**: Nishioka (Tmcmc202601 project)

---

## Overview

This directory extends the **5-species Hamilton biofilm model** (TMCMC-estimated,
`Tmcmc202601/data_5species/`) into a full **3D FEM stress analysis pipeline**.

```
TMCMC estimation (0-D)         →  θ_MAP  (20 parameters per condition)
          ↓
3D FEM reaction–diffusion       →  φᵢ(x,y,z)  species fields
          ↓
Dysbiotic Index (DI) field     →  DI(x,y,z) = 1 − H/log5
          ↓
DI → E(DI) mapping             →  E(x,y,z) = Emax(1−r)ⁿ + Emin·r
          ↓
Abaqus FEM stress analysis     →  S_Mises, U, RF  (substrate / surface)
```

---

## Directory Structure

```
FEM/
├── ── Source scripts ──────────────────────────────────────────────────────
│   abaqus_biofilm_aniso_3d.py     C1: Abaqus transversely isotropic model
│   abaqus_biofilm_cohesive_3d.py  B3: Abaqus 3D cohesive zone model
│   abaqus_biofilm_cohesive.py     (legacy) 2D CZM base
│   abaqus_biofilm_demo_3d.py      A: 3D isotropic stress model
│   abaqus_biofilm_demo.py         A: 2D base model
│   abaqus_biofilm_thread.py       Threaded Abaqus runner (legacy)
│   compare_biofilm_abaqus.py      ODB stress extractor (abaqus python)
│   aggregate_di_credible.py       B1: DI field credible interval
│   fem_aniso_analysis.py          C1: ∇φ_Pg gradient / aniso direction
│   run_material_sensitivity_sweep.py  A1+A2+A3 sweep runner
│   run_czm3d_sweep.py             B3 runner
│   run_aniso_comparison.py        C1 runner
│   run_posterior_abaqus_ensemble.py   Posterior ensemble Abaqus
│   run_posterior_pipeline.py      End-to-end pipeline
│   export_for_abaqus.py           CSV export for Abaqus field import
│   fem_*.py                       FEM computation (1D/2D/3D, Lie splitting)
│   posterior_sensitivity*.py      Sensitivity / stress uncertainty
│   plot_*.py                      Visualization helpers
│   analyze_abaqus_profiles.py     Depth-profile analysis
│   usdfld_biofilm.f               Fortran USDFLD (DI → E mapping)
│
├── ── Output directories (prefix _) ───────────────────────────────────────
│   _posterior_abaqus/      Posterior ensemble: 20 samples × 4 conditions
│   _di_credible/           B1: DI credible interval fields (p05/p50/p95)
│   _material_sweep/        A1+A2+A3: E_max/E_min/n sensitivity results
│   _aniso/                 C1: ∇φ_Pg gradient fields + aniso_summary.json
│   _aniso_sweep/           C1: Abaqus anisotropy sweep results
│   _results/, _results_3d/ FEM species field results
│   _posterior_plots/       Posterior ensemble figures
│   _posterior_sensitivity/ Sensitivity analysis outputs
│   _benchmarks/            FEM convergence / benchmark data
│
├── ── Archive ─────────────────────────────────────────────────────────────
│   _job_archive/           Abaqus job files (.odb, .inp, etc.) archived
│       aniso/              C1 anisotropy jobs
│       biofilm_demo/       A: isotropic demo jobs
│       di_credible/        B1: DI credible interval jobs
│       material_sweep/     A: material sensitivity jobs
│       old_field_csv/      Legacy field CSVs / PNGs
│
├── ── Tests ───────────────────────────────────────────────────────────────
│   tests/test_di_e_mapping.py   Unit tests for DI and E(DI) (pytest)
│
├── ── Reports ─────────────────────────────────────────────────────────────
│   FEM_README.md           This file
│   methods_supplement_fem.md  DI time point, E(DI) physics, TMCMC→Monod
│   fem_report.pdf/.tex     Earlier isotropic FEM report
│   abaqus_implementation_report.pdf/.tex  Abaqus scripting report
│   docs/
│       fem_pipeline.md     Extended pipeline documentation
│       fem_pipeline.tex    LaTeX version for publication
│       fem_pipeline.pdf    Compiled PDF
```

---

## 4 Conditions

| Key | Label | Description |
|-----|-------|-------------|
| `dh_baseline` | dh-baseline | dh θ (a₃₅=21.4), original model |
| `commensal_static` | Comm. Static | healthy θ, no HOBIC |
| `commensal_hobic` | Comm. HOBIC | healthy θ + HOBIC perturbation |
| `dysbiotic_static` | Dysb. Static | dysbiotic θ, no perturbation |

---

## Physics

### Dysbiotic Index (DI)

Shannon entropy-based measure of biofilm state:

```
H = -Σᵢ φᵢ/Φ · log(φᵢ/Φ)     (i = species present)
DI = 1 - H / log(5)            (0 = commensal, 1 = dysbiotic)
```

### E(DI) Power-Law Mapping

```
r(x)     = clamp(DI(x) / s,  0, 1)        s = 0.025778 (global scale)
E(DI)    = E_max · (1−r)^n + E_min · r    n = 2  (power-law exponent)
E_max    = 10.0 GPa   (commensal stiffness)
E_min    = 0.5  GPa   (dysbiotic stiffness)
ν        = 0.30
```

### Biofilm material model (substrate vs. biofilm modes)

- Substrate mode (`--mode substrate` in biofilm_conformal_tet.py)
  - GPa-scale effective stiffness of the biofilm-covered dental surface.
  - Material: linear elastic (engineering constants or isotropic), NLGEOM as needed.
- Biofilm mode (`--mode biofilm` in biofilm_conformal_tet.py)
  - Pa-scale EPS matrix (Billings 2015; Klempt 2024).
  - Default: linear elastic + NLGEOM (qualitative for large strains).
  - Optional: Neo-Hookean hyperelastic via `--neo-hookean`, using Abaqus built-in
    `*Hyperelastic, Neo Hooke` with parameters derived from E(DI) and ν.

---

## Pipeline Steps

### A. Material Sensitivity Sweep (A1 + A2 + A3)

**Script**: `run_material_sensitivity_sweep.py`
**Output**: `_material_sweep/results.csv`, `_material_sweep/figures/`

#### A1 — E_max × E_min Grid
4×4 grid: E_max ∈ {5, 10, 15, 20} GPa × E_min ∈ {0.1, 0.5, 1.0, 2.0} GPa
Fixed: n=2, θ = dh_old (a₃₅=21.4)

#### A2 — Power-Law Exponent Comparison
n ∈ {1, 2, 3}
Fixed: E_max=10 GPa, E_min=0.5 GPa, θ = dh_old

#### A3 — θ Variant Comparison

| Tag | Source | a₃₅ | a₄₅ |
|-----|--------|-----|-----|
| `mild_weight` | `_sweeps/K0.05_n4.0/theta_MAP.json` | **3.56** | 2.41 |
| `dh_old` | `data_5species/_runs/.../theta_MAP.json` | 21.4 | 3.97 |
| `nolambda` | `_sweeps/K0.05_n4.0_baseline/theta_MAP.json` | 20.9 | — |

**57 total Abaqus jobs** (3 FEM runs cached + 57 stress solves, ~10 min)

Key result: mild_weight θ (a₃₅=3.56) gives ~30% lower S_Mises at substrate
vs dh_old (a₃₅=21.4), confirming Pg suppression reduces mechanical risk.

---

### B1. DI Field Credible Interval

**Script**: `aggregate_di_credible.py`
**Output**: `_di_credible/{cond}/`

From 20 posterior samples per condition:
- Computes nodal DI quantiles: p05 / p50 / p95 at each of 3375 nodes
- Exports `p05_field.csv`, `p50_field.csv`, `p95_field.csv` for Abaqus
- Runs Abaqus on p05/p50/p95 → stress credible bands
- P.g center-of-mass depth profiles across samples

| Condition | DI_mean (p50) | S_Mises substrate (MPa) |
|-----------|--------------|------------------------|
| dh-baseline | ~0.015 | ~0.84 |
| Comm. Static | ~0.010 | ~0.86 |
| Comm. HOBIC | ~0.010 | ~0.85 |
| Dysb. Static | ~0.011 | ~0.86 |

---

### B3. 3D Cohesive Zone Model

**Script**: `run_czm3d_sweep.py`
**Abaqus script**: `abaqus_biofilm_cohesive_3d.py`
**Output**: `_czm3d/czm_results.csv`, `_czm3d/figures/`

Interface cohesive properties (DI-dependent):

```
t_max(DI) = t_max,0 · (1 − r)^n     t_max,0 = 1.0 MPa
G_c(DI)   = G_c,0  · (1 − r)^n     G_c,0   = 10.0 J/m²
```

Mixed-mode damage: Benzeggagh-Kenane (BK) law
Pull displacement: u_max = 5 mm, N_steps = 20

---

### C1. Transverse Isotropy (Anisotropy)

**Scripts**: `fem_aniso_analysis.py` → `run_aniso_comparison.py`
**Output**: `_aniso/`, `_aniso_sweep/`

#### Step 1 — Gradient Field (`fem_aniso_analysis.py`)

Loads `_di_credible/{cond}/phi_pg_stack.npy` (20 samples × 3375 nodes),
takes median, reshapes to 15×15×15 grid, computes ∇φ_Pg via `np.gradient`.

Dominant direction **e₁** = weighted mean of strongest gradients:

| Condition | e₁ | angle from x-axis |
|-----------|-----|------------------|
| dh-baseline | [-0.972, +0.211, -0.105] | **13.6°** |
| Comm. Static | [-0.956, +0.258, -0.142] | 17.1° |
| Comm. HOBIC | [-0.959, +0.245, -0.145] | 16.5° |
| Dysb. Static | [-0.952, +0.262, -0.160] | **17.9°** |

All conditions: dominant gradient is nearly along −x (depth toward substrate),
confirming P.g colonizes close to the tooth surface.

#### Step 2 — Abaqus Sweep (`run_aniso_comparison.py`)

Material model: **transversely isotropic**, stiff axis = e₁:

```
E₁(DI) = E(DI)             (stiff, along ∇φ_Pg)
E₂ = E₃ = β · E₁           (transverse, β = aniso_ratio)
ν₁₂ = ν₁₃ = ν₂₃ = 0.30
G₁₂ = G₁₃ = E₁/(2(1+ν))
G₂₃       = E₂/(2(1+ν))
```

Implemented via Abaqus `*ELASTIC, TYPE=ENGINEERING CONSTANTS`.

Sweep: **β ∈ {1.0, 0.7, 0.5, 0.3}** × 4 conditions = **16 jobs**
Loading: 1 MPa compressive pressure on top face, fixed bottom face.

#### C1 Results (1 MPa compression)

| Condition | β=1.0 sub | β=0.5 sub | Δ sub | β=1.0 surf | β=0.5 surf | Δ surf |
|-----------|-----------|-----------|-------|-----------|-----------|-------|
| dh-baseline | 0.839 MPa | 0.817 MPa | **−2.6%** | 0.979 MPa | 0.981 MPa | +0.2% |
| Comm. Static | 0.860 MPa | 0.849 MPa | −1.3% | 1.020 MPa | 1.020 MPa | 0.0% |
| Comm. HOBIC | 0.854 MPa | 0.843 MPa | −1.3% | 1.020 MPa | 1.020 MPa | 0.0% |
| Dysb. Static | 0.856 MPa | 0.849 MPa | −0.8% | 1.020 MPa | 1.020 MPa | 0.0% |

**Key findings**:
- Reducing β (more anisotropic) **decreases substrate S_Mises** by 1–3%
- Surface stress is largely **insensitive** to β (load-controlled BC dominates)
- dh-baseline shows the **largest anisotropy sensitivity** (steeper gradient angle)
- Effect is modest but consistent with a stiff-in-depth, soft-in-transverse biofilm

---

## Running the Pipeline

```bash
cd Tmcmc202601/FEM

# Step 0: FEM field + posterior ensemble (prerequisite)
python run_posterior_abaqus_ensemble.py

# Step B1: DI credible interval fields (requires Step 0)
python aggregate_di_credible.py

# Step A: Material sensitivity sweep (requires Step 0)
python run_material_sensitivity_sweep.py

# Step C1: Gradient analysis (requires B1)
python fem_aniso_analysis.py

# Step C1: Abaqus anisotropy sweep (requires B1 + C1 grad)
python run_aniso_comparison.py

# Step B3: CZM sweep (requires B1)
python run_czm3d_sweep.py

# Re-plot only (no Abaqus)
python run_aniso_comparison.py --plot-only
python run_material_sensitivity_sweep.py --plot-only
python run_czm3d_sweep.py --plot-only
```

---

## Key Parameters Summary

| Parameter | Value | Description |
|-----------|-------|-------------|
| `DI_SCALE` (s) | 0.025778 | Global DI normalization scale (primary displacement knob) |
| `E_MAX` | 10.0 GPa | Stiffness at DI=0 (commensal), fixed from literature |
| `E_MIN` | 0.5 GPa | Stiffness at DI=s (dysbiotic), fixed from literature |
| `DI_EXPONENT` (n) | 2.0 | Power-law exponent (primary displacement knob) |
| `NU` | 0.30 | Poisson's ratio |
| `N_BINS` | 20 | DI bins for material assignment |
| Grid size | 15×15×15 | FEM nodal grid (3375 nodes) |
| Pressure | 1.0 MPa | Applied compressive load |

---

## Environment

| Variable | Description |
|----------|-------------|
| `ABAQUS_CMD` | Path to Abaqus command (default: `/home/nishioka/.../abaqus`) |

Used by `tmcmc_to_fem_coupling.py` when invoking assembly. Set for your install:
```bash
export ABAQUS_CMD=/path/to/abaqus
```

## Abaqus API Notes

Abaqus Python (CAE) environment quirks:

| Issue | Fix |
|-------|-----|
| No generator in `math.sqrt(sum(...))` | Use explicit arithmetic |
| `mat.Elastic(type="STRING")` fails | Use constant: `ENGINEERING_CONSTANTS` |
| `Region(cells=[cell])` fails (needs GeomSequence) | Use `elements.sequenceFromLabels()` |
| `model.DatumCsysByThreePoints` → AttributeError | Use `part.DatumCsysByThreePoints` |
| `fieldOutputRequests["F-Output-1"]` on new model → AttributeError | Remove; use defaults |
| Material orientation: use `part.MaterialOrientation(orientationType=SYSTEM, axis=AXIS_1, ...)` | |

---

## References

- Wriggers & Junker (2024): *A Hamilton principle-based model for diffusion-driven biofilm growth*, CMAME
- Junker & Balzani (2021): *Hamilton model for biofilm mechanics*
- Abaqus Documentation: `*ELASTIC, TYPE=ENGINEERING CONSTANTS` (transverse isotropy)
