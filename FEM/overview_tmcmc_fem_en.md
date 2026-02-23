# TMCMC + FEM Integrated Overview

**One-line summary:**  
*From experimental time-series → Bayesian interaction parameters (TMCMC) → Dysbiotic Index field → 3-tooth conformal biofilm FEM → stress and displacement under masticatory load.*

---

## Part A — TMCMC: Parameter Estimation

### A.1 What TMCMC Does

- **Goal:** Quantify inter-species interactions in oral biofilm via Bayesian inference.
- **Input:** Experimental volume fractions — 6 time points × 5 species (So, An, Vd, Fn, Pg); constraint Σφᵢ = 1.
- **Model:** ODE dφ/dt = f(φ,ψ;θ) with 20 parameters (interaction matrix A, viability ψᵢ, decay bᵢ), Hill-function gating (Fn → Pg).
- **Inference:** TMCMC (β: 0→1, 8 stages, 150 particles); prior → posterior; MAP + 95% CI.
- **Output:** `parameter_summary.csv`, `posterior_samples.npy`, diagnostics (ESS, Rhat), figures.

### A.2 Conditions (4)

| Key | Description |
|-----|--------------|
| dh_baseline | Dysbiotic cascade (extreme Pg support; a₃₅ high) |
| commensal_static | Balanced commensal |
| dysbiotic_static | Moderate dysbiotic |
| commensal_hobic | Commensal with HOBIC dynamics |

### A.3 Flow (TMCMC only)

```
Experimental data [6×5] → Prior [20 params] → ODE model → Likelihood → TMCMC → MAP + 95% CI
                                    ↑_____________________________|
                                    (Numba JIT, TSM-ROM, parallel)
```

---

## Part B — FEM: 3-Tooth Biofilm Mechanics

### B.1 What the FEM Pipeline Does

- **Goal:** Compute von Mises stress and nodal displacement of conformal biofilm around three real teeth under inward masticatory pressure.
- **Geometry:** OpenJaw Patient 1 — T23 (crown), T30/T31 (slit / inter-proximal). STL → conformal C3D4 tet mesh (0.5 mm thickness, 8 layers) → 82,080 nodes, 437,472 elements.
- **Material:** DI (Dysbiotic Index) from 5-species composition → bins → E_eff(DI) = E_max(1−r)^α + E_min·r (r = clip(DI/s_DI, 0, 1)); E_max=10 MPa, E_min=0.5 MPa, α=2, s_DI from TMCMC MAP.
- **Assembly:** 3-tooth INP, Tie constraint at slit (T30↔T31), ENCASTRE on inner (tooth) face, Cload 1 MPa inward on outer face.
- **Solver:** Abaqus/Standard, Static/General, 4 CPUs; ODB → CSV extraction → 10+ figures (displacement, MISES, histograms, slit view, etc.).

### B.2 Data Flow (FEM)

```
STL (T23/30/31) → biofilm_conformal_tet.py → conformal mesh + DI bins
       → biofilm_3tooth_assembly.py → biofilm_3tooth.inp (26 MB)
       → Abaqus job=BioFilm3T → BioFilm3T.odb (181 MB)
       → odb_extract.py → odb_nodes.csv, odb_elements.csv
       → odb_visualize.py → Fig1–Fig12, CompFig1–4, LateFig1–7, etc.
```

### B.3 Key FEM Outputs (Report)

- **Baseline (DH-baseline):** MISES median T23/T30/T31 ≈ 0.52–0.55 MPa; |U|_outer median ≈ 6.9×10⁻⁵ mm.
- **Condition comparison (P0/P0b):** Four conditions; DI field and E_eff differ; late-time: DH-baseline DI ≈ 0.51 → E_eff = E_min = 0.5 MPa; commensal DI ≈ 0 → E_eff ≈ 9.85 MPa.
- **Late-time finding:** Under force control, MISES is identical across conditions; **displacement is ~19.7× larger for DH-baseline** (dysbiotic biofilm much softer).

---

## Part C — Coupling: TMCMC → FEM

### C.1 Link

- **tmcmc_to_fem_coupling.py:** Reads per-condition MAP parameters (from TMCMC runs), exports DI field from 3-D FD simulation (or snapshot), writes condition-specific field CSV and INP.
- **DI definition:** DI = 1 − H/H_max, H = −Σ p_i ln p_i, p_i = φᵢ/Σφⱼ (Shannon-based).
- **Result:** Same 3-tooth geometry and load; only the DI→E_eff field changes by condition → different displacement (and under displacement-controlled BC, different MISES).

### C.2 End-to-End Loop

```
Experiment [6×5] → TMCMC → θ_MAP (20 params) per condition
       → 3D FD / snapshot → φᵢ(x,y,z) → DI(x,y,z)
       → E_eff(DI) → per-condition INP → Abaqus → MISES, U
       → Compare conditions (e.g. DH-baseline vs Commensal-static)
```

### C.3 Takeaway (Integrated)

- **What:** Bayesian 5-species interaction parameters (TMCMC) + 3-tooth conformal biofilm FEM (Abaqus).
- **How:** ODE×TMCMC → MAP; MAP → DI field → E_eff → INP → stress/displacement; condition comparison (4 conditions).
- **Why it matters:** Late-time dysbiotic (Pg-dominated) biofilm is ~19.7× softer → ~19.7× larger displacement under 1 MPa; displacement is the biomechanically relevant biomarker under force control.

---

## One-Page Summary (TMCMC + FEM)

| Stage | Input | Process | Output |
|-------|--------|---------|--------|
| **TMCMC** | Exp. data [6×5] | ODE + TMCMC (8 st., 150 pt.) | MAP + 95% CI, 20 params |
| **Coupling** | θ_MAP, snapshot | DI = 1−H/log5, E_eff(DI) | Field CSV, INP per condition |
| **FEM** | INP (3 teeth, Tie, Cload) | Abaqus Static | ODB → MISES, U, figures |

**Key numbers:** 82,080 nodes, 437,472 C3D4; 1 MPa inward; DH-baseline vs commensal displacement ratio ≈ **19.7×** at late time.

---

## References (in-report)

- biofilm_3tooth_report.tex / biofilm_3tooth_report.pdf — full pipeline, equations, P0/P0b, sensitivity, BC comparison.
- FEM_README.md — directory layout, conditions, physics (DI, E_eff), pipeline steps.
- overview2602_en.md — TMCMC-only overview.
