# FEM Pipeline Plan — 3-Tooth Biofilm Assembly
> Last updated: 2026-02-23 (P0b/P3/P7 complete; late-time results added)

---

## 1. Pipeline Overview

```
STL (crown/slit geometry)
        ↓
biofilm_conformal_tet.py   ← conformal C3D4 tet mesh, DI-bin material assignment
        ↓
biofilm_3tooth_assembly.py ← 3-tooth assembly (T23 crown + T30/T31 slit),
                              approximal Tie constraint, Cload
        ↓
biofilm_3tooth.inp         ← Abaqus INP file
        ↓
abaqus job=BioFilm3T       ← FEM solve
        ↓
BioFilm3T.odb              ← 181 MB result
        ↓
abaqus python odb_extract.py   ← CSV extraction (odb_nodes.csv, odb_elements.csv)
        ↓
python3 odb_visualize.py   ← 12-figure PNG visualization
        ↓
figures/Fig1–Fig12
```

**Unit system (Abaqus):** mm / N / MPa (= N mm⁻²)

---

## 2. Key Files

| File | Role |
|------|------|
| `biofilm_conformal_tet.py` | STL → C3D4 mesh; E in **MPa** (÷1e6 from Pa) |
| `biofilm_3tooth_assembly.py` | 3-tooth INP writer; approximal Tie; Cload |
| `odb_extract.py` | Abaqus-Python ODB → `odb_nodes.csv`, `odb_elements.csv` |
| `odb_visualize.py` | Standard Python, **12-figure** visualization |
| `tmcmc_to_fem_coupling.py` | **[NEW]** TMCMC MAP → DI field CSV → per-condition INP |
| `biofilm_3tooth.inp` | Generated INP (do not hand-edit) |
| `BioFilm3T.odb` | FEM result (181 MB, pressure = 1 MPa) |
| `figures/` | Fig1–Fig12 PNG outputs |
| `biofilm_3tooth_report.tex/.pdf` | **[NEW]** 15-page LaTeX technical report |
| `abaqus_field_dh_3d.csv` | DI field for current run (dh_baseline, snapshot 20) |
| `abaqus_field_dh_baseline_snap20.csv` | DI field exported by coupling script |
| `p0b_long_sim_runner.py` | **[NEW]** Full FD+FEM pipeline for 4 conditions, late-time (n_macro=1000) |
| `compare_conditions_late.py` | **[NEW]** Late-time (t≈0.5) 4-condition FEM comparison (LateFig1–7) |
| `compare_conditions_fem.py` | **[NEW]** Snap20 4-condition FEM comparison (CompFig1–4) |
| `p3_pressure_sweep.py` | **[NEW]** Pressure parameter sweep 0.01–10 MPa |
| `p4_mesh_convergence.py` | **[NEW]** Mesh convergence study (N_layers=4,8,16) |
| `p7_tie_diagnostic.py` | **[NEW]** Tie gap distribution diagnostic |
| `abaqus_field_{cond}_late.csv` | Late-time DI field CSVs for 4 conditions |
| `odb_elements_{cond}_late.csv` | Late-time ODB element CSVs for 4 conditions |

---

## 3. Current Status (as of 2026-02-23, post-session)

### 3a. Mesh  ✅ COMPLETE
- **82,080 nodes**, **437,472 C3D4 elements** (T23: 86,160 | T30: 190,272 | T31: 161,040)
- Conformal prism → tet split; inner surface = exact STL surface
- DI-bin assignment: `n_bins=20`, 7 active bins (3–11)
- DI source: `abaqus_field_dh_3d.csv` (dh_baseline MAP, snapshot 20, t=0.05)

### 3b. Material  ✅ COMPLETE
- `E_eff = E_max·(1−r)^α + E_min·r`
- E_max = 10 MPa, E_min = 0.5 MPa, α = 2.0, ν = 0.30
- DI scale = 0.025778 (from entropy-based dysbiotic index formula)
- Bug fixed: Pa → MPa conversion (÷1e6)

### 3c. Crown (T23)  ✅ COMPLETE
- MISES: min=0.249 / median=**0.546** / max=0.856 MPa
- |U| outer median: **6.91×10⁻⁵ mm** (under 1 MPa inward pressure)

### 3d. Slit (T30 + T31)  ✅ COMPLETE
- `*Tie name=SLIT_TIE, adjust=NO` (303 zero-vol elements avoided)
- 110 nodes tied within 0.5 mm; 1168 outside tolerance (accepted — gap mean 4.6 mm)
- APPROX nodes: T30=1,614 / T31=1,278; excluded from Cload
- MISES median: T30=**0.515** MPa, T31=**0.522** MPa
- |U| APPROX ≈ **1.17×10⁻⁶ mm** (inter-proximal coupling visible)

### 3e. Figures (Fig1–12)  ✅ COMPLETE

| Fig | Content | Status |
|-----|---------|--------|
| Fig1 | 3D node scatter, colored by \|U\| | ✅ |
| Fig2 | 3D element centroid scatter, colored by MISES | ✅ |
| Fig3 | MISES histogram per tooth | ✅ |
| Fig4 | MISES vs DI bin radial profile | ✅ |
| Fig5 | \|U\| boxplot INNER vs OUTER per tooth | ✅ |
| Fig6 | MISES boxplot per tooth | ✅ |
| Fig7 | Numeric summary table | ✅ |
| Fig8 | 3D slit view: T30+T31, APPROX nodes in orange | ✅ |
| Fig9 | APPROX vs OUTER displacement boxplot+violin (T30, T31) | ✅ |
| Fig10 | Crown vs Slit summary (MISES + displacement by surface type) | ✅ |
| **Fig11** | **[NEW P6]** MISES cross-sections at y=const planes (4 cuts) | ✅ |
| **Fig12** | **[NEW P6]** Strain energy density w≈σ²/(3E_eff) | ✅ |

### 3f. TMCMC → FEM Coupling  ✅ SCRIPT READY
- `tmcmc_to_fem_coupling.py` reads MAP parameters from TMCMC sweep runs
- Computes DI via entropy: `DI = 1 − H/H_max`, H_max = ln(5)
- Exports DI field CSV from `_results_3d/{condition}/snapshots_phi.npy`
- Tested: `dh_baseline`, snap=20, t=0.05 → 3375 points, mean DI=0.0070
- dh_baseline MAP: **a35=21.41**, **a45=2.50**, **a55=0.12** (extreme dysbiotic cascade)

### 3g. LaTeX Documentation  ✅ COMPLETE
- `biofilm_3tooth_report.tex` → `biofilm_3tooth_report.pdf` (15 pages)
- Covers: pipeline, mesh equations, material model, results tables, all 12 figures,
  known issues, next actions, appendices

### 3h. Late-time FEM Comparison (P0b)  ✅ COMPLETE (2026-02-23)
- **Script:** `p0b_long_sim_runner.py` (n_macro=1000, t≈0.5)
- **4 conditions:** dh_baseline, commensal_static, Dysbiotic_Static, Commensal_HOBIC
- **FD simulation:** 101 snapshots, ~540 s per condition

**Late-time DI (t≈0.5):**

| Condition | DI_mean | E_eff (MPa) | |U| T23 median |
|-----------|---------|-------------|----------------|
| dh_baseline | **0.5135** | **0.50** (clamped at E_MIN) | ~0.358 µm |
| commensal_static | 0.0002 | 9.85 | ~0.018 µm |
| Dysbiotic_Static | 0.0005 | 9.78 | ~0.018 µm |
| Commensal_HOBIC | 0.0003 | 9.87 | ~0.018 µm |

- **Key finding — MISES:** identical across all conditions (force-controlled BC → geometry-dominated stress)
- **Key finding — Displacement:** dh_baseline ~**19.7×** larger (E_eff ratio 0.50/9.85 MPa)
- **Figures:** LateFig1 (MISES violin), LateFig2 (DI comparison), LateFig3 (E_eff), LateFig4 (ΔMISES), LateFig5 (summary table), LateFig6 (displacement violin), LateFig7 (displacement ratio bar)

### 3i. Pressure Sweep (P3)  ✅ COMPLETE (2026-02-23)
- **Script:** `p3_pressure_sweep.py`
- **Pressures:** 0.01, 0.1, 0.5, 1.0, 5.0, 10.0 MPa (all completed)
- At 10 MPa: MISES median=5.2880 MPa, max=17.6020 MPa
- **Result:** linear elastic behaviour confirmed across full range
- **Outputs:** `_pressure_sweep/pressure_sweep_results.json`, `_pressure_sweep/P3_pressure_sweep.png`

### 3j. Tie Diagnostic (P7)  ✅ COMPLETE (2026-02-23)
- **Script:** `p7_tie_diagnostic.py`
- T30 APPROX: 1,614 nodes | T31 APPROX: 1,278 nodes
- Only **8.6%** of T31 APPROX nodes within 0.5 mm of T30 APPROX
- **Recommendation:** `--slit-max-dist` 0.4–0.6 mm
- **Interpretation:** 0.5 mm Abaqus Tie tolerance is appropriate for the narrow proximal slit; geometry is correct
- **Figures:** `figures/P7_tie_diagnostic.png`, `figures/P7_tie_3d_gap.png`

---

## 4. Known Issues / Warnings

| Issue | Status |
|-------|--------|
| 1168 slave nodes outside 0.5 mm tie tolerance | Accepted — proximal pocket only needs 110 |
| 2 unconnected regions warning before Tie | Resolved by Tie; benign |
| DI from TMCMC snapshot (not real clinical data) | **Pending** — Next Action P1 |
| Pressure = 1 MPa (single value, not patient-specific) | **Pending** — Next Action P3 |
| T31 MISES peak 1.76 MPa (slit corner stress concentration) | Noted — expected near approximal edge |

---

## 5. Next Actions (Prioritized)

### [P0] ✅ COMPLETE (2026-02-22) Condition Comparison (snap20)

- 4 conditions compared at snapshot 20 (t=0.05): dh_baseline, commensal_static, Dysbiotic_Static, Commensal_HOBIC
- Script: `compare_conditions_fem.py`; figures: CompFig1–4 in `figures/`
- **Key finding (snap20):** DH-baseline DI=0.0070 (lowest at early time — Pg cascade not yet developed)
- **Snap20 E_eff:** dh_baseline stiffest (median 5.55 MPa); Commensal-HOBIC softest (4.13 MPa)

### [P0b] ✅ COMPLETE (2026-02-23) Late-snapshot Condition Comparison

- Script: `p0b_long_sim_runner.py` (n_macro=1000, t≈0.5, fully developed Pg cascade)
- 4 conditions: dh_baseline, commensal_static, Dysbiotic_Static, Commensal_HOBIC

| Condition | DI_mean (late) | E_eff (MPa) | |U| T23 median |
|-----------|---------------|-------------|----------------|
| dh_baseline | **0.5135** | **0.50** (clamped) | ~0.358 µm |
| commensal_static | 0.0002 | 9.85 | ~0.018 µm |
| Dysbiotic_Static | 0.0005 | 9.78 | ~0.018 µm |
| Commensal_HOBIC | 0.0003 | 9.87 | ~0.018 µm |

- **MISES:** identical across conditions (force-controlled BC → geometry-dominated)
- **Displacement:** dh_baseline **~19.7× larger** (E_eff ratio 0.50/9.85 MPa)
- Figures: LateFig1–7 in `figures/`

---

### [P1] Real DI Map Projection  ← High priority

- Load actual clinical DI measurements (CSV with tooth/location/DI)
- Project onto element centroids via KDTree spatial lookup
- Replace entropy-based snapshot DI in the coupling pipeline
- **Blocker:** requires real clinical data CSV (currently none available)
- **Impact:** physically valid E distribution → clinically meaningful MISES

---

### [P3] ✅ COMPLETE (2026-02-23) Pressure Parameter Study

- Swept 0.01, 0.1, 0.5, 1.0, 5.0, 10.0 MPa — all 6 completed
- **Result:** Linear elastic confirmed across full range
- At 10 MPa: MISES median=5.2880 MPa, max=17.6020 MPa
- **Outputs:** `_pressure_sweep/pressure_sweep_results.json`, `P3_pressure_sweep.png`

---

### [P4] ✅ COMPLETE (2026-02-23) Mesh Convergence Study

- Script: `p4_mesh_convergence.py`; N_layers=4,8,16
- N=4: 0.5447 MPa | N=8: 0.5461 MPa | N=16: 0.5432 MPa
- **Verdict:** coarsest vs finest Δ=0.27% — **CONVERGED**; N=8 is adequate
- Figure: `figures/P4_mesh_convergence.png`

---

### [P5] Improve Slit Coupling  ← Low priority

- Replace `*Tie` with `*Contact` + friction (µ~0.1) for physiological sliding
- Biofilm between T30↔T31 can slide under masticatory load
- Requires surface-to-surface formulation; node-based Tie is too rigid
- **Impact:** more realistic slit mechanics (quantify via APPROX |U| change)

---

### [P7] ✅ COMPLETE (2026-02-23) Tie Coverage Investigation

- **Script:** `p7_tie_diagnostic.py`
- **Results:** T30 APPROX=1,614 nodes | T31 APPROX=1,278 nodes
- Only 8.6% of T31 APPROX nodes within 0.5 mm of T30 APPROX
- **Recommendation:** `--slit-max-dist` 0.4–0.6 mm
- **Interpretation:** 0.5 mm Abaqus Tie tolerance is appropriate; proximal slit geometry correct
- **Figures:** `figures/P7_tie_diagnostic.png`, `figures/P7_tie_3d_gap.png`

---

### [P8] ✅ COMPLETE (2026-02-23) Element Quality Report

- **Script:** `p8_element_quality.py`; INP: `biofilm_3tooth.inp`
- 437,472 C3D4 elements, 82,080 nodes
- **Negative volumes:** 0 ✅
- **Aspect ratio:** median=8.20, p99=10.87, max=13.89
- 19,765 elements (4.5%) with AR > 10 — acceptable for prism-to-tet split at thin surfaces
- **Shape quality Q:** median=0.343; only 3 elements Q<0.1 (0.001%) ✅
- **Figure:** `figures/P8_element_quality.png`

### [P9] ✅ COMPLETE (2026-02-23) E_eff Material-Model Sensitivity

- **Scripts:** `run_eeff_sensitivity_3tooth.py` (12 Abaqus runs), `plot_eeff_sensitivity.py` (4 figures)
- **Swept:** E_max ∈ {7.5,10,12.5} MPa; E_min ∈ {0.25,0.5,1.0} MPa; α ∈ {1,2,3}; s_DI ∈ {0.019,0.026,0.032}

| Parameter variation | ΔU_T23 | ΔMISES |
|--------------------|--------|--------|
| E_max: ±25% | +32%/−20% | <0.1% |
| E_min: ×0.5/×2 | +1.5%/−2.8% | <0.1% |
| α: 1/3 (baseline 2) | −27%/+39% | <0.6% |
| s_DI: −25%/+25% | +32%/−13% | <0.7% |

- **Key finding:** MISES insensitive (<1%) to all material parameters; displacement driven by E_max and α (snap20 regime)
- **Figures:** SensFig1–4 in `figures/`

### [P10] ✅ COMPLETE (2026-02-23) BC Sensitivity Study

- **Script:** `p10_bc_sensitivity.py`
- **BC types:** force-ctrl (Cload=1 MPa) vs displacement-ctrl (δ=1 µm inward-normal on 6,228 OUTER nodes)
- **Conditions:** dh_baseline and commensal_static, late-time DI fields

| BC type | dh_baseline MISES | commensal MISES | dh_baseline \|U\| | commensal \|U\| |
|---------|------------------|-----------------|------------------|----------------|
| Force-ctrl | ≈0.546 MPa | ≈0.546 MPa | 0.358 µm | 0.018 µm |
| Disp-ctrl (δ=1µm) | 0.669 MPa | 13.19 MPa | 0.270 µm | 0.270 µm |

- **MISES ratio (disp-ctrl):** 13.19/0.669 = **19.7×** (= E_eff ratio: 9.85/0.50)
- **Key finding:** BC type determines *which* quantity carries the biological signal; both yield the same ×19.7 contrast
- **Figures:** BCFig1–3 in `figures/`

---

## 6. Documentation Roadmap

| Document | Content | Status |
|----------|---------|--------|
| `FEM_PLAN.md` (this file) | Pipeline status, next steps | ✅ Done |
| `biofilm_3tooth_report.tex/.pdf` | Full 15-page pipeline report + LaTeX equations | ✅ Done |
| `tmcmc_to_fem_coupling.py` | P2 coupling script | ✅ Done |
| `FEM_RESULTS.md` | Tabulated MISES/U per condition run, parameter log | **High — next** |
| `compare_conditions_fem.py` | P0/P7 condition comparison script | **Planned** |
| `FEM_METHODS.md` | Meshing algorithm, material model deep-dive | Medium |
| Jupyter `fem_postprocess.ipynb` | Interactive CSV exploration | Medium |

---

## 7. Lessons Learned (Key Bugs)

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| Displacement ~0 (machine precision) | E in Pa, not MPa → 10⁶× too stiff | `E_MPa = bin_E_stiff[b] * 1e-6` |
| ODB sets not found | Sets at `inst.elementSets`, not `assembly.elementSets` | Changed to `inst` level |
| APPROX tag overwritten by OUTER | Alphabetical ODB set order | Sort: INNER=0, OUTER=1, APPROX=2 |
| 303 zero-volume elements | `adjust=YES` moved T31 slave nodes 0.041 mm into T30 | Changed to `adjust=NO` |

---

## 8. Run Commands

```bash
# ── Standard baseline run ─────────────────────────────────────────────────
cd Tmcmc202601/FEM

python3 biofilm_3tooth_assembly.py          # Step 1: Generate INP
abaqus job=BioFilm3T input=biofilm_3tooth.inp cpus=4 interactive  # Step 2
abaqus python odb_extract.py BioFilm3T.odb  # Step 3: Extract CSV
python3 odb_visualize.py                    # Step 4: Figures (Fig1–Fig12)

# ── P0: Condition comparison run ─────────────────────────────────────────
python3 tmcmc_to_fem_coupling.py --list     # Check available conditions

# Export DI field for a condition:
python3 tmcmc_to_fem_coupling.py --condition commensal_static --snapshot 20

# Regenerate INP and run Abaqus:
python3 tmcmc_to_fem_coupling.py --condition commensal_static --snapshot 20 --regen-inp
abaqus job=BioFilm3T_cs input=biofilm_3tooth_commensal_static.inp cpus=4 interactive
abaqus python odb_extract.py BioFilm3T_cs.odb

# ── P2: Coupling script standalone usage ─────────────────────────────────
python3 tmcmc_to_fem_coupling.py --condition dh_baseline --snapshot 20
python3 tmcmc_to_fem_coupling.py --condition dh_baseline --snapshot -1  # last snap
```

---

## 9. Condition Parameter Summary

| Condition | a35 | a45 | a55 | DI (snap20, t=0.05) | DI (late, t≈0.5) | E_eff late (MPa) | Character |
|-----------|-----|-----|-----|---------------------|------------------|-----------------|-----------|
| `dh_baseline` | **21.41** | 2.50 | 0.12 | **0.0070** | **0.5135** | **0.50** (clamped) | Dysbiotic (Pg-cascade dominates at late time) |
| `commensal_static` | 1.37 | 2.79 | 2.62 | 0.0095 | 0.0002 | 9.85 | Balanced commensal |
| `Dysbiotic_Static` | 2.03 | 2.16 | 2.95 | 0.0093 | 0.0005 | 9.78 | Moderate dysbiotic |
| `Commensal_HOBIC` | N/A | N/A | N/A | 0.0099 | 0.0003 | 9.87 | HOBIC commensal |

**Key findings:**
- **Snap20 (t=0.05):** dh_baseline DI is *lowest* (0.0070) — Pg cascade not yet developed at this early timepoint.
  Lower DI → more diverse species → lower r → *stiffer* biofilm (E_eff higher).
- **Late time (t≈0.5):** dh_baseline DI=0.5135 → r=1 (clamped at E_MIN=0.5 MPa); all other conditions maintain DI≈0 → E_eff≈9.8 MPa.
  **Displacement contrast ≈19.7× (clinically meaningful).**

> **Note:** theta_MAP files for Dysbiotic/Commensal conditions live in
> `_runs/{ConditionName}_{date}/theta_MAP.json`, not in the sweep directory.
> Run `tmcmc_to_fem_coupling.py --list` and look for matching condition names.
