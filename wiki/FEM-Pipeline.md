# FEM Pipeline

## Overview

The FEM pipeline takes TMCMC posterior samples and produces stress maps on a 3D tooth model:

```
posterior_samples.npy
        │
        ▼
 φᵢ(x) composition fields  (5-species spatial distribution)
        │
        ▼
 DI(x) = 1 − H(x)/log(5)  (Dysbiotic Index, 0=healthy, 1=fully dysbiotic)
        │
        ▼
 E(x) = E_max·(1−r)ⁿ + E_min·r  (stiffness power-law mapping)
        │
        ▼
 Abaqus 3D FEM  (INP generation → job submission)
        │
        ▼
 S_Mises · U_max per condition
```

---

## Step 1: Posterior Ensemble → DI Fields

```bash
cd Tmcmc202601/FEM
python run_posterior_abaqus_ensemble.py \
    --runs-dir ../data_5species/_runs \
    --condition Dysbiotic_HOBIC \
    --n-samples 100
```

Output: `FEM/_posterior_abaqus/<condition>/` — one DI field CSV per posterior sample.

---

## Step 2: Aggregate Credible Intervals

```bash
python aggregate_di_credible.py
```

Output: `FEM/_di_credible/<condition>/fig_di_spatial_ci.png`, `fig_di_depth_profile.png`

---

## Step 3: Generate Abaqus INP

### Substrate mode (GPa, S_Mises primary)

```bash
python biofilm_conformal_tet.py \
    --stl external_tooth_models/Open-Full-Jaw-main/P1_Tooth_23.stl \
    --di-csv abaqus_field_dh_3d.csv \
    --out p23_substrate.inp \
    --mode substrate
```

### Biofilm mode (Pa-scale EPS, NLGEOM)

```bash
python biofilm_conformal_tet.py \
    --stl external_tooth_models/.../P1_Tooth_23.stl \
    --di-csv abaqus_field_dh_3d.csv \
    --out p23_biofilm.inp \
    --mode biofilm
```

### Neo-Hookean (large strain)

```bash
python biofilm_conformal_tet.py ... --neo-hookean
```

---

## Step 4: Submit to Abaqus

On HPC cluster:

```bash
abaqus job=p23_biofilm cpus=8 interactive
```

Output: `p23_biofilm.odb` → post-process with `abaqus viewer` or `abaqus python`.

---

## Material Parameters

### Stiffness Mapping

```
DI(x) = 1 − H(x) / log(5)       H = Shannon entropy of φᵢ
r(x)  = clamp(DI(x) / s, 0, 1)  s = 0.025778 (scaling)
E(x)  = E_max · (1−r)ⁿ + E_min · r
```

### Default Values

| Parameter | Substrate mode | Biofilm mode |
|-----------|:--------------:|:------------:|
| E_max | 10 GPa | 100 Pa |
| E_min | 0.5 GPa | 10 Pa |
| n (exponent) | 2 | 2 |
| ν (Poisson) | 0.3 | 0.45 |

Sweep over E_max/E_min/n: `python run_material_sensitivity_sweep.py`

---

## Tooth Geometry

Patient-specific mandible meshes from **Open-Full-Jaw** [Gholamalizadeh et al. 2022]:

```
FEM/external_tooth_models/Open-Full-Jaw-main/
├── P1_Tooth_23.stl     # lower left canine
├── P1_Tooth_30.stl     # lower left first molar
└── P1_Tooth_31.stl     # lower left second molar
```

- CBCT-derived, PDL gap ≈ 0.2 mm
- Volumetric tet mesh generated with fTetWild

---

## JAX-FEM: Nutrient Transport (Klempt 2024)

Separate from the Abaqus pipeline. Solves the steady-state PDE:

```
−D_c Δc + g φ₀(x) c/(k+c) = 0    in Ω = [0,1]²
c = 1                              on ∂Ω
```

```bash
~/.pyenv/versions/miniconda3-latest/envs/klempt_fem/bin/python \
    FEM/jax_fem_reaction_diffusion_demo.py
# → FEM/klempt2024_results/klempt2024_nutrient.vtu + .png
```

**Result**: c_min ≈ 0.31 inside biofilm (Thiele modulus ~4, diffusion-limited).

### jax-fem 0.0.11 API Notes

| Element | API |
|---------|-----|
| Mesh | `rectangle_mesh(Nx, Ny, Lx, Ly)` → `meshio.Mesh` |
| Problem | `@dataclass(mesh, vec, dim, ele_type, dirichlet_bc_info)` |
| tensor_map | `(u_grads: (vec,dim)) → (vec,dim)` |
| Solver | `solver(problem, {})` → `sol_list` |
| Save | `save_sol(problem.fes[0], sol, path)` |

---

## Output Files

```
FEM/
├── _results_3d/<condition>/
│   ├── fig1_3d_slices.png
│   ├── fig4_dysbiotic_3d.png
│   ├── fig6_overview_pg_3d.png
│   └── fig7_overview_all_species_3d.png
├── _di_credible/<condition>/
│   ├── fig_di_spatial_ci.png
│   └── fig_di_depth_profile.png
├── _posterior_uncertainty/
│   ├── Fig1_stress_violin.png
│   └── Fig3_sensitivity_heatmap.png
└── klempt2024_results/
    ├── klempt2024_nutrient.vtu
    └── klempt2024_nutrient.png
```
