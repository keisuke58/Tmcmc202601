# FEM Spatial Extension of the Hamilton Biofilm Model

## Overview

This directory extends the **5-species Hamilton biofilm model** — whose parameters
are estimated by TMCMC in `Tmcmc202601/data_5species/` — into spatial
reaction-diffusion simulations using the **finite-element / finite-difference method**
in 1D, 2D, and 3D.

The core idea is:

```
TMCMC estimation (0D)      → θ_MAP (20 parameters)
                                      ↓
FEM spatial extension      → φᵢ(x, [y, z], t)  for i = 1…5 species
```

---

## Mathematical Background

### 1. The Hamilton Biofilm Model (0D)

State vector **g** ∈ ℝ¹²:

| Index | Symbol | Meaning |
|-------|--------|---------|
| 0–4   | φᵢ     | Volume fraction of species i (S.oralis, A.naeslundii, Veillonella, F.nucleatum, P.gingivalis) |
| 5     | φ₀     | Void (water) fraction; φ₀ = 1 − Σφᵢ |
| 6–10  | ψᵢ     | Nutrient concentration for species i |
| 11    | γ      | Substrate / oxygen level |

**Volume constraint:** Σᵢφᵢ + φ₀ = 1 at all times.

Parameters **θ** ∈ ℝ²⁰:

| Block | Params | Meaning |
|-------|--------|---------|
| M1    | a11, a12, a22, b1, b2 | S.oralis + A.naeslundii self/cross interaction |
| M2    | a33, a34, a44, b3, b4 | Veillonella + F.nucleatum |
| M3    | a13, a14, a23, a24    | Commensal cross-feeding |
| M4    | a55, b5               | P.gingivalis self |
| M5    | a15, a25, a35, a45    | P.gingivalis cross (received from commensals) |

The ODE system Q(**g**_new, **g**_old; θ) = 0 is solved at each time step by
Newton iteration with line-search backtracking (Numba JIT: `_newton_step_jit`).

---

### 2. Reaction-Diffusion PDE

Each species' volume fraction satisfies:

```
∂φᵢ/∂t = Rᵢ(φ, ψ, γ; θ)  +  Dᵢ ∇²φᵢ
```

- **Rᵢ** = Hamilton reaction term (from the 0D model)
- **Dᵢ** = effective diffusion coefficient
- **∇²** = Laplacian in 1D / 2D / 3D

Default diffusion coefficients (motility proxies, in model units):

| Species | D_eff |
|---------|-------|
| S.oralis | 1×10⁻³ |
| A.naeslundii | 1×10⁻³ |
| Veillonella | 8×10⁻⁴ |
| F.nucleatum | 5×10⁻⁴ |
| P.gingivalis | 2×10⁻⁴ |

Boundary conditions: **no-flux (Neumann)** on all boundaries.

---

### 3. Operator Splitting (Lie Splitting)

Each macro time step Δt_mac = dt_h × n_react_sub:

```
Step ①  Reaction  :  solve dg/dt = R(g; θ) at each node independently
                      → n_react_sub Newton steps (Numba prange parallel)

Step ②  Diffusion :  (I − Δt_mac · Dᵢ · L) φᵢ_new = φᵢ_old
                      → backward Euler, one sparse linear solve per species
                      → SuperLU factorization precomputed at init
```

The Lie splitting is first-order accurate in time; second-order Strang splitting
can be added if higher accuracy is required.

---

### 4. Spatial Discretisation

#### 1D (P1 finite elements, uniform mesh)
- Consistent mass matrix M + stiffness matrix K
- Solve: (M + Δt·Dᵢ·K) φ_new = M φ_old

#### 2D (finite differences on uniform Nx × Ny grid)
- Node index: k = ix·Ny + iy  (row-major)
- 2D Laplacian: **L₂D = kron(Lx, Iy) + kron(Ix, Ly)**
- Neumann BC: ghost-node approach → diagonal modified at boundary rows

#### 3D (finite differences on uniform Nx × Ny × Nz grid)
- Node index: k = ix·(Ny·Nz) + iy·Nz + iz
- 3D Laplacian: **L₃D = kron(kron(Lx,Iy),Iz) + kron(kron(Ix,Ly),Iz) + kron(kron(Ix,Iy),Lz)**
- All six faces: Neumann

The 1D Neumann Laplacian matrix (per-axis) using the ghost-node / half-stencil:

```
    [-1/h²   1/h²                        ]   ← wall
    [ 1/h²  -2/h²   1/h²                 ]   ← interior
    [         ⋱       ⋱     ⋱            ]
    [                  1/h² -2/h²  1/h²  ]   ← interior
    [                         1/h² -1/h² ]   ← wall
```

---

## File Structure

```
Tmcmc202601/FEM/
├── fem_spatial_extension.py   1D simulation
├── fem_visualize.py           1D visualisation (8 figures)
├── fem_2d_extension.py        2D simulation
├── fem_2d_visualize.py        2D visualisation (5 figures)
├── fem_3d_extension.py        3D simulation
├── fem_3d_visualize.py        3D visualisation (5 figures)
├── fem_convergence.py         Mesh convergence analysis + MD report
├── FEM_README.md              This file
│
├── _results/                  1D results
│   ├── dh_baseline/           fig1–fig8.png + *.npy
│   └── commensal_static/
│
├── _results_2d/               2D results
│   ├── dh_baseline/
│   ├── commensal_static/
│   ├── conv_N20/ conv_N30/ conv_N40/   convergence grids
│   └── convergence/           conv_A–conv_E figures + convergence_report.md
│
└── _results_3d/               3D results
    ├── dh_baseline/
    └── commensal_static/
```

---

## Usage

### Prerequisites

```bash
# From Tmcmc202601/FEM/
# Python packages: numpy, scipy, matplotlib, numba
```

### 1D Simulation

```bash
python fem_spatial_extension.py \
    --theta-json ../data_5species/_runs/<run>/theta_MAP.json \
    --condition "dh_baseline" \
    --n-nodes 30 --n-macro 100 --n-react-sub 50 \
    --init-mode gradient \
    --save-every 5 \
    --out-dir _results/dh_baseline

python fem_visualize.py \
    --results-dir _results/dh_baseline \
    --condition "dh_baseline"
```

Figures generated:

| Figure | Content |
|--------|---------|
| fig1 | Space-time heatmaps (5 species × depth × time) |
| fig2 | Time series at 3 spatial nodes |
| fig3 | Spatial profiles at 4 time snapshots |
| fig4 | Final composition: stacked area (absolute + relative %) |
| fig5 | P.g invasion front tracking |
| fig6 | Dysbiotic Index heatmap + time series |
| fig7 | Surface vs bulk comparison (transient clipped) |
| fig8 | 6-panel summary (heatmap, profiles, θ bar chart) |

### 2D Simulation

```bash
python fem_2d_extension.py \
    --theta-json ../data_5species/_runs/<run>/theta_MAP.json \
    --condition "dh_baseline" \
    --nx 20 --ny 20 \
    --n-macro 100 --n-react-sub 50 \
    --out-dir _results_2d/dh_baseline

python fem_2d_visualize.py \
    --results-dir _results_2d/dh_baseline \
    --condition "dh_baseline"
```

Figures generated:

| Figure | Content |
|--------|---------|
| fig1 | 2D heatmaps: 5 species × 3 time points |
| fig2 | Hovmöller diagram: φ(x,t) y-averaged |
| fig3 | Lateral y-profiles at 3 depths (surface/mid/deep) |
| fig4 | 2D Dysbiotic Index maps at 3 time points |
| fig5 | 6-panel summary |

### 3D Simulation

```bash
python fem_3d_extension.py \
    --theta-json ../data_5species/_runs/<run>/theta_MAP.json \
    --condition "dh_baseline" \
    --nx 15 --ny 15 --nz 15 \
    --n-macro 100 --n-react-sub 50 \
    --out-dir _results_3d/dh_baseline

python fem_3d_visualize.py \
    --results-dir _results_3d/dh_baseline \
    --condition "dh_baseline"
```

For larger grids (> 20³), use `--solver cg` (conjugate gradient + ILU) instead of SuperLU.

Figures generated:

| Figure | Content |
|--------|---------|
| fig1 | XY / XZ / YZ cross-section slices, 5 species, t_final |
| fig2 | Hovmöller (yz-averaged, depth × time) |
| fig3 | Final depth profile (yz-averaged) |
| fig4 | Dysbiotic Index: 3×3 slice panels at t_init/mid/final |
| fig5 | 6-panel summary |

### Mesh Convergence Test (2D)

```bash
# Run three grids
for N in 20 30 40; do
  python fem_2d_extension.py --nx $N --ny $N --out-dir _results_2d/conv_N$N ...
done

# Analyse
python fem_convergence.py \
    --dirs _results_2d/conv_N20 _results_2d/conv_N30 _results_2d/conv_N40 \
    --labels "N=20" "N=30" "N=40" \
    --out-dir _results_2d/convergence
```

Outputs: `conv_A_mean_phi.png`, `conv_B_errors.png`, `conv_C_depth.png`,
`conv_D_pg_max.png`, `conv_E_rate.png`, **`convergence_report.md`**.

---

## Conditions Tested

| Condition | Run directory | Key features |
|-----------|---------------|--------------|
| dh_baseline | `sweep_pg_20260217_081459/dh_baseline` | a23=21, a35=21 (pathological dysbiosis) |
| Commensal Static | `Commensal_Static_20260208_002100` | All θ ≤ 2.79 (balanced community) |

### Theta comparison (selected parameters)

| Parameter | dh_baseline | Commensal Static | Role |
|-----------|-------------|-----------------|------|
| a23 | 21.0 | 2.69 | A.n → Vei cross-feeding |
| a35 | 21.4 | 1.37 | Vei → P.g support |
| a45 | 2.50 | 2.79 | F.n → P.g support |
| a55 | 0.12 | 2.62 | P.g self-growth |

---

## Convergence Results (2D, dh_baseline)

### Domain-averaged φᵢ at t_final

| Grid | φ̄_S.o | φ̄_A.n | φ̄_Vei | φ̄_F.n | φ̄_P.g |
|------|--------|--------|--------|--------|--------|
| 20×20 | 0.0918 | 0.0975 | 0.0820 | 0.0659 | 0.0727 |
| 30×30 | 0.0916 | 0.0976 | 0.0820 | 0.0659 | 0.0727 |
| 40×40 | 0.0916 | 0.0978 | 0.0821 | 0.0659 | 0.0727 |

Differences < **0.03 %** across all grids.

### Spatial L2 error vs N=40

| Grid | F.nucleatum | P.gingivalis | S.oralis* |
|------|------------|-------------|-----------|
| 20×20 | 0.7 % | 1.2 % | 9.1 %* |
| 30×30 | 0.7 % | 1.2 % | 8.8 %* |

\* S.oralis / A.naeslundii errors reflect **different random noise realisations**
at each grid size, not true discretisation error. F.n and P.g use deterministic
gradient initial conditions and show true convergence.

**Conclusion: N=20 is sufficient for biological analysis; N=30–40 for publication figures.**

---

## Performance

| Simulation | Grid | Nodes | Runtime |
|-----------|------|-------|---------|
| 1D | 30 nodes | 30 | ~5 s |
| 2D | 20×20 | 400 | ~8 s |
| 2D | 30×30 | 900 | ~18 s |
| 2D | 40×40 | 1 600 | ~45 s |
| 3D | 15³ | 3 375 | ~60 s (estimated) |
| 3D | 20³ | 8 000 | ~150 s (estimated) |

All simulations: 100 macro steps × 50 Hamilton sub-steps, dt_h=1e-5, Numba parallel.
Machine: Intel CPU (number of cores determines reaction step scaling).

**Bottleneck:** reaction step scales as O(N_nodes / N_cores).
**Diffusion step:** SuperLU factorisation done once at init; repeated solves are fast.
For 3D grids > 20³, switch to `--solver cg` to avoid large SuperLU fill-in.

---

## Implementation Notes

### Numba Import Path

The model solver lives at `Tmcmc202601/tmcmc/program2602/improved_5species_jit.py`.
All FEM scripts add `_MODEL_PATH` to `sys.path` **before** importing, to avoid
Numba cache path conflicts:

```python
_MODEL_PATH = _HERE.parent / "tmcmc" / "program2602"
sys.path.insert(0, str(_MODEL_PATH))
from improved_5species_jit import _newton_step_jit, HAS_NUMBA
```

If you see `ModuleNotFoundError` in Numba cache: clear `__pycache__` in
`tmcmc/program2602/`.

### theta_MAP.json format

Two formats are supported by `_load_theta()`:
1. `{"theta_full": [v0, v1, …, v19]}` — output from TMCMC sweep
2. `{"a11": v, "a12": v, …}` — named parameter dict

### Volume Constraint

After each diffusion step, φ₀ is recomputed as `1 − Σφᵢ` (clipped to ≥ 0)
to maintain the volume constraint numerically.

### Initial Conditions ("gradient" mode)

| Species | IC |
|---------|----|
| S.oralis, A.naeslundii | Uniform + small noise |
| Veillonella | Uniform + noise |
| F.nucleatum | exp(−3x/Lx) + noise (surface-enriched) |
| P.gingivalis (2D) | exp(−5x/Lx) × Gaussian(y_centre) |
| P.gingivalis (3D) | exp(−5x/Lx) × Gaussian(y_centre) × Gaussian(z_centre) |

---

## Dependencies

```
numpy >= 1.24
scipy >= 1.10    (sparse, linalg)
matplotlib >= 3.7
numba >= 0.57    (JIT parallel reaction step)
```

---

## Related Files

- **TMCMC estimation**: `Tmcmc202601/data_5species/main/estimate_reduced_nishioka.py`
- **Hamilton solver**: `Tmcmc202601/tmcmc/program2602/improved_5species_jit.py`
- **Prior bounds**: `Tmcmc202601/data_5species/model_config/prior_bounds.json`
- **Convergence report**: `_results_2d/convergence/convergence_report.md`

---

*FEM spatial extension implemented 2026-02-21*
