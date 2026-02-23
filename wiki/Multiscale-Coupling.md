# Multiscale Micro→Macro Coupling

> **Added 2026-02-24.** Bridges the 0D/1D ecology ODE with a spatially varying growth eigenstrain ε(x) that feeds directly into Abaqus.

---

## Motivation

The TMCMC posterior yields a *mean-field* community composition θ. Feeding it directly into a 3D FEM gives a uniform material field. The multiscale pipeline adds **spatial resolution**: nutrient transport depletes toward the tooth surface, so the inner biofilm barely grows while the saliva-exposed outer layer grows at up to 14% volumetric strain.

---

## Pipeline

```
TMCMC θ_MAP
    │
    ├─► 0D JAX ODE  (T* = 25, n = 2500 steps, dt = 0.01)
    │       → DI_0D        commensal ≈ 0.05,  dysbiotic ≈ 0.84  (18× difference)
    │       → α_0D         condition-specific mean metabolic activity
    │
    └─► 1D Hamilton ODE + Nutrient PDE  (N = 30 nodes, T* = 20, dt_h = 1e-3)
            → c(x, T)      nutrient concentration field
            → φᵢ(x, T)     species abundance profiles along depth x ∈ [0,1]
            │              (x=0: tooth surface, x=1: saliva interface)
            ▼
        α_Monod(x) = k_α · ∫₀ᵀ φ_total(x,t) · c(x,t)/(k + c(x,t)) dt
                                                    ↑ KEY spatial bridge
        ε_growth(x) = α_Monod(x) / 3               (isotropic volumetric eigenstrain)
            │
            ▼
        Abaqus T3D2 bar-element INP
        Thermal-analogy:  α_th = 1.0,  ΔT(x) = ε_growth(x)
        → spatially non-uniform eigenstrain field along depth
```

---

## Key Equations

### Nutrient PDE (1D steady-state, solved at each macro time step)

```
∂c/∂t = D_c · ∂²c/∂x² − g · φ_total(x) · c/(k + c)
BC:  c(x=1, t) = 1   (Dirichlet at saliva interface)
     ∂c/∂x|_{x=0} = 0  (no-flux at tooth surface)
```

### Hamilton ODE for species dynamics

```
dφᵢ/dt = φᵢ · (rᵢ − dᵢ·φᵢ + Σⱼ aᵢⱼ · H(φⱼ))
H(φⱼ) = φⱼⁿ / (Kⁿ + φⱼⁿ)   Hill gate, K=0.05, n=4
```

Parameters θ taken from `data_5species/_runs/<condition>/theta_MAP.json`.

### Stiffness from DI (Hybrid mode)

```
r   = clamp(DI_0D / s, 0, 1)   s = 0.025778
E(x) = [E_max · (1−r)² + E_min · r] · (α_Monod(x) / α_Monod_max)
```

---

## Numerical Results (2026-02-24)

### 0D ODE — Condition Discrimination

| Quantity | Commensal | Dysbiotic | Ratio |
|----------|:---------:|:---------:|:-----:|
| DI_0D | 0.047 | 0.845 | **18×** |
| α_0D (mean activity) | ~low | ~high | — |

### 1D Spatial Fields

| Location | α_Monod | ε_growth |
|----------|:-------:|:--------:|
| x = 0 (tooth surface) | 0.004 | 0.0013 (~0%) |
| x = 1 (saliva side) | 0.420 | 0.14 (14%) |
| **Gradient (x=1 / x=0)** | **101×** | **101×** |

> **Key finding**: The 1D diffusion homogenises the species composition (DI ≈ 0 everywhere in 1D), so spatial variation comes entirely from the nutrient field c(x). Condition discrimination (commensal vs dysbiotic) therefore lives in the 0D DI_0D scalar, not in the spatial profile.

### Effective Stiffness (Hybrid CSV)

| Condition | E_eff mean (Pa) | σ_0 at x=0 (Pa) |
|-----------|:--------------:|:---------------:|
| Commensal | ~909 | −1.26 |
| Dysbiotic | ~33 | −0.045 |
| **Ratio** | **28×** | **28×** |

---

## Implementation Files

| File | Purpose |
|------|---------|
| `FEM/multiscale_coupling_1d.py` | Full pipeline: 0D ODE + 1D PDE → α_Monod(x) CSV |
| `FEM/generate_hybrid_macro_csv.py` | Hybrid CSV: combines 0D DI scalar with 1D spatial α |
| `FEM/generate_abaqus_eigenstrain.py` | Generates Abaqus T3D2 bar INP via thermal analogy |
| `FEM/generate_pipeline_summary.py` | 9-panel summary figure of the full pipeline |

### Output Directories

```
FEM/
├── _multiscale_results/
│   ├── macro_eigenstrain_commensal.csv    # x, alpha_Monod, eps_growth
│   ├── macro_eigenstrain_dysbiotic.csv
│   └── multiscale_comparison.png
├── _abaqus_input/
│   ├── biofilm_1d_bar_commensal.inp       # Abaqus T3D2 bar element
│   ├── biofilm_1d_bar_dysbiotic.inp
│   ├── eigenstrain_field_commensal.csv
│   └── eigenstrain_field_dysbiotic.csv
└── _pipeline_summary/
    └── pipeline_summary.png               # 9-panel overview
```

---

## Running the Pipeline

### Step 1: Full 0D + 1D coupling

```bash
~/.pyenv/versions/miniconda3-latest/envs/klempt_fem/bin/python \
    FEM/multiscale_coupling_1d.py
```

Outputs:
- `_multiscale_results/macro_eigenstrain_{commensal,dysbiotic}.csv`
- `_multiscale_results/multiscale_comparison.png`

### Step 2: Generate Hybrid CSV (0D DI × 1D spatial α)

```bash
~/.pyenv/versions/miniconda3-latest/envs/klempt_fem/bin/python \
    FEM/generate_hybrid_macro_csv.py
```

### Step 3: Generate Abaqus INP

```bash
~/.pyenv/versions/miniconda3-latest/envs/klempt_fem/bin/python \
    FEM/generate_abaqus_eigenstrain.py
# → FEM/_abaqus_input/biofilm_1d_bar_{commensal,dysbiotic}.inp
```

Submit to Abaqus on cluster:

```bash
abaqus job=biofilm_1d_bar_dysbiotic cpus=4 interactive
```

### Step 4: Pipeline summary figure

```bash
~/.pyenv/versions/miniconda3-latest/envs/klempt_fem/bin/python \
    FEM/generate_pipeline_summary.py
# → FEM/_pipeline_summary/pipeline_summary.png
```

---

## Environment Notes

- Requires the **klempt_fem** conda environment (Python 3.11, JAX 0.9.0.1, numpy ≥2.0)
- `numpy.trapz` was removed in NumPy 2.x — use `numpy.trapezoid` instead
- `K_hill` must be wrapped as `jnp.array(K_hill)` in the 0D JAX ODE solver
- **No pandas** in klempt_fem — use `open()` + manual line parsing, or `np.loadtxt()`
- See [Troubleshooting](Troubleshooting) for JAX-FEM-specific issues

---

## Known Limitations

| Limitation | Impact | Next step |
|-----------|--------|-----------|
| 1D diffusion homogenises species composition | DI ≈ 0 everywhere in 1D → condition discrimination lost | Use 0D DI as scalar multiplier (Hybrid mode) |
| 0D ODE ignores spatial nutrient gradient | α_0D does not reflect depth-dependent starvation | 1D coupling provides α_Monod(x) |
| T3D2 bar is a proof-of-concept geometry | Not anatomically realistic | Extend to 3D tet mesh using spatial interpolation |
