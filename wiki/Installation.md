# Installation

## Requirements

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.11 | system or pyenv |
| NumPy | ≥1.24 | |
| SciPy | ≥1.11 | |
| JAX | 0.9.0.1 | x64 enabled; JAX-FEM only |
| jax-fem | 0.0.11 | JAX-FEM only |
| Abaqus | 2023 | HPC cluster; FEM only |

---

## 1. TMCMC Environment (main estimation)

The TMCMC pipeline uses the system Python (no special conda env required).

```bash
pip install numpy scipy pandas matplotlib
```

Clone and verify:

```bash
git clone https://github.com/keisuke58/Tmcmc202601.git
cd Tmcmc202601/data_5species

# Smoke test
python -m py_compile core/nishioka_model.py core/tmcmc.py core/evaluator.py core/mcmc.py
python -c "from core.nishioka_model import INTERACTION_GRAPH_JSON; print('OK')"
```

---

## 2. JAX-FEM Environment (nutrient transport PDE)

A separate conda environment is required for JAX-FEM demos.

```bash
# Create env (Python 3.11)
conda create -n klempt_fem python=3.11
conda activate klempt_fem

# Install JAX (CPU)
pip install "jax[cpu]==0.9.0.1"

# Install jax-fem 0.0.11
pip install jax-fem==0.0.11

# Install supporting packages
pip install basix==0.10.0 meshio matplotlib
```

The environment lives at:
```
~/.pyenv/versions/miniconda3-latest/envs/klempt_fem/
```

Run JAX-FEM scripts with the full path:
```bash
~/.pyenv/versions/miniconda3-latest/envs/klempt_fem/bin/python \
    FEM/jax_fem_reaction_diffusion_demo.py
```

### Patches applied to jax-fem 0.0.11

The `solver.py` file requires patching since petsc4py is not installed:

```python
# site-packages/jax_fem/solver.py — line 8
# BEFORE:
from petsc4py import PETSc

# AFTER:
try:
    from petsc4py import PETSc
    PETSC_AVAILABLE = True
except ImportError:
    PETSC_AVAILABLE = False
```

See `FEM/JAXFEM/` for the patched version.

---

## 3. Abaqus (FEM stress analysis)

Abaqus 2023 is used on an HPC cluster. The Python scripts generate `.inp` files that are submitted to Abaqus:

```bash
# Generate INP
python FEM/biofilm_conformal_tet.py \
    --stl external_tooth_models/.../P1_Tooth_23.stl \
    --di-csv abaqus_field_dh_3d.csv \
    --out p23_biofilm.inp --mode biofilm

# Submit to Abaqus (on cluster)
abaqus job=p23_biofilm interactive
```

---

## Directory Structure After Setup

```
Tmcmc202601/
├── data_5species/
│   ├── core/           # ← core modules
│   ├── model_config/   # ← prior_bounds.json
│   └── _runs/          # ← output (created at runtime)
├── FEM/
│   └── klempt2024_results/   # ← created at runtime
└── wiki/               # ← this wiki (source)
```
