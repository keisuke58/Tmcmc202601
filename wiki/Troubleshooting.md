# Troubleshooting

## TMCMC Issues

### Particle Collapse (ESS near zero)

**Symptom**: `WARNING: ESS = 3 / 150` after first tempering stage.

**Cause**: Likelihood is too narrow relative to prior — all particles get near-zero weight.

**Fix**:
1. Reduce `--n-stages` (fewer, larger β steps)
2. Widen prior bounds in `model_config/prior_bounds.json`
3. Reduce `--lambda-pg` (de-emphasize Pg likelihood)

---

### Pg Hitting Upper Prior Bound

**Symptom**: `a35 = 5.0` (== upper bound), posterior concentrated at boundary.

**Cause**: Prior too narrow for bridge facilitation strength.

**Fix**: Widen `a35` and `a45` bounds:

```json
{
  "a35": [0, 10],
  "a45": [0, 10]
}
```

Or use `prior_bounds_original.json` as a starting point.

---

### Poor Pg Fit (RMSE > 0.5)

**Symptom**: Pg trajectory doesn't follow the measured surge.

**Fix options** (in order of severity):
1. Increase `--lambda-pg 3.0` or `4.0`
2. Increase `--n-particles 500` for more posterior coverage
3. Check if the data condition is correct (`--condition Dysbiotic_HOBIC`)
4. Verify Hill gate parameters in `improved_5species_jit.py`: K=0.05, n=4

---

### NaN in ODE Integration

**Symptom**: `RuntimeWarning: invalid value encountered in multiply`

**Cause**: φᵢ goes negative due to aggressive interaction terms.

**Fix**: Tighten growth rate bounds or add a clipping floor in the ODE:

```python
phi = np.maximum(phi, 0.0)   # in nishioka_model.py RHS
```

---

### Slow Run on Single Core

**Symptom**: 1000-particle run takes > 24 h.

**Fix**: Use the parallel sweep runner which distributes conditions across cores:

```bash
python run_bridge_sweep.py --n-particles 1000 --n-jobs 4
```

Or run all 4 conditions in background:

```bash
nohup python run_bridge_sweep.py --n-particles 1000 > sweep.log 2>&1 &
```

---

## JAX-FEM Issues

### `ImportError: No module named 'petsc4py'`

**Fix**: This is expected — petsc4py is not installed. The solver.py patch routes to the scipy fallback. Verify the patch is applied:

```bash
grep "PETSC_AVAILABLE" \
  ~/.pyenv/versions/miniconda3-latest/envs/klempt_fem/lib/python3.11/site-packages/jax_fem/solver.py
```

If missing, re-apply the patch (see [Installation](Installation#patches-applied-to-jax-fem-011)).

---

### `AttributeError: 'Mesh' object has no attribute 'points'`

**Cause**: jax-fem 0.0.11 API changed — `rectangle_mesh()` now returns `meshio.Mesh`.

**Fix**:

```python
# WRONG (old API):
nodes, cells = rectangle_mesh(Nx, Ny, Lx, Ly)

# CORRECT (0.0.11):
mesh_data = rectangle_mesh(Nx, Ny, Lx, Ly)
mesh = Mesh(mesh_data.points, mesh_data.cells[0].data, ele_type='QUAD4')
```

---

### `tensor_map` Shape Error

**Cause**: `tensor_map` signature changed in 0.0.11.

```python
# WRONG:
def tensor_map(self, x):   # x is position

# CORRECT:
def tensor_map(self, u_grads):   # u_grads has shape (vec, dim)
    return -self.D_c * u_grads
```

---

### Newton Solver Not Converging

**Symptom**: `Newton iteration did not converge after 20 steps`

**Cause**: Thiele modulus too high or initial guess far from solution.

**Fix**:
1. Reduce `g_eff` (consumption rate)
2. Provide better initial guess: `sol = problem.set_params(c0=1.0)`
3. Check boundary conditions are correctly specified

---

## Abaqus Issues

### Element Distortion Warning

**Symptom**: `WARNING: 3 elements have excessive distortion`

**Fix**: Regenerate tet mesh with finer resolution:

```bash
python biofilm_conformal_tet.py ... --mesh-size 0.05
```

---

### `.odb` File Not Written

**Cause**: Abaqus job terminated early (check `.msg` file).

```bash
grep "ABAQUS" p23_biofilm.msg | tail -20
```

Common cause: material properties set to zero (DI field CSV not loaded correctly).

---

## CI Failures

### `AssertionError: Expected 15 active, got X`

**Cause**: `get_nishioka_mask()` in `nishioka_model.py` returned unexpected count.

**Fix**: Verify that the mask definition has exactly:
- 5 diagonal entries (self-inhibition)
- 10 off-diagonal active edges

```python
mask = get_nishioka_mask()
print(mask)  # should be 5×5 boolean array
```

---

## Getting Help

If you encounter an issue not listed here:

1. Check existing [Issues](https://github.com/keisuke58/Tmcmc202601/issues)
2. Open a [Discussion](https://github.com/keisuke58/Tmcmc202601/discussions) with:
   - Your run command and configuration
   - The full error traceback
   - `config.json` from the failing run
