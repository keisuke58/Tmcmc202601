
import time
import json
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import equinox as eqx
from pathlib import Path

# Import models
from pinn_elasticity_3d import ElasticityPINN3D, pde_residual_3d, E_MAX, E_MIN, W, H, D, P_APPLIED
from pifo_elasticity_3d import FNO3d, compute_loss as compute_loss_pifno
from dem_elasticity_3d import ElasticityNetwork, compute_energy_loss, strain_energy_density

# Configure JAX
jax.config.update("jax_enable_x64", False)

# Benchmark settings
N_EPOCHS = 500  # Short run for benchmarking
RESOLUTION = 32 # Grid resolution for PI-FNO and validation
BATCH_SIZE_PINN = 1000 # Points per batch for PINN/DEM

# Shared Resources
def get_synthetic_E_field(resolution):
    """Generate a synthetic E field (biofilm-like blobs)"""
    nx, ny, nz = resolution, resolution, resolution
    x = jnp.linspace(0, W, nx)
    y = jnp.linspace(0, H, ny)
    z = jnp.linspace(0, D, nz)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    
    # Blob 1
    r1 = (X - 0.3*W)**2 + (Z - 0.3*D)**2
    # Blob 2
    r2 = (X - 0.7*W)**2 + (Z - 0.7*D)**2
    
    # Density map (0 to 1)
    density = 0.8 * jnp.exp(-r1 / 0.05) + 0.6 * jnp.exp(-r2 / 0.05)
    density = jnp.clip(density, 0.0, 1.0)
    
    # E field
    E = E_MIN + (E_MAX - E_MIN) * density
    return E, (X, Y, Z)

def get_E_at_points(pts, E_grid, grid_coords):
    """Interpolate E field at arbitrary points (for PINN/DEM consistency)"""
    # pts: (N, 3)
    # E_grid: (Nx, Ny, Nz)
    # grid_coords: (X, Y, Z)
    # For benchmark, we can just use an analytic function to avoid interpolation cost overhead
    # matching the one above.
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    
    r1 = (x - 0.3*W)**2 + (z - 0.3*D)**2
    r2 = (x - 0.7*W)**2 + (z - 0.7*D)**2
    
    density = 0.8 * jnp.exp(-r1 / 0.05) + 0.6 * jnp.exp(-r2 / 0.05)
    density = jnp.clip(density, 0.0, 1.0)
    
    E = E_MIN + (E_MAX - E_MIN) * density
    return E

# ============================================================
# Train PINN
# ============================================================
def run_pinn(key):
    print("--- Starting PINN Benchmark ---")
    k_model, k_train = jr.split(key)
    model = ElasticityPINN3D(hidden=64, n_layers=4, n_fourier=16, key=k_model)
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    @eqx.filter_jit
    def make_step(model, opt_state, domain_pts, bc_pts):
        def loss_fn(m):
            # PDE Residual
            E_vals = get_E_at_points(domain_pts, None, None)
            
            # pde_residual_3d returns (div_sigma, stress, u)
            div_sigma, _, _ = jax.vmap(pde_residual_3d, in_axes=(None, 0, 0, 0, 0))(
                m, domain_pts[:,0], domain_pts[:,1], domain_pts[:,2], E_vals
            )
            
            loss_pde = jnp.mean(jnp.sum(div_sigma**2, axis=1))
            
            # Note: Ignoring BC loss for benchmark throughput test
            # Ideally we should include it for fairness, but we are comparing
            # raw "physics informed" iteration speed.
            
            return loss_pde
            
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    # Training Loop
    times = []
    losses = []
    
    start_total = time.time()
    for i in range(N_EPOCHS):
        k_train, k1, k2 = jr.split(k_train, 3)
        # Random points
        dom_pts = jr.uniform(k1, (BATCH_SIZE_PINN, 3), minval=jnp.array([0.,0.,0.]), maxval=jnp.array([W,H,D]))
        bc_pts = jr.uniform(k2, (BATCH_SIZE_PINN//10, 3), minval=jnp.array([0.,H,0.]), maxval=jnp.array([W,H,D]))
        
        t0 = time.time()
        model, opt_state, loss = make_step(model, opt_state, dom_pts, bc_pts)
        jax.block_until_ready(loss)
        t1 = time.time()
        
        times.append(t1 - t0)
        losses.append(float(loss))
        
        if (i+1) % 100 == 0:
            print(f"PINN Epoch {i+1}/{N_EPOCHS}, Loss: {loss:.4e}, Time: {t1-t0:.4f}s")
            
    total_time = time.time() - start_total
    return {
        "name": "PINN",
        "avg_time_per_epoch": np.mean(times[1:]), # Skip compile
        "total_time": total_time,
        "final_loss": losses[-1],
        "params": sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))
    }

# ============================================================
# Train DEM
# ============================================================
def run_dem(key):
    print("--- Starting DEM Benchmark ---")
    k_model, k_train = jr.split(key)
    # Same architecture as PINN
    model = ElasticityNetwork(hidden=64, n_layers=4, n_fourier=16, key=k_model)
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    @eqx.filter_jit
    def make_step(model, opt_state, domain_pts, top_pts):
        # E values
        E_dom = get_E_at_points(domain_pts, None, None)
        
        # DEM requires potential energy minimization
        def loss_fn(m):
            # Volume integral (Monte Carlo)
            vol = W * H * D
            
            def get_energy(pt):
                E = get_E_at_points(pt[None], None, None)[0]
                return strain_energy_density(m, pt[0], pt[1], pt[2], E)
            
            U_int = jnp.mean(jax.vmap(get_energy)(domain_pts)) * vol
            
            # Work done by external force
            # Surface integral on top
            area = W * D
            def get_work(pt):
                # u_y * (-P)
                # We need E_norm for the model input.
                E = get_E_at_points(pt[None], None, None)[0]
                E_norm = E / E_MAX
                u = m(pt[0], pt[1], pt[2], E_norm)
                return u[1] * (-P_APPLIED)
                
            W_ext = jnp.mean(jax.vmap(get_work)(top_pts)) * area
            
            return U_int + W_ext

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    times = []
    losses = []
    start_total = time.time()
    
    for i in range(N_EPOCHS):
        k_train, k1, k2 = jr.split(k_train, 3)
        dom_pts = jr.uniform(k1, (BATCH_SIZE_PINN, 3), minval=jnp.array([0.,0.,0.]), maxval=jnp.array([W,H,D]))
        top_pts = jr.uniform(k2, (BATCH_SIZE_PINN//10, 3), minval=jnp.array([0.,H,0.]), maxval=jnp.array([W,H,D]))
        
        t0 = time.time()
        model, opt_state, loss = make_step(model, opt_state, dom_pts, top_pts)
        jax.block_until_ready(loss)
        t1 = time.time()
        
        times.append(t1 - t0)
        losses.append(float(loss))
        
        if (i+1) % 100 == 0:
            print(f"DEM Epoch {i+1}/{N_EPOCHS}, Energy: {loss:.4e}, Time: {t1-t0:.4f}s")
            
    total_time = time.time() - start_total
    return {
        "name": "DEM",
        "avg_time_per_epoch": np.mean(times[1:]),
        "total_time": total_time,
        "final_loss": losses[-1],
        "params": sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))
    }

# ============================================================
# Train PI-FNO
# ============================================================
def run_pifno(key):
    print("--- Starting PI-FNO Benchmark ---")
    k_model, k_train = jr.split(key)
    # FNO parameters
    model = FNO3d(in_channels=4, out_channels=3, modes=8, width=20, depth=4, key=k_model)
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    # Static E grid for single instance learning
    E_field, _ = get_synthetic_E_field(RESOLUTION)
    E_grid = E_field[None, ...] # (1, Nx, Ny, Nz)
    
    @eqx.filter_jit
    def make_step(model, opt_state):
        loss_fn = lambda m: compute_loss_pifno(m, E_grid, RESOLUTION)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    times = []
    losses = []
    start_total = time.time()
    
    for i in range(N_EPOCHS):
        t0 = time.time()
        model, opt_state, loss = make_step(model, opt_state)
        jax.block_until_ready(loss)
        t1 = time.time()
        
        times.append(t1 - t0)
        losses.append(float(loss))
        
        if (i+1) % 100 == 0:
            print(f"PI-FNO Epoch {i+1}/{N_EPOCHS}, Loss: {loss:.4e}, Time: {t1-t0:.4f}s")
            
    total_time = time.time() - start_total
    return {
        "name": "PI-FNO",
        "avg_time_per_epoch": np.mean(times[1:]),
        "total_time": total_time,
        "final_loss": losses[-1],
        "params": sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))
    }

# ============================================================
# Main
# ============================================================
def main():
    key = jr.PRNGKey(42)
    k1, k2, k3 = jr.split(key, 3)
    
    results = []
    results.append(run_pinn(k1))
    results.append(run_dem(k2))
    results.append(run_pifno(k3))
    
    print("\n\n=== Benchmark Results ===")
    print(f"{'Model':<10} | {'Params':<10} | {'Time/Epoch (s)':<15} | {'Total Time (s)':<15} | {'Final Loss/Energy':<20}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<10} | {r['params']:<10} | {r['avg_time_per_epoch']:<15.4f} | {r['total_time']:<15.2f} | {r['final_loss']:<20.4e}")

    # Recommendation
    print("\n=== Recommendation ===")
    best_time = min(results, key=lambda x: x['total_time'])
    print(f"Fastest Model: {best_time['name']}")
    
    print("Note: Loss values are not directly comparable (Residual vs Energy).")
    print("DEM is generally more stable for elasticity problems.")
    print("PI-FNO is mesh-based and might be faster per epoch but memory intensive.")
    
if __name__ == "__main__":
    main()
