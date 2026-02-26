import jax
import jax.numpy as jnp

from .core_hamilton_1d import (
    theta_to_matrices,
    reaction_step,
    make_initial_state,
)


jax.config.update("jax_enable_x64", True)


def make_initial_state_2d(Nx, Ny, active_mask):
    G1d = make_initial_state(Nx, active_mask)
    G2d = jnp.tile(G1d[:, jnp.newaxis, :], (1, Ny, 1))
    G_flat = G2d.reshape((Nx * Ny, 12))
    return G_flat


def diffusion_step_2d(G_flat, params):
    D_eff = params["D_eff"]
    dt_diff = params["dt_h"] * params["n_react_sub"]
    dx = params["dx"]
    dy = params["dy"]
    Nx = params["Nx"]
    Ny = params["Ny"]
    phi_flat = G_flat[:, 0:5]
    phi = phi_flat.reshape((Nx, Ny, 5))
    lap = jnp.zeros_like(phi)
    lap_x = (
        phi[0 : Nx - 2, 1 : Ny - 1, :]
        + phi[2:Nx, 1 : Ny - 1, :]
        - 2.0 * phi[1 : Nx - 1, 1 : Ny - 1, :]
    ) / (dx * dx)
    lap_y = (
        phi[1 : Nx - 1, 0 : Ny - 2, :]
        + phi[1 : Nx - 1, 2:Ny, :]
        - 2.0 * phi[1 : Nx - 1, 1 : Ny - 1, :]
    ) / (dy * dy)
    interior = lap_x + lap_y
    lap = lap.at[1 : Nx - 1, 1 : Ny - 1, :].set(interior)
    phi_new = phi + dt_diff * D_eff * lap
    phi_new = jnp.clip(phi_new, 0.0, 1.0)
    phi_sum = jnp.sum(phi_new, axis=2)
    phi_sum = jnp.minimum(phi_sum, 1.0)
    phi0_new = 1.0 - phi_sum
    G_new = G_flat.reshape((Nx, Ny, 12))
    G_new = G_new.at[:, :, 0:5].set(phi_new)
    G_new = G_new.at[:, :, 5].set(phi0_new)
    G_new_flat = G_new.reshape((Nx * Ny, 12))
    return G_new_flat


def simulate_hamilton_2d(
    theta,
    D_eff,
    n_macro=20,
    n_react_sub=10,
    Nx=20,
    Ny=20,
    Lx=1.0,
    Ly=1.0,
    dt_h=1e-5,
):
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    A, b_diag = theta_to_matrices(theta)
    active_mask = jnp.ones(5, dtype=jnp.int64)
    params = {
        "dt_h": dt_h,
        "Kp1": 1e-4,
        "Eta": jnp.ones(5),
        "EtaPhi": jnp.ones(5),
        "c": 100.0,
        "alpha": 100.0,
        "K_hill": 0.05,
        "n_hill": 4.0,
        "A": A,
        "b_diag": b_diag,
        "active_mask": active_mask,
        "n_react_sub": n_react_sub,
        "D_eff": D_eff,
        "dx": dx,
        "dy": dy,
        "Nx": Nx,
        "Ny": Ny,
    }
    G0_flat = make_initial_state_2d(Nx, Ny, active_mask)

    def body(carry, _):
        G_flat = carry
        G_flat = reaction_step(G_flat, params)
        G_flat = diffusion_step_2d(G_flat, params)
        return G_flat, G_flat

    _, G_traj = jax.lax.scan(body, G0_flat, jnp.arange(n_macro))
    G_all = jnp.concatenate([G0_flat[jnp.newaxis, :, :], G_traj], axis=0)
    return G_all
