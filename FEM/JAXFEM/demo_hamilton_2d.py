import jax.numpy as jnp

from .core_hamilton_1d import THETA_DEMO
from .core_hamilton_2d import simulate_hamilton_2d


def main():
    Nx = 20
    Ny = 20
    n_macro = 20
    D_eff = jnp.array([0.001, 0.001, 0.0008, 0.0005, 0.0002])
    G_all = simulate_hamilton_2d(
        theta=THETA_DEMO,
        D_eff=D_eff,
        n_macro=n_macro,
        n_react_sub=10,
        Nx=Nx,
        Ny=Ny,
        Lx=1.0,
        Ly=1.0,
        dt_h=1e-5,
    )
    G_final = G_all[-1]
    phi = G_final[:, 0:5]
    phi_mean = jnp.mean(phi, axis=0)
    print("JAXFEM 2D Hamilton-like 5-species demo")
    print(f"Nx = {Nx}, Ny = {Ny}")
    print(f"n_macro = {n_macro}")
    print("mean phi per species:", ", ".join(f"{float(v):.4f}" for v in phi_mean))


if __name__ == "__main__":
    main()
