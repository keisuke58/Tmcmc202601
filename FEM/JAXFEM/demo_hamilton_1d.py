import jax.numpy as jnp

from .core_hamilton_1d import THETA_DEMO, simulate_hamilton_1d


def main():
    N = 30
    n_macro = 60
    D_eff = jnp.array([0.001, 0.001, 0.0008, 0.0005, 0.0002])
    G_all = simulate_hamilton_1d(
        theta=THETA_DEMO,
        D_eff=D_eff,
        n_macro=n_macro,
        n_react_sub=20,
        N=N,
        L=1.0,
        dt_h=1e-5,
    )
    G_final = G_all[-1]
    phi_mean = jnp.mean(G_final[:, 0:5], axis=0)
    print("JAXFEM 1D Hamilton-like 5-species demo")
    print(f"N = {N}")
    print(f"n_macro = {n_macro}")
    print("mean phi per species:", ", ".join(f"{float(v):.4f}" for v in phi_mean))


if __name__ == "__main__":
    main()
