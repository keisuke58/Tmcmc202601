from .core_rd import solve_0d, solve_nd
import jax.numpy as jnp


def main():
    u0_final, t_arr, traj0 = solve_0d()
    print("JAXFEM 0D final u:", float(u0_final))
    u1_final, _ = solve_nd(dim=1, N=64)
    print(
        "JAXFEM 1D final u stats:",
        float(jnp.mean(u1_final)),
        float(jnp.max(u1_final)),
    )
    u2_final, _ = solve_nd(dim=2, N=32)
    print(
        "JAXFEM 2D final u stats:",
        float(jnp.mean(u2_final)),
        float(jnp.max(u2_final)),
    )
    u3_final, _ = solve_nd(dim=3, N=16)
    print(
        "JAXFEM 3D final u stats:",
        float(jnp.mean(u3_final)),
        float(jnp.max(u3_final)),
    )


if __name__ == "__main__":
    main()
