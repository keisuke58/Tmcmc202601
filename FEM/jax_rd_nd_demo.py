import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def reaction_term(u, k):
    return -k * u


def source_0d(t, s0):
    return s0 * jnp.exp(-((t - 0.5) ** 2) / 0.02)


def solve_0d(T=1.0, dt=1e-3, k=1.0, s0=1.0):
    n_steps = int(T / dt)
    u0 = 0.0

    def step(u, i):
        t = dt * i
        du = reaction_term(u, k) + source_0d(t, s0)
        return u + dt * du, u + dt * du

    uT, traj = jax.lax.scan(step, u0, jnp.arange(n_steps))
    t_arr = dt * jnp.arange(n_steps)
    return uT, t_arr, traj


def laplacian_nd(u, dx):
    ndim = u.ndim
    if ndim == 1:
        interior = (u[:-2] + u[2:] - 2.0 * u[1:-1]) / (dx * dx)
        z = jnp.zeros_like(u)
        z = z.at[1:-1].set(interior)
        return z
    if ndim == 2:
        interior = (
            u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:] - 4.0 * u[1:-1, 1:-1]
        ) / (dx * dx)
        z = jnp.zeros_like(u)
        z = z.at[1:-1, 1:-1].set(interior)
        return z
    if ndim == 3:
        interior = (
            u[:-2, 1:-1, 1:-1]
            + u[2:, 1:-1, 1:-1]
            + u[1:-1, :-2, 1:-1]
            + u[1:-1, 2:, 1:-1]
            + u[1:-1, 1:-1, :-2]
            + u[1:-1, 1:-1, 2:]
            - 6.0 * u[1:-1, 1:-1, 1:-1]
        ) / (dx * dx)
        z = jnp.zeros_like(u)
        z = z.at[1:-1, 1:-1, 1:-1].set(interior)
        return z
    raise ValueError("Unsupported ndim")


def source_nd(coords, s0):
    r2 = jnp.sum((coords - 0.5) ** 2)
    return s0 * jnp.exp(-r2 / 0.02)


def solve_nd(dim, N=32, T=0.1, dt=1e-4, D=0.01, k=1.0, s0=10.0):
    if dim == 1:
        x = jnp.linspace(0.0, 1.0, N)
        dx = x[1] - x[0]
        u0 = jnp.zeros((N,))

        def step(u, _):
            coords = x
            src = jax.vmap(lambda xx: source_nd(jnp.array([xx]), s0))(coords)
            lap = laplacian_nd(u, dx)
            du = D * lap + reaction_term(u, k) + src
            u_next = u + dt * du
            u_next = u_next.at[0].set(0.0).at[-1].set(0.0)
            return u_next, u_next

        n_steps = int(T / dt)
        uT, traj = jax.lax.scan(step, u0, jnp.arange(n_steps))
        return uT, traj

    if dim == 2:
        x = jnp.linspace(0.0, 1.0, N)
        y = jnp.linspace(0.0, 1.0, N)
        dx = x[1] - x[0]
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        u0 = jnp.zeros((N, N))

        def step(u, _):
            coords = jnp.stack([X, Y], axis=-1)
            src = jax.vmap(lambda row: jax.vmap(lambda p: source_nd(p, s0))(row))(coords)
            lap = laplacian_nd(u, dx)
            du = D * lap + reaction_term(u, k) + src
            u_next = u + dt * du
            u_next = (
                u_next.at[0, :].set(0.0).at[-1, :].set(0.0).at[:, 0].set(0.0).at[:, -1].set(0.0)
            )
            return u_next, u_next

        n_steps = int(T / dt)
        uT, traj = jax.lax.scan(step, u0, jnp.arange(n_steps))
        return uT, traj

    if dim == 3:
        x = jnp.linspace(0.0, 1.0, N)
        y = jnp.linspace(0.0, 1.0, N)
        z = jnp.linspace(0.0, 1.0, N)
        dx = x[1] - x[0]
        X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
        u0 = jnp.zeros((N, N, N))

        def step(u, _):
            coords = jnp.stack([X, Y, Z], axis=-1)
            src = jax.vmap(
                lambda plane: jax.vmap(lambda row: jax.vmap(lambda p: source_nd(p, s0))(row))(plane)
            )(coords)
            lap = laplacian_nd(u, dx)
            du = D * lap + reaction_term(u, k) + src
            u_next = u + dt * du
            u_next = (
                u_next.at[0, :, :]
                .set(0.0)
                .at[-1, :, :]
                .set(0.0)
                .at[:, 0, :]
                .set(0.0)
                .at[:, -1, :]
                .set(0.0)
                .at[:, :, 0]
                .set(0.0)
                .at[:, :, -1]
                .set(0.0)
            )
            return u_next, u_next

        n_steps = int(T / dt)
        uT, traj = jax.lax.scan(step, u0, jnp.arange(n_steps))
        return uT, traj

    raise ValueError("dim must be 1,2,3")


def main():
    u0_final, t_arr, traj0 = solve_0d()
    print("0D final u:", float(u0_final))
    u1_final, _ = solve_nd(dim=1, N=64)
    print("1D final u stats:", float(jnp.mean(u1_final)), float(jnp.max(u1_final)))
    u2_final, _ = solve_nd(dim=2, N=32)
    print("2D final u stats:", float(jnp.mean(u2_final)), float(jnp.max(u2_final)))
    u3_final, _ = solve_nd(dim=3, N=16)
    print("3D final u stats:", float(jnp.mean(u3_final)), float(jnp.max(u3_final)))


if __name__ == "__main__":
    main()
