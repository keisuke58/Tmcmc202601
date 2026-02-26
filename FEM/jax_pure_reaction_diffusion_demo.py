import jax
import jax.numpy as jnp


def solve_pde(D, n_steps=500, N=64, dt=1e-4, k=1.0, s0=10.0):
    x = jnp.linspace(0.0, 1.0, N)
    y = jnp.linspace(0.0, 1.0, N)
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    dx = x[1] - x[0]
    r2 = (X - 0.5) ** 2 + (Y - 0.5) ** 2
    source = s0 * jnp.exp(-r2 / 0.02)

    def laplacian(u):
        interior = (
            u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:] - 4.0 * u[1:-1, 1:-1]
        ) / (dx * dx)
        z = jnp.zeros_like(u)
        z = z.at[1:-1, 1:-1].set(interior)
        return z

    def step(u, _):
        lap = laplacian(u)
        du = D * lap - k * u + source
        return u + dt * du, None

    u0 = jnp.zeros((N, N))
    u_final, _ = jax.lax.scan(step, u0, jnp.arange(n_steps))
    return u_final, X, Y


def loss_fn(D):
    N = 64
    u, X, Y = solve_pde(D, n_steps=800, N=N)
    i1 = int(0.25 * (N - 1))
    j1 = int(0.25 * (N - 1))
    i2 = int(0.75 * (N - 1))
    j2 = int(0.75 * (N - 1))
    v1 = u[i1, j1]
    v2 = u[i2, j2]
    t1 = 0.2
    t2 = 0.1
    return (v1 - t1) ** 2 + (v2 - t2) ** 2


def main():
    D0 = 0.01
    loss_grad = jax.grad(loss_fn)
    loss_value = loss_fn(D0)
    grad_value = loss_grad(D0)
    u, X, Y = solve_pde(D0)
    u_mean = jnp.mean(u)
    u_max = jnp.max(u)
    print("Pure JAX reaction-diffusion demo")
    print(f"D = {float(D0):.6f}")
    print(f"loss(D) = {float(loss_value):.6f}")
    print(f"d loss / d D = {float(grad_value):.6f}")
    print(f"u_mean = {float(u_mean):.6f}")
    print(f"u_max = {float(u_max):.6f}")


if __name__ == "__main__":
    main()
