import jax
import jax.numpy as jnp

from .core_rd import solve_nd


jax.config.update("jax_enable_x64", True)


def solve_2d_with_params(params, N, T, dt):
    D = jnp.exp(params[0])
    k = jnp.exp(params[1])
    s0 = 10.0
    uT, traj = solve_nd(dim=2, N=N, T=T, dt=dt, D=D, k=k, s0=s0)
    return traj


def generate_data(N=32, T=0.05, dt=5e-5, D_true=0.01, k_true=1.0, noise_std=0.01):
    uT, traj_true = solve_nd(dim=2, N=N, T=T, dt=dt, D=D_true, k=k_true, s0=10.0)
    n_steps = traj_true.shape[0]
    idx = jnp.array([n_steps // 4, n_steps // 2, 3 * n_steps // 4, n_steps - 1])
    traj_sel = traj_true[idx]
    key = jax.random.PRNGKey(1)
    noise = noise_std * jax.random.normal(key, traj_sel.shape)
    u_obs = traj_sel + noise
    return traj_sel, u_obs, idx


def loss_params(params, u_obs, N, T, dt, idx):
    traj_sim = solve_2d_with_params(params, N, T, dt)
    traj_sel = traj_sim[idx]
    return jnp.mean((traj_sel - u_obs) ** 2)


def main():
    N = 32
    T = 0.05
    dt = 5e-5
    D_true = 0.01
    k_true = 1.0
    u_true, u_obs, idx = generate_data(
        N=N, T=T, dt=dt, D_true=D_true, k_true=k_true
    )
    params = jnp.array([jnp.log(0.02), jnp.log(1.0)])
    lr = 5e-3
    loss_fn = jax.jit(lambda p: loss_params(p, u_obs, N, T, dt, idx))
    loss_grad = jax.jit(lambda p: jax.grad(loss_params)(p, u_obs, N, T, dt, idx))
    for i in range(200):
        l = loss_fn(params)
        g = loss_grad(params)
        params = params - lr * g
    D_est = float(jnp.exp(params[0]))
    k_est = float(jnp.exp(params[1]))
    print("JAXFEM 2D reaction-diffusion parameter estimation")
    print(f"D_true = {D_true:.5f}, k_true = {k_true:.5f}")
    print(f"D_est  = {D_est:.5f}, k_est  = {k_est:.5f}")


if __name__ == "__main__":
    main()

