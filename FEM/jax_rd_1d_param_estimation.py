import jax
import jax.numpy as jnp

from jax_rd_nd_demo import solve_nd

jax.config.update("jax_enable_x64", True)


def solve_1d_with_params(params, N, T, dt):
    D = jax.nn.softplus(params[0])
    k = jax.nn.softplus(params[1])
    s0 = 10.0
    uT, _ = solve_nd(dim=1, N=N, T=T, dt=dt, D=D, k=k, s0=s0)
    return uT


def generate_data(N=64, T=0.1, dt=1e-4, D_true=0.01, k_true=1.0, noise_std=0.01):
    u_true, _ = solve_nd(dim=1, N=N, T=T, dt=dt, D=D_true, k=k_true, s0=10.0)
    key = jax.random.PRNGKey(0)
    noise = noise_std * jax.random.normal(key, u_true.shape)
    u_obs = u_true + noise
    return u_true, u_obs


def loss_params(params, u_obs, N, T, dt):
    u_sim = solve_1d_with_params(params, N, T, dt)
    return jnp.mean((u_sim - u_obs) ** 2)


def main():
    N = 64
    T = 0.1
    dt = 1e-4
    D_true = 0.01
    k_true = 1.0
    u_true, u_obs = generate_data(N=N, T=T, dt=dt, D_true=D_true, k_true=k_true)
    params = jnp.array([0.005, 0.5])
    lr = 1e-2
    loss_fn = jax.jit(lambda p: loss_params(p, u_obs, N, T, dt))
    loss_grad = jax.jit(lambda p: jax.grad(loss_params)(p, u_obs, N, T, dt))
    for i in range(50):
        l = loss_fn(params)
        g = loss_grad(params)
        params = params - lr * g
    D_est = float(jax.nn.softplus(params[0]))
    k_est = float(jax.nn.softplus(params[1]))
    print("1D reaction-diffusion parameter estimation demo")
    print(f"D_true = {D_true:.5f}, k_true = {k_true:.5f}")
    print(f"D_est  = {D_est:.5f}, k_est  = {k_est:.5f}")


if __name__ == "__main__":
    main()
