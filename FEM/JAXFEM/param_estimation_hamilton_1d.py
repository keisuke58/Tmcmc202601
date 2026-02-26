import jax
import jax.numpy as jnp

from .core_hamilton_1d import THETA_DEMO, simulate_hamilton_1d


jax.config.update("jax_enable_x64", True)


D_BASE = jnp.array([0.001, 0.001, 0.0008, 0.0005, 0.0002])


def summarize_traj(G_all, idx_time, species_idx=3):
    phi = G_all[:, :, 0:5]
    phi_spec = phi[:, :, species_idx]
    phi_mean = jnp.mean(phi_spec, axis=1)
    return phi_mean[idx_time]


def simulate_with_scale(log_scale, n_macro, n_react_sub, N):
    scale = jnp.exp(log_scale)
    D_eff = D_BASE * scale
    G_all = simulate_hamilton_1d(
        theta=THETA_DEMO,
        D_eff=D_eff,
        n_macro=n_macro,
        n_react_sub=n_react_sub,
        N=N,
        L=1.0,
        dt_h=1e-5,
    )
    return G_all


def generate_data(
    n_macro=20,
    n_react_sub=10,
    N=30,
    log_scale_true=0.0,
    noise_std=0.001,
):
    G_all_true = simulate_with_scale(log_scale_true, n_macro=n_macro, n_react_sub=n_react_sub, N=N)
    idx_time = jnp.array([n_macro // 4, n_macro // 2, 3 * n_macro // 4, n_macro])
    y_true = summarize_traj(G_all_true, idx_time, species_idx=3)
    key = jax.random.PRNGKey(0)
    noise = noise_std * jax.random.normal(key, y_true.shape)
    y_obs = y_true + noise
    return y_true, y_obs, idx_time


def loss_fn(log_scale, y_obs, idx_time, n_macro, n_react_sub, N):
    G_all = simulate_with_scale(log_scale, n_macro, n_react_sub, N)
    y_pred = summarize_traj(G_all, idx_time, species_idx=3)
    return jnp.mean((y_pred - y_obs) ** 2)


def main():
    n_macro = 20
    n_react_sub = 10
    N = 30
    log_scale_true = jnp.log(1.0)
    y_true, y_obs, idx_time = generate_data(
        n_macro=n_macro,
        n_react_sub=n_react_sub,
        N=N,
        log_scale_true=log_scale_true,
        noise_std=0.001,
    )
    log_scale = jnp.log(0.5)
    lr = 1e-2
    loss_jit = jax.jit(
        lambda p: loss_fn(p, y_obs, idx_time, n_macro=n_macro, n_react_sub=n_react_sub, N=N)
    )
    grad_jit = jax.jit(lambda p: jax.grad(loss_fn)(p, y_obs, idx_time, n_macro, n_react_sub, N))
    for i in range(30):
        l = loss_jit(log_scale)
        g = grad_jit(log_scale)
        log_scale = log_scale - lr * g
    D_true = float(jnp.exp(log_scale_true))
    D_est = float(jnp.exp(log_scale))
    print("Hamilton 1D effective diffusion scale estimation")
    print(f"D_scale_true = {D_true:.4f}")
    print(f"D_scale_est  = {D_est:.4f}")


if __name__ == "__main__":
    main()
