import jax
import jax.numpy as jnp
import os


jax.config.update("jax_enable_x64", True)


def init_mlp_params(layer_sizes, key, scale=0.1):
    keys = jax.random.split(key, len(layer_sizes) - 1)
    params = []
    for k, (m, n) in zip(keys, zip(layer_sizes[:-1], layer_sizes[1:])):
        w_key, b_key = jax.random.split(k)
        w = scale * jax.random.normal(w_key, (n, m))
        b = jnp.zeros((n,))
        params.append((w, b))
    return params


def mlp_forward(params, x):
    h = x
    for w, b in params[:-1]:
        h = jnp.dot(w, h) + b
        h = jnp.tanh(h)
    w_last, b_last = params[-1]
    out = jnp.dot(w_last, h) + b_last
    return out[0]


def pinn_u(params, x, t):
    inp = jnp.array([x, t])
    return mlp_forward(params, inp)


def pde_residual(params, x, t, D, k, s0):
    def u_fun(x_, t_):
        return pinn_u(params, x_, t_)

    u_t = jax.grad(lambda tt: u_fun(x, tt))(t)
    u_x = jax.grad(lambda xx: u_fun(xx, t))(x)
    u_xx = jax.grad(lambda xx: jax.grad(lambda yy: u_fun(yy, t))(xx))(x)
    source = s0 * jnp.exp(-((x - 0.5) ** 2) / 0.01)
    return u_t - D * u_xx + k * u_fun(x, t) - source


def loss_pinn(params, xs, ts, xs_bc, ts_bc, xs_ic, D, k, s0, x_data, t_data, u_data):
    res_pde = jax.vmap(lambda x, t: pde_residual(params, x, t, D, k, s0))(xs, ts)
    loss_pde = jnp.mean(res_pde**2)
    u_bc = jax.vmap(lambda x, t: pinn_u(params, x, t))(xs_bc, ts_bc)
    loss_bc = jnp.mean(u_bc**2)
    u_ic = jax.vmap(lambda x: pinn_u(params, x, 0.0))(xs_ic)
    loss_ic = jnp.mean(u_ic**2)
    u_data_pred = jax.vmap(lambda x, t: pinn_u(params, x, t))(x_data, t_data)
    loss_data = jnp.mean((u_data_pred - u_data) ** 2)
    return loss_pde + 0.1 * loss_bc + 0.1 * loss_ic + loss_data


loss_pinn_grad = jax.jit(
    lambda p, xs, ts, xs_bc, ts_bc, xs_ic, D, k, s0, xd, td, ud: jax.grad(loss_pinn)(
        p, xs, ts, xs_bc, ts_bc, xs_ic, D, k, s0, xd, td, ud
    )
)


def main():
    key = jax.random.PRNGKey(0)
    layer_sizes = [2, 64, 64, 1]
    params = init_mlp_params(layer_sizes, key)
    D = 0.01
    k = 1.0
    s0 = 10.0
    n_colloc = 128
    x_colloc = jax.random.uniform(key, (n_colloc,), minval=0.0, maxval=1.0)
    t_colloc = jax.random.uniform(key, (n_colloc,), minval=0.0, maxval=1.0)
    n_bc = 32
    x_bc_left = jnp.zeros(n_bc // 2)
    x_bc_right = jnp.ones(n_bc // 2)
    x_bc = jnp.concatenate([x_bc_left, x_bc_right], axis=0)
    t_bc = jax.random.uniform(key, (n_bc,), minval=0.0, maxval=1.0)
    n_ic = 64
    x_ic = jnp.linspace(0.0, 1.0, n_ic)
    xs = x_colloc
    ts = t_colloc
    xs_bc = x_bc
    ts_bc = t_bc
    xs_ic = x_ic
    base = "_jax_rd_datasets"
    x_data_all = jnp.load(os.path.join(base, "x_1d.npy"))
    t_all = jnp.load(os.path.join(base, "t_1d.npy"))
    u_traj = jnp.load(os.path.join(base, "u_1d_traj.npy"))
    t_data = t_all[-1] * jnp.ones_like(x_data_all)
    u_data_all = u_traj[-1]
    x_data = x_data_all
    u_data = u_data_all
    lr = 1e-3
    params_current = params
    for i in range(20):
        loss_val = loss_pinn(
            params_current,
            xs,
            ts,
            xs_bc,
            ts_bc,
            xs_ic,
            D,
            k,
            s0,
            x_data,
            t_data,
            u_data,
        )
        grads = loss_pinn_grad(
            params_current,
            xs,
            ts,
            xs_bc,
            ts_bc,
            xs_ic,
            D,
            k,
            s0,
            x_data,
            t_data,
            u_data,
        )
        new_params = []
        for (w, b), (gw, gb) in zip(params_current, grads):
            new_w = w - lr * gw
            new_b = b - lr * gb
            new_params.append((new_w, new_b))
        params_current = new_params
    final_loss = loss_pinn(
        params_current,
        xs,
        ts,
        xs_bc,
        ts_bc,
        xs_ic,
        D,
        k,
        s0,
        x_data,
        t_data,
        u_data,
    )
    print("JAX PINN 1D reaction-diffusion demo")
    print(f"final loss = {float(final_loss):.6e}")


if __name__ == "__main__":
    main()
