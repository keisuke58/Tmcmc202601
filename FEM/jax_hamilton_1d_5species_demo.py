import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


THETA_DEMO = jnp.array(
    [
        1.34,
        -0.18,
        1.79,
        1.17,
        2.58,
        3.51,
        2.73,
        0.71,
        2.1,
        0.37,
        2.05,
        -0.15,
        3.56,
        0.16,
        0.12,
        0.32,
        1.49,
        2.1,
        2.41,
        2.5,
    ]
)


def theta_to_matrices(theta):
    A = jnp.zeros((5, 5))
    b = jnp.zeros(5)
    A = A.at[0, 0].set(theta[0])
    A = A.at[0, 1].set(theta[1])
    A = A.at[1, 0].set(theta[1])
    A = A.at[1, 1].set(theta[2])
    b = b.at[0].set(theta[3])
    b = b.at[1].set(theta[4])
    A = A.at[2, 2].set(theta[5])
    A = A.at[2, 3].set(theta[6])
    A = A.at[3, 2].set(theta[6])
    A = A.at[3, 3].set(theta[7])
    b = b.at[2].set(theta[8])
    b = b.at[3].set(theta[9])
    A = A.at[0, 2].set(theta[10])
    A = A.at[2, 0].set(theta[10])
    A = A.at[0, 3].set(theta[11])
    A = A.at[3, 0].set(theta[11])
    A = A.at[1, 2].set(theta[12])
    A = A.at[2, 1].set(theta[12])
    A = A.at[1, 3].set(theta[13])
    A = A.at[3, 1].set(theta[13])
    A = A.at[4, 4].set(theta[14])
    b = b.at[4].set(theta[15])
    A = A.at[0, 4].set(theta[16])
    A = A.at[4, 0].set(theta[16])
    A = A.at[1, 4].set(theta[17])
    A = A.at[4, 1].set(theta[17])
    A = A.at[2, 4].set(theta[18])
    A = A.at[4, 2].set(theta[18])
    A = A.at[3, 4].set(theta[19])
    A = A.at[4, 3].set(theta[19])
    return A, b


def residual(g_new, g_prev, params):
    dt = params["dt_h"]
    Kp1 = params["Kp1"]
    Eta = params["Eta"]
    EtaPhi = params["EtaPhi"]
    c = params["c"]
    alpha = params["alpha"]
    K_hill = params["K_hill"]
    n_hill = params["n_hill"]
    A = params["A"]
    b_diag = params["b_diag"]
    active_mask = params["active_mask"]
    eps = 1e-12
    phi_new = g_new[0:5]
    phi0_new = g_new[5]
    psi_new = g_new[6:11]
    gamma_new = g_new[11]
    phi_old = g_prev[0:5]
    phi0_old = g_prev[5]
    psi_old = g_prev[6:11]
    phidot = (phi_new - phi_old) / dt
    phi0dot = (phi0_new - phi0_old) / dt
    psidot = (psi_new - psi_old) / dt
    Ia = A @ (phi_new * psi_new)
    hill_mask = (K_hill > 1e-9).astype(jnp.float64) * (active_mask[4] == 1).astype(jnp.float64)
    fn = jnp.maximum(phi_new[3] * psi_new[3], 0.0)
    num = fn**n_hill
    den = K_hill**n_hill + num
    factor = jnp.where(den > eps, num / den, 0.0) * hill_mask
    Ia = Ia.at[4].set(Ia[4] * factor)
    Q = jnp.zeros(12, dtype=jnp.float64)

    def body_i_phi(carry, i):
        Q_local = carry
        active = active_mask[i] == 1

        def active_branch():
            t1 = Kp1 * (2.0 - 4.0 * phi_new[i]) / ((phi_new[i] - 1.0) ** 3 * phi_new[i] ** 3)
            t2 = (1.0 / Eta[i]) * (
                gamma_new
                + (EtaPhi[i] + Eta[i] * psi_new[i] ** 2) * phidot[i]
                + Eta[i] * phi_new[i] * psi_new[i] * psidot[i]
            )
            t3 = (c / Eta[i]) * psi_new[i] * Ia[i]
            return Q_local.at[i].set(t1 + t2 - t3)

        def inactive_branch():
            return Q_local.at[i].set(phi_new[i])

        return jax.lax.cond(active, active_branch, inactive_branch), None

    Q, _ = jax.lax.scan(body_i_phi, Q, jnp.arange(5))
    Q = Q.at[5].set(
        gamma_new + Kp1 * (2.0 - 4.0 * phi0_new) / ((phi0_new - 1.0) ** 3 * phi0_new**3) + phi0dot
    )

    def body_i_psi(carry, i):
        Q_local = carry
        active = active_mask[i] == 1

        def active_branch():
            t1 = (-2.0 * Kp1) / ((psi_new[i] - 1.0) ** 2 * psi_new[i] ** 3) - (2.0 * Kp1) / (
                (psi_new[i] - 1.0) ** 3 * psi_new[i] ** 2
            )
            t2 = (b_diag[i] * alpha / Eta[i]) * psi_new[i]
            t3 = phi_new[i] * psi_new[i] * phidot[i] + phi_new[i] ** 2 * psidot[i]
            t4 = (c / Eta[i]) * phi_new[i] * Ia[i]
            return Q_local.at[6 + i].set(t1 + t2 + t3 - t4)

        def inactive_branch():
            return Q_local.at[6 + i].set(psi_new[i])

        return jax.lax.cond(active, active_branch, inactive_branch), None

    Q, _ = jax.lax.scan(body_i_psi, Q, jnp.arange(5))
    Q = Q.at[11].set(jnp.sum(phi_new) + phi0_new - 1.0)
    return Q


def clip_state(g, active_mask):
    eps = 1e-10
    phi = g[0:5]
    phi0 = g[5]
    psi = g[6:11]
    gamma = g[11]
    mask = active_mask.astype(jnp.float64)
    phi_clipped = jnp.clip(phi, eps, 1.0 - eps)
    psi_clipped = jnp.clip(psi, eps, 1.0 - eps)
    phi = mask * phi_clipped
    psi = mask * psi_clipped
    phi0 = jnp.clip(phi0, eps, 1.0 - eps)
    gamma = jnp.clip(gamma, -1e6, 1e6)
    g_new = jnp.zeros_like(g)
    g_new = g_new.at[0:5].set(phi)
    g_new = g_new.at[5].set(phi0)
    g_new = g_new.at[6:11].set(psi)
    g_new = g_new.at[11].set(gamma)
    return g_new


def newton_step(g_prev, params):
    active_mask = params["active_mask"]
    n_steps = 6

    def body(carry, _):
        g = carry
        g = clip_state(g, active_mask)

        def F(gg):
            return residual(gg, g_prev, params)

        Q = F(g)
        J = jax.jacfwd(F)(g)
        delta = jnp.linalg.solve(J, -Q)
        g_next = g + delta
        g_next = clip_state(g_next, active_mask)
        return g_next, None

    g0 = clip_state(g_prev, active_mask)
    g_final, _ = jax.lax.scan(body, g0, jnp.arange(n_steps))
    return g_final


newton_step_vmap = jax.jit(jax.vmap(newton_step, in_axes=(0, None)))


def reaction_step(G, params):
    n_sub = params["n_react_sub"]

    def body(carry, _):
        return newton_step_vmap(carry, params), None

    G_final, _ = jax.lax.scan(body, G, jnp.arange(n_sub))
    return G_final


def diffusion_step(G, params):
    D_eff = params["D_eff"]
    dt_diff = params["dt_h"] * params["n_react_sub"]
    dx = params["dx"]
    phi = G[:, 0:5]
    N = phi.shape[0]
    lap = jnp.zeros_like(phi)
    interior = (phi[0 : N - 2, :] + phi[2:N, :] - 2.0 * phi[1 : N - 1, :]) / (dx * dx)
    lap = lap.at[1 : N - 1, :].set(interior)
    phi_new = phi + dt_diff * D_eff * lap
    phi_new = jnp.clip(phi_new, 0.0, 1.0)
    phi_sum = jnp.sum(phi_new, axis=1)
    phi_sum = jnp.minimum(phi_sum, 1.0)
    phi0_new = 1.0 - phi_sum
    G_new = G.at[:, 0:5].set(phi_new)
    G_new = G_new.at[:, 5].set(phi0_new)
    return G_new


def make_initial_state(N, active_mask):
    x = jnp.linspace(0.0, 1.0, N)
    G = jnp.zeros((N, 12), dtype=jnp.float64)
    phi_base = jnp.array([0.12, 0.12, 0.08, 0.05, 0.0])

    def init_node(xk):
        phi = phi_base
        fn_val = 0.05 + 0.1 * jnp.exp(-15.0 * xk)
        pg_val = 0.02 * jnp.exp(-30.0 * xk)
        phi = phi.at[3].set(fn_val)
        phi = phi.at[4].set(pg_val)
        for i in range(5):
            if active_mask[i] == 0:
                phi = phi.at[i].set(0.0)
        phi_sum = jnp.sum(phi)
        phi_sum = jnp.minimum(phi_sum, 0.999999)
        phi = phi * (0.999999 / phi_sum)
        phi0 = 1.0 - jnp.sum(phi)
        psi = jnp.where(active_mask == 1, 0.999, 0.0)
        g = jnp.zeros(12, dtype=jnp.float64)
        g = g.at[0:5].set(phi)
        g = g.at[5].set(phi0)
        g = g.at[6:11].set(psi)
        g = g.at[11].set(0.0)
        return g

    G = jax.vmap(init_node)(x)
    return G


def run_simulation():
    N = 30
    L = 1.0
    dx = L / (N - 1)
    dt_h = 1e-5
    n_react_sub = 20
    n_macro = 60
    D_eff = jnp.array([0.001, 0.001, 0.0008, 0.0005, 0.0002])
    A, b_diag = theta_to_matrices(THETA_DEMO)
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
        "newton_steps": 6,
        "n_react_sub": n_react_sub,
        "D_eff": D_eff,
        "dx": dx,
    }
    G = make_initial_state(N, active_mask)
    for step in range(n_macro):
        G = reaction_step(G, params)
        G = diffusion_step(G, params)
    phi_mean = jnp.mean(G[:, 0:5], axis=0)
    print("JAX 1D Hamilton-like 5-species demo")
    print(f"N = {N}, dx = {dx:.5f}")
    print(f"dt_h = {dt_h:.1e}, n_react_sub = {n_react_sub}, n_macro = {n_macro}")
    print("mean phi per species:", ", ".join(f"{float(v):.4f}" for v in phi_mean))


def main():
    run_simulation()


if __name__ == "__main__":
    main()
