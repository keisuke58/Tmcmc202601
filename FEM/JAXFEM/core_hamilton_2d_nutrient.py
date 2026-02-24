"""
core_hamilton_2d_nutrient.py
============================
Pure-JAX 2D Hamilton 5-species biofilm solver with nutrient PDE coupling.

Grid    : (Nx, Ny) uniform nodes on [0, Lx] x [0, Ly]
State   : G (Nx*Ny, 12)  Hamilton state [phi_1..5, phi_0, psi_1..5, gamma]
Nutrient: c (Nx, Ny)     nutrient concentration field

Method  : Lie operator splitting per macro step
  (1) Reaction  -- vmap of 0D Newton step over all spatial nodes
  (2) Species diffusion -- explicit 2D Laplacian per species (Neumann BCs)
  (3) Nutrient PDE -- backward-Euler diffusion + Monod consumption

The nutrient concentration c modulates the interaction strength via a
Monod factor c/(k_M + c), linking the macro-scale nutrient field to
micro-scale Hamilton dynamics.

Usage
-----
    from JAXFEM.core_hamilton_2d_nutrient import run_simulation, Config2D
    cfg = Config2D(Nx=20, Ny=20, n_macro=100)
    result = run_simulation(theta, cfg)
"""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class Config2D:
    """Simulation parameters for 2D Hamilton + nutrient coupling."""

    def __init__(
        self,
        Nx: int = 20,
        Ny: int = 20,
        Lx: float = 1.0,
        Ly: float = 1.0,
        dt_h: float = 1e-5,
        n_react_sub: int = 20,
        n_macro: int = 60,
        save_every: int = 10,
        # Species diffusion (S.o, A.n, Vei, Fn, Pg)
        D_eff: np.ndarray | None = None,
        # Nutrient PDE
        D_c: float = 0.01,
        k_monod: float = 1.0,
        g_consumption: np.ndarray | None = None,
        c_boundary: float = 1.0,
        # Hamilton physics
        Kp1: float = 1e-4,
        c_hamilton: float = 100.0,
        alpha: float = 100.0,
        K_hill: float = 0.05,
        n_hill: float = 4.0,
        newton_iters: int = 6,
    ):
        self.Nx = Nx
        self.Ny = Ny
        self.Lx = Lx
        self.Ly = Ly
        self.dx = Lx / max(Nx - 1, 1)
        self.dy = Ly / max(Ny - 1, 1)
        self.dt_h = dt_h
        self.n_react_sub = n_react_sub
        self.n_macro = n_macro
        self.save_every = save_every

        self.D_eff = (
            np.array([1e-3, 1e-3, 8e-4, 5e-4, 2e-4])
            if D_eff is None else np.asarray(D_eff)
        )
        self.D_c = D_c
        self.k_monod = k_monod
        self.g_consumption = (
            np.array([1.0, 1.0, 0.8, 0.5, 0.3])
            if g_consumption is None else np.asarray(g_consumption)
        )
        self.c_boundary = c_boundary

        self.Kp1 = Kp1
        self.c_hamilton = c_hamilton
        self.alpha = alpha
        self.K_hill = K_hill
        self.n_hill = n_hill
        self.newton_iters = newton_iters

        # Derived
        self.dt_macro = dt_h * n_react_sub


# ---------------------------------------------------------------------------
# theta -> A, b
# ---------------------------------------------------------------------------

def theta_to_matrices(theta):
    """Convert 20-vector theta to interaction matrix A (5x5) and b_diag (5,)."""
    A = jnp.zeros((5, 5))
    b = jnp.zeros(5)
    # M1: S.oralis, A.naeslundii
    A = A.at[0, 0].set(theta[0])
    A = A.at[0, 1].set(theta[1]);  A = A.at[1, 0].set(theta[1])
    A = A.at[1, 1].set(theta[2])
    b = b.at[0].set(theta[3]);     b = b.at[1].set(theta[4])
    # M2: Veillonella, F.nucleatum
    A = A.at[2, 2].set(theta[5])
    A = A.at[2, 3].set(theta[6]);  A = A.at[3, 2].set(theta[6])
    A = A.at[3, 3].set(theta[7])
    b = b.at[2].set(theta[8]);     b = b.at[3].set(theta[9])
    # M3: cross-species
    A = A.at[0, 2].set(theta[10]); A = A.at[2, 0].set(theta[10])
    A = A.at[0, 3].set(theta[11]); A = A.at[3, 0].set(theta[11])
    A = A.at[1, 2].set(theta[12]); A = A.at[2, 1].set(theta[12])
    A = A.at[1, 3].set(theta[13]); A = A.at[3, 1].set(theta[13])
    # P.gingivalis self
    A = A.at[4, 4].set(theta[14])
    b = b.at[4].set(theta[15])
    # P.gingivalis cross
    A = A.at[0, 4].set(theta[16]); A = A.at[4, 0].set(theta[16])
    A = A.at[1, 4].set(theta[17]); A = A.at[4, 1].set(theta[17])
    A = A.at[2, 4].set(theta[18]); A = A.at[4, 2].set(theta[18])
    A = A.at[3, 4].set(theta[19]); A = A.at[4, 3].set(theta[19])
    return A, b


# ---------------------------------------------------------------------------
# 0D Hamilton residual & Newton step (reused from jax_hamilton_0d_5species_demo)
# ---------------------------------------------------------------------------

def clip_state(g, active_mask):
    eps = 1e-10
    phi = g[0:5]
    phi0 = g[5]
    psi = g[6:11]
    gamma = g[11]
    mask = active_mask.astype(jnp.float64)
    phi = mask * jnp.clip(phi, eps, 1.0 - eps)
    psi = mask * jnp.clip(psi, eps, 1.0 - eps)
    phi0 = jnp.clip(phi0, eps, 1.0 - eps)
    gamma = jnp.clip(gamma, -1e6, 1e6)
    g_new = jnp.zeros_like(g)
    g_new = g_new.at[0:5].set(phi)
    g_new = g_new.at[5].set(phi0)
    g_new = g_new.at[6:11].set(psi)
    g_new = g_new.at[11].set(gamma)
    return g_new


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

    # Hill gate for P.gingivalis (species 4), gated by F.nucleatum (species 3)
    hill_mask = (K_hill > 1e-9).astype(jnp.float64) * (active_mask[4] == 1).astype(
        jnp.float64
    )
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
            t1 = Kp1 * (2.0 - 4.0 * phi_new[i]) / (
                (phi_new[i] - 1.0) ** 3 * phi_new[i] ** 3
            )
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
        gamma_new
        + Kp1 * (2.0 - 4.0 * phi0_new) / ((phi0_new - 1.0) ** 3 * phi0_new ** 3)
        + phi0dot
    )

    def body_i_psi(carry, i):
        Q_local = carry
        active = active_mask[i] == 1
        def active_branch():
            t1 = (-2.0 * Kp1) / (
                (psi_new[i] - 1.0) ** 2 * psi_new[i] ** 3
            ) - (2.0 * Kp1) / (
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


def _make_newton_step_vmap(n_iters):
    """Create jitted vmapped Newton step with fixed iteration count.

    n_iters must be a compile-time constant (Python int), not a traced value.
    Different n_iters values produce separate compiled functions.
    """
    def newton_step(g_prev, params):
        active_mask = params["active_mask"]

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
        g_final, _ = jax.lax.scan(body, g0, jnp.arange(n_iters))
        return g_final

    return jax.jit(jax.vmap(newton_step, in_axes=(0, None)))


def _make_reaction_step(n_sub, n_iters):
    """Create reaction step with fixed sub-step and Newton iteration counts."""
    _newton_vmap = _make_newton_step_vmap(n_iters)

    def reaction_step(G, params):
        def body(carry, _):
            return _newton_vmap(carry, params), None
        G_final, _ = jax.lax.scan(body, G, jnp.arange(n_sub))
        return G_final

    return reaction_step


# ---------------------------------------------------------------------------
# 2D Laplacian (Neumann BCs)
# ---------------------------------------------------------------------------

def laplacian_2d_neumann(u, dx, dy):
    """
    2D 5-point Laplacian with Neumann (zero-flux) BCs on all walls.
    u : (Nx, Ny)  scalar field
    Returns lap : (Nx, Ny)
    """
    Nx, Ny = u.shape

    # x-direction: d^2u/dx^2
    lap_x = jnp.zeros_like(u)
    # Interior
    lap_x = lap_x.at[1:-1, :].set(
        (u[:-2, :] + u[2:, :] - 2.0 * u[1:-1, :]) / (dx * dx)
    )
    # Neumann at x=0: ghost u[-1] = u[0], so d2u/dx2 = (u[1] - u[0]) / dx^2
    lap_x = lap_x.at[0, :].set((u[1, :] - u[0, :]) / (dx * dx))
    # Neumann at x=Lx:
    lap_x = lap_x.at[-1, :].set((u[-2, :] - u[-1, :]) / (dx * dx))

    # y-direction: d^2u/dy^2
    lap_y = jnp.zeros_like(u)
    lap_y = lap_y.at[:, 1:-1].set(
        (u[:, :-2] + u[:, 2:] - 2.0 * u[:, 1:-1]) / (dy * dy)
    )
    lap_y = lap_y.at[:, 0].set((u[:, 1] - u[:, 0]) / (dy * dy))
    lap_y = lap_y.at[:, -1].set((u[:, -2] - u[:, -1]) / (dy * dy))

    return lap_x + lap_y


def laplacian_2d_dirichlet(u, dx, dy, c_bc):
    """
    2D 5-point Laplacian with Dirichlet BCs: u = c_bc on all walls.
    Used for the nutrient field c.
    """
    Nx, Ny = u.shape
    # Pad with boundary values
    u_pad = jnp.pad(u, 1, mode="constant", constant_values=c_bc)
    lap = (
        u_pad[:-2, 1:-1] + u_pad[2:, 1:-1] - 2.0 * u_pad[1:-1, 1:-1]
    ) / (dx * dx) + (
        u_pad[1:-1, :-2] + u_pad[1:-1, 2:] - 2.0 * u_pad[1:-1, 1:-1]
    ) / (dy * dy)
    return lap


# ---------------------------------------------------------------------------
# Species diffusion step (explicit Euler, Neumann BCs)
# ---------------------------------------------------------------------------

def diffusion_step_species(G, D_eff, dt_diff, dx, dy):
    """
    Explicit forward-Euler diffusion for 5 species volume fractions.
    G : (N_nodes, 12)
    Returns updated G with diffused phi.
    """
    Nx_Ny = G.shape[0]
    # We need (Nx, Ny) but it's passed flat — caller reshapes before calling
    raise NotImplementedError("Use diffusion_step_species_2d instead")


def diffusion_step_species_2d(phi_2d, D_eff, dt_diff, dx, dy):
    """
    Explicit diffusion for 5 species on 2D grid.
    phi_2d : (Nx, Ny, 5) volume fractions
    Returns updated phi_2d.
    """
    def diffuse_one(phi_i, D_i):
        lap = laplacian_2d_neumann(phi_i, dx, dy)
        return phi_i + dt_diff * D_i * lap

    phi_new = jnp.stack(
        [diffuse_one(phi_2d[:, :, i], D_eff[i]) for i in range(5)],
        axis=-1,
    )
    phi_new = jnp.clip(phi_new, 0.0, 1.0)
    # Enforce sum constraint
    phi_sum = jnp.sum(phi_new, axis=-1)
    scale = jnp.where(phi_sum > 1.0, 1.0 / phi_sum, 1.0)
    phi_new = phi_new * scale[:, :, None]
    return phi_new


# ---------------------------------------------------------------------------
# Nutrient PDE step (semi-implicit: backward-Euler diffusion, explicit consumption)
# ---------------------------------------------------------------------------

def nutrient_step(c, phi_2d, cfg, dt_nutrient):
    """
    One time step for nutrient field c(x,y).

    dc/dt = D_c * lap(c) - sum_i g_i * phi_i * c / (k_M + c)

    Uses explicit Euler for simplicity.  Dirichlet BC: c = c_boundary on all walls.
    """
    D_c = cfg.D_c
    k_M = cfg.k_monod
    g_cons = jnp.array(cfg.g_consumption)
    c_bc = cfg.c_boundary

    # Diffusion (Dirichlet BCs)
    lap_c = laplacian_2d_dirichlet(c, cfg.dx, cfg.dy, c_bc)

    # Monod consumption: sum_i g_i phi_i c/(k+c)
    phi_total_weighted = jnp.sum(phi_2d * g_cons[None, None, :], axis=-1)  # (Nx, Ny)
    monod = c / (k_M + c)
    consumption = phi_total_weighted * monod

    # Update
    c_new = c + dt_nutrient * (D_c * lap_c - consumption)
    c_new = jnp.clip(c_new, 0.0, c_bc)
    return c_new


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------

def make_initial_state_2d(cfg):
    """Build initial Hamilton state G (Nx*Ny, 12) and nutrient c (Nx, Ny)."""
    Nx, Ny = cfg.Nx, cfg.Ny
    x = jnp.linspace(0.0, cfg.Lx, Nx)
    y = jnp.linspace(0.0, cfg.Ly, Ny)
    active_mask = jnp.ones(5, dtype=jnp.int64)

    G_2d = jnp.zeros((Nx, Ny, 12), dtype=jnp.float64)

    # Commensals: uniform
    phi_base = jnp.array([0.12, 0.12, 0.08, 0.05, 0.02])

    # F.nucleatum: enriched near substrate (x=0)
    xp_fn = jnp.exp(-3.0 * x / cfg.Lx)
    fn_2d = 0.05 * xp_fn[:, None] * jnp.ones((1, Ny))

    # P.gingivalis: focal seed near substrate centre
    yc = 0.5 * cfg.Ly
    ys = 0.15 * cfg.Ly
    xp_pg = jnp.exp(-5.0 * x / cfg.Lx)
    yp_pg = jnp.exp(-0.5 * ((y - yc) / ys) ** 2)
    pg_2d = 0.01 * xp_pg[:, None] * yp_pg[None, :]

    # Assemble
    G_2d = G_2d.at[:, :, 0].set(phi_base[0])  # S.oralis
    G_2d = G_2d.at[:, :, 1].set(phi_base[1])  # A.naeslundii
    G_2d = G_2d.at[:, :, 2].set(phi_base[2])  # Veillonella
    G_2d = G_2d.at[:, :, 3].set(fn_2d)        # F.nucleatum
    G_2d = G_2d.at[:, :, 4].set(pg_2d)        # P.gingivalis

    # Normalise: sum(phi) < 1
    phi_sum = jnp.sum(G_2d[:, :, :5], axis=-1)
    scale = jnp.where(phi_sum > 0.999, 0.999 / phi_sum, 1.0)
    G_2d = G_2d.at[:, :, :5].set(G_2d[:, :, :5] * scale[:, :, None])

    # phi_0 = 1 - sum(phi)
    G_2d = G_2d.at[:, :, 5].set(1.0 - jnp.sum(G_2d[:, :, :5], axis=-1))

    # psi = phi initial (nutrient order parameter)
    G_2d = G_2d.at[:, :, 6:11].set(
        jnp.where(active_mask[None, None, :] == 1, 0.999, 0.0)
    )

    # gamma (Lagrange multiplier)
    G_2d = G_2d.at[:, :, 11].set(0.0)

    # Flatten for Newton vmap: (Nx*Ny, 12)
    G_flat = G_2d.reshape(Nx * Ny, 12)

    # Nutrient field: initially uniform at boundary value
    c = jnp.ones((Nx, Ny), dtype=jnp.float64) * cfg.c_boundary

    return G_flat, c


# ---------------------------------------------------------------------------
# Full simulation
# ---------------------------------------------------------------------------

def run_simulation(theta, cfg):
    """
    Run 2D Hamilton + nutrient coupling.

    Parameters
    ----------
    theta : array (20,)  TMCMC parameter vector
    cfg   : Config2D     simulation configuration

    Returns
    -------
    dict with keys:
        phi_snaps : (n_snap, 5, Nx, Ny)  species fractions
        c_snaps   : (n_snap, Nx, Ny)     nutrient field
        t_snaps   : (n_snap,)            time values
    """
    A, b_diag = theta_to_matrices(jnp.asarray(theta, dtype=jnp.float64))
    active_mask = jnp.ones(5, dtype=jnp.int64)

    # params dict: only JAX-traceable values (no Python ints used in jnp.arange)
    params = {
        "dt_h": cfg.dt_h,
        "Kp1": cfg.Kp1,
        "Eta": jnp.ones(5),
        "EtaPhi": jnp.ones(5),
        "c": cfg.c_hamilton,
        "alpha": cfg.alpha,
        "K_hill": jnp.array(cfg.K_hill),
        "n_hill": jnp.array(cfg.n_hill),
        "A": A,
        "b_diag": b_diag,
        "active_mask": active_mask,
    }

    # Build reaction step with compile-time constants for scan lengths
    _reaction_step = _make_reaction_step(cfg.n_react_sub, cfg.newton_iters)

    D_eff = jnp.array(cfg.D_eff)
    dt_macro = cfg.dt_macro
    Nx, Ny = cfg.Nx, cfg.Ny

    G, c = make_initial_state_2d(cfg)

    phi_snaps = [G.reshape(Nx, Ny, 12)[:, :, :5].transpose(2, 0, 1)]
    c_snaps = [c.copy()]
    t_snaps = [0.0]

    print(f"\n{'='*62}")
    print(f"2D Hamilton+Nutrient  |  Nx={Nx} Ny={Ny}")
    print(f"  dt_h={cfg.dt_h:.1e}  n_sub={cfg.n_react_sub}  n_macro={cfg.n_macro}")
    print(f"  D_c={cfg.D_c}  k_M={cfg.k_monod}  c_bc={cfg.c_boundary}")
    print(f"{'='*62}\n")

    for step in range(1, cfg.n_macro + 1):
        t = step * dt_macro

        # (1) Reaction: Hamilton Newton over all nodes
        G = _reaction_step(G, params)

        # (2) Species diffusion: explicit Euler with Neumann BCs
        phi_2d = G.reshape(Nx, Ny, 12)[:, :, :5]
        phi_2d = diffusion_step_species_2d(phi_2d, D_eff, dt_macro, cfg.dx, cfg.dy)

        # Write back to G
        G_2d = G.reshape(Nx, Ny, 12)
        G_2d = G_2d.at[:, :, :5].set(phi_2d)
        G_2d = G_2d.at[:, :, 5].set(1.0 - jnp.sum(phi_2d, axis=-1))
        G = G_2d.reshape(Nx * Ny, 12)

        # (3) Nutrient PDE step
        c = nutrient_step(c, phi_2d, cfg, dt_macro)

        # Save snapshot
        if step % cfg.save_every == 0 or step == cfg.n_macro:
            phi_snap = phi_2d.transpose(2, 0, 1)  # (5, Nx, Ny)
            phi_snaps.append(np.asarray(phi_snap))
            c_snaps.append(np.asarray(c))
            t_snaps.append(float(t))

            phi_mean = jnp.mean(phi_2d, axis=(0, 1))
            c_mean = float(jnp.mean(c))
            bar = ", ".join(f"{float(v):.4f}" for v in phi_mean)
            print(
                f"  [{100*step/cfg.n_macro:5.1f}%] t={t:.5f}  "
                f"phi_mean=[{bar}]  c_mean={c_mean:.4f}"
            )

    result = {
        "phi_snaps": np.array(phi_snaps),      # (n_snap, 5, Nx, Ny)
        "c_snaps": np.array(c_snaps),           # (n_snap, Nx, Ny)
        "t_snaps": np.array(t_snaps),           # (n_snap,)
    }
    return result


# ---------------------------------------------------------------------------
# Derived fields
# ---------------------------------------------------------------------------

def compute_di_field(phi_snaps):
    """
    Compute Dysbiosis Index field from phi snapshots.
    DI = 1 - H / ln(5) where H = -sum(p_i ln p_i).

    phi_snaps : (n_snap, 5, Nx, Ny)
    Returns   : (n_snap, Nx, Ny)
    """
    # p_i = phi_i / sum(phi)
    phi_sum = np.sum(phi_snaps, axis=1, keepdims=True)
    phi_sum_safe = np.where(phi_sum > 0, phi_sum, 1.0)
    p = phi_snaps / phi_sum_safe
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(p > 0, np.log(p), 0.0)
    H = -(p * log_p).sum(axis=1)
    di = 1.0 - H / np.log(5.0)
    return di


def compute_alpha_monod(phi_snaps, c_snaps, t_snaps, k_alpha=0.05, k_monod=1.0):
    """
    Compute Monod-weighted growth activity field.
    alpha_Monod(x,y) = k_alpha * integral(phi_total * c/(k+c) dt)

    phi_snaps : (n_snap, 5, Nx, Ny)
    c_snaps   : (n_snap, Nx, Ny)
    t_snaps   : (n_snap,)
    Returns   : (Nx, Ny)
    """
    phi_total = np.sum(phi_snaps, axis=1)  # (n_snap, Nx, Ny)
    monod = c_snaps / (k_monod + c_snaps)  # (n_snap, Nx, Ny)
    integrand = phi_total * monod           # (n_snap, Nx, Ny)
    alpha = k_alpha * np.trapezoid(integrand, t_snaps, axis=0)
    return alpha


# ---------------------------------------------------------------------------
# Nutrient-coupled Hamilton reaction (two-way coupling: c <-> phi)
# ---------------------------------------------------------------------------

def residual_c(g_new, g_prev, c_node, params):
    """
    Hamilton PDE residual with local nutrient concentration c_node.

    Same as residual() but c_node (scalar) replaces params["c"].
    This creates two-way coupling: nutrient affects species interactions.

    In the original model, params["c"] = 100.0 (Hamilton coupling constant).
    Here, c_node = local nutrient concentration (0-1), so interactions
    weaken in nutrient-depleted regions.
    """
    dt = params["dt_h"]
    Kp1 = params["Kp1"]
    Eta = params["Eta"]
    EtaPhi = params["EtaPhi"]
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

    # Hill gate for P.gingivalis (species 4)
    hill_mask = (K_hill > 1e-9).astype(jnp.float64) * (
        active_mask[4] == 1
    ).astype(jnp.float64)
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
            t1 = Kp1 * (2.0 - 4.0 * phi_new[i]) / (
                (phi_new[i] - 1.0) ** 3 * phi_new[i] ** 3
            )
            t2 = (1.0 / Eta[i]) * (
                gamma_new
                + (EtaPhi[i] + Eta[i] * psi_new[i] ** 2) * phidot[i]
                + Eta[i] * phi_new[i] * psi_new[i] * psidot[i]
            )
            t3 = (c_node / Eta[i]) * psi_new[i] * Ia[i]
            return Q_local.at[i].set(t1 + t2 - t3)
        def inactive_branch():
            return Q_local.at[i].set(phi_new[i])
        return jax.lax.cond(active, active_branch, inactive_branch), None

    Q, _ = jax.lax.scan(body_i_phi, Q, jnp.arange(5))
    Q = Q.at[5].set(
        gamma_new
        + Kp1 * (2.0 - 4.0 * phi0_new) / ((phi0_new - 1.0) ** 3 * phi0_new ** 3)
        + phi0dot
    )

    def body_i_psi(carry, i):
        Q_local = carry
        active = active_mask[i] == 1
        def active_branch():
            t1 = (-2.0 * Kp1) / (
                (psi_new[i] - 1.0) ** 2 * psi_new[i] ** 3
            ) - (2.0 * Kp1) / (
                (psi_new[i] - 1.0) ** 3 * psi_new[i] ** 2
            )
            t2 = (b_diag[i] * alpha / Eta[i]) * psi_new[i]
            t3 = phi_new[i] * psi_new[i] * phidot[i] + phi_new[i] ** 2 * psidot[i]
            t4 = (c_node / Eta[i]) * phi_new[i] * Ia[i]
            return Q_local.at[6 + i].set(t1 + t2 + t3 - t4)
        def inactive_branch():
            return Q_local.at[6 + i].set(psi_new[i])
        return jax.lax.cond(active, active_branch, inactive_branch), None

    Q, _ = jax.lax.scan(body_i_psi, Q, jnp.arange(5))
    Q = Q.at[11].set(jnp.sum(phi_new) + phi0_new - 1.0)
    return Q


def _make_newton_step_c_vmap(n_iters):
    """Create jitted vmapped Newton step with nutrient coupling.

    Vmaps over (g_prev, c_node) with shared params.
    """
    def newton_step_c(g_prev, c_node, params):
        active_mask = params["active_mask"]

        def body(carry, _):
            g = carry
            g = clip_state(g, active_mask)
            def F(gg):
                return residual_c(gg, g_prev, c_node, params)
            Q = F(g)
            J = jax.jacfwd(F)(g)
            delta = jnp.linalg.solve(J, -Q)
            g_next = g + delta
            g_next = clip_state(g_next, active_mask)
            return g_next, None

        g0 = clip_state(g_prev, active_mask)
        g_final, _ = jax.lax.scan(body, g0, jnp.arange(n_iters))
        return g_final

    return jax.jit(jax.vmap(newton_step_c, in_axes=(0, 0, None)))


def _make_reaction_step_c(n_sub, n_iters):
    """Create reaction step with nutrient coupling.

    c_flat stays constant during reaction sub-steps (only updated
    in the nutrient PDE step).
    """
    _newton_c_vmap = _make_newton_step_c_vmap(n_iters)

    def reaction_step_c(G, c_flat, params):
        def body(G_local, _):
            return _newton_c_vmap(G_local, c_flat, params), None
        G_final, _ = jax.lax.scan(body, G, jnp.arange(n_sub))
        return G_final

    return reaction_step_c


# ---------------------------------------------------------------------------
# CFL-stable nutrient PDE step (with sub-stepping)
# ---------------------------------------------------------------------------

def _nutrient_sub_step(c, phi_2d, D_c, k_M, g_cons, c_bc, dx, dy, dt_sub):
    """Single explicit Euler step for nutrient c(x,y)."""
    lap_c = laplacian_2d_dirichlet(c, dx, dy, c_bc)
    phi_total_w = jnp.sum(phi_2d * g_cons[None, None, :], axis=-1)
    monod = c / (k_M + c)
    consumption = phi_total_w * monod
    c_new = c + dt_sub * (D_c * lap_c - consumption)
    return jnp.clip(c_new, 0.0, c_bc)


def _make_nutrient_step_stable(n_sub_c):
    """Create CFL-stable nutrient step with n_sub_c sub-steps."""
    def step(c, phi_2d, D_c, k_M, g_cons, c_bc, dx, dy, dt_macro):
        dt_sub = dt_macro / n_sub_c
        def body(c_local, _):
            return _nutrient_sub_step(
                c_local, phi_2d, D_c, k_M, g_cons, c_bc, dx, dy, dt_sub
            ), None
        c_final, _ = jax.lax.scan(body, c, jnp.arange(n_sub_c))
        return c_final
    return step


# ---------------------------------------------------------------------------
# Biofilm geometry
# ---------------------------------------------------------------------------

def egg_shape_mask(Nx, Ny, Lx=1.0, Ly=1.0,
                   ax=0.35, ay=0.25, cx=0.5, cy=0.5,
                   skew=0.3, eps=0.1):
    """Klempt 2024 egg-shaped biofilm indicator phi_biofilm(x,y) in [0,1]."""
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    xn = (X - cx * Lx) / (ax * Lx)
    yn = (Y - cy * Ly + skew * (X - cx * Lx) ** 2) / (ay * Ly)
    r2 = xn ** 2 + yn ** 2
    return 0.5 * (1.0 - np.tanh((r2 - 1.0) / eps))


def make_initial_state_2d_biofilm(cfg, biofilm_mask):
    """Build initial Hamilton state with egg-shaped biofilm geometry.

    Species are initialized ONLY inside the biofilm region.
    Outside: phi=0, phi0=1 (void), psi=eps (inactive).
    """
    Nx, Ny = cfg.Nx, cfg.Ny
    active_mask = jnp.ones(5, dtype=jnp.int64)
    mask = jnp.array(biofilm_mask)

    G_2d = jnp.zeros((Nx, Ny, 12), dtype=jnp.float64)

    # Base composition (inside biofilm only)
    phi_base = jnp.array([0.12, 0.12, 0.08, 0.05, 0.02])
    for i in range(5):
        G_2d = G_2d.at[:, :, i].set(phi_base[i] * mask)

    # Normalize: sum(phi) < 1
    phi_sum = jnp.sum(G_2d[:, :, :5], axis=-1)
    scale = jnp.where(phi_sum > 0.999, 0.999 / phi_sum, 1.0)
    G_2d = G_2d.at[:, :, :5].set(G_2d[:, :, :5] * scale[:, :, None])

    # phi_0 = 1 - sum(phi)
    G_2d = G_2d.at[:, :, 5].set(1.0 - jnp.sum(G_2d[:, :, :5], axis=-1))

    # psi: active inside biofilm, inactive outside
    for i in range(5):
        psi_val = jnp.where(mask > 0.5, 0.999, 1e-10)
        G_2d = G_2d.at[:, :, 6 + i].set(psi_val)

    # gamma = 0
    G_2d = G_2d.at[:, :, 11].set(0.0)

    G_flat = G_2d.reshape(Nx * Ny, 12)
    c = jnp.ones((Nx, Ny), dtype=jnp.float64) * cfg.c_boundary

    return G_flat, c


# ---------------------------------------------------------------------------
# Full coupled simulation
# ---------------------------------------------------------------------------

def run_simulation_coupled(theta, cfg, biofilm_mask=None, n_sub_c=30,
                           reaction_fn=None, nutrient_fn=None,
                           c_hamilton_scale=1.0):
    """
    Run 2D Hamilton + nutrient with two-way coupling.

    Key difference from run_simulation():
      The local nutrient concentration c(x,y) modulates the Hamilton
      interaction strength, creating a two-way coupling:
        c -> phi (nutrient affects species interactions)
        phi -> c (species consume nutrients)

    With biofilm_mask: species are confined to the biofilm region,
    creating spatially heterogeneous composition (unlike 1D).

    Parameters
    ----------
    theta : array (20,)
    cfg   : Config2D
    biofilm_mask : (Nx, Ny) or None
    n_sub_c : int  nutrient PDE sub-steps for CFL stability
    reaction_fn : pre-compiled reaction step (reuse across conditions)
    nutrient_fn : pre-compiled nutrient step (reuse across conditions)
    c_hamilton_scale : float
        Scale factor for nutrient → Hamilton coupling.
        Nutrient PDE gives c ∈ [0, 1], but Hamilton model was calibrated
        with c = 100.  Set c_hamilton_scale = 100 to match.
    """
    A, b_diag = theta_to_matrices(jnp.asarray(theta, dtype=jnp.float64))
    active_mask = jnp.ones(5, dtype=jnp.int64)

    params = {
        "dt_h": cfg.dt_h,
        "Kp1": cfg.Kp1,
        "Eta": jnp.ones(5),
        "EtaPhi": jnp.ones(5),
        "alpha": cfg.alpha,
        "K_hill": jnp.array(cfg.K_hill),
        "n_hill": jnp.array(cfg.n_hill),
        "A": A,
        "b_diag": b_diag,
        "active_mask": active_mask,
    }

    _reaction_c = reaction_fn or _make_reaction_step_c(cfg.n_react_sub, cfg.newton_iters)
    _nutrient_stable = nutrient_fn or _make_nutrient_step_stable(n_sub_c)

    D_eff = jnp.array(cfg.D_eff)
    D_c = cfg.D_c
    k_M = cfg.k_monod
    g_cons = jnp.array(cfg.g_consumption)
    c_bc = cfg.c_boundary
    dt_macro = cfg.dt_macro
    Nx, Ny = cfg.Nx, cfg.Ny

    if biofilm_mask is not None:
        G, c = make_initial_state_2d_biofilm(cfg, biofilm_mask)
        mask_jax = jnp.array(biofilm_mask)
    else:
        G, c = make_initial_state_2d(cfg)
        mask_jax = None

    phi_snaps = [np.asarray(G.reshape(Nx, Ny, 12)[:, :, :5].transpose(2, 0, 1))]
    c_snaps = [np.asarray(c)]
    t_snaps = [0.0]

    print(f"\n{'='*65}")
    print(f"  2D Coupled Hamilton+Nutrient  |  Nx={Nx} Ny={Ny}")
    print(f"  dt_h={cfg.dt_h:.1e}  n_sub={cfg.n_react_sub}  n_macro={cfg.n_macro}")
    print(f"  D_c={D_c}  k_M={k_M}  n_sub_c={n_sub_c}")
    print(f"  c_hamilton_scale={c_hamilton_scale}")
    print(f"  biofilm={'egg-shape' if biofilm_mask is not None else 'full-domain'}")
    print(f"{'='*65}")

    _c_scale = float(c_hamilton_scale)

    for step in range(1, cfg.n_macro + 1):
        t = step * dt_macro

        # (1) Reaction with nutrient coupling
        # Scale c from PDE units [0,1] to Hamilton units [0, c_scale]
        c_flat = c.reshape(Nx * Ny) * _c_scale
        G = _reaction_c(G, c_flat, params)

        # (2) Species diffusion (explicit Euler, Neumann BCs)
        phi_2d = G.reshape(Nx, Ny, 12)[:, :, :5]
        phi_2d = diffusion_step_species_2d(phi_2d, D_eff, dt_macro, cfg.dx, cfg.dy)

        # Apply biofilm mask (zero species outside biofilm)
        if mask_jax is not None:
            phi_2d = phi_2d * mask_jax[:, :, None]

        # Write back to G
        G_2d = G.reshape(Nx, Ny, 12)
        G_2d = G_2d.at[:, :, :5].set(phi_2d)
        G_2d = G_2d.at[:, :, 5].set(1.0 - jnp.sum(phi_2d, axis=-1))
        G = G_2d.reshape(Nx * Ny, 12)

        # (3) Nutrient PDE (CFL-stable sub-stepping)
        c = _nutrient_stable(c, phi_2d, D_c, k_M, g_cons, c_bc, cfg.dx, cfg.dy, dt_macro)

        # Save snapshot
        if step % cfg.save_every == 0 or step == cfg.n_macro:
            phi_snap = np.asarray(phi_2d.transpose(2, 0, 1))
            phi_snaps.append(phi_snap)
            c_snaps.append(np.asarray(c))
            t_snaps.append(float(t))

            phi_mean = jnp.mean(phi_2d, axis=(0, 1))
            c_mean = float(jnp.mean(c))
            bar = ", ".join(f"{float(v):.4f}" for v in phi_mean)
            print(
                f"  [{100*step/cfg.n_macro:5.1f}%] t={t:.5f}  "
                f"phi_mean=[{bar}]  c_mean={c_mean:.4f}"
            )

    return {
        "phi_snaps": np.array(phi_snaps),
        "c_snaps": np.array(c_snaps),
        "t_snaps": np.array(t_snaps),
    }


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

THETA_DEMO = np.array([
    1.34, -0.18, 1.79, 1.17, 2.58,
    3.51, 2.73, 0.71, 2.1, 0.37,
    2.05, -0.15, 3.56, 0.16, 0.12,
    0.32, 1.49, 2.1, 2.41, 2.5,
])


def main():
    cfg = Config2D(
        Nx=15, Ny=15,
        n_macro=30,
        n_react_sub=10,
        dt_h=1e-5,
        save_every=10,
    )
    result = run_simulation(THETA_DEMO, cfg)

    phi_snaps = result["phi_snaps"]
    c_snaps = result["c_snaps"]
    t_snaps = result["t_snaps"]

    print(f"\nSnapshots: {len(t_snaps)}")
    print(f"phi shape: {phi_snaps.shape}")
    print(f"c shape:   {c_snaps.shape}")

    # Derived fields
    di = compute_di_field(phi_snaps)
    alpha = compute_alpha_monod(phi_snaps, c_snaps, t_snaps)
    print(f"\nFinal DI range: [{di[-1].min():.4f}, {di[-1].max():.4f}]")
    print(f"alpha_Monod range: [{alpha.min():.6f}, {alpha.max():.6f}]")


if __name__ == "__main__":
    main()
