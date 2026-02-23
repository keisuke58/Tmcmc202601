"""TMCMC -> Hamilton ODE -> Klempt Reaction-Diffusion 1D Pipeline

Full scientific pipeline demonstrating:
  1. TMCMC MAP parameters for commensal vs dysbiotic biofilm
  2. Hamilton 0D ODE integration -> final species compositions phi_i
  3. Construction of 1D biofilm depth profile phi(x) from 0D compositions
  4. Klempt 2024 steady-state nutrient transport PDE (1D):
       -D_c * d^2c/dx^2 + g * phi(x) * c / (k + c) = 0
     BC: c(x=L)=1 (saliva side), dc/dx|_{x=0}=0 (tooth surface, no flux)
  5. Comparison figure: commensal (mild) vs dysbiotic parameter sets

Physical interpretation:
  x = 0 : tooth surface (biofilm attachment site, no external nutrient supply)
  x = L : saliva (c = c_inf = 1, abundant nutrients)
  phi(x) : total biofilm volume fraction (concentrated near x=0, decreases toward saliva)
  c(x)   : steady-state nutrient concentration (depleted inside biofilm)

Environment: klempt_fem (Python 3.11, jax 0.9.0.1) -- JAX only, no jax-fem needed.
"""

import os
import sys

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "klempt2024_results", "hamilton_rd_pipeline")

# ---------------------------------------------------------------------------
# TMCMC MAP parameter sets (theta: 20 parameters)
# ---------------------------------------------------------------------------
# Commensal (mild_weight equivalent): a35=theta[18]=2.41, a45=theta[19]=2.5
THETA_COMMENSAL = np.array([
    1.34, -0.18, 1.79, 1.17, 2.58,    # theta[0-4]
    3.51,  2.73, 0.71, 2.10, 0.37,    # theta[5-9]
    2.05, -0.15, 3.56, 0.16, 0.12,    # theta[10-14]
    0.32,  1.49, 2.10, 2.41, 2.50,    # theta[15-19]  <- a35=2.41, a45=2.50
])

# Dysbiotic (baseline MAP from K0.05_n4.0_baseline sweep): a35=20.94
THETA_DYSBIOTIC = np.array([
    0.9679,  2.6813, 0.6423, 0.9248, 1.0350,   # theta[0-4]
    0.1015,  1.3271, 0.2119, 2.4925, 0.5537,   # theta[5-9]
    1.3808,  1.4155, 2.4463, 2.2082, 3.4005,   # theta[10-14]
    0.2979,  0.2864, 2.3241, 20.9445, 2.8122,  # theta[15-19]  <- a35=20.94
])

SPECIES_NAMES = ["S. oralis", "A. naeslundii", "V. dispar", "F. nucleatum", "P. gingivalis"]
SPECIES_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


# ---------------------------------------------------------------------------
# Hamilton ODE: 0D model (no spatial diffusion)
# From jax_hamilton_0d_5species_demo.py
# ---------------------------------------------------------------------------

def theta_to_matrices(theta):
    """Convert 20-parameter theta to interaction matrix A and growth vector b."""
    theta = jnp.array(theta)
    A = jnp.zeros((5, 5))
    b = jnp.zeros(5)
    # Self-interactions and growth rates for So, An
    A = A.at[0, 0].set(theta[0])
    A = A.at[0, 1].set(theta[1])
    A = A.at[1, 0].set(theta[1])
    A = A.at[1, 1].set(theta[2])
    b = b.at[0].set(theta[3])
    b = b.at[1].set(theta[4])
    # Vd, Fn
    A = A.at[2, 2].set(theta[5])
    A = A.at[2, 3].set(theta[6])
    A = A.at[3, 2].set(theta[6])
    A = A.at[3, 3].set(theta[7])
    b = b.at[2].set(theta[8])
    b = b.at[3].set(theta[9])
    # Cross: So-Vd, So-Fn
    A = A.at[0, 2].set(theta[10])
    A = A.at[2, 0].set(theta[10])
    A = A.at[0, 3].set(theta[11])
    A = A.at[3, 0].set(theta[11])
    # Cross: An-Vd, An-Fn
    A = A.at[1, 2].set(theta[12])
    A = A.at[2, 1].set(theta[12])
    A = A.at[1, 3].set(theta[13])
    A = A.at[3, 1].set(theta[13])
    # Pg: self-interaction and growth
    A = A.at[4, 4].set(theta[14])
    b = b.at[4].set(theta[15])
    # Pg cross-interactions: So-Pg, An-Pg, Vd-Pg (a35=theta[18]), Fn-Pg (a45=theta[19])
    A = A.at[0, 4].set(theta[16])
    A = A.at[4, 0].set(theta[16])
    A = A.at[1, 4].set(theta[17])
    A = A.at[4, 1].set(theta[17])
    A = A.at[2, 4].set(theta[18])   # a35: Vd -> Pg (theta[18])
    A = A.at[4, 2].set(theta[18])
    A = A.at[3, 4].set(theta[19])   # a45: Fn -> Pg (theta[19])
    A = A.at[4, 3].set(theta[19])
    return A, b


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
    hill_mask = (K_hill > 1e-9).astype(jnp.float64) * (active_mask[4] == 1).astype(jnp.float64)
    fn = jnp.maximum(phi_new[3] * psi_new[3], 0.0)
    num = fn ** n_hill
    den = K_hill ** n_hill + num
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
        gamma_new
        + Kp1 * (2.0 - 4.0 * phi0_new) / ((phi0_new - 1.0) ** 3 * phi0_new ** 3)
        + phi0dot
    )

    def body_i_psi(carry, i):
        Q_local = carry
        active = active_mask[i] == 1

        def active_branch():
            t1 = (-2.0 * Kp1) / ((psi_new[i] - 1.0) ** 2 * psi_new[i] ** 3) \
               - (2.0 * Kp1) / ((psi_new[i] - 1.0) ** 3 * psi_new[i] ** 2)
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


def newton_step_0d(g_prev, params):
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
    g_final, _ = jax.lax.scan(body, g0, jnp.arange(6))
    return g_final


newton_step_0d_jit = jax.jit(newton_step_0d)


def run_hamilton_0d(theta, t_final=0.05, dt_h=1e-5):
    """Integrate Hamilton 0D ODE to t_final and return trajectory.

    Returns
    -------
    phi_traj : np.ndarray (n_steps, 5)   -- phi_i trajectory
    phi_final : np.ndarray (5,)           -- final species fractions
    """
    A, b_diag = theta_to_matrices(theta)
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
        "active_mask": jnp.ones(5, dtype=jnp.int64),
    }
    n_steps = int(t_final / dt_h)

    # Initial state matching the TMCMC experiment (phi_init from ODE IC)
    phi_init = jnp.array([0.018, 0.0045, 0.423, 0.00225, 0.00225])
    phi_init = phi_init * (0.999999 / jnp.sum(phi_init))
    phi0_init = 1.0 - jnp.sum(phi_init)
    g = jnp.zeros(12, dtype=jnp.float64)
    g = g.at[0:5].set(phi_init)
    g = g.at[5].set(phi0_init)
    g = g.at[6:11].set(jnp.ones(5) * 0.999)

    def step(carry, _):
        g_new = newton_step_0d_jit(carry, params)
        return g_new, g_new[0:5]

    g_final, phi_traj = jax.lax.scan(step, g, jnp.arange(n_steps))
    phi_final = np.array(g_final[0:5])
    phi_traj = np.array(phi_traj)
    return phi_traj, phi_final


# ---------------------------------------------------------------------------
# 1D biofilm spatial profile from 0D composition
# ---------------------------------------------------------------------------

def make_biofilm_profile(phi_final, x, depth_scale=0.4):
    """Construct 1D depth-dependent biofilm volume fraction.

    Model: biofilm is attached at x=0 (tooth), decays toward x=L (saliva).
    Each species has the same spatial shape (a first approximation);
    the amplitude is given by phi_final from the 0D ODE.

    phi_i(x) = phi_i_final * shape(x)
    shape(x) = exp(-x / depth_scale)   [concentrated near tooth]

    Parameters
    ----------
    phi_final : array (5,)   -- final species fractions from 0D ODE
    x : array (N,)            -- spatial grid [0, L]
    depth_scale : float       -- decay length in [0,1] coordinates

    Returns
    -------
    phi_species : array (N, 5)  -- phi_i(x) for each species
    phi_total : array (N,)      -- sum of species fractions
    """
    L = x[-1]
    shape = np.exp(-x / (depth_scale * L))   # shape: 1 at x=0, decays toward x=L
    # Normalize shape so mean matches 0D value
    phi_species = np.outer(shape, phi_final)
    phi_total = phi_species.sum(axis=1)
    # Cap at 1
    phi_total = np.minimum(phi_total, 0.99)
    return phi_species, phi_total


# ---------------------------------------------------------------------------
# 1D steady-state nutrient transport: Newton solver
# ---------------------------------------------------------------------------

def solve_1d_nutrient(phi_x, x, D_c=1.0, k_monod=1.0, g_eff=10.0, c_inf=1.0,
                      n_newton=30):
    """Solve -D_c d^2c/dx^2 + g*phi(x)*c/(k+c) = 0 in 1D with Newton's method.

    Boundary conditions:
      c(x=x[-1]) = c_inf  (Dirichlet at saliva side)
      dc/dx|_{x=x[0]} = 0 (Neumann zero-flux at tooth surface)

    The Dirichlet BC at j=N-1 is enforced by fixing c[-1]=c_inf.
    The Neumann BC at j=0 uses ghost-point: c_{-1}=c_1, so:
      F_0 = D_c * 2*(c_1 - c_0)/dx^2 - g*phi_0*c_0/(k+c_0) = 0

    Parameters
    ----------
    phi_x : array (N,)   -- biofilm volume fraction at each node
    x : array (N,)       -- spatial grid (uniform)
    D_c, k_monod, g_eff : float   -- nutrient transport parameters
    c_inf : float        -- Dirichlet BC value at x[-1] (saliva)
    n_newton : int       -- Newton iterations

    Returns
    -------
    c : array (N,)   -- steady-state nutrient concentration
    """
    N = len(x)
    dx = x[1] - x[0]

    # Initial guess: linear from c_inf (x=L) to c_inf*0.3 (x=0)
    c = np.linspace(c_inf * 0.3, c_inf, N)
    c[-1] = c_inf   # enforce Dirichlet BC

    for _ in range(n_newton):
        F = np.zeros(N)
        dF = np.zeros(N)   # diagonal of Jacobian

        # Node j=0: Neumann ghost-point (dc/dx=0 -> c_{-1}=c_1)
        j = 0
        F[j] = D_c * 2.0 * (c[j+1] - c[j]) / dx**2 - g_eff * phi_x[j] * c[j] / (k_monod + c[j])
        dF[j] = -2.0 * D_c / dx**2 - g_eff * phi_x[j] * k_monod / (k_monod + c[j])**2

        # Interior nodes j=1,...,N-2
        for j in range(1, N-1):
            F[j] = D_c * (c[j+1] - 2*c[j] + c[j-1]) / dx**2 \
                   - g_eff * phi_x[j] * c[j] / (k_monod + c[j])
            dF[j] = -2.0 * D_c / dx**2 - g_eff * phi_x[j] * k_monod / (k_monod + c[j])**2

        # Node j=N-1: Dirichlet BC (c=c_inf, residual always 0)
        F[N-1] = 0.0
        dF[N-1] = 1.0

        # Build full tridiagonal Jacobian
        J = np.diag(dF)
        # Off-diagonals from Laplacian
        for j in range(1, N-1):
            J[j, j-1] += D_c / dx**2
            J[j, j+1] += D_c / dx**2
        # j=0: Neumann ghost point (only upper off-diagonal)
        J[0, 1] += 2.0 * D_c / dx**2

        # Newton update
        delta = np.linalg.solve(J, -F)
        c = c + delta
        c = np.clip(c, 0.0, c_inf)
        c[-1] = c_inf   # keep Dirichlet BC

    return c


# ---------------------------------------------------------------------------
# Thiele modulus (dimensionless)
# ---------------------------------------------------------------------------

def thiele_modulus(phi_mean, g_eff, D_c, k_monod, L=1.0):
    """Phi_T = L * sqrt(g * phi_mean / (D_c * k_monod)).

    Large Phi_T -> diffusion-limited (strong nutrient depletion inside biofilm).
    """
    return L * np.sqrt(g_eff * phi_mean / (D_c * k_monod))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("TMCMC -> Hamilton ODE -> Klempt 1D Nutrient Transport Pipeline")
    print("=" * 70)

    os.makedirs(OUT_DIR, exist_ok=True)

    # Spatial grid: x=0 (tooth surface), x=1 (saliva)
    N = 120
    x = np.linspace(0.0, 1.0, N)
    L = 1.0

    # RD parameters (dimensionless, Klempt 2024 Table 1)
    D_c = 1.0
    k_monod = 1.0
    c_inf = 1.0

    # Two parameter sets
    cases = {
        "Commensal\n(a35=2.41)": {
            "theta": THETA_COMMENSAL,
            "color": "#2ca02c",
            "linestyle": "-",
        },
        "Dysbiotic\n(a35=20.94)": {
            "theta": THETA_DYSBIOTIC,
            "color": "#d62728",
            "linestyle": "--",
        },
    }

    # --- Step 1: Hamilton 0D ODE ---
    print("\n[1/4] Hamilton 0D ODE integration (t_final=0.05) ...")
    results = {}
    for name, cfg in cases.items():
        label = name.replace("\n", " ")
        print(f"  {label}: integrating...")
        phi_traj, phi_final = run_hamilton_0d(cfg["theta"], t_final=0.05, dt_h=1e-5)
        cfg["phi_traj"] = phi_traj
        cfg["phi_final"] = phi_final
        print(f"    phi_final: So={phi_final[0]:.3f}, An={phi_final[1]:.3f}, "
              f"Vd={phi_final[2]:.3f}, Fn={phi_final[3]:.3f}, Pg={phi_final[4]:.3f}")
        results[name] = phi_final

    # --- Step 2: 1D spatial profiles ---
    print("\n[2/4] Constructing 1D biofilm depth profiles ...")
    for name, cfg in cases.items():
        phi_species, phi_total = make_biofilm_profile(cfg["phi_final"], x, depth_scale=0.4)
        cfg["phi_species_1d"] = phi_species
        cfg["phi_total_1d"] = phi_total
        phi_mean = float(phi_total.mean())
        label = name.replace("\n", " ")
        print(f"  {label}: phi_total mean={phi_mean:.4f}, max={phi_total.max():.4f}")

    # --- Step 3: Nutrient transport PDE (sweep over g_eff) ---
    print("\n[3/4] Solving 1D steady-state RD (Newton) ...")
    g_eff_values = [5.0, 20.0, 50.0]   # low / medium / Klempt-benchmark

    for name, cfg in cases.items():
        cfg["nutrient_profiles"] = {}
        for g_eff in g_eff_values:
            c_sol = solve_1d_nutrient(
                cfg["phi_total_1d"], x, D_c=D_c, k_monod=k_monod,
                g_eff=g_eff, c_inf=c_inf,
            )
            cfg["nutrient_profiles"][g_eff] = c_sol
        label = name.replace("\n", " ")
        c_g50 = cfg["nutrient_profiles"][50.0]
        phi_m = float(cfg["phi_total_1d"].mean())
        Phi_T = thiele_modulus(phi_m, 50.0, D_c, k_monod)
        print(f"  {label}: g=50 -> c_min={c_g50.min():.4f}, c_mean={c_g50.mean():.4f}, "
              f"Thiele={Phi_T:.2f}")

    # --- Step 4: Plot ---
    print("\n[4/4] Plotting ...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    n_steps_traj = next(iter(cases.values()))["phi_traj"].shape[0]
    t_arr = np.linspace(0, 0.05, n_steps_traj)

    # Panel (0,0): 0D ODE trajectory - commensal
    ax = axes[0, 0]
    cfg_c = cases["Commensal\n(a35=2.41)"]
    for i, (sname, col) in enumerate(zip(SPECIES_NAMES, SPECIES_COLORS)):
        ax.plot(t_arr, cfg_c["phi_traj"][:, i], color=col, label=sname, lw=1.5)
    ax.set_xlabel("Dimensionless time")
    ax.set_ylabel("Volume fraction phi_i")
    ax.set_title("Commensal (a35=2.41)\nHamilton ODE trajectory")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_xlim(0, 0.05)

    # Panel (0,1): 0D ODE trajectory - dysbiotic
    ax = axes[0, 1]
    cfg_d = cases["Dysbiotic\n(a35=20.94)"]
    for i, (sname, col) in enumerate(zip(SPECIES_NAMES, SPECIES_COLORS)):
        ax.plot(t_arr, cfg_d["phi_traj"][:, i], color=col, label=sname, lw=1.5)
    ax.set_xlabel("Dimensionless time")
    ax.set_ylabel("Volume fraction phi_i")
    ax.set_title("Dysbiotic (a35=20.94)\nHamilton ODE trajectory")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_xlim(0, 0.05)

    # Panel (0,2): Final composition bar chart
    ax = axes[0, 2]
    phi_c = cfg_c["phi_final"]
    phi_d = cfg_d["phi_final"]
    bar_x = np.arange(5)
    width = 0.35
    ax.bar(bar_x - width/2, phi_c, width, color="#2ca02c", alpha=0.8, label="Commensal")
    ax.bar(bar_x + width/2, phi_d, width, color="#d62728", alpha=0.8, label="Dysbiotic")
    ax.set_xticks(bar_x)
    ax.set_xticklabels(["So", "An", "Vd", "Fn", "Pg"], fontsize=9)
    ax.set_ylabel("Final volume fraction phi_i")
    ax.set_title("Final composition (0D ODE, t=0.05)")
    ax.legend()

    # Panel (1,0): 1D biofilm profiles (phi_total)
    ax = axes[1, 0]
    ax.plot(x, cfg_c["phi_total_1d"], color="#2ca02c", lw=2, label="Commensal total phi")
    ax.plot(x, cfg_d["phi_total_1d"], color="#d62728", lw=2, ls="--", label="Dysbiotic total phi")
    # Also Pg only
    ax.plot(x, cfg_c["phi_species_1d"][:, 4], color="#2ca02c", lw=1, ls=":", label="Comm. Pg")
    ax.plot(x, cfg_d["phi_species_1d"][:, 4], color="#d62728", lw=1, ls=":", label="Dysb. Pg")
    ax.axvline(0, color="gray", lw=0.5, ls="-", label="tooth surface")
    ax.set_xlabel("Depth x  [0=tooth, 1=saliva]")
    ax.set_ylabel("Volume fraction")
    ax.set_title("1D Biofilm Depth Profile\nsolid=total phi, dotted=Pg")
    ax.legend(fontsize=7)

    # Panel (1,1): Nutrient profiles at g_eff=50
    ax = axes[1, 1]
    g_eff_plot = 50.0
    c_c = cfg_c["nutrient_profiles"][g_eff_plot]
    c_d = cfg_d["nutrient_profiles"][g_eff_plot]
    ax.plot(x, c_c, color="#2ca02c", lw=2, label="Commensal")
    ax.plot(x, c_d, color="#d62728", lw=2, ls="--", label="Dysbiotic")
    ax.fill_between(x, c_c, c_d, alpha=0.15, color="gray",
                    label=f"Difference (Dysb-Comm at tooth: {c_d[0]-c_c[0]:.3f})")
    ax.set_xlabel("Depth x  [0=tooth, 1=saliva]")
    ax.set_ylabel("Nutrient concentration c")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Steady-State Nutrient Profile\ng_eff={g_eff_plot}, D_c={D_c}, k={k_monod}")
    ax.legend(fontsize=8)

    # Panel (1,2): Nutrient profiles across g_eff values
    ax = axes[1, 2]
    linestyles_g = {5.0: ":", 20.0: "--", 50.0: "-"}
    for g_eff in g_eff_values:
        ls = linestyles_g[g_eff]
        c_c = cfg_c["nutrient_profiles"][g_eff]
        c_d = cfg_d["nutrient_profiles"][g_eff]
        Phi_T_c = thiele_modulus(float(cfg_c["phi_total_1d"].mean()), g_eff, D_c, k_monod)
        Phi_T_d = thiele_modulus(float(cfg_d["phi_total_1d"].mean()), g_eff, D_c, k_monod)
        ax.plot(x, c_c, color="#2ca02c", lw=1.5, ls=ls,
                label=f"Comm g={g_eff:.0f} (Phi_T={Phi_T_c:.1f})")
        ax.plot(x, c_d, color="#d62728", lw=1.5, ls=ls,
                label=f"Dysb g={g_eff:.0f} (Phi_T={Phi_T_d:.1f})")
    ax.set_xlabel("Depth x  [0=tooth, 1=saliva]")
    ax.set_ylabel("Nutrient concentration c")
    ax.set_ylim(0, 1.05)
    ax.set_title("Nutrient Profiles vs. Consumption Rate\n(Thiele modulus in legend)")
    ax.legend(fontsize=6, ncol=2)

    plt.suptitle(
        "TMCMC → Hamilton ODE → Klempt 1D Nutrient Transport Pipeline\n"
        "Commensal (a35=2.41) vs. Dysbiotic (a35=20.94) biofilm",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig_path = os.path.join(OUT_DIR, "hamilton_rd_pipeline_comparison.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"  Figure saved: {fig_path}")

    # --- Summary table ---
    print("\n--- Summary Table ---")
    print(f"{'Case':<25} {'phi_Pg':>8} {'phi_total_mean':>15} {'c_min(g=50)':>12} {'c_tooth(g=50)':>14} {'Thiele(g=50)':>12}")
    print("-" * 88)
    for name, cfg in cases.items():
        label = name.replace("\n", " ")
        phi_m = float(cfg["phi_total_1d"].mean())
        c50 = cfg["nutrient_profiles"][50.0]
        Phi_T = thiele_modulus(phi_m, 50.0, D_c, k_monod)
        print(f"{label:<25} {cfg['phi_final'][4]:>8.4f} {phi_m:>15.4f} "
              f"{c50.min():>12.4f} {c50[0]:>14.4f} {Phi_T:>12.2f}")

    print("\nDone. Results in:", OUT_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
