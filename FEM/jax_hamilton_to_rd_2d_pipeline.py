"""TMCMC -> Hamilton 0D ODE -> 2D Nutrient PDE: Commensal vs Dysbiotic

Issue #3 (P2): 1D パイプラインの 2D 拡張。

アプローチ (Issue #3 の記載通り):
  1. Hamilton 0D ODE で条件別 phi_i_final を取得
  2. 2D メッシュ上に phi_i(x,y) を生成 (x方向: 深さプロファイル, y方向: 均一)
  3. 2D 定常栄養 PDE を Newton 法で解く:
       -D_c (∂²c/∂x² + ∂²c/∂y²) + g·φ(x,y)·c/(k+c) = 0
     BC: c(x=Lx, y) = 1.0  (saliva, Dirichlet)
         Neumann (no-flux) on x=0, y=0, y=Ly
  4. Commensal vs Dysbiotic の 2D 栄養場を比較

座標系:
  x = 深さ方向 (0=歯面, Lx=唾液)
  y = 横方向 (歯面に沿った方向)

Environment: klempt_fem (Python 3.11, jax 0.9.0.1)
Run: ~/.pyenv/versions/miniconda3-latest/envs/klempt_fem/bin/python \\
     Tmcmc202601/FEM/jax_hamilton_to_rd_2d_pipeline.py
"""

import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "klempt2024_results", "hamilton_rd_2d_pipeline")

# ---------------------------------------------------------------------------
# TMCMC MAP parameter sets (same as 1D pipeline)
# ---------------------------------------------------------------------------
THETA_COMMENSAL = jnp.array([
    1.34, -0.18, 1.79, 1.17, 2.58,
    3.51,  2.73, 0.71, 2.10, 0.37,
    2.05, -0.15, 3.56, 0.16, 0.12,
    0.32,  1.49, 2.10, 2.41, 2.50,
])

THETA_DYSBIOTIC = jnp.array([
    0.9679,  2.6813, 0.6423, 0.9248, 1.0350,
    0.1015,  1.3271, 0.2119, 2.4925, 0.5537,
    1.3808,  1.4155, 2.4463, 2.2082, 3.4005,
    0.2979,  0.2864, 2.3241, 20.9445, 2.8122,
])

SPECIES_NAMES = ["S. oralis", "A. naeslundii", "V. dispar",
                 "F. nucleatum", "P. gingivalis"]
SPECIES_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


# ---------------------------------------------------------------------------
# Hamilton 0D ODE (from jax_hamilton_to_rd_1d_pipeline.py)
# ---------------------------------------------------------------------------

def theta_to_matrices(theta):
    theta = jnp.array(theta)
    A = jnp.zeros((5, 5))
    b = jnp.zeros(5)
    A = A.at[0, 0].set(theta[0]); A = A.at[0, 1].set(theta[1])
    A = A.at[1, 0].set(theta[1]); A = A.at[1, 1].set(theta[2])
    b = b.at[0].set(theta[3]);    b = b.at[1].set(theta[4])
    A = A.at[2, 2].set(theta[5]); A = A.at[2, 3].set(theta[6])
    A = A.at[3, 2].set(theta[6]); A = A.at[3, 3].set(theta[7])
    b = b.at[2].set(theta[8]);    b = b.at[3].set(theta[9])
    A = A.at[0, 2].set(theta[10]); A = A.at[2, 0].set(theta[10])
    A = A.at[0, 3].set(theta[11]); A = A.at[3, 0].set(theta[11])
    A = A.at[1, 2].set(theta[12]); A = A.at[2, 1].set(theta[12])
    A = A.at[1, 3].set(theta[13]); A = A.at[3, 1].set(theta[13])
    A = A.at[4, 4].set(theta[14]); b = b.at[4].set(theta[15])
    A = A.at[0, 4].set(theta[16]); A = A.at[4, 0].set(theta[16])
    A = A.at[1, 4].set(theta[17]); A = A.at[4, 1].set(theta[17])
    A = A.at[2, 4].set(theta[18]); A = A.at[4, 2].set(theta[18])
    A = A.at[3, 4].set(theta[19]); A = A.at[4, 3].set(theta[19])
    return A, b


def clip_state(g, active_mask):
    eps = 1e-10
    mask = active_mask.astype(jnp.float64)
    phi = mask * jnp.clip(g[0:5], eps, 1.0 - eps)
    psi = mask * jnp.clip(g[6:11], eps, 1.0 - eps)
    phi0 = jnp.clip(g[5], eps, 1.0 - eps)
    gamma = jnp.clip(g[11], -1e6, 1e6)
    out = jnp.zeros_like(g)
    out = out.at[0:5].set(phi)
    out = out.at[5].set(phi0)
    out = out.at[6:11].set(psi)
    out = out.at[11].set(gamma)
    return out


def residual_0d(g_new, g_prev, params):
    dt = params["dt_h"]; Kp1 = params["Kp1"]
    Eta = params["Eta"]; EtaPhi = params["EtaPhi"]
    c = params["c"]; alpha = params["alpha"]
    K_hill = params["K_hill"]; n_hill = params["n_hill"]
    A = params["A"]; b_diag = params["b_diag"]
    active_mask = params["active_mask"]; eps = 1e-12

    phi_new = g_new[0:5]; phi0_new = g_new[5]
    psi_new = g_new[6:11]; gamma_new = g_new[11]
    phi_old = g_prev[0:5]; phi0_old = g_prev[5]; psi_old = g_prev[6:11]
    phidot = (phi_new - phi_old) / dt
    phi0dot = (phi0_new - phi0_old) / dt
    psidot = (psi_new - psi_old) / dt

    Ia = A @ (phi_new * psi_new)
    hill_mask = (K_hill > 1e-9).astype(jnp.float64) * (active_mask[4] == 1).astype(jnp.float64)
    fn = jnp.maximum(phi_new[3] * psi_new[3], 0.0)
    num = fn ** n_hill; den = K_hill ** n_hill + num
    factor = jnp.where(den > eps, num / den, 0.0) * hill_mask
    Ia = Ia.at[4].set(Ia[4] * factor)

    Q = jnp.zeros(12, dtype=jnp.float64)

    def body_phi(carry, i):
        Q_l = carry; active = active_mask[i] == 1
        def act():
            t1 = Kp1 * (2.0 - 4.0 * phi_new[i]) / ((phi_new[i] - 1.0)**3 * phi_new[i]**3)
            t2 = (1.0/Eta[i]) * (gamma_new + (EtaPhi[i] + Eta[i]*psi_new[i]**2)*phidot[i]
                                  + Eta[i]*phi_new[i]*psi_new[i]*psidot[i])
            t3 = (c/Eta[i]) * psi_new[i] * Ia[i]
            return Q_l.at[i].set(t1 + t2 - t3)
        def inact(): return Q_l.at[i].set(phi_new[i])
        return jax.lax.cond(active, act, inact), None

    Q, _ = jax.lax.scan(body_phi, Q, jnp.arange(5))
    Q = Q.at[5].set(gamma_new + Kp1*(2.0 - 4.0*phi0_new)/((phi0_new-1.0)**3*phi0_new**3) + phi0dot)

    def body_psi(carry, i):
        Q_l = carry; active = active_mask[i] == 1
        def act():
            t1 = (-2.0*Kp1)/((psi_new[i]-1.0)**2*psi_new[i]**3) - (2.0*Kp1)/((psi_new[i]-1.0)**3*psi_new[i]**2)
            t2 = (b_diag[i]*alpha/Eta[i]) * psi_new[i]
            t3 = phi_new[i]*psi_new[i]*phidot[i] + phi_new[i]**2*psidot[i]
            t4 = (c/Eta[i]) * phi_new[i] * Ia[i]
            return Q_l.at[6+i].set(t1 + t2 + t3 - t4)
        def inact(): return Q_l.at[6+i].set(psi_new[i])
        return jax.lax.cond(active, act, inact), None

    Q, _ = jax.lax.scan(body_psi, Q, jnp.arange(5))
    Q = Q.at[11].set(jnp.sum(phi_new) + phi0_new - 1.0)
    return Q


def newton_step_0d(g_prev, params):
    active_mask = params["active_mask"]
    def body(carry, _):
        g = clip_state(carry, active_mask)
        def F(gg): return residual_0d(gg, g_prev, params)
        Q = F(g); J = jax.jacfwd(F)(g)
        g_next = clip_state(g + jnp.linalg.solve(J, -Q), active_mask)
        return g_next, None
    g_final, _ = jax.lax.scan(body, clip_state(g_prev, active_mask), jnp.arange(6))
    return g_final


newton_step_0d_jit = jax.jit(newton_step_0d)


def run_hamilton_0d(theta, t_final=0.05, dt_h=1e-5):
    """Run Hamilton 0D ODE → phi_traj, phi_final."""
    A, b_diag = theta_to_matrices(theta)
    params = {
        "dt_h": dt_h, "Kp1": 1e-4,
        "Eta": jnp.ones(5), "EtaPhi": jnp.ones(5),
        "c": 100.0, "alpha": 100.0,
        "K_hill": jnp.array(0.05), "n_hill": jnp.array(4.0),
        "A": A, "b_diag": b_diag,
        "active_mask": jnp.ones(5, dtype=jnp.int64),
    }
    n_steps = int(t_final / dt_h)
    phi_init = jnp.array([0.018, 0.0045, 0.423, 0.00225, 0.00225])
    phi_init = phi_init * (0.999999 / jnp.sum(phi_init))
    g = jnp.zeros(12, dtype=jnp.float64)
    g = g.at[0:5].set(phi_init)
    g = g.at[5].set(1.0 - jnp.sum(phi_init))
    g = g.at[6:11].set(jnp.ones(5) * 0.999)

    def step(carry, _):
        g_new = newton_step_0d_jit(carry, params)
        return g_new, g_new[0:5]

    g_final, phi_traj = jax.lax.scan(step, g, jnp.arange(n_steps))
    return np.array(phi_traj), np.array(g_final[0:5])


# ---------------------------------------------------------------------------
# 2D biofilm profile: x=depth (exponential decay), y=lateral (uniform)
# ---------------------------------------------------------------------------

def make_biofilm_profile_2d(phi_final, x_grid, y_grid, depth_scale=0.4):
    """Construct 2D phi_i(x,y) from 0D composition.

    phi_i(x,y) = phi_i_final * exp(-x / depth_scale)  [uniform in y]

    Returns: phi_species (Nx, Ny, 5), phi_total (Nx, Ny)
    """
    Nx, Ny = len(x_grid), len(y_grid)
    L = x_grid[-1]
    shape_x = np.exp(-x_grid / (depth_scale * L))  # (Nx,)
    # shape_x[i] * phi_final[s] → (Nx, 5), then tile in y
    phi_1d = np.outer(shape_x, phi_final)           # (Nx, 5)
    phi_species = np.tile(phi_1d[:, np.newaxis, :], (1, Ny, 1))  # (Nx, Ny, 5)
    phi_total = phi_species.sum(axis=2)
    phi_total = np.minimum(phi_total, 0.99)
    return phi_species, phi_total


# ---------------------------------------------------------------------------
# 2D steady-state nutrient PDE: Newton solver
# ---------------------------------------------------------------------------

def solve_2d_nutrient(phi_2d, x_grid, y_grid, D_c=1.0, k_monod=1.0,
                      g_eff=50.0, c_inf=1.0, n_newton=40):
    """Solve -D_c Δc + g·φ(x,y)·c/(k+c) = 0 on 2D grid.

    BC:
      c(x=Lx, y) = c_inf   (Dirichlet: saliva)
      ∂c/∂x|_{x=0} = 0     (Neumann: tooth)
      ∂c/∂y|_{y=0} = 0     (Neumann: symmetry)
      ∂c/∂y|_{y=Ly} = 0    (Neumann: symmetry)

    Uses Picard (fixed-point) iteration linearizing Monod term:
      -D_c Δc^{k+1} + g·φ·c^k/(k_m+c^k) · (c^{k+1}/c^k) ≈ 0
    → -D_c Δc^{k+1} + [g·φ/(k_m+c^k)] · c^{k+1} = 0
    This is a linear system Ax=b at each iteration.

    Parameters
    ----------
    phi_2d : (Nx, Ny) total biofilm volume fraction
    x_grid, y_grid : 1D coordinate arrays
    """
    Nx, Ny = len(x_grid), len(y_grid)
    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]
    N_total = Nx * Ny

    def idx(i, j):
        return i * Ny + j

    # Initial guess: linear in x from c_inf*0.3 to c_inf
    c = np.zeros((Nx, Ny))
    for i in range(Nx):
        c[i, :] = c_inf * (0.3 + 0.7 * x_grid[i] / x_grid[-1])
    c[-1, :] = c_inf

    for it in range(n_newton):
        # Build sparse system: A c_new = rhs
        # Linearized: -D_c Δc + sigma·c = 0, sigma = g·phi/(k+c_old)
        sigma = g_eff * phi_2d / (k_monod + np.abs(c) + 1e-12)

        # Use direct assembly (dense, OK for 20x20=400)
        A_mat = np.zeros((N_total, N_total))
        rhs = np.zeros(N_total)

        for i in range(Nx):
            for j in range(Ny):
                k = idx(i, j)

                # Dirichlet BC at x=Lx (i=Nx-1)
                if i == Nx - 1:
                    A_mat[k, k] = 1.0
                    rhs[k] = c_inf
                    continue

                # Diagonal: Laplacian stencil + reaction
                diag = 0.0
                # x-direction Laplacian
                if i == 0:
                    # Neumann at x=0: ghost → c[-1,j]=c[1,j]
                    # lap_x = 2*(c[1,j]-c[0,j])/dx²
                    diag += 2.0 * D_c / dx**2
                    A_mat[k, idx(1, j)] -= 2.0 * D_c / dx**2
                else:
                    diag += 2.0 * D_c / dx**2
                    A_mat[k, idx(i-1, j)] -= D_c / dx**2
                    A_mat[k, idx(i+1, j)] -= D_c / dx**2

                # y-direction Laplacian
                if j == 0:
                    # Neumann at y=0: ghost → c[i,-1]=c[i,1]
                    diag += 2.0 * D_c / dy**2
                    if Ny > 1:
                        A_mat[k, idx(i, 1)] -= 2.0 * D_c / dy**2
                elif j == Ny - 1:
                    # Neumann at y=Ly
                    diag += 2.0 * D_c / dy**2
                    A_mat[k, idx(i, Ny-2)] -= 2.0 * D_c / dy**2
                else:
                    diag += 2.0 * D_c / dy**2
                    A_mat[k, idx(i, j-1)] -= D_c / dy**2
                    A_mat[k, idx(i, j+1)] -= D_c / dy**2

                # Reaction term (linearized Monod)
                diag += sigma[i, j]

                A_mat[k, k] = diag
                # rhs = 0 (steady state, no source except BC)

        c_flat = np.linalg.solve(A_mat, rhs)
        c_new = c_flat.reshape((Nx, Ny))
        c_new = np.clip(c_new, 0.0, c_inf)
        c_new[-1, :] = c_inf

        # Check convergence
        delta = np.max(np.abs(c_new - c))
        c = c_new
        if delta < 1e-10:
            print(f"    Newton converged at iteration {it+1} (delta={delta:.2e})")
            break

    return c


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def compute_di_2d(phi_species):
    """DI = 1 - H/ln(5) from (Nx, Ny, 5) species field."""
    phi_sum = phi_species.sum(axis=2, keepdims=True)
    phi_sum = np.maximum(phi_sum, 1e-12)
    p = phi_species / phi_sum
    p = np.clip(p, 1e-12, 1.0)
    H = -np.sum(p * np.log(p), axis=2)
    return np.clip(1.0 - H / np.log(5.0), 0.0, 1.0)


def thiele_modulus(phi_mean, g_eff, D_c, k_monod, L=1.0):
    return L * np.sqrt(g_eff * phi_mean / (D_c * k_monod))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("TMCMC -> Hamilton 0D ODE -> 2D Nutrient PDE Pipeline (Issue #3)")
    print("=" * 70)

    os.makedirs(OUT_DIR, exist_ok=True)

    # Grid
    Nx, Ny = 40, 40
    Lx, Ly = 1.0, 1.0
    x_grid = np.linspace(0, Lx, Nx)
    y_grid = np.linspace(0, Ly, Ny)

    # RD parameters (Klempt 2024)
    D_c = 1.0
    k_monod = 1.0
    c_inf = 1.0
    depth_scale = 0.4

    cases = {
        "Commensal (a35=2.41)": {"theta": THETA_COMMENSAL, "color": "#2ca02c"},
        "Dysbiotic (a35=20.94)": {"theta": THETA_DYSBIOTIC, "color": "#d62728"},
    }

    # --- Step 1: Hamilton 0D ODE ---
    print("\n[1/3] Hamilton 0D ODE (t_final=0.05) ...")
    for name, cfg in cases.items():
        t0 = time.time()
        phi_traj, phi_final = run_hamilton_0d(cfg["theta"], t_final=0.05, dt_h=1e-5)
        cfg["phi_traj"] = phi_traj
        cfg["phi_final"] = phi_final
        elapsed = time.time() - t0
        print(f"  {name}: {elapsed:.1f}s")
        print(f"    phi_final: So={phi_final[0]:.4f}, An={phi_final[1]:.4f}, "
              f"Vd={phi_final[2]:.4f}, Fn={phi_final[3]:.4f}, Pg={phi_final[4]:.4f}")

    # --- Step 2: 2D spatial profiles ---
    print("\n[2/3] Constructing 2D biofilm profiles ...")
    for name, cfg in cases.items():
        phi_species, phi_total = make_biofilm_profile_2d(
            cfg["phi_final"], x_grid, y_grid, depth_scale=depth_scale)
        cfg["phi_species_2d"] = phi_species
        cfg["phi_total_2d"] = phi_total
        phi_mean = float(phi_total.mean())
        print(f"  {name}: phi_total mean={phi_mean:.4f}, "
              f"max={phi_total.max():.4f}, min={phi_total.min():.4f}")

    # --- Step 3: 2D steady-state nutrient PDE ---
    g_eff_values = [5.0, 20.0, 50.0]
    print(f"\n[3/3] Solving 2D steady-state nutrient PDE (g_eff={g_eff_values}) ...")
    for name, cfg in cases.items():
        cfg["c_2d"] = {}
        for g_eff in g_eff_values:
            print(f"  {name}, g_eff={g_eff}:")
            t0 = time.time()
            c_sol = solve_2d_nutrient(
                cfg["phi_total_2d"], x_grid, y_grid,
                D_c=D_c, k_monod=k_monod, g_eff=g_eff, c_inf=c_inf)
            elapsed = time.time() - t0
            cfg["c_2d"][g_eff] = c_sol
            phi_m = float(cfg["phi_total_2d"].mean())
            Phi_T = thiele_modulus(phi_m, g_eff, D_c, k_monod)
            print(f"    c_min={c_sol.min():.4f}, c(tooth,mid)={c_sol[0, Ny//2]:.4f}, "
                  f"Thiele={Phi_T:.2f}, time={elapsed:.1f}s")

    # --- Visualization ---
    print("\n[Plotting] ...")
    Y, X = np.meshgrid(y_grid, x_grid)
    comm = cases["Commensal (a35=2.41)"]
    dysb = cases["Dysbiotic (a35=20.94)"]

    # ==== Figure 1: 2D heatmaps at g_eff=50 ====
    g_main = 50.0
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # Row 0: Commensal, Row 1: Dysbiotic
    # Col 0: phi_total, Col 1: c(x,y), Col 2: Pg
    for row, (cname, data) in enumerate(
            [("Commensal", comm), ("Dysbiotic", dysb)]):
        # phi_total
        ax = axes[row, 0]
        im = ax.pcolormesh(X, Y, data["phi_total_2d"], cmap="viridis", shading="auto")
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(f"{cname}\nphi_total(x,y)")
        ax.set_xlabel("x (depth)"); ax.set_ylabel("y (lateral)")
        ax.set_aspect("equal")

        # c(x,y)
        ax = axes[row, 1]
        vmin_c = min(comm["c_2d"][g_main].min(), dysb["c_2d"][g_main].min())
        im = ax.pcolormesh(X, Y, data["c_2d"][g_main], cmap="plasma",
                           shading="auto", vmin=vmin_c, vmax=1.0)
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(f"{cname}\nNutrient c (g={g_main})")
        ax.set_xlabel("x (depth)"); ax.set_ylabel("y (lateral)")
        ax.set_aspect("equal")

        # Pg
        ax = axes[row, 2]
        pg = data["phi_species_2d"][:, :, 4]
        im = ax.pcolormesh(X, Y, pg, cmap="Purples", shading="auto")
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(f"{cname}\nP. gingivalis phi_Pg")
        ax.set_xlabel("x (depth)"); ax.set_ylabel("y (lateral)")
        ax.set_aspect("equal")

    plt.suptitle(
        "TMCMC -> Hamilton 0D -> 2D Nutrient PDE\n"
        f"Commensal vs Dysbiotic (grid {Nx}x{Ny}, g_eff={g_main})",
        fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig_path = os.path.join(OUT_DIR, "hamilton_rd_2d_comparison.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {fig_path}")
    plt.close()

    # ==== Figure 2: 1D cross-sections at y=Ly/2 ====
    j_mid = Ny // 2
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    # (0,0): 0D trajectory - Commensal
    ax = axes[0, 0]
    n_traj = comm["phi_traj"].shape[0]
    t_arr = np.linspace(0, 0.05, n_traj)
    for i, (sn, col) in enumerate(zip(SPECIES_NAMES, SPECIES_COLORS)):
        ax.plot(t_arr, comm["phi_traj"][:, i], color=col, label=sn, lw=1.5)
    ax.set_xlabel("Time"); ax.set_ylabel("phi_i")
    ax.set_title("Commensal: 0D ODE"); ax.legend(fontsize=6)

    # (0,1): 0D trajectory - Dysbiotic
    ax = axes[0, 1]
    for i, (sn, col) in enumerate(zip(SPECIES_NAMES, SPECIES_COLORS)):
        ax.plot(t_arr, dysb["phi_traj"][:, i], color=col, label=sn, lw=1.5)
    ax.set_xlabel("Time"); ax.set_ylabel("phi_i")
    ax.set_title("Dysbiotic: 0D ODE"); ax.legend(fontsize=6)

    # (0,2): Final composition bar chart
    ax = axes[0, 2]
    bar_x = np.arange(5); width = 0.35
    ax.bar(bar_x - width/2, comm["phi_final"], width, color="#2ca02c", alpha=0.8, label="Commensal")
    ax.bar(bar_x + width/2, dysb["phi_final"], width, color="#d62728", alpha=0.8, label="Dysbiotic")
    ax.set_xticks(bar_x); ax.set_xticklabels(["So", "An", "Vd", "Fn", "Pg"])
    ax.set_ylabel("phi_i"); ax.set_title("Final Composition (0D)"); ax.legend()

    # (1,0): phi_total 1D cross-section
    ax = axes[1, 0]
    ax.plot(x_grid, comm["phi_total_2d"][:, j_mid], "g-", lw=2, label="Comm total")
    ax.plot(x_grid, dysb["phi_total_2d"][:, j_mid], "r--", lw=2, label="Dysb total")
    ax.plot(x_grid, comm["phi_species_2d"][:, j_mid, 4], "g:", lw=1, label="Comm Pg")
    ax.plot(x_grid, dysb["phi_species_2d"][:, j_mid, 4], "r:", lw=1, label="Dysb Pg")
    ax.set_xlabel("x (0=tooth, 1=saliva)"); ax.set_ylabel("Volume fraction")
    ax.set_title("Depth Profile (y=0.5 cross-section)"); ax.legend(fontsize=7)

    # (1,1): Nutrient at g=50
    ax = axes[1, 1]
    c_c = comm["c_2d"][g_main][:, j_mid]
    c_d = dysb["c_2d"][g_main][:, j_mid]
    ax.plot(x_grid, c_c, "g-", lw=2, label="Commensal")
    ax.plot(x_grid, c_d, "r--", lw=2, label="Dysbiotic")
    ax.fill_between(x_grid, c_c, c_d, alpha=0.15, color="gray")
    ax.set_xlabel("x (0=tooth, 1=saliva)"); ax.set_ylabel("c")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Nutrient (g={g_main}, y=0.5)"); ax.legend(fontsize=8)

    # (1,2): Nutrient sweep over g_eff
    ax = axes[1, 2]
    ls_map = {5.0: ":", 20.0: "--", 50.0: "-"}
    for g_eff in g_eff_values:
        ls = ls_map[g_eff]
        c_c = comm["c_2d"][g_eff][:, j_mid]
        c_d = dysb["c_2d"][g_eff][:, j_mid]
        phi_m_c = float(comm["phi_total_2d"].mean())
        phi_m_d = float(dysb["phi_total_2d"].mean())
        ax.plot(x_grid, c_c, "g", lw=1.5, ls=ls,
                label=f"Comm g={g_eff:.0f} (Φ={thiele_modulus(phi_m_c,g_eff,D_c,k_monod):.1f})")
        ax.plot(x_grid, c_d, "r", lw=1.5, ls=ls,
                label=f"Dysb g={g_eff:.0f} (Φ={thiele_modulus(phi_m_d,g_eff,D_c,k_monod):.1f})")
    ax.set_xlabel("x (0=tooth, 1=saliva)"); ax.set_ylabel("c")
    ax.set_ylim(0, 1.05); ax.set_title("Nutrient vs g_eff"); ax.legend(fontsize=5, ncol=2)

    plt.suptitle(
        "TMCMC -> Hamilton 0D -> 2D Nutrient PDE: Cross-Sections",
        fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig_path2 = os.path.join(OUT_DIR, "hamilton_rd_2d_cross_section.png")
    plt.savefig(fig_path2, dpi=150, bbox_inches="tight")
    print(f"  Saved: {fig_path2}")
    plt.close()

    # ==== Figure 3: Difference map (dysbiotic - commensal) ====
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    diff_phi = dysb["phi_total_2d"] - comm["phi_total_2d"]
    diff_c = dysb["c_2d"][g_main] - comm["c_2d"][g_main]
    diff_pg = dysb["phi_species_2d"][:, :, 4] - comm["phi_species_2d"][:, :, 4]

    for ax, data, label, cmap in zip(
            axes,
            [diff_phi, diff_c, diff_pg],
            ["Δ phi_total", "Δ nutrient c", "Δ P. gingivalis"],
            ["RdBu_r", "RdBu", "PiYG"]):
        vabs = max(abs(data.min()), abs(data.max()))
        if vabs < 1e-15:
            vabs = 1.0
        im = ax.pcolormesh(X, Y, data, cmap=cmap, shading="auto",
                           vmin=-vabs, vmax=vabs)
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(f"Dysbiotic - Commensal\n{label}")
        ax.set_xlabel("x (depth)"); ax.set_ylabel("y (lateral)")
        ax.set_aspect("equal")

    plt.suptitle("Condition Difference Maps (2D)", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    fig_path3 = os.path.join(OUT_DIR, "hamilton_rd_2d_difference.png")
    plt.savefig(fig_path3, dpi=150, bbox_inches="tight")
    print(f"  Saved: {fig_path3}")
    plt.close()

    # --- Summary table ---
    print("\n--- Summary Table (g_eff=50) ---")
    print(f"{'Metric':<25} {'Commensal':>12} {'Dysbiotic':>12} {'Ratio':>8}")
    print("-" * 60)

    phi_m_c = float(comm["phi_total_2d"].mean())
    phi_m_d = float(dysb["phi_total_2d"].mean())
    print(f"{'phi_total mean':<25} {phi_m_c:>12.4f} {phi_m_d:>12.4f} "
          f"{phi_m_d/max(phi_m_c,1e-12):>8.2f}")

    c_tooth_c = comm["c_2d"][g_main][0, j_mid]
    c_tooth_d = dysb["c_2d"][g_main][0, j_mid]
    print(f"{'c(tooth, y=0.5) g=50':<25} {c_tooth_c:>12.4f} {c_tooth_d:>12.4f} "
          f"{c_tooth_d/max(c_tooth_c,1e-12):>8.2f}")

    c_mean_c = float(comm["c_2d"][g_main].mean())
    c_mean_d = float(dysb["c_2d"][g_main].mean())
    print(f"{'c mean g=50':<25} {c_mean_c:>12.4f} {c_mean_d:>12.4f} "
          f"{c_mean_d/max(c_mean_c,1e-12):>8.2f}")

    Phi_c = thiele_modulus(phi_m_c, 50.0, D_c, k_monod)
    Phi_d = thiele_modulus(phi_m_d, 50.0, D_c, k_monod)
    print(f"{'Thiele mod g=50':<25} {Phi_c:>12.2f} {Phi_d:>12.2f} "
          f"{Phi_d/max(Phi_c,1e-12):>8.2f}")

    pg_max_c = comm["phi_species_2d"][:, :, 4].max()
    pg_max_d = dysb["phi_species_2d"][:, :, 4].max()
    print(f"{'Pg max':<25} {pg_max_c:>12.6f} {pg_max_d:>12.6f} "
          f"{pg_max_d/max(pg_max_c,1e-12):>8.2f}")

    print(f"\nAll results in: {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
