#!/usr/bin/env python3
"""
run_coupled_staggered.py — Klempt-style staggered growth-mechanics coupling
============================================================================

Quasi-static staggered (operator-split) time loop — each growth step dt_growth:
  (0) Hamilton reaction with LOCAL nutrient c(x,y) → φ evolves spatially
  (1) Species diffusion (explicit Euler, Neumann BCs)
  (2) Nutrient PDE (CFL-stable, mixed BCs: Dirichlet top, Neumann bottom/sides)
  (3) Accumulate growth α(x,y) += Σ wᵢ·φᵢ · Monod · f(σ) · dt
  (4) DI(x,y) → E(x,y)  (spatially varying due to nutrient-modulated φ)
  (5) Quasi-static FEM solve → σ(x,y), u(x,y)

Key improvements over previous adiabatic version:
  - Hamilton reaction at each growth step with per-node nutrient coupling
    (c_effective = c_hamilton × c_nutrient / c_boundary)
  - Species-specific growth weights (EPS production capacity)
  - Stress-dependent growth modulation: f(σ) = max(0, 1 - σ_vm/σ_crit)
  - Mixed nutrient BCs (Dirichlet top / Neumann bottom,sides)
  - Plane stress option for thin biofilm
  - Equilibrium ψ from 0D solver (not hardcoded 0.999)

Usage:
    python run_coupled_staggered.py --condition dh_baseline --nx 25 --ny 25
    python run_coupled_staggered.py --condition all
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))

from core_hamilton_2d_nutrient import (
    Config2D,
    _make_newton_step_vmap_localc,
    _make_nutrient_step_mixed,
    _make_nutrient_step_stable,
    diffusion_step_species_2d,
    theta_to_matrices,
)
from solve_stress_2d import solve_2d_fem
from material_models import compute_di, compute_E_phi_pg, E_MAX_PA, E_MIN_PA

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


# Species-specific growth weights (EPS production capacity)
# Weights reflect relative contribution to biofilm matrix expansion:
#   So (1.0): Primary EPS producer — glucosyltransferases (gtfB/C/D) synthesize
#             water-insoluble glucans, the main structural scaffold.
#             Ref: Bowen & Koo (2011) Caries Res 45:69; Xiao & Koo (2010)
#   An (0.8): Moderate EPS — fimbriae-mediated coaggregation + heteropolysaccharide.
#             Ref: Palmer et al. (2003) Microbiology 149:3507
#   Vei (0.6): Low EPS — primarily metabolic (lactate consumer), limited matrix.
#             Ref: Mashima & Nakazawa (2014) Curr Oral Health Rep 1:138
#   Fn (0.5): Bridge organism — outer membrane adhesins (RadD, Fap2) rather than
#             polysaccharide EPS. Moderate structural via coaggregation.
#             Ref: Kolenbrander et al. (2010) Nat Rev Microbiol 8:471
#   Pg (0.3): Keystone pathogen — gingipains degrade matrix components.
#             Net negative contribution to structural EPS.
#             Ref: Hajishengallis et al. (2012) Nat Rev Microbiol 10:717
# Ranking consistent with Kenney & Ash (1969) classification of EPS producers
# and Flemming et al. (2016) Nat Rev Microbiol 14:563 (general biofilm matrix).
GROWTH_WEIGHTS = np.array([1.0, 0.8, 0.6, 0.5, 0.3])


# ============================================================================
# 0D equilibration (Numba) — match TMCMC calibration
# ============================================================================


def _equilibrate_0d_numba(theta, maxtimestep=2500, dt=1e-5, K_hill=0.05, n_hill=4.0):
    """
    Run 0D Hamilton ODE (Numba solver, 50 Newton iter + line search) to find
    the quasi-stationary state at the TMCMC calibration time.

    Returns (phi_eq, psi_eq, gamma_eq):
      phi_eq   : (5,) species fractions
      psi_eq   : (5,) viability variables
      gamma_eq : float  Lagrange multiplier
    """
    sys.path.insert(0, str(_HERE.parent.parent / "tmcmc" / "program2602"))
    from improved_5species_jit import BiofilmNewtonSolver5S

    solver = BiofilmNewtonSolver5S(
        dt=dt, maxtimestep=maxtimestep, eps=1e-8, K_hill=K_hill, n_hill=n_hill
    )
    _t_arr, g_arr = solver.run_deterministic(theta[:20])
    g_final = g_arr[-1].copy()
    phi_eq = g_final[:5]
    psi_eq = g_final[6:11]
    gamma_eq = g_final[11]
    return phi_eq, psi_eq, gamma_eq


def _make_initial_state_from_0d(cfg, phi_eq, psi_eq, gamma_eq):
    """
    Build 2D initial state G (Nx*Ny, 12) using the FULL 0D equilibrium
    (φ, ψ, γ), plus small spatial perturbation for symmetry breaking.
    """
    Nx, Ny = cfg.Nx, cfg.Ny
    G_2d = jnp.zeros((Nx, Ny, 12), dtype=jnp.float64)

    # Uniform equilibrium φ with tiny random perturbation (~1%)
    rng = np.random.RandomState(42)
    for s in range(5):
        noise = rng.uniform(-0.01, 0.01, (Nx, Ny)) * phi_eq[s]
        G_2d = G_2d.at[:, :, s].set(phi_eq[s] + noise)

    # Clip and normalize
    G_2d = G_2d.at[:, :, :5].set(jnp.clip(G_2d[:, :, :5], 1e-10, None))
    phi_sum = jnp.sum(G_2d[:, :, :5], axis=-1)
    scale = jnp.where(phi_sum > 0.999, 0.999 / phi_sum, 1.0)
    G_2d = G_2d.at[:, :, :5].set(G_2d[:, :, :5] * scale[:, :, None])

    # phi_0 = 1 - sum(phi)
    G_2d = G_2d.at[:, :, 5].set(1.0 - jnp.sum(G_2d[:, :, :5], axis=-1))

    # ψ from 0D equilibrium (NOT hardcoded 0.999)
    for s in range(5):
        G_2d = G_2d.at[:, :, 6 + s].set(jnp.clip(psi_eq[s], 1e-6, 1.0 - 1e-6))
    G_2d = G_2d.at[:, :, 11].set(gamma_eq)

    G = G_2d.reshape(Nx * Ny, 12)

    # Nutrient field: uniform c = c_boundary
    c = jnp.full((Nx, Ny), cfg.c_boundary, dtype=jnp.float64)

    return G, c


# ============================================================================
# Helpers
# ============================================================================


def interpolate_elem_to_node(elem_field, Nx, Ny):
    """Interpolate element-centered field to nodes by averaging adjacent elements."""
    n_ex, n_ey = Nx - 1, Ny - 1
    field_2d = elem_field.reshape(n_ex, n_ey)
    node_field = np.zeros((Nx, Ny))
    count = np.zeros((Nx, Ny))
    for ex in range(n_ex):
        for ey in range(n_ey):
            val = field_2d[ex, ey]
            for ni, nj in [(ex, ey), (ex + 1, ey), (ex + 1, ey + 1), (ex, ey + 1)]:
                node_field[ni, nj] += val
                count[ni, nj] += 1
    return node_field / np.maximum(count, 1)


# ============================================================================
# Staggered coupled solver
# ============================================================================


def run_staggered_coupled(
    theta,
    cfg,
    nu=0.30,
    k_alpha=0.05,
    e_model="phi_pg",
    fem_every=None,
    dt_growth=0.1,
    n_growth_steps=50,
    ode_init_steps=2500,
    ode_adjust_steps=100,
    sigma_crit=0.0,
    stress_type="plane_strain",
    nutrient_bc="mixed",
    alpha_max=0.0,
):
    """
    Quasi-static staggered growth-mechanics coupling (Klempt 2024 style).

    At each growth step dt_growth:
      (0) Hamilton reaction with local c  (ode_adjust_steps Newton steps)
      (1) Species diffusion               (explicit Euler)
      (2) Nutrient PDE                    (CFL-stable mixed BCs)
      (3) α += Σ wᵢ·φᵢ · Monod · f(σ)   (species-specific + stress mod)
      (4) DI → E                          (from evolved φ)
      (5) FEM → σ, u                      (quasi-static)

    Parameters
    ----------
    sigma_crit : float — stress threshold for growth inhibition [Pa].
        0 = no stress feedback (backward compatible).
        f(σ) = max(0, 1 - σ_vm/σ_crit)
    stress_type : str — "plane_strain" or "plane_stress"
    nutrient_bc : str — "mixed" (Dirichlet top, Neumann others) or "dirichlet" (all walls)
    alpha_max : float — logistic growth saturation limit.
        0 = no saturation (linear growth, backward compatible).
        > 0: α_dot *= (1 - α/α_max), giving logistic saturation.
        Klempt (2024) uses α_max ≈ 0.3 for volumetric growth.
    """
    A, b_diag = theta_to_matrices(jnp.asarray(theta, dtype=jnp.float64))
    active_mask = jnp.ones(5, dtype=jnp.int64)

    params = {
        "dt_h": cfg.dt_h,
        "Kp1": cfg.Kp1,
        "Eta": jnp.ones(5),
        "EtaPhi": jnp.ones(5),
        "c": cfg.c_hamilton,  # unused when local c is provided
        "alpha": cfg.alpha,
        "K_hill": jnp.array(cfg.K_hill),
        "n_hill": jnp.array(cfg.n_hill),
        "A": A,
        "b_diag": b_diag,
        "active_mask": active_mask,
    }

    # JIT-compiled Newton step with per-node nutrient c
    _newton_vmap_c = _make_newton_step_vmap_localc(cfg.newton_iters)

    # Nutrient PDE step
    if nutrient_bc == "mixed":
        _nutrient_fn = _make_nutrient_step_mixed(30)
    else:
        _nutrient_fn = _make_nutrient_step_stable(30)

    D_eff = jnp.array(cfg.D_eff)
    D_c = cfg.D_c
    k_M = cfg.k_monod
    g_cons = jnp.array(cfg.g_consumption)
    c_bc = cfg.c_boundary
    Nx, Ny = cfg.Nx, cfg.Ny
    Lx, Ly = cfg.Lx, cfg.Ly

    save_every = fem_every or max(1, n_growth_steps // 10)

    # --- 0D equilibration: full state using Numba solver (TMCMC-consistent) ---
    print(f"  Equilibrating ODE via 0D Numba solver ({ode_init_steps} steps)...")
    phi_eq, psi_eq, gamma_eq = _equilibrate_0d_numba(
        theta,
        maxtimestep=ode_init_steps,
        dt=cfg.dt_h,
        K_hill=cfg.K_hill,
        n_hill=cfg.n_hill,
    )
    print(f"  φ_eq = {phi_eq.round(4)}")
    print(f"  ψ_eq = {psi_eq.round(4)}")
    di_eq = compute_di(phi_eq.reshape(1, 1, 5))[0, 0]
    print(f"  DI_eq = {di_eq:.4f}")

    # Initialize 2D state from full 0D equilibrium (φ + ψ + γ)
    G, c = _make_initial_state_from_0d(cfg, phi_eq, psi_eq, gamma_eq)

    # Growth variable α(x,y) — accumulates over time
    alpha_field = np.zeros((Nx, Ny))
    # Stress at nodes (for growth modulation feedback)
    sigma_vm_node = np.zeros((Nx, Ny))

    # Storage for time-evolution snapshots
    snaps = {
        "t": [],
        "phi": [],  # (5, Nx, Ny)
        "c": [],  # (Nx, Ny)
        "DI": [],  # (Nx, Ny)
        "E": [],  # (Nx, Ny)
        "alpha": [],  # (Nx, Ny)
        "eps_growth": [],  # (Nx, Ny)
        "sigma_vm": [],  # (n_elem,)
        "u_max": [],  # scalar
        "sigma_vm_max": [],  # scalar
        "sigma_vm_mean": [],  # scalar
        "geom_nonlin": [],  # scalar: |∇u|/|ε| ratio (< 0.05 ≈ linear OK)
    }

    # Save initial state
    phi_2d_init = np.asarray(G.reshape(Nx, Ny, 12)[:, :, :5])
    _save_snapshot(
        snaps,
        0.0,
        phi_2d_init,
        np.asarray(c),
        alpha_field,
        Nx,
        Ny,
        Lx,
        Ly,
        nu,
        e_model,
        stress_type,
    )

    mode_str = "QUASI-STATIC" if ode_adjust_steps > 0 else "ADIABATIC"
    stress_str = f"σ_crit={sigma_crit:.0f}Pa" if sigma_crit > 0 else "no σ feedback"
    alpha_str = f"α_max={alpha_max:.2f}" if alpha_max > 0 else "no saturation"
    print(f"\n{'='*70}")
    print(f"  {mode_str} STAGGERED  |  Nx={Nx} Ny={Ny}  |  {n_growth_steps} growth steps")
    print(f"  dt_growth={dt_growth:.2e} [hours]  ODE adjust: {ode_adjust_steps} steps/growth")
    print(f"  k_alpha={k_alpha}  FEM every {save_every} steps  |  E model: {e_model}")
    print(f"  {stress_str}  |  {alpha_str}  |  {stress_type}  |  nutrient BC: {nutrient_bc}")
    print(f"{'='*70}\n")

    t0_wall = time.perf_counter()
    c_hamilton = float(cfg.c_hamilton)
    c_boundary = float(cfg.c_boundary)

    for step in range(1, n_growth_steps + 1):
        t_growth = step * dt_growth

        # --- (0) Hamilton reaction with LOCAL nutrient coupling ---
        if ode_adjust_steps > 0:
            # c_effective(x,y) = c_hamilton × c_nutrient(x,y) / c_boundary
            # At boundary: c_eff = c_hamilton (matches TMCMC calibration)
            # In depleted regions: c_eff < c_hamilton → weaker interactions → φ shifts
            c_np_for_ode = np.asarray(c)
            c_eff_flat = jnp.asarray(c_np_for_ode.reshape(-1) * (c_hamilton / c_boundary))
            for _ in range(ode_adjust_steps):
                G = _newton_vmap_c(G, c_eff_flat, params)

        # --- (1) Species diffusion ---
        phi_2d = G.reshape(Nx, Ny, 12)[:, :, :5]
        phi_2d = diffusion_step_species_2d(phi_2d, D_eff, dt_growth, cfg.dx, cfg.dy)

        # Write back to G
        G_2d = G.reshape(Nx, Ny, 12)
        G_2d = G_2d.at[:, :, :5].set(phi_2d)
        G_2d = G_2d.at[:, :, 5].set(1.0 - jnp.sum(phi_2d, axis=-1))
        G = G_2d.reshape(Nx * Ny, 12)

        # --- (2) Nutrient PDE ---
        c = _nutrient_fn(c, phi_2d, D_c, k_M, g_cons, c_bc, cfg.dx, cfg.dy, dt_growth)

        # --- (3) Accumulate growth α with species weights + stress modulation ---
        phi_2d_np = np.asarray(phi_2d)
        c_np = np.asarray(c)
        # Species-specific weighted sum: Σ wᵢ · φᵢ
        weighted_phi = np.sum(GROWTH_WEIGHTS[None, None, :] * phi_2d_np, axis=-1)
        monod = c_np / (k_M + c_np)
        # Stress-dependent growth modulation
        if sigma_crit > 0:
            stress_mod = np.clip(1.0 - sigma_vm_node / sigma_crit, 0.0, 1.0)
        else:
            stress_mod = 1.0
        # Logistic saturation: dα/dt *= (1 - α/α_max)
        if alpha_max > 0:
            logistic = np.clip(1.0 - alpha_field / alpha_max, 0.0, 1.0)
        else:
            logistic = 1.0
        alpha_field += k_alpha * weighted_phi * monod * stress_mod * logistic * dt_growth

        # --- (4-5) FEM solve at snapshot intervals ---
        if step % save_every == 0 or step == n_growth_steps:
            _save_snapshot(
                snaps,
                float(t_growth),
                phi_2d_np,
                c_np,
                alpha_field,
                Nx,
                Ny,
                Lx,
                Ly,
                nu,
                e_model,
                stress_type,
            )

            # Update σ_vm at nodes for next step's stress modulation
            svm_elem = snaps["sigma_vm"][-1]
            if svm_elem.size == (Nx - 1) * (Ny - 1):
                sigma_vm_node = interpolate_elem_to_node(svm_elem, Nx, Ny)

            pct = 100 * step / n_growth_steps
            svm_max = snaps["sigma_vm_max"][-1]
            u_max_val = snaps["u_max"][-1]
            alpha_cur_max = alpha_field.max()
            DI_min = snaps["DI"][-1].min()
            DI_max = snaps["DI"][-1].max()
            E_min_val = snaps["E"][-1].min()
            E_max_val = snaps["E"][-1].max()
            print(
                f"  [{pct:5.1f}%] t_g={t_growth:.2f}  "
                f"DI=[{DI_min:.3f},{DI_max:.3f}]  "
                f"E=[{E_min_val:.0f},{E_max_val:.0f}]  "
                f"α={alpha_cur_max:.4f}  "
                f"σ_vm={svm_max:.1f} Pa  "
                f"|u|={u_max_val:.2e}"
            )

    elapsed = time.perf_counter() - t0_wall
    print(f"\n  Total wall time: {elapsed:.1f}s")
    print(f"  Snapshots: {len(snaps['t'])}")

    # Convert lists to arrays
    for key in ["t", "u_max", "sigma_vm_max", "sigma_vm_mean", "geom_nonlin"]:
        snaps[key] = np.array(snaps[key])
    for key in ["phi", "c", "DI", "E", "alpha", "eps_growth", "sigma_vm"]:
        snaps[key] = np.array(snaps[key])

    snaps["Nx"] = Nx
    snaps["Ny"] = Ny
    snaps["Lx"] = Lx
    snaps["Ly"] = Ly
    snaps["e_model"] = e_model
    snaps["elapsed_s"] = round(elapsed, 1)

    return snaps


def _save_snapshot(
    snaps,
    t,
    phi_2d_np,
    c_np,
    alpha_field,
    Nx,
    Ny,
    Lx,
    Ly,
    nu,
    e_model,
    stress_type="plane_strain",
):
    """Compute derived fields + FEM solve and store snapshot."""
    DI = compute_di(phi_2d_np)

    if e_model == "phi_pg":
        E_field = compute_E_phi_pg(phi_2d_np)
    elif e_model == "virulence":
        from material_models import compute_E_virulence

        E_field = compute_E_virulence(phi_2d_np)
    else:
        from material_models import compute_E_di

        E_field = compute_E_di(DI)

    eps_growth = alpha_field / 3.0

    # FEM solve
    geom_nonlin = 0.0
    if alpha_field.max() > 1e-12:
        fem = solve_2d_fem(
            E_field,
            nu,
            eps_growth,
            Nx,
            Ny,
            Lx,
            Ly,
            bc_type="bottom_fixed",
            stress_type=stress_type,
        )
        sigma_vm = fem["sigma_vm"]
        u_mag = np.sqrt(fem["u"][:, 0] ** 2 + fem["u"][:, 1] ** 2)
        u_max = float(u_mag.max())
        svm_max = float(sigma_vm.max())
        svm_mean = float(sigma_vm.mean())
        geom_nonlin = fem.get("geom_nonlin_ratio", 0.0)
    else:
        n_elem = (Nx - 1) * (Ny - 1)
        sigma_vm = np.zeros(n_elem)
        u_max = 0.0
        svm_max = 0.0
        svm_mean = 0.0

    snaps["t"].append(t)
    snaps["phi"].append(phi_2d_np.transpose(2, 0, 1).copy())  # (5, Nx, Ny)
    snaps["c"].append(c_np.copy())
    snaps["DI"].append(DI.copy())
    snaps["E"].append(E_field.copy())
    snaps["alpha"].append(alpha_field.copy())
    snaps["eps_growth"].append(eps_growth.copy())
    snaps["sigma_vm"].append(sigma_vm.copy())
    snaps["u_max"].append(u_max)
    snaps["sigma_vm_max"].append(svm_max)
    snaps["sigma_vm_mean"].append(svm_mean)
    snaps["geom_nonlin"].append(geom_nonlin)


# ============================================================================
# Visualization: time-evolution panel
# ============================================================================


def plot_time_evolution(snaps, outpath, condition="demo"):
    """6-row time-evolution figure showing coupled growth-mechanics."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t_arr = snaps["t"]
    n_snap = len(t_arr)
    Nx, Ny = snaps["Nx"], snaps["Ny"]
    n_ex, n_ey = Nx - 1, Ny - 1

    if n_snap <= 5:
        idx = list(range(n_snap))
    else:
        idx = [0]
        step = (n_snap - 1) / 4
        for i in range(1, 4):
            idx.append(int(round(i * step)))
        idx.append(n_snap - 1)
        idx = sorted(set(idx))

    n_cols = len(idx)
    fig, axes = plt.subplots(6, n_cols, figsize=(3.5 * n_cols, 18))
    if n_cols == 1:
        axes = axes[:, None]

    row_labels = [
        "φ_total",
        "Nutrient c",
        "DI(x,y)",
        "E(x,y) [Pa]",
        "α(x,y)",
        "σ_vm [Pa]",
    ]
    cmaps = ["YlGn", "Blues", "RdYlGn_r", "viridis", "hot", "jet"]

    for col_i, snap_i in enumerate(idx):
        t_val = t_arr[snap_i]
        phi_snap = snaps["phi"][snap_i]
        c_snap = snaps["c"][snap_i]
        DI_snap = snaps["DI"][snap_i]
        E_snap = snaps["E"][snap_i]
        alpha_snap = snaps["alpha"][snap_i]
        svm_snap = snaps["sigma_vm"][snap_i]

        phi_total = phi_snap.sum(axis=0)
        fields = [phi_total, c_snap, DI_snap, E_snap, alpha_snap]

        if svm_snap.size == n_ex * n_ey:
            svm_2d = svm_snap.reshape(n_ex, n_ey)
        else:
            svm_2d = np.zeros((n_ex, n_ey))

        for row_i in range(5):
            ax = axes[row_i, col_i]
            im = ax.imshow(
                fields[row_i].T,
                origin="lower",
                cmap=cmaps[row_i],
                aspect="equal",
                extent=[0, 1, 0, 1],
            )
            if col_i == n_cols - 1:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xticks([])
            ax.set_yticks([])
            if col_i == 0:
                ax.set_ylabel(row_labels[row_i], fontsize=10)
            if row_i == 0:
                ax.set_title(f"t = {t_val:.4f}", fontsize=10)

        ax = axes[5, col_i]
        im = ax.imshow(
            svm_2d.T,
            origin="lower",
            cmap="jet",
            aspect="equal",
            extent=[0, 1, 0, 1],
        )
        if col_i == n_cols - 1:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks([])
        ax.set_yticks([])
        if col_i == 0:
            ax.set_ylabel(row_labels[5], fontsize=10)

    fig.suptitle(
        f"Staggered Coupled Growth-Mechanics: {condition}\n"
        f"(Klempt 2024 style — {n_snap} time steps, "
        f"{snaps['elapsed_s']}s wall time)",
        fontsize=13,
        weight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure: {outpath}")


def plot_time_series(snaps, outpath, condition="demo"):
    """Time series of σ_vm_max, |u|_max, α_max."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = snaps["t"]
    svm_max = snaps["sigma_vm_max"]
    u_max = snaps["u_max"]
    alpha_max = (
        np.array([a.max() for a in snaps["alpha"]])
        if isinstance(snaps["alpha"], np.ndarray)
        else np.array([a.max() for a in snaps["alpha"]])
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    ax = axes[0]
    ax.plot(t, svm_max, "r-o", ms=3, lw=1.5)
    ax.set_xlabel("t (growth time)")
    ax.set_ylabel("σ_vm,max [Pa]")
    ax.set_title("Max von Mises stress")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t, u_max, "b-o", ms=3, lw=1.5)
    ax.set_xlabel("t")
    ax.set_ylabel("|u|_max")
    ax.set_title("Max displacement")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(t, alpha_max, "g-o", ms=3, lw=1.5)
    ax.set_xlabel("t")
    ax.set_ylabel("α_max")
    ax.set_title("Max growth variable")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Time evolution: {condition}", fontsize=12, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(outpath, dpi=180)
    plt.close(fig)
    print(f"  Figure: {outpath}")


# ============================================================================
# Multi-condition comparison
# ============================================================================


def _run_single_condition_subprocess(cond_dir, args):
    """Run a single condition in a subprocess to isolate JAX LLVM memory."""
    import subprocess as sp

    cmd = [
        sys.executable,
        str(_HERE / "run_coupled_staggered.py"),
        "--condition",
        cond_dir,
        "--nx",
        str(args.nx),
        "--ny",
        str(args.ny),
        "--dt-h",
        str(args.dt_h),
        "--dt-growth",
        str(args.dt_growth),
        "--n-growth-steps",
        str(args.n_growth_steps),
        "--ode-init-steps",
        str(args.ode_init_steps),
        "--ode-adjust-steps",
        str(args.ode_adjust_steps),
        "--k-hill",
        str(args.k_hill),
        "--n-hill",
        str(args.n_hill),
        "--nu",
        str(args.nu),
        "--k-alpha",
        str(args.k_alpha),
        "--e-model",
        args.e_model,
        "--sigma-crit",
        str(args.sigma_crit),
        "--stress-type",
        args.stress_type,
        "--nutrient-bc",
        args.nutrient_bc,
        "--alpha-max",
        str(args.alpha_max),
        "--outdir",
        str(args.outdir),
        "--save-npz",
    ]
    print(f"\n  >>> subprocess: {cond_dir}")
    proc = sp.run(cmd, capture_output=False, text=True)
    return proc.returncode


def run_4_conditions(args):
    """Run staggered coupling for all 4 conditions via subprocess isolation."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    conditions = {
        "commensal_static": "CS",
        "commensal_hobic": "CH",
        "dh_baseline": "DH",
        "dysbiotic_static": "DS",
    }

    for cond_dir, label in conditions.items():
        rc = _run_single_condition_subprocess(cond_dir, args)
        if rc != 0:
            print(f"  WARNING: {cond_dir} failed (rc={rc})")

    all_snaps = {}
    for cond_dir, label in conditions.items():
        npz_path = outdir / f"coupled_snaps_{cond_dir}.npz"
        if not npz_path.exists():
            print(f"  SKIP {label}: {npz_path} not found")
            continue
        data = np.load(npz_path, allow_pickle=True)
        snaps = {k: data[k] for k in data.files}
        snaps["Nx"] = int(snaps["Nx"])
        snaps["Ny"] = int(snaps["Ny"])
        snaps["Lx"] = float(snaps["Lx"])
        snaps["Ly"] = float(snaps["Ly"])
        snaps["e_model"] = str(snaps["e_model"])
        snaps["elapsed_s"] = float(snaps["elapsed_s"])
        all_snaps[label] = snaps

    if len(all_snaps) < 2:
        print("Not enough conditions for comparison.")
        return all_snaps

    # --- Comparison figure ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = {"CS": "#2196F3", "CH": "#4CAF50", "DH": "#FF9800", "DS": "#F44336"}

    for label, snaps in all_snaps.items():
        t = snaps["t"]
        col = colors.get(label, "gray")
        axes[0].plot(t, snaps["sigma_vm_max"], "-o", color=col, ms=3, lw=1.5, label=label)
        axes[1].plot(t, snaps["u_max"], "-o", color=col, ms=3, lw=1.5, label=label)
        alpha_max = np.array([snaps["alpha"][i].max() for i in range(len(t))])
        axes[2].plot(t, alpha_max, "-o", color=col, ms=3, lw=1.5, label=label)

    for i, (ylabel, title) in enumerate(
        [
            ("σ_vm,max [Pa]", "Max von Mises stress"),
            ("|u|_max", "Max displacement"),
            ("α_max", "Max growth variable"),
        ]
    ):
        axes[i].set_xlabel("t (growth time)")
        axes[i].set_ylabel(ylabel)
        axes[i].set_title(title)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    fig.suptitle(
        "Staggered Coupled: 4-Condition Comparison (Klempt-style)",
        fontsize=13,
        weight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    cmp_path = outdir / "coupled_comparison_4cond.png"
    fig.savefig(cmp_path, dpi=200)
    plt.close(fig)
    print(f"\n  Comparison figure: {cmp_path}")

    # --- Final-state comparison ---
    fig, axes = plt.subplots(3, len(all_snaps), figsize=(4 * len(all_snaps), 10))
    if len(all_snaps) == 1:
        axes = axes[:, None]

    for col_i, (label, snaps) in enumerate(all_snaps.items()):
        Nx, Ny = snaps["Nx"], snaps["Ny"]
        n_ex, n_ey = Nx - 1, Ny - 1

        DI_final = snaps["DI"][-1]
        svm_final = snaps["sigma_vm"][-1]
        E_final = snaps["E"][-1]

        if svm_final.size == n_ex * n_ey:
            svm_2d = svm_final.reshape(n_ex, n_ey)
        else:
            svm_2d = np.zeros((n_ex, n_ey))

        ax = axes[0, col_i]
        im = ax.imshow(
            DI_final.T,
            origin="lower",
            cmap="RdYlGn_r",
            aspect="equal",
            extent=[0, 1, 0, 1],
            vmin=0,
            vmax=1,
        )
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(f"{label}: DI", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

        ax = axes[1, col_i]
        im = ax.imshow(
            E_final.T,
            origin="lower",
            cmap="viridis",
            aspect="equal",
            extent=[0, 1, 0, 1],
            vmin=E_MIN_PA,
            vmax=E_MAX_PA,
        )
        plt.colorbar(im, ax=ax, fraction=0.046, label="Pa")
        ax.set_title("E(x,y)", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

        ax = axes[2, col_i]
        im = ax.imshow(svm_2d.T, origin="lower", cmap="jet", aspect="equal", extent=[0, 1, 0, 1])
        plt.colorbar(im, ax=ax, fraction=0.046, label="Pa")
        ax.set_title(f"σ_vm (max={svm_2d.max():.1f})", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        "Final State: DI / E / σ_vm across 4 conditions (staggered coupled)",
        fontsize=13,
        weight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    final_path = outdir / "coupled_final_state_4cond.png"
    fig.savefig(final_path, dpi=200)
    plt.close(fig)
    print(f"  Final state figure: {final_path}")

    return all_snaps


# ============================================================================
# CLI
# ============================================================================


def main():
    ap = argparse.ArgumentParser(
        description="Klempt-style staggered coupled growth-mechanics solver"
    )
    ap.add_argument(
        "--condition",
        default=None,
        help="Single condition or 'all' for 4-condition comparison",
    )
    ap.add_argument("--nx", type=int, default=20)
    ap.add_argument("--ny", type=int, default=20)
    # ODE parameters
    ap.add_argument("--dt-h", type=float, default=1e-5, help="Hamilton ODE time step")
    ap.add_argument(
        "--ode-init-steps",
        type=int,
        default=2500,
        help="ODE steps for initial equilibration (matching TMCMC maxtimestep)",
    )
    ap.add_argument(
        "--ode-adjust-steps",
        type=int,
        default=100,
        help="ODE steps per growth step with local nutrient coupling (0=adiabatic)",
    )
    # Growth (slow) timescale parameters
    ap.add_argument("--dt-growth", type=float, default=0.1, help="Growth step size")
    ap.add_argument("--n-growth-steps", type=int, default=50, help="Number of growth steps")
    # Physics
    ap.add_argument("--k-hill", type=float, default=0.05)
    ap.add_argument("--n-hill", type=float, default=4.0)
    ap.add_argument("--nu", type=float, default=0.30)
    ap.add_argument("--k-alpha", type=float, default=0.05)
    ap.add_argument("--e-model", choices=["phi_pg", "virulence", "di"], default="phi_pg")
    # New physics options
    ap.add_argument(
        "--sigma-crit",
        type=float,
        default=0.0,
        help="Stress threshold for growth inhibition [Pa]. 0=no feedback.",
    )
    ap.add_argument(
        "--stress-type",
        choices=["plane_strain", "plane_stress"],
        default="plane_strain",
        help="Plane strain (thick) or plane stress (thin biofilm)",
    )
    ap.add_argument(
        "--nutrient-bc",
        choices=["mixed", "dirichlet"],
        default="mixed",
        help="Nutrient BC: mixed (Dirichlet top, Neumann others) or dirichlet (all walls)",
    )
    ap.add_argument(
        "--alpha-max",
        type=float,
        default=0.0,
        help="Logistic growth saturation limit. 0=no cap. Klempt uses ~0.3.",
    )
    ap.add_argument("--outdir", default=None)
    ap.add_argument(
        "--save-npz",
        action="store_true",
        help="Save results as .npz for later comparison",
    )
    ap.add_argument(
        "--theta-json",
        default=None,
        help="Direct path to theta JSON file (overrides condition-based lookup)",
    )
    args = ap.parse_args()

    _TMCMC = _HERE.parent.parent
    outdir = args.outdir or str(_TMCMC / "FEM" / "figures" / "coupled_staggered")
    args.outdir = outdir
    Path(outdir).mkdir(parents=True, exist_ok=True)

    if args.condition is None or args.condition == "all":
        run_4_conditions(args)
    else:
        theta = None
        # Direct theta file takes priority
        if args.theta_json and Path(args.theta_json).exists():
            with open(args.theta_json) as f:
                d = json.load(f)
            theta = np.array(d.get("theta_full", d.get("theta_sub")), dtype=np.float64)
        else:
            _RUNS = _TMCMC / "data_5species" / "_runs"
            theta_path = _RUNS / args.condition / "theta_MAP.json"
            if theta_path.exists():
                with open(theta_path) as f:
                    d = json.load(f)
                theta = np.array(d.get("theta_full", d.get("theta_sub")), dtype=np.float64)

        if theta is None:
            from core_hamilton_2d_nutrient import THETA_DEMO

            theta = THETA_DEMO
            print(f"  WARNING: No theta found, using demo theta")

        cfg = Config2D(
            Nx=args.nx,
            Ny=args.ny,
            dt_h=args.dt_h,
            n_react_sub=1,
            n_macro=args.n_growth_steps,
            K_hill=args.k_hill,
            n_hill=args.n_hill,
        )

        snaps = run_staggered_coupled(
            theta,
            cfg,
            nu=args.nu,
            k_alpha=args.k_alpha,
            e_model=args.e_model,
            dt_growth=args.dt_growth,
            n_growth_steps=args.n_growth_steps,
            ode_init_steps=args.ode_init_steps,
            ode_adjust_steps=args.ode_adjust_steps,
            sigma_crit=args.sigma_crit,
            stress_type=args.stress_type,
            nutrient_bc=args.nutrient_bc,
            alpha_max=args.alpha_max,
        )

        plot_time_evolution(
            snaps,
            Path(outdir) / f"coupled_evolution_{args.condition}.png",
            condition=args.condition,
        )
        plot_time_series(
            snaps,
            Path(outdir) / f"coupled_timeseries_{args.condition}.png",
            condition=args.condition,
        )

        if args.save_npz:
            npz_path = Path(outdir) / f"coupled_snaps_{args.condition}.npz"
            np.savez_compressed(
                npz_path,
                t=snaps["t"],
                phi=snaps["phi"],
                c=snaps["c"],
                DI=snaps["DI"],
                E=snaps["E"],
                alpha=snaps["alpha"],
                eps_growth=snaps["eps_growth"],
                sigma_vm=snaps["sigma_vm"],
                u_max=snaps["u_max"],
                sigma_vm_max=snaps["sigma_vm_max"],
                sigma_vm_mean=snaps["sigma_vm_mean"],
                geom_nonlin=snaps["geom_nonlin"],
                Nx=snaps["Nx"],
                Ny=snaps["Ny"],
                Lx=snaps["Lx"],
                Ly=snaps["Ly"],
                e_model=snaps["e_model"],
                elapsed_s=snaps["elapsed_s"],
            )
            print(f"  Saved: {npz_path}")

        print(f"\n  Final DI: [{snaps['DI'][-1].min():.3f}, {snaps['DI'][-1].max():.3f}]")
        print(f"  Final E:  [{snaps['E'][-1].min():.1f}, {snaps['E'][-1].max():.1f}] Pa")
        print(f"  Final α:  [{snaps['alpha'][-1].min():.5f}, {snaps['alpha'][-1].max():.5f}]")
        print(f"  Final σ_vm_max: {snaps['sigma_vm_max'][-1]:.2f} Pa")
        print(f"  Final |u|_max:  {snaps['u_max'][-1]:.2e}")
        gnl = snaps["geom_nonlin"][-1]
        print(
            f"  Geom nonlin ratio: {gnl:.4f} ({'LINEAR OK' if gnl < 0.05 else 'WARNING: nonlinear'})"
        )

    print(f"\n  Output: {outdir}")


if __name__ == "__main__":
    main()
