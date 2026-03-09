#!/usr/bin/env python3
"""
run_coupled_staggered.py — Klempt-style staggered growth-mechanics coupling
============================================================================

Staggered (operator-split) time loop:
  For each macro time step:
    (1) Hamilton reaction (5-species ODE per node)
    (2) Species diffusion (explicit Euler, Neumann BCs)
    (3) Nutrient PDE (CFL-stable sub-stepping)
    (4) Accumulate growth variable α(x,y)
    (5) Compute DI(x,y) → E(x,y)
    (6) Quasi-static FEM solve → σ(x,y), u(x,y)

This replaces the previous decoupled pipeline:
  OLD:  ODE(all steps) → final DI → final E → FEM(once)
  NEW:  each step → ODE + FEM together (Klempt 2024 style)

Usage:
    python run_coupled_staggered.py
    python run_coupled_staggered.py --condition dh_baseline
    python run_coupled_staggered.py --condition commensal_static --nx 25 --ny 25
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
    _make_reaction_step,
    _make_nutrient_step_stable,
    _make_newton_step_vmap,
    diffusion_step_species_2d,
    make_initial_state_2d,
    theta_to_matrices,
)
from solve_stress_2d import solve_2d_fem
from material_models import compute_di, compute_E_phi_pg, E_MAX_PA, E_MIN_PA

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


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
):
    """
    Staggered growth-mechanics coupling (Klempt 2024 style).

    At each macro time step:
      (1-3) Hamilton reaction + diffusion + nutrient PDE
      (4)   α(x,y) += k_α · φ_total · c/(k+c) · dt
      (5)   DI(x,y) → E(x,y)
      (6)   FEM solve with current E and ε_growth = α/3

    Parameters
    ----------
    theta : (20,) array
    cfg : Config2D
    nu : float — Poisson's ratio
    k_alpha : float — growth rate coefficient
    e_model : str — "phi_pg", "di", or "virulence"
    fem_every : int or None — solve FEM every N macro steps (None = save_every)

    Returns
    -------
    dict with time-evolution snapshots of all fields
    """
    A, b_diag = theta_to_matrices(jnp.asarray(theta, dtype=jnp.float64))
    active_mask = jnp.ones(5, dtype=jnp.int64)

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

    # Use Python loop (not lax.scan) to avoid LLVM memory accumulation
    _newton_vmap = _make_newton_step_vmap(cfg.newton_iters)

    def _reaction_step_pyloop(G, params):
        for _ in range(cfg.n_react_sub):
            G = _newton_vmap(G, params)
        return G

    _reaction_step = _reaction_step_pyloop
    _nutrient_stable = _make_nutrient_step_stable(30)

    D_eff = jnp.array(cfg.D_eff)
    D_c = cfg.D_c
    k_M = cfg.k_monod
    g_cons = jnp.array(cfg.g_consumption)
    c_bc = cfg.c_boundary
    dt_macro = cfg.dt_macro
    Nx, Ny = cfg.Nx, cfg.Ny
    Lx, Ly = cfg.Lx, cfg.Ly

    fem_every = fem_every or cfg.save_every

    G, c = make_initial_state_2d(cfg)

    # Growth variable α(x,y) — accumulates over time
    alpha_field = np.zeros((Nx, Ny))

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
    }

    # Save initial state
    phi_2d_init = np.asarray(G.reshape(Nx, Ny, 12)[:, :, :5])
    _save_snapshot(snaps, 0.0, phi_2d_init, np.asarray(c), alpha_field, Nx, Ny, Lx, Ly, nu, e_model)

    print(f"\n{'='*70}")
    print(f"  STAGGERED COUPLED  |  Nx={Nx} Ny={Ny}  |  {cfg.n_macro} macro steps")
    print(f"  dt_h={cfg.dt_h:.1e}  n_sub={cfg.n_react_sub}  k_alpha={k_alpha}")
    print(f"  FEM every {fem_every} steps  |  E model: {e_model}")
    print(f"{'='*70}\n")

    t0_wall = time.perf_counter()

    for step in range(1, cfg.n_macro + 1):
        t = step * dt_macro

        # --- (1) Reaction: Hamilton Newton over all nodes ---
        G = _reaction_step(G, params)

        # --- (2) Species diffusion ---
        phi_2d = G.reshape(Nx, Ny, 12)[:, :, :5]
        phi_2d = diffusion_step_species_2d(phi_2d, D_eff, dt_macro, cfg.dx, cfg.dy)

        # Write back to G
        G_2d = G.reshape(Nx, Ny, 12)
        G_2d = G_2d.at[:, :, :5].set(phi_2d)
        G_2d = G_2d.at[:, :, 5].set(1.0 - jnp.sum(phi_2d, axis=-1))
        G = G_2d.reshape(Nx * Ny, 12)

        # --- (3) Nutrient PDE ---
        c = _nutrient_stable(c, phi_2d, D_c, k_M, g_cons, c_bc, cfg.dx, cfg.dy, dt_macro)

        # --- (4) Accumulate growth α ---
        phi_2d_np = np.asarray(phi_2d)
        c_np = np.asarray(c)
        phi_total = phi_2d_np.sum(axis=-1)
        monod = c_np / (k_M + c_np)
        alpha_field += k_alpha * phi_total * monod * float(dt_macro)

        # --- (5-6) FEM solve at snapshot intervals ---
        if step % fem_every == 0 or step == cfg.n_macro:
            _save_snapshot(
                snaps, float(t), phi_2d_np, c_np, alpha_field, Nx, Ny, Lx, Ly, nu, e_model
            )

            i_snap = len(snaps["t"]) - 1
            pct = 100 * step / cfg.n_macro
            svm_max = snaps["sigma_vm_max"][-1]
            u_max = snaps["u_max"][-1]
            alpha_max = alpha_field.max()
            print(
                f"  [{pct:5.1f}%] t={t:.5f}  "
                f"α_max={alpha_max:.4f}  "
                f"σ_vm_max={svm_max:.2f} Pa  "
                f"|u|_max={u_max:.2e}"
            )

    elapsed = time.perf_counter() - t0_wall
    print(f"\n  Total wall time: {elapsed:.1f}s")
    print(f"  Snapshots: {len(snaps['t'])}")

    # Convert lists to arrays
    for key in ["t", "u_max", "sigma_vm_max", "sigma_vm_mean"]:
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


def _save_snapshot(snaps, t, phi_2d_np, c_np, alpha_field, Nx, Ny, Lx, Ly, nu, e_model):
    """Compute derived fields + FEM solve and store snapshot."""
    # phi_2d_np: (Nx, Ny, 5)
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
    if alpha_field.max() > 1e-12:
        fem = solve_2d_fem(E_field, nu, eps_growth, Nx, Ny, Lx, Ly, bc_type="bottom_fixed")
        sigma_vm = fem["sigma_vm"]
        u_mag = np.sqrt(fem["u"][:, 0] ** 2 + fem["u"][:, 1] ** 2)
        u_max = float(u_mag.max())
        svm_max = float(sigma_vm.max())
        svm_mean = float(sigma_vm.mean())
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


# ============================================================================
# Visualization: time-evolution panel
# ============================================================================


def plot_time_evolution(snaps, outpath, condition="demo"):
    """6-row time-evolution figure showing coupled growth-mechanics."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    t_arr = snaps["t"]
    n_snap = len(t_arr)
    Nx, Ny = snaps["Nx"], snaps["Ny"]
    n_ex, n_ey = Nx - 1, Ny - 1

    # Select up to 5 snapshots (evenly spaced + final)
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

    # 6 rows: φ_total, c, DI, E, ε_growth, σ_vm
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
        phi_snap = snaps["phi"][snap_i]  # (5, Nx, Ny)
        c_snap = snaps["c"][snap_i]
        DI_snap = snaps["DI"][snap_i]
        E_snap = snaps["E"][snap_i]
        alpha_snap = snaps["alpha"][snap_i]
        svm_snap = snaps["sigma_vm"][snap_i]

        phi_total = phi_snap.sum(axis=0)

        fields = [phi_total, c_snap, DI_snap, E_snap, alpha_snap]
        # σ_vm is on element grid
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

        # σ_vm row
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
    ax.set_xlabel("t (Hamilton time)")
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
        "--n-macro",
        str(args.n_macro),
        "--dt-h",
        str(args.dt_h),
        "--n-react-sub",
        str(args.n_react_sub),
        "--save-every",
        str(args.save_every),
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
        "--outdir",
        str(args.outdir),
        "--save-npz",  # save results for later comparison
    ]
    print(f"\n  >>> subprocess: {cond_dir}")
    proc = sp.run(cmd, capture_output=False, text=True)
    return proc.returncode


def run_4_conditions(args):
    """Run staggered coupling for all 4 conditions via subprocess isolation."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _TMCMC = _HERE.parent.parent
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    conditions = {
        "commensal_static": "CS",
        "commensal_hobic": "CH",
        "dh_baseline": "DH",
        "dysbiotic_static": "DS",
    }

    # Run each condition in a separate subprocess (JAX LLVM memory isolation)
    for cond_dir, label in conditions.items():
        rc = _run_single_condition_subprocess(cond_dir, args)
        if rc != 0:
            print(f"  WARNING: {cond_dir} failed (rc={rc})")

    # Load saved npz files
    all_snaps = {}
    for cond_dir, label in conditions.items():
        npz_path = outdir / f"coupled_snaps_{cond_dir}.npz"
        if not npz_path.exists():
            print(f"  SKIP {label}: {npz_path} not found")
            continue
        data = np.load(npz_path, allow_pickle=True)
        snaps = {k: data[k] for k in data.files}
        # Restore scalar metadata
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

    # --- Comparison figure: σ_vm_max(t) and u_max(t) for all conditions ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = {"CS": "#2196F3", "CH": "#4CAF50", "DH": "#FF9800", "DS": "#F44336"}

    for label, snaps in all_snaps.items():
        t = snaps["t"]
        col = colors.get(label, "gray")

        axes[0].plot(t, snaps["sigma_vm_max"], "-o", color=col, ms=3, lw=1.5, label=label)
        axes[1].plot(t, snaps["u_max"], "-o", color=col, ms=3, lw=1.5, label=label)

        alpha_max = np.array([snaps["alpha"][i].max() for i in range(len(t))])
        axes[2].plot(t, alpha_max, "-o", color=col, ms=3, lw=1.5, label=label)

    axes[0].set_xlabel("t (Hamilton time)")
    axes[0].set_ylabel("σ_vm,max [Pa]")
    axes[0].set_title("Max von Mises stress")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("t")
    axes[1].set_ylabel("|u|_max")
    axes[1].set_title("Max displacement")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel("t")
    axes[2].set_ylabel("α_max")
    axes[2].set_title("Max growth variable")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

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

    # --- Final-state comparison: 4 columns (CS/CH/DH/DS) × 3 rows (DI, σ_vm, u) ---
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
        ax.set_title(f"E(x,y)", fontsize=11)
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
    ap = argparse.ArgumentParser(description="Staggered coupled growth-mechanics solver")
    ap.add_argument(
        "--condition", default=None, help="Single condition or 'all' for 4-condition comparison"
    )
    ap.add_argument("--nx", type=int, default=20)
    ap.add_argument("--ny", type=int, default=20)
    ap.add_argument("--n-macro", type=int, default=60)
    ap.add_argument("--dt-h", type=float, default=1e-5)
    ap.add_argument("--n-react-sub", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--k-hill", type=float, default=0.05)
    ap.add_argument("--n-hill", type=float, default=4.0)
    ap.add_argument("--nu", type=float, default=0.30)
    ap.add_argument("--k-alpha", type=float, default=0.05)
    ap.add_argument("--e-model", choices=["phi_pg", "virulence", "di"], default="phi_pg")
    ap.add_argument("--outdir", default=None)
    ap.add_argument(
        "--save-npz", action="store_true", help="Save results as .npz for later comparison"
    )
    args = ap.parse_args()

    _TMCMC = _HERE.parent.parent
    outdir = args.outdir or str(_TMCMC / "FEM" / "figures" / "coupled_staggered")
    args.outdir = outdir
    Path(outdir).mkdir(parents=True, exist_ok=True)

    if args.condition is None or args.condition == "all":
        run_4_conditions(args)
    else:
        # Single condition
        _RUNS = _TMCMC / "data_5species" / "_runs"
        theta_path = _RUNS / args.condition / "theta_MAP.json"
        if theta_path.exists():
            with open(theta_path) as f:
                d = json.load(f)
            theta = np.array(d.get("theta_full", d.get("theta_sub")), dtype=np.float64)
        else:
            from core_hamilton_2d_nutrient import THETA_DEMO

            theta = THETA_DEMO
            print(f"  WARNING: {theta_path} not found, using demo theta")

        cfg = Config2D(
            Nx=args.nx,
            Ny=args.ny,
            n_macro=args.n_macro,
            dt_h=args.dt_h,
            n_react_sub=args.n_react_sub,
            save_every=args.save_every,
            K_hill=args.k_hill,
            n_hill=args.n_hill,
        )

        snaps = run_staggered_coupled(
            theta,
            cfg,
            nu=args.nu,
            k_alpha=args.k_alpha,
            e_model=args.e_model,
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

        # Save npz if requested
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
                Nx=snaps["Nx"],
                Ny=snaps["Ny"],
                Lx=snaps["Lx"],
                Ly=snaps["Ly"],
                e_model=snaps["e_model"],
                elapsed_s=snaps["elapsed_s"],
            )
            print(f"  Saved: {npz_path}")

        # Print summary
        print(f"\n  Final DI: [{snaps['DI'][-1].min():.3f}, {snaps['DI'][-1].max():.3f}]")
        print(f"  Final E:  [{snaps['E'][-1].min():.1f}, {snaps['E'][-1].max():.1f}] Pa")
        print(f"  Final α:  [{snaps['alpha'][-1].min():.5f}, {snaps['alpha'][-1].max():.5f}]")
        print(f"  Final σ_vm_max: {snaps['sigma_vm_max'][-1]:.2f} Pa")
        print(f"  Final |u|_max:  {snaps['u_max'][-1]:.2e}")

    print(f"\n  Output: {outdir}")


if __name__ == "__main__":
    main()
