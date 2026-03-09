#!/usr/bin/env python3
"""
compare_glv_hamilton.py
========================
Reviewer defense: Hamilton ODE vs generalized Lotka-Volterra (gLV) comparison.

Shows that gLV does NOT conserve total volume fraction (Σφ_i ≠ const),
while Hamilton formulation does by construction (Lagrange multiplier γ).

Also compares fit quality when both are calibrated to the same data.

Usage:
    python compare_glv_hamilton.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tmcmc", "program2602"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "data_5species", "main"))

import numpy as np
import json
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from improved_5species_jit import BiofilmNewtonSolver5S

# ── Load data ──
RUN_DIRS = {
    "CS": "data_5species/_runs/commensal_static_posterior",
    "CH": "data_5species/_runs/commensal_hobic_posterior",
    "DH": "data_5species/_runs/dh_baseline",
    "DS": "data_5species/_runs/dysbiotic_static_posterior",
}

DAYS = np.array([1, 3, 6, 10, 15, 21])
SPECIES = ["S. oralis", "A. naeslundii", "Veillonella", "F. nucleatum", "P. gingivalis"]
SP_COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]


# ═══════════════════════════════════════════════════════
# gLV model: dφ_i/dt = φ_i (r_i + Σ_j A_ij φ_j)
# ═══════════════════════════════════════════════════════


def glv_rhs(t, phi, r, A):
    """Generalized Lotka-Volterra RHS."""
    phi = np.clip(phi, 0, None)
    return phi * (r + A @ phi)


def simulate_glv(r, A, phi0, t_span, t_eval):
    """Simulate gLV. Returns (n_time, 5) array."""
    sol = solve_ivp(
        glv_rhs,
        t_span,
        phi0,
        t_eval=t_eval,
        args=(r, A),
        method="Radau",
        rtol=1e-6,
        atol=1e-8,
    )
    if sol.success and sol.y.shape[1] == len(t_eval):
        return sol.y.T  # (n_time, 5)
    return None


def glv_params_from_vector(x, n_sp=5):
    """Unpack optimization vector into (r, A).
    x = [r1..r5, A11, A12, ..., A55] (5 + 25 = 30 params)
    """
    r = x[:n_sp]
    A = x[n_sp:].reshape(n_sp, n_sp)
    return r, A


def glv_loss(x, phi0, data, t_eval, t_span):
    """SSE for gLV fit."""
    r, A = glv_params_from_vector(x)
    phi_pred = simulate_glv(r, A, phi0, t_span, t_eval)
    if phi_pred is None:
        return 1e10
    if phi_pred.shape[0] != data.shape[0]:
        return 1e10
    return np.sum((phi_pred - data) ** 2)


def fit_glv(data, t_eval, n_restarts=5):
    """Fit gLV to data using multi-start optimization."""
    n_sp = 5
    phi0 = data[0].copy()
    phi0 = np.maximum(phi0, 1e-6)
    t_span = (t_eval[0], t_eval[-1])

    best_loss = np.inf
    best_x = None

    for trial in range(n_restarts):
        np.random.seed(trial * 42)
        r0 = np.random.uniform(-1, 1, n_sp)
        A0 = np.random.uniform(-2, 2, (n_sp, n_sp))
        x0 = np.concatenate([r0, A0.flatten()])

        try:
            # First rough pass with Nelder-Mead
            res = minimize(
                glv_loss,
                x0,
                args=(phi0, data, t_eval, t_span),
                method="Nelder-Mead",
                options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-8},
            )
            # Refine with Powell
            if res.fun < 1e8:
                res = minimize(
                    glv_loss,
                    res.x,
                    args=(phi0, data, t_eval, t_span),
                    method="Powell",
                    options={"maxiter": 2000, "xtol": 1e-8, "ftol": 1e-10},
                )
            if res.fun < best_loss:
                best_loss = res.fun
                best_x = res.x
                print(f"    Trial {trial}: loss={res.fun:.6f}")
        except Exception:
            continue

    if best_x is None:
        return None, None, np.inf

    r, A = glv_params_from_vector(best_x)
    return r, A, best_loss


# ═══════════════════════════════════════════════════════
# Hamilton model (from TMCMC MAP)
# ═══════════════════════════════════════════════════════


def simulate_hamilton_map(cond):
    """Simulate Hamilton ODE at MAP theta."""
    run_dir = RUN_DIRS[cond]

    theta_info = json.load(open(f"{run_dir}/theta_MAP.json"))
    theta = np.array(theta_info["theta_full"])

    cfg_path = f"{run_dir}/config.json"
    try:
        cfg = json.load(open(cfg_path))
        alpha = cfg.get("alpha_const", 0.0)
        n_hill = cfg.get("n_hill", 4.0)
    except FileNotFoundError:
        alpha = 0.0
        n_hill = 2.0 if cond == "DH" else 4.0

    solver = BiofilmNewtonSolver5S(
        dt=1e-4,
        maxtimestep=2500,
        c_const=25.0,
        alpha_const=alpha,
        K_hill=0.05,
        n_hill=n_hill,
        phi_init=0.02,
    )
    t_arr, g_arr = solver.run_deterministic(theta)
    # φ̄ = φ × ψ
    phi = g_arr[:, :5]
    psi = g_arr[:, 6:11]
    phibar = phi * psi
    return t_arr, phibar, g_arr


def main():
    np.random.seed(42)
    results = {}

    for cond in ["CS", "CH", "DS", "DH"]:
        print(f"\n{'='*60}")
        print(f"Condition: {cond}")

        run_dir = RUN_DIRS[cond]
        data_path = f"{run_dir}/data.npy"
        idx_path = f"{run_dir}/idx_sparse.npy"

        if not os.path.exists(data_path):
            print("  No data.npy, skipping")
            continue

        data = np.load(data_path)
        idx_sparse = np.load(idx_path).astype(int)
        print(f"  Data shape: {data.shape}")

        # ── Hamilton MAP ──
        t_arr, phibar_ham, g_arr = simulate_hamilton_map(cond)
        ham_pred = phibar_ham[idx_sparse]
        ham_rmse = np.sqrt(np.mean((ham_pred[1:] - data[1:]) ** 2))
        ham_sum = np.sum(g_arr[:, :5], axis=1) + g_arr[:, 5]  # Σφ_i + φ0
        ham_sum_dev = np.max(np.abs(ham_sum - 1.0))
        print(f"  Hamilton MAP: RMSE={ham_rmse:.4f}, max|Σφ-1|={ham_sum_dev:.2e}")

        # ── gLV fit ──
        t_eval = DAYS.astype(float)
        print("  Fitting gLV (10 restarts)...")
        r, A, glv_loss_val = fit_glv(data, t_eval, n_restarts=20)

        if r is not None:
            phi0_glv = np.maximum(data[0].copy(), 1e-6)
            t_span = (t_eval[0], t_eval[-1])
            glv_pred = simulate_glv(r, A, phi0_glv, t_span, t_eval)
            if glv_pred is not None:
                glv_rmse = np.sqrt(np.mean((glv_pred[1:] - data[1:]) ** 2))
                glv_sum = np.sum(glv_pred, axis=1)
                glv_sum_dev = np.max(np.abs(glv_sum - glv_sum[0]))
                n_params_glv = 5 + 25  # r(5) + A(5x5)
                print(
                    f"  gLV fit: RMSE={glv_rmse:.4f}, max|ΔΣφ|={glv_sum_dev:.4f}, params={n_params_glv}"
                )
            else:
                glv_rmse = np.inf
                glv_sum = None
                glv_sum_dev = np.inf
                print("  gLV: simulation failed")
        else:
            glv_rmse = np.inf
            glv_pred = None
            glv_sum = None
            glv_sum_dev = np.inf
            print("  gLV: optimization failed")

        results[cond] = {
            "data": data,
            "idx_sparse": idx_sparse,
            "ham_pred": ham_pred,
            "ham_rmse": ham_rmse,
            "ham_sum_dev": ham_sum_dev,
            "ham_phibar": phibar_ham,
            "ham_g_arr": g_arr,
            "glv_pred": glv_pred,
            "glv_rmse": glv_rmse,
            "glv_sum_dev": glv_sum_dev,
            "glv_r": r,
            "glv_A": A,
        }

    plot_comparison(results)
    print_summary(results)


def plot_comparison(results):
    """2 rows × 4 cols: top = fit comparison, bottom = volume conservation."""
    conditions = [c for c in ["CS", "CH", "DS", "DH"] if c in results]
    n_cond = len(conditions)

    fig, axes = plt.subplots(2, n_cond, figsize=(5 * n_cond, 8))
    fig.suptitle("Hamilton ODE vs generalized Lotka-Volterra (gLV)", fontsize=14, y=0.98)

    for col, cond in enumerate(conditions):
        r = results[cond]
        data = r["data"]

        # ── Top: fit comparison ──
        ax = axes[0, col]
        for s in range(5):
            ax.plot(DAYS, data[:, s], "o", color=SP_COLORS[s], ms=7, zorder=10)
            ax.plot(
                DAYS,
                r["ham_pred"][:, s],
                "-",
                color=SP_COLORS[s],
                lw=2,
                label=SPECIES[s] if col == 0 else None,
            )
            if r["glv_pred"] is not None:
                ax.plot(DAYS, r["glv_pred"][:, s], "--", color=SP_COLORS[s], lw=1.5, alpha=0.7)

        ax.set_title(f"{cond}\nHam RMSE={r['ham_rmse']:.3f}, gLV RMSE={r['glv_rmse']:.3f}")
        ax.set_xlabel("Day")
        if col == 0:
            ax.set_ylabel(r"$\bar{\varphi}_i$")
            ax.legend(fontsize=6, loc="upper left")
        # Add legend for line styles
        if col == n_cond - 1:
            from matplotlib.lines import Line2D

            custom = [
                Line2D([0], [0], color="gray", lw=2, ls="-"),
                Line2D([0], [0], color="gray", lw=1.5, ls="--"),
                Line2D([0], [0], color="gray", marker="o", ls="none", ms=6),
            ]
            ax.legend(custom, ["Hamilton", "gLV", "Data"], fontsize=8, loc="upper right")

        # ── Bottom: volume conservation ──
        ax2 = axes[1, col]
        # Hamilton: Σφ + φ0 should = 1
        g_arr = r["ham_g_arr"]
        t_ham = np.arange(g_arr.shape[0]) * 1e-4
        ham_sum = np.sum(g_arr[:, :5], axis=1) + g_arr[:, 5]
        ax2.plot(
            t_ham,
            ham_sum,
            "-",
            color="#1565C0",
            lw=2,
            label="Hamilton $\\Sigma\\varphi_i + \\varphi_0$",
        )

        # gLV: Σφ
        if r["glv_pred"] is not None:
            glv_sum = np.sum(r["glv_pred"], axis=1)
            ax2.plot(
                DAYS, glv_sum, "s--", color="#E65100", lw=2, ms=7, label="gLV $\\Sigma\\varphi_i$"
            )

        ax2.axhline(1.0, color="gray", ls=":", alpha=0.5)
        ax2.set_xlabel("Time")
        ax2.set_title(
            f"Volume conservation\nHam dev={r['ham_sum_dev']:.1e}, gLV dev={r['glv_sum_dev']:.3f}"
        )
        if col == 0:
            ax2.set_ylabel("Total volume fraction")
            ax2.legend(fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(os.path.dirname(__file__), "paper_final", "glv_hamilton_comparison.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nSaved: {out}")
    plt.close()


def print_summary(results):
    print("\n" + "=" * 70)
    print("Hamilton vs gLV Summary")
    print("=" * 70)
    print(
        f"{'Cond':4s} {'Ham RMSE':>10s} {'gLV RMSE':>10s} {'Ham |ΔΣ|':>12s} {'gLV |ΔΣ|':>12s} {'Ham params':>10s} {'gLV params':>10s}"
    )
    for cond in ["CS", "CH", "DS", "DH"]:
        if cond not in results:
            continue
        r = results[cond]
        print(
            f"{cond:4s} {r['ham_rmse']:10.4f} {r['glv_rmse']:10.4f} {r['ham_sum_dev']:12.2e} {r['glv_sum_dev']:12.4f} {'9-20':>10s} {'30':>10s}"
        )

    print("\nKey points:")
    print("  1. Hamilton: Σφ_i + φ_0 = 1 by construction (Lagrange multiplier γ)")
    print("  2. gLV: no volume conservation → Σφ_i drifts freely")
    print("  3. Hamilton uses 9-20 params (with biological constraints), gLV needs 30")
    print("  4. Hamilton ensures non-negative φ via Cahn-Hilliard potential")

    print("\n" + "=" * 70)
    print("LaTeX Table")
    print("=" * 70)
    print(r"\begin{tabular}{@{}lcccc@{}}")
    print(r"\toprule")
    print(r" & \multicolumn{2}{c}{RMSE} & \multicolumn{2}{c}{$|\Delta\Sigma\varphi|$} \\")
    print(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}")
    print(r" & Hamilton & gLV & Hamilton & gLV \\")
    print(r"\midrule")
    for cond in ["CS", "CH", "DS", "DH"]:
        if cond not in results:
            continue
        r = results[cond]
        print(
            f"{cond} & {r['ham_rmse']:.3f} & {r['glv_rmse']:.3f} & {r['ham_sum_dev']:.1e} & {r['glv_sum_dev']:.3f} \\\\"
        )
    print(r"\midrule")
    print(r"Parameters & 9--20 & 30 & \multicolumn{2}{c}{constraint: $\gamma$ vs none} \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == "__main__":
    main()
