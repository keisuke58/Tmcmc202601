#!/usr/bin/env python3
"""
Train-Early -> Predict-Late Validation (Validation C)

For each condition:
  1. Use scipy.minimize to find MAP on days 1,3,6,10 only
  2. Forward simulate with that MAP
  3. Compute prediction RMSE on held-out days 15, 21
  4. Compare with full-data MAP RMSE on the same late timepoints

Data is absolute volume phibar = phi * psi.
"""

import sys
import json
import time
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize

# --- Path setup ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tmcmc" / "program2602"))

from improved_5species_jit import BiofilmNewtonSolver5S

# --- Constants ---
SPECIES_NAMES = ["S. oralis", "A. naeslundii", "V. dispar", "F. nucleatum", "P. gingivalis"]
SPECIES_SHORT = ["So", "An", "Vd", "Fn", "Pg"]
DAYS = np.array([1, 3, 6, 10, 15, 21])
IDX_SPARSE = np.array([113, 339, 679, 1131, 1696, 2375])
ACTIVE_SPECIES = [0, 1, 2, 3, 4]

EARLY_IDX = [0, 1, 2, 3]  # days 1, 3, 6, 10
LATE_IDX = [4, 5]  # days 15, 21

SOLVER_KWARGS = dict(
    dt=0.0001,
    maxtimestep=2500,
    c_const=25.0,
    alpha_const=0.0,
    phi_init=0.02,
    Kp1=0.0001,
    K_hill=0.05,
    n_hill=4.0,
)

RUNS_DIR = PROJECT_ROOT / "data_5species" / "_runs"

CONDITIONS = {
    "CS": {
        "run": RUNS_DIR / "commensal_static_posterior",
        "label": "Commensal Static",
        "config": RUNS_DIR / "commensal_static_posterior" / "config.json",
    },
    "CH": {
        "run": RUNS_DIR / "commensal_hobic_posterior",
        "label": "Commensal HOBIC",
        "config": RUNS_DIR / "commensal_hobic_posterior" / "config.json",
    },
    "DS": {
        "run": RUNS_DIR / "dysbiotic_static_posterior",
        "label": "Dysbiotic Static",
        "config": RUNS_DIR / "dysbiotic_static_posterior" / "config.json",
    },
    "DH": {
        "run": RUNS_DIR / "Dysbiotic_HOBIC_20260226_040637",
        "label": "Dysbiotic HOBIC",
        "theta_run": RUNS_DIR / "dh_baseline",
        "config": RUNS_DIR / "Dysbiotic_HOBIC_20260226_040637" / "config.json",
    },
}

# Cached solver (created once, reused)
_SOLVER = None


def get_solver():
    global _SOLVER
    if _SOLVER is None:
        _SOLVER = BiofilmNewtonSolver5S(
            **SOLVER_KWARGS,
            active_species=ACTIVE_SPECIES,
            use_numba=False,
        )
    return _SOLVER


def load_theta_map(cond_key: str):
    cfg = CONDITIONS[cond_key]
    theta_dir = cfg.get("theta_run", cfg["run"])
    with open(theta_dir / "theta_MAP.json") as f:
        d = json.load(f)
    return np.array(d["theta_full"]), d["active_indices"]


def load_data(cond_key: str) -> np.ndarray:
    cfg = CONDITIONS[cond_key]
    return np.load(cfg["run"] / "data.npy")


def load_prior_bounds(cond_key: str, active_indices: list):
    bounds_file = PROJECT_ROOT / "data_5species" / "model_config" / "prior_bounds.json"
    with open(bounds_file) as f:
        bounds_cfg = json.load(f)
    strategy_map = {
        "CS": "Commensal_Static",
        "CH": "Commensal_HOBIC",
        "DS": "Dysbiotic_Static",
        "DH": "Dysbiotic_HOBIC",
    }
    strategy = bounds_cfg["strategies"][strategy_map[cond_key]]
    default = bounds_cfg["default_bounds"]
    lower, upper = [], []
    for idx in active_indices:
        b = strategy.get("bounds", {}).get(str(idx), default)
        lower.append(b[0])
        upper.append(b[1])
    return np.array(lower), np.array(upper)


def extract_phibar(x0: np.ndarray) -> np.ndarray:
    """Extract phibar at observation times from state array."""
    n_state = x0.shape[1]
    n_total = (n_state - 2) // 2
    psi_offset = n_total + 1
    phibar = np.zeros((len(IDX_SPARSE), 5))
    for i, sp in enumerate(ACTIVE_SPECIES):
        phibar[:, i] = x0[IDX_SPARSE, sp] * x0[IDX_SPARSE, psi_offset + sp]
    return phibar


def run_forward(theta_full: np.ndarray) -> np.ndarray:
    """Run ODE and return phibar at observation times, shape (6, 5)."""
    solver = get_solver()
    _, x0 = solver.run_deterministic(theta_full)
    return extract_phibar(x0)


def compute_metrics(pred, data):
    resid = pred - data
    rmse = float(np.sqrt(np.mean(resid**2)))
    mae = float(np.mean(np.abs(resid)))
    rmse_per = np.sqrt(np.mean(resid**2, axis=0))
    return {"rmse": rmse, "mae": mae, "rmse_per_species": rmse_per.tolist()}


def optimize_early(data_early, active_indices, theta_init_full, lb, ub, sigma_obs):
    """Find MAP on early data using Nelder-Mead with dimension-scaled budget."""
    theta_base = np.zeros(20)
    sigma2 = sigma_obs**2
    n_eval = [0]
    n_active = len(active_indices)

    def neg_logL(theta_sub):
        n_eval[0] += 1
        full = theta_base.copy()
        for i, idx in enumerate(active_indices):
            full[idx] = np.clip(theta_sub[i], lb[i], ub[i])
        try:
            pred = run_forward(full)
        except Exception:
            return 1e20
        pred_early = pred[EARLY_IDX]
        if not np.all(np.isfinite(pred_early)):
            return 1e20
        resid = data_early - pred_early
        return 0.5 * np.sum(resid**2 / sigma2)

    # Start from full MAP (warm start), clipped to bounds
    theta_sub_init = np.array([theta_init_full[idx] for idx in active_indices])
    theta_sub_init = np.clip(theta_sub_init, lb, ub)

    # Scale maxfev with dimensionality: 200*n for proper convergence
    maxfev = max(500, 200 * n_active)

    t0 = time.time()
    res = minimize(
        neg_logL,
        theta_sub_init,
        method="Nelder-Mead",
        options={"maxfev": maxfev, "xatol": 1e-4, "fatol": 1e-8, "adaptive": True},
    )
    elapsed = time.time() - t0

    # Clip to bounds
    theta_opt = np.clip(res.x, lb, ub)
    print(
        f"    Optimization: {n_eval[0]} fn evals (budget {maxfev}), {elapsed:.1f}s, "
        f"f={res.fun:.6f}, success={res.success}"
    )

    theta_full = theta_base.copy()
    for i, idx in enumerate(active_indices):
        theta_full[idx] = theta_opt[i]
    return theta_full, res


def main():
    out_dir = SCRIPT_DIR / "results_early_late"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    all_preds = {}

    for cond_key in CONDITIONS:
        cfg = CONDITIONS[cond_key]
        print(f"\n{'='*60}")
        print(f"  {cfg['label']} ({cond_key})")
        print(f"{'='*60}")

        data = load_data(cond_key)
        theta_full_map, active_indices = load_theta_map(cond_key)

        with open(cfg["config"]) as f:
            config = json.load(f)
        sigma_obs = config.get("sigma_obs", 0.1)

        data_early = data[EARLY_IDX]
        data_late = data[LATE_IDX]

        # Full MAP prediction
        pred_full = run_forward(theta_full_map)
        m_full_early = compute_metrics(pred_full[EARLY_IDX], data_early)
        m_full_late = compute_metrics(pred_full[LATE_IDX], data_late)
        m_full_all = compute_metrics(pred_full, data)

        print(
            f"  Full MAP RMSE: all={m_full_all['rmse']:.4f}, "
            f"early={m_full_early['rmse']:.4f}, late={m_full_late['rmse']:.4f}"
        )

        # Optimize on early data
        lb, ub = load_prior_bounds(cond_key, active_indices)
        theta_early_full, opt_res = optimize_early(
            data_early, active_indices, theta_full_map, lb, ub, sigma_obs
        )

        pred_early_trained = run_forward(theta_early_full)
        m_et_early = compute_metrics(pred_early_trained[EARLY_IDX], data_early)
        m_et_late = compute_metrics(pred_early_trained[LATE_IDX], data_late)
        m_et_all = compute_metrics(pred_early_trained, data)

        late_deg = (
            m_et_late["rmse"] / m_full_late["rmse"] if m_full_late["rmse"] > 1e-6 else float("inf")
        )

        print(
            f"  Early MAP RMSE: all={m_et_all['rmse']:.4f}, "
            f"early={m_et_early['rmse']:.4f}, late={m_et_late['rmse']:.4f}"
        )
        print(f"  Late prediction degradation: {late_deg:.2f}x vs full MAP")

        print("  Late RMSE per species (early-trained / full-MAP):")
        for i, sp in enumerate(SPECIES_SHORT):
            et_val = m_et_late["rmse_per_species"][i]
            full_val = m_full_late["rmse_per_species"][i]
            ratio = et_val / full_val if full_val > 1e-6 else float("inf")
            print(f"    {sp}: {et_val:.4f} / {full_val:.4f} = {ratio:.2f}x")

        results[cond_key] = {
            "label": cfg["label"],
            "full_map": {
                "rmse_all": m_full_all["rmse"],
                "rmse_early": m_full_early["rmse"],
                "rmse_late": m_full_late["rmse"],
                "rmse_late_per_species": m_full_late["rmse_per_species"],
            },
            "early_trained": {
                "rmse_all": m_et_all["rmse"],
                "rmse_early": m_et_early["rmse"],
                "rmse_late": m_et_late["rmse"],
                "rmse_late_per_species": m_et_late["rmse_per_species"],
                "theta_full": theta_early_full.tolist(),
                "n_fun_evals": opt_res.nfev,
            },
            "late_degradation": late_deg,
        }
        all_preds[cond_key] = {
            "data": data,
            "pred_full": pred_full,
            "pred_early": pred_early_trained,
        }

    # Save results
    with open(out_dir / "early_late_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ============================================================
    # Visualization
    # ============================================================
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, cond_key in zip(axes.flat, CONDITIONS):
        if cond_key not in all_preds:
            ax.set_visible(False)
            continue
        cfg = CONDITIONS[cond_key]
        p = all_preds[cond_key]
        r = results[cond_key]

        ax.axvspan(13, 22, alpha=0.1, color="red")
        ax.axvline(12, color="gray", linestyle=":", alpha=0.5, label="Train/Test split")

        for i, sp_name in enumerate(SPECIES_NAMES):
            ax.plot(DAYS, p["data"][:, i], "o", color=colors[i], markersize=7, zorder=3)
            ax.plot(DAYS, p["pred_full"][:, i], "-", color=colors[i], alpha=0.4, linewidth=1.5)
            ax.plot(
                DAYS, p["pred_early"][:, i], "--", color=colors[i], linewidth=2.5, label=sp_name
            )

        ax.set_title(
            f"{cfg['label']}\n"
            f"Late RMSE: {r['early_trained']['rmse_late']:.4f} "
            f"(full: {r['full_map']['rmse_late']:.4f}, "
            f"{r['late_degradation']:.2f}x)",
            fontsize=9,
            fontweight="bold",
        )
        ax.set_xlabel("Day")
        ax.set_ylabel(r"$\bar{\varphi}$")
        ax.set_xlim(0, 22)
        ax.legend(fontsize=7, loc="best")

    fig.suptitle(
        "Early -> Late Prediction Validation\n"
        "(train days 1-10, predict days 15-21; solid=full MAP, dashed=early-trained)",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(out_dir / "early_late_validation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {out_dir / 'early_late_validation.png'}")

    # Bar chart
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    cond_keys = [k for k in CONDITIONS if k in results]
    x = np.arange(len(cond_keys))
    w = 0.35
    full_late = [results[k]["full_map"]["rmse_late"] for k in cond_keys]
    early_late = [results[k]["early_trained"]["rmse_late"] for k in cond_keys]

    bars1 = ax2.bar(
        x - w / 2, full_late, w, label="Full MAP (late RMSE)", color="#4daf4a", alpha=0.8
    )
    bars2 = ax2.bar(
        x + w / 2, early_late, w, label="Early-trained (late RMSE)", color="#e41a1c", alpha=0.8
    )
    for bar, val in zip(bars1, full_late):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for bar, val in zip(bars2, early_late):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax2.set_ylabel("Late RMSE (days 15, 21)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([CONDITIONS[k]["label"] for k in cond_keys], fontsize=9)
    ax2.legend()
    ax2.set_title("Early->Late: Late-Period RMSE Comparison", fontweight="bold")
    plt.tight_layout()
    fig2.savefig(out_dir / "early_late_rmse_bar.png", dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"Bar chart saved: {out_dir / 'early_late_rmse_bar.png'}")

    # Summary
    print("\n========================================")
    print("SUMMARY: Early->Late Prediction")
    print("========================================")
    for k in cond_keys:
        r = results[k]
        deg = r["late_degradation"]
        tag = "OK" if deg < 2.0 else "!!"
        print(
            f"  [{tag}] {k} ({r['label']}): "
            f"late RMSE {r['early_trained']['rmse_late']:.4f} "
            f"({deg:.2f}x vs full-MAP {r['full_map']['rmse_late']:.4f})"
        )


if __name__ == "__main__":
    main()
