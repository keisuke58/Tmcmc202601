#!/usr/bin/env python3
"""
Cross-Condition Prediction Validation (Validation B)

Uses MAP θ from one condition to predict another condition's data.
Pairs:
  - CS MAP → predict CH  (same commensal, different cultivation)
  - CH MAP → predict CS
  - DS MAP → predict DH  (same dysbiotic, different cultivation)
  - DH MAP → predict DS

Data is absolute volume phibar = phi * psi (NOT normalized fractions).
"""

import sys
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

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

# Condition label → run directory (with data.npy and theta_MAP.json)
CONDITIONS = {
    "CS": {
        "run": RUNS_DIR / "commensal_static_posterior",
        "label": "Commensal Static",
    },
    "CH": {
        "run": RUNS_DIR / "commensal_hobic_posterior",
        "label": "Commensal HOBIC",
    },
    "DS": {
        "run": RUNS_DIR / "dysbiotic_static_posterior",
        "label": "Dysbiotic Static",
    },
    "DH": {
        "run": RUNS_DIR / "Dysbiotic_HOBIC_20260226_040637",
        "label": "Dysbiotic HOBIC",
        "theta_run": RUNS_DIR / "dh_baseline",  # MAP from dh_baseline
    },
}

# Cross-prediction pairs: (source_MAP, target_data)
PAIRS = [
    ("CS", "CH"),
    ("CH", "CS"),
    ("DS", "DH"),
    ("DH", "DS"),
]


def load_theta_map(cond_key: str) -> np.ndarray:
    cfg = CONDITIONS[cond_key]
    theta_dir = cfg.get("theta_run", cfg["run"])
    with open(theta_dir / "theta_MAP.json") as f:
        d = json.load(f)
    return np.array(d["theta_full"])


def load_data(cond_key: str) -> np.ndarray:
    """Load absolute volume data from data.npy, shape (6, 5)."""
    cfg = CONDITIONS[cond_key]
    return np.load(cfg["run"] / "data.npy")


def run_forward(theta_full: np.ndarray) -> np.ndarray:
    """Run ODE forward and return raw phibar at observation times, shape (6, 5)."""
    solver = BiofilmNewtonSolver5S(
        **SOLVER_KWARGS,
        active_species=ACTIVE_SPECIES,
        use_numba=False,
    )
    _, x0 = solver.run_deterministic(theta_full)
    # phibar = phi * psi (absolute volume, no normalization)
    n_state = x0.shape[1]
    n_total = (n_state - 2) // 2
    psi_offset = n_total + 1
    phibar = np.zeros((len(IDX_SPARSE), 5))
    for i, sp in enumerate(ACTIVE_SPECIES):
        phibar[:, i] = x0[IDX_SPARSE, sp] * x0[IDX_SPARSE, psi_offset + sp]
    return phibar


def compute_metrics(pred: np.ndarray, data: np.ndarray):
    resid = pred - data
    rmse_total = float(np.sqrt(np.mean(resid**2)))
    mae_total = float(np.mean(np.abs(resid)))
    rmse_per = np.sqrt(np.mean(resid**2, axis=0))
    return {
        "rmse_total": rmse_total,
        "mae_total": mae_total,
        "rmse_per_species": rmse_per.tolist(),
        "max_abs": float(np.max(np.abs(resid))),
    }


def main():
    out_dir = SCRIPT_DIR / "results_cross_condition"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all MAP thetas and data
    thetas = {}
    datasets = {}
    for key in CONDITIONS:
        thetas[key] = load_theta_map(key)
        datasets[key] = load_data(key)
        print(
            f"Loaded {key}: theta shape={thetas[key].shape}, "
            f"data shape={datasets[key].shape}, "
            f"data range=[{datasets[key].min():.4f}, {datasets[key].max():.4f}]"
        )

    # Self-fit RMSE (sanity check — should match fit_metrics.json)
    self_rmses = {}
    print("\n--- Self-fit RMSE ---")
    for key in CONDITIONS:
        pred = run_forward(thetas[key])
        m = compute_metrics(pred, datasets[key])
        self_rmses[key] = m["rmse_total"]
        print(f"  {key}: RMSE = {m['rmse_total']:.4f}  MAE = {m['mae_total']:.4f}")

    # Cross-prediction
    results = {}
    print("\n--- Cross-Condition Prediction ---")
    for src, tgt in PAIRS:
        pred = run_forward(thetas[src])
        m = compute_metrics(pred, datasets[tgt])
        pair_key = f"{src}->{tgt}"
        deg = m["rmse_total"] / self_rmses[tgt] if self_rmses[tgt] > 0 else float("inf")
        results[pair_key] = {
            "source": src,
            "target": tgt,
            "source_label": CONDITIONS[src]["label"],
            "target_label": CONDITIONS[tgt]["label"],
            "metrics": m,
            "self_rmse_src": self_rmses[src],
            "self_rmse_tgt": self_rmses[tgt],
            "degradation": deg,
        }
        print(
            f"  {src}->{tgt}: RMSE = {m['rmse_total']:.4f}  "
            f"(vs self-fit {self_rmses[tgt]:.4f}, {deg:.2f}x degradation)"
        )
        # Per-species breakdown
        for i, sp in enumerate(SPECIES_SHORT):
            print(f"    {sp}: {m['rmse_per_species'][i]:.4f}")

    # Save results
    with open(out_dir / "cross_condition_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ============================================================
    # Visualization
    # ============================================================
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]

    # --- Fig 1: 4-panel time series ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (src, tgt) in zip(axes.flat, PAIRS):
        data_tgt = datasets[tgt]
        pred_self = run_forward(thetas[tgt])
        pred_cross = run_forward(thetas[src])
        pair_key = f"{src}->{tgt}"
        r = results[pair_key]

        for i, sp_name in enumerate(SPECIES_NAMES):
            ax.plot(DAYS, data_tgt[:, i], "o", color=colors[i], markersize=7, zorder=3)
            ax.plot(DAYS, pred_self[:, i], "-", color=colors[i], alpha=0.4, linewidth=1.5)
            ax.plot(DAYS, pred_cross[:, i], "--", color=colors[i], linewidth=2.5, label=sp_name)

        ax.set_title(
            f"{CONDITIONS[src]['label']} MAP -> {CONDITIONS[tgt]['label']} data\n"
            f"RMSE={r['metrics']['rmse_total']:.4f} ({r['degradation']:.2f}x)",
            fontsize=10,
            fontweight="bold",
        )
        ax.set_xlabel("Day")
        ax.set_ylabel(r"$\bar{\varphi}$ (absolute volume)")
        ax.set_xlim(0, 22)
        ax.legend(fontsize=7, loc="best")
        ax.text(
            0.02,
            0.95,
            f"Self-fit RMSE: {r['self_rmse_tgt']:.4f}",
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.5),
        )

    fig.suptitle(
        "Cross-Condition Prediction Validation\n"
        "(solid thin = self-fit, dashed thick = cross-prediction, dots = data)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(out_dir / "cross_condition_validation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {out_dir / 'cross_condition_validation.png'}")

    # --- Fig 2: Bar chart ---
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    pair_labels = [f"{s}->{t}" for s, t in PAIRS]
    self_vals = [results[f"{s}->{t}"]["self_rmse_tgt"] for s, t in PAIRS]
    cross_vals = [results[f"{s}->{t}"]["metrics"]["rmse_total"] for s, t in PAIRS]

    x = np.arange(len(pair_labels))
    w = 0.35
    bars1 = ax2.bar(x - w / 2, self_vals, w, label="Self-fit RMSE", color="#4daf4a", alpha=0.8)
    bars2 = ax2.bar(
        x + w / 2, cross_vals, w, label="Cross-prediction RMSE", color="#e41a1c", alpha=0.8
    )

    for bar, val in zip(bars1, self_vals):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for bar, val in zip(bars2, cross_vals):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax2.set_ylabel("RMSE")
    ax2.set_xticks(x)
    ax2.set_xticklabels(pair_labels, fontsize=11)
    ax2.legend()
    ax2.set_title("Cross-Condition Prediction: RMSE Comparison", fontweight="bold")
    plt.tight_layout()
    fig2.savefig(out_dir / "cross_condition_rmse_bar.png", dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"Bar chart saved: {out_dir / 'cross_condition_rmse_bar.png'}")

    # --- Fig 3: Per-species heatmap ---
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    rmse_matrix = np.zeros((len(PAIRS), 5))
    for i, (s, t) in enumerate(PAIRS):
        rmse_matrix[i] = results[f"{s}->{t}"]["metrics"]["rmse_per_species"]

    im = ax3.imshow(rmse_matrix, cmap="YlOrRd", aspect="auto")
    ax3.set_xticks(range(5))
    ax3.set_xticklabels(SPECIES_NAMES, rotation=30, ha="right")
    ax3.set_yticks(range(len(PAIRS)))
    ax3.set_yticklabels([f"{s}->{t}" for s, t in PAIRS])
    for i in range(len(PAIRS)):
        for j in range(5):
            ax3.text(
                j,
                i,
                f"{rmse_matrix[i, j]:.4f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if rmse_matrix[i, j] > 0.15 else "black",
            )
    plt.colorbar(im, ax=ax3, label="RMSE")
    ax3.set_title("Per-Species Cross-Prediction RMSE", fontweight="bold")
    plt.tight_layout()
    fig3.savefig(out_dir / "cross_condition_species_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig3)
    print(f"Heatmap saved: {out_dir / 'cross_condition_species_heatmap.png'}")

    # --- Summary ---
    print("\n========================================")
    print("SUMMARY")
    print("========================================")
    for src, tgt in PAIRS:
        r = results[f"{src}->{tgt}"]
        emoji = "OK" if r["degradation"] < 2.0 else "!!"
        print(
            f"  [{emoji}] {src}->{tgt}: {r['metrics']['rmse_total']:.4f} "
            f"({r['degradation']:.2f}x vs self-fit {r['self_rmse_tgt']:.4f})"
        )


if __name__ == "__main__":
    main()
