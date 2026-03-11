#!/usr/bin/env python3
"""Generate cross-condition comparison figures for paper.

Usage:
    python3 generate_comparison_figures.py \
        --cs _runs/CS_1000p_expIC_repSigma_... \
        --ch _runs/CH_1000p_expIC_repSigma_... \
        --ds _runs/DS_1000p_expIC_repSigma_... \
        --dh _runs/DH_1000p_expIC_repSigma_...

    # Auto-detect latest runs:
    python3 generate_comparison_figures.py --auto
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "../.."))

from data_5species.visualization.helpers import compute_phibar, load_exp_boxplot
from data_5species.visualization.plot_manager import PlotManager
from tmcmc.program2602.improved_5species_jit import BiofilmNewtonSolver5S

SPECIES = ["S. oralis", "A. naeslundii", "V. dispar", "F. nucleatum", "P. gingivalis"]
SP_SHORT = ["So", "An", "Vd", "Fn", "Pg"]
COND_LABELS = {
    "CS": "Commensal Static",
    "CH": "Commensal HOBIC",
    "DS": "Dysbiotic Static",
    "DH": "Dysbiotic HOBIC",
}
COND_COLORS = {"CS": "#2196F3", "CH": "#4CAF50", "DS": "#FF9800", "DH": "#F44336"}
ACTIVE_SPECIES = [0, 1, 2, 3, 4]


def load_run(run_dir: Path) -> dict:
    """Load all relevant data from a run directory."""
    run_dir = Path(run_dir)
    info = {"dir": run_dir}

    cfg = json.load(open(run_dir / "config.json"))
    rs = json.load(open(run_dir / "results_summary.json"))
    info["config"] = cfg
    info["MAP"] = np.array(rs["MAP"])
    info["mean"] = np.array(rs["mean"])
    info["samples"] = np.load(run_dir / "samples.npy")
    info["data"] = np.load(run_dir / "data.npy")
    info["idx_sparse"] = np.load(run_dir / "idx_sparse.npy")
    info["t_days"] = np.load(run_dir / "t_days.npy")
    info["condition"] = cfg.get("condition", "?")
    info["cultivation"] = cfg.get("cultivation", "?")

    phi_init = np.array(cfg["phi_init"]) if cfg.get("phi_init_is_array") else None
    info["phi_init"] = phi_init

    # Diagnostics
    diag_path = run_dir / "mcmc_diagnostics.json"
    if diag_path.exists():
        info["diagnostics"] = json.load(open(diag_path))

    # Fit metrics
    fm_path = run_dir / "fit_metrics.json"
    if fm_path.exists():
        info["fit_metrics"] = json.load(open(fm_path))

    return info


def theta_to_AB(theta):
    """Convert 20-param theta to A(5x5), b(5)."""
    A = np.zeros((5, 5))
    b = np.zeros(5)
    # M1
    A[0, 0], A[0, 1], A[1, 1] = theta[0], theta[1], theta[2]
    A[1, 0] = theta[1]
    b[0], b[1] = theta[3], theta[4]
    # M2
    A[2, 2], A[2, 3], A[3, 3] = theta[5], theta[6], theta[7]
    A[3, 2] = theta[6]
    b[2], b[3] = theta[8], theta[9]
    # M3 cross
    A[0, 2] = A[2, 0] = theta[10]
    A[0, 3] = A[3, 0] = theta[11]
    A[1, 2] = A[2, 1] = theta[12]
    A[1, 3] = A[3, 1] = theta[13]
    # Sp5
    A[4, 4] = theta[14]
    b[4] = theta[15]
    A[0, 4] = A[4, 0] = theta[16]
    A[1, 4] = A[4, 1] = theta[17]
    A[2, 4] = A[4, 2] = theta[18]
    A[3, 4] = A[4, 3] = theta[19]
    return A, b


# ============================================================
# Fig 1: A matrix heatmap (4 conditions)
# ============================================================
def fig_A_heatmap(runs: dict, out_dir: Path):
    """4-panel A matrix heatmap comparison."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    vmin_all, vmax_all = np.inf, -np.inf
    A_maps = {}
    for key in ["CS", "CH", "DS", "DH"]:
        A, _ = theta_to_AB(runs[key]["MAP"])
        A_maps[key] = A
        vmin_all = min(vmin_all, A.min())
        vmax_all = max(vmax_all, A.max())

    # Symmetric colorbar
    vabs = max(abs(vmin_all), abs(vmax_all))

    for ax_idx, key in enumerate(["CS", "CH", "DS", "DH"]):
        ax = axes[ax_idx]
        A = A_maps[key]
        im = ax.imshow(A, cmap="RdBu_r", vmin=-vabs, vmax=vabs, aspect="equal")
        ax.set_xticks(range(5))
        ax.set_xticklabels(SP_SHORT, fontsize=10)
        ax.set_yticks(range(5))
        ax.set_yticklabels(SP_SHORT, fontsize=10)
        ax.set_title(COND_LABELS[key], fontsize=12, fontweight="bold")

        # Annotate values
        for i in range(5):
            for j in range(5):
                val = A[i, j]
                if abs(val) > 0.001:
                    color = "white" if abs(val) > vabs * 0.6 else "black"
                    ax.text(
                        j,
                        i,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=color,
                        fontweight="bold",
                    )

    fig.colorbar(im, ax=axes, shrink=0.8, label="Interaction strength $a_{ij}$")
    fig.suptitle(
        "Interaction Matrix A — MAP Estimates (4 Conditions)", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    path = out_dir / "comparison_A_heatmap.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ============================================================
# Fig 2: B vector comparison (bar + CI)
# ============================================================
def fig_B_comparison(runs: dict, out_dir: Path):
    """Growth rate b comparison across 4 conditions with posterior CI."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=False)

    b_indices = [3, 4, 8, 9, 15]  # theta indices for b1..b5
    cond_keys = ["CS", "CH", "DS", "DH"]

    for sp_idx in range(5):
        ax = axes[sp_idx]
        b_theta_idx = b_indices[sp_idx]

        positions = np.arange(4)
        for ci, key in enumerate(cond_keys):
            samples_b = runs[key]["samples"][:, b_theta_idx]
            median = np.median(samples_b)
            q16 = np.percentile(samples_b, 15.865)
            q84 = np.percentile(samples_b, 84.135)
            q2 = np.percentile(samples_b, 2.275)
            q98 = np.percentile(samples_b, 97.725)

            color = COND_COLORS[key]
            # 2σ whisker
            ax.plot([ci, ci], [q2, q98], color=color, linewidth=1.5, alpha=0.4)
            # 1σ bar
            ax.plot([ci, ci], [q16, q84], color=color, linewidth=4, alpha=0.6)
            # Median dot
            ax.plot(
                ci,
                median,
                "o",
                color=color,
                markersize=8,
                markeredgecolor="black",
                markeredgewidth=1.0,
                zorder=5,
            )
            # MAP marker
            b_map = runs[key]["MAP"][b_theta_idx]
            ax.plot(
                ci,
                b_map,
                "D",
                color="white",
                markersize=5,
                markeredgecolor=color,
                markeredgewidth=1.5,
                zorder=6,
            )

        ax.set_xticks(positions)
        ax.set_xticklabels(cond_keys, fontsize=10)
        ax.set_title(f"$b_{sp_idx+1}$ ({SP_SHORT[sp_idx]})", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        if sp_idx == 0:
            ax.set_ylabel("Growth rate", fontsize=11)

    # Legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="gray",
            markersize=8,
            markeredgecolor="black",
            linestyle="None",
            label="Median",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="white",
            markersize=5,
            markeredgecolor="gray",
            markeredgewidth=1.5,
            linestyle="None",
            label="MAP",
        ),
        Line2D([0], [0], color="gray", linewidth=4, alpha=0.6, label="1σ"),
        Line2D([0], [0], color="gray", linewidth=1.5, alpha=0.4, label="2σ"),
    ]
    fig.legend(
        handles=legend_elements, loc="upper right", fontsize=9, ncol=4, bbox_to_anchor=(0.98, 1.0)
    )

    fig.suptitle(
        "Growth Rates b — Posterior Comparison (4 Conditions)", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    path = out_dir / "comparison_B_growth_rates.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ============================================================
# Fig 3: 4×5 panel (all conditions × all species)
# ============================================================
def fig_4x5_panel(runs: dict, out_dir: Path, n_draws: int = 50):
    """4 rows (conditions) × 5 cols (species) posterior predictive."""
    cond_keys = ["CS", "CH", "DS", "DH"]
    fig, axes = plt.subplots(4, 5, figsize=(22, 16), sharex=False, sharey=False)

    colors = PlotManager.COLORS

    for row, key in enumerate(cond_keys):
        r = runs[key]
        cfg = r["config"]
        phi_init = r["phi_init"]

        solver = BiofilmNewtonSolver5S(
            maxtimestep=cfg.get("maxtimestep", 2500),
            dt=cfg.get("dt", 1.0),
            active_species=ACTIVE_SPECIES,
            c_const=cfg.get("c_const", 0.01),
            Kp1=cfg.get("Kp1", 0.01),
            K_hill=cfg.get("K_hill", 0.05),
            n_hill=cfg.get("n_hill", 4),
            phi_init=phi_init if phi_init is not None else 0.2,
        )

        # MAP trajectory
        t_fit, x_map = solver.solve(r["MAP"])
        phibar_map = compute_phibar(x_map, ACTIVE_SPECIES)

        # Sample trajectories
        np.random.seed(42)
        n_samp = min(n_draws, r["samples"].shape[0])
        indices = np.random.choice(r["samples"].shape[0], n_samp, replace=False)
        phibar_samples = []
        for idx in indices:
            theta_full = r["MAP"].copy()
            theta_full[:20] = r["samples"][idx]
            _, x_sim = solver.solve(theta_full)
            phibar_samples.append(compute_phibar(x_sim, ACTIVE_SPECIES))
        phibar_samples = np.array(phibar_samples)

        # Time mapping
        t_known = t_fit[r["idx_sparse"][-1]]
        d_known = float(r["t_days"][-1])
        t_plot = t_fit * (d_known / t_known) if t_known > 0 else t_fit

        # Percentiles
        q_2lo = np.nanpercentile(phibar_samples, 2.275, axis=0)
        q_1lo = np.nanpercentile(phibar_samples, 15.865, axis=0)
        q50 = np.nanpercentile(phibar_samples, 50, axis=0)
        q_1hi = np.nanpercentile(phibar_samples, 84.135, axis=0)
        q_2hi = np.nanpercentile(phibar_samples, 97.725, axis=0)

        # Exp boxplot
        try:
            exp_bp = load_exp_boxplot(r["condition"], r["cultivation"])
        except Exception:
            exp_bp = None

        for col, sp in enumerate(ACTIVE_SPECIES):
            ax = axes[row, col]
            c = colors[sp]

            # Bands
            ax.fill_between(t_plot, q_2lo[:, col], q_2hi[:, col], alpha=0.12, color=c)
            ax.fill_between(t_plot, q_1lo[:, col], q_1hi[:, col], alpha=0.25, color=c)
            ax.plot(t_plot, q50[:, col], linewidth=1.5, color=c)
            ax.plot(t_plot, phibar_map[:, col], "--", color="black", linewidth=1, alpha=0.7)

            # Boxplot
            if exp_bp and sp in exp_bp:
                bp = exp_bp[sp]
                hw = 0.4
                for j, day in enumerate(bp["days"]):
                    q1 = bp["q1"][j] if "q1" in bp else bp["median"][j]
                    q3 = bp["q3"][j] if "q3" in bp else bp["median"][j]
                    rect = plt.Rectangle(
                        (day - hw, q1),
                        2 * hw,
                        q3 - q1,
                        linewidth=0.8,
                        edgecolor=c,
                        facecolor=c,
                        alpha=0.2,
                        zorder=5,
                    )
                    ax.add_patch(rect)
                    ax.plot(
                        [day - hw, day + hw],
                        [bp["median"][j]] * 2,
                        color=c,
                        linewidth=1.5,
                        alpha=0.8,
                        zorder=6,
                    )
                    ax.plot(
                        [day, day], [bp["low"][j], q1], color=c, linewidth=0.8, alpha=0.4, zorder=4
                    )
                    ax.plot(
                        [day, day], [q3, bp["high"][j]], color=c, linewidth=0.8, alpha=0.4, zorder=4
                    )

            # Data
            ax.scatter(
                r["t_days"],
                r["data"][:, col],
                s=30,
                color=c,
                edgecolors="black",
                zorder=10,
                linewidth=0.8,
            )

            # IC
            if phi_init is not None and col < len(phi_init):
                ax.scatter(
                    [1],
                    [phi_init[col]],
                    s=40,
                    facecolors="white",
                    edgecolors=c,
                    zorder=11,
                    linewidth=1.2,
                    marker="D",
                )

            # Labels
            ax.grid(True, alpha=0.2)
            all_days = (
                sorted(set([1] + list(r["t_days"]))) if phi_init is not None else list(r["t_days"])
            )
            ax.set_xticks(all_days)
            ax.tick_params(labelsize=8)
            if row == 0:
                ax.set_title(SPECIES[sp], fontsize=11, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{COND_LABELS[key]}\n" + r"$\bar{\varphi}$", fontsize=9)
            if row == 3:
                ax.set_xlabel("Days", fontsize=9)

    fig.suptitle(
        "Posterior Predictive (1σ/2σ) — All Conditions × Species", fontsize=15, fontweight="bold"
    )
    plt.tight_layout()
    path = out_dir / "comparison_4x5_panel.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ============================================================
# Main
# ============================================================
def find_latest_run(runs_dir: Path, prefix: str) -> Path:
    """Find latest run dir matching prefix."""
    candidates = sorted(runs_dir.glob(f"{prefix}_*p_expIC_repSigma_*"), key=lambda p: p.name)
    if not candidates:
        candidates = sorted(runs_dir.glob(f"{prefix}_*"), key=lambda p: p.name)
    if not candidates:
        raise FileNotFoundError(f"No run found for prefix={prefix} in {runs_dir}")
    return candidates[-1]


def main():
    parser = argparse.ArgumentParser(description="Generate cross-condition comparison figures")
    parser.add_argument("--cs", type=str, default=None, help="CS run dir")
    parser.add_argument("--ch", type=str, default=None, help="CH run dir")
    parser.add_argument("--ds", type=str, default=None, help="DS run dir")
    parser.add_argument("--dh", type=str, default=None, help="DH run dir")
    parser.add_argument("--auto", action="store_true", help="Auto-detect latest runs")
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Output dir (default: _runs/comparison)"
    )
    parser.add_argument("--n-draws", type=int, default=50, help="Posterior draws for 4x5 panel")
    args = parser.parse_args()

    runs_dir = Path(__file__).parent / "_runs"

    if args.auto:
        dirs = {
            "CS": find_latest_run(runs_dir, "CS"),
            "CH": find_latest_run(runs_dir, "CH"),
            "DS": find_latest_run(runs_dir, "DS"),
            "DH": find_latest_run(runs_dir, "DH"),
        }
    else:
        dirs = {}
        for key, val in [("CS", args.cs), ("CH", args.ch), ("DS", args.ds), ("DH", args.dh)]:
            if val is None:
                try:
                    dirs[key] = find_latest_run(runs_dir, key)
                except FileNotFoundError:
                    print(f"Warning: No run found for {key}, skipping", file=sys.stderr)
            else:
                dirs[key] = Path(val)

    if len(dirs) < 4:
        print(
            f"Warning: Only {len(dirs)}/4 conditions found. Some figures may be incomplete.",
            file=sys.stderr,
        )

    # Load runs
    runs = {}
    for key, d in dirs.items():
        print(f"Loading {key}: {d.name}")
        try:
            runs[key] = load_run(d)
        except Exception as e:
            print(f"  Failed: {e}", file=sys.stderr)

    if len(runs) < 2:
        print("Need at least 2 conditions. Exiting.", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir) if args.output_dir else runs_dir / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")

    # Generate figures
    if len(runs) == 4:
        print("\n[1/3] A matrix heatmap...")
        fig_A_heatmap(runs, out_dir)

        print("[2/3] B growth rate comparison...")
        fig_B_comparison(runs, out_dir)

        print(f"[3/3] 4×5 panel ({args.n_draws} draws)...")
        fig_4x5_panel(runs, out_dir, n_draws=args.n_draws)
    else:
        avail = list(runs.keys())
        print(f"\nOnly {avail} available. Generating partial figures...")
        if len(runs) >= 2:
            fig_A_heatmap(runs, out_dir)
            fig_B_comparison(runs, out_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
