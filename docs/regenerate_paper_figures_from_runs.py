#!/usr/bin/env python3
"""
regenerate_paper_figures_from_runs.py
=====================================
TMCMC 完了後に新 run ディレクトリから論文用 Fig 2 (posterior predictive)
& Fig 3 (interaction matrix) を再生成する。

Usage:
    python regenerate_paper_figures_from_runs.py \
        --cs-dir _runs/CS_1000p_day1ic_XXXXXX \
        --ch-dir _runs/CH_1000p_day1ic_XXXXXX \
        --ds-dir _runs/DS_1000p_day1ic_XXXXXX \
        --dh-dir _runs/DH_1000p_day1ic_XXXXXX

    # Or auto-detect latest runs:
    python regenerate_paper_figures_from_runs.py --auto
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ============================================================
# Paths
# ============================================================
ROOT = Path(__file__).resolve().parent.parent
if ROOT.name == "docs":
    ROOT = ROOT.parent
RUNS = ROOT / "data_5species" / "main" / "_runs"
OUT = ROOT / "docs" / "paper_figures"
OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# Publication style
# ============================================================
SPECIES_NAMES = ["S. oralis", "A. naeslundii", "V. dispar", "F. nucleatum", "P. gingivalis"]
SPECIES_SHORT = ["So", "An", "Vd", "Fn", "Pg"]
SPECIES_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
COND_LABELS = {
    "CS": "Commensal Static",
    "CH": "Commensal HOBIC",
    "DS": "Dysbiotic Static",
    "DH": "Dysbiotic HOBIC",
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 17,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 12,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.linewidth": 1.3,
        "xtick.major.width": 1.1,
        "ytick.major.width": 1.1,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
    }
)


# ============================================================
# Data loading
# ============================================================
def load_run(run_dir: Path):
    """Load theta_MAP, theta_mean, fit_metrics, config from a run directory."""
    result = {}
    for name in ["theta_MAP", "theta_mean", "fit_metrics", "config"]:
        p = run_dir / f"{name}.json"
        if p.exists():
            with open(p) as f:
                result[name] = json.load(f)
    return result


def theta_to_matrices(theta):
    """Convert 20-element theta to A(5x5) and b(5)."""
    A = np.zeros((5, 5))
    b = np.zeros(5)
    A[0, 0], A[0, 1], A[1, 1] = theta[0], theta[1], theta[2]
    A[1, 0] = theta[1]
    b[0], b[1] = theta[3], theta[4]
    A[2, 2], A[2, 3], A[3, 3] = theta[5], theta[6], theta[7]
    A[3, 2] = theta[6]
    b[2], b[3] = theta[8], theta[9]
    A[0, 2], A[2, 0] = theta[10], theta[10]
    A[0, 3], A[3, 0] = theta[11], theta[11]
    A[1, 2], A[2, 1] = theta[12], theta[12]
    A[1, 3], A[3, 1] = theta[13], theta[13]
    A[4, 4] = theta[14]
    b[4] = theta[15]
    A[0, 4], A[4, 0] = theta[16], theta[16]
    A[1, 4], A[4, 1] = theta[17], theta[17]
    A[2, 4], A[4, 2] = theta[18], theta[18]
    A[3, 4], A[4, 3] = theta[19], theta[19]
    return A, b


# ============================================================
# Fig 3: Interaction matrix heatmaps
# ============================================================
def _is_locked(val):
    """Check if a cell is locked (inactive parameter = 0)."""
    return abs(val) < 0.01


def _heatmap_text_color(val, vmax):
    """Choose text color for readability: white on dark, black on light."""
    if _is_locked(val):
        return "#c0c0c0"  # very light gray for locked "—"
    if abs(val) > vmax * 0.55:
        return "white"
    return "black"


def generate_fig3(runs: dict):
    """Generate Fig 3 panels from new TMCMC results."""
    print("=== Fig 3: Interaction matrices ===")

    from matplotlib.colors import LinearSegmentedColormap

    # Custom diverging colormap: deep blue -> white -> deep red, with more contrast
    cmap_custom = LinearSegmentedColormap.from_list(
        "custom_div",
        [
            "#2166ac",
            "#4393c3",
            "#92c5de",
            "#d1e5f0",
            "#f7f7f7",
            "#fddbc7",
            "#f4a582",
            "#d6604d",
            "#b2182b",
        ],
        N=256,
    )

    for i, (cond, run_dir) in enumerate(runs.items()):
        data = load_run(run_dir)
        if "theta_MAP" not in data or "theta_mean" not in data:
            print(f"  {cond}: SKIP (missing theta_MAP/theta_mean in {run_dir})")
            continue

        theta_map = np.array(data["theta_MAP"]["theta_full"])
        theta_mean = np.array(data["theta_mean"]["theta_full"])
        A_map, b_map = theta_to_matrices(theta_map)
        A_mean, b_mean = theta_to_matrices(theta_mean)

        fig, axes = plt.subplots(
            1, 3, figsize=(14, 5.5), gridspec_kw={"width_ratios": [4.5, 4.5, 1.2], "wspace": 0.15}
        )

        vmax = max(abs(A_map).max(), abs(A_mean).max())
        # Ensure vmax is at least 0.5 to avoid overly saturated maps
        vmax = max(vmax, 0.5)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        for ax, A, title in [(axes[0], A_map, "MAP Estimate"), (axes[1], A_mean, "Posterior Mean")]:
            # Create masked array: gray out locked (zero) cells
            masked = np.ma.array(A)
            im = ax.imshow(masked, cmap=cmap_custom, norm=norm, aspect="equal")

            # Draw cell borders
            for edge in range(6):
                ax.axhline(edge - 0.5, color="white", linewidth=1.5)
                ax.axvline(edge - 0.5, color="white", linewidth=1.5)

            # Gray overlay for locked cells
            for ii in range(5):
                for jj in range(5):
                    if abs(A[ii, jj]) < 0.01:
                        ax.add_patch(
                            plt.Rectangle(
                                (jj - 0.5, ii - 0.5),
                                1,
                                1,
                                facecolor="#f0f0f0",
                                edgecolor="white",
                                linewidth=1.5,
                            )
                        )

            ax.set_xticks(range(5))
            ax.set_yticks(range(5))
            ax.set_xticklabels(SPECIES_SHORT, fontsize=15)
            ax.set_yticklabels(SPECIES_SHORT, fontsize=15)
            ax.set_title(title, fontsize=17, fontweight="bold", pad=8)
            for ii in range(5):
                for jj in range(5):
                    # Show "—" for locked cells instead of "0.00"
                    cell_text = "—" if _is_locked(A[ii, jj]) else f"{A[ii,jj]:.2f}"
                    ax.text(
                        jj,
                        ii,
                        cell_text,
                        ha="center",
                        va="center",
                        fontsize=14 if not _is_locked(A[ii, jj]) else 12,
                        fontweight="bold" if not _is_locked(A[ii, jj]) else "normal",
                        color=_heatmap_text_color(A[ii, jj], vmax),
                    )

        # Colorbar below the two heatmaps (horizontal)
        cbar_ax = fig.add_axes([0.08, -0.02, 0.55, 0.025])  # [left, bottom, width, height]
        cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label("Interaction strength $a_{ij}$", fontsize=13)

        ax3 = axes[2]
        y_pos = np.arange(5)
        ax3.barh(y_pos, b_map, 0.35, label="MAP", color="#4393c3", edgecolor="white", linewidth=0.5)
        ax3.barh(
            y_pos + 0.37,
            b_mean,
            0.35,
            label="Mean",
            color="#d6604d",
            edgecolor="white",
            linewidth=0.5,
        )
        ax3.set_yticks(y_pos + 0.175)
        ax3.set_yticklabels(SPECIES_SHORT, fontsize=15)
        ax3.set_xlabel(r"$b_i$", fontsize=16)
        ax3.set_title("Decay", fontsize=17, fontweight="bold", pad=8)
        ax3.legend(fontsize=11, loc="lower right", framealpha=0.9)
        ax3.grid(True, alpha=0.3, axis="x")
        ax3.invert_yaxis()

        fig.suptitle(
            f"{COND_LABELS[cond]}: Interaction Matrix $A$ & Decay $b$",
            fontsize=19,
            fontweight="bold",
            y=1.01,
        )

        tag = chr(97 + i)
        for ext in ["png", "pdf"]:
            out = OUT / f"fig3{tag}_{cond}_interaction_matrix.{ext}"
            fig.savefig(out, dpi=300 if ext == "png" else None)
        plt.close()

        if "fit_metrics" in data:
            rmse_map = data["fit_metrics"]["MAP"]["rmse_total"]
            rmse_mean = data["fit_metrics"]["Mean"]["rmse_total"]
            print(f"  {cond}: MAP RMSE={rmse_map:.4f}, Mean RMSE={rmse_mean:.4f}")
        else:
            print(f"  {cond}: saved (no fit_metrics)")


# ============================================================
# Fig 2: Posterior predictive per-species panels
# ============================================================
def _backup_existing(filepath: Path):
    """Backup existing file by appending _prev suffix before overwriting."""
    if filepath.exists():
        stem = filepath.stem
        suffix = filepath.suffix
        backup = filepath.parent / f"{stem}_prev{suffix}"
        # If _prev already exists, keep it (don't chain backups)
        if not backup.exists():
            import shutil

            shutil.copy2(filepath, backup)
            print(f"    backed up: {backup.name}")


def _setup_solver_and_data(runs, cond, run_dir):
    """Common setup: load data, create solver, return components."""
    sys.path.insert(0, str(ROOT / "data_5species"))
    sys.path.insert(0, str(ROOT / "data_5species" / "main"))

    from improved_5species_jit import BiofilmNewtonSolver5S
    from estimate_reduced_nishioka import load_experimental_data

    data_run = load_run(run_dir)
    if "theta_MAP" not in data_run or "config" not in data_run:
        return None

    config = data_run["config"]
    cond_map = {
        "CS": ("Commensal", "Static"),
        "CH": ("Commensal", "HOBIC"),
        "DS": ("Dysbiotic", "Static"),
        "DH": ("Dysbiotic", "HOBIC"),
    }
    condition, cultivation = cond_map.get(cond, ("Commensal", "Static"))

    data_dir = ROOT / "data_5species"
    exp_data, t_days, sigma_obs, phi_init_exp, metadata = load_experimental_data(
        data_dir, condition, cultivation, start_from_day=3
    )

    theta_map = np.array(data_run["theta_MAP"]["theta_full"])
    dt = config.get("dt", 0.001)
    maxtimestep = config.get("maxtimestep", 50000)

    def make_solver(phi_init):
        return BiofilmNewtonSolver5S(
            dt=dt,
            maxtimestep=maxtimestep,
            c_const=config.get("c_const", 1.0),
            alpha_const=config.get("alpha_const", 1.0),
            phi_init=phi_init / phi_init.sum() if phi_init.sum() > 0 else phi_init,
            Kp1=config.get("kp1", 1.0),
            K_hill=config.get("K_hill", 0.05),
            n_hill=config.get("n_hill", 4),
        )

    # Load posterior samples if available
    samples_path = run_dir / "samples.npy"
    samples = np.load(samples_path) if samples_path.exists() else None

    return {
        "data_run": data_run,
        "config": config,
        "condition": condition,
        "cultivation": cultivation,
        "exp_data": exp_data,
        "t_days": t_days,
        "phi_init_exp": phi_init_exp,
        "theta_map": theta_map,
        "make_solver": make_solver,
        "samples": samples,
    }


def _forward_solve_phibar(make_solver, theta, phi_init):
    """Run forward model, return (t_arr, phibar[n_time, 5])."""
    solver = make_solver(phi_init)
    t_arr, x0 = solver.solve(theta)
    phibar = np.zeros((x0.shape[0], 5))
    for si in range(5):
        phibar[:, si] = x0[:, si] * x0[:, 5 + si]
    return t_arr, phibar


def generate_fig2_new(runs: dict):
    """Generate Fig 2 NEW style: 1x5 compact panels with MAP only."""
    print("\n=== Fig 2 (new style): 1x5 MAP posterior predictive ===")

    sys.path.insert(0, str(ROOT / "data_5species"))
    sys.path.insert(0, str(ROOT / "data_5species" / "main"))
    try:
        from improved_5species_jit import BiofilmNewtonSolver5S  # noqa: F401
    except ImportError as e:
        print(f"  Cannot import solver: {e}")
        return

    for i, (cond, run_dir) in enumerate(runs.items()):
        ctx = _setup_solver_and_data(runs, cond, run_dir)
        if ctx is None:
            print(f"  {cond}: SKIP (missing theta_MAP/config)")
            continue

        t_arr, phibar = _forward_solve_phibar(
            ctx["make_solver"], ctx["theta_map"], ctx["phi_init_exp"]
        )
        day_max = ctx["t_days"].max()
        t_plot = (t_arr - t_arr.min()) / (t_arr.max() - t_arr.min()) * (day_max / 0.95)

        fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=False)
        fig.suptitle(COND_LABELS[cond], fontsize=20, fontweight="bold", y=1.02)

        for si in range(5):
            ax = axes[si]
            ax.plot(t_plot, phibar[:, si], color=SPECIES_COLORS[si], linewidth=2, label="MAP")
            ax.scatter(
                ctx["t_days"],
                ctx["exp_data"][:, si],
                color=SPECIES_COLORS[si],
                edgecolor="k",
                s=60,
                zorder=10,
                label="Data",
            )
            ax.scatter(
                [1],
                [ctx["phi_init_exp"][si]],
                color=SPECIES_COLORS[si],
                marker="D",
                edgecolor="k",
                s=80,
                zorder=10,
                alpha=0.6,
            )
            ax.set_title(SPECIES_NAMES[si], fontsize=15, fontweight="bold")
            ax.set_xlabel("Days", fontsize=14)
            if si == 0:
                ax.set_ylabel(r"$\bar{\varphi}_i$", fontsize=16)
            ax.set_xticks([1, 3, 7, 10, 15, 21])
            ax.set_xlim(0, 22)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        tag = chr(97 + i)
        for ext in ["png", "pdf"]:
            out = OUT / f"fig2{tag}_{cond}_posterior_predictive.{ext}"
            _backup_existing(out)
            fig.savefig(out, dpi=300 if ext == "png" else None)
        plt.close()
        print(f"  {cond}: saved fig2{tag} (new style)")


def generate_fig2_classic(runs: dict):
    """Generate Fig 2 CLASSIC style: 2x3 layout with CI bands, paper-compatible naming."""
    print("\n=== Fig 2 (classic style): 2x3 with CI bands ===")

    sys.path.insert(0, str(ROOT / "data_5species"))
    sys.path.insert(0, str(ROOT / "data_5species" / "main"))
    try:
        from improved_5species_jit import BiofilmNewtonSolver5S  # noqa: F401
    except ImportError as e:
        print(f"  Cannot import solver: {e}")
        return

    PAPER_OUT = ROOT / "data_5species" / "docs" / "paper_comprehensive_figs"
    PAPER_OUT.mkdir(parents=True, exist_ok=True)

    # Light fill colors per species (for CI bands)
    fill_colors = [
        (0.12, 0.47, 0.71, 0.15),  # So blue
        (1.00, 0.50, 0.05, 0.15),  # An orange
        (0.17, 0.63, 0.17, 0.15),  # Vd green
        (0.84, 0.15, 0.16, 0.15),  # Fn red
        (0.58, 0.40, 0.74, 0.15),  # Pg purple
    ]
    fill_colors_dark = [
        (0.12, 0.47, 0.71, 0.30),
        (1.00, 0.50, 0.05, 0.30),
        (0.17, 0.63, 0.17, 0.30),
        (0.84, 0.15, 0.16, 0.30),
        (0.58, 0.40, 0.74, 0.30),
    ]

    for i, (cond, run_dir) in enumerate(runs.items()):
        ctx = _setup_solver_and_data(runs, cond, run_dir)
        if ctx is None:
            print(f"  {cond}: SKIP")
            continue

        # Forward solve MAP
        t_arr, phibar_map = _forward_solve_phibar(
            ctx["make_solver"], ctx["theta_map"], ctx["phi_init_exp"]
        )
        day_max = ctx["t_days"].max()
        t_plot = (t_arr - t_arr.min()) / (t_arr.max() - t_arr.min()) * (day_max / 0.95)

        # Compute posterior CI bands from samples
        samples = ctx["samples"]
        n_subsample = min(100, len(samples)) if samples is not None else 0
        phibar_all = None

        if n_subsample > 0:
            idx = np.random.choice(len(samples), n_subsample, replace=False)
            phibar_all = np.zeros((n_subsample, len(t_arr), 5))
            n_failed = 0
            for j, si_idx in enumerate(idx):
                try:
                    _, pb = _forward_solve_phibar(
                        ctx["make_solver"], samples[si_idx], ctx["phi_init_exp"]
                    )
                    if pb.shape[0] == len(t_arr):
                        phibar_all[j] = pb
                    else:
                        # Interpolate to match t_arr length
                        for sp in range(5):
                            phibar_all[j, :, sp] = np.interp(
                                np.linspace(0, 1, len(t_arr)),
                                np.linspace(0, 1, pb.shape[0]),
                                pb[:, sp],
                            )
                except Exception:
                    phibar_all[j] = phibar_map  # fallback
                    n_failed += 1
            if n_failed > 0:
                print(f"    {cond}: {n_failed}/{n_subsample} samples failed, used MAP fallback")

        # 2x3 layout (3 top, 2 bottom centered)
        fig = plt.figure(figsize=(14, 8))
        gs = fig.add_gridspec(2, 6, hspace=0.35, wspace=0.4)
        axes = [
            fig.add_subplot(gs[0, 0:2]),  # So
            fig.add_subplot(gs[0, 2:4]),  # An
            fig.add_subplot(gs[0, 4:6]),  # Vd
            fig.add_subplot(gs[1, 1:3]),  # Fn
            fig.add_subplot(gs[1, 3:5]),  # Pg
        ]

        title = f"{ctx['condition']} {ctx['cultivation']}: Per-Species Posterior Predictive Fits"
        fig.suptitle(title, fontsize=18, fontweight="bold", y=0.98)

        for si in range(5):
            ax = axes[si]

            # CI bands
            if phibar_all is not None:
                p5 = np.percentile(phibar_all[:, :, si], 5, axis=0)
                p25 = np.percentile(phibar_all[:, :, si], 25, axis=0)
                p50 = np.percentile(phibar_all[:, :, si], 50, axis=0)
                p75 = np.percentile(phibar_all[:, :, si], 75, axis=0)
                p95 = np.percentile(phibar_all[:, :, si], 95, axis=0)

                ax.fill_between(t_plot, p5, p95, color=fill_colors[si], label="5-95%")
                ax.fill_between(t_plot, p25, p75, color=fill_colors_dark[si], label="25-75%")
                ax.plot(t_plot, p50, color=SPECIES_COLORS[si], linewidth=1.2, label="Median")

            # MAP line
            ax.plot(
                t_plot,
                phibar_map[:, si],
                color=SPECIES_COLORS[si],
                linewidth=2,
                linestyle="--",
                label="MAP",
            )

            # Data points
            ax.scatter(
                ctx["t_days"],
                ctx["exp_data"][:, si],
                color=SPECIES_COLORS[si],
                edgecolor="k",
                s=70,
                zorder=10,
                label="Data",
            )

            # Day 1 IC
            ax.scatter(
                [1],
                [ctx["phi_init_exp"][si]],
                color=SPECIES_COLORS[si],
                marker="D",
                edgecolor="k",
                s=80,
                zorder=10,
                alpha=0.7,
            )

            ax.set_title(SPECIES_NAMES[si], fontsize=15, fontweight="bold")
            ax.set_xlabel("Days", fontsize=13)
            ax.set_ylabel(r"$\bar{\varphi}$", fontsize=14)
            ax.set_xticks([1, 3, 6, 10, 15, 21])
            ax.set_xlim(0, 22)
            ax.legend(fontsize=8, loc="best", framealpha=0.8)

        # Save to both paper_figures/ and paper_comprehensive_figs/
        tag = chr(97 + i)
        cond_name = f"{ctx['condition']}_{ctx['cultivation']}"
        for ext in ["png", "pdf"]:
            # paper_figures/ output
            out1 = OUT / f"fig2{tag}_{cond}_classic.{ext}"
            fig.savefig(out1, dpi=300 if ext == "png" else None)

            # paper_comprehensive_figs/ output (paper-compatible naming)
            out2 = PAPER_OUT / f"{cond_name}_Fig_A02_per_species_panel.{ext}"
            _backup_existing(out2)
            fig.savefig(out2, dpi=300 if ext == "png" else None)

        plt.close()
        print(f"  {cond}: saved classic style (CI bands, 2x3)")


def generate_fig2(runs: dict):
    """Generate Fig 2 in both styles."""
    generate_fig2_new(runs)
    generate_fig2_classic(runs)


# ============================================================
# Fig 2 + RMSE summary table
# ============================================================
def print_rmse_table(runs: dict):
    """Print RMSE comparison table."""
    print("\n=== RMSE Summary ===")
    print(f"{'Cond':<6} {'MAP RMSE':>10} {'Mean RMSE':>11} {'Dir'}")
    print("-" * 60)
    for cond, run_dir in runs.items():
        data = load_run(run_dir)
        if "fit_metrics" in data:
            m = data["fit_metrics"]
            print(
                f"{cond:<6} {m['MAP']['rmse_total']:>10.4f} {m['Mean']['rmse_total']:>11.4f} {run_dir.name}"
            )
        else:
            print(f"{cond:<6} {'N/A':>10} {'N/A':>11} {run_dir.name}")


# ============================================================
# Auto-detect latest runs
# ============================================================
def find_latest_runs(pattern_prefix: str = ""):
    """Find latest completed run for each condition."""
    cond_prefixes = {
        "CS": ["CS_", "Commensal_Static_"],
        "CH": ["CH_", "Commensal_HOBIC_"],
        "DS": ["DS_", "Dysbiotic_Static_"],
        "DH": ["DH_", "Dysbiotic_HOBIC_"],
    }

    runs = {}
    for cond, prefixes in cond_prefixes.items():
        candidates = []
        for prefix in prefixes:
            full_prefix = pattern_prefix + prefix if pattern_prefix else prefix
            for d in RUNS.iterdir():
                if d.is_dir() and d.name.startswith(full_prefix):
                    # Check if run completed (has theta_MAP.json)
                    if (d / "theta_MAP.json").exists():
                        candidates.append(d)
        if candidates:
            # Sort by modification time, pick latest
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            runs[cond] = candidates[0]
            print(f"  {cond}: {candidates[0].name}")
        else:
            print(f"  {cond}: NOT FOUND")

    return runs


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Regenerate paper figures from TMCMC runs")
    parser.add_argument("--cs-dir", type=str, help="CS run directory")
    parser.add_argument("--ch-dir", type=str, help="CH run directory")
    parser.add_argument("--ds-dir", type=str, help="DS run directory")
    parser.add_argument("--dh-dir", type=str, help="DH run directory")
    parser.add_argument("--auto", action="store_true", help="Auto-detect latest day1ic runs")
    parser.add_argument(
        "--prefix", type=str, default="", help="Run dir prefix filter (e.g. '1000p_day1ic')"
    )
    parser.add_argument("--fig3-only", action="store_true", help="Only generate Fig 3")
    parser.add_argument("--fig2-only", action="store_true", help="Only generate Fig 2")
    args = parser.parse_args()

    if args.auto:
        print("Auto-detecting latest completed runs...")
        runs = find_latest_runs(args.prefix)
    else:
        runs = {}
        for cond, arg_name in [
            ("CS", "cs_dir"),
            ("CH", "ch_dir"),
            ("DS", "ds_dir"),
            ("DH", "dh_dir"),
        ]:
            val = getattr(args, arg_name)
            if val:
                p = Path(val)
                if not p.is_absolute():
                    p = RUNS / val
                if p.exists():
                    runs[cond] = p
                else:
                    print(f"  {cond}: directory not found: {p}")

    if not runs:
        print("No runs found. Use --auto or specify --cs-dir etc.")
        return

    print(f"\nUsing {len(runs)} conditions: {list(runs.keys())}")
    print(f"Output: {OUT}\n")

    if not args.fig2_only:
        generate_fig3(runs)
    if not args.fig3_only:
        generate_fig2(runs)

    print_rmse_table(runs)
    print(f"\nDone! Figures saved to {OUT}")


if __name__ == "__main__":
    main()
