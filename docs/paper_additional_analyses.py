#!/usr/bin/env python3
"""
Paper Additional Analyses — standalone script producing publication figures + LaTeX tables.

Three analyses using Phase 2 (free-psi) TMCMC posterior results:
  1. So Initial-Condition Sensitivity
  2. Leave-One-Condition-Out Cross-Prediction
  3. Effective Dimensionality per Condition

Output:
  figures/so_ic_sensitivity.pdf
  figures/cross_condition_prediction.pdf
  figures/effective_dimensionality.pdf
  + LaTeX table fragments to stdout

Usage:
    python3 paper_additional_analyses.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Publication-quality matplotlib rcParams
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42,  # TrueType embedding for journals
        "ps.fonttype": 42,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJ = Path(__file__).resolve().parent.parent  # Tmcmc202601
RUNS = PROJ / "data_5species" / "main" / "_runs"
EXP_CSV = PROJ / "data_5species" / "experiment_data" / "fig3_species_distribution_summary.csv"
PRIOR_JSON = PROJ / "data_5species" / "model_config" / "prior_bounds.json"
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Condition -> latest run directory mapping (auto-detect latest)
COND_KEYS = {
    "CS": ("Commensal", "Static"),
    "CH": ("Commensal", "HOBIC"),
    "DS": ("Dysbiotic", "Static"),
    "DH": ("Dysbiotic", "HOBIC"),
}

COND_LABELS = {
    "CS": "Commensal Static",
    "CH": "Commensal HOBIC",
    "DS": "Dysbiotic Static",
    "DH": "Dysbiotic HOBIC",
}

SPECIES = ["So", "An", "Vd", "Fn", "Pg"]
SPECIES_FULL = ["S. oralis", "A. naeslundii", "V. dispar", "F. nucleatum", "P. gingivalis"]
SPECIES_COLORS = ["#1f77b4", "#2ca02c", "#ffc107", "#9467bd", "#d62728"]

EXP_DAYS = np.array([1, 3, 6, 10, 15, 21])

# Species name mapping in experimental CSV (different between conditions)
SP_MAP_COMMENSAL = {
    "S. oralis": 0,
    "A. naeslundii": 1,
    "V. dispar": 2,
    "F. nucleatum": 3,
    "P. gingivalis_20709": 4,
}
SP_MAP_DYSBIOTIC = {
    "S. oralis": 0,
    "A. naeslundii": 1,
    "V. parvula": 2,  # V. dispar analog
    "F. nucleatum": 3,
    "P. gingivalis_W83": 4,
}

# Parameter names (20 parameters)
PARAM_NAMES = [
    r"$a_{11}$",
    r"$a_{12}$",
    r"$a_{22}$",
    r"$b_1$",
    r"$b_2$",
    r"$a_{33}$",
    r"$a_{34}$",
    r"$a_{44}$",
    r"$b_3$",
    r"$b_4$",
    r"$a_{13}$",
    r"$a_{14}$",
    r"$a_{23}$",
    r"$a_{24}$",
    r"$a_{55}$",
    r"$b_5$",
    r"$a_{15}$",
    r"$a_{25}$",
    r"$a_{35}$",
    r"$a_{45}$",
]
PARAM_NAMES_LATEX = [
    "a_{11}",
    "a_{12}",
    "a_{22}",
    "b_1",
    "b_2",
    "a_{33}",
    "a_{34}",
    "a_{44}",
    "b_3",
    "b_4",
    "a_{13}",
    "a_{14}",
    "a_{23}",
    "a_{24}",
    "a_{55}",
    "b_5",
    "a_{15}",
    "a_{25}",
    "a_{35}",
    "a_{45}",
]


# ========================================================================
# ODE Model (Hamilton replicator dynamics with Hill gate)
# ========================================================================


def hill(x: np.ndarray, K: float = 0.05, n: float = 4.0) -> np.ndarray:
    xn = np.abs(x) ** n
    return xn / (K**n + xn)


def ode_rhs(phi: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Replicator dynamics with Hill-gated interactions."""
    a = np.zeros((5, 5))
    b = np.zeros(5)

    a[0, 0] = theta[0]
    a[0, 1] = theta[1]
    a[1, 1] = theta[2]
    b[0] = theta[3]
    b[1] = theta[4]
    a[2, 2] = theta[5]
    a[2, 3] = theta[6]
    a[3, 3] = theta[7]
    b[2] = theta[8]
    b[3] = theta[9]
    a[0, 2] = theta[10]
    a[0, 3] = theta[11]
    a[1, 2] = theta[12]
    a[1, 3] = theta[13]
    a[4, 4] = theta[14]
    b[4] = theta[15]
    a[0, 4] = theta[16]
    a[1, 4] = theta[17]
    a[2, 4] = theta[18]
    a[3, 4] = theta[19]

    for i in range(5):
        for j in range(i + 1, 5):
            a[j, i] = a[i, j]

    fitness = b.copy()
    for i in range(5):
        for j in range(5):
            if a[i, j] != 0:
                fitness[i] += a[i, j] * hill(phi[j : j + 1])[0]

    mean_f = np.dot(phi, fitness)
    return phi * (fitness - mean_f)


def simulate(
    theta: np.ndarray, ic: np.ndarray, n_steps: int = 2500, dt: float = 1e-4
) -> np.ndarray:
    """Run forward model. Returns trajectory (n_steps+1, 5)."""
    x = np.array(ic, dtype=np.float64)
    traj = [x.copy()]
    for _ in range(n_steps):
        dx = ode_rhs(x, theta)
        x = x + dt * dx
        x = np.clip(x, 1e-12, None)
        x = x / x.sum()
        traj.append(x.copy())
    return np.array(traj)


# ========================================================================
# Data Loading Helpers
# ========================================================================


def find_latest_run(cond_key: str) -> Path:
    """Find the latest NUTS run directory for a condition."""
    condition, cultivation = COND_KEYS[cond_key]
    prefix = f"jax_ode_nuts_{condition}_{cultivation}_"
    candidates = sorted(RUNS.glob(prefix + "*"))
    if not candidates:
        raise FileNotFoundError(f"No run directories matching {prefix}* in {RUNS}")
    return candidates[-1]


def load_map_theta(run_dir: Path) -> np.ndarray:
    """Load MAP theta from a run directory."""
    with open(run_dir / "theta_MAP.json") as f:
        d = json.load(f)
    return np.array([d[str(i)] for i in range(20)])


def load_samples(run_dir: Path) -> np.ndarray:
    """Load posterior samples (N, 20)."""
    return np.load(run_dir / "samples.npy")


def load_config(run_dir: Path) -> dict:
    """Load config.json from a run directory."""
    with open(run_dir / "config.json") as f:
        return json.load(f)


def load_prior_bounds() -> Dict[str, np.ndarray]:
    """Load per-condition prior bounds. Returns dict of (20, 2) arrays."""
    with open(PRIOR_JSON) as f:
        pj = json.load(f)

    result = {}
    for cond_key in COND_KEYS:
        condition, cultivation = COND_KEYS[cond_key]
        strat_key = f"{condition}_{cultivation}"
        if strat_key in pj["strategies"]:
            strat = pj["strategies"][strat_key]
            bounds = np.zeros((20, 2))
            for i in range(20):
                if str(i) in strat["bounds"]:
                    bounds[i] = strat["bounds"][str(i)]
                else:
                    bounds[i] = pj["default_bounds"]
            result[cond_key] = bounds
        else:
            result[cond_key] = np.tile(pj["default_bounds"], (20, 1))
    return result


def load_exp_data() -> Dict[str, np.ndarray]:
    """
    Load experimental species fractions.
    Returns dict: cond_key -> (6, 5) array of fractions (days x species).
    """
    df = pd.read_csv(EXP_CSV)
    result = {}
    for cond_key, (condition, cultivation) in COND_KEYS.items():
        sub = df[(df["condition"] == condition) & (df["cultivation"] == cultivation)]
        sp_map = SP_MAP_COMMENSAL if condition == "Commensal" else SP_MAP_DYSBIOTIC
        fracs = np.zeros((len(EXP_DAYS), 5))
        for _, row in sub.iterrows():
            day = int(row["day"])
            sp = row["species"]
            if day in EXP_DAYS and sp in sp_map:
                d_idx = np.where(EXP_DAYS == day)[0][0]
                fracs[d_idx, sp_map[sp]] = row["mean"] / 100.0
        # Renormalize each timepoint
        for d in range(len(EXP_DAYS)):
            s = fracs[d].sum()
            if s > 0:
                fracs[d] /= s
        result[cond_key] = fracs
    return result


def days_to_step_indices(days: np.ndarray, n_steps: int = 2500) -> np.ndarray:
    """Convert experimental days to simulation step indices.
    Maps [1..21] to [0..n_steps] with 95% coverage."""
    t_max_model = n_steps * 1e-4
    day_scale = (t_max_model * 0.95) / days.max()
    t_model = days * day_scale
    idx = np.round(t_model / 1e-4).astype(int)
    return np.clip(idx, 0, n_steps)


# ========================================================================
# Analysis 1: So Initial-Condition Sensitivity
# ========================================================================


def analysis_so_sensitivity():
    """Perturb Day 1 So fraction and show trajectory sensitivity."""
    print("=" * 70)
    print("ANALYSIS 1: So Initial-Condition Sensitivity")
    print("=" * 70)

    exp_data = load_exp_data()
    step_idx = days_to_step_indices(EXP_DAYS)
    model_days = np.linspace(0, 21, 2501)  # for plotting

    # Per-condition sigma for So (from experimental data)
    sigma_so = {
        "CS": 0.19,
        "CH": 0.19,
        "DS": 0.08,
        "DH": 0.08,
    }

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()

    for ax_idx, cond_key in enumerate(["CS", "CH", "DS", "DH"]):
        ax = axes[ax_idx]
        run_dir = find_latest_run(cond_key)
        theta_map = load_map_theta(run_dir)
        obs = exp_data[cond_key]
        ic_nominal = obs[0].copy()  # Day 1 fractions
        sig = sigma_so[cond_key]

        perturbations = {
            r"$-2\sigma$": -2 * sig,
            r"$-1\sigma$": -1 * sig,
            "nominal": 0.0,
            r"$+1\sigma$": +1 * sig,
            r"$+2\sigma$": +2 * sig,
        }

        line_styles = {
            r"$-2\sigma$": ("--", 0.5),
            r"$-1\sigma$": ("-.", 0.7),
            "nominal": ("-", 1.0),
            r"$+1\sigma$": ("-.", 0.7),
            r"$+2\sigma$": ("--", 0.5),
        }

        colors_pert = {
            r"$-2\sigma$": "#4575b4",
            r"$-1\sigma$": "#91bfdb",
            "nominal": "#000000",
            r"$+1\sigma$": "#fc8d59",
            r"$+2\sigma$": "#d73027",
        }

        for label, delta in perturbations.items():
            ic = ic_nominal.copy()
            ic[0] += delta  # perturb So
            ic = np.clip(ic, 1e-6, None)
            ic /= ic.sum()  # renormalize

            traj = simulate(theta_map, ic)
            ls, alpha = line_styles[label]
            ax.plot(
                model_days,
                traj[:, 0],
                ls=ls,
                alpha=alpha,
                color=colors_pert[label],
                label=label,
                lw=1.3,
            )

        # Overlay experimental So with errorbars
        so_std = np.array([obs[d, 0] * 0.15 for d in range(len(EXP_DAYS))])
        ax.errorbar(
            EXP_DAYS,
            obs[:, 0],
            yerr=so_std,
            fmt="ko",
            ms=4,
            capsize=3,
            capthick=0.8,
            lw=0.8,
            label="data",
        )

        ax.set_title(COND_LABELS[cond_key], fontweight="bold")
        ax.set_xlabel("Day")
        ax.set_ylabel(r"$\phi_{\mathrm{So}}$")
        ax.set_xlim(0, 22)
        ax.set_ylim(-0.02, 1.02)
        if ax_idx == 0:
            ax.legend(loc="upper right", framealpha=0.8, ncol=2, fontsize=8)

    fig.suptitle(
        r"$S.\ oralis$ trajectory sensitivity to Day-1 initial condition",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = FIG_DIR / "so_ic_sensitivity.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")

    # Print sensitivity summary
    print("\n  Sensitivity summary (So at Day 21):")
    print(f"  {'Condition':>8s}  {'nom':>6s}  {'-2sig':>6s}  {'+2sig':>6s}  {'range':>6s}")
    for cond_key in ["CS", "CH", "DS", "DH"]:
        run_dir = find_latest_run(cond_key)
        theta_map = load_map_theta(run_dir)
        obs = exp_data[cond_key]
        ic_nominal = obs[0].copy()
        sig = sigma_so[cond_key]

        vals = []
        for delta in [-2 * sig, 0, +2 * sig]:
            ic = ic_nominal.copy()
            ic[0] += delta
            ic = np.clip(ic, 1e-6, None)
            ic /= ic.sum()
            traj = simulate(theta_map, ic)
            vals.append(traj[-1, 0])
        print(
            f"  {cond_key:>8s}  {vals[1]:.4f}  {vals[0]:.4f}  {vals[2]:.4f}  {vals[2]-vals[0]:.4f}"
        )


# ========================================================================
# Analysis 2: Leave-One-Condition-Out Cross-Prediction
# ========================================================================


def compute_rmse_per_species(pred: np.ndarray, obs: np.ndarray) -> Tuple[float, np.ndarray]:
    """Compute overall RMSE and per-species RMSE.
    pred, obs: (n_time, 5) arrays."""
    resid = pred - obs
    rmse_total = np.sqrt(np.mean(resid**2))
    rmse_sp = np.sqrt(np.mean(resid**2, axis=0))
    return rmse_total, rmse_sp


def analysis_cross_prediction():
    """Cross-condition prediction: use MAP from one condition to predict another."""
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Leave-One-Condition-Out Cross-Prediction")
    print("=" * 70)

    exp_data = load_exp_data()
    step_idx = days_to_step_indices(EXP_DAYS)
    model_days = np.linspace(0, 21, 2501)

    # Load MAP theta for each condition
    theta_map = {}
    for ck in COND_KEYS:
        rd = find_latest_run(ck)
        theta_map[ck] = load_map_theta(rd)

    # Cross-prediction pairs:
    # Within same health state + cross-state
    pairs = [
        ("CS", "CH"),  # commensal: static -> HOBIC
        ("CH", "CS"),  # commensal: HOBIC -> static
        ("DS", "DH"),  # dysbiotic: static -> HOBIC
        ("DH", "DS"),  # dysbiotic: HOBIC -> static
    ]
    # Additional cross-state pairs for full matrix
    cross_state_pairs = [
        ("DS", "CS"),
        ("CS", "DS"),
        ("DH", "CH"),
        ("CH", "DH"),
        ("DS", "CH"),
        ("CS", "DH"),
        ("DH", "CS"),
        ("CH", "DS"),
    ]

    # Compute RMSE table
    rmse_table = {}
    for src, tgt in pairs:
        ic = exp_data[tgt][0].copy()  # target's own Day 1 IC
        traj = simulate(theta_map[src], ic)
        pred = traj[step_idx]
        obs = exp_data[tgt]
        rmse_tot, rmse_sp = compute_rmse_per_species(pred, obs)
        rmse_table[(src, tgt)] = (rmse_tot, rmse_sp)

    # Cross-state predictions
    for src, tgt in cross_state_pairs:
        ic = exp_data[tgt][0].copy()
        traj = simulate(theta_map[src], ic)
        pred = traj[step_idx]
        obs = exp_data[tgt]
        rmse_tot, rmse_sp = compute_rmse_per_species(pred, obs)
        rmse_table[(src, tgt)] = (rmse_tot, rmse_sp)

    # Self-prediction for reference
    for ck in COND_KEYS:
        ic = exp_data[ck][0].copy()
        traj = simulate(theta_map[ck], ic)
        pred = traj[step_idx]
        obs = exp_data[ck]
        rmse_tot, rmse_sp = compute_rmse_per_species(pred, obs)
        rmse_table[(ck, ck)] = (rmse_tot, rmse_sp)

    # ---- Figure: 2x2 grid ----
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    for ax_idx, (src, tgt) in enumerate(pairs):
        ax = axes[ax_idx // 2, ax_idx % 2]
        ic = exp_data[tgt][0].copy()
        obs = exp_data[tgt]

        # Self prediction
        traj_self = simulate(theta_map[tgt], ic)
        # Cross prediction
        traj_cross = simulate(theta_map[src], ic)

        for sp_i in range(5):
            # Experimental data
            ax.scatter(
                EXP_DAYS,
                obs[:, sp_i],
                marker="o",
                s=25,
                color=SPECIES_COLORS[sp_i],
                zorder=5,
                edgecolors="k",
                linewidths=0.3,
            )
            # Self-prediction (thin)
            ax.plot(
                model_days, traj_self[:, sp_i], "-", color=SPECIES_COLORS[sp_i], alpha=0.3, lw=0.8
            )
            # Cross-prediction (thick dashed)
            ax.plot(
                model_days,
                traj_cross[:, sp_i],
                "--",
                color=SPECIES_COLORS[sp_i],
                alpha=0.9,
                lw=1.5,
                label=SPECIES[sp_i] if ax_idx == 0 else None,
            )

        rmse_self = rmse_table[(tgt, tgt)][0]
        rmse_cross = rmse_table[(src, tgt)][0]
        ratio = rmse_cross / rmse_self if rmse_self > 0 else float("inf")

        ax.set_title(
            f"{COND_LABELS[src]} $\\to$ {COND_LABELS[tgt]}\n"
            f"RMSE: {rmse_cross:.3f} (self: {rmse_self:.3f}, ratio: {ratio:.1f}$\\times$)",
            fontsize=10,
        )
        ax.set_xlabel("Day")
        ax.set_ylabel("Species fraction")
        ax.set_xlim(0, 22)
        ax.set_ylim(-0.02, 1.02)

    # Legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.01),
        framealpha=0.9,
    )

    fig.suptitle(
        "Cross-condition prediction: MAP $\\theta$ transfer", fontsize=13, fontweight="bold", y=0.98
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_path = FIG_DIR / "cross_condition_prediction.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")

    # ---- LaTeX Table ----
    print("\n  LaTeX table fragment (RMSE cross-prediction matrix):")
    print(
        r"""
\begin{table}[htbp]
\centering
\caption{Cross-condition prediction RMSE. Rows: source $\theta_{\mathrm{MAP}}$; columns: target data.
  Diagonal = self-prediction (reference). Off-diagonal = transfer prediction.}
\label{tab:cross_prediction}
\begin{tabular}{lcccc}
\toprule
Source $\backslash$ Target & CS & CH & DS & DH \\
\midrule"""
    )

    for src in ["CS", "CH", "DS", "DH"]:
        row_vals = []
        for tgt in ["CS", "CH", "DS", "DH"]:
            if (src, tgt) in rmse_table:
                v = rmse_table[(src, tgt)][0]
                if src == tgt:
                    row_vals.append(f"\\textbf{{{v:.3f}}}")
                elif src[0] != tgt[0]:  # cross-state
                    row_vals.append(f"\\textit{{{v:.3f}}}")
                else:
                    row_vals.append(f"{v:.3f}")
            else:
                row_vals.append("---")
        print(f"{src} & {' & '.join(row_vals)} \\\\")

    print(
        r"""\bottomrule
\end{tabular}
\end{table}
"""
    )

    # Per-species RMSE detail
    print("  Per-species RMSE (within-state transfers):")
    print(f"  {'Pair':>10s}  {'RMSE':>6s}  " + "  ".join(f"{s:>5s}" for s in SPECIES))
    for src, tgt in pairs:
        rmse_tot, rmse_sp = rmse_table[(src, tgt)]
        sp_str = "  ".join(f"{v:.3f}" for v in rmse_sp)
        print(f"  {src}>{tgt:>2s}  {rmse_tot:.4f}  {sp_str}")

    print("\n  Per-species RMSE (cross-state transfers):")
    print(f"  {'Pair':>10s}  {'RMSE':>6s}  " + "  ".join(f"{s:>5s}" for s in SPECIES))
    for src, tgt in cross_state_pairs:
        rmse_tot, rmse_sp = rmse_table[(src, tgt)]
        sp_str = "  ".join(f"{v:.3f}" for v in rmse_sp)
        print(f"  {src}>{tgt:>2s}  {rmse_tot:.4f}  {sp_str}")


# ========================================================================
# Analysis 3: Effective Dimensionality (Posterior Informativeness)
# ========================================================================


def analysis_effective_dimensionality():
    """Compute posterior_std / prior_width ratio per parameter per condition."""
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Effective Dimensionality (Posterior Informativeness)")
    print("=" * 70)

    prior_bounds = load_prior_bounds()

    # Compute ratios using BOTH condition-specific priors and a common wide prior
    # The wide uniform prior [-2, 4] (width=6) provides a condition-independent baseline
    UNIFORM_WIDTH = 6.0  # common baseline: [-2, 4] for all params

    ratios = {}  # cond -> (20,) array of std/width  (condition-specific prior)
    ratios_uni = {}  # cond -> (20,) array of std/UNIFORM_WIDTH
    n_identified = {}
    n_identified_uni = {}
    threshold = 0.80

    for cond_key in ["CS", "CH", "DS", "DH"]:
        run_dir = find_latest_run(cond_key)
        samples = load_samples(run_dir)
        bounds = prior_bounds[cond_key]

        post_std = samples.std(axis=0)
        prior_width = bounds[:, 1] - bounds[:, 0]
        safe_width = np.where(prior_width > 1e-10, prior_width, 1.0)
        ratio = post_std / safe_width
        ratio_uni = post_std / UNIFORM_WIDTH
        ratios[cond_key] = ratio
        ratios_uni[cond_key] = ratio_uni

        identified = ratio < threshold
        identified_uni = ratio_uni < threshold
        n_identified[cond_key] = identified.sum()
        n_identified_uni[cond_key] = identified_uni.sum()
        print(
            f"  {cond_key}: {n_identified[cond_key]}/20 (cond-specific prior), "
            f"{n_identified_uni[cond_key]}/20 (uniform [-2,4] prior) "
            f"well-identified (ratio < {threshold})"
        )
        print(
            f"         post_std range: [{post_std.min():.3f}, {post_std.max():.3f}], "
            f"prior_width range: [{prior_width.min():.3f}, {prior_width.max():.3f}]"
        )

    # ---- Figure: Two heatmaps (condition-specific prior + uniform prior) + bar chart ----
    fig, axes = plt.subplots(
        1, 3, figsize=(14, 8), gridspec_kw={"width_ratios": [2.5, 2.5, 1], "wspace": 0.3}
    )
    cond_order = ["CS", "CH", "DS", "DH"]

    for panel_idx, (ratios_dict, title_str, label_str) in enumerate(
        [
            (
                ratios,
                "Condition-specific prior",
                r"$\sigma_{\mathrm{post}} / \Delta_{\mathrm{cond}}$",
            ),
            (
                ratios_uni,
                "Uniform prior ($\\Delta=6$)",
                r"$\sigma_{\mathrm{post}} / \Delta_{\mathrm{unif}}$",
            ),
        ]
    ):
        ax = axes[panel_idx]
        mat = np.array([ratios_dict[ck] for ck in cond_order])  # (4, 20)

        im = ax.imshow(
            mat.T, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1.0, interpolation="nearest"
        )
        ax.set_xticks(range(4))
        ax.set_xticklabels(cond_order, fontsize=11)
        ax.set_yticks(range(20))
        ax.set_yticklabels(PARAM_NAMES if panel_idx == 0 else [""] * 20, fontsize=9)
        ax.set_xlabel("Condition")
        if panel_idx == 0:
            ax.set_ylabel("Parameter")

        for i in range(4):
            for j in range(20):
                v = mat[i, j]
                color = "white" if v > 0.6 else "black"
                ax.text(i, j, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color=color)

        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label(label_str, fontsize=9)
        ax.set_title(title_str, fontsize=10)

    # Right: Bar chart of effective dimensionality (both metrics)
    ax_bar = axes[2]
    x_bar = np.arange(4)
    width = 0.35
    n_eff_cond = [n_identified[ck] for ck in cond_order]
    n_eff_uni = [n_identified_uni[ck] for ck in cond_order]
    colors_bar = ["#2166ac", "#67a9cf", "#ef8a62", "#b2182b"]

    bars1 = ax_bar.barh(
        x_bar - width / 2,
        n_eff_cond,
        height=width,
        color=[c + "80" for c in colors_bar],
        edgecolor="k",
        lw=0.5,
        label="cond-specific",
    )
    bars2 = ax_bar.barh(
        x_bar + width / 2,
        n_eff_uni,
        height=width,
        color=colors_bar,
        edgecolor="k",
        lw=0.5,
        label="uniform",
    )
    ax_bar.set_yticks(x_bar)
    ax_bar.set_yticklabels(cond_order, fontsize=11)
    ax_bar.set_xlabel(f"$N_{{\\mathrm{{eff}}}}$ (ratio < {threshold})", fontsize=10)
    ax_bar.set_title("Effective\ndimensionality", fontsize=10)
    ax_bar.set_xlim(0, 21)
    ax_bar.invert_yaxis()
    ax_bar.legend(fontsize=8, loc="lower right")

    for bar, n in zip(bars1, n_eff_cond):
        ax_bar.text(
            bar.get_width() + 0.2,
            bar.get_y() + bar.get_height() / 2,
            str(n),
            va="center",
            fontsize=9,
        )
    for bar, n in zip(bars2, n_eff_uni):
        ax_bar.text(
            bar.get_width() + 0.2,
            bar.get_y() + bar.get_height() / 2,
            str(n),
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    fig.suptitle(
        "Parameter identifiability: posterior contraction relative to prior",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    fig.subplots_adjust(top=0.93, bottom=0.06, left=0.08, right=0.97)

    out_path = FIG_DIR / "effective_dimensionality.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")

    # ---- LaTeX Table (uniform prior baseline, more discriminative) ----
    print("\n  LaTeX table fragment (effective dimensionality, uniform prior baseline):")
    print(
        r"""
\begin{table}[htbp]
\centering
\caption{Effective dimensionality per condition. $r_i = \sigma_{\mathrm{post},i} / \Delta_{\mathrm{unif}}$
  with $\Delta_{\mathrm{unif}}=6$ (common baseline).
  A parameter is ``well-identified'' if $r_i < 0.80$.}
\label{tab:effective_dim}
\begin{tabular}{lcccc}
\toprule
Parameter & CS & CH & DS & DH \\
\midrule"""
    )

    for j in range(20):
        vals = []
        for ck in cond_order:
            r = ratios_uni[ck][j]
            if r < threshold:
                vals.append(f"\\cellcolor{{green!20}}{r:.2f}")
            else:
                vals.append(f"\\cellcolor{{red!15}}{r:.2f}")
        print(f"${PARAM_NAMES_LATEX[j]}$ & {' & '.join(vals)} \\\\")

    print(
        r"""\midrule
$N_{\mathrm{eff}}$""",
        end="",
    )
    for ck in cond_order:
        print(f" & \\textbf{{{n_identified_uni[ck]}}}", end="")
    print(
        r""" \\
\bottomrule
\end{tabular}
\end{table}
"""
    )

    # Print which parameters are NOT identified (using uniform prior)
    always_prior_uni = []
    for j in range(20):
        if all(ratios_uni[ck][j] >= threshold for ck in cond_order):
            always_prior_uni.append(PARAM_NAMES_LATEX[j])
    if always_prior_uni:
        print(
            f"  Parameters prior-dominated in ALL conditions (uniform baseline): "
            f"{', '.join(always_prior_uni)}"
        )
    else:
        print("  No parameter is prior-dominated in all conditions (uniform baseline).")

    never_prior_uni = []
    for j in range(20):
        if all(ratios_uni[ck][j] < threshold for ck in cond_order):
            never_prior_uni.append(PARAM_NAMES_LATEX[j])
    if never_prior_uni:
        print(
            f"  Parameters well-identified in ALL conditions (uniform baseline): "
            f"{', '.join(never_prior_uni)}"
        )

    # Posterior standard deviation summary
    print("\n  Posterior std summary:")
    print(f"  {'Cond':>4s}  {'min_std':>8s}  {'max_std':>8s}  {'mean_std':>8s}")
    for ck in cond_order:
        run_dir = find_latest_run(ck)
        samples = load_samples(run_dir)
        post_std = samples.std(axis=0)
        print(f"  {ck:>4s}  {post_std.min():8.4f}  {post_std.max():8.4f}  {post_std.mean():8.4f}")


# ========================================================================
# Main
# ========================================================================


def main():
    print("Paper Additional Analyses")
    print(f"Output directory: {FIG_DIR}")
    print(f"Run directories: {RUNS}")
    print()

    # Show which run directories are being used
    for cond_key in ["CS", "CH", "DS", "DH"]:
        rd = find_latest_run(cond_key)
        cfg = load_config(rd)
        n = cfg.get("n_particles", "?")
        rmse = cfg.get("rmse", float("nan"))
        print(f"  {cond_key}: {rd.name}  (N={n}, RMSE={rmse:.4f})")
    print()

    analysis_so_sensitivity()
    analysis_cross_prediction()
    analysis_effective_dimensionality()

    print("\n" + "=" * 70)
    print("All analyses complete.")
    print(f"Figures saved in: {FIG_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
