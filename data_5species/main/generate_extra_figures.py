#!/usr/bin/env python3
"""
Generate additional publication-quality figures for completed TMCMC runs.
Reads samples.npy, logL.npy, theta_MAP.json, theta_mean.json, data.npy, config.json
and produces:
  1. Interaction matrix heatmap (A matrix from MAP)
  2. Violin plots of posterior distributions
  3. Pairwise correlation heatmap of posterior samples
  4. MAP vs Mean parameter comparison
  5. Log-likelihood distribution with MAP/Mean marked
  6. Species composition stacked bar (data timepoints)
  7. Parameter sensitivity (logL correlation)

Usage:
    python generate_extra_figures.py --run_dir <RUN_DIR>
    python generate_extra_figures.py --run_dir DIR1 --run_dir DIR2
"""

import argparse
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── Parameter labels (20 params, Nishioka 5-species model) ──
PARAM_LABELS = [
    r"$a_{11}$", r"$a_{12}$", r"$a_{22}$", r"$b_1$", r"$b_2$",
    r"$a_{33}$", r"$a_{34}$", r"$a_{44}$", r"$b_3$", r"$b_4$",
    r"$a_{13}$", r"$a_{14}$", r"$a_{23}$", r"$a_{24}$",
    r"$a_{55}$", r"$b_5$",
    r"$a_{15}$", r"$a_{25}$", r"$a_{35}$", r"$a_{45}$",
]

PARAM_LABELS_PLAIN = [
    "a11", "a12", "a22", "b1", "b2",
    "a33", "a34", "a44", "b3", "b4",
    "a13", "a14", "a23", "a24",
    "a55", "b5",
    "a15", "a25", "a35", "a45",
]

SPECIES_NAMES = ["S. oralis", "A. naeslundii", "V. dispar", "F. nucleatum", "P. gingivalis"]
SPECIES_SHORT = ["S.o", "A.n", "V.d", "F.n", "P.g"]

# Species colors
COLORS_COMMENSAL = ["#1f77b4", "#2ca02c", "#bcbd22", "#9467bd", "#d62728"]
COLORS_DYSBIOTIC = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#d62728"]


def theta_to_A_matrix(theta):
    """Convert 20-parameter theta to 5x5 interaction matrix A and growth vector b."""
    A = np.zeros((5, 5))
    b = np.zeros(5)

    # M1: S0-S1 block
    A[0, 0] = theta[0]   # a11
    A[0, 1] = theta[1]   # a12
    A[1, 0] = theta[1]   # a21 = a12 (symmetric)
    A[1, 1] = theta[2]   # a22
    b[0]    = theta[3]    # b1
    b[1]    = theta[4]    # b2

    # M2: S2-S3 block
    A[2, 2] = theta[5]   # a33
    A[2, 3] = theta[6]   # a34
    A[3, 2] = theta[6]   # a43 = a34
    A[3, 3] = theta[7]   # a44
    b[2]    = theta[8]    # b3
    b[3]    = theta[9]    # b4

    # M3: Cross interactions S0/S1 - S2/S3
    A[0, 2] = theta[10]  # a13
    A[2, 0] = theta[10]
    A[0, 3] = theta[11]  # a14
    A[3, 0] = theta[11]
    A[1, 2] = theta[12]  # a23
    A[2, 1] = theta[12]
    A[1, 3] = theta[13]  # a24
    A[3, 1] = theta[13]

    # M4: S4 self
    A[4, 4] = theta[14]  # a55
    b[4]    = theta[15]   # b5

    # M5: S4 cross
    A[0, 4] = theta[16]  # a15
    A[4, 0] = theta[16]
    A[1, 4] = theta[17]  # a25
    A[4, 1] = theta[17]
    A[2, 4] = theta[18]  # a35
    A[4, 2] = theta[18]
    A[3, 4] = theta[19]  # a45
    A[4, 3] = theta[19]

    return A, b


def load_run(run_dir):
    """Load all data from a run directory."""
    run_dir = Path(run_dir)
    data = {}

    with open(run_dir / "config.json") as f:
        data["config"] = json.load(f)

    with open(run_dir / "theta_MAP.json") as f:
        data["theta_MAP"] = np.array(json.load(f)["theta_full"])

    with open(run_dir / "theta_mean.json") as f:
        tm = json.load(f)
        data["theta_mean"] = np.array(tm.get("theta_full", tm.get("theta_sub")))

    data["samples"] = np.load(run_dir / "samples.npy")
    data["logL"] = np.load(run_dir / "logL.npy")
    data["exp_data"] = np.load(run_dir / "data.npy")
    data["idx_sparse"] = np.load(run_dir / "idx_sparse.npy")

    with open(run_dir / "fit_metrics.json") as f:
        data["fit_metrics"] = json.load(f)

    cond = data["config"].get("condition", "")
    cult = data["config"].get("cultivation", "")
    data["label"] = f"{cond}_{cult}"
    data["condition"] = cond
    data["colors"] = COLORS_DYSBIOTIC if "Dysbiotic" in cond else COLORS_COMMENSAL
    data["days"] = np.array(data["config"]["metadata"]["days"])

    return data


def save_plot_with_preview(fig, output_path):
    """Save plot in high res (300dpi) and preview (100dpi) formats."""
    # High resolution
    fig.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix('.pdf'), dpi=300, bbox_inches="tight")
    
    # Preview (max 2000px width)
    # 100 dpi is usually safe for typical figure sizes (up to 20 inches)
    fig.savefig(output_path.with_name(output_path.stem + "_preview.png"), dpi=100, bbox_inches="tight")
    print(f"  Saved: {output_path.name} (+preview)")

# ═══════════════════════════════════════════════════════════════════
# Figure 1: Interaction Matrix Heatmap
# ═══════════════════════════════════════════════════════════════════
def fig_interaction_heatmap(data, out_dir):
    """5x5 interaction matrix heatmap from MAP estimate."""
    A, b = theta_to_A_matrix(data["theta_MAP"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5),
                                    gridspec_kw={"width_ratios": [5, 1.2]})

    # Heatmap
    vmax = np.max(np.abs(A)) * 1.05
    im = ax1.imshow(A, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")

    for i in range(5):
        for j in range(5):
            val = A[i, j]
            color = "white" if abs(val) > vmax * 0.6 else "black"
            ax1.text(j, i, f"{val:.2f}", ha="center", va="center",
                     fontsize=11, fontweight="bold", color=color)

    ax1.set_xticks(range(5))
    ax1.set_yticks(range(5))
    ax1.set_xticklabels(SPECIES_SHORT, fontsize=12)
    ax1.set_yticklabels(SPECIES_NAMES, fontsize=11)
    ax1.set_xlabel("Effect of species (column) on ...", fontsize=12)
    ax1.set_ylabel("... species (row)", fontsize=12)
    ax1.set_title(f"Interaction Matrix A — {data['label']}", fontsize=14, fontweight="bold")

    cb = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cb.set_label("Interaction strength", fontsize=11)

    # Growth rates bar
    colors_bar = data["colors"]
    bars = ax2.barh(range(5), b, color=colors_bar, edgecolor="black", height=0.6)
    ax2.set_yticks(range(5))
    ax2.set_yticklabels(SPECIES_SHORT, fontsize=11)
    ax2.set_xlabel("Growth rate $b_i$", fontsize=11)
    ax2.set_title("Growth", fontsize=12, fontweight="bold")
    ax2.invert_yaxis()
    for i, v in enumerate(b):
        ax2.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=10)
    ax2.axvline(0, color="gray", lw=0.5)
    ax2.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    save_plot_with_preview(fig, out_dir / f"pub_interaction_heatmap_{data['label']}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════
# Figure 2: Violin Plots
# ═══════════════════════════════════════════════════════════════════
def fig_violin_plots(data, out_dir):
    """Violin plots for all 20 posterior parameters with MAP/Mean markers."""
    samples = data["samples"]
    theta_map = data["theta_MAP"]
    theta_mean = data["theta_mean"]
    n_params = samples.shape[1]

    fig, ax = plt.subplots(figsize=(16, 6))

    parts = ax.violinplot(samples, positions=range(n_params), showmedians=True,
                          showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor("#4C72B0")
        pc.set_alpha(0.6)
    parts["cmedians"].set_color("black")

    # MAP and Mean markers
    ax.scatter(range(n_params), theta_map, color="red", marker="D", s=50,
               zorder=5, label="MAP", edgecolors="black", linewidths=0.5)
    ax.scatter(range(n_params), theta_mean, color="limegreen", marker="s", s=40,
               zorder=5, label="Mean", edgecolors="black", linewidths=0.5)

    # Group shading
    groups = [(0, 5, "#E8F0FE", "M1\n(S.o–A.n)"),
              (5, 10, "#FEF3E8", "M2\n(V.d–F.n)"),
              (10, 14, "#E8FEE8", "M3\n(Cross)"),
              (14, 16, "#FEE8FE", "M4\n(P.g self)"),
              (16, 20, "#FEE8E8", "M5\n(P.g cross)")]
    for start, end, color, label in groups:
        ax.axvspan(start - 0.5, end - 0.5, alpha=0.3, color=color)
        ax.text((start + end) / 2 - 0.5, ax.get_ylim()[0] if ax.get_ylim()[0] < 0 else -0.1,
                label, ha="center", va="top", fontsize=8, fontstyle="italic")

    ax.set_xticks(range(n_params))
    ax.set_xticklabels(PARAM_LABELS, fontsize=10)
    ax.set_ylabel("Parameter Value", fontsize=12)
    ax.set_title(f"Posterior Distributions — {data['label']}", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_plot_with_preview(fig, out_dir / f"pub_violin_posterior_{data['label']}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════
# Figure 3: Posterior Correlation Heatmap
# ═══════════════════════════════════════════════════════════════════
def fig_correlation_heatmap(data, out_dir):
    """Correlation matrix of posterior samples."""
    samples = data["samples"]
    corr = np.corrcoef(samples.T)
    n = corr.shape[0]

    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")

    # Annotate strong correlations only
    for i in range(n):
        for j in range(n):
            if abs(corr[i, j]) > 0.3 and i != j:
                ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center",
                        fontsize=7, color="black" if abs(corr[i, j]) < 0.7 else "white")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(PARAM_LABELS, fontsize=9, rotation=45, ha="right")
    ax.set_yticklabels(PARAM_LABELS, fontsize=9)
    ax.set_title(f"Posterior Parameter Correlations — {data['label']}", fontsize=14, fontweight="bold")

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Pearson correlation", fontsize=11)

    plt.tight_layout()
    plt.savefig(out_dir / f"pub_correlation_{data['label']}.png", dpi=300, bbox_inches="tight")
    plt.savefig(out_dir / f"pub_correlation_{data['label']}.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [3] Correlation heatmap saved")


# ═══════════════════════════════════════════════════════════════════
# Figure 4: MAP vs Mean Comparison
# ═══════════════════════════════════════════════════════════════════
def fig_map_vs_mean(data, out_dir):
    """Side-by-side bar chart comparing MAP and Mean estimates."""
    theta_map = data["theta_MAP"]
    theta_mean = data["theta_mean"]
    n = len(theta_map)

    x = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=(16, 5))
    bars1 = ax.bar(x - width/2, theta_map, width, label="MAP", color="#d62728", alpha=0.8, edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width/2, theta_mean, width, label="Mean", color="#2ca02c", alpha=0.8, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(PARAM_LABELS, fontsize=10)
    ax.set_ylabel("Parameter Value", fontsize=12)
    ax.set_title(f"MAP vs Posterior Mean — {data['label']}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, color="gray", lw=0.5)

    plt.tight_layout()
    save_plot_with_preview(fig, out_dir / f"pub_MAP_vs_Mean_{data['label']}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════
# Figure 5: Log-Likelihood Distribution
# ═══════════════════════════════════════════════════════════════════
def fig_logL_distribution(data, out_dir):
    """Histogram of log-likelihood values with MAP/Mean marked."""
    logL = data["logL"]
    theta_map = data["theta_MAP"]
    theta_mean = data["theta_mean"]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(logL, bins=60, color="#4C72B0", alpha=0.7, edgecolor="black", linewidth=0.3, density=True)

    # Mark MAP logL (highest logL sample)
    map_idx = np.argmax(logL)
    ax.axvline(logL[map_idx], color="red", lw=2, ls="--", label=f"MAP (logL={logL[map_idx]:.2f})")

    # Percentiles
    p5 = np.percentile(logL, 5)
    p50 = np.percentile(logL, 50)
    p95 = np.percentile(logL, 95)
    ax.axvline(p50, color="orange", lw=1.5, ls=":", label=f"Median (logL={p50:.2f})")
    ax.axvspan(p5, p95, alpha=0.1, color="green", label=f"90% CI [{p5:.1f}, {p95:.1f}]")

    ax.set_xlabel("Log-Likelihood", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Posterior Log-Likelihood — {data['label']}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / f"pub_logL_distribution_{data['label']}.png", dpi=300, bbox_inches="tight")
    plt.savefig(out_dir / f"pub_logL_distribution_{data['label']}.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [5] Log-likelihood distribution saved")


# ═══════════════════════════════════════════════════════════════════
# Figure 6: Species Composition Stacked Bar (Data)
# ═══════════════════════════════════════════════════════════════════
def fig_species_composition(data, out_dir):
    """Stacked bar chart of species composition at each timepoint."""
    exp_data = data["exp_data"]  # shape (n_time, 5)
    days = data["days"]
    colors = data["colors"]

    # Normalize to fractions
    totals = exp_data.sum(axis=1, keepdims=True)
    totals = np.where(totals == 0, 1, totals)
    fracs = exp_data / totals

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Absolute volumes
    bottom = np.zeros(len(days))
    for i in range(5):
        ax1.bar([f"Day {d}" for d in days], exp_data[:, i], bottom=bottom,
                color=colors[i], edgecolor="white", linewidth=0.5, label=SPECIES_NAMES[i])
        bottom += exp_data[:, i]

    ax1.set_ylabel("Absolute Volume (φ̄)", fontsize=11)
    ax1.set_title(f"Species Volumes — {data['label']}", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax1.grid(axis="y", alpha=0.3)

    # Right: Relative composition
    bottom = np.zeros(len(days))
    for i in range(5):
        ax2.bar([f"Day {d}" for d in days], fracs[:, i], bottom=bottom,
                color=colors[i], edgecolor="white", linewidth=0.5, label=SPECIES_NAMES[i])
        bottom += fracs[:, i]

    ax2.set_ylabel("Relative Fraction", fontsize=11)
    ax2.set_title(f"Species Composition — {data['label']}", fontsize=13, fontweight="bold")
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_plot_with_preview(fig, out_dir / f"pub_species_composition_{data['label']}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════
# Figure 7: Parameter Sensitivity (logL-parameter correlation)
# ═══════════════════════════════════════════════════════════════════
def fig_parameter_sensitivity(data, out_dir):
    """Bar chart of correlation between each parameter and log-likelihood."""
    samples = data["samples"]
    logL = data["logL"]
    n_params = samples.shape[1]

    correlations = np.array([np.corrcoef(samples[:, i], logL)[0, 1] for i in range(n_params)])

    fig, ax = plt.subplots(figsize=(14, 5))

    colors_bar = ["#d62728" if c > 0 else "#2ca02c" for c in correlations]
    bars = ax.bar(range(n_params), correlations, color=colors_bar, edgecolor="black", linewidth=0.5, alpha=0.8)

    ax.set_xticks(range(n_params))
    ax.set_xticklabels(PARAM_LABELS, fontsize=10)
    ax.set_ylabel("Correlation with logL", fontsize=12)
    ax.set_title(f"Parameter Sensitivity (Corr with Log-Likelihood) — {data['label']}", fontsize=13, fontweight="bold")
    ax.axhline(0, color="gray", lw=0.5)
    ax.axhline(0.1, color="gray", lw=0.5, ls=":")
    ax.axhline(-0.1, color="gray", lw=0.5, ls=":")
    ax.grid(axis="y", alpha=0.3)

    # Annotate strongest
    top_idx = np.argsort(np.abs(correlations))[-3:]
    for idx in top_idx:
        ax.annotate(f"{correlations[idx]:.3f}",
                    (idx, correlations[idx]),
                    textcoords="offset points", xytext=(0, 8 if correlations[idx] > 0 else -12),
                    ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    save_plot_with_preview(fig, out_dir / f"pub_sensitivity_{data['label']}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════
# Figure 8: Fit Metrics Summary Table
# ═══════════════════════════════════════════════════════════════════
def fig_fit_metrics_table(data, out_dir):
    """Visual table of fit metrics per species."""
    fm = data["fit_metrics"]
    map_m = fm["MAP"]
    mean_m = fm["Mean"]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    col_labels = ["Species", "RMSE (MAP)", "MAE (MAP)", "RMSE (Mean)", "MAE (Mean)"]
    table_data = []
    for i, name in enumerate(SPECIES_NAMES):
        table_data.append([
            name,
            f"{map_m['rmse_per_species'][i]:.4f}",
            f"{map_m['mae_per_species'][i]:.4f}",
            f"{mean_m['rmse_per_species'][i]:.4f}",
            f"{mean_m['mae_per_species'][i]:.4f}",
        ])
    table_data.append([
        "TOTAL",
        f"{map_m['rmse_total']:.4f}",
        f"{map_m['mae_total']:.4f}",
        f"{mean_m['rmse_total']:.4f}",
        f"{mean_m['mae_total']:.4f}",
    ])

    table = ax.table(cellText=table_data, colLabels=col_labels, loc="center",
                     cellLoc="center", colColours=["#D6EAF8"] * 5)
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.6)

    # Color the total row
    for j in range(5):
        table[len(table_data), j].set_facecolor("#FADBD8")
        table[len(table_data), j].set_text_props(fontweight="bold")

    ax.set_title(f"Fit Metrics Summary — {data['label']}", fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    save_plot_with_preview(fig, out_dir / f"pub_fit_metrics_table_{data['label']}")
    plt.close()
    print(f"  [8] Fit metrics table saved")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
def process_run(run_dir):
    run_dir = Path(run_dir)
    print(f"\n{'='*60}")
    print(f"Processing: {run_dir.name}")
    print(f"{'='*60}")

    data = load_run(run_dir)
    out_dir = run_dir / "figures"
    out_dir.mkdir(exist_ok=True)

    fig_interaction_heatmap(data, out_dir)
    fig_violin_plots(data, out_dir)
    fig_correlation_heatmap(data, out_dir)
    fig_map_vs_mean(data, out_dir)
    fig_logL_distribution(data, out_dir)
    fig_species_composition(data, out_dir)
    fig_parameter_sensitivity(data, out_dir)
    fig_fit_metrics_table(data, out_dir)

    print(f"\n  All 8 figures saved to: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate extra publication figures")
    parser.add_argument("--run_dir", type=str, action="append", required=True,
                        help="Run directory (can specify multiple)")
    args = parser.parse_args()

    for rd in args.run_dir:
        process_run(rd)

    print("\nDone!")


if __name__ == "__main__":
    main()
