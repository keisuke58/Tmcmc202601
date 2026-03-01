#!/usr/bin/env python3
"""
plot_cross_condition_ci.py
==========================
Reads summary.json from each condition's CI run and generates
a publication-quality cross-condition comparison figure.
"""

import json
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_OUT_BASE = _HERE / "_uncertainty_propagation"


def load_summaries():
    """Load all available condition summaries."""
    results = {}
    for cond_dir in sorted(_OUT_BASE.iterdir()):
        if not cond_dir.is_dir():
            continue
        summary_path = cond_dir / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                results[cond_dir.name] = json.load(f)
    return results


def load_sensitivity(cond):
    """Load sensitivity_indices.json for a condition."""
    path = _OUT_BASE / cond / "aggregated" / "sensitivity_indices.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def plot_comparison(results):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conds = list(results.keys())
    n = len(conds)
    if n < 2:
        print(f"Need >= 2 conditions, found {n}")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    colors = {
        "dh_baseline": "#1f77b4",
        "baseline_original_bounds": "#d62728",
        "commensal_static": "#2ca02c",
        "commensal_hobic": "#ff7f0e",
        "dysbiotic_static": "#9467bd",
    }
    x = np.arange(n)
    bar_colors = [colors.get(c, "#333333") for c in conds]

    labels = []
    for c in conds:
        lab = c.replace("_", "\n")
        if c == "dh_baseline":
            lab = "Mild-weight\n(narrow bounds)"
        elif c == "baseline_original_bounds":
            lab = "Baseline\n(original bounds)"
        labels.append(lab)

    # --- Row 1: bar charts with CI ---

    # (a) DI mean + CI
    ax = axes[0, 0]
    means = [results[c]["di_mean_global"] for c in conds]
    widths = [results[c]["di_ci_width"] for c in conds]
    ax.bar(x, means, color=bar_colors, alpha=0.8, edgecolor="k", linewidth=0.5)
    ax.errorbar(
        x,
        means,
        yerr=[w / 2 for w in widths],
        fmt="none",
        ecolor="k",
        capsize=5,
        capthick=1.5,
        linewidth=1.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("DI (global mean)", fontsize=11)
    ax.set_title("(a) Dysbiosis Index", fontsize=12, weight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # (b) E_phi_pg + CI
    ax = axes[0, 1]
    means = [results[c]["epg_mean_pa"] for c in conds]
    widths = [results[c]["epg_ci_width_pa"] for c in conds]
    ax.bar(x, means, color=bar_colors, alpha=0.8, edgecolor="k", linewidth=0.5)
    ax.errorbar(
        x,
        means,
        yerr=[w / 2 for w in widths],
        fmt="none",
        ecolor="k",
        capsize=5,
        capthick=1.5,
        linewidth=1.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("$E_{\\phi_{Pg}}$ [Pa]", fontsize=11)
    ax.set_title("(b) Young's Modulus (Pg Hill)", fontsize=12, weight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # (c) E_virulence + CI
    ax = axes[0, 2]
    means = [results[c]["evir_mean_pa"] for c in conds]
    widths = [results[c]["evir_ci_width_pa"] for c in conds]
    ax.bar(x, means, color=bar_colors, alpha=0.8, edgecolor="k", linewidth=0.5)
    ax.errorbar(
        x,
        means,
        yerr=[w / 2 for w in widths],
        fmt="none",
        ecolor="k",
        capsize=5,
        capthick=1.5,
        linewidth=1.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("$E_{vir}$ [Pa]", fontsize=11)
    ax.set_title("(c) Young's Modulus (Pg+Fn)", fontsize=12, weight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # --- Row 2: sensitivity comparison ---

    # (d) Top-5 sensitivity comparison
    ax = axes[1, 0]
    all_sens = {}
    for c in conds:
        s = load_sensitivity(c)
        if s:
            all_sens[c] = s
    if all_sens:
        # Get union of top-5 params from all conditions
        all_params = set()
        for c, s in all_sens.items():
            ranking = s.get("_ranking", [])
            all_params.update(ranking[:5])
        params = sorted(all_params)
        xp = np.arange(len(params))
        width = 0.35
        for i, c in enumerate(conds):
            s = all_sens.get(c, {})
            vals = [abs(s.get(p, {}).get("spearman_di_mean", 0)) for p in params]
            offset = (i - (n - 1) / 2) * width
            ax.bar(
                xp + offset,
                vals,
                width * 0.9,
                color=colors.get(c, "#333"),
                alpha=0.8,
                label=c.replace("_", " "),
                edgecolor="k",
                linewidth=0.3,
            )
        ax.set_xticks(xp)
        ax.set_xticklabels(params, fontsize=9, rotation=30)
        ax.set_ylabel("|Spearman $\\rho$|", fontsize=11)
        ax.set_title("(d) Sensitivity |$\\rho$(param, DI)|", fontsize=12, weight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
    else:
        ax.text(
            0.5,
            0.5,
            "No sensitivity data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )

    # (e) DI CI width comparison
    ax = axes[1, 1]
    widths = [results[c]["di_ci_width"] for c in conds]
    ax.bar(x, widths, color=bar_colors, alpha=0.8, edgecolor="k", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("90% CI width", fontsize=11)
    ax.set_title("(e) DI Uncertainty Width", fontsize=12, weight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # (f) Summary table
    ax = axes[1, 2]
    ax.axis("off")
    cell_text = []
    col_labels = ["Condition", "DI mean", "DI CI", "E_Pg [Pa]", "n_samples"]
    for c in conds:
        r = results[c]
        cell_text.append(
            [
                c.replace("_", " "),
                f"{r['di_mean_global']:.4f}",
                f"{r['di_ci_width']:.4f}",
                f"{r['epg_mean_pa']:.1f}",
                str(r["n_samples"]),
            ]
        )
    table = ax.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)
    ax.set_title("(f) Summary", fontsize=12, weight="bold", pad=20)

    fig.suptitle(
        "Cross-Condition Posterior Uncertainty Comparison (50 samples, 90% CI)",
        fontsize=14,
        weight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out = _OUT_BASE / "cross_condition_ci_comparison.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved: {out}")

    # Also save master summary
    master = _OUT_BASE / "master_summary.json"
    with open(master, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {master}")


if __name__ == "__main__":
    results = load_summaries()
    # Remove duplicate (baseline_original_bounds == dh_baseline)
    results.pop("baseline_original_bounds", None)
    print(f"Found {len(results)} conditions: {list(results.keys())}")
    for c, r in results.items():
        print(
            f"  {c}: DI={r['di_mean_global']:.6f} (CI={r['di_ci_width']:.6f}), "
            f"E_Pg={r['epg_mean_pa']:.1f} Pa, n={r['n_samples']}"
        )
    if len(results) >= 2:
        plot_comparison(results)
    else:
        print("Need at least 2 conditions for comparison")
