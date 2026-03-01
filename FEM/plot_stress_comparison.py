#!/usr/bin/env python3
"""
plot_stress_comparison.py
==========================
A3: 3D Abaqus stress comparison across 4 biofilm conditions.

Reads _stress.json from _abaqus_auto_jobs/{cond}_T23_v2/ directories,
extracts E_biofilm from INP files, and generates comparison figures.

Usage
-----
  python plot_stress_comparison.py
  python plot_stress_comparison.py --suffix v2
  python plot_stress_comparison.py --conditions commensal_static dh_baseline
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_FIG_DIR = _HERE / "figures"
_FIG_DIR.mkdir(exist_ok=True)

CONDITIONS = ["commensal_static", "commensal_hobic", "dh_baseline", "dysbiotic_static"]

COND_INFO = {
    "commensal_static": {
        "label": "Commensal\nStatic",
        "short": "Comm-Static",
        "color": "#2ca02c",
    },
    "commensal_hobic": {
        "label": "Commensal\nHOBIC",
        "short": "Comm-HOBIC",
        "color": "#17becf",
    },
    "dh_baseline": {
        "label": "Dysbiotic\nHOBIC",
        "short": "Dysb-HOBIC",
        "color": "#d62728",
    },
    "dysbiotic_static": {
        "label": "Dysbiotic\nStatic",
        "short": "Dysb-Static",
        "color": "#ff7f0e",
    },
}


def load_stress_json(jobs_dir, cond, suffix):
    """Load stress summary JSON for a condition."""
    job_dir = jobs_dir / f"{cond}_T23_{suffix}"
    job_name = f"two_layer_T23_{cond}"
    json_path = job_dir / f"{job_name}_stress.json"
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f)
    return None


def extract_e_biofilm_from_inp(jobs_dir, cond, suffix):
    """Extract biofilm E [MPa] from INP file (second *Elastic section)."""
    job_dir = jobs_dir / f"{cond}_T23_{suffix}"
    job_name = f"two_layer_T23_{cond}"
    inp_path = job_dir / f"{job_name}.inp"
    if not inp_path.exists():
        return None
    elastic_count = 0
    with open(inp_path) as f:
        for line in f:
            if line.strip().startswith("*Elastic"):
                elastic_count += 1
                next_line = next(f, "").strip()
                if elastic_count == 2:
                    # Second *Elastic = biofilm
                    parts = next_line.split(",")
                    return float(parts[0].strip())
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS)
    ap.add_argument("--suffix", default="v2")
    args = ap.parse_args()

    jobs_dir = _HERE / "_abaqus_auto_jobs"
    conditions = args.conditions

    # --- Load all data ---
    results = {}
    for cond in conditions:
        stress = load_stress_json(jobs_dir, cond, args.suffix)
        e_bio = extract_e_biofilm_from_inp(jobs_dir, cond, args.suffix)
        if stress is None:
            print(f"  [SKIP] {cond}: no stress JSON")
            continue
        stress["E_biofilm_MPa"] = e_bio
        stress["E_biofilm_Pa"] = e_bio * 1e6 if e_bio else None
        results[cond] = stress
        print(
            f"  {cond}: E_bio={e_bio:.6e} MPa ({e_bio*1e6:.1f} Pa), "
            f"mises_max={stress['mises']['max']:.2f}, "
            f"disp_max={stress['displacement']['max_mag']:.1f}"
        )

    if len(results) < 2:
        print("Not enough data for comparison")
        return

    conds = [c for c in conditions if c in results]

    # --- Figure: 4-panel comparison ---
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    # ============ Panel (a): E_biofilm bar chart ============
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(conds))
    e_vals = [results[c]["E_biofilm_Pa"] for c in conds]
    colors = [COND_INFO[c]["color"] for c in conds]
    labels = [COND_INFO[c]["short"] for c in conds]

    bars = ax.bar(x, e_vals, color=colors, edgecolor="k", linewidth=0.5)
    for i, (bar, val) in enumerate(zip(bars, e_vals)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 15,
            f"{val:.0f} Pa",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("$E_{biofilm}$ [Pa]", fontsize=12)
    ax.set_title("(a) Biofilm Effective Stiffness", fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(e_vals) * 1.25)

    # ============ Panel (b): Displacement bar chart ============
    ax = fig.add_subplot(gs[0, 1])
    disp_max = [results[c]["displacement"]["max_mag"] for c in conds]
    disp_mean = [results[c]["displacement"]["mean_mag"] for c in conds]

    w = 0.35
    ax.bar(
        x - w / 2, disp_max, w, color=colors, edgecolor="k", linewidth=0.5, alpha=0.9, label="Max"
    )
    ax.bar(
        x + w / 2, disp_mean, w, color=colors, edgecolor="k", linewidth=0.5, alpha=0.5, label="Mean"
    )
    for i, val in enumerate(disp_max):
        ax.text(
            x[i] - w / 2,
            val + max(disp_max) * 0.02,
            f"{val:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Displacement [mm]", fontsize=12)
    ax.set_title("(b) Biofilm Layer Displacement", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)

    # ============ Panel (c): von Mises stress comparison ============
    ax = fig.add_subplot(gs[1, 0])
    mises_max = [results[c]["mises"]["max"] for c in conds]
    mises_mean = [results[c]["mises"]["mean"] for c in conds]
    mises_p95 = [results[c]["mises"]["p95"] for c in conds]

    w3 = 0.25
    ax.bar(
        x - w3, mises_mean, w3, color=colors, alpha=0.5, edgecolor="k", linewidth=0.5, label="Mean"
    )
    ax.bar(x, mises_p95, w3, color=colors, alpha=0.75, edgecolor="k", linewidth=0.5, label="P95")
    ax.bar(
        x + w3, mises_max, w3, color=colors, alpha=1.0, edgecolor="k", linewidth=0.5, label="Max"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("von Mises Stress [MPa]", fontsize=12)
    ax.set_title("(c) von Mises Stress (tooth+biofilm)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.text(
        0.02,
        0.95,
        "Dominated by dentin\n(tooth structure)",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        style="italic",
        color="gray",
    )

    # ============ Panel (d): E vs Displacement scatter ============
    ax = fig.add_subplot(gs[1, 1])
    for i, cond in enumerate(conds):
        e_pa = results[cond]["E_biofilm_Pa"]
        d_max = results[cond]["displacement"]["max_mag"]
        ax.scatter(e_pa, d_max, s=200, c=colors[i], edgecolors="k", linewidth=1.5, zorder=5)
        ax.annotate(
            COND_INFO[cond]["short"],
            (e_pa, d_max),
            textcoords="offset points",
            xytext=(10, 10),
            fontsize=10,
            fontweight="bold",
            color=colors[i],
        )

    ax.set_xlabel("$E_{biofilm}$ [Pa]", fontsize=12)
    ax.set_ylabel("Max Displacement [mm]", fontsize=12)
    ax.set_title("(d) Stiffness vs Deformation", fontsize=13, fontweight="bold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")

    # Fit power law line
    e_arr = np.array(e_vals)
    d_arr = np.array(disp_max)
    if len(e_arr) > 1:
        log_e = np.log10(e_arr)
        log_d = np.log10(d_arr)
        coeffs = np.polyfit(log_e, log_d, 1)
        e_fit = np.logspace(np.log10(min(e_arr) * 0.5), np.log10(max(e_arr) * 2), 50)
        d_fit = 10 ** (coeffs[0] * np.log10(e_fit) + coeffs[1])
        ax.plot(e_fit, d_fit, "k--", alpha=0.4, linewidth=1, label=f"slope={coeffs[0]:.2f}")
        ax.legend(fontsize=10)

    fig.suptitle(
        "3D Abaqus Stress Analysis: Hybrid DI Approach\n"
        "0D Hamilton ODE (condition scale) + 2D PDE (spatial pattern)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    out = _FIG_DIR / "stress_comparison_hybrid_3d.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")

    # --- Summary table to stdout ---
    print("\n" + "=" * 80)
    print(
        f"{'Condition':<20} {'E_bio [Pa]':>12} {'Mises max':>12} "
        f"{'Mises mean':>12} {'Disp max [mm]':>15}"
    )
    print("-" * 80)
    for cond in conds:
        r = results[cond]
        print(
            f"{COND_INFO[cond]['short']:<20} {r['E_biofilm_Pa']:>12.1f} "
            f"{r['mises']['max']:>12.2f} {r['mises']['mean']:>12.4f} "
            f"{r['displacement']['max_mag']:>15.1f}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
