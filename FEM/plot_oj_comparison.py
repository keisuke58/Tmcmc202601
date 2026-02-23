#!/usr/bin/env python3
"""
plot_oj_comparison.py
=====================
Generate comparison figures from oj_crown_slit_stress.csv
(produced by compare_biofilm_abaqus.py on multiple ODB files).

Output (written to _aniso_sweep/figures/):
  fig_C2_geometry_comparison.png   – S_Mises bar chart across geometry types (β=0.5)
  fig_C2_beta_sweep_t23.png        – S_Mises vs β for T23 geometry
  fig_C2_gradient_crown_slit.png   – depth gradient (inner→outer) for crown & slit
"""

import csv
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE    = Path(__file__).resolve().parent
CSV_IN  = HERE / "oj_crown_slit_stress.csv"
FIG_DIR = HERE / "_aniso_sweep" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── ODB label metadata ────────────────────────────────────────────────────────
ODB_META = {
    "OJ_Crown_T23_b050.odb": dict(label="OJ Crown T23\n(hollow, β=0.5)",
                                   group="openjaw_hollow", beta=0.50, color="#17becf"),
    "OJ_Slit_T3031_b050.odb": dict(label="OJ Slit T30-T31\n(hollow, β=0.5)",
                                    group="openjaw_hollow", beta=0.50, color="#bcbd22"),
    "oj_t23_b100.odb":  dict(label="OJ T23 solid\nβ=1.0 (iso)",
                              group="openjaw_solid_t23", beta=1.00, color="#9467bd"),
    "oj_t23_b070.odb":  dict(label="OJ T23 solid\nβ=0.7",
                              group="openjaw_solid_t23", beta=0.70, color="#9467bd"),
    "oj_t23_b050.odb":  dict(label="OJ T23 solid\nβ=0.5",
                              group="openjaw_solid_t23", beta=0.50, color="#9467bd"),
    "oj_t23_b030.odb":  dict(label="OJ T23 solid\nβ=0.3",
                              group="openjaw_solid_t23", beta=0.30, color="#9467bd"),
    "oj_t30_b050.odb":  dict(label="OJ T30 solid\nβ=0.5",
                              group="openjaw_solid_other", beta=0.50, color="#8c564b"),
    "oj_t31_b050.odb":  dict(label="OJ T31 solid\nβ=0.5",
                              group="openjaw_solid_other", beta=0.50, color="#e377c2"),
    "dh_cube_v3.odb":   dict(label="Idealized Cube\nβ=0.5",
                              group="idealized", beta=0.50, color="#d62728"),
    "dh_crown_v3.odb":  dict(label="Idealized Crown\nβ=0.5",
                              group="idealized", beta=0.50, color="#ff7f0e"),
    "dh_slit_v3.odb":   dict(label="Idealized Slit\nβ=0.5",
                              group="idealized", beta=0.50, color="#1f77b4"),
}


def _load_csv(path):
    data = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            odb  = row["odb"]
            frac = float(row["depth_frac"])
            sm   = float(row["S_Mises"])
            data.setdefault(odb, {})[frac] = sm
    return data


def plot_geometry_comparison(data):
    """Bar chart: S_Mises at β=0.5 for all geometries."""
    groups = [
        ("openjaw_hollow",       "OpenJaw\nHollow"),
        ("openjaw_solid_t23",    "OJ Solid\nT23"),
        ("openjaw_solid_other",  "OJ Solid\nT30/T31"),
        ("idealized",            "Idealized"),
    ]

    # Select β=0.5 cases only (or only instance for hollow)
    bar_data = []
    for odb, meta in ODB_META.items():
        if odb not in data:
            continue
        if meta["beta"] != 0.50 and meta["group"] == "openjaw_solid_t23":
            continue
        fracs = data[odb]
        inner = fracs.get(0.0, math.nan)
        outer = fracs.get(1.0, math.nan)
        bar_data.append((meta["label"].replace("\n", " "),
                         meta["group"], meta["color"],
                         inner / 1e6, outer / 1e6))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    labels  = [d[0] for d in bar_data]
    colors  = [d[2] for d in bar_data]
    inners  = [d[3] for d in bar_data]
    outers  = [d[4] for d in bar_data]
    x = np.arange(len(labels))

    for si, (vals, title) in enumerate([
        (inners, "Inner surface S_Mises (depth_frac=0)"),
        (outers, "Outer surface S_Mises (depth_frac=1)"),
    ]):
        ax = axes[si]
        bars = ax.bar(x, vals, color=colors, edgecolor="k", linewidth=0.6, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("S_Mises (MPa)")
        ax.set_title(title, fontsize=10)
        ax.axhline(1.0, color="gray", ls="--", lw=0.8, label="Applied pressure (1 MPa)")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                    "%.3f" % v, ha="center", va="bottom", fontsize=7)

    fig.suptitle(
        "C2: S_Mises Comparison across Biofilm Geometries  (β=0.5, 1 MPa applied)",
        fontsize=12, fontweight="bold"
    )
    out = FIG_DIR / "fig_C2_geometry_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("[plot] %s" % out.name)


def plot_beta_sweep(data):
    """S_Mises vs β for T23 solid biofilm."""
    t23_odbs = [k for k, m in ODB_META.items() if m["group"] == "openjaw_solid_t23"]
    betas  = sorted([ODB_META[k]["beta"] for k in t23_odbs])
    inner  = []
    outer  = []
    for b in betas:
        odb = next(k for k in t23_odbs if ODB_META[k]["beta"] == b)
        fd  = data.get(odb, {})
        inner.append(fd.get(0.0, math.nan) / 1e6)
        outer.append(fd.get(1.0, math.nan) / 1e6)

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.plot(betas, inner, "o-", color="#4c72b0", lw=2, ms=8, label="Inner surface (depth_frac=0)")
    ax.plot(betas, outer, "s--", color="#dd8452", lw=2, ms=8, label="Outer surface (depth_frac=1)")
    ax.axhline(1.0, color="gray", ls=":", lw=0.8, label="Applied pressure (1 MPa)")
    ax.set_xlabel("Anisotropy ratio β = E_trans / E_stiff", fontsize=11)
    ax.set_ylabel("S_Mises (MPa)", fontsize=11)
    ax.set_title("C2: OJ T23 Solid Biofilm  –  S_Mises vs β", fontsize=12)
    ax.invert_xaxis()
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    out = FIG_DIR / "fig_C2_beta_sweep_t23.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("[plot] %s" % out.name)


def plot_depth_gradient(data):
    """S_Mises at 3 depth fractions for hollow crown & slit vs idealized."""
    target_odbs = [
        ("OJ_Crown_T23_b050.odb",  "#17becf", "OJ Crown T23 (hollow, β=0.5)"),
        ("OJ_Slit_T3031_b050.odb", "#bcbd22", "OJ Slit T30-T31 (hollow, β=0.5)"),
        ("dh_crown_v3.odb",        "#ff7f0e", "Idealized Crown (β=0.5)"),
        ("dh_slit_v3.odb",         "#1f77b4", "Idealized Slit (β=0.5)"),
        ("dh_cube_v3.odb",         "#d62728", "Idealized Cube (β=0.5)"),
    ]
    fracs = [0.0, 0.5, 1.0]
    frac_labels = ["Inner\n(substrate)", "Mid", "Outer\n(surface)"]

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for odb, color, label in target_odbs:
        fd = data.get(odb, {})
        vals = [fd.get(f, math.nan) / 1e6 for f in fracs]
        ls = "--" if odb.startswith("dh_") else "-"
        ax.plot(range(len(fracs)), vals, "o" + ls, color=color,
                lw=2, ms=8, label=label)

    ax.set_xticks(range(len(fracs)))
    ax.set_xticklabels(frac_labels, fontsize=10)
    ax.axhline(1.0, color="gray", ls=":", lw=0.8, label="Applied pressure (1 MPa)")
    ax.set_ylabel("S_Mises (MPa)", fontsize=11)
    ax.set_title("C2: S_Mises Depth Profile  –  Real vs Idealized Biofilm Geometry",
                 fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    out = FIG_DIR / "fig_C2_depth_gradient.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("[plot] %s" % out.name)


def main():
    if not CSV_IN.exists():
        print("CSV not found: %s" % CSV_IN)
        return
    data = _load_csv(CSV_IN)
    print("Loaded %d ODB entries from %s" % (len(data), CSV_IN.name))

    plot_geometry_comparison(data)
    plot_beta_sweep(data)
    plot_depth_gradient(data)

    print("\n[done] figures → %s/" % FIG_DIR)


if __name__ == "__main__":
    main()
