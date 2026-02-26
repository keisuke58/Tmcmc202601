#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
odb_visualize.py
================
Run with standard Python (NOT abaqus python):
    python3 odb_visualize.py

Reads odb_nodes.csv and odb_elements.csv produced by odb_extract.py.
Generates 7 PNG figures in ./figures/:

  Fig1 – 3D point cloud of nodes, colored by |U| (displacement)
  Fig2 – 3D point cloud of element centroids, colored by MISES
  Fig3 – MISES histogram per tooth (T23, T30, T31)
  Fig4 – MISES radial profile vs DI bin (depth gradient)
  Fig5 – |U| box plot: INNER vs OUTER nodes per tooth
  Fig6 – MISES box plot per tooth (summary)
  Fig7 – Numeric summary table
"""
from __future__ import print_function
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless – no X display needed
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NODES_CSV = os.path.join(SCRIPT_DIR, "odb_nodes.csv")
ELEMS_CSV = os.path.join(SCRIPT_DIR, "odb_elements.csv")
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

TEETH = ["T23", "T30", "T31"]
COLORS = {"T23": "#e41a1c", "T30": "#377eb8", "T31": "#4daf4a", "BULK": "#aaaaaa"}
CMAP_U = "viridis"  # displacement
CMAP_S = "plasma"  # von Mises stress
DPI = 150


# ── Load CSVs ─────────────────────────────────────────────────────────────────
def load_csv(path, dtypes):
    """Load CSV with named columns. dtypes: list of (name, dtype) matching header order."""
    print("Loading", path)
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    return data


print("=" * 60)
nodes = load_csv(NODES_CSV, None)
elems = load_csv(ELEMS_CSV, None)
print("  Nodes: %d   Elements: %d" % (len(nodes), len(elems)))

# ── Parse arrays ──────────────────────────────────────────────────────────────
n_x = nodes["x"].astype(float)
n_y = nodes["y"].astype(float)
n_z = nodes["z"].astype(float)
Ux = nodes["Ux"].astype(float)
Uy = nodes["Uy"].astype(float)
Uz = nodes["Uz"].astype(float)
Umag = nodes["Umag"].astype(float)
n_tooth = np.array([str(t) for t in nodes["tooth"]], dtype="U4")
n_region = np.array([str(r) for r in nodes["region"]], dtype="U8")

e_cx = elems["cx"].astype(float)
e_cy = elems["cy"].astype(float)
e_cz = elems["cz"].astype(float)
mises = elems["mises"].astype(float)
e_bin = elems["bin"].astype(int)
e_tooth = np.array([str(t) for t in elems["tooth"]], dtype="U4")

# ── Global color scales ───────────────────────────────────────────────────────
umag_max = float(np.percentile(Umag, 99))
umag_min = 0.0
mises_min = float(np.percentile(mises, 1))
mises_max = float(np.percentile(mises, 99))

norm_u = mcolors.Normalize(vmin=umag_min, vmax=umag_max)
norm_s = mcolors.Normalize(vmin=mises_min, vmax=mises_max)


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────
def subsample(arr, max_pts=5000):
    """Return indices for a uniform subsample."""
    if len(arr) <= max_pts:
        return np.arange(len(arr))
    step = max(1, len(arr) // max_pts)
    return np.arange(0, len(arr), step)


def set_3d_labels(ax, fs=7):
    ax.set_xlabel("x (mm)", fontsize=fs)
    ax.set_ylabel("y (mm)", fontsize=fs)
    ax.set_zlabel("z (mm)", fontsize=fs)
    ax.tick_params(labelsize=fs)


# ═════════════════════════════════════════════════════════════════════════════
# Fig 1 – 3D node scatter colored by |U|
# ═════════════════════════════════════════════════════════════════════════════
print("\n[Fig 1] 3D node scatter – displacement |U|")
fig = plt.figure(figsize=(16, 5))
sc_ref = None
for col, tooth in enumerate(TEETH):
    ax = fig.add_subplot(1, 3, col + 1, projection="3d")
    mask = n_tooth == tooth
    idx = np.where(mask)[0]
    idx = idx[subsample(idx)]
    sc_ref = ax.scatter(
        n_x[idx],
        n_y[idx],
        n_z[idx],
        c=Umag[idx],
        cmap=CMAP_U,
        norm=norm_u,
        s=4,
        alpha=0.75,
        linewidths=0,
        depthshade=True,
    )
    ax.set_title("%s  (n_nodes=%d)" % (tooth, mask.sum()), fontsize=10)
    set_3d_labels(ax)
cbar = fig.colorbar(sc_ref, ax=fig.axes, label="|U| displacement (mm)", shrink=0.55, pad=0.05)
cbar.ax.tick_params(labelsize=8)
fig.suptitle("Biofilm nodal displacement magnitude – 1 MPa inward pressure", fontsize=12)
plt.tight_layout()
out = os.path.join(FIG_DIR, "Fig1_displacement_3D.png")
plt.savefig(out, dpi=DPI, bbox_inches="tight")
plt.close()
print("  Saved:", out)

# ═════════════════════════════════════════════════════════════════════════════
# Fig 2 – 3D element centroid scatter colored by MISES
# ═════════════════════════════════════════════════════════════════════════════
print("[Fig 2] 3D element scatter – MISES stress")
fig = plt.figure(figsize=(16, 5))
sc_ref = None
for col, tooth in enumerate(TEETH):
    ax = fig.add_subplot(1, 3, col + 1, projection="3d")
    mask = e_tooth == tooth
    idx = np.where(mask)[0]
    idx = idx[subsample(idx)]
    sc_ref = ax.scatter(
        e_cx[idx],
        e_cy[idx],
        e_cz[idx],
        c=mises[idx],
        cmap=CMAP_S,
        norm=norm_s,
        s=3,
        alpha=0.65,
        linewidths=0,
        depthshade=True,
    )
    ax.set_title("%s  (n_elems=%d)" % (tooth, mask.sum()), fontsize=10)
    set_3d_labels(ax)
cbar = fig.colorbar(sc_ref, ax=fig.axes, label="von Mises (MPa)", shrink=0.55, pad=0.05)
cbar.ax.tick_params(labelsize=8)
fig.suptitle(
    "Biofilm von Mises stress – 1 MPa inward pressure (2nd–98th pct color scale)", fontsize=12
)
plt.tight_layout()
out = os.path.join(FIG_DIR, "Fig2_mises_3D.png")
plt.savefig(out, dpi=DPI, bbox_inches="tight")
plt.close()
print("  Saved:", out)

# ═════════════════════════════════════════════════════════════════════════════
# Fig 3 – MISES histogram per tooth
# ═════════════════════════════════════════════════════════════════════════════
print("[Fig 3] MISES histogram per tooth")
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
for ax, tooth in zip(axes, TEETH):
    mask = e_tooth == tooth
    m_t = mises[mask]
    ax.hist(m_t, bins=80, color=COLORS[tooth], alpha=0.85, edgecolor="none")
    med = np.median(m_t)
    mu = np.mean(m_t)
    ax.axvline(med, color="k", linewidth=1.5, linestyle="--", label="Median")
    ax.axvline(mu, color="red", linewidth=1.5, linestyle=":", label="Mean")
    ax.set_title("%s  (n=%d)" % (tooth, mask.sum()), fontsize=11)
    ax.set_xlabel("von Mises (MPa)", fontsize=9)
    ax.set_ylabel("# elements", fontsize=9)
    ax.legend(fontsize=8)
    stats_txt = (
        "min   = %.4f\n"
        "q25   = %.4f\n"
        "med   = %.4f\n"
        "mean  = %.4f\n"
        "q75   = %.4f\n"
        "max   = %.4f"
    ) % (m_t.min(), np.percentile(m_t, 25), med, mu, np.percentile(m_t, 75), m_t.max())
    ax.text(
        0.98,
        0.97,
        stats_txt,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.75),
    )
fig.suptitle("von Mises stress histogram per tooth (1 MPa inward pressure)", fontsize=12)
plt.tight_layout()
out = os.path.join(FIG_DIR, "Fig3_mises_histogram.png")
plt.savefig(out, dpi=DPI, bbox_inches="tight")
plt.close()
print("  Saved:", out)

# ═════════════════════════════════════════════════════════════════════════════
# Fig 4 – MISES radial profile vs DI bin
# ═════════════════════════════════════════════════════════════════════════════
print("[Fig 4] MISES vs DI bin (depth profile)")
bins_present = np.unique(e_bin[e_bin >= 0])
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
for ax, tooth in zip(axes, TEETH):
    mask_t = e_tooth == tooth
    b_ids, means, meds, q25s, q75s = [], [], [], [], []
    for b in sorted(bins_present):
        mb = mask_t & (e_bin == b)
        if mb.sum() < 3:
            continue
        m_b = mises[mb]
        b_ids.append(b)
        means.append(np.mean(m_b))
        meds.append(np.median(m_b))
        q25s.append(np.percentile(m_b, 25))
        q75s.append(np.percentile(m_b, 75))
    b_ids = np.array(b_ids)
    means = np.array(means)
    meds = np.array(meds)
    q25s = np.array(q25s)
    q75s = np.array(q75s)
    ax.fill_between(b_ids, q25s, q75s, alpha=0.25, color=COLORS[tooth], label="IQR (25-75%)")
    ax.plot(b_ids, meds, "o-", color=COLORS[tooth], linewidth=2, markersize=5, label="Median")
    ax.plot(
        b_ids,
        means,
        "s--",
        color=COLORS[tooth],
        linewidth=1.2,
        markersize=4,
        alpha=0.8,
        label="Mean",
    )
    ax.set_title(tooth, fontsize=11)
    ax.set_xlabel("DI bin  (low bin = outer surface,\nhigh bin = inner/tooth contact)", fontsize=8)
    ax.set_ylabel("von Mises (MPa)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
fig.suptitle(
    "MISES radial profile vs DI bin\n"
    "(outer biofilm = lower bin, lower E  →  higher strain per unit stress)",
    fontsize=11,
)
plt.tight_layout()
out = os.path.join(FIG_DIR, "Fig4_mises_bin_profile.png")
plt.savefig(out, dpi=DPI, bbox_inches="tight")
plt.close()
print("  Saved:", out)

# ═════════════════════════════════════════════════════════════════════════════
# Fig 5 – |U| box plot: INNER vs OUTER nodes per tooth
# ═════════════════════════════════════════════════════════════════════════════
print("[Fig 5] Displacement |U| – INNER vs OUTER per tooth")
fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
for ax, tooth in zip(axes, TEETH):
    mask_t = n_tooth == tooth
    u_inner = Umag[mask_t & (n_region == "INNER")]
    u_outer = Umag[mask_t & (n_region == "OUTER")]
    bplot = ax.boxplot(
        [u_inner, u_outer],
        labels=["INNER\n(tooth\nadhesion)", "OUTER\n(free\nsurface)"],
        patch_artist=True,
        notch=False,
        showfliers=True,
        flierprops=dict(marker=".", markersize=2, alpha=0.3),
    )
    bplot["boxes"][0].set_facecolor(COLORS[tooth] + "66")
    bplot["boxes"][1].set_facecolor(COLORS[tooth] + "bb")
    ax.set_title(tooth, fontsize=11)
    ax.set_ylabel("|U| (mm)", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    for j, (lbl, u) in enumerate([("INNER", u_inner), ("OUTER", u_outer)]):
        ax.text(
            j + 1,
            ax.get_ylim()[1] * 0.98 if ax.get_ylim()[1] > 0 else u.max() * 1.05,
            "n=%d\nmed=%.3e" % (len(u), np.median(u)),
            ha="center",
            va="top",
            fontsize=7,
        )
fig.suptitle(
    "Displacement |U| at inner (ENCASTRE→0) and outer (free) surface nodes\n"
    "Inner = 0 by BC (ENCASTRE); outer shows actual deformation",
    fontsize=11,
)
plt.tight_layout()
out = os.path.join(FIG_DIR, "Fig5_displacement_inner_outer.png")
plt.savefig(out, dpi=DPI, bbox_inches="tight")
plt.close()
print("  Saved:", out)

# ═════════════════════════════════════════════════════════════════════════════
# Fig 6 – MISES box plot: per tooth (summary)
# ═════════════════════════════════════════════════════════════════════════════
print("[Fig 6] MISES box plot – per tooth")
fig, ax = plt.subplots(figsize=(8, 6))
data_per_tooth = [mises[e_tooth == t] for t in TEETH]
bplot = ax.boxplot(
    data_per_tooth,
    labels=TEETH,
    patch_artist=True,
    notch=False,
    showfliers=False,
    medianprops=dict(color="k", linewidth=2),
)
for patch, tooth in zip(bplot["boxes"], TEETH):
    patch.set_facecolor(COLORS[tooth] + "aa")
for i, (tooth, d) in enumerate(zip(TEETH, data_per_tooth)):
    ax.text(
        i + 1,
        float(np.percentile(d, 75)) * 1.01,
        "n=%d\nmed=%.3f\nmax=%.2f" % (len(d), float(np.median(d)), float(d.max())),
        ha="center",
        va="bottom",
        fontsize=8,
    )
ax.set_ylabel("von Mises stress (MPa)", fontsize=11)
ax.set_title("MISES stress per tooth – 1 MPa inward pressure\n(outliers hidden)", fontsize=11)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
out = os.path.join(FIG_DIR, "Fig6_mises_boxplot.png")
plt.savefig(out, dpi=DPI, bbox_inches="tight")
plt.close()
print("  Saved:", out)

# ═════════════════════════════════════════════════════════════════════════════
# Fig 7 – Numeric summary table
# ═════════════════════════════════════════════════════════════════════════════
print("[Fig 7] Summary table")
fig, ax = plt.subplots(figsize=(14, 3.5))
ax.axis("off")
col_labels = [
    "Tooth",
    "Elements",
    "Nodes",
    "U_max (mm)",
    "U_outer_med (mm)",
    "MISES_min",
    "MISES_mean",
    "MISES_max",
    "(MPa)",
]
rows = []
for tooth in TEETH:
    em = e_tooth == tooth
    nm = n_tooth == tooth
    nm_out = nm & (n_region == "OUTER")
    rows.append(
        [
            tooth,
            "%d" % em.sum(),
            "%d" % nm.sum(),
            "%.4e" % float(Umag[nm].max()),
            "%.4e" % float(np.median(Umag[nm_out])) if nm_out.sum() else "N/A",
            "%.4f" % float(mises[em].min()),
            "%.4f" % float(mises[em].mean()),
            "%.4f" % float(mises[em].max()),
            "",
        ]
    )
table = ax.table(cellText=rows, colLabels=col_labels, cellLoc="center", loc="center")
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.1, 2.2)
ax.set_title(
    "BioFilm3T – Abaqus/Standard 2024  |  Conformal C3D4  |  1 MPa inward pressure",
    fontsize=11,
    pad=20,
)
plt.tight_layout()
out = os.path.join(FIG_DIR, "Fig7_summary_table.png")
plt.savefig(out, dpi=DPI, bbox_inches="tight")
plt.close()
print("  Saved:", out)

# ═════════════════════════════════════════════════════════════════════════════
# Fig 8 – Slit: 3D view of T30 + T31 with APPROX region highlighted
# ═════════════════════════════════════════════════════════════════════════════
print("[Fig 8] Slit 3D – T30 + T31 with approximal region")
fig = plt.figure(figsize=(14, 6))

for col, tooth in enumerate(["T30", "T31"]):
    ax = fig.add_subplot(1, 2, col + 1, projection="3d")
    mask_t = n_tooth == tooth

    # OUTER (non-approx) nodes
    m_out = mask_t & (n_region == "OUTER")
    idx = np.where(m_out)[0][subsample(np.where(m_out)[0])]
    ax.scatter(
        n_x[idx],
        n_y[idx],
        n_z[idx],
        c=Umag[idx],
        cmap=CMAP_U,
        norm=norm_u,
        s=3,
        alpha=0.5,
        linewidths=0,
        label="OUTER",
    )

    # APPROX nodes – highlighted in orange, larger
    m_ap = mask_t & (n_region == "APPROX")
    if m_ap.sum() > 0:
        idx_ap = np.where(m_ap)[0]
        ax.scatter(
            n_x[idx_ap],
            n_y[idx_ap],
            n_z[idx_ap],
            c="orange",
            s=20,
            alpha=0.9,
            linewidths=0,
            zorder=5,
            label="APPROX (slit %d)" % m_ap.sum(),
        )

    # INNER nodes – small grey
    m_in = mask_t & (n_region == "INNER")
    idx_in = np.where(m_in)[0][subsample(np.where(m_in)[0], 2000)]
    ax.scatter(n_x[idx_in], n_y[idx_in], n_z[idx_in], c="lightgrey", s=1, alpha=0.3, linewidths=0)

    ax.set_title("%s biofilm\nOUTER|APPROX(orange)|INNER(grey)" % tooth, fontsize=10)
    set_3d_labels(ax)
    ax.legend(fontsize=7, loc="upper left")

fig.suptitle(
    "Slit region: T30 ↔ T31 inter-proximal biofilm\n"
    "Orange = approximal nodes (slit face, no Cload, Tie constraint active near contact)",
    fontsize=11,
)
plt.tight_layout()
out = os.path.join(FIG_DIR, "Fig8_slit_3D.png")
plt.savefig(out, dpi=DPI, bbox_inches="tight")
plt.close()
print("  Saved:", out)

# ═════════════════════════════════════════════════════════════════════════════
# Fig 9 – Slit: |U| and MISES at APPROX vs OUTER (T30 + T31 side-by-side)
# ═════════════════════════════════════════════════════════════════════════════
print("[Fig 9] Slit – APPROX vs OUTER displacement and stress comparison")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: displacement |U| boxplot – INNER / OUTER / APPROX per tooth
ax = axes[0]
data_list, tick_list = [], []
for tooth in ["T30", "T31"]:
    mask_t = n_tooth == tooth
    for region, lbl in [("INNER", "INNER"), ("APPROX", "APPROX\n(slit)"), ("OUTER", "OUTER")]:
        m = mask_t & (n_region == region)
        if m.sum() > 0:
            data_list.append(Umag[m])
            tick_list.append("%s\n%s" % (tooth, lbl))

bplot = ax.boxplot(
    data_list,
    tick_labels=tick_list,
    patch_artist=True,
    notch=False,
    showfliers=True,
    flierprops=dict(marker=".", markersize=1.5, alpha=0.3),
)
colors = [
    "#aaaacc",
    "#ff8800",
    "#5599ff",  # T30: INNER, APPROX, OUTER
    "#aaccaa",
    "#ff8800",
    "#44aa44",
]  # T31: INNER, APPROX, OUTER
for patch, col in zip(bplot["boxes"], colors):
    patch.set_facecolor(col + "aa")
ax.set_title("Displacement |U| by surface region\n(APPROX = slit face)", fontsize=10)
ax.set_ylabel("|U| (mm)", fontsize=9)
ax.grid(True, alpha=0.3, axis="y")
ax.tick_params(axis="x", labelsize=7)

# Right: MISES distribution – APPROX vs non-approx outer nodes using violin
# Need element data close to APPROX node positions → approximate via node membership
# Use Umag at each region as proxy for mechanical state
ax2 = axes[1]
slit_teeth = ["T30", "T31"]
positions, violin_data, xlabels = [], [], []
pos = 1
for tooth in slit_teeth:
    for region, lbl in [("INNER", "INNER"), ("APPROX", "APPROX\n(slit)"), ("OUTER", "OUTER")]:
        m = (n_tooth == tooth) & (n_region == region)
        if m.sum() > 0:
            violin_data.append(Umag[m] * 1e6)  # convert mm → nm for readability
            xlabels.append("%s\n%s" % (tooth, lbl))
            positions.append(pos)
        pos += 1
    pos += 0.5  # gap between teeth

parts = ax2.violinplot(violin_data, positions=positions, showmedians=True, showextrema=True)
vcolors = ["#9999dd", "#ff8800", "#4488ff", "#99bb99", "#ff8800", "#44aa44"]  # T30  # T31
for pc, col in zip(parts["bodies"], vcolors):
    pc.set_facecolor(col)
    pc.set_alpha(0.7)
ax2.set_xticks(positions)
ax2.set_xticklabels(xlabels, fontsize=7)
ax2.set_title("Displacement |U| violin per surface region\n(×10⁻⁶ mm)", fontsize=10)
ax2.set_ylabel("|U| × 10⁻⁶ (mm)", fontsize=9)
ax2.grid(True, alpha=0.3, axis="y")

fig.suptitle(
    "Slit (inter-proximal) analysis – T30 ↔ T31\n"
    "APPROX nodes: Cload excluded, Tie constraint active at contact zone",
    fontsize=11,
)
plt.tight_layout()
out = os.path.join(FIG_DIR, "Fig9_slit_approx_vs_outer.png")
plt.savefig(out, dpi=DPI, bbox_inches="tight")
plt.close()
print("  Saved:", out)

# ═════════════════════════════════════════════════════════════════════════════
# Fig 10 – Crown vs Slit summary comparison
# ═════════════════════════════════════════════════════════════════════════════
print("[Fig 10] Crown vs Slit summary")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: MISES box plot for all 3 teeth (crown + slit)
ax = axes[0]
data_all = [mises[e_tooth == t] for t in TEETH]
bplot = ax.boxplot(
    data_all,
    tick_labels=TEETH,
    patch_artist=True,
    notch=False,
    showfliers=False,
    medianprops=dict(color="k", linewidth=2),
)
for patch, tooth in zip(bplot["boxes"], TEETH):
    patch.set_facecolor(COLORS[tooth] + "aa")
# Annotate crown vs slit
ax.axvspan(0.5, 1.5, alpha=0.07, color="red", label="Crown (T23)")
ax.axvspan(1.5, 3.5, alpha=0.07, color="blue", label="Slit (T30+T31)")
for i, (tooth, d) in enumerate(zip(TEETH, data_all)):
    ax.text(
        i + 1,
        float(np.percentile(d, 75)) * 1.01,
        "med=%.3f" % float(np.median(d)),
        ha="center",
        va="bottom",
        fontsize=8,
    )
ax.set_ylabel("von Mises (MPa)", fontsize=10)
ax.set_title("MISES: Crown (T23) vs Slit (T30,T31)", fontsize=10)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="y")

# Right: max displacement per region type across all teeth
ax2 = axes[1]
region_data = {}
for region in ["INNER", "APPROX", "OUTER"]:
    m = n_region == region
    if m.sum() > 0:
        region_data[region] = Umag[m]

bplot2 = ax2.boxplot(
    list(region_data.values()),
    tick_labels=list(region_data.keys()),
    patch_artist=True,
    notch=False,
    showfliers=True,
    flierprops=dict(marker=".", markersize=1.5, alpha=0.2),
)
reg_colors = ["#5566aa", "#ff8800", "#44aa55"]
for patch, col in zip(bplot2["boxes"], reg_colors):
    patch.set_facecolor(col + "aa")
ax2.set_ylabel("|U| (mm)", fontsize=10)
ax2.set_title("Displacement by surface type\n(all teeth)", fontsize=10)
ax2.grid(True, alpha=0.3, axis="y")
for i, (reg, d) in enumerate(region_data.items()):
    ax2.text(
        i + 1,
        float(np.percentile(d, 75)) * 1.01,
        "n=%d\nmed=%.2e" % (len(d), float(np.median(d))),
        ha="center",
        va="bottom",
        fontsize=7,
    )

fig.suptitle(
    "BioFilm3T – Crown vs Slit comparison\n"
    "Conformal C3D4  |  1 MPa inward pressure  |  Slit: ENCASTRE inner + Tie at contact",
    fontsize=11,
)
plt.tight_layout()
out = os.path.join(FIG_DIR, "Fig10_crown_vs_slit.png")
plt.savefig(out, dpi=DPI, bbox_inches="tight")
plt.close()
print("  Saved:", out)

# ═════════════════════════════════════════════════════════════════════════════
# Fig 11 – Cross-section slices at y = const (P6: visualization enhancements)
# ═════════════════════════════════════════════════════════════════════════════
print("\n[Fig 11] Cross-section slices (y = const)")

# Pick 4 evenly-spaced y planes spanning the model
y_cuts = [float(np.percentile(e_cy, p)) for p in [20, 40, 60, 80]]
half_width = (e_cy.max() - e_cy.min()) * 0.025  # ±2.5% of range per slice

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for k, y0 in enumerate(y_cuts):
    ax = axes[k]
    in_slice = np.abs(e_cy - y0) < half_width
    if in_slice.sum() < 10:
        ax.set_title("y=%.1f mm  (no data)" % y0)
        continue
    cx_s = e_cx[in_slice]
    cz_s = e_cz[in_slice]
    ms_s = mises[in_slice]
    sc = ax.scatter(cx_s, cz_s, c=ms_s, cmap=CMAP_S, norm=norm_s, s=6, alpha=0.75, linewidths=0)
    # Colour teeth differently with marker shape
    for tooth, marker in [("T23", "o"), ("T30", "s"), ("T31", "^")]:
        mt = in_slice & (e_tooth == tooth)
        if mt.sum() > 0:
            ax.scatter(
                e_cx[mt],
                e_cz[mt],
                c=mises[mt],
                cmap=CMAP_S,
                norm=norm_s,
                s=8,
                alpha=0.8,
                marker=marker,
                linewidths=0,
                label=tooth,
            )
    plt.colorbar(sc, ax=ax, label="MISES (MPa)", shrink=0.75)
    ax.set_xlabel("x (mm)", fontsize=8)
    ax.set_ylabel("z (mm)", fontsize=8)
    ax.set_title(
        "Cross-section y = %.1f mm  (±%.1f mm, n=%d elements)" % (y0, half_width, in_slice.sum()),
        fontsize=9,
    )
    ax.legend(fontsize=7, markerscale=1.5, loc="upper right")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.2)

fig.suptitle(
    "MISES cross-sections at y = const planes\n"
    "(x-z slice through biofilm, coloured by von Mises stress)",
    fontsize=12,
)
plt.tight_layout()
out = os.path.join(FIG_DIR, "Fig11_cross_section_mises.png")
plt.savefig(out, dpi=DPI, bbox_inches="tight")
plt.close()
print("  Saved:", out)

# ═════════════════════════════════════════════════════════════════════════════
# Fig 12 – Strain energy density  w ≈ σ_vM² / (3 E_eff)   (P6)
# ═════════════════════════════════════════════════════════════════════════════
print("[Fig 12] Strain energy density (SED ≈ σ_vM²/(3·E_eff))")

# Reconstruct E_eff from DI bin using same formula as biofilm_conformal_tet.py
E_MAX_MPA = 10.0  # MPa
E_MIN_MPA = 0.5  # MPa
DI_SCALE = 0.025778
DI_EXP = 2.0
N_BINS = 20

# DI field statistics (from abaqus_field_dh_3d.csv snapshot 20)
# bin midpoint DI_b = (b + 0.5) * (di_max / N_BINS), di_max ~= 0.0135
# Approximate di_max from bins used (3..11 → di_max ~ 0.0135)
DI_MAX_APPROX = 0.0135
bin_w = DI_MAX_APPROX / N_BINS

bin_E_mpa = np.zeros(N_BINS)
for b in range(N_BINS):
    di_b = (b + 0.5) * bin_w
    r = float(np.clip(di_b / DI_SCALE, 0.0, 1.0))
    bin_E_mpa[b] = E_MAX_MPA * (1.0 - r) ** DI_EXP + E_MIN_MPA * r

# Map each element's bin to E_eff
E_eff_per_elem = np.array([bin_E_mpa[max(0, min(N_BINS - 1, b))] for b in e_bin])

# SED (MPa·mm/mm² = MPa) – approximate isotropic formula
# w = σ_vM² / (3·E) is valid for uniaxial; better estimate = σ_vM²/(2·E·(1+ν)/3·...)
# For simplicity use:  w = σ_vM² / (3·E_eff)  [MPa × MPa / MPa = MPa]
# Note: 1 MPa = 1 N/mm² = 1 mJ/mm³ in mm/N/MPa unit system
sed = mises**2 / (3.0 * np.where(E_eff_per_elem > 0, E_eff_per_elem, 1e-6))

sed_min = float(np.percentile(sed, 1))
sed_max = float(np.percentile(sed, 99))
norm_sed = mcolors.Normalize(vmin=sed_min, vmax=sed_max)

fig, axes2 = plt.subplots(1, 3, figsize=(16, 5))
for col, tooth in enumerate(TEETH):
    ax = axes2[col]
    mask = e_tooth == tooth
    idx = np.where(mask)[0][subsample(np.where(mask)[0])]
    sc = ax.scatter(
        e_cx[idx],
        e_cz[idx],
        c=sed[idx],
        cmap="inferno",
        norm=norm_sed,
        s=4,
        alpha=0.75,
        linewidths=0,
    )
    plt.colorbar(sc, ax=ax, label="SED (mJ/mm³)", shrink=0.75)
    ax.set_xlabel("x (mm)", fontsize=8)
    ax.set_ylabel("z (mm)", fontsize=8)
    ax.set_title(
        "%s  SED median=%.4f  max=%.3f  mJ/mm³"
        % (tooth, float(np.median(sed[mask])), float(sed[mask].max())),
        fontsize=9,
    )
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.2)

fig.suptitle(
    "Approximate strain energy density  $w \\approx \\sigma_\\mathrm{vM}^2 / (3E_\\mathrm{eff})$\n"
    "(x-z projection, 1 MPa inward pressure, inferno colour scale)",
    fontsize=12,
)
plt.tight_layout()
out = os.path.join(FIG_DIR, "Fig12_strain_energy_density.png")
plt.savefig(out, dpi=DPI, bbox_inches="tight")
plt.close()
print("  Saved:", out)

# ── Print final stats to console ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY  (all teeth combined)")
print("  Total nodes   : %d" % len(nodes))
print("  Total elements: %d" % len(elems))
print("  Umag  – min=%.3e  max=%.3e  mm" % (float(Umag.min()), float(Umag.max())))
print(
    "  MISES – min=%.4f  mean=%.4f  max=%.4f  MPa"
    % (float(mises.min()), float(mises.mean()), float(mises.max()))
)
print("\nPer-tooth MISES:")
for tooth in TEETH:
    m = mises[e_tooth == tooth]
    print(
        "  %-4s  min=%.4f  mean=%.4f  max=%.4f  MPa  (n=%d)"
        % (tooth, float(m.min()), float(m.mean()), float(m.max()), len(m))
    )
print("\nFigures saved to:", FIG_DIR)
for fname in sorted(os.listdir(FIG_DIR)):
    if fname.endswith(".png"):
        fpath = os.path.join(FIG_DIR, fname)
        print("  %-35s  %.0f KB" % (fname, os.path.getsize(fpath) / 1024))
