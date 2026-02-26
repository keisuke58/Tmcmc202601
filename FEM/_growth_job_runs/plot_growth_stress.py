#!/usr/bin/env python3
"""
plot_growth_stress.py
=====================
Visualize stress results from the 4-condition growth load analysis (P23, Tooth 23).

Figures produced:
  fig_growth_1_condition_bar.png   – bar chart: max/mean σ_Mises (GROWTH & LOAD)
  fig_growth_2_depth_profile.png   – σ_Mises vs normalised depth (z_norm) per condition
  fig_growth_3_spatial_load.png    – 2D scatter (x-z plane) coloured by σ_Mises_LOAD
  fig_growth_4_spatial_growth.png  – 2D scatter (x-z plane) coloured by σ_Mises_GROWTH
"""
import os
import csv
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

HERE = os.path.dirname(os.path.abspath(__file__))

CONDITIONS = ["commensal_static", "commensal_hobic", "dh_baseline", "dysbiotic_static"]
LABELS = ["Commensal\nStatic", "Commensal\nHOBIC", "DH\nBaseline", "Dysbiotic\nStatic"]
COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

ALPHA_EFF = {
    "commensal_static": 0.01717,
    "commensal_hobic": 0.01436,
    "dh_baseline": 0.007055,
    "dysbiotic_static": 0.01759,
}


def load_csv(cond):
    path = os.path.join(HERE, "stress_%s.csv" % cond)
    data = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k in reader.fieldnames:
                if k not in data:
                    data[k] = []
                data[k].append(float(row[k]))
    for k in data:
        data[k] = np.array(data[k])
    return data


# ── Load all data ─────────────────────────────────────────────────────────────
all_data = {}
for cond in CONDITIONS:
    all_data[cond] = load_csv(cond)

# Convert MPa → Pa
for cond, d in all_data.items():
    d["mises_growth_Pa"] = d["mises_growth"] * 1e6
    d["mises_load_Pa"] = d["mises_load"] * 1e6
    d["s11_Pa"] = d["s11_load"] * 1e6
    d["s22_Pa"] = d["s22_load"] * 1e6
    d["s33_Pa"] = d["s33_load"] * 1e6

# ─────────────────────────────────────────────────────────────────────────────
# Fig 1: Bar chart – max & mean σ_Mises for GROWTH and LOAD
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle("P23 Biofilm Growth Load – Stress Summary by Condition", fontsize=13)

x = np.arange(len(CONDITIONS))
w = 0.35

for ax_i, (step, step_key) in enumerate(
    [("GROWTH (eigenstrain)", "mises_growth_Pa"), ("LOAD (GCF 100 Pa)", "mises_load_Pa")]
):
    ax = axes[ax_i]
    maxv = [all_data[c][step_key].max() for c in CONDITIONS]
    meanv = [all_data[c][step_key].mean() for c in CONDITIONS]

    bars_max = ax.bar(
        x - w / 2, maxv, w, label="max", color=COLORS, alpha=0.9, edgecolor="k", lw=0.7
    )
    bars_mean = ax.bar(
        x + w / 2,
        meanv,
        w,
        label="mean",
        color=COLORS,
        alpha=0.5,
        edgecolor="k",
        lw=0.7,
        hatch="//",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=9)
    ax.set_ylabel("σ_Mises [Pa]", fontsize=10)
    ax.set_title("Step: %s" % step, fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # annotate max values
    for bar, val in zip(bars_max, maxv):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val * 1.02,
            "%.2g" % val,
            ha="center",
            va="bottom",
            fontsize=8,
        )

plt.tight_layout()
out = os.path.join(HERE, "fig_growth_1_condition_bar.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print("Saved: %s" % out)

# ─────────────────────────────────────────────────────────────────────────────
# Fig 2: Depth profile – σ_Mises vs normalised z-depth
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("P23 Biofilm – σ_Mises vs Normalised Depth (z_norm)", fontsize=13)

for ax_i, (step_key, step_label) in enumerate(
    [("mises_growth_Pa", "GROWTH step (eigenstrain)"), ("mises_load_Pa", "LOAD step (GCF 100 Pa)")]
):
    ax = axes[ax_i]
    for cond, label, color in zip(CONDITIONS, LABELS, COLORS):
        d = all_data[cond]
        cz = d["cz"]
        z_min, z_max = cz.min(), cz.max()
        z_norm = (cz - z_min) / (z_max - z_min + 1e-12)
        mises = d[step_key]

        # Bin by z_norm into 20 slices
        n_slices = 20
        bins = np.linspace(0, 1, n_slices + 1)
        z_mid, m_mean, m_std = [], [], []
        for bi in range(n_slices):
            mask = (z_norm >= bins[bi]) & (z_norm < bins[bi + 1])
            if mask.sum() == 0:
                continue
            z_mid.append(0.5 * (bins[bi] + bins[bi + 1]))
            m_mean.append(mises[mask].mean())
            m_std.append(mises[mask].std())
        z_mid = np.array(z_mid)
        m_mean = np.array(m_mean)
        m_std = np.array(m_std)

        ax.plot(z_mid, m_mean, "-o", color=color, label=label.replace("\n", " "), ms=4, lw=1.5)
        ax.fill_between(z_mid, m_mean - m_std, m_mean + m_std, color=color, alpha=0.15)

    ax.set_xlabel("Normalised depth z_norm (0=apical, 1=coronal)", fontsize=9)
    ax.set_ylabel("σ_Mises [Pa]", fontsize=10)
    ax.set_title(step_label, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
out = os.path.join(HERE, "fig_growth_2_depth_profile.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print("Saved: %s" % out)

# ─────────────────────────────────────────────────────────────────────────────
# Fig 3: Spatial scatter (x-z plane) coloured by σ_Mises_LOAD – 4 panels
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("P23 Biofilm – σ_Mises_LOAD Spatial Field (x-z projection)", fontsize=13)

# Global colour limits across all conditions
all_load = np.concatenate([all_data[c]["mises_load_Pa"] for c in CONDITIONS])
vmin, vmax = all_load.min(), all_load.max()
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
cmap = "plasma"

for i, (cond, label) in enumerate(zip(CONDITIONS, LABELS)):
    ax = axes[i // 2][i % 2]
    d = all_data[cond]
    sc = ax.scatter(
        d["cx"],
        d["cz"],
        c=d["mises_load_Pa"],
        cmap=cmap,
        norm=norm,
        s=0.5,
        alpha=0.6,
        rasterized=True,
    )
    ax.set_xlabel("x [mm]", fontsize=8)
    ax.set_ylabel("z [mm]", fontsize=8)
    ax.set_title("%s  (alpha_eff=%.4f)" % (label.replace("\n", " "), ALPHA_EFF[cond]), fontsize=9)
    ax.set_aspect("equal")
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("σ_Mises [Pa]", fontsize=8)

plt.tight_layout()
out = os.path.join(HERE, "fig_growth_3_spatial_load.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print("Saved: %s" % out)

# ─────────────────────────────────────────────────────────────────────────────
# Fig 4: Spatial scatter (x-z plane) coloured by σ_Mises_GROWTH – 4 panels
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("P23 Biofilm – σ_Mises_GROWTH Spatial Field (x-z projection)", fontsize=13)

all_grow = np.concatenate([all_data[c]["mises_growth_Pa"] for c in CONDITIONS])
vmin_g, vmax_g = all_grow.min(), all_grow.max()
norm_g = mcolors.Normalize(vmin=vmin_g, vmax=vmax_g)

for i, (cond, label) in enumerate(zip(CONDITIONS, LABELS)):
    ax = axes[i // 2][i % 2]
    d = all_data[cond]
    sc = ax.scatter(
        d["cx"],
        d["cz"],
        c=d["mises_growth_Pa"],
        cmap="viridis",
        norm=norm_g,
        s=0.5,
        alpha=0.6,
        rasterized=True,
    )
    ax.set_xlabel("x [mm]", fontsize=8)
    ax.set_ylabel("z [mm]", fontsize=8)
    ax.set_title("%s  (alpha_eff=%.4f)" % (label.replace("\n", " "), ALPHA_EFF[cond]), fontsize=9)
    ax.set_aspect("equal")
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("σ_Mises [Pa]", fontsize=8)

plt.tight_layout()
out = os.path.join(HERE, "fig_growth_4_spatial_growth.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print("Saved: %s" % out)

# ─────────────────────────────────────────────────────────────────────────────
# Fig 5: α_eff vs stress relationship
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle("P23 – alpha_eff vs Stress (GROWTH/LOAD Steps)", fontsize=12)

alphas = [ALPHA_EFF[c] for c in CONDITIONS]
max_grow = [all_data[c]["mises_growth_Pa"].max() for c in CONDITIONS]
max_load = [all_data[c]["mises_load_Pa"].max() for c in CONDITIONS]
mean_load = [all_data[c]["mises_load_Pa"].mean() for c in CONDITIONS]

ax = axes[0]
ax.scatter(alphas, max_grow, c=COLORS, s=80, zorder=5, edgecolors="k", lw=0.8)
for a, v, l in zip(alphas, max_grow, LABELS):
    ax.annotate(l.replace("\n", " "), (a, v), textcoords="offset points", xytext=(5, 3), fontsize=8)
# Fit linear
m, b = np.polyfit(alphas, max_grow, 1)
xfit = np.linspace(min(alphas) * 0.9, max(alphas) * 1.05, 50)
ax.plot(xfit, m * xfit + b, "k--", lw=1, alpha=0.6, label="linear fit")
ax.set_xlabel("alpha_eff (= alpha_final × 0.85)", fontsize=9)
ax.set_ylabel("max σ_Mises_GROWTH [Pa]", fontsize=9)
ax.set_title(
    "GROWTH: σ_max ∝ alpha_eff  (R²≈%.3f)" % np.corrcoef(alphas, max_grow)[0, 1] ** 2, fontsize=9
)
ax.grid(alpha=0.3)
ax.legend(fontsize=8)

ax = axes[1]
ax.scatter(alphas, max_load, c=COLORS, s=80, zorder=5, edgecolors="k", lw=0.8, label="max")
ax.scatter(
    alphas, mean_load, c=COLORS, s=50, zorder=5, marker="^", edgecolors="k", lw=0.8, label="mean"
)
for a, v, l in zip(alphas, max_load, LABELS):
    ax.annotate(l.replace("\n", " "), (a, v), textcoords="offset points", xytext=(5, 3), fontsize=8)
ax.set_xlabel("alpha_eff", fontsize=9)
ax.set_ylabel("σ_Mises_LOAD [Pa]", fontsize=9)
ax.set_title("LOAD: σ dominated by GCF pressure (100 Pa)", fontsize=9)
ax.grid(alpha=0.3)
ax.legend(fontsize=8)

plt.tight_layout()
out = os.path.join(HERE, "fig_growth_5_alpha_vs_stress.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print("Saved: %s" % out)

print("\nAll figures written to %s" % HERE)
