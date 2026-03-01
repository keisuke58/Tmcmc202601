#!/usr/bin/env python3
"""
compare_conditions_late.py  --  Late-time (t≈0.5) 4-condition comparison
=========================================================================

Uses late-snapshot ODB results (BioFilm3T_*_late.odb → odb_elements_*_late.csv)
and late DI fields (abaqus_field_*_late.csv) to produce scientifically meaningful
condition comparison where the dysbiotic cascade has fully developed.

Outputs
-------
  figures/LateFig1_mises_violin.png      -- MISES violin (4 conditions, 3 teeth)
  figures/LateFig2_di_comparison.png     -- Late DI field (t=0.5) side by side
  figures/LateFig3_eeff_comparison.png   -- E_eff distribution (late)
  figures/LateFig4_delta_mises.png       -- ΔMISES(dh_baseline − others) per tooth/bin
  figures/LateFig5_summary_table.png     -- numeric summary table

Usage
-----
  python3 compare_conditions_late.py
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_HERE = os.path.dirname(os.path.abspath(__file__))
_FIGS = os.path.join(_HERE, "figures")
os.makedirs(_FIGS, exist_ok=True)

# ── Material model (mirrors biofilm_3tooth_assembly.py) ────────────────────────
E_MAX = 10.0
E_MIN = 0.5
ALPHA = 2.0
DI_SCALE = 0.025778

CONDS = [
    {"key": "dh_baseline", "label": "DH-baseline\n(dysbiotic)", "color": "#d62728"},
    {"key": "commensal_static", "label": "Commensal-static", "color": "#2ca02c"},
    {"key": "Dysbiotic_Static", "label": "Dysbiotic-static", "color": "#ff7f0e"},
    {"key": "Commensal_HOBIC", "label": "Commensal-HOBIC", "color": "#1f77b4"},
]
TEETH = ["T23", "T30", "T31"]


def load_elem_csv(path):
    """Load odb_elements CSV → dict of numpy arrays."""
    if not os.path.isfile(path):
        return None
    mises, bins, teeth = [], [], []
    with open(path) as f:
        header = f.readline().strip().split(",")
        im = header.index("mises")
        ib = header.index("bin")
        it = header.index("tooth")
        for line in f:
            p = line.strip().split(",")
            if len(p) <= max(im, ib, it):
                continue
            mises.append(float(p[im]))
            bins.append(int(p[ib]))
            teeth.append(p[it].strip())
    return {"mises": np.array(mises), "bin": np.array(bins), "tooth": np.array(teeth)}


def load_di_late(cond_key):
    """Load abaqus_field_{cond}_late.csv → (di, r_pg, t) arrays."""
    path = os.path.join(_HERE, f"abaqus_field_{cond_key}_late.csv")
    if not os.path.isfile(path):
        return None, None, None
    di, r_pg = [], []
    t_val = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("x,"):
                continue
            p = line.split(",")
            if len(p) >= 8:
                di.append(float(p[4]))
                r_pg.append(float(p[6]))
                if t_val is None:
                    t_val = float(p[7])
    return np.array(di), np.array(r_pg), t_val


def di_to_eeff(di):
    r = np.clip(di / DI_SCALE, 0.0, 1.0)
    return E_MAX * (1.0 - r) ** ALPHA + E_MIN * r


# ── Load all data ──────────────────────────────────────────────────────────────
print("=" * 62)
print("  compare_conditions_late.py")
print("  Late-time (t≈0.5) 4-condition FEM comparison")
print("=" * 62)

elem_data = {}
di_data = {}

print("\n[Loading data]")
for c in CONDS:
    key = c["key"]
    ep = os.path.join(_HERE, f"odb_elements_{key}_late.csv")
    ed = load_elem_csv(ep)
    elem_data[key] = ed
    di, r_pg, t = load_di_late(key)
    di_data[key] = {"di": di, "r_pg": r_pg, "t": t}
    n_elem = len(ed["mises"]) if ed else 0
    n_di = len(di) if di is not None else 0
    print(f"  {key:30s}  elems={n_elem:6d}  di_pts={n_di}  t={t}")


# ── Fig 1: MISES violin per tooth, 4 conditions ────────────────────────────────
print("\n[LateFig1] MISES violin (late) ...")

fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=False)
fig.suptitle(
    "Late-time FEM: von Mises Stress per Tooth (t ≈ 0.5)\n"
    "1 MPa inward pressure · 437,472 C3D4 elements per condition",
    fontsize=12,
    fontweight="bold",
)

summary = {}  # key → tooth → (median, max)

for ax, tooth in zip(axes, TEETH):
    vals, colors, labels = [], [], []
    for c in CONDS:
        key = c["key"]
        ed = elem_data[key]
        if ed is None:
            continue
        m = ed["mises"][ed["tooth"] == tooth]
        vals.append(m)
        colors.append(c["color"])
        labels.append(c["label"])
        summary.setdefault(key, {})[tooth] = (float(np.median(m)), float(m.max()))

    xs = np.arange(len(vals))
    vp = ax.violinplot(vals, positions=xs, widths=0.55, showmedians=True, showextrema=True)
    for body, col in zip(vp["bodies"], colors):
        body.set_facecolor(col)
        body.set_alpha(0.55)
    vp["cmedians"].set_color("black")
    vp["cmedians"].set_linewidth(2)
    for part in ["cbars", "cmins", "cmaxes"]:
        vp[part].set_color("gray")

    # Annotate medians
    for i, (v, col) in enumerate(zip(vals, colors)):
        ax.text(
            xs[i],
            np.median(v) + 0.01,
            f"{np.median(v):.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
            color=col,
            fontweight="bold",
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("von Mises stress (MPa)", fontsize=9)
    ax.set_title(f"Tooth {tooth}", fontsize=11)
    ax.grid(axis="y", alpha=0.3)

handles = [mpatches.Patch(color=c["color"], label=c["label"].replace("\n", " ")) for c in CONDS]
fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=[0, 0.06, 1, 1])
out1 = os.path.join(_FIGS, "LateFig1_mises_violin.png")
fig.savefig(out1, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {out1}")


# ── Fig 2: Late DI comparison (histogram + boxplot) ────────────────────────────
print("\n[LateFig2] Late DI comparison ...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    "Late-time Dysbiotic Index (DI) Field  —  t ≈ 0.5\n"
    "(n = 3375 spatial grid points per condition)",
    fontsize=12,
    fontweight="bold",
)

ax_h, ax_b = axes
bins_di = np.linspace(0.0, 0.55, 80)

di_lists, di_labels, di_colors = [], [], []
for c in CONDS:
    key = c["key"]
    di = di_data[key]["di"]
    t = di_data[key]["t"]
    if di is None:
        continue
    col = c["color"]
    lbl = c["label"].replace("\n", " ")
    ax_h.hist(
        di,
        bins=bins_di,
        color=col,
        alpha=0.55,
        density=True,
        label=f"{lbl}  (μ={di.mean():.4f})",
        edgecolor="none",
    )
    ax_h.axvline(di.mean(), color=col, lw=1.5, ls="--", alpha=0.9)
    di_lists.append(di)
    di_labels.append(c["label"])
    di_colors.append(col)
    print(
        f"  {key:30s}  t={t:.2f}  DI mean={di.mean():.4f}  max={di.max():.4f}"
        f"  r_pg mean={di_data[key]['r_pg'].mean():.4f}"
    )

ax_h.set_xlabel("Dysbiotic Index (DI)", fontsize=10)
ax_h.set_ylabel("Probability density", fontsize=10)
ax_h.set_title("DI histogram (dashed = mean)", fontsize=10)
ax_h.legend(fontsize=8)
ax_h.grid(alpha=0.3)

bp = ax_b.boxplot(di_lists, patch_artist=True, notch=False, medianprops=dict(color="black", lw=2))
for patch, col in zip(bp["boxes"], di_colors):
    patch.set_facecolor(col)
    patch.set_alpha(0.65)
ax_b.set_xticklabels(di_labels, fontsize=8)
ax_b.set_ylabel("DI", fontsize=10)
ax_b.set_title("DI boxplot (late, t≈0.5)", fontsize=10)
ax_b.grid(axis="y", alpha=0.3)

fig.tight_layout()
out2 = os.path.join(_FIGS, "LateFig2_di_comparison.png")
fig.savefig(out2, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {out2}")


# ── Fig 3: E_eff distribution (late) ──────────────────────────────────────────
print("\n[LateFig3] E_eff distribution (late) ...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    r"Effective Young's Modulus $E_\mathrm{eff}$ — Late-time DI"
    "\n"
    r"$E_\mathrm{eff} = 10(1-r)^2 + 0.5r$,  $r = \min(\mathrm{DI}/0.025778,\,1)$",
    fontsize=11,
    fontweight="bold",
)

ax_h, ax_s = axes
bins_e = np.linspace(E_MIN - 0.1, E_MAX + 0.1, 70)

for c in CONDS:
    key = c["key"]
    di = di_data[key]["di"]
    if di is None:
        continue
    eeff = di_to_eeff(di)
    col = c["color"]
    lbl = c["label"].replace("\n", " ")
    ax_h.hist(
        eeff,
        bins=bins_e,
        color=col,
        alpha=0.5,
        density=True,
        label=f"{lbl}  (μ={eeff.mean():.2f} MPa)",
        edgecolor="none",
    )
    ax_h.axvline(eeff.mean(), color=col, lw=1.5, ls="--", alpha=0.9)
    print(
        f"  {key:30s}  E_eff mean={eeff.mean():.3f}  median={np.median(eeff):.3f}"
        f"  min={eeff.min():.3f}  max={eeff.max():.3f}"
    )

ax_h.set_xlabel("E_eff (MPa)", fontsize=10)
ax_h.set_ylabel("Probability density", fontsize=10)
ax_h.set_title("E_eff histogram (late DI)", fontsize=10)
ax_h.legend(fontsize=8)
ax_h.grid(alpha=0.3)

# DI vs E_eff scatter for dh_baseline and commensal_static
for c in CONDS[:2]:
    key = c["key"]
    di = di_data[key]["di"]
    if di is None:
        continue
    eeff = di_to_eeff(di)
    ax_s.scatter(di, eeff, s=5, c=c["color"], alpha=0.4, label=c["label"].replace("\n", " "))

di_curve = np.linspace(0.0, 0.55, 300)
ax_s.plot(di_curve, di_to_eeff(di_curve), "k-", lw=1.2, label="Model curve")
ax_s.axvline(DI_SCALE, color="gray", ls=":", lw=1, label=f"DI_scale={DI_SCALE:.4f}")
ax_s.set_xlabel("DI", fontsize=10)
ax_s.set_ylabel("E_eff (MPa)", fontsize=10)
ax_s.set_title("DI → E_eff mapping\n(DH-baseline vs Commensal-static)", fontsize=10)
ax_s.legend(fontsize=8)
ax_s.grid(alpha=0.3)

fig.tight_layout()
out3 = os.path.join(_FIGS, "LateFig3_eeff_comparison.png")
fig.savefig(out3, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {out3}")


# ── Fig 4: ΔMISES per tooth (dh_baseline vs each other condition) ──────────────
print("\n[LateFig4] ΔMISES = DH-baseline minus each condition ...")

ref_key = "dh_baseline"
ref_ed = elem_data[ref_key]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(
    "ΔMISES = DH-baseline − other condition (per DI bin, per tooth)\n"
    "Positive (red) → DH-baseline is MORE stressed; Negative (blue) → LESS stressed",
    fontsize=11,
    fontweight="bold",
)

for ax, tooth in zip(axes, TEETH):
    ref_m = ref_ed["mises"][ref_ed["tooth"] == tooth]
    ref_b = ref_ed["bin"][ref_ed["tooth"] == tooth]
    bins_uniq = sorted(set(ref_b))

    xs_offset = np.arange(len(bins_uniq))
    width = 0.25
    offsets = [-0.25, 0.0, 0.25]

    for oi, c in enumerate([c for c in CONDS if c["key"] != ref_key]):
        key = c["key"]
        ed = elem_data[key]
        if ed is None:
            continue
        delta = []
        for b in bins_uniq:
            rm = ref_m[ref_b == b]
            cm = ed["mises"][(ed["tooth"] == tooth) & (ed["bin"] == b)]
            if len(rm) > 0 and len(cm) > 0:
                delta.append(np.median(rm) - np.median(cm))
            else:
                delta.append(0.0)
        bar_colors = ["#d62728" if v > 0 else "#1f77b4" for v in delta]
        ax.bar(
            xs_offset + offsets[oi],
            delta,
            width,
            color=bar_colors,
            alpha=0.75,
            label=c["label"].replace("\n", " "),
        )

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(xs_offset)
    ax.set_xticklabels(bins_uniq, fontsize=7)
    ax.set_xlabel("DI bin", fontsize=9)
    ax.set_ylabel("Δ MISES (MPa)", fontsize=9)
    ax.set_title(f"Tooth {tooth}", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    if tooth == "T23":
        ax.legend(fontsize=7)

fig.tight_layout()
out4 = os.path.join(_FIGS, "LateFig4_delta_mises.png")
fig.savefig(out4, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {out4}")


# ── Fig 5: Summary table ───────────────────────────────────────────────────────
print("\n[LateFig5] Summary table ...")

fig, ax = plt.subplots(figsize=(14, 5))
ax.axis("off")
fig.suptitle(
    "Late-time FEM Summary  (t ≈ 0.5, 1 MPa inward pressure)", fontsize=12, fontweight="bold"
)

col_labels = [
    "Condition",
    "t",
    "DI_mean",
    "DI_max",
    "r_pg_mean",
    "E_eff_mean\n(MPa)",
    "T23 MISES\nmedian (MPa)",
    "T30 MISES\nmedian (MPa)",
    "T31 MISES\nmedian (MPa)",
]
rows = []
for c in CONDS:
    key = c["key"]
    di = di_data[key]["di"]
    t = di_data[key]["t"]
    rpg = di_data[key]["r_pg"]
    ed = elem_data[key]
    eeff = di_to_eeff(di) if di is not None else np.array([float("nan")])

    m_t23 = np.median(ed["mises"][ed["tooth"] == "T23"]) if ed else float("nan")
    m_t30 = np.median(ed["mises"][ed["tooth"] == "T30"]) if ed else float("nan")
    m_t31 = np.median(ed["mises"][ed["tooth"] == "T31"]) if ed else float("nan")

    rows.append(
        [
            c["label"].replace("\n", " "),
            f"{t:.2f}" if t else "—",
            f"{di.mean():.4f}" if di is not None else "—",
            f"{di.max():.4f}" if di is not None else "—",
            f"{rpg.mean():.4f}" if rpg is not None else "—",
            f"{eeff.mean():.3f}",
            f"{m_t23:.4f}",
            f"{m_t30:.4f}",
            f"{m_t31:.4f}",
        ]
    )
    print(
        f"  {key:28s}  DI={di.mean():.4f}  E={eeff.mean():.3f}"
        f"  T23={m_t23:.4f}  T30={m_t30:.4f}  T31={m_t31:.4f}"
    )

tbl = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.0, 2.0)
for j in range(len(col_labels)):
    tbl[0, j].set_facecolor("#2c3e50")
    tbl[0, j].set_text_props(color="white", fontweight="bold")
for i, c in enumerate(CONDS):
    tbl[i + 1, 0].set_facecolor(c["color"])
    tbl[i + 1, 0].set_text_props(color="white", fontweight="bold")
    for j in range(1, len(col_labels)):
        tbl[i + 1, j].set_facecolor(c["color"] + "22")

fig.tight_layout()
out5 = os.path.join(_FIGS, "LateFig5_summary_table.png")
fig.savefig(out5, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {out5}")


# ── Done ───────────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  Done.  5 figures written to figures/")
for out in [out1, out2, out3, out4, out5]:
    print(f"    {os.path.basename(out)}")
print("=" * 62)
