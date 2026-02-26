#!/usr/bin/env python3
"""
compare_conditions_fem.py  --  P0 / P7  Condition Comparison
=============================================================

Compares FEM stress/DI results across biological conditions.

Data sources (used in order of availability):
  [A] _posterior_abaqus/{cond}/sample_*/stress.json   -- 20-sample ensemble
  [B] abaqus_field_{cond}_snap20.csv                  -- DI field (3375 pts)
  [C] odb_elements.csv + odb_elements_{cond}.csv      -- 3-tooth ODB [optional]

Outputs
-------
  figures/CompFig1_mises_violin.png      -- posterior ensemble Mises (substrate + surface)
  figures/CompFig2_di_histogram.png      -- DI field distribution per condition
  figures/CompFig3_eeff_distribution.png -- E_eff from DI model per condition
  figures/CompFig4_3tooth_comparison.png -- 3-tooth MISES overlay (if both ODBs present)
                                            OR summary statistics table (fallback)

Usage
-----
  python3 compare_conditions_fem.py                    # use all available data
  python3 compare_conditions_fem.py --no-3tooth        # skip 3-tooth ODB mode
  python3 compare_conditions_fem.py --snapshot 40      # use different snapshot for DI

Material model (mirrors biofilm_3tooth_assembly.py)
---------------------------------------------------
  E_eff = E_max*(1 - r)^alpha + E_min*r
  r     = DI / DI_scale   (clamped to [0, 1])
  E_max = 10 MPa,  E_min = 0.5 MPa,  alpha = 2.0,  DI_scale = 0.025778
"""

from __future__ import print_function, division
import os
import json
import argparse

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_POST = os.path.join(_HERE, "_posterior_abaqus")
_FIGS = os.path.join(_HERE, "figures")
os.makedirs(_FIGS, exist_ok=True)

# ── Material model constants ───────────────────────────────────────────────────
E_MAX = 10.0  # MPa (fixed from literature)
E_MIN = 0.5  # MPa (fixed from literature)
ALPHA = 2.0  # DI exponent (tunable)
DI_SCALE = 0.025778  # DI scale (tunable)

# ── Condition registry ─────────────────────────────────────────────────────────
# Maps internal name → display label, plot color, DI field CSV filename
COND_INFO = {
    "dh_baseline": {
        "label": "DH-baseline\n(dysbiotic cascade)",
        "color": "#d62728",  # red
        "di_csv_candidates": [
            "abaqus_field_dh_baseline_snap20.csv",
            "abaqus_field_dh_3d.csv",
        ],
    },
    "commensal_static": {
        "label": "Commensal-static\n(balanced)",
        "color": "#2ca02c",  # green
        "di_csv_candidates": [
            "abaqus_field_commensal_static_snap20.csv",
            "abaqus_field_Commensal_Static_snap20.csv",
        ],
    },
    "dysbiotic_static": {
        "label": "Dysbiotic-static",
        "color": "#ff7f0e",  # orange
        "di_csv_candidates": [
            "abaqus_field_Dysbiotic_Static_snap20.csv",
        ],
    },
    "commensal_hobic": {
        "label": "Commensal-HOBIC",
        "color": "#1f77b4",  # blue
        "di_csv_candidates": [
            "abaqus_field_Commensal_HOBIC_snap20.csv",
        ],
    },
}

COND_ORDER = ["dh_baseline", "commensal_static", "dysbiotic_static", "commensal_hobic"]


# ── Helpers ────────────────────────────────────────────────────────────────────


def _find_di_csv(cond):
    for fname in COND_INFO[cond]["di_csv_candidates"]:
        p = os.path.join(_HERE, fname)
        if os.path.isfile(p):
            return p
    return None


def load_posterior_stress(cond):
    """Return (substrate_mpa, surface_mpa) arrays from sample stress.json files."""
    d = os.path.join(_POST, cond)
    if not os.path.isdir(d):
        return None, None
    sub, sur = [], []
    for s in sorted(os.listdir(d)):
        sj = os.path.join(d, s, "stress.json")
        if os.path.isfile(sj):
            j = json.load(open(sj))
            if "substrate_smises" in j and "surface_smises" in j:
                sub.append(j["substrate_smises"] / 1e6)
                sur.append(j["surface_smises"] / 1e6)
    if not sub:
        return None, None
    return np.array(sub), np.array(sur)


def load_di_field(cond):
    """Return (di, phi_pg, r_pg) 1-D arrays from abaqus_field CSV."""
    csv = _find_di_csv(cond)
    if csv is None:
        return None, None, None
    di, phi_pg, r_pg = [], [], []
    with open(csv) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("x,"):
                continue
            parts = line.split(",")
            if len(parts) >= 8:
                di.append(float(parts[4]))
                phi_pg.append(float(parts[3]))
                r_pg.append(float(parts[6]))
    return np.array(di), np.array(phi_pg), np.array(r_pg)


def di_to_eeff(di):
    """Apply material model: E_eff = E_max*(1-r)^alpha + E_min*r."""
    r = np.clip(di / DI_SCALE, 0.0, 1.0)
    return E_MAX * (1.0 - r) ** ALPHA + E_MIN * r


def load_odb_elements(path):
    """Load odb_elements CSV: returns dict with arrays mises, bin, tooth."""
    if not os.path.isfile(path):
        return None
    mises, bins, teeth = [], [], []
    with open(path) as f:
        header = f.readline().strip().split(",")
        idx_m = header.index("mises")
        idx_b = header.index("bin")
        idx_t = header.index("tooth")
        for line in f:
            p = line.strip().split(",")
            if len(p) <= max(idx_m, idx_b, idx_t):
                continue
            mises.append(float(p[idx_m]))
            bins.append(int(p[idx_b]))
            teeth.append(p[idx_t])
    return {
        "mises": np.array(mises),
        "bin": np.array(bins),
        "tooth": np.array(teeth),
    }


# ── Figure 1: Posterior ensemble Mises violin ──────────────────────────────────


def fig1_mises_violin(save=True):
    print("\n[CompFig1] Posterior ensemble Mises stress comparison ...")

    available = []
    data_sub, data_sur = {}, {}
    for cond in COND_ORDER:
        sub, sur = load_posterior_stress(cond)
        if sub is not None:
            available.append(cond)
            data_sub[cond] = sub
            data_sur[cond] = sur
            info = COND_INFO[cond]
            print(
                "  %-20s  n=%d  sub=%.4f±%.4f MPa  sur=%.4f±%.4f MPa"
                % (cond, len(sub), np.median(sub), sub.std(), np.median(sur), sur.std())
            )

    if not available:
        print("  No posterior stress data found — skipping CompFig1.")
        return

    n = len(available)
    xs = np.arange(n)
    w = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=False)
    fig.suptitle(
        "Posterior Ensemble: von Mises Stress by Condition\n"
        "(20 posterior samples per condition, cube-model geometry)",
        fontsize=12,
        fontweight="bold",
    )

    for ax, key, title in zip(
        axes, [data_sub, data_sur], ["Substrate (inner, depth=0)", "Surface (outer, depth=0.5)"]
    ):
        vals = [key[c] for c in available]
        colors = [COND_INFO[c]["color"] for c in available]

        vp = ax.violinplot(vals, positions=xs, widths=w * 1.6, showmedians=True, showextrema=True)
        for body, col in zip(vp["bodies"], colors):
            body.set_facecolor(col)
            body.set_alpha(0.6)
        vp["cmedians"].set_color("black")
        vp["cmedians"].set_linewidth(2)
        for part in ["cbars", "cmins", "cmaxes"]:
            vp[part].set_color("gray")

        # overlay individual points (jittered)
        rng = np.random.default_rng(0)
        for i, (v, col) in enumerate(zip(vals, colors)):
            jitter = rng.uniform(-0.08, 0.08, size=len(v))
            ax.scatter(
                xs[i] + jitter,
                v,
                s=18,
                color=col,
                alpha=0.7,
                edgecolors="white",
                linewidths=0.4,
                zorder=3,
            )

        ax.set_xticks(xs)
        ax.set_xticklabels([COND_INFO[c]["label"] for c in available], fontsize=9)
        ax.set_ylabel("von Mises stress (MPa)", fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))

    # Annotate p-values (Wilcoxon) if scipy available
    try:
        from scipy.stats import ranksums

        ax = axes[0]
        dh = data_sub.get("dh_baseline")
        cs = data_sub.get("commensal_static")
        if dh is not None and cs is not None:
            _, p = ranksums(dh, cs)
            ax.text(
                0.5,
                0.97,
                "DH vs CS: p=%.3f" % p,
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=8,
                color="gray",
            )
    except ImportError:
        pass

    fig.tight_layout()
    out = os.path.join(_FIGS, "CompFig1_mises_violin.png")
    if save:
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print("  Saved: %s" % out)
    plt.close(fig)
    return out


# ── Figure 2: DI field histogram ───────────────────────────────────────────────


def fig2_di_histogram(save=True):
    print("\n[CompFig2] DI field distribution comparison ...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Dysbiotic Index (DI) Field Distribution\n"
        "(snapshot 20, t=0.05, 3375 spatial points per condition)",
        fontsize=12,
        fontweight="bold",
    )

    ax_kde, ax_box = axes

    handles = []
    stats_rows = []
    bins_hist = np.linspace(0.0, 0.03, 50)

    for cond in COND_ORDER:
        di, phi_pg, r_pg = load_di_field(cond)
        if di is None:
            continue
        col = COND_INFO[cond]["color"]
        label = COND_INFO[cond]["label"].replace("\n", " ")
        print(
            "  %-20s  n=%d  DI mean=%.4f  median=%.4f  max=%.4f"
            % (cond, len(di), di.mean(), np.median(di), di.max())
        )

        # KDE via histogram
        ax_kde.hist(
            di, bins=bins_hist, color=col, alpha=0.55, density=True, label=label, edgecolor="none"
        )

        # Vertical median line
        ax_kde.axvline(np.median(di), color=col, lw=1.5, ls="--", alpha=0.85)

        handles.append(mpatches.Patch(color=col, label=label))
        stats_rows.append((cond, len(di), di.mean(), np.median(di), di.std(), di.max()))

    ax_kde.set_xlabel("Dysbiotic Index (DI)", fontsize=10)
    ax_kde.set_ylabel("Probability density", fontsize=10)
    ax_kde.set_title("DI histogram (dashed = median)", fontsize=10)
    ax_kde.legend(handles=handles, fontsize=8, loc="upper right")
    ax_kde.grid(alpha=0.3)

    # Boxplot panel
    avail = [c for c in COND_ORDER if _find_di_csv(c)]
    di_list = []
    di_labels = []
    di_colors = []
    for cond in avail:
        di, _, _ = load_di_field(cond)
        if di is not None:
            di_list.append(di)
            di_labels.append(COND_INFO[cond]["label"])
            di_colors.append(COND_INFO[cond]["color"])

    if di_list:
        bp = ax_box.boxplot(
            di_list, patch_artist=True, notch=False, medianprops=dict(color="black", lw=2)
        )
        for patch, col in zip(bp["boxes"], di_colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.65)
        ax_box.set_xticklabels(di_labels, fontsize=8)
        ax_box.set_ylabel("Dysbiotic Index (DI)", fontsize=10)
        ax_box.set_title("DI boxplot per condition", fontsize=10)
        ax_box.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = os.path.join(_FIGS, "CompFig2_di_histogram.png")
    if save:
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print("  Saved: %s" % out)
    plt.close(fig)
    return out


# ── Figure 3: E_eff distribution ───────────────────────────────────────────────


def fig3_eeff_distribution(save=True):
    print("\n[CompFig3] E_eff distribution comparison ...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Effective Young's Modulus (E_eff) Distribution\n"
        r"$E_\mathrm{eff} = E_\mathrm{max}(1-r)^\alpha + E_\mathrm{min}\,r$,"
        "  r = DI / %.6f,  E_max=%.1f MPa,  E_min=%.1f MPa,  α=%.1f"
        % (DI_SCALE, E_MAX, E_MIN, ALPHA),
        fontsize=10,
        fontweight="bold",
    )

    ax_hist, ax_box = axes
    handles = []
    eeff_all = {}
    bins_e = np.linspace(E_MIN - 0.05, E_MAX + 0.1, 60)

    for cond in COND_ORDER:
        di, _, _ = load_di_field(cond)
        if di is None:
            continue
        eeff = di_to_eeff(di)
        eeff_all[cond] = eeff
        col = COND_INFO[cond]["color"]
        lbl = COND_INFO[cond]["label"].replace("\n", " ")
        print(
            "  %-20s  E_eff mean=%.4f MPa  median=%.4f MPa  min=%.4f  max=%.4f"
            % (cond, eeff.mean(), np.median(eeff), eeff.min(), eeff.max())
        )

        ax_hist.hist(
            eeff, bins=bins_e, color=col, alpha=0.5, density=True, label=lbl, edgecolor="none"
        )
        ax_hist.axvline(np.median(eeff), color=col, lw=1.5, ls="--", alpha=0.85)
        handles.append(mpatches.Patch(color=col, label=lbl))

    ax_hist.set_xlabel("E_eff (MPa)", fontsize=10)
    ax_hist.set_ylabel("Probability density", fontsize=10)
    ax_hist.set_title("E_eff histogram (dashed = median)", fontsize=10)
    ax_hist.legend(handles=handles, fontsize=8)
    ax_hist.grid(alpha=0.3)

    # Scatter: DI vs E_eff for two key conditions
    for cond in ["dh_baseline", "commensal_static"]:
        di, _, _ = load_di_field(cond)
        if di is None:
            continue
        eeff = di_to_eeff(di)
        col = COND_INFO[cond]["color"]
        lbl = COND_INFO[cond]["label"].replace("\n", " ")
        ax_box.scatter(di, eeff, s=6, c=col, alpha=0.5, label=lbl)

    di_plot = np.linspace(0.0, DI_SCALE * 1.1, 200)
    eeff_curve = di_to_eeff(di_plot)
    ax_box.plot(di_plot, eeff_curve, "k-", lw=1.2, label="Model curve")
    ax_box.set_xlabel("DI", fontsize=10)
    ax_box.set_ylabel("E_eff (MPa)", fontsize=10)
    ax_box.set_title("DI → E_eff mapping\n(DH-baseline vs Commensal-static)", fontsize=10)
    ax_box.legend(fontsize=8)
    ax_box.grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(_FIGS, "CompFig3_eeff_distribution.png")
    if save:
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print("  Saved: %s" % out)
    plt.close(fig)
    return out


# ── Figure 4: 3-tooth ODB comparison OR summary table ─────────────────────────


def fig4_comparison(skip_3tooth=False, save=True):
    print("\n[CompFig4] 3-tooth comparison / summary ...")

    # Try to load 3-tooth ODB CSVs
    odb_baseline = os.path.join(_HERE, "odb_elements.csv")
    odb_commensal = os.path.join(_HERE, "odb_elements_commensal_static.csv")

    has_baseline = os.path.isfile(odb_baseline)
    has_commensal = os.path.isfile(odb_commensal)

    if has_baseline and has_commensal and not skip_3tooth:
        return _fig4_odb_overlay(odb_baseline, odb_commensal, save)
    else:
        if not skip_3tooth:
            if has_baseline and not has_commensal:
                print(
                    "  Note: odb_elements.csv found but odb_elements_commensal_static.csv missing."
                )
                print("  Run Abaqus for commensal_static first:")
                print(
                    "    abaqus job=BioFilm3T_commensal_static "
                    "input=biofilm_3tooth_commensal_static.inp cpus=4 interactive"
                )
                print("  Then extract: abaqus python odb_extract.py BioFilm3T_commensal_static.odb")
                print("  Falling back to summary table ...")
        return _fig4_summary_table(odb_baseline if has_baseline else None, save)


def _fig4_odb_overlay(path_b, path_cs, save):
    """Side-by-side 3-tooth MISES comparison."""
    print("  Loading 3-tooth ODB data for both conditions ...")
    db = load_odb_elements(path_b)
    dc = load_odb_elements(path_cs)
    if db is None or dc is None:
        print("  ERROR loading ODB CSVs.")
        return None

    teeth = ["T23", "T30", "T31"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(
        "3-Tooth FEM: MISES Comparison — DH-baseline vs Commensal-static\n"
        "(437,472 C3D4 elements, 1 MPa inward pressure)",
        fontsize=12,
        fontweight="bold",
    )

    bins_m = np.linspace(0, 2.0, 60)
    for col_i, tooth in enumerate(teeth):
        mb = db["mises"][db["tooth"] == tooth]
        mc = dc["mises"][dc["tooth"] == tooth]

        ax_top = axes[0, col_i]
        ax_bot = axes[1, col_i]

        # Histogram overlay
        ax_top.hist(mb, bins=bins_m, color="#d62728", alpha=0.5, density=True, label="DH-baseline")
        ax_top.hist(
            mc, bins=bins_m, color="#2ca02c", alpha=0.5, density=True, label="Commensal-static"
        )
        ax_top.axvline(np.median(mb), color="#d62728", lw=1.5, ls="--")
        ax_top.axvline(np.median(mc), color="#2ca02c", lw=1.5, ls="--")
        ax_top.set_title(tooth, fontsize=11)
        ax_top.set_xlabel("MISES (MPa)")
        ax_top.set_ylabel("Density")
        ax_top.legend(fontsize=8)
        ax_top.grid(alpha=0.3)

        # ΔMISES per bin
        bins_b = sorted(set(db["bin"][db["tooth"] == tooth]))
        delta_med = []
        bin_labels = []
        for b in bins_b:
            mb_b = db["mises"][(db["tooth"] == tooth) & (db["bin"] == b)]
            mc_b = dc["mises"][(dc["tooth"] == tooth) & (dc["bin"] == b)]
            if len(mb_b) > 0 and len(mc_b) > 0:
                delta_med.append(np.median(mb_b) - np.median(mc_b))
                bin_labels.append(b)
        if delta_med:
            bar_colors = ["#d62728" if v > 0 else "#2ca02c" for v in delta_med]
            ax_bot.bar(bin_labels, delta_med, color=bar_colors, alpha=0.75)
            ax_bot.axhline(0, color="black", lw=0.8)
            ax_bot.set_xlabel("DI bin")
            ax_bot.set_ylabel("Δ MISES (DH − CS) MPa")
            ax_bot.set_title("Δ MISES per DI bin — %s" % tooth, fontsize=9)
            ax_bot.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = os.path.join(_FIGS, "CompFig4_3tooth_comparison.png")
    if save:
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print("  Saved: %s" % out)
    plt.close(fig)
    return out


def _fig4_summary_table(odb_baseline_path, save):
    """Fallback: numeric summary table."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")
    fig.suptitle(
        "Condition Comparison — Summary Statistics\n"
        "(CompFig4: 3-tooth ODB comparison pending commensal Abaqus solve)",
        fontsize=12,
        fontweight="bold",
    )

    rows = []
    col_labels = [
        "Condition",
        "DI mean",
        "DI median",
        "DI max",
        "E_eff mean\n(MPa)",
        "E_eff median\n(MPa)",
        "Sub. MISES\nmedian (MPa)",
        "Sur. MISES\nmedian (MPa)",
        "N samples",
    ]

    for cond in COND_ORDER:
        di, _, _ = load_di_field(cond)
        sub, sur = load_posterior_stress(cond)
        if di is None and sub is None:
            continue
        eeff = di_to_eeff(di) if di is not None else np.array([float("nan")])

        row = [
            COND_INFO[cond]["label"].replace("\n", " "),
            "%.4f" % (di.mean() if di is not None else float("nan")),
            "%.4f" % (np.median(di) if di is not None else float("nan")),
            "%.4f" % (di.max() if di is not None else float("nan")),
            "%.3f" % (eeff.mean() if di is not None else float("nan")),
            "%.3f" % (np.median(eeff) if di is not None else float("nan")),
            "%.4f" % (np.median(sub) if sub is not None else float("nan")),
            "%.4f" % (np.median(sur) if sur is not None else float("nan")),
            "%d" % (len(sub) if sub is not None else 0),
        ]
        rows.append(row)
        print("  " + " | ".join(row))

    if not rows:
        ax.text(
            0.5,
            0.5,
            "No data available.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
        )
        fig.tight_layout()
    else:
        tbl = ax.table(
            cellText=rows,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.0, 1.8)

        # Color header
        for j in range(len(col_labels)):
            tbl[0, j].set_facecolor("#2c3e50")
            tbl[0, j].set_text_props(color="white", fontweight="bold")

        # Color rows by condition
        for i, cond in enumerate(
            [c for c in COND_ORDER if _find_di_csv(c) or os.path.isdir(os.path.join(_POST, c))]
        ):
            col = COND_INFO[cond]["color"]
            tbl[i + 1, 0].set_facecolor(col)
            tbl[i + 1, 0].set_text_props(color="white", fontweight="bold")
            for j in range(1, len(col_labels)):
                tbl[i + 1, j].set_facecolor(col + "22")

        # Footnote
        ax.text(
            0.5,
            0.02,
            "3-tooth MISES comparison pending: run Abaqus for biofilm_3tooth_commensal_static.inp → odb_elements_commensal_static.csv",
            ha="center",
            va="bottom",
            transform=ax.transAxes,
            fontsize=7.5,
            color="#555555",
            style="italic",
        )

    out = os.path.join(_FIGS, "CompFig4_summary_table.png")
    if save:
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print("  Saved: %s" % out)
    plt.close(fig)
    return out


# ── CLI ────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="P0/P7 Condition comparison: DH-baseline vs Commensal-static"
    )
    p.add_argument(
        "--no-3tooth", action="store_true", help="Skip 3-tooth ODB comparison even if CSVs exist"
    )
    p.add_argument(
        "--snapshot",
        type=int,
        default=20,
        help="Snapshot index used when regenerating DI CSVs [20]",
    )
    p.add_argument("--figs", default=None, help="Override output directory for figures")
    p.add_argument(
        "--di-exp", type=float, default=ALPHA, help="DI exponent alpha used in E_eff mapping [2.0]"
    )
    p.add_argument(
        "--di-scale",
        type=float,
        default=DI_SCALE,
        help="DI normalisation scale s_DI used in E_eff mapping [0.025778]",
    )
    return p.parse_args()


def main():
    args = parse_args()
    global _FIGS, ALPHA, DI_SCALE
    ALPHA = float(args.di_exp)
    DI_SCALE = float(args.di_scale)
    if args.figs:
        _FIGS = args.figs
        os.makedirs(_FIGS, exist_ok=True)

    print("=" * 62)
    print("  compare_conditions_fem.py  (P0 / P7)")
    print("  Output dir: %s" % _FIGS)
    print("=" * 62)

    # Check data availability
    print("\n[Data availability]")
    for cond in COND_ORDER:
        has_post = os.path.isdir(os.path.join(_POST, cond))
        di_csv = _find_di_csv(cond)
        print(
            "  %-22s  posterior=%s  di_csv=%s"
            % (cond, "✓" if has_post else "✗", os.path.basename(di_csv) if di_csv else "✗")
        )
    print(
        "  odb_elements.csv              : %s"
        % ("✓" if os.path.isfile(os.path.join(_HERE, "odb_elements.csv")) else "✗")
    )
    print(
        "  odb_elements_commensal_static : %s"
        % (
            "✓"
            if os.path.isfile(os.path.join(_HERE, "odb_elements_commensal_static.csv"))
            else "✗ (need Abaqus)"
        )
    )

    out1 = fig1_mises_violin()
    out2 = fig2_di_histogram()
    out3 = fig3_eeff_distribution()
    out4 = fig4_comparison(skip_3tooth=args.no_3tooth)

    print("\n" + "=" * 62)
    print("  Done.  Figures written:")
    for p in [out1, out2, out3, out4]:
        if p:
            print("    %s" % p)
    print()
    print("  Next step (P0 completion):")
    print("    abaqus job=BioFilm3T_commensal_static \\")
    print("           input=biofilm_3tooth_commensal_static.inp \\")
    print("           cpus=4 ask=off interactive")
    print("    abaqus python odb_extract.py BioFilm3T_commensal_static.odb")
    print("    # → renames output to odb_elements_commensal_static.csv")
    print("    python3 compare_conditions_fem.py   # re-run → full CompFig4")
    print("=" * 62)


if __name__ == "__main__":
    main()
