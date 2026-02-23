#!/usr/bin/env python3
"""
plot_czm_analytical.py
======================
B3 (analytical): CZM interface properties from B1 DI fields.

Abaqus CAE runs crash on this system (API incompatibility).
This script computes all key CZM quantities directly from the
p05/p50/p95 DI quantile arrays produced by aggregate_di_credible.py,
and generates the same figures that run_czm3d_sweep.py would produce.

Physical model
--------------
Interface cohesive properties at the biofilm-substrate interface:

  r(DI)     = clamp(DI / s, 0, 1)       s = DI_SCALE = 0.025778
  t_max(DI) = t_max_0 * (1 - r)^n       t_max_0 = 1.0 MPa
  G_c(DI)   = G_c_0  * (1 - r)^n       G_c_0   = 10.0 J/m^2

Approximation: RF_peak ≈ t_max_eff × A_interface
  (exact for uniform properties, lower bound for DI-heterogeneous interface)
  A_interface = Lx × Ly = 1.0 × 1.0 = 1.0 m^2 (normalised domain)

Outputs  (_czm3d/):
  czm_analytical.csv
  figures/fig_B3_czm_summary.png
  figures/fig_B3_czm_4panel.png
  figures/fig_B3_czm_posterior_bands.png
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── paths ───────────────────────────────────────────────────────────────────
_HERE    = Path(__file__).resolve().parent
_DI_BASE = _HERE / "_di_credible"
_PA_BASE = _HERE / "_posterior_abaqus"
_OUT     = _HERE / "_czm3d"
_OUT.mkdir(exist_ok=True)
(_OUT / "figures").mkdir(exist_ok=True)

# ── CZM physics constants ────────────────────────────────────────────────────
T_MAX_0     = 1.0e6    # Pa   nominal normal strength
GC_MAX      = 10.0     # J/m² fracture energy baseline
DI_SCALE    = 0.025778
DI_EXPONENT = 2.0
N_LAYERS_CZ = 3        # bottom layers for interface DI mean
A_INTERFACE = 1.0      # m² (normalised domain Lx=Ly=1)

CONDITIONS = ["dh_baseline", "commensal_static", "commensal_hobic", "dysbiotic_static"]
LABELS = {
    "dh_baseline":      "dh-baseline",
    "commensal_static": "Comm. Static",
    "commensal_hobic":  "Comm. HOBIC",
    "dysbiotic_static": "Dysb. Static",
}
COLORS = {
    "dh_baseline":      "#d62728",
    "commensal_static": "#2ca02c",
    "commensal_hobic":  "#1f77b4",
    "dysbiotic_static": "#ff7f0e",
}
QTAGS = ["p05", "p50", "p95"]
QTAG_ALPHA = {"p05": 0.35, "p50": 1.0, "p95": 0.35}

# ── helpers ──────────────────────────────────────────────────────────────────
def _czm_from_di(di_mean):
    r       = np.clip(di_mean / DI_SCALE, 0.0, 1.0)
    factor  = (1.0 - r) ** DI_EXPONENT
    t_max   = T_MAX_0 * factor
    gc      = GC_MAX  * factor
    rf_peak = t_max * A_INTERFACE        # N (normalised domain)
    return t_max, gc, rf_peak


def _di_bottom_layers(di_quantile_npy, n_layers=N_LAYERS_CZ):
    """
    di_quantile_npy: (3375,) or (N,) array of DI values at each node.
    Assumes 15^3 grid; bottom N z-slices (z lowest).
    Returns mean DI of the bottom layers.
    """
    q = np.asarray(di_quantile_npy).reshape(15, 15, 15)  # (x, y, z)
    return float(q[:, :, :n_layers].mean())


# ── load B1 DI data ──────────────────────────────────────────────────────────
def load_b1():
    """
    Returns dict:
      { cond -> { qtag -> {"di_mean": float, "t_max": float, "gc": float, "rf_peak": float} } }
    Also returns full DI stacks for posterior bands:
      { cond -> (20, 3375) di_stack }
    """
    results   = {}
    di_stacks = {}

    for cond in CONDITIONS:
        d = _DI_BASE / cond
        results[cond] = {}

        # p05/p50/p95 from precomputed quantile arrays
        q_arr = np.load(d / "di_quantiles.npy")  # (3, 3375)
        for qi, qtag in enumerate(QTAGS):
            di_mean = _di_bottom_layers(q_arr[qi])
            t_max, gc, rf_peak = _czm_from_di(di_mean)
            results[cond][qtag] = {
                "di_mean":  di_mean,
                "t_max":    t_max,
                "gc":       gc,
                "rf_peak":  rf_peak,
            }

        # full stack (20 samples) for posterior bands
        di_stack = np.load(d / "di_stack.npy")   # (20, 3375) or (3375, 20)
        if di_stack.ndim == 2 and di_stack.shape[0] != 20:
            di_stack = di_stack.T
        di_stacks[cond] = di_stack

    return results, di_stacks


def compute_posterior_czm(di_stacks):
    """
    For each condition, compute CZM metrics for all 20 posterior samples.
    Returns { cond -> {"di_mean": (20,), "t_max": (20,), "gc": (20,), "rf_peak": (20,)} }
    """
    post = {}
    for cond, stack in di_stacks.items():
        n = stack.shape[0]
        di_means  = np.array([_di_bottom_layers(stack[i]) for i in range(n)])
        r         = np.clip(di_means / DI_SCALE, 0.0, 1.0)
        factor    = (1.0 - r) ** DI_EXPONENT
        post[cond] = {
            "di_mean":  di_means,
            "t_max":    T_MAX_0 * factor,
            "gc":       GC_MAX  * factor,
            "rf_peak":  T_MAX_0 * factor * A_INTERFACE,
        }
    return post


# ── save CSV ─────────────────────────────────────────────────────────────────
def save_csv(results):
    import csv
    rows = []
    for cond in CONDITIONS:
        for qtag in QTAGS:
            r = results[cond][qtag]
            rows.append({
                "condition": cond, "di_qtag": qtag,
                "di_mean": r["di_mean"],
                "t_max_eff": r["t_max"],
                "gc_eff": r["gc"],
                "rf_peak": r["rf_peak"],
            })
    path = _OUT / "czm_analytical.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"[csv] {path.name}")


# ── Fig B3-1: Summary (2-panel) ───────────────────────────────────────────────
def fig_b3_summary(results, post):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    x = np.arange(len(CONDITIONS))
    w = 0.5

    # Panel left: RF_peak with p05-p95 error bars
    ax = axes[0]
    rf_p50 = [results[c]["p50"]["rf_peak"] for c in CONDITIONS]
    rf_p05 = [results[c]["p05"]["rf_peak"] for c in CONDITIONS]
    rf_p95 = [results[c]["p95"]["rf_peak"] for c in CONDITIONS]
    colors = [COLORS[c] for c in CONDITIONS]
    bars = ax.bar(x, rf_p50, width=w, color=colors, alpha=0.85,
                  edgecolor="k", linewidth=0.6, zorder=3)
    # DI↑ → RF↓: p05_DI gives highest RF (upper bound), p95_DI gives lowest
    rf_lo = np.maximum(0, np.array(rf_p50) - np.array(rf_p95))   # lower error
    rf_hi = np.maximum(0, np.array(rf_p05) - np.array(rf_p50))   # upper error
    ax.errorbar(x, rf_p50,
                yerr=[rf_lo, rf_hi],
                fmt="none", color="black", capsize=5, lw=1.5, zorder=4)
    # posterior IQR as scatter
    for xi, cond in enumerate(CONDITIONS):
        rf_post = post[cond]["rf_peak"]
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(rf_post))
        ax.scatter(xi + jitter, rf_post, color=COLORS[cond],
                   s=20, alpha=0.5, zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in CONDITIONS], fontsize=10,
                       rotation=10, ha="right")
    ax.set_ylabel("Peak reaction force (N)", fontsize=11)
    ax.set_title("B3: RF_peak = t_max_eff × A_interface\n(dots = posterior samples)", fontsize=10)
    ax.grid(axis="y", alpha=0.3, ls="--")
    ax.set_ylim(0)

    # Panel right: Gc_eff vs DI_mean scatter with theoretical curve
    ax = axes[1]
    di_range = np.linspace(0, max(results[c]["p95"]["di_mean"] for c in CONDITIONS) * 1.4, 200)
    r_range  = np.clip(di_range / DI_SCALE, 0, 1)
    gc_curve = GC_MAX * (1 - r_range) ** DI_EXPONENT
    ax.plot(di_range, gc_curve, "k--", lw=1.2, alpha=0.6,
            label=r"$G_c(DI)=G_{c,0}(1-DI/s)^n$")

    for cond in CONDITIONS:
        # posterior cloud
        di_post = post[cond]["di_mean"]
        gc_post = post[cond]["gc"]
        ax.scatter(di_post, gc_post, color=COLORS[cond], s=20, alpha=0.4, zorder=3)
        # p05/p50/p95 markers
        for qtag, marker, ms in [("p05","v",8),("p50","o",12),("p95","^",8)]:
            r = results[cond][qtag]
            ax.scatter(r["di_mean"], r["gc"],
                       color=COLORS[cond], s=ms**2, marker=marker,
                       edgecolors="k", linewidths=0.7, zorder=5)
        # label p50
        r50 = results[cond]["p50"]
        ax.annotate(LABELS[cond], (r50["di_mean"], r50["gc"]),
                    textcoords="offset points", xytext=(6,4), fontsize=8)
    ax.set_xlabel("Mean DI (bottom interface layers)", fontsize=11)
    ax.set_ylabel("$G_c$ (J/m²)", fontsize=11)
    ax.set_title("B3: Fracture Energy vs DI\n(▼=p05  ●=p50  ▲=p95,  cloud=posterior)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, ls="--")

    fig.suptitle(
        "B3: Cohesive Zone Model — Interface Strength vs Biofilm Condition\n"
        r"$t_\mathrm{max}(DI)=t_0(1-DI/s)^n$,  $G_c(DI)=G_{c,0}(1-DI/s)^n$",
        fontsize=12, fontweight="bold",
    )
    out = _OUT / "figures" / "fig_B3_czm_summary.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[done] {out.name}")


# ── Fig B3-2: 4-panel ────────────────────────────────────────────────────────
def fig_b3_4panel(results, post):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    x = np.arange(len(CONDITIONS))
    w = 0.55
    colors = [COLORS[c] for c in CONDITIONS]

    panels = [
        ("di_mean",  "Mean DI (interface)",   "B3: Interface DI",       "DI"),
        ("t_max",    "t_max_eff (Pa)",         "B3: Normal Strength",    "Pa"),
        ("gc",       "$G_c$ (J/m²)",           "B3: Fracture Energy",    "J/m²"),
        ("rf_peak",  "RF_peak (N)",            "B3: Peak Pull Force",    "N"),
    ]

    for ax, (key, ylabel, title, unit) in zip(axes.flat, panels):
        p50 = [results[c]["p50"][key] for c in CONDITIONS]
        p05 = [results[c]["p05"][key] for c in CONDITIONS]
        p95 = [results[c]["p95"][key] for c in CONDITIONS]

        bars = ax.bar(x, p50, width=w, color=colors, alpha=0.85,
                      edgecolor="k", linewidth=0.5, zorder=3)
        # DI↑→metric↓: swap p05/p95 for lower/upper error
        lo = np.maximum(0, np.array(p50) - np.array(p95))
        hi = np.maximum(0, np.array(p05) - np.array(p50))
        ax.errorbar(x, p50, yerr=[lo, hi],
                    fmt="none", color="black", capsize=4, lw=1.3, zorder=4)

        # posterior dots
        for xi, cond in enumerate(CONDITIONS):
            vals = post[cond][key]
            jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(vals))
            ax.scatter(xi + jitter, vals, color=COLORS[cond],
                       s=16, alpha=0.45, zorder=5)

        # annotate p50
        for xi, v in enumerate(p50):
            ax.text(xi, v + (p95[xi]-p50[xi])*1.05 + abs(v)*0.01,
                    f"{v:.3g}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[c] for c in CONDITIONS],
                           fontsize=9, rotation=10, ha="right")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3, ls="--")
        ax.set_ylim(0)

    fig.suptitle(
        "B3: CZM Interface Properties — 4 Metrics × 4 Conditions\n"
        "(bars=p50,  error=p05–p95 DI credible interval,  dots=20 posterior samples)",
        fontsize=12, fontweight="bold",
    )
    out = _OUT / "figures" / "fig_B3_czm_4panel.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[done] {out.name}")


# ── Fig B3-3: Posterior violin (RF_peak + Gc) ────────────────────────────────
def fig_b3_posterior_bands(post):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    colors = [COLORS[c] for c in CONDITIONS]

    for ax, (key, ylabel, title) in zip(axes, [
        ("rf_peak", "Peak Pull Force (N)",   "B3 Posterior: RF_peak"),
        ("gc",      "$G_c$ (J/m²)",           "B3 Posterior: Fracture Energy"),
    ]):
        vals_list = [post[c][key] for c in CONDITIONS]
        parts = ax.violinplot(vals_list, positions=range(len(CONDITIONS)),
                              showmedians=False, showextrema=False, widths=0.6)
        for body, col in zip(parts["bodies"], colors):
            body.set_facecolor(col)
            body.set_alpha(0.4)

        bp = ax.boxplot(vals_list, positions=range(len(CONDITIONS)),
                        widths=0.28, patch_artist=True,
                        medianprops=dict(color="black", lw=2),
                        whiskerprops=dict(lw=1.2),
                        capprops=dict(lw=1.2),
                        flierprops=dict(marker="o", ms=4, alpha=0.4))
        for patch, col in zip(bp["boxes"], colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.75)

        for xi, (vals, col) in enumerate(zip(vals_list, colors)):
            jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(vals))
            ax.scatter(xi + jitter, vals, color=col, s=22, zorder=5, alpha=0.85)

        ax.set_xticks(range(len(CONDITIONS)))
        ax.set_xticklabels([LABELS[c] for c in CONDITIONS], fontsize=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.3, ls="--")
        ax.set_ylim(0)

    fig.suptitle(
        "B3: Posterior Distribution of CZM Interface Properties\n"
        "(20 TMCMC samples × 4 conditions)",
        fontsize=12, fontweight="bold",
    )
    out = _OUT / "figures" / "fig_B3_czm_posterior_bands.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[done] {out.name}")


# ── print summary table ──────────────────────────────────────────────────────
def print_summary(results, post):
    print("\n{'='*65}")
    print(f"{'Condition':20s}  {'DI_mean(p50)':12s}  {'t_max (Pa)':12s}  "
          f"{'Gc (J/m²)':10s}  {'RF_peak (N)':11s}")
    print("-" * 65)
    for cond in CONDITIONS:
        r = results[cond]["p50"]
        print(f"{LABELS[cond]:20s}  {r['di_mean']:12.5f}  {r['t_max']:12.4g}  "
              f"{r['gc']:10.4f}  {r['rf_peak']:11.4g}")
    print()

    # compare to commensal_static reference
    ref_rf = results["commensal_static"]["p50"]["rf_peak"]
    ref_gc = results["commensal_static"]["p50"]["gc"]
    print(f"{'Δ vs Comm.Static':20s}  {'':12s}  {'':12s}  {'ΔGc':10s}  {'ΔRF':11s}")
    print("-" * 65)
    for cond in CONDITIONS:
        r = results[cond]["p50"]
        delta_gc = (r["gc"] - ref_gc) / ref_gc * 100
        delta_rf = (r["rf_peak"] - ref_rf) / ref_rf * 100
        print(f"{LABELS[cond]:20s}  {'':12s}  {'':12s}  {delta_gc:+9.1f}%  {delta_rf:+10.1f}%")


# ── main ────────────────────────────────────────────────────────────────────
def main():
    print("Loading B1 DI quantile data...")
    results, di_stacks = load_b1()

    print("Computing posterior CZM metrics (20 samples × 4 conditions)...")
    post = compute_posterior_czm(di_stacks)

    save_csv(results)
    print_summary(results, post)

    print("\nGenerating figures...")
    np.random.seed(42)
    fig_b3_summary(results, post)
    fig_b3_4panel(results, post)
    fig_b3_posterior_bands(post)

    print(f"\nAll B3 analytical results → {_OUT}/")


if __name__ == "__main__":
    main()
