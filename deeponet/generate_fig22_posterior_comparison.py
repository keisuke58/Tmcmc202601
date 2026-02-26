#!/usr/bin/env python3
"""Fig 22: DeepONet vs ODE posterior comparison (paper quality)."""
import json
import numpy as np
from pathlib import Path
from scipy.stats import gaussian_kde
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
PAPER_FIG_DIR = PROJECT_ROOT / "FEM" / "figures" / "paper_final"
PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR = PROJECT_ROOT / "data_5species" / "_runs"

PARAM_NAMES = [
    r"$\mu_{\mathrm{So}}$",
    r"$\mu_{\mathrm{An}}$",
    r"$\mu_{\mathrm{Vd}}$",
    r"$\mu_{\mathrm{Fn}}$",
    r"$\mu_{\mathrm{Pg}}$",
    r"$a_{12}$",
    r"$a_{13}$",
    r"$a_{14}$",
    r"$a_{21}$",
    r"$a_{23}$",
    r"$a_{24}$",
    r"$a_{31}$",
    r"$a_{32}$",
    r"$a_{34}$",
    r"$a_{41}$",
    r"$a_{15}$",
    r"$a_{25}$",
    r"$a_{35}^{\mathrm{inv}}$",
    r"$a_{35}$",
    r"$a_{45}$",
]

CONDITION_DIRS = {
    "CS": ("commensal_static_posterior", "deeponet_Commensal_Static"),
    "CH": ("commensal_hobic_posterior", "deeponet_Commensal_HOBIC"),
    "DS": ("dysbiotic_static_posterior", "deeponet_Dysbiotic_Static"),
    "DH": ("dh_baseline", "deeponet_Dysbiotic_HOBIC"),
}


def load_samples_and_map(run_dir):
    samples_path = run_dir / "samples.npy"
    map_path = run_dir / "theta_MAP.json"
    logL_path = run_dir / "logL.npy"
    samples, theta_map, logL = None, None, None
    if samples_path.exists():
        samples = np.load(str(samples_path))
    if logL_path.exists():
        logL = np.load(str(logL_path))
    if map_path.exists():
        with open(map_path) as f:
            data = json.load(f)
        theta_map = np.array(data["theta_full"]) if isinstance(data, dict) else np.array(data)
    return samples, theta_map, logL


def compute_overlap(s1, s2, idx):
    d1, d2 = s1[:, idx], s2[:, idx]
    m1, std1, m2, std2 = d1.mean(), d1.std(), d2.mean(), d2.std()
    if std1 < 1e-8 or std2 < 1e-8:
        return 0.0
    bd = 0.25 * ((m1 - m2) ** 2 / (std1**2 + std2**2)) + 0.5 * np.log(
        0.5 * (std1 / std2 + std2 / std1)
    )
    return float(np.exp(-bd))


def plot_kde(ax, data, color, label, alpha_fill=0.25):
    if len(data) < 5 or data.std() < 1e-10:
        ax.axvline(data.mean(), color=color, lw=2, label=label)
        return
    try:
        kde = gaussian_kde(data, bw_method="silverman")
        xg = np.linspace(data.min() - 0.1 * data.ptp(), data.max() + 0.1 * data.ptp(), 200)
        y = kde(xg)
        ax.plot(xg, y, color=color, lw=1.5, label=label)
        ax.fill_between(xg, y, alpha=alpha_fill, color=color)
    except:
        ax.hist(data, bins=30, alpha=0.4, density=True, color=color, label=label)


def main():
    ode_s, ode_m, _ = load_samples_and_map(RUNS_DIR / "dh_baseline")
    don_s, don_m, _ = load_samples_and_map(RUNS_DIR / "deeponet_Dysbiotic_HOBIC")
    if ode_s is None or don_s is None:
        print("ERROR: DH samples not found")
        return

    overlaps = [compute_overlap(ode_s, don_s, i) for i in range(20)]
    fig = plt.figure(figsize=(15, 14))
    gs = GridSpec(5, 5, figure=fig, hspace=0.45, wspace=0.35, height_ratios=[1, 1, 1, 1, 1.3])
    c_ode, c_don = "#D32F2F", "#1565C0"

    for idx in range(20):
        ax = fig.add_subplot(gs[idx // 5, idx % 5])
        plot_kde(ax, ode_s[:, idx], c_ode, "ODE-TMCMC")
        plot_kde(ax, don_s[:, idx], c_don, "DeepONet-TMCMC")
        if ode_m is not None:
            ax.axvline(ode_m[idx], color=c_ode, lw=1.2, ls="--", alpha=0.7)
        if don_m is not None:
            ax.axvline(don_m[idx], color=c_don, lw=1.2, ls=":", alpha=0.7)
        ol = overlaps[idx]
        olc = "#2E7D32" if ol > 0.9 else ("#F57F17" if ol > 0.5 else "#C62828")
        ax.set_title(f"{PARAM_NAMES[idx]}  OL={ol:.2f}", fontsize=8.5, color=olc)
        ax.tick_params(labelsize=6)
        ax.set_yticks([])
        if idx == 0:
            ax.legend(fontsize=6, loc="upper right", framealpha=0.7)

    ax_ol = fig.add_subplot(gs[4, :3])
    cols = ["#2E7D32" if o > 0.9 else ("#F57F17" if o > 0.5 else "#C62828") for o in overlaps]
    ax_ol.bar(range(20), overlaps, color=cols, alpha=0.85, edgecolor="white", lw=0.5)
    ax_ol.axhline(0.95, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax_ol.axhline(0.50, color="gray", ls=":", lw=0.8, alpha=0.6)
    ax_ol.set_xticks(range(20))
    ax_ol.set_xticklabels(PARAM_NAMES, fontsize=7, rotation=45, ha="right")
    ax_ol.set_ylabel("Bhattacharyya Overlap", fontsize=9)
    ax_ol.set_title("(b) Posterior Overlap per Parameter (DH)", fontsize=10, fontweight="bold")
    ax_ol.set_ylim(0, 1.08)
    ax_ol.grid(axis="y", alpha=0.2)
    for i, ol in enumerate(overlaps):
        if ol < 0.6:
            ax_ol.text(
                i,
                ol + 0.03,
                f"{ol:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
                fontweight="bold",
                color="#C62828",
            )
    hi = sum(1 for o in overlaps if o > 0.95)
    md = sum(1 for o in overlaps if 0.5 <= o <= 0.95)
    lo = sum(1 for o in overlaps if o < 0.5)
    ax_ol.text(
        0.02,
        0.95,
        f"High (>0.95): {hi}/20\nMedium: {md}/20\nLow (<0.5): {lo}/20",
        transform=ax_ol.transAxes,
        fontsize=7,
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    ax_sp = fig.add_subplot(gs[4, 3:])
    conds = ["CS", "CH", "DS", "DH"]
    ot = {"CS": 1800, "CH": 1800, "DS": 1800, "DH": 1800}
    dt = {"CS": 18.4, "CH": 17.0, "DS": 18.0, "DH": 12.5}
    ov, dv = [ot[c] for c in conds], [dt[c] for c in conds]
    su = [o / d for o, d in zip(ov, dv)]
    x = np.arange(4)
    w = 0.35
    ax_sp.bar(x - w / 2, ov, w, color=c_ode, alpha=0.7, label="ODE-TMCMC")
    ax_sp.bar(x + w / 2, dv, w, color=c_don, alpha=0.7, label="DeepONet-TMCMC")
    ax_sp.set_yscale("log")
    ax_sp.set_xticks(x)
    ax_sp.set_xticklabels(conds, fontsize=10)
    ax_sp.set_ylabel("Wall Time [s]", fontsize=9)
    ax_sp.set_title("(c) TMCMC Computational Cost", fontsize=10, fontweight="bold")
    ax_sp.legend(fontsize=8, loc="upper right")
    ax_sp.grid(axis="y", alpha=0.2, which="both")
    for i, (s, d) in enumerate(zip(su, dv)):
        ax_sp.annotate(
            f"{s:.0f}x",
            xy=(x[i] + w / 2, d),
            xytext=(0, -15),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            fontweight="bold",
            color="#2E7D32",
        )

    fig.suptitle(
        "(a) DeepONet-TMCMC vs ODE-TMCMC Posterior Comparison\n(Dysbiotic HOBIC, dashed/dotted = MAP)",
        fontsize=12,
        fontweight="bold",
        y=0.995,
    )
    for ext in ["png", "pdf"]:
        out = PAPER_FIG_DIR / f"Fig22_deeponet_posterior_comparison.{ext}"
        fig.savefig(str(out), dpi=300, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)

    print()
    print("=" * 70)
    print("Summary: DeepONet vs ODE TMCMC")
    print("=" * 70)
    for c in conds:
        od, dd = CONDITION_DIRS[c]
        _, om, _ = load_samples_and_map(RUNS_DIR / od)
        _, dm, _ = load_samples_and_map(RUNS_DIR / dd)
        if om is not None and dm is not None:
            diff = np.abs(om - dm).mean()
            print(
                f"{c:<8} ODE={ot[c]:>6.0f}s DON={dt[c]:>6.1f}s Speedup={ot[c]/dt[c]:>5.0f}x MAP_diff={diff:.4f}"
            )
    print(f"\nDH Overlap: High={hi}/20 Med={md}/20 Low={lo}/20 Mean={np.mean(overlaps):.3f}")


if __name__ == "__main__":
    main()
