#!/usr/bin/env python3
"""
Compare two joint TMCMC runs (e.g. 50p vs 200p).

Usage:
    python compare_joint_runs.py \
        _runs/joint_4cond_50p_20260318_031450 \
        _runs/joint_4cond_200p_20260318_162834
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path

MAIN_DIR = Path(__file__).parent
sys.path.insert(0, str(MAIN_DIR.parent))
sys.path.insert(0, str(MAIN_DIR.parent.parent / "tmcmc" / "program2602"))

from core.joint_evaluator import ALL_AIJ_INDICES, PARAM_NAMES

JOINT_PARAM_NAMES = [PARAM_NAMES[i] for i in ALL_AIJ_INDICES]


def bhatt_1d(x, y, n_bins=50):
    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    bins = np.linspace(lo, hi, n_bins + 1)
    hx, _ = np.histogram(x, bins=bins, density=True)
    hy, _ = np.histogram(y, bins=bins, density=True)
    dx = bins[1] - bins[0]
    return float(np.sum(np.sqrt(hx * hy)) * dx)


def load_run(run_dir):
    d = Path(run_dir)
    if not d.is_absolute():
        d = MAIN_DIR / d
    data = {}
    data["dir"] = d
    data["tag"] = d.name

    sf = d / "samples.npy"
    lf = d / "logL.npy"
    if sf.exists():
        data["samples"] = np.load(sf)
        data["logL"] = np.load(lf)
    else:
        raise FileNotFoundError(f"No samples.npy in {d}")

    for jf in ["mcmc_diagnostics.json", "credible_intervals.json", "theta_MAP.json"]:
        p = d / jf
        if p.exists():
            with open(p) as f:
                data[jf.replace(".json", "")] = json.load(f)
    return data


def compare(run_a, run_b):
    """Compare two runs and print summary."""
    tag_a = run_a["tag"]
    tag_b = run_b["tag"]

    print("=" * 70)
    print(f"COMPARISON: {tag_a} vs {tag_b}")
    print("=" * 70)

    # Basic stats
    for label, r in [("A", run_a), ("B", run_b)]:
        n = r["samples"].shape[0]
        logL_max = r["logL"].max()
        diag = r.get("mcmc_diagnostics", {})
        rhat = diag.get("rhat_max", "N/A")
        ess = diag.get("ess_min", "N/A")
        print(f"  [{label}] {r['tag']}: N={n}, max_logL={logL_max:.2f}, R-hat={rhat}, ESS={ess}")

    # MAP comparison
    print(
        f"\n{'Param':>6s} | {'MAP_A':>8s} | {'MAP_B':>8s} | {'Δ':>8s} | {'CI_A width':>10s} | {'CI_B width':>10s} | {'CI↓%':>6s}"
    )
    print("-" * 70)

    ci_a = run_a.get("credible_intervals", {})
    ci_b = run_b.get("credible_intervals", {})
    map_a = run_a.get("theta_MAP", {}).get("theta_joint", [None] * 15)
    map_b = run_b.get("theta_MAP", {}).get("theta_joint", [None] * 15)

    ci_improvements = []
    for pi in range(len(ALL_AIJ_INDICES)):
        pname = JOINT_PARAM_NAMES[pi]
        ma = map_a[pi] if pi < len(map_a) else None
        mb = map_b[pi] if pi < len(map_b) else None

        wa = None
        wb = None
        if ci_a and "hdi_lower" in ci_a and pi < len(ci_a["hdi_lower"]):
            wa = ci_a["hdi_upper"][pi] - ci_a["hdi_lower"][pi]
        if ci_b and "hdi_lower" in ci_b and pi < len(ci_b["hdi_lower"]):
            wb = ci_b["hdi_upper"][pi] - ci_b["hdi_lower"][pi]

        delta = (mb - ma) if (ma is not None and mb is not None) else None
        ci_pct = ((wa - wb) / wa * 100) if (wa and wb and wa > 0) else None
        if ci_pct is not None:
            ci_improvements.append(ci_pct)

        ma_s = f"{ma:+.4f}" if ma is not None else "N/A"
        mb_s = f"{mb:+.4f}" if mb is not None else "N/A"
        d_s = f"{delta:+.4f}" if delta is not None else "N/A"
        wa_s = f"{wa:.4f}" if wa is not None else "N/A"
        wb_s = f"{wb:.4f}" if wb is not None else "N/A"
        ci_s = f"{ci_pct:+.1f}%" if ci_pct is not None else "N/A"

        print(
            f"  {pname:>5s} | {ma_s:>8s} | {mb_s:>8s} | {d_s:>8s} | {wa_s:>10s} | {wb_s:>10s} | {ci_s:>6s}"
        )

    if ci_improvements:
        print(f"\n  Mean CI width improvement: {np.mean(ci_improvements):+.1f}%")
        print(
            f"  Params with narrower CI: {sum(1 for x in ci_improvements if x > 0)}/{len(ci_improvements)}"
        )

    # Bhattacharyya overlap between runs
    print("\nBhattacharyya overlap (A vs B marginals):")
    overlaps = []
    for pi in range(min(run_a["samples"].shape[1], run_b["samples"].shape[1])):
        bc = bhatt_1d(run_a["samples"][:, pi], run_b["samples"][:, pi])
        overlaps.append(bc)
        pname = JOINT_PARAM_NAMES[pi] if pi < len(JOINT_PARAM_NAMES) else f"p{pi}"
        print(f"  {pname}: {bc:.3f}")
    print(f"  Mean overlap: {np.mean(overlaps):.3f}")

    # LogL improvement
    dll = run_b["logL"].max() - run_a["logL"].max()
    print(f"\nΔ max_logL (B-A): {dll:+.2f}")


def plot_comparison(run_a, run_b, fig_dir):
    """Generate comparison plots."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_params = len(ALL_AIJ_INDICES)
    ncols = 5
    nrows = (n_params + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4 * nrows))
    axes = axes.flatten()

    for pi in range(n_params):
        ax = axes[pi]
        pname = JOINT_PARAM_NAMES[pi]
        ax.hist(
            run_a["samples"][:, pi],
            bins=35,
            density=True,
            alpha=0.5,
            color="tab:blue",
            label=run_a["tag"][:20],
            edgecolor="none",
        )
        ax.hist(
            run_b["samples"][:, pi],
            bins=35,
            density=True,
            alpha=0.5,
            color="tab:red",
            label=run_b["tag"][:20],
            edgecolor="none",
        )
        ax.set_title(pname, fontsize=10)
        if pi == 0:
            ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

    for k in range(pi + 1, len(axes)):
        axes[k].set_visible(False)

    fig.suptitle(f"Run Comparison: {run_a['tag'][:25]} vs {run_b['tag'][:25]}", fontsize=12)
    fig.tight_layout()
    out = fig_dir / "joint_run_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nComparison figure: {out}")


def main():
    parser = argparse.ArgumentParser(description="Compare two joint TMCMC runs")
    parser.add_argument("run_a", help="First run directory (baseline)")
    parser.add_argument("run_b", help="Second run directory (improved)")
    parser.add_argument("--no-plot", action="store_true", help="Skip figure generation")
    args = parser.parse_args()

    run_a = load_run(args.run_a)
    run_b = load_run(args.run_b)

    compare(run_a, run_b)

    if not args.no_plot:
        fig_dir = run_b["dir"] / "figures"
        fig_dir.mkdir(exist_ok=True)
        plot_comparison(run_a, run_b, fig_dir)


if __name__ == "__main__":
    main()
