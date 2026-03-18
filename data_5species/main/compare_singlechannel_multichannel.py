#!/usr/bin/env python3
"""
Compare single-channel vs multi-channel TMCMC posteriors.

For each condition, quantifies how adding total species / viability / pH
channels changes the posterior (CI width, MAP shift, Bhattacharyya overlap).

Usage:
    python compare_singlechannel_multichannel.py \
        --single-CS _runs/CS_500p_expIC_... \
        --multi-CS  _runs/CS_500p_expIC_mc_... \
        --single-CH _runs/CH_500p_expIC_... \
        --multi-CH  _runs/CH_500p_expIC_mc_... \
        --single-DS _runs/DS_500p_expIC_... \
        --multi-DS  _runs/DS_500p_expIC_mc_... \
        --single-DH _runs/DH_500p_expIC_... \
        --multi-DH  _runs/DH_500p_expIC_mc_...
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path

MAIN_DIR = Path(__file__).parent
sys.path.insert(0, str(MAIN_DIR.parent))
sys.path.insert(0, str(MAIN_DIR.parent.parent / "tmcmc" / "program2602"))

from core.joint_evaluator import ALL_AIJ_INDICES, PARAM_NAMES, CONDITION_AIJ_LOCKS

CONDITION_KEYS = {
    "CS": "Commensal_Static",
    "CH": "Commensal_HOBIC",
    "DS": "Dysbiotic_Static",
    "DH": "Dysbiotic_HOBIC",
}


def bhatt_1d(x, y, n_bins=50):
    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    if hi - lo < 1e-12:
        return 1.0
    bins = np.linspace(lo, hi, n_bins + 1)
    hx, _ = np.histogram(x, bins=bins, density=True)
    hy, _ = np.histogram(y, bins=bins, density=True)
    dx = bins[1] - bins[0]
    return float(np.sum(np.sqrt(hx * hy)) * dx)


def load_run(run_dir):
    d = Path(run_dir)
    if not d.is_absolute():
        d = MAIN_DIR / d
    samples = np.load(d / "samples.npy")
    config = json.load(open(d / "config.json"))
    logL = np.load(d / "logL.npy") if (d / "logL.npy").exists() else None
    cond = config.get("condition", "")
    cult = config.get("cultivation", "")
    cond_key = f"{cond}_{cult}"
    locks = CONDITION_AIJ_LOCKS.get(cond_key, set())
    active_aij = [i for i in ALL_AIJ_INDICES if i not in locks]
    return {
        "samples": samples,
        "logL": logL,
        "config": config,
        "cond_key": cond_key,
        "active_aij": active_aij,
        "locks": locks,
        "dir": d,
    }


def compare_single_multi(label, single, multi):
    """Compare single-channel vs multi-channel for one condition."""
    print(f"\n{'=' * 70}")
    print(f"  {label}: single-channel vs multi-channel")
    print(f"  Single: {single['dir'].name}  (n={single['samples'].shape[0]})")
    print(f"  Multi:  {multi['dir'].name}  (n={multi['samples'].shape[0]})")
    print(f"{'=' * 70}")

    active = single["active_aij"]
    print(
        f"\n{'Param':>5s} | {'MAP_s':>8s} | {'MAP_m':>8s} | {'ΔMAP':>8s} | {'CI_s':>8s} | {'CI_m':>8s} | {'CI↓%':>6s} | {'BC':>6s}"
    )
    print("-" * 75)

    results = {}
    for aij_idx in active:
        pname = PARAM_NAMES[aij_idx]
        s_samples = single["samples"][:, aij_idx]
        m_samples = multi["samples"][:, aij_idx]

        # MAP (mode via KDE or just max-logL sample)
        if single["logL"] is not None:
            s_map = s_samples[np.argmax(single["logL"])]
        else:
            s_map = np.median(s_samples)
        if multi["logL"] is not None:
            m_map = m_samples[np.argmax(multi["logL"])]
        else:
            m_map = np.median(m_samples)

        # 95% HDI width
        s_lo, s_hi = np.percentile(s_samples, [2.5, 97.5])
        m_lo, m_hi = np.percentile(m_samples, [2.5, 97.5])
        s_ci = s_hi - s_lo
        m_ci = m_hi - m_lo
        ci_change = (s_ci - m_ci) / s_ci * 100 if s_ci > 1e-10 else 0

        # Bhattacharyya
        bc = bhatt_1d(s_samples, m_samples)

        results[pname] = {
            "map_single": float(s_map),
            "map_multi": float(m_map),
            "ci_single": float(s_ci),
            "ci_multi": float(m_ci),
            "ci_change_pct": float(ci_change),
            "bc": float(bc),
        }

        print(
            f"  {pname:>5s} | {s_map:+8.4f} | {m_map:+8.4f} | {m_map - s_map:+8.4f} | "
            f"{s_ci:8.4f} | {m_ci:8.4f} | {ci_change:+5.1f}% | {bc:6.3f}"
        )

    # Summary
    bcs = [v["bc"] for v in results.values()]
    ci_changes = [v["ci_change_pct"] for v in results.values()]
    print(f"\n  Mean BC (single vs multi): {np.mean(bcs):.3f}")
    print(f"  Mean CI change: {np.mean(ci_changes):+.1f}%")
    print(f"  Params narrowed (CI↓>0): {sum(1 for x in ci_changes if x > 0)}/{len(ci_changes)}")

    return results


def plot_comparison(all_results, out_dir):
    """Generate before/after plots."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = list(all_results.keys())
    n_cond = len(conditions)

    # Bar chart: CI change per param per condition
    all_params = []
    for cond_results in all_results.values():
        for pname in cond_results:
            if pname not in all_params:
                all_params.append(pname)

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(all_params))
    width = 0.8 / n_cond
    colors = {"CS": "tab:blue", "CH": "tab:cyan", "DS": "tab:red", "DH": "tab:orange"}

    for i, cond in enumerate(conditions):
        vals = [all_results[cond].get(p, {}).get("ci_change_pct", 0) for p in all_params]
        ax.bar(x + i * width, vals, width, label=cond, color=colors.get(cond, f"C{i}"), alpha=0.7)

    ax.set_xticks(x + width * (n_cond - 1) / 2)
    ax.set_xticklabels(all_params, rotation=45, ha="right")
    ax.set_ylabel("CI Width Change (%)")
    ax.set_title("Effect of Multi-Channel Likelihood on 95% CI Width")
    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax.legend()
    fig.tight_layout()
    out = out_dir / "multichannel_ci_change.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nCI change figure: {out}")

    # BC heatmap
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    bc_matrix = np.zeros((len(all_params), n_cond))
    for j, cond in enumerate(conditions):
        for i, p in enumerate(all_params):
            bc_matrix[i, j] = all_results[cond].get(p, {}).get("bc", float("nan"))

    im = ax2.imshow(bc_matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax2.set_xticks(range(n_cond))
    ax2.set_xticklabels(conditions)
    ax2.set_yticks(range(len(all_params)))
    ax2.set_yticklabels(all_params)
    plt.colorbar(im, ax=ax2, label="BC (single vs multi)")
    ax2.set_title("Single-Channel vs Multi-Channel Posterior Stability")
    for i in range(len(all_params)):
        for j in range(n_cond):
            if not np.isnan(bc_matrix[i, j]):
                ax2.text(j, i, f"{bc_matrix[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig2.tight_layout()
    out2 = out_dir / "multichannel_bc_stability.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"BC stability figure: {out2}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare single-channel vs multi-channel posteriors"
    )
    for cond in ["CS", "CH", "DS", "DH"]:
        parser.add_argument(f"--single-{cond}", help=f"Single-channel {cond} run dir")
        parser.add_argument(f"--multi-{cond}", help=f"Multi-channel {cond} run dir")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    all_results = {}
    for cond in ["CS", "CH", "DS", "DH"]:
        s_dir = getattr(args, f"single_{cond}")
        m_dir = getattr(args, f"multi_{cond}")
        if s_dir and m_dir:
            single = load_run(s_dir)
            multi = load_run(m_dir)
            all_results[cond] = compare_single_multi(cond, single, multi)

    if not all_results:
        print("No condition pairs provided. Use --single-XX and --multi-XX flags.")
        sys.exit(1)

    if not args.no_plot and all_results:
        out_dir = (
            Path(args.out_dir) if args.out_dir else MAIN_DIR / "_runs" / "multichannel_comparison"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_comparison(all_results, out_dir)

        # Save JSON
        save = {}
        for cond, res in all_results.items():
            save[cond] = res
        with open(out_dir / "multichannel_comparison.json", "w") as f:
            json.dump(save, f, indent=2)
        print(f"\nResults JSON: {out_dir / 'multichannel_comparison.json'}")


if __name__ == "__main__":
    main()
