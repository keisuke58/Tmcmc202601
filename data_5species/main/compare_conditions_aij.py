#!/usr/bin/env python3
"""
Cross-condition A_ij posterior comparison.

Computes Bhattacharyya overlap between all condition pairs for each A_ij parameter.
Identifies which interactions are condition-independent vs environment-dependent.

Usage:
    python compare_conditions_aij.py \
        --CS _runs/CS_1000p_expIC_repSigma_20260314_195803 \
        --CH _runs/CH_1000p_expIC_repSigma_20260314_195803 \
        --DS _runs/DS_1000p_expIC_repSigma_20260314_195803 \
        --DH _runs/DH_1000p_expIC_repSigma_20260311_191354

    # Compare multichannel vs single-channel:
    python compare_conditions_aij.py \
        --CS _runs/CS_500p_expIC_mc_2ch_... \
        --CH _runs/CH_500p_expIC_mc_2ch_... \
        --DS _runs/DS_500p_expIC_mc_2ch_... \
        --DH _runs/DH_500p_expIC_mc_2ch_... \
        --tag multichannel
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from itertools import combinations

MAIN_DIR = Path(__file__).parent
sys.path.insert(0, str(MAIN_DIR.parent))
sys.path.insert(0, str(MAIN_DIR.parent.parent / "tmcmc" / "program2602"))

from core.joint_evaluator import ALL_AIJ_INDICES, PARAM_NAMES, CONDITION_AIJ_LOCKS

# Biological annotation for each A_ij
AIJ_BIO = {
    "a11": "So self",
    "a12": "So←An",
    "a22": "An self",
    "a33": "Vd self",
    "a34": "Vd←Fn",
    "a44": "Fn self",
    "a13": "So←Vd",
    "a14": "So←Fn",
    "a23": "An←Vd",
    "a24": "An←Fn",
    "a55": "Pg self",
    "a15": "So←Pg",
    "a25": "An←Pg",
    "a35": "Vei←Pg",
    "a45": "Fn←Pg",
}

CONDITION_KEYS = {
    "CS": "Commensal_Static",
    "CH": "Commensal_HOBIC",
    "DS": "Dysbiotic_Static",
    "DH": "Dysbiotic_HOBIC",
}


def bhatt_1d(x, y, n_bins=50):
    """Bhattacharyya coefficient between two 1D sample sets."""
    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    if hi - lo < 1e-12:
        return 1.0
    bins = np.linspace(lo, hi, n_bins + 1)
    hx, _ = np.histogram(x, bins=bins, density=True)
    hy, _ = np.histogram(y, bins=bins, density=True)
    dx = bins[1] - bins[0]
    return float(np.sum(np.sqrt(hx * hy)) * dx)


def load_condition(run_dir):
    """Load samples and config from a run directory."""
    d = Path(run_dir)
    if not d.is_absolute():
        d = MAIN_DIR / d
    samples = np.load(d / "samples.npy")
    config = json.load(open(d / "config.json"))

    # Determine active_indices from config or locks
    cond = config.get("condition", "")
    cult = config.get("cultivation", "")
    cond_key = f"{cond}_{cult}"
    locks = CONDITION_AIJ_LOCKS.get(cond_key, set())

    # Map: which columns in samples correspond to which A_ij
    # samples has shape (n, n_active) where n_active = 20 - len(locks_all)
    # But actually samples always has 20 columns (all params including b_i)
    # Active A_ij = ALL_AIJ_INDICES minus locked ones
    active_aij = [i for i in ALL_AIJ_INDICES if i not in locks]

    return {
        "samples": samples,
        "config": config,
        "cond_key": cond_key,
        "active_aij": active_aij,
        "locks": locks,
        "dir": d,
    }


def compare_all_conditions(conditions, tag=""):
    """Compare A_ij posteriors across all condition pairs."""
    cond_labels = list(conditions.keys())
    pairs = list(combinations(cond_labels, 2))

    # For each A_ij, compute pairwise Bhattacharyya overlap
    print("=" * 90)
    print(f"CROSS-CONDITION A_ij POSTERIOR COMPARISON{f' ({tag})' if tag else ''}")
    print("=" * 90)

    # Header
    pair_labels = [f"{a}-{b}" for a, b in pairs]
    header = (
        f"{'Param':>5s} {'Bio':>10s} | "
        + " | ".join(f"{p:>7s}" for p in pair_labels)
        + " | Mean   | Verdict"
    )
    print(header)
    print("-" * len(header))

    results = {}
    for aij_idx in ALL_AIJ_INDICES:
        pname = PARAM_NAMES[aij_idx]
        bio = AIJ_BIO.get(pname, "")

        overlaps = []
        overlap_strs = []
        for c1, c2 in pairs:
            d1 = conditions[c1]
            d2 = conditions[c2]

            # Check if both conditions have this param free
            if aij_idx in d1["locks"] or aij_idx in d2["locks"]:
                overlap_strs.append("  lock ")
                continue

            bc = bhatt_1d(d1["samples"][:, aij_idx], d2["samples"][:, aij_idx])
            overlaps.append(bc)

            # Color coding
            if bc > 0.8:
                marker = " "
            elif bc > 0.5:
                marker = "*"
            else:
                marker = "!"
            overlap_strs.append(f"{bc:6.3f}{marker}")

        if overlaps:
            mean_bc = np.mean(overlaps)
            if mean_bc > 0.8:
                verdict = "SHARED"
            elif mean_bc > 0.5:
                verdict = "WEAK"
            elif mean_bc > 0.3:
                verdict = "DIVERGE"
            else:
                verdict = "COND-DEP"
        else:
            mean_bc = float("nan")
            verdict = "N/A"

        results[pname] = {
            "overlaps": overlaps,
            "mean": mean_bc,
            "verdict": verdict,
        }

        mean_str = f"{mean_bc:.3f}" if not np.isnan(mean_bc) else "  N/A"
        print(
            f"  {pname:>5s} {bio:>10s} | "
            + " | ".join(f"{s:>7s}" for s in overlap_strs)
            + f" | {mean_str} | {verdict}"
        )

    # Summary
    shared = [k for k, v in results.items() if v["verdict"] == "SHARED"]
    weak = [k for k, v in results.items() if v["verdict"] == "WEAK"]
    diverge = [k for k, v in results.items() if v["verdict"] in ("DIVERGE", "COND-DEP")]

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"  SHARED (BC > 0.8): {len(shared)} params — {', '.join(shared)}")
    print(f"  WEAK   (0.5-0.8): {len(weak)} params — {', '.join(weak)}")
    print(f"  DIVERGE (< 0.5):  {len(diverge)} params — {', '.join(diverge)}")

    all_means = [v["mean"] for v in results.values() if not np.isnan(v["mean"])]
    if all_means:
        print(f"\n  Global mean BC: {np.mean(all_means):.3f}")
        print(
            f"  Min BC: {min(all_means):.3f} ({min(results, key=lambda k: results[k]['mean'] if not np.isnan(results[k]['mean']) else 999)})"
        )

    print("\n  Legend: BC > 0.8 = shared, 0.5-0.8 = weak*, < 0.5 = diverge!")
    print("  'lock' = parameter locked to 0 in that condition")

    return results


def plot_comparison(conditions, results, out_dir):
    """Generate comparison violin/box plots."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cond_labels = list(conditions.keys())
    n_aij = len(ALL_AIJ_INDICES)
    ncols = 5
    nrows = (n_aij + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(22, 4 * nrows))
    axes = axes.flatten()

    colors = {"CS": "tab:blue", "CH": "tab:cyan", "DS": "tab:red", "DH": "tab:orange"}

    for pi, aij_idx in enumerate(ALL_AIJ_INDICES):
        ax = axes[pi]
        pname = PARAM_NAMES[aij_idx]
        bio = AIJ_BIO.get(pname, "")

        plot_data = []
        plot_labels = []
        plot_colors = []
        for cl in cond_labels:
            d = conditions[cl]
            if aij_idx not in d["locks"]:
                plot_data.append(d["samples"][:, aij_idx])
                plot_labels.append(cl)
                plot_colors.append(colors[cl])

        if plot_data:
            bp = ax.boxplot(plot_data, labels=plot_labels, patch_artist=True, widths=0.6)
            for patch, color in zip(bp["boxes"], plot_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

        verdict = results.get(pname, {}).get("verdict", "N/A")
        mean_bc = results.get(pname, {}).get("mean", float("nan"))
        title_color = "green" if verdict == "SHARED" else "orange" if verdict == "WEAK" else "red"
        ax.set_title(
            f"{pname} ({bio})\nBC={mean_bc:.2f} [{verdict}]", fontsize=9, color=title_color
        )
        ax.tick_params(labelsize=7)

    for k in range(pi + 1, len(axes)):
        axes[k].set_visible(False)

    fig.suptitle("Cross-Condition A_ij Posterior Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = out_dir / "cross_condition_aij_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {out}")

    # Heatmap of pairwise BC
    pairs = list(combinations(cond_labels, 2))
    n_free = sum(
        1 for idx in ALL_AIJ_INDICES if all(idx not in conditions[c]["locks"] for c in cond_labels)
    )

    # Build matrix: rows = A_ij, cols = pairs
    free_aij = [
        idx
        for idx in ALL_AIJ_INDICES
        if all(idx not in conditions[c]["locks"] for c in cond_labels)
    ]
    if len(free_aij) > 0 and len(pairs) > 0:
        bc_matrix = np.zeros((len(free_aij), len(pairs)))
        for i, aij_idx in enumerate(free_aij):
            for j, (c1, c2) in enumerate(pairs):
                bc_matrix[i, j] = bhatt_1d(
                    conditions[c1]["samples"][:, aij_idx],
                    conditions[c2]["samples"][:, aij_idx],
                )

        fig2, ax2 = plt.subplots(figsize=(8, max(6, len(free_aij) * 0.5)))
        im = ax2.imshow(bc_matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax2.set_xticks(range(len(pairs)))
        ax2.set_xticklabels([f"{a}-{b}" for a, b in pairs], rotation=45, ha="right")
        ax2.set_yticks(range(len(free_aij)))
        ax2.set_yticklabels([PARAM_NAMES[idx] for idx in free_aij])
        plt.colorbar(im, ax=ax2, label="Bhattacharyya Coefficient")
        ax2.set_title("Pairwise Bhattacharyya Overlap (A_ij × Condition Pairs)")

        # Annotate
        for i in range(len(free_aij)):
            for j in range(len(pairs)):
                ax2.text(j, i, f"{bc_matrix[i, j]:.2f}", ha="center", va="center", fontsize=7)

        fig2.tight_layout()
        out2 = out_dir / "cross_condition_aij_heatmap.png"
        fig2.savefig(out2, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"Heatmap saved: {out2}")


def main():
    parser = argparse.ArgumentParser(description="Cross-condition A_ij posterior comparison")
    parser.add_argument("--CS", required=True, help="CS run directory")
    parser.add_argument("--CH", required=True, help="CH run directory")
    parser.add_argument("--DS", required=True, help="DS run directory")
    parser.add_argument("--DH", required=True, help="DH run directory")
    parser.add_argument("--tag", default="", help="Tag for output label")
    parser.add_argument("--no-plot", action="store_true", help="Skip figure generation")
    parser.add_argument("--out-dir", default=None, help="Output directory for figures")
    args = parser.parse_args()

    conditions = {}
    for label, run_dir in [("CS", args.CS), ("CH", args.CH), ("DS", args.DS), ("DH", args.DH)]:
        conditions[label] = load_condition(run_dir)
        print(
            f"Loaded {label}: {conditions[label]['cond_key']}, "
            f"samples={conditions[label]['samples'].shape}, "
            f"locks={conditions[label]['locks']}"
        )

    results = compare_all_conditions(conditions, tag=args.tag)

    if not args.no_plot:
        out_dir = (
            Path(args.out_dir)
            if args.out_dir
            else MAIN_DIR / "_runs" / "cross_condition_comparison"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_comparison(conditions, results, out_dir)

    # Save results JSON
    out_dir = (
        Path(args.out_dir) if args.out_dir else MAIN_DIR / "_runs" / "cross_condition_comparison"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    save_results = {}
    for pname, r in results.items():
        save_results[pname] = {
            "mean_bc": float(r["mean"]) if not np.isnan(r["mean"]) else None,
            "verdict": r["verdict"],
            "overlaps": [float(x) for x in r["overlaps"]],
        }
    with open(out_dir / "cross_condition_results.json", "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults JSON: {out_dir / 'cross_condition_results.json'}")


if __name__ == "__main__":
    main()
