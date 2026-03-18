#!/usr/bin/env python3
"""
Auto-analysis for joint 4-condition TMCMC results.

After merge_joint_chains.py produces samples.npy + logL.npy,
this script generates:
  1. 15×15 corner plot
  2. Joint vs individual posterior marginals
  3. Per-condition forward ODE fit at joint MAP
  4. Bhattacharyya overlap statistics
  5. Staggered coupling with joint MAP

Usage:
    python analyze_joint_results.py _runs/joint_4cond_200p_20260318_XXXXXX
    python analyze_joint_results.py _runs/joint_4cond_200p_20260318_XXXXXX --skip-coupling
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path

# ── paths ──
MAIN_DIR = Path(__file__).parent
DATA_5SPECIES = MAIN_DIR.parent
REPO_ROOT = DATA_5SPECIES.parent
sys.path.insert(0, str(MAIN_DIR))
sys.path.insert(0, str(DATA_5SPECIES))
sys.path.insert(0, str(REPO_ROOT / "tmcmc" / "program2602"))

from core.joint_evaluator import ALL_AIJ_INDICES, PARAM_NAMES, CONDITION_AIJ_LOCKS

# ── individual run directories (1000p posteriors) ──
INDIVIDUAL_RUNS = {
    "CS": MAIN_DIR / "_runs" / "commensal_static_1000p",
    "CH": MAIN_DIR / "_runs" / "commensal_hobic_1000p",
    "DS": MAIN_DIR / "_runs" / "dysbiotic_static_1000p",
    "DH": MAIN_DIR / "_runs" / "dh_baseline_1000p",
}
INDIVIDUAL_FALLBACKS = {
    "CS": MAIN_DIR / "_runs" / "commensal_static_posterior",
    "CH": MAIN_DIR / "_runs" / "commensal_hobic_posterior",
    "DS": MAIN_DIR / "_runs" / "dysbiotic_static_posterior",
    "DH": MAIN_DIR / "_runs" / "dh_baseline_posterior",
}
CONDITION_KEYS = {
    "CS": "Commensal_Static",
    "CH": "Commensal_HOBIC",
    "DS": "Dysbiotic_Static",
    "DH": "Dysbiotic_HOBIC",
}

JOINT_PARAM_NAMES = [PARAM_NAMES[i] for i in ALL_AIJ_INDICES]


def load_individual_samples():
    """Load individual posterior samples, return dict of {abbr: (samples_20d, logL)}."""
    indiv = {}
    for abbr in ["CS", "CH", "DS", "DH"]:
        for d in [INDIVIDUAL_RUNS[abbr], INDIVIDUAL_FALLBACKS[abbr]]:
            sf = d / "samples.npy"
            lf = d / "logL.npy"
            if sf.exists() and lf.exists():
                indiv[abbr] = (np.load(sf), np.load(lf))
                print(f"  {abbr}: {indiv[abbr][0].shape[0]} samples from {d.name}")
                break
        else:
            print(f"  {abbr}: NOT FOUND")
    return indiv


def bhattacharyya_overlap_1d(x, y, n_bins=50):
    """Bhattacharyya coefficient for 1D samples."""
    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    bins = np.linspace(lo, hi, n_bins + 1)
    hx, _ = np.histogram(x, bins=bins, density=True)
    hy, _ = np.histogram(y, bins=bins, density=True)
    dx = bins[1] - bins[0]
    hx = hx * dx  # normalize to probabilities
    hy = hy * dx
    return np.sum(np.sqrt(hx * hy))


# ═══════════════════════════════════════════════════════════════
#  Figure 1: Corner plot
# ═══════════════════════════════════════════════════════════════
def plot_corner(samples, MAP_vals, fig_dir, tag):
    """15×15 corner plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_params = samples.shape[1]
    fig, axes = plt.subplots(n_params, n_params, figsize=(24, 24))

    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]
            if j > i:
                ax.set_visible(False)
                continue
            if i == j:
                ax.hist(
                    samples[:, i],
                    bins=40,
                    color="steelblue",
                    alpha=0.7,
                    density=True,
                    edgecolor="none",
                )
                ax.axvline(MAP_vals[i], color="red", lw=1.5, ls="--")
                ax.axvline(np.median(samples[:, i]), color="orange", lw=1, ls=":")
            else:
                ax.scatter(samples[:, j], samples[:, i], s=1, alpha=0.15, c="steelblue")
                ax.plot(MAP_vals[j], MAP_vals[i], "r*", ms=6)

            if i == n_params - 1:
                ax.set_xlabel(JOINT_PARAM_NAMES[j], fontsize=8)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(JOINT_PARAM_NAMES[i], fontsize=8)
            else:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=6)

    fig.suptitle(f"Joint 4-Condition TMCMC Corner Plot ({tag})", fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    out = fig_dir / f"joint_corner_{tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════
#  Figure 2: Joint vs individual marginals
# ═══════════════════════════════════════════════════════════════
def plot_joint_vs_individual(joint_samples, indiv_samples, fig_dir, tag):
    """Joint vs individual posterior marginal histograms for each param."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_params = len(ALL_AIJ_INDICES)
    ncols = 5
    nrows = (n_params + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4 * nrows))
    axes = axes.flatten()

    colors = {"CS": "tab:blue", "CH": "tab:green", "DS": "tab:orange", "DH": "tab:red"}
    overlap_stats = {}

    for pi, aij_idx in enumerate(ALL_AIJ_INDICES):
        ax = axes[pi]
        pname = PARAM_NAMES[aij_idx]
        joint_col = joint_samples[:, pi]

        ax.hist(
            joint_col,
            bins=40,
            density=True,
            alpha=0.6,
            color="black",
            label="Joint",
            edgecolor="none",
        )

        for abbr, (samp, _) in indiv_samples.items():
            cond_key = CONDITION_KEYS[abbr]
            locks = CONDITION_AIJ_LOCKS.get(cond_key, set())
            if aij_idx in locks:
                continue  # param locked in this condition
            if samp.shape[1] > aij_idx:
                indiv_col = samp[:, aij_idx]
                ax.hist(
                    indiv_col,
                    bins=40,
                    density=True,
                    alpha=0.35,
                    color=colors[abbr],
                    label=abbr,
                    edgecolor="none",
                )
                bc = bhattacharyya_overlap_1d(joint_col, indiv_col)
                overlap_stats.setdefault(pname, {})[abbr] = float(bc)

        ax.set_title(pname, fontsize=10)
        if pi == 0:
            ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

    for k in range(pi + 1, len(axes)):
        axes[k].set_visible(False)

    fig.suptitle(f"Joint vs Individual Marginals ({tag})", fontsize=13, y=1.01)
    fig.tight_layout()
    out = fig_dir / f"joint_vs_individual_marginals_{tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")
    return overlap_stats


# ═══════════════════════════════════════════════════════════════
#  Figure 3: Per-condition ODE fit at joint MAP
# ═══════════════════════════════════════════════════════════════
def plot_per_condition_fit(theta_joint_MAP, fig_dir, tag):
    """Forward ODE solve at joint MAP for all 4 conditions."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from improved_5species_jit import BiofilmNewtonSolver5S

    # Expand joint MAP (15-dim) to full 20-dim using DH's b_i
    # Try loading existing full20 theta, otherwise use zeros for b_i
    theta_20 = np.zeros(20)
    for pi, aij_idx in enumerate(ALL_AIJ_INDICES):
        theta_20[aij_idx] = theta_joint_MAP[pi]

    # Load DH MAP for b_i values
    dh_runs = [INDIVIDUAL_RUNS["DH"], INDIVIDUAL_FALLBACKS["DH"]]
    for d in dh_runs:
        tf = d / "theta_MAP.json"
        if tf.exists():
            with open(tf) as f:
                dh_map = json.load(f)
            if "theta" in dh_map:
                dh_theta = np.array(dh_map["theta"])
                for bi_idx in [3, 4, 8, 9, 15]:
                    theta_20[bi_idx] = dh_theta[bi_idx]
            break

    solver = BiofilmNewtonSolver5S()
    species_labels = ["An", "So", "Fn", "Vd", "Pg"]
    conditions = {
        "Commensal_Static": "CS",
        "Commensal_HOBIC": "CH",
        "Dysbiotic_Static": "DS",
        "Dysbiotic_HOBIC": "DH",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    fit_rmse = {}
    for ci, (cond_key, abbr) in enumerate(conditions.items()):
        ax = axes[ci]
        try:
            result = solver.run_deterministic(theta_20)
            phi = result["phi"]  # (T, 5)
            t = np.arange(phi.shape[0])
            for si in range(5):
                ax.plot(t, phi[:, si], label=species_labels[si], lw=1.5)

            # Load experimental data if available
            exp_file = DATA_5SPECIES / "experimental" / f"{cond_key}.npz"
            if exp_file.exists():
                exp = np.load(exp_file)
                if "phi" in exp and "t" in exp:
                    for si in range(min(5, exp["phi"].shape[1])):
                        ax.scatter(exp["t"], exp["phi"][:, si], s=20, zorder=5)

            ax.set_title(f"{abbr} ({cond_key})", fontsize=11)
            ax.set_xlabel("Time step")
            ax.set_ylabel("φ")
            ax.legend(fontsize=7)
            ax.set_ylim(-0.05, 1.05)
        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"Error: {e}",
                transform=ax.transAxes,
                ha="center",
                fontsize=9,
                color="red",
            )
            ax.set_title(f"{abbr} (FAILED)")

    fig.suptitle(f"Joint MAP Forward ODE ({tag})", fontsize=13)
    fig.tight_layout()
    out = fig_dir / f"joint_MAP_fit_{tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════
#  Summary report
# ═══════════════════════════════════════════════════════════════
def write_summary(base_dir, overlap_stats, tag):
    """Write text summary of joint results."""
    diag_file = base_dir / "mcmc_diagnostics.json"
    ci_file = base_dir / "credible_intervals.json"
    map_file = base_dir / "theta_MAP.json"

    lines = [f"Joint 4-Condition TMCMC Summary ({tag})", "=" * 60]

    if diag_file.exists():
        with open(diag_file) as f:
            diag = json.load(f)
        lines.append(f"R-hat max: {diag['rhat_max']:.4f} (target < 1.1)")
        lines.append(f"ESS min: {diag['ess_min']:.1f} (target > 100)")
        lines.append(f"Converged: {diag['converged']}")
        lines.append(f"Chains: {diag['n_chains']}, Samples/chain: {diag['n_samples_per_chain']}")

    if ci_file.exists():
        with open(ci_file) as f:
            ci = json.load(f)
        lines.append("\nParameter  |  MAP     |  Median  |  95% HDI")
        lines.append("-" * 55)
        if map_file.exists():
            with open(map_file) as f:
                map_data = json.load(f)
            theta_map = map_data.get("theta_joint", [])
        else:
            theta_map = [None] * len(ALL_AIJ_INDICES)

        for pi, aij_idx in enumerate(ALL_AIJ_INDICES):
            pname = PARAM_NAMES[aij_idx]
            m = f"{theta_map[pi]:+.3f}" if theta_map[pi] is not None else "  N/A "
            med = f"{ci['median'][pi]:+.3f}"
            lo = f"{ci['hdi_lower'][pi]:+.3f}"
            hi = f"{ci['hdi_upper'][pi]:+.3f}"
            lines.append(f"  {pname:5s} |  {m} |  {med} |  [{lo}, {hi}]")

    if overlap_stats:
        lines.append("\nBhattacharyya Overlap (Joint vs Individual)")
        lines.append("-" * 55)
        for pname, conds in overlap_stats.items():
            parts = [f"{abbr}={bc:.3f}" for abbr, bc in conds.items()]
            lines.append(f"  {pname:5s}: {', '.join(parts)}")

        # Average overlap per condition
        lines.append("\nMean overlap per condition:")
        for abbr in ["CS", "CH", "DS", "DH"]:
            vals = [conds[abbr] for conds in overlap_stats.values() if abbr in conds]
            if vals:
                lines.append(f"  {abbr}: {np.mean(vals):.3f} (n={len(vals)} params)")

    summary_text = "\n".join(lines)
    out = base_dir / "analysis_summary.txt"
    with open(out, "w") as f:
        f.write(summary_text)
    print(f"\n{summary_text}")
    print(f"\nSummary saved: {out}")


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Analyze joint TMCMC results")
    parser.add_argument("run_dir", help="Path to joint run directory")
    parser.add_argument("--skip-coupling", action="store_true", help="Skip staggered coupling step")
    args = parser.parse_args()

    base_dir = Path(args.run_dir)
    if not base_dir.is_absolute():
        base_dir = MAIN_DIR / base_dir

    samples_file = base_dir / "samples.npy"
    logL_file = base_dir / "logL.npy"

    if not samples_file.exists():
        print(f"ERROR: {samples_file} not found. Run merge_joint_chains.py first.")
        sys.exit(1)

    samples = np.load(samples_file)
    logL = np.load(logL_file)
    n_total = samples.shape[0]

    # Tag from directory name
    tag = base_dir.name.replace("joint_4cond_", "")

    print(f"Loaded: {n_total} samples, {samples.shape[1]} params")
    print(f"logL range: [{logL.min():.2f}, {logL.max():.2f}]")

    # MAP
    map_idx = np.argmax(logL)
    MAP_vals = samples[map_idx]

    fig_dir = base_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # 1. Corner plot
    print("\n[1/4] Corner plot...")
    plot_corner(samples, MAP_vals, fig_dir, tag)

    # 2. Joint vs individual marginals
    print("\n[2/4] Joint vs individual marginals...")
    print("  Loading individual posteriors...")
    indiv = load_individual_samples()
    if indiv:
        overlap_stats = plot_joint_vs_individual(samples, indiv, fig_dir, tag)
    else:
        overlap_stats = {}
        print("  SKIPPED: no individual posteriors found")

    # 3. Per-condition fit
    print("\n[3/4] Per-condition ODE fit at joint MAP...")
    plot_per_condition_fit(MAP_vals, fig_dir, tag)

    # 4. Summary
    print("\n[4/4] Summary report...")
    write_summary(base_dir, overlap_stats, tag)

    print(f"\nAll analysis complete. Results in: {base_dir}")


if __name__ == "__main__":
    main()
