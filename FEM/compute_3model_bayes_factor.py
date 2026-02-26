#!/usr/bin/env python3
"""
compute_3model_bayes_factor.py
==============================
3-Model Material Model Discrimination Analysis (Fig 18 candidate)

Posterior samples (θ) → Hamilton ODE → φ_species → E under 3 models
→ Pairwise Bhattacharyya distance & discrimination Bayes factor

Models:
  M1: E(DI)       — Shannon entropy
  M2: E(φ_Pg)     — P. gingivalis fraction
  M3: E(V)        — Virulence index (weighted Pg+Fn)

Usage:
  python compute_3model_bayes_factor.py [--n-samples 50] [--workers 4]
"""
import json
import sys
import time
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "tmcmc" / "program2602"))
sys.path.insert(0, str(_ROOT / "FEM"))

from improved_5species_jit import BiofilmNewtonSolver5S
from material_models import compute_di, compute_E_di, compute_E_phi_pg, compute_E_virulence

# ── Config ───────────────────────────────────────────────────────────────────
RUNS_DIR = _ROOT / "data_5species" / "_runs"
OUT_DIR = _HERE / "_3model_bayes_factor"

CONDITIONS = {
    "commensal_static": {
        "samples": RUNS_DIR / "commensal_static_posterior" / "samples.npy",
        "label": "Commensal Static",
        "short": "CS",
        "color": "#2ca02c",
    },
    "commensal_hobic": {
        "samples": RUNS_DIR / "commensal_hobic_posterior" / "samples.npy",
        "label": "Commensal HOBIC",
        "short": "CH",
        "color": "#17becf",
    },
    "dh_baseline": {
        "samples": RUNS_DIR / "dh_baseline" / "samples.npy",
        "label": "Dysbiotic HOBIC",
        "short": "DH",
        "color": "#d62728",
    },
}


def _solve_single(theta):
    """Run Hamilton ODE for a single theta → return phi_final (5,)."""
    solver = BiofilmNewtonSolver5S(
        dt=1e-4,
        maxtimestep=750,
        active_species=[0, 1, 2, 3, 4],
        c_const=25.0,
        alpha_const=0.0,
        phi_init=0.02,
        K_hill=0.05,
        n_hill=4.0,
        use_numba=True,
    )
    try:
        _, g_arr = solver.solve(theta)
        return g_arr[-1, 0:5]  # (5,) species fractions
    except Exception:
        return np.full(5, np.nan)


def run_condition(cond_name, n_samples, workers):
    """ODE forward for n_samples from posterior → {phi_final, DI, E_di, E_phipg, E_vir}."""
    meta = CONDITIONS[cond_name]
    samples = np.load(meta["samples"])  # (N, 20)
    N = samples.shape[0]

    # Subsample if needed
    if n_samples < N:
        idx = np.linspace(0, N - 1, n_samples, dtype=int)
        samples = samples[idx]
    else:
        n_samples = N

    print(f"  [{cond_name}] Running ODE for {n_samples} samples (workers={workers})...")
    t0 = time.time()

    if workers <= 1:
        phi_arr = np.array([_solve_single(samples[i]) for i in range(n_samples)])
    else:
        phi_arr = np.full((n_samples, 5), np.nan)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_solve_single, samples[i]): i for i in range(n_samples)}
            for fut in as_completed(futures):
                i = futures[fut]
                phi_arr[i] = fut.result()

    elapsed = time.time() - t0
    print(f"  [{cond_name}] Done in {elapsed:.1f}s")

    # Remove NaN rows (failed solves)
    valid = ~np.isnan(phi_arr[:, 0])
    phi_arr = phi_arr[valid]

    # Compute DI and E under 3 models
    di = compute_di(phi_arr)  # (n,)
    E_di = compute_E_di(di, di_scale=1.0)  # 0D scale
    E_phipg = compute_E_phi_pg(phi_arr)  # (n,)
    E_vir = compute_E_virulence(phi_arr)  # (n,)

    return {
        "condition": cond_name,
        "n_valid": int(valid.sum()),
        "phi_final": phi_arr,
        "di": di,
        "E_di": E_di,
        "E_phipg": E_phipg,
        "E_vir": E_vir,
        "elapsed_s": elapsed,
    }


# ── Discrimination Metrics ───────────────────────────────────────────────────


def bhattacharyya_distance(x, y):
    """Bhattacharyya distance between two 1D distributions (Gaussian approx)."""
    mu1, s1 = np.mean(x), np.std(x) + 1e-12
    mu2, s2 = np.mean(y), np.std(y) + 1e-12
    sigma = 0.5 * (s1**2 + s2**2)
    return 0.125 * (mu1 - mu2) ** 2 / sigma + 0.5 * np.log(sigma / (s1 * s2 + 1e-30))


def cohens_d(x, y):
    """Effect size (Cohen's d)."""
    n1, n2 = len(x), len(y)
    s_pool = np.sqrt(((n1 - 1) * np.var(x) + (n2 - 1) * np.var(y)) / (n1 + n2 - 2))
    return abs(np.mean(x) - np.mean(y)) / (s_pool + 1e-12)


def kruskal_h(*groups):
    """Kruskal-Wallis H statistic (non-parametric ANOVA)."""
    from scipy.stats import kruskal

    stat, pval = kruskal(*groups)
    return stat, pval


def compute_discrimination_metrics(results):
    """Compute pairwise and overall discrimination for each model."""
    conds = list(results.keys())
    models = {"DI": "E_di", "φ_Pg": "E_phipg", "Virulence": "E_vir"}

    metrics = {}
    for model_name, key in models.items():
        E_arrays = {c: results[c][key] for c in conds}

        # Pairwise Bhattacharyya & Cohen's d
        pairwise = {}
        for i, c1 in enumerate(conds):
            for c2 in conds[i + 1 :]:
                pair = f"{CONDITIONS[c1]['short']}-{CONDITIONS[c2]['short']}"
                pairwise[pair] = {
                    "bhattacharyya": float(bhattacharyya_distance(E_arrays[c1], E_arrays[c2])),
                    "cohens_d": float(cohens_d(E_arrays[c1], E_arrays[c2])),
                    "mean_diff": float(abs(np.mean(E_arrays[c1]) - np.mean(E_arrays[c2]))),
                }

        # Overall Kruskal-Wallis
        groups = [E_arrays[c] for c in conds]
        kw_stat, kw_pval = kruskal_h(*groups)

        # E range (max mean - min mean)
        means = [np.mean(E_arrays[c]) for c in conds]
        e_range = max(means) - min(means)
        e_ratio = max(means) / (min(means) + 1e-12)

        metrics[model_name] = {
            "pairwise": pairwise,
            "kruskal_wallis_H": float(kw_stat),
            "kruskal_wallis_p": float(kw_pval),
            "E_range_Pa": float(e_range),
            "E_ratio": float(e_ratio),
            "per_condition": {
                c: {
                    "mean": float(np.mean(E_arrays[c])),
                    "std": float(np.std(E_arrays[c])),
                    "ci90": [
                        float(np.percentile(E_arrays[c], 5)),
                        float(np.percentile(E_arrays[c], 95)),
                    ],
                }
                for c in conds
            },
        }

    return metrics


# ── Plotting ─────────────────────────────────────────────────────────────────


def plot_3model_comparison(results, metrics, outdir):
    """Generate publication-quality 3-model discrimination figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conds = list(results.keys())
    models = [("DI", "E_di"), ("φ_Pg", "E_phipg"), ("Virulence", "E_vir")]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), gridspec_kw={"height_ratios": [2, 1]})

    # ── Row 1: E distribution violin/swarm for each model ────────────────
    for j, (model_name, key) in enumerate(models):
        ax = axes[0, j]

        positions = []
        data_list = []
        colors = []
        labels = []
        for i, c in enumerate(conds):
            E = results[c][key]
            data_list.append(E)
            positions.append(i)
            colors.append(CONDITIONS[c]["color"])
            labels.append(CONDITIONS[c]["short"])

        # Violin plot
        parts = ax.violinplot(
            data_list, positions=positions, showmeans=True, showextrema=True, showmedians=True
        )
        for pc, col in zip(parts["bodies"], colors):
            pc.set_facecolor(col)
            pc.set_alpha(0.5)

        # Scatter individual points
        for i, (pos, data, col) in enumerate(zip(positions, data_list, colors)):
            jitter = np.random.normal(0, 0.04, len(data))
            ax.scatter(pos + jitter, data, s=8, alpha=0.3, color=col, zorder=3)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel("E [Pa]", fontsize=11)
        ax.set_title(f"Model: {model_name}", fontsize=13, fontweight="bold")
        ax.grid(alpha=0.2, axis="y")

        # Annotate range
        m = metrics[model_name]
        ax.annotate(
            f"Range: {m['E_range_Pa']:.0f} Pa\nRatio: {m['E_ratio']:.1f}×\nKW p={m['kruskal_wallis_p']:.2e}",
            xy=(0.97, 0.97),
            xycoords="axes fraction",
            fontsize=9,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9),
        )

    # ── Row 2: Bhattacharyya distance comparison bars ─────────────────────
    for j, (model_name, key) in enumerate(models):
        ax = axes[1, j]

        pairs = list(metrics[model_name]["pairwise"].keys())
        bhatt = [metrics[model_name]["pairwise"][p]["bhattacharyya"] for p in pairs]
        cohen = [metrics[model_name]["pairwise"][p]["cohens_d"] for p in pairs]

        x = np.arange(len(pairs))
        w = 0.35
        bars1 = ax.bar(x - w / 2, bhatt, w, label="Bhattacharyya D", color="#4c72b0", alpha=0.8)
        ax2 = ax.twinx()
        bars2 = ax2.bar(x + w / 2, cohen, w, label="Cohen's d", color="#dd8452", alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(pairs, fontsize=10)
        ax.set_ylabel("Bhattacharyya D", fontsize=10, color="#4c72b0")
        ax2.set_ylabel("Cohen's d", fontsize=10, color="#dd8452")
        ax.set_title(f"{model_name}: Pairwise Discrimination", fontsize=11)

        # Cohen's d thresholds
        for thresh, lbl in [(0.2, "small"), (0.8, "large"), (1.2, "very large")]:
            ax2.axhline(thresh, ls=":", color="#dd8452", alpha=0.3, lw=0.8)

        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

    fig.suptitle(
        "Three Material Model Discrimination:\n"
        "DI (entropy) vs φ_Pg (pathogen fraction) vs Virulence (Pg+Fn weighted)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    outpath = outdir / "fig_3model_discrimination.png"
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figure: {outpath}")
    return str(outpath)


def plot_summary_table(metrics, outdir):
    """Generate summary comparison table as figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")

    models = ["DI", "φ_Pg", "Virulence"]
    conds_short = ["CS", "CH", "DH"]

    # Table data
    header = ["Model", "KW H (p-value)", "E range [Pa]", "E ratio"]
    for cs in conds_short:
        header.append(f"E_{cs} [Pa]")
    header.append("Max Cohen's d")

    rows = []
    for m in models:
        d = metrics[m]
        # Find max Cohen's d
        max_d = max(d["pairwise"][p]["cohens_d"] for p in d["pairwise"])
        row = [
            m,
            f"{d['kruskal_wallis_H']:.1f} ({d['kruskal_wallis_p']:.1e})",
            f"{d['E_range_Pa']:.0f}",
            f"{d['E_ratio']:.1f}×",
        ]
        for c_full in ["commensal_static", "commensal_hobic", "dh_baseline"]:
            pc = d["per_condition"][c_full]
            row.append(f"{pc['mean']:.0f} ± {pc['std']:.0f}")
        row.append(f"{max_d:.2f}")
        rows.append(row)

    table = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)

    # Color best model row (DI = row 1)
    for j in range(len(header)):
        table[1, j].set_facecolor("#d4edda")  # green for DI
        table[2, j].set_facecolor("#f8d7da")  # red for φ_Pg
        table[3, j].set_facecolor("#f8d7da")  # red for Virulence
        table[0, j].set_facecolor("#e2e3e5")  # header gray

    ax.set_title("Material Model Discrimination Summary", fontsize=13, fontweight="bold", pad=20)

    outpath = outdir / "table_3model_summary.png"
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Table:  {outpath}")
    return str(outpath)


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-samples",
        type=int,
        default=50,
        help="Number of posterior samples per condition (default: 50)",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Parallel workers for ODE solves (default: 4)"
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("3-Model Material Model Discrimination Analysis")
    print("=" * 60)
    print(f"  Samples per condition: {args.n_samples}")
    print(f"  Workers: {args.workers}")

    # 1. Run ODE for all conditions
    results = {}
    total_t0 = time.time()
    for cond in CONDITIONS:
        results[cond] = run_condition(cond, args.n_samples, args.workers)
    total_elapsed = time.time() - total_t0
    print(f"\nTotal ODE time: {total_elapsed:.1f}s")

    # 2. Compute discrimination metrics
    print("\nComputing discrimination metrics...")
    metrics = compute_discrimination_metrics(results)

    # 3. Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for model_name in ["DI", "φ_Pg", "Virulence"]:
        m = metrics[model_name]
        print(f"\n--- {model_name} Model ---")
        print(f"  KW H={m['kruskal_wallis_H']:.1f}, p={m['kruskal_wallis_p']:.2e}")
        print(f"  E range: {m['E_range_Pa']:.0f} Pa, ratio: {m['E_ratio']:.1f}×")
        for c in CONDITIONS:
            pc = m["per_condition"][c]
            print(
                f"    {CONDITIONS[c]['short']}: E = {pc['mean']:.1f} ± {pc['std']:.1f} Pa "
                f"[{pc['ci90'][0]:.0f}, {pc['ci90'][1]:.0f}]"
            )
        print("  Pairwise:")
        for pair, pv in m["pairwise"].items():
            print(
                f"    {pair}: Bhattacharyya={pv['bhattacharyya']:.2f}, "
                f"Cohen's d={pv['cohens_d']:.2f}, ΔE={pv['mean_diff']:.0f} Pa"
            )

    # 4. Pseudo Bayes Factor interpretation
    # B = exp(D_bhatt_DI - D_bhatt_altmodel) for worst-case pair
    print("\n" + "=" * 60)
    print("PSEUDO BAYES FACTOR (DI vs alternatives)")
    print("=" * 60)
    # Use CS-DH pair (extreme case)
    target_pair = "CS-DH"
    for alt in ["φ_Pg", "Virulence"]:
        d_di = metrics["DI"]["pairwise"][target_pair]["bhattacharyya"]
        d_alt = metrics[alt]["pairwise"][target_pair]["bhattacharyya"]
        pseudo_bf = np.exp(d_di - d_alt)
        print(
            f"  DI vs {alt} ({target_pair}): "
            f"B_DI = exp({d_di:.2f} - {d_alt:.2f}) = {pseudo_bf:.1f}"
        )
        if pseudo_bf > 100:
            interp = "decisive (>100)"
        elif pseudo_bf > 10:
            interp = "strong (>10)"
        elif pseudo_bf > 3:
            interp = "moderate (>3)"
        else:
            interp = "weak (<3)"
        print(f"    Jeffreys' interpretation: {interp}")

    # 5. Save metrics
    metrics_path = OUT_DIR / "3model_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics: {metrics_path}")

    # 6. Generate figures
    print("\nGenerating figures...")
    fig_path = plot_3model_comparison(results, metrics, OUT_DIR)
    tab_path = plot_summary_table(metrics, OUT_DIR)

    # 7. Save numpy arrays for reproducibility
    for c in results:
        np.savez(
            OUT_DIR / f"{c}_3model.npz",
            phi_final=results[c]["phi_final"],
            di=results[c]["di"],
            E_di=results[c]["E_di"],
            E_phipg=results[c]["E_phipg"],
            E_vir=results[c]["E_vir"],
        )

    print(f"\nAll outputs in: {OUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
