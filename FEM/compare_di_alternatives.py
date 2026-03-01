#!/usr/bin/env python3
"""
compare_di_alternatives.py
==========================
学術的妥当性順に実装した DI 代替モデルの比較検討。

Models (学術的妥当性順):
  1. Shannon DI (既存) — 生態学の標準
  2. Simpson DI — dominance に敏感、生態学で広く使用
  3. Voigt mixing — 有効媒質理論、変分整合的
  4. Gini evenness — 不均等度の直接表現
  5. Pielou evenness — 種数に正規化された均等度
  6. Reuss mixing — 直列負荷の下限
  7. φ_Pg, Virulence — メカニズムベース（参考）

Usage:
  python compare_di_alternatives.py              # synthetic test data
  python compare_di_alternatives.py --samples    # posterior samples (if available)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "tmcmc" / "program2602"))
sys.path.insert(0, str(_HERE))

from material_models import (
    compute_all_E,
    compute_di,
    compute_di_gini,
    compute_di_pielou,
    compute_di_simpson,
    compute_E_di,
    compute_E_phi_pg,
    compute_E_reuss,
    compute_E_virulence,
    compute_E_voigt,
)

OUT_DIR = _HERE / "_di_alternatives_comparison"

# Synthetic test compositions (paper-consistent)
# Commensal: uniform; Dysbiotic: S. oralis dominant (Hamilton ODE の attractor)
SYNTHETIC_PHI = {
    "commensal": np.array([0.20, 0.20, 0.20, 0.20, 0.20]),
    "dysbiotic_So": np.array([0.80, 0.05, 0.05, 0.05, 0.05]),  # S. oralis dominant
    "dysbiotic_Pg": np.array([0.05, 0.05, 0.05, 0.05, 0.80]),  # P. gingivalis dominant
}


def run_synthetic_comparison() -> dict:
    """Synthetic compositions で全モデルを比較。"""
    results = {}
    for label, phi in SYNTHETIC_PHI.items():
        phi_2d = phi.reshape(1, 5)
        res = compute_all_E(phi_2d, mode="all", di_scale=1.0)
        results[label] = {}
        for k, v in res.items():
            arr = np.asarray(v)
            results[label][k] = float(arr.flat[0]) if arr.size == 1 else arr
    return results


def run_from_posterior(n_samples: int = 50) -> dict | None:
    """Posterior samples から ODE を回し、全モデルで E を計算。"""
    run_dir = _ROOT / "data_5species" / "_runs"
    samples_path = run_dir / "dh_baseline" / "samples.npy"
    if not samples_path.exists():
        samples_path = run_dir / "commensal_static" / "samples.npy"
    if not samples_path.exists():
        return None

    sys.path.insert(0, str(_ROOT / "tmcmc" / "program2602"))
    from improved_5species_jit import BiofilmNewtonSolver5S

    samples = np.load(samples_path)
    N = min(n_samples, len(samples))
    idx = np.linspace(0, len(samples) - 1, N, dtype=int)
    thetas = samples[idx]

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

    phi_arr = []
    for i in range(N):
        try:
            _, g_arr = solver.solve(thetas[i])
            phi_arr.append(g_arr[-1, 0:5])
        except Exception:
            phi_arr.append(np.full(5, np.nan))

    phi_arr = np.array(phi_arr)
    valid = ~np.isnan(phi_arr[:, 0])
    phi_arr = phi_arr[valid]

    if len(phi_arr) == 0:
        return None

    res = compute_all_E(phi_arr, mode="all", di_scale=1.0)
    return {
        "condition": "posterior_samples",
        "n_valid": int(valid.sum()),
        "phi_final": phi_arr,
        **{k: v for k, v in res.items()},
    }


def compute_discrimination_metrics(results: dict) -> dict:
    """Commensal vs Dysbiotic の差別化指標を計算。"""
    if "commensal" not in results or "dysbiotic_So" not in results:
        return {}

    metrics = {}
    e_keys = [
        "E_di",
        "E_simpson",
        "E_gini",
        "E_pielou",
        "E_voigt",
        "E_reuss",
        "E_phi_pg",
        "E_virulence",
    ]
    for key in e_keys:
        if key not in results.get("commensal", {}):
            continue
        e_comm = results["commensal"][key]
        e_dysb = results["dysbiotic_So"][key]
        if np.isscalar(e_comm) and np.isscalar(e_dysb):
            ratio = e_comm / (e_dysb + 1e-12)
            diff = e_comm - e_dysb
            metrics[key] = {
                "E_commensal": float(e_comm),
                "E_dysbiotic": float(e_dysb),
                "ratio": float(ratio),
                "diff_Pa": float(diff),
            }
    return metrics


def plot_comparison(results: dict, metrics: dict, outdir: Path) -> str:
    """比較図を生成。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)

    models = [
        ("E_di", "DI (Shannon)", "#1f77b4"),
        ("E_simpson", "DI (Simpson)", "#ff7f0e"),
        ("E_voigt", "Voigt mixing", "#2ca02c"),
        ("E_gini", "DI (Gini)", "#d62728"),
        ("E_pielou", "DI (Pielou)", "#9467bd"),
        ("E_reuss", "Reuss mixing", "#8c564b"),
        ("E_phi_pg", "φ_Pg (Hill)", "#e377c2"),
        ("E_virulence", "Virulence", "#7f7f7f"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    def _scalar(val):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return np.nan
        arr = np.asarray(val)
        return float(arr.flat[0]) if arr.size >= 1 else np.nan

    # (a) Bar: E by model, commensal vs dysbiotic
    ax = axes[0, 0]
    x = np.arange(len(models))
    w = 0.35
    e_comm = [_scalar(results["commensal"].get(m[0])) for m in models]
    e_dysb = [_scalar(results["dysbiotic_So"].get(m[0])) for m in models]
    ax.bar(x - w / 2, e_comm, w, label="Commensal", color="#2ca02c", alpha=0.8)
    ax.bar(x + w / 2, e_dysb, w, label="Dysbiotic (So)", color="#d62728", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in models], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("E [Pa]")
    ax.set_title("(a) Elastic Modulus by Model")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    # (b) Discrimination ratio
    ax = axes[0, 1]
    ratios = [metrics.get(m[0], {}).get("ratio", 1.0) for m in models]
    colors = [m[2] for m in models]
    bars = ax.barh(range(len(models)), ratios, color=colors, alpha=0.8)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([m[1] for m in models], fontsize=9)
    ax.set_xlabel("E_commensal / E_dysbiotic")
    ax.set_title("(b) Condition Discrimination Ratio")
    ax.set_xscale("log")
    ax.axvline(1.0, color="k", ls="--", lw=0.8)
    ax.grid(alpha=0.3, axis="x")

    # (c) DI values comparison (Shannon, Simpson, Gini, Pielou)
    ax = axes[1, 0]
    di_models = [
        ("DI", "Shannon", "#1f77b4"),
        ("DI_simpson", "Simpson", "#ff7f0e"),
        ("DI_gini", "Gini", "#d62728"),
        ("DI_pielou", "Pielou", "#9467bd"),
    ]
    x = np.arange(3)
    w = 0.2
    for i, (key, label, col) in enumerate(di_models):
        vals = []
        for cond in ["commensal", "dysbiotic_So", "dysbiotic_Pg"]:
            v = results.get(cond, {}).get(key, np.nan)
            vals.append(_scalar(v))
        ax.bar(x + (i - 1.5) * w, vals, w, label=label, color=col, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Commensal", "Dysb.(So)", "Dysb.(Pg)"])
    ax.set_ylabel("DI value")
    ax.set_title("(c) Diversity Index Comparison")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    # (d) Summary table
    ax = axes[1, 1]
    ax.axis("off")
    rows = []
    for m in models:
        key = m[0]
        if key not in metrics:
            continue
        mm = metrics[key]
        rows.append(
            [
                m[1],
                f"{mm['E_commensal']:.0f}",
                f"{mm['E_dysbiotic']:.0f}",
                f"{mm['ratio']:.1f}×",
            ]
        )
    if rows:
        table = ax.table(
            cellText=rows,
            colLabels=["Model", "E_comm [Pa]", "E_dysb [Pa]", "Ratio"],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
    ax.set_title("(d) Summary", fontsize=12, weight="bold", pad=10)

    fig.suptitle(
        "DI Alternative Models: Academic Plausibility Order Comparison",
        fontsize=14,
        weight="bold",
        y=1.02,
    )
    plt.tight_layout()

    outpath = outdir / "fig_di_alternatives_comparison.png"
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(outpath)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples",
        action="store_true",
        help="Use posterior samples if available (otherwise synthetic)",
    )
    parser.add_argument("--n-samples", type=int, default=50, help="Samples when using posterior")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DI Alternative Models Comparison")
    print("=" * 60)

    if args.samples:
        print("\nAttempting to load posterior samples...")
        results_raw = run_from_posterior(args.n_samples)
        if results_raw is not None:
            # Convert to same format as synthetic
            phi_mean = results_raw["phi_final"].mean(axis=0)
            results = {
                "commensal": {
                    k: float(np.mean(v))
                    for k, v in results_raw.items()
                    if isinstance(v, np.ndarray) and v.ndim == 1
                },
                "dysbiotic_So": {
                    k: float(np.mean(v))
                    for k, v in results_raw.items()
                    if isinstance(v, np.ndarray) and v.ndim == 1
                },
            }
            # Actually we need per-composition results. For posterior, we have distribution.
            # Use mean phi as "dysbiotic" and synthetic commensal for comparison
            res_all = compute_all_E(results_raw["phi_final"], mode="all", di_scale=1.0)
            results = {
                "commensal": {
                    k: float(v[0]) if hasattr(v, "__len__") and len(v) > 0 else float(v)
                    for k, v in compute_all_E(
                        SYNTHETIC_PHI["commensal"].reshape(1, 5), di_scale=1.0
                    ).items()
                },
                "dysbiotic_So": {
                    k: float(np.mean(v)) for k, v in res_all.items() if isinstance(v, np.ndarray)
                },
                "dysbiotic_Pg": {
                    k: float(v[0]) if hasattr(v, "__len__") and len(v) > 0 else float(v)
                    for k, v in compute_all_E(
                        SYNTHETIC_PHI["dysbiotic_Pg"].reshape(1, 5), di_scale=1.0
                    ).items()
                },
            }
            # Fix: res_all has arrays, we need scalar for dysbiotic_So
            for k, v in res_all.items():
                if isinstance(v, np.ndarray) and v.size > 0:
                    results["dysbiotic_So"][k] = float(np.mean(v))
            print(f"  Loaded {results_raw['n_valid']} posterior samples")
        else:
            print("  No posterior samples found, using synthetic data")
            results = run_synthetic_comparison()
    else:
        print("\nUsing synthetic test compositions")
        results = run_synthetic_comparison()

    metrics = compute_discrimination_metrics(results)

    print("\n--- Results ---")
    for cond, data in results.items():
        print(f"\n{cond}:")
        for k in ["E_di", "E_simpson", "E_voigt", "E_gini", "E_pielou", "E_reuss"]:
            if k in data:
                print(f"  {k}: {data[k]:.1f} Pa")

    print("\n--- Discrimination (Commensal vs Dysbiotic So) ---")
    for key, m in metrics.items():
        print(
            f"  {key}: E_comm={m['E_commensal']:.0f}, E_dysb={m['E_dysbiotic']:.0f}, ratio={m['ratio']:.1f}×"
        )

    print("\n--- Interpretation ---")
    print("  DI-based (Shannon, Simpson, Gini, Pielou): diversity loss → E↓ (correct direction)")
    print(
        "  Voigt/Reuss: composition-based; dysbiotic So (E_So high) → E↑ (opposite for diversity)"
    )
    print("  φ_Pg, Virulence: pathogen-based; φ_Pg low in So-dominant → no discrimination")

    with open(OUT_DIR / "comparison_results.json", "w") as f:
        json.dump(
            {
                "results": {
                    k: {
                        kk: (
                            float(vv)
                            if np.isscalar(vv)
                            else vv.tolist() if hasattr(vv, "tolist") else vv
                        )
                        for kk, vv in v.items()
                    }
                    for k, v in results.items()
                },
                "metrics": metrics,
            },
            f,
            indent=2,
        )

    fig_path = plot_comparison(results, metrics, OUT_DIR)
    print(f"\nFigure: {fig_path}")
    print(f"Output: {OUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
