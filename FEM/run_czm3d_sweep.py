#!/usr/bin/env python3
"""
run_czm3d_sweep.py
==================
B3: 3D Cohesive Zone Model sweep across conditions + theta variants.

For each (condition × theta_variant):
  1. Use existing FEM field CSV  (from _posterior_abaqus or _material_sweep)
  2. abaqus cae  →  3D CZM job  →  ODB
  3. Parse <job>_czm_out.csv  →  RF_peak, Gc_eff, t_max_eff

Then plot:
  - fig_B3_rf_peak_conditions.png   RF_peak per condition × theta
  - fig_B3_gc_eff_conditions.png    Gc_eff vs DI_mean scatter
  - fig_B3_comparison.png           2×2 panel summary

Usage:
  python run_czm3d_sweep.py                  # all conditions, p50 DI field
  python run_czm3d_sweep.py --plot-only      # re-plot from czm_results.csv
  python run_czm3d_sweep.py --conditions commensal_static dh_baseline
  python run_czm3d_sweep.py --u-max 0.003   # smaller pull displacement
"""

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_PA_BASE = _HERE / "_posterior_abaqus"
_DI_BASE = _HERE / "_di_credible"
_MS_BASE = _HERE / "_material_sweep"
_OUT = _HERE / "_czm3d"

CONDITIONS = [
    "dh_baseline",
    "commensal_static",
    "commensal_hobic",
    "dysbiotic_static",
]

COND_LABELS = {
    "dh_baseline": "dh-baseline",
    "commensal_static": "Comm. Static",
    "commensal_hobic": "Comm. HOBIC",
    "dysbiotic_static": "Dysb. Static",
}

COND_COLORS = {
    "dh_baseline": "#d62728",
    "commensal_static": "#2ca02c",
    "commensal_hobic": "#1f77b4",
    "dysbiotic_static": "#ff7f0e",
}

# DI quantile to use from B1 output ("p05", "p50", "p95")
# p50 = median DI field (default)
DI_QTAG = "p50"

# Abaqus CZM parameters
T_MAX_0 = 1.0e6  # Pa  baseline interface normal strength
GC_MAX = 10.0  # J/m^2 baseline fracture energy
DI_EXPONENT = 2.0
DI_SCALE = 0.025778
U_MAX = 5.0e-3  # m
N_STEPS = 20

# ---------------------------------------------------------------------------
# Field CSV source resolution
# ---------------------------------------------------------------------------


def _get_field_csv(cond: str, qtag: str) -> Path | None:
    """
    Priority:
      1. _di_credible/{cond}/{qtag}_field.csv   (B1 output, preferred)
      2. _posterior_abaqus/{cond}/sample_0000/field.csv  (fallback)
    """
    b1_csv = _DI_BASE / cond / ("%s_field.csv" % qtag)
    if b1_csv.exists():
        return b1_csv

    # fallback: use MAP sample (sample_0000) from posterior ensemble
    fallback = _PA_BASE / cond / "sample_0000" / "field.csv"
    if fallback.exists():
        print("  [warn] B1 output not found; using sample_0000 field.csv")
        return fallback

    return None


# ---------------------------------------------------------------------------
# Abaqus runner
# ---------------------------------------------------------------------------


def _run_czm_job(
    field_csv: Path,
    job_name: str,
    u_max: float,
    out_dir: Path,
) -> dict | None:
    """Run one 3D CZM Abaqus job (resume-safe). Return result dict."""
    done_flag = out_dir / "done.flag"
    res_json = out_dir / "result.json"
    czm_csv = _HERE / ("%s_czm_out.csv" % job_name)

    if done_flag.exists() and res_json.exists():
        with res_json.open() as f:
            return json.load(f)

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "abaqus",
        "cae",
        "noGUI=%s" % str(_HERE / "abaqus_biofilm_cohesive_3d.py"),
        "--",
        "--field-csv",
        str(field_csv),
        "--job-name",
        job_name,
        "--t-max",
        "%.6g" % T_MAX_0,
        "--gc-max",
        "%.6g" % GC_MAX,
        "--di-exponent",
        "%.2f" % DI_EXPONENT,
        "--di-scale",
        "%.6f" % DI_SCALE,
        "--u-max",
        "%.6g" % u_max,
        "--n-steps",
        str(N_STEPS),
    ]
    t0 = time.perf_counter()
    ret = subprocess.run(cmd, cwd=str(_HERE), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - t0

    if ret.returncode != 0:
        print("    [warn] Abaqus CAE rc=%d" % ret.returncode)
        return None

    # Parse CZM output CSV
    result = None
    if czm_csv.exists():
        with czm_csv.open() as f:
            rows = list(csv.DictReader(f))
        if rows:
            r = rows[0]
            result = {
                "di_mean": float(r["di_mean"]),
                "t_max_eff": float(r["t_max_eff"]),
                "gc_eff": float(r["gc_eff"]),
                "u_max": float(r["u_max"]),
                "rf_peak": float(r["rf_peak"]),
                "rf_at_umax": float(r["rf_at_umax"]),
                "elapsed_s": elapsed,
            }

    if result is None:
        print("    [warn] CZM output CSV not found or empty.")
        return None

    with res_json.open("w") as f:
        json.dump(result, f, indent=2)
    done_flag.touch()

    print(
        "    done in %.1fs  di=%.4f  Gc=%.3g J/m2  RF_peak=%.4g N"
        % (elapsed, result["di_mean"], result["gc_eff"], result["rf_peak"])
    )
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS, choices=CONDITIONS)
    ap.add_argument(
        "--di-qtag",
        default=DI_QTAG,
        choices=["p05", "p50", "p95"],
        help="Which DI quantile field to use (default: p50)",
    )
    ap.add_argument(
        "--u-max",
        type=float,
        default=U_MAX,
        help="Max pull displacement in m (default %.4g)" % U_MAX,
    )
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()

    _OUT.mkdir(parents=True, exist_ok=True)
    csv_path = _OUT / "czm_results.csv"

    results = []

    if not args.plot_only:
        for cond in args.conditions:
            print("\n[%s]" % cond)
            field_csv = _get_field_csv(cond, args.di_qtag)
            if field_csv is None:
                print("  [skip] no field CSV found for condition=%s" % cond)
                continue

            job_name = "czm3d_%s_%s" % (cond[:4], args.di_qtag)
            out_dir = _OUT / cond

            print("  field: %s" % field_csv.relative_to(_HERE))
            rec = _run_czm_job(field_csv, job_name, args.u_max, out_dir)
            if rec:
                results.append({"condition": cond, "di_qtag": args.di_qtag, **rec})

        # Save CSV
        if results:
            with csv_path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=results[0].keys())
                w.writeheader()
                w.writerows(results)
            print("\n[done] %d results → %s" % (len(results), csv_path))
        else:
            # Try loading existing
            if csv_path.exists():
                with csv_path.open() as f:
                    for r in csv.DictReader(f):
                        results.append(
                            {
                                k: (float(v) if k not in ("condition", "di_qtag") else v)
                                for k, v in r.items()
                            }
                        )
            if not results:
                print("[warn] no results.")
                return

    else:
        if not csv_path.exists():
            print("[plot-only] no czm_results.csv found.")
            return
        with csv_path.open() as f:
            for r in csv.DictReader(f):
                results.append(
                    {
                        k: (float(v) if k not in ("condition", "di_qtag") else v)
                        for k, v in r.items()
                    }
                )

    _plot_results(results)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_results(results: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = _OUT / "figures"
    fig_dir.mkdir(exist_ok=True)

    conditions = [r["condition"] for r in results]
    di_means = [r["di_mean"] for r in results]
    gc_effs = [r["gc_eff"] for r in results]
    rf_peaks = [r["rf_peak"] for r in results]
    t_maxs = [r["t_max_eff"] for r in results]

    colors = [COND_COLORS.get(c, "gray") for c in conditions]
    labels = [COND_LABELS.get(c, c) for c in conditions]

    # ── Fig B3-1: RF_peak bar chart ───────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)

    ax = axes[0]
    bars = ax.bar(
        range(len(results)), rf_peaks, color=colors, alpha=0.85, edgecolor="k", linewidth=0.5
    )
    for bar, v in zip(bars, rf_peaks):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.02,
            "%.2g N" % v,
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Peak reaction force RF_peak (N)")
    ax.set_title("B3: Peak Pull Force per Condition")
    ax.grid(axis="y", alpha=0.35, linestyle="--")

    # ── Fig B3-2: Gc_eff vs DI_mean scatter ──────────────────────────────
    ax = axes[1]
    for i, (c, di, gc) in enumerate(zip(conditions, di_means, gc_effs)):
        ax.scatter(
            di,
            gc,
            color=COND_COLORS.get(c, "gray"),
            s=120,
            zorder=5,
            edgecolors="k",
            linewidths=0.5,
        )
        ax.annotate(
            COND_LABELS.get(c, c), (di, gc), textcoords="offset points", xytext=(6, 4), fontsize=8
        )

    # overlay theoretical curve
    di_curve = np.linspace(0, max(di_means) * 1.3, 100)
    di_scale = DI_SCALE
    r_curve = np.clip(di_curve / di_scale, 0, 1)
    gc_curve = GC_MAX * (1 - r_curve) ** DI_EXPONENT
    ax.plot(di_curve, gc_curve, "k--", lw=1.2, alpha=0.6, label=r"$G_c = G_{c,0}(1-DI/s)^n$")
    ax.set_xlabel("Mean DI (bottom layers)", fontsize=10)
    ax.set_ylabel("Effective Gc (J/m²)", fontsize=10)
    ax.set_title("B3: Gc_eff vs DI_mean")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, linestyle="--")

    fig.suptitle(
        "B3: 3D Cohesive Zone Model  –  Interface Strength vs Condition",
        fontsize=12,
        fontweight="bold",
    )
    out = fig_dir / "fig_B3_czm_summary.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("[plot] %s" % out.name)

    # ── Fig B3-3: 4-panel (di_mean, t_max, gc, RF) ───────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    panel_data = [
        (di_means, "DI mean (bottom layers)", "B3: Interface DI"),
        (t_maxs, "t_max_eff (Pa)", "B3: Effective Normal Strength"),
        (gc_effs, "Gc_eff (J/m²)", "B3: Effective Fracture Energy"),
        (rf_peaks, "RF_peak (N)", "B3: Peak Pull Force"),
    ]
    for ax, (vals, ylabel, title) in zip(axes.flat, panel_data):
        bars = ax.bar(
            range(len(results)), vals, color=colors, alpha=0.85, edgecolor="k", linewidth=0.5
        )
        ax.set_xticks(range(len(results)))
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02,
                "%.3g" % v,
                ha="center",
                va="bottom",
                fontsize=7,
            )

    fig.suptitle("B3: 3D CZM  –  Condition Comparison  (4 metrics)", fontsize=12, fontweight="bold")
    out = fig_dir / "fig_B3_czm_4panel.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("[plot] %s" % out.name)
    print("\n[plot] all → %s/" % fig_dir)


if __name__ == "__main__":
    main()
