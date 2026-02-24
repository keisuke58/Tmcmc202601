#!/usr/bin/env python3
"""
plot_stress_comparison.py
==========================
A3: von Mises / principal stress extraction, visualization, and
commensal vs dysbiotic stress distribution comparison figures.

Generates paper-quality comparison figures:
  Fig A: von Mises stress violin/box plots across conditions
  Fig B: Principal stress distribution per condition
  Fig C: DI → E_eff → stress cascade diagram
  Fig D: Spatial stress profile comparison (if ODB data available)

Data sources (in priority order):
  [1] _abaqus_auto_jobs/{cond}_T23/result_*.json  (from run_abaqus_auto.py)
  [2] _posterior_abaqus/{cond}/sample_*/stress.json (from ensemble)
  [3] _3d_conformal_auto/{cond}/di_field.npy       (DI field for material model)

Usage
-----
  python plot_stress_comparison.py                # all available data
  python plot_stress_comparison.py --from-di-only # synthetic from DI field only
  python plot_stress_comparison.py --conditions dh_baseline commensal_static
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import FancyBboxPatch

_HERE = Path(__file__).resolve().parent
_FIG_DIR = _HERE / "figures"
_FIG_DIR.mkdir(exist_ok=True)

# Material model (matching biofilm_3tooth_assembly.py)
E_MAX_PA = 10.0e9      # Pa
E_MIN_PA = 0.5e9       # Pa
DI_SCALE = 0.025778
DI_EXP = 2.0
NU = 0.30

COND_INFO = {
    "commensal_static": {
        "label": "Commensal\n(balanced)",
        "short": "Commensal",
        "color": "#2ca02c",
    },
    "commensal_hobic": {
        "label": "Commensal\n(HOBIC)",
        "short": "Comm-HOBIC",
        "color": "#17becf",
    },
    "dh_baseline": {
        "label": "Dysbiotic\n(DH cascade)",
        "short": "Dysbiotic",
        "color": "#d62728",
    },
    "dysbiotic_static": {
        "label": "Dysbiotic\n(static)",
        "short": "Dysb-Static",
        "color": "#ff7f0e",
    },
}


def di_to_eeff(di, e_max=E_MAX_PA, e_min=E_MIN_PA,
               di_scale=DI_SCALE, di_exp=DI_EXP):
    """DI → effective stiffness (power-law model)."""
    r = np.clip(di / di_scale, 0, 1)
    return e_max * (1 - r)**di_exp + e_min * r


def load_abaqus_results(conditions):
    """Load results from run_abaqus_auto.py outputs."""
    data = {}
    for cond in conditions:
        # Try auto jobs directory
        auto_dir = _HERE / "_abaqus_auto_jobs" / f"{cond}_T23"
        result_files = list(auto_dir.glob("result_*.json")) if auto_dir.exists() else []
        if result_files:
            with open(result_files[0]) as f:
                d = json.load(f)
            if d.get("stress"):
                data[cond] = d["stress"]
                continue

        # Try element CSV
        elem_csvs = list(auto_dir.glob("*_elements.csv")) if auto_dir.exists() else []
        if elem_csvs:
            mises = []
            max_p = []
            with open(elem_csvs[0]) as f:
                f.readline()  # header
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) >= 3:
                        try:
                            mises.append(float(parts[1]))
                            max_p.append(float(parts[2]))
                        except ValueError:
                            pass
            if mises:
                data[cond] = {
                    "mises_arr": np.array(mises),
                    "max_principal_arr": np.array(max_p),
                    "mises": {
                        "max": float(np.max(mises)),
                        "mean": float(np.mean(mises)),
                        "p95": float(np.percentile(mises, 95)),
                    },
                }
    return data


def load_posterior_ensemble(conditions):
    """Load stress data from posterior Abaqus ensemble."""
    data = {}
    for cond in conditions:
        post_dir = _HERE / "_posterior_abaqus" / cond
        if not post_dir.exists():
            continue

        subs, surfs = [], []
        for sample_dir in sorted(post_dir.glob("sample_*")):
            stress_json = sample_dir / "stress.json"
            done_flag = sample_dir / "done.flag"
            if not (stress_json.exists() and done_flag.exists()):
                continue
            with open(stress_json) as f:
                s = json.load(f)
            if "substrate_mises_mean" in s:
                subs.append(s["substrate_mises_mean"])
            if "surface_mises_mean" in s:
                surfs.append(s["surface_mises_mean"])

        if subs:
            data[cond] = {
                "substrate": np.array(subs),
                "surface": np.array(surfs) if surfs else None,
                "n_samples": len(subs),
            }
    return data


def load_di_fields(conditions):
    """Load DI fields from _3d_conformal_auto or _results_2d_nutrient."""
    data = {}
    for cond in conditions:
        # Try conformal auto first
        di_path = _HERE / "_3d_conformal_auto" / cond / "di_field.npy"
        if not di_path.exists():
            # Try _results_2d_nutrient
            for d in (_HERE / "_results_2d_nutrient").glob(f"{cond}*"):
                dp = d / "di_field.npy"
                if dp.exists():
                    di_path = dp
                    break
        if di_path.exists():
            di = np.load(di_path)
            data[cond] = di
    return data


def fig_mises_comparison(abaqus_data, ensemble_data, conditions):
    """Fig A: von Mises stress comparison across conditions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Summary bar chart
    ax = axes[0]
    conds_with_data = [c for c in conditions if c in abaqus_data or c in ensemble_data]
    if not conds_with_data:
        ax.text(0.5, 0.5, "No Abaqus data available",
                ha="center", va="center", transform=ax.transAxes, fontsize=14)
    else:
        x_pos = np.arange(len(conds_with_data))
        means, maxs, p95s = [], [], []
        colors = []
        labels = []
        for cond in conds_with_data:
            info = COND_INFO.get(cond, {"color": "gray", "short": cond})
            colors.append(info["color"])
            labels.append(info["short"])
            if cond in abaqus_data:
                s = abaqus_data[cond]["mises"]
                means.append(s["mean"])
                maxs.append(s["max"])
                p95s.append(s["p95"])
            elif cond in ensemble_data:
                e = ensemble_data[cond]
                means.append(float(np.mean(e["substrate"])))
                maxs.append(float(np.max(e["substrate"])))
                p95s.append(float(np.percentile(e["substrate"], 95)))

        w = 0.25
        ax.bar(x_pos - w, means, w, color=colors, alpha=0.6, label="Mean")
        ax.bar(x_pos, p95s, w, color=colors, alpha=0.8, label="P95")
        ax.bar(x_pos + w, maxs, w, color=colors, alpha=1.0, label="Max")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel("von Mises Stress [MPa]", fontsize=12)
        ax.set_title("(a) von Mises Stress Summary", fontsize=13)
        ax.legend(fontsize=9)

    # Panel 2: Posterior ensemble violins (if available)
    ax = axes[1]
    ens_conds = [c for c in conditions if c in ensemble_data]
    if not ens_conds:
        ax.text(0.5, 0.5, "No posterior ensemble data",
                ha="center", va="center", transform=ax.transAxes, fontsize=14)
    else:
        vdata = []
        vlabels = []
        vcolors = []
        for cond in ens_conds:
            info = COND_INFO.get(cond, {"color": "gray", "short": cond})
            vdata.append(ensemble_data[cond]["substrate"])
            vlabels.append(info["short"])
            vcolors.append(info["color"])

        parts = ax.violinplot(vdata, showmeans=True, showmedians=True)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(vcolors[i])
            pc.set_alpha(0.6)
        ax.set_xticks(range(1, len(vlabels)+1))
        ax.set_xticklabels(vlabels, fontsize=10)
        ax.set_ylabel("Substrate Mises [MPa]", fontsize=12)
        ax.set_title("(b) Posterior Ensemble", fontsize=13)
        n_str = ", ".join(f"n={ensemble_data[c]['n_samples']}" for c in ens_conds)
        ax.text(0.02, 0.98, n_str, transform=ax.transAxes, fontsize=8,
                va="top", ha="left", style="italic")

    fig.tight_layout()
    out = _FIG_DIR / "StressFig1_mises_comparison.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def fig_di_eeff_cascade(di_fields, conditions):
    """Fig C: DI → E_eff material model cascade."""
    conds = [c for c in conditions if c in di_fields]
    if not conds:
        print("  [SKIP] No DI data for cascade figure")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: DI histogram
    ax = axes[0]
    for cond in conds:
        di = di_fields[cond].flatten()
        info = COND_INFO.get(cond, {"color": "gray", "short": cond})
        ax.hist(di, bins=50, alpha=0.5, color=info["color"],
                label=info["short"], density=True)
    ax.set_xlabel("Dysbiotic Index (DI)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("(a) DI Distribution", fontsize=13)
    ax.legend(fontsize=9)

    # Panel 2: DI → E_eff mapping curve + condition distributions
    ax = axes[1]
    di_range = np.linspace(0, DI_SCALE * 1.5, 200)
    e_range = di_to_eeff(di_range) * 1e-9  # GPa
    ax.plot(di_range, e_range, "k-", lw=2, label=r"$E(DI) = E_{max}(1-r)^n + E_{min}\cdot r$")
    for cond in conds:
        di = di_fields[cond].flatten()
        e = di_to_eeff(di) * 1e-9
        info = COND_INFO.get(cond, {"color": "gray", "short": cond})
        ax.scatter(di, e, s=3, alpha=0.3, color=info["color"], label=info["short"])
    ax.set_xlabel("DI", fontsize=12)
    ax.set_ylabel("$E_{eff}$ [GPa]", fontsize=12)
    ax.set_title("(b) DI → Effective Stiffness", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    # Panel 3: E_eff histogram
    ax = axes[2]
    for cond in conds:
        di = di_fields[cond].flatten()
        e = di_to_eeff(di) * 1e-6  # MPa
        info = COND_INFO.get(cond, {"color": "gray", "short": cond})
        ax.hist(e, bins=50, alpha=0.5, color=info["color"],
                label=info["short"], density=True)
    ax.set_xlabel("$E_{eff}$ [MPa]", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("(c) Stiffness Distribution", fontsize=13)
    ax.legend(fontsize=9)

    fig.tight_layout()
    out = _FIG_DIR / "StressFig2_di_eeff_cascade.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def fig_spatial_stress_profile(abaqus_data, conditions):
    """Fig D: Spatial stress profile if element-level data available."""
    conds_with_arr = [c for c in conditions
                      if c in abaqus_data and "mises_arr" in abaqus_data[c]]
    if not conds_with_arr:
        print("  [SKIP] No element-level data for spatial profile")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Mises CDF
    ax = axes[0]
    for cond in conds_with_arr:
        mises = np.sort(abaqus_data[cond]["mises_arr"])
        cdf = np.arange(1, len(mises)+1) / len(mises)
        info = COND_INFO.get(cond, {"color": "gray", "short": cond})
        ax.plot(mises, cdf, color=info["color"], lw=2, label=info["short"])
    ax.set_xlabel("von Mises Stress [MPa]", fontsize=12)
    ax.set_ylabel("CDF", fontsize=12)
    ax.set_title("(a) Mises Stress CDF", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel 2: Principal stress comparison
    ax = axes[1]
    for cond in conds_with_arr:
        if "max_principal_arr" not in abaqus_data[cond]:
            continue
        s_max = np.sort(abaqus_data[cond]["max_principal_arr"])
        cdf = np.arange(1, len(s_max)+1) / len(s_max)
        info = COND_INFO.get(cond, {"color": "gray", "short": cond})
        ax.plot(s_max, cdf, color=info["color"], lw=2, label=info["short"])
    ax.set_xlabel("Max Principal Stress [MPa]", fontsize=12)
    ax.set_ylabel("CDF", fontsize=12)
    ax.set_title("(b) Principal Stress CDF", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = _FIG_DIR / "StressFig3_spatial_profiles.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def fig_summary_table(abaqus_data, ensemble_data, di_fields, conditions):
    """Summary table figure for paper."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")

    cols = ["Condition", "DI mean", "DI max", "E_eff mean\n[GPa]",
            "Mises mean\n[MPa]", "Mises P95\n[MPa]", "U_max\n[mm]"]
    rows = []
    colors_row = []
    for cond in conditions:
        info = COND_INFO.get(cond, {"short": cond, "color": "gray"})
        row = [info["short"]]
        if cond in di_fields:
            di = di_fields[cond]
            row.append(f"{np.mean(di):.4f}")
            row.append(f"{np.max(di):.4f}")
            e_eff_mean = np.mean(di_to_eeff(di)) * 1e-9
            row.append(f"{e_eff_mean:.2f}")
        else:
            row.extend(["--", "--", "--"])

        if cond in abaqus_data:
            s = abaqus_data[cond]["mises"]
            row.append(f"{s['mean']:.4f}")
            row.append(f"{s['p95']:.4f}")
            u = abaqus_data[cond].get("displacement", {}).get("max_mag", 0)
            row.append(f"{u:.6f}" if u else "--")
        elif cond in ensemble_data:
            e = ensemble_data[cond]
            row.append(f"{np.mean(e['substrate']):.4f}")
            row.append(f"{np.percentile(e['substrate'], 95):.4f}")
            row.append("--")
        else:
            row.extend(["--", "--", "--"])

        rows.append(row)
        colors_row.append(info["color"])

    if rows:
        table = ax.table(cellText=rows, colLabels=cols,
                         loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width(range(len(cols)))
        # Color condition cells
        for i, c in enumerate(colors_row):
            table[(i+1, 0)].set_facecolor(c)
            table[(i+1, 0)].set_text_props(color="white", weight="bold")
        for j in range(len(cols)):
            table[(0, j)].set_facecolor("#404040")
            table[(0, j)].set_text_props(color="white", weight="bold")

    ax.set_title("Biofilm Mechanics: Condition Comparison Summary",
                 fontsize=14, weight="bold", pad=20)

    out = _FIG_DIR / "StressFig4_summary_table.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def main():
    ap = argparse.ArgumentParser(
        description="Stress comparison visualization")
    ap.add_argument("--conditions", nargs="+",
                    default=["commensal_static", "dh_baseline"])
    ap.add_argument("--from-di-only", action="store_true",
                    help="Use DI field only (no Abaqus data needed)")
    args = ap.parse_args()

    conditions = args.conditions
    print(f"Conditions: {conditions}")

    # Load data
    print("\nLoading data...")
    abaqus_data = load_abaqus_results(conditions)
    ensemble_data = load_posterior_ensemble(conditions)
    di_fields = load_di_fields(conditions)

    print(f"  Abaqus results: {list(abaqus_data.keys())}")
    print(f"  Posterior ensemble: {list(ensemble_data.keys())}")
    print(f"  DI fields: {list(di_fields.keys())}")

    # Generate figures
    print("\nGenerating figures...")
    fig_mises_comparison(abaqus_data, ensemble_data, conditions)
    fig_di_eeff_cascade(di_fields, conditions)
    fig_spatial_stress_profile(abaqus_data, conditions)
    fig_summary_table(abaqus_data, ensemble_data, di_fields, conditions)

    print(f"\nAll figures saved to: {_FIG_DIR}")


if __name__ == "__main__":
    main()
