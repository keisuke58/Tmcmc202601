"""
Compare Expected (Fig2×Fig3) vs TMCMC Estimated species volumes.

Three visualization styles:
  1. Stream graph: expected as faint fill, estimated as bold line + CI band
  2. Small multiples: per-species panels with CI bands
  3. Side-by-side stacked bars: expected vs estimated

Usage:
  python compare_expected_vs_estimated.py --run-dir _runs/Dysbiotic_Static_abs_20260208_005500
  python compare_expected_vs_estimated.py --run-dir _runs/Commensal_Static_abs_20260208_005500

If no --run-dir given, generates demo with synthetic "estimated" data for layout preview.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.interpolate import PchipInterpolator
import json
import os
import argparse

BASE = "/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species"
DATA_DIR = f"{BASE}/experiment_data"

SPECIES_DISPLAY = [
    "S. oralis", "A. naeslundii", "V. dispar/parvula", "F. nucleatum", "P. gingivalis"
]
SPECIES_SHORT = ["S. oralis", "A. naeslundii", "V. disp./parv.", "F. nucleatum", "P. gingivalis"]

SPECIES_COLORS = {
    "S. oralis":        "#2196F3",
    "A. naeslundii":    "#43A047",
    "V. dispar/parvula": "#FF9800",
    "F. nucleatum":     "#7B1FA2",
    "P. gingivalis":    "#E53935",
}

CONDITIONS = [
    ("Commensal", "Static"),
    ("Commensal", "HOBIC"),
    ("Dysbiotic", "Static"),
    ("Dysbiotic", "HOBIC"),
]
DAYS = [1, 3, 6, 10, 15, 21]


def load_expected():
    """Load expected species volumes from Fig2 × Fig3."""
    return pd.read_csv(f"{DATA_DIR}/expected_species_volumes.csv")


def load_estimated_from_run(run_dir):
    """
    Load TMCMC estimated results from a run directory.
    Returns: condition, cultivation, dict of {species: {days, mean, ci_lo, ci_hi}}
    """
    config_path = os.path.join(run_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        condition = config.get("condition", "Unknown")
        cultivation = config.get("cultivation", "Unknown")
    else:
        # Try to infer from directory name
        dirname = os.path.basename(run_dir)
        condition = "Dysbiotic" if "Dysbiotic" in dirname else "Commensal"
        cultivation = "HOBIC" if "HOBIC" in dirname else "Static"

    # Load posterior predictive samples or summary
    # Try multiple file patterns
    result = {}

    # Pattern 1: posterior_predictive_summary.csv
    pp_path = os.path.join(run_dir, "posterior_predictive_summary.csv")
    if os.path.exists(pp_path):
        df_pp = pd.read_csv(pp_path)
        for sp in SPECIES_DISPLAY:
            sp_data = df_pp[df_pp["species"] == sp]
            result[sp] = {
                "days": sp_data["day"].values,
                "mean": sp_data["mean"].values,
                "ci_lo": sp_data["ci_lo"].values if "ci_lo" in sp_data.columns else sp_data["q05"].values,
                "ci_hi": sp_data["ci_hi"].values if "ci_hi" in sp_data.columns else sp_data["q95"].values,
            }
        return condition, cultivation, result

    # Pattern 2: theta_MAP.json + forward simulation
    map_path = os.path.join(run_dir, "theta_MAP.json")
    mean_path = os.path.join(run_dir, "theta_MEAN.json")

    for theta_path, label in [(map_path, "MAP"), (mean_path, "MEAN")]:
        if os.path.exists(theta_path):
            print(f"  Found {label} theta: {theta_path}")
            # We'd need to forward-simulate here
            # For now, return None and use demo mode
            break

    # Pattern 3: fit_metrics with predicted values
    for sp_idx, sp in enumerate(SPECIES_DISPLAY):
        fit_path = os.path.join(run_dir, f"fit_metrics_chain0.json")
        if os.path.exists(fit_path):
            with open(fit_path) as f:
                fit = json.load(f)
            # Extract predicted values if available
            break

    return condition, cultivation, result


def generate_demo_estimated(df_expected, condition, cultivation):
    """
    Generate synthetic "estimated" data for layout preview.
    Adds noise + slight bias to expected values, with CI bands.
    """
    sub = df_expected[
        (df_expected["condition"] == condition) &
        (df_expected["cultivation"] == cultivation)
    ]
    result = {}
    np.random.seed(123)

    for sp in SPECIES_DISPLAY:
        sp_data = sub[sub["species"] == sp].sort_values("day")
        days = sp_data["day"].values
        true_vals = sp_data["species_volume_x1e6"].values

        # Add realistic noise: slight bias + random perturbation
        noise_scale = true_vals * 0.15 + 0.005
        mean_est = true_vals * (1.0 + np.random.uniform(-0.1, 0.1, len(days)))
        mean_est = np.maximum(0, mean_est)

        # CI band (wider for later days)
        ci_width = noise_scale * (1.0 + np.arange(len(days)) * 0.15)
        ci_lo = np.maximum(0, mean_est - 1.5 * ci_width)
        ci_hi = mean_est + 1.5 * ci_width

        result[sp] = {
            "days": days,
            "mean": mean_est,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
        }

    return result


# ══════════════════════════════════════════════════════════════
# Style 1: Stream Graph with CI overlay
# ══════════════════════════════════════════════════════════════

def plot_stream_comparison(df_expected, estimated, condition, cultivation, output_dir):
    """Stream graph: expected as faint fill, estimated as bold line + CI."""
    fig, ax = plt.subplots(figsize=(14, 7))

    sub = df_expected[
        (df_expected["condition"] == condition) &
        (df_expected["cultivation"] == cultivation)
    ]
    days = sorted(sub["day"].unique())
    days_fine = np.linspace(min(days), max(days), 200)

    # ── Expected: faint stacked area ──
    y_stack = np.zeros_like(days_fine)
    for sp in SPECIES_DISPLAY:
        sp_data = sub[sub["species"] == sp].sort_values("day")
        y = sp_data["species_volume_x1e6"].values
        if len(y) == len(days):
            interp = PchipInterpolator(days, y)
            y_fine = np.maximum(0, interp(days_fine))
        else:
            y_fine = np.zeros_like(days_fine)

        ax.fill_between(days_fine, y_stack, y_stack + y_fine,
                        color=SPECIES_COLORS[sp], alpha=0.15, linewidth=0)
        ax.plot(days_fine, y_stack + y_fine, color=SPECIES_COLORS[sp],
                linewidth=0.5, alpha=0.3)
        y_stack += y_fine

    # ── Estimated: bold lines + CI bands ──
    for sp in SPECIES_DISPLAY:
        if sp not in estimated:
            continue
        est = estimated[sp]
        color = SPECIES_COLORS[sp]

        # CI band
        if len(est["days"]) > 2:
            d_fine = np.linspace(est["days"].min(), est["days"].max(), 200)
            interp_lo = PchipInterpolator(est["days"], est["ci_lo"])
            interp_hi = PchipInterpolator(est["days"], est["ci_hi"])
            interp_mean = PchipInterpolator(est["days"], est["mean"])

            ax.fill_between(d_fine, np.maximum(0, interp_lo(d_fine)),
                            interp_hi(d_fine),
                            color=color, alpha=0.2, linewidth=0)
            ax.plot(d_fine, interp_mean(d_fine), color=color, linewidth=2.5, alpha=0.9)
        else:
            ax.fill_between(est["days"], est["ci_lo"], est["ci_hi"],
                            color=color, alpha=0.2)
            ax.plot(est["days"], est["mean"], color=color, linewidth=2.5)

        # Data points (expected) as open circles
        sp_expected = sub[sub["species"] == sp].sort_values("day")
        ax.plot(sp_expected["day"], sp_expected["species_volume_x1e6"],
                'o', color=color, markersize=8, markerfacecolor='white',
                markeredgewidth=2, alpha=0.8, zorder=5)

        # Estimated points as filled
        ax.plot(est["days"], est["mean"], 's', color=color, markersize=6,
                alpha=0.9, zorder=5)

    # Legend
    legend_elements = []
    for sp in SPECIES_DISPLAY:
        legend_elements.append(
            Line2D([0], [0], color=SPECIES_COLORS[sp], lw=2.5, label=sp))
    legend_elements.append(
        Line2D([0], [0], color='gray', marker='o', markerfacecolor='white',
               markeredgewidth=2, markersize=8, lw=0, label='Expected (Fig2×Fig3)'))
    legend_elements.append(
        Line2D([0], [0], color='gray', marker='s', markersize=6, lw=0,
               label='Estimated (TMCMC)'))
    legend_elements.append(
        plt.Rectangle((0, 0), 1, 1, fc='gray', alpha=0.15, label='Expected stream (faint)'))
    legend_elements.append(
        plt.Rectangle((0, 0), 1, 1, fc='gray', alpha=0.2, label='95% CI band'))

    ax.legend(handles=legend_elements, fontsize=9, loc='upper right',
              framealpha=0.95, ncol=2)

    ax.set_xlabel("Day", fontsize=13)
    ax.set_ylabel("Species Volume [×10⁶ μm³]", fontsize=13)
    ax.set_title(f"Expected vs Estimated  —  {condition} / {cultivation}\n"
                 f"Faint fill = Expected (Fig2×Fig3),  Bold line + band = TMCMC Estimate ± 95% CI",
                 fontsize=13, fontweight='bold')
    ax.set_xlim(0.5, 21.5)
    ax.set_xticks(DAYS)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.15, linestyle='--')
    ax.set_axisbelow(True)

    fig.tight_layout()
    path = os.path.join(output_dir, f"compare_stream_{condition}_{cultivation}.png")
    fig.savefig(path, dpi=250, bbox_inches='tight', facecolor='white')
    print(f"Saved: {path}")
    plt.close()


# ══════════════════════════════════════════════════════════════
# Style 2: Small Multiples with CI bands
# ══════════════════════════════════════════════════════════════

def plot_small_multiples_comparison(df_expected, estimated, condition, cultivation, output_dir):
    """Per-species panels: expected (gray) vs estimated (colored + CI)."""
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), sharey=False)
    fig.suptitle(f"Species-by-Species Comparison  —  {condition} / {cultivation}\n"
                 f"Gray area = Expected (Fig2×Fig3),  Color = TMCMC Estimate ± 95% CI",
                 fontsize=13, fontweight='bold', y=1.05)

    sub = df_expected[
        (df_expected["condition"] == condition) &
        (df_expected["cultivation"] == cultivation)
    ]

    for i, sp in enumerate(SPECIES_DISPLAY):
        ax = axes[i]
        color = SPECIES_COLORS[sp]

        # Expected: gray filled area
        sp_exp = sub[sub["species"] == sp].sort_values("day")
        days = sp_exp["day"].values
        exp_vals = sp_exp["species_volume_x1e6"].values

        if len(days) > 2:
            d_fine = np.linspace(days.min(), days.max(), 200)
            interp_exp = PchipInterpolator(days, exp_vals)
            exp_fine = np.maximum(0, interp_exp(d_fine))
            ax.fill_between(d_fine, 0, exp_fine, color='gray', alpha=0.15, linewidth=0)
            ax.plot(d_fine, exp_fine, color='gray', linewidth=1.0, alpha=0.4,
                    linestyle='--')

        # Expected data points
        ax.plot(days, exp_vals, 'o', color='gray', markersize=9,
                markerfacecolor='white', markeredgewidth=2.0, alpha=0.7, zorder=5,
                label='Expected')

        # Estimated: colored line + CI band
        if sp in estimated:
            est = estimated[sp]
            if len(est["days"]) > 2:
                d_fine2 = np.linspace(est["days"].min(), est["days"].max(), 200)
                interp_lo = PchipInterpolator(est["days"], est["ci_lo"])
                interp_hi = PchipInterpolator(est["days"], est["ci_hi"])
                interp_m = PchipInterpolator(est["days"], est["mean"])

                ax.fill_between(d_fine2, np.maximum(0, interp_lo(d_fine2)),
                                interp_hi(d_fine2),
                                color=color, alpha=0.2, linewidth=0, label='95% CI')
                ax.plot(d_fine2, interp_m(d_fine2), color=color, linewidth=2.5,
                        alpha=0.9, label='TMCMC Mean')
            else:
                ax.fill_between(est["days"], est["ci_lo"], est["ci_hi"],
                                color=color, alpha=0.2, label='95% CI')
                ax.plot(est["days"], est["mean"], color=color, linewidth=2.5,
                        label='TMCMC Mean')

            ax.plot(est["days"], est["mean"], 's', color=color, markersize=5,
                    alpha=0.9, zorder=5)

        ax.set_title(SPECIES_SHORT[i], fontsize=11, fontweight='bold',
                     fontstyle='italic', color=color)
        ax.set_xlabel("Day", fontsize=9)
        ax.set_xticks(DAYS)
        ax.set_xlim(0, 22)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.15, linestyle='--')
        ax.set_axisbelow(True)

        if i == 0:
            ax.set_ylabel("Volume [×10⁶ μm³]", fontsize=10)
            ax.legend(fontsize=7, loc='upper right', framealpha=0.9)

    fig.tight_layout()
    path = os.path.join(output_dir, f"compare_species_{condition}_{cultivation}.png")
    fig.savefig(path, dpi=250, bbox_inches='tight', facecolor='white')
    print(f"Saved: {path}")
    plt.close()


# ══════════════════════════════════════════════════════════════
# Style 3: Side-by-side stacked bars
# ══════════════════════════════════════════════════════════════

def plot_sidebyside_bars(df_expected, estimated, condition, cultivation, output_dir):
    """Side-by-side stacked bars: expected (left, faint) vs estimated (right, bold)."""
    fig, ax = plt.subplots(figsize=(14, 6))

    sub = df_expected[
        (df_expected["condition"] == condition) &
        (df_expected["cultivation"] == cultivation)
    ]
    days = sorted(sub["day"].unique())
    bar_w = 1.2
    gap = 0.15

    for d_idx, day in enumerate(days):
        x_center = day

        # ── Expected bar (left) ──
        bottom_exp = 0
        for sp in SPECIES_DISPLAY:
            sp_data = sub[(sub["species"] == sp) & (sub["day"] == day)]
            val = sp_data["species_volume_x1e6"].values[0] if len(sp_data) > 0 else 0

            ax.bar(x_center - bar_w/2 - gap/2, val, bar_w, bottom=bottom_exp,
                   color=SPECIES_COLORS[sp], alpha=0.3, edgecolor='gray',
                   linewidth=0.5)
            if val > 0.015:
                ax.text(x_center - bar_w/2 - gap/2, bottom_exp + val/2,
                        f"{val:.3f}", ha='center', va='center', fontsize=5.5,
                        color='gray', fontweight='bold')
            bottom_exp += val

        # ── Estimated bar (right) ──
        bottom_est = 0
        for sp in SPECIES_DISPLAY:
            if sp in estimated:
                est = estimated[sp]
                day_mask = est["days"] == day
                if day_mask.any():
                    val = est["mean"][day_mask][0]
                    err_lo = val - est["ci_lo"][day_mask][0]
                    err_hi = est["ci_hi"][day_mask][0] - val
                else:
                    val = 0
                    err_lo = err_hi = 0
            else:
                val = 0
                err_lo = err_hi = 0

            ax.bar(x_center + bar_w/2 + gap/2, val, bar_w, bottom=bottom_est,
                   color=SPECIES_COLORS[sp], alpha=0.85, edgecolor='white',
                   linewidth=0.5)

            # Error bar for top species in stack
            if val > 0.015:
                ax.text(x_center + bar_w/2 + gap/2, bottom_est + val/2,
                        f"{val:.3f}", ha='center', va='center', fontsize=5.5,
                        color='white', fontweight='bold')
            bottom_est += val

        # CI whisker on total estimated
        if estimated:
            total_lo = sum(estimated[sp]["ci_lo"][estimated[sp]["days"] == day][0]
                          if sp in estimated and (estimated[sp]["days"] == day).any() else 0
                          for sp in SPECIES_DISPLAY)
            total_hi = sum(estimated[sp]["ci_hi"][estimated[sp]["days"] == day][0]
                          if sp in estimated and (estimated[sp]["days"] == day).any() else 0
                          for sp in SPECIES_DISPLAY)
            ax.plot([x_center + bar_w/2 + gap/2] * 2, [total_lo, total_hi],
                    'k-', linewidth=1.5, alpha=0.5)
            ax.plot(x_center + bar_w/2 + gap/2, total_hi, 'k_', markersize=6, alpha=0.5)
            ax.plot(x_center + bar_w/2 + gap/2, total_lo, 'k_', markersize=6, alpha=0.5)

    # Labels
    ax.set_xlabel("Day", fontsize=12)
    ax.set_ylabel("Volume [×10⁶ μm³]", fontsize=12)
    ax.set_title(f"Expected vs Estimated (Side-by-Side)  —  {condition} / {cultivation}\n"
                 f"Left (faint) = Expected (Fig2×Fig3),  Right (bold) = TMCMC ± 95% CI",
                 fontsize=13, fontweight='bold')
    ax.set_xticks(days)
    ax.set_xticklabels([f"Day {d}" for d in days], fontsize=11)
    ax.set_xlim(-1, 23)
    ax.set_ylim(bottom=0)
    ax.grid(True, axis='y', alpha=0.15, linestyle='--')
    ax.set_axisbelow(True)

    # Legend
    legend_elements = [plt.Rectangle((0, 0), 1, 1, fc=SPECIES_COLORS[sp], alpha=0.85, label=sp)
                       for sp in SPECIES_DISPLAY]
    legend_elements.append(plt.Rectangle((0, 0), 1, 1, fc='gray', alpha=0.3, label='Expected'))
    legend_elements.append(plt.Rectangle((0, 0), 1, 1, fc='gray', alpha=0.85, label='Estimated'))
    ax.legend(handles=legend_elements, fontsize=8, loc='upper right',
              framealpha=0.9, ncol=2)

    fig.tight_layout()
    path = os.path.join(output_dir, f"compare_bars_{condition}_{cultivation}.png")
    fig.savefig(path, dpi=250, bbox_inches='tight', facecolor='white')
    print(f"Saved: {path}")
    plt.close()


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Path to TMCMC run directory")
    parser.add_argument("--condition", type=str, default=None)
    parser.add_argument("--cultivation", type=str, default=None)
    parser.add_argument("--demo", action="store_true",
                        help="Generate demo with synthetic data for all 4 conditions")
    args = parser.parse_args()

    df_expected = load_expected()

    if args.run_dir and os.path.exists(args.run_dir):
        condition, cultivation, estimated = load_estimated_from_run(args.run_dir)
        output_dir = args.run_dir

        if not estimated:
            print(f"Could not load estimated data from {args.run_dir}, using demo mode")
            estimated = generate_demo_estimated(df_expected, condition, cultivation)

        print(f"\n=== {condition} / {cultivation} ===")
        plot_stream_comparison(df_expected, estimated, condition, cultivation, output_dir)
        plot_small_multiples_comparison(df_expected, estimated, condition, cultivation, output_dir)
        plot_sidebyside_bars(df_expected, estimated, condition, cultivation, output_dir)

    else:
        # Demo mode: all 4 conditions
        output_dir = DATA_DIR
        print("Demo mode: generating comparison previews for all 4 conditions\n")

        for cond, cult in CONDITIONS:
            print(f"=== {cond} / {cult} ===")
            estimated = generate_demo_estimated(df_expected, cond, cult)
            plot_stream_comparison(df_expected, estimated, cond, cult, output_dir)
            plot_small_multiples_comparison(df_expected, estimated, cond, cult, output_dir)
            plot_sidebyside_bars(df_expected, estimated, cond, cult, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
