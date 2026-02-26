#!/usr/bin/env python3
"""Fig 23: GNN Prior Effect on TMCMC Posterior (paper quality).

Compares:
  - Uniform prior TMCMC (dh_baseline)
  - GNN-informed prior TMCMC (gnn_test_dh)
  - GNN predicted a_ij (vertical lines)

Shows how GNN prior concentrates posterior around plausible interaction
parameters, especially for a_ij edges (theta[1,10,11,18,19]).
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import gaussian_kde

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
PAPER_FIG_DIR = PROJECT_ROOT / 'FEM' / 'figures' / 'paper_final'
PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR = PROJECT_ROOT / 'data_5species' / '_runs'

PARAM_NAMES = [
    r'$\mu_{\mathrm{So}}$', r'$a_{\mathrm{So \to An}}$',
    r'$\mu_{\mathrm{An}}$', r'$b_{\mathrm{So}}$', r'$b_{\mathrm{An}}$',
    r'$\mu_{\mathrm{Vd}}$', r'$a_{\mathrm{An \to Vd}}$',
    r'$a_{\mathrm{Vd \to An}}$', r'$b_{\mathrm{Vd}}$',
    r'$a_{\mathrm{An \to Fn}}$',
    r'$a_{\mathrm{So \to Vd}}$', r'$a_{\mathrm{So \to Fn}}$',
    r'$\mu_{\mathrm{Fn}}$', r'$a_{\mathrm{Fn \to An}}$',
    r'$a_{\mathrm{Vd \to Fn}}$', r'$b_{\mathrm{Fn}}$',
    r'$\mu_{\mathrm{Pg}}$', r'$a_{\mathrm{Fn \to Vd}}^{\mathrm{inv}}$',
    r'$a_{\mathrm{Vd \to Pg}}$', r'$a_{\mathrm{Fn \to Pg}}$',
]

# GNN-predicted a_ij indices
ACTIVE_THETA_IDX = [1, 10, 11, 18, 19]
EDGE_NAMES = ['So→An', 'So→Vd', 'So→Fn', 'Vd→Pg', 'Fn→Pg']


def load_samples_and_map(run_dir):
    """Load posterior samples and MAP estimate."""
    samples_path = run_dir / 'samples.npy'
    map_path = run_dir / 'theta_MAP.json'
    logL_path = run_dir / 'logL.npy'
    samples, theta_map, logL = None, None, None
    if samples_path.exists():
        samples = np.load(str(samples_path))
    if logL_path.exists():
        logL = np.load(str(logL_path))
    if map_path.exists():
        with open(map_path) as f:
            data = json.load(f)
        theta_map = np.array(data['theta_full']) if isinstance(data, dict) and 'theta_full' in data else np.array(data)
    return samples, theta_map, logL


def compute_overlap(s1, s2, idx):
    """Bhattacharyya coefficient approximation."""
    d1, d2 = s1[:, idx], s2[:, idx]
    m1, std1, m2, std2 = d1.mean(), d1.std(), d2.mean(), d2.std()
    if std1 < 1e-8 or std2 < 1e-8:
        return 0.0
    bd = 0.25 * ((m1 - m2)**2 / (std1**2 + std2**2)) + 0.5 * np.log(0.5 * (std1/std2 + std2/std1))
    return float(np.exp(-bd))


def compute_width_ratio(s1, s2, idx):
    """Compute posterior width ratio (uniform/GNN) for parameter idx."""
    std1 = s1[:, idx].std()
    std2 = s2[:, idx].std()
    if std2 < 1e-8:
        return float('inf')
    return std1 / std2


def plot_kde(ax, data, color, label, alpha_fill=0.25):
    if len(data) < 5 or data.std() < 1e-10:
        ax.axvline(data.mean(), color=color, lw=2, label=label)
        return
    try:
        kde = gaussian_kde(data, bw_method='silverman')
        xg = np.linspace(data.min() - 0.15 * data.ptp(), data.max() + 0.15 * data.ptp(), 300)
        y = kde(xg)
        ax.plot(xg, y, color=color, lw=1.5, label=label)
        ax.fill_between(xg, y, alpha=alpha_fill, color=color)
    except Exception:
        ax.hist(data, bins=30, alpha=0.4, density=True, color=color, label=label)


def main():
    # Load GNN predictions
    gnn_pred_path = SCRIPT_DIR / 'data' / 'gnn_prior_predictions.json'
    gnn_pred = None
    if gnn_pred_path.exists():
        with open(gnn_pred_path) as f:
            gnn_data = json.load(f)
        if 'Dysbiotic_HOBIC' in gnn_data:
            gnn_pred = np.array(gnn_data['Dysbiotic_HOBIC']['a_ij_pred'])
            print(f"GNN predictions: {gnn_pred}")

    # Load posterior samples
    uniform_dir = RUNS_DIR / 'dh_baseline'
    gnn_dir = RUNS_DIR / 'gnn_test_dh'

    uni_s, uni_m, _ = load_samples_and_map(uniform_dir)
    gnn_s, gnn_m, _ = load_samples_and_map(gnn_dir)

    if uni_s is None:
        print(f"ERROR: No samples in {uniform_dir}")
        return
    if gnn_s is None:
        print(f"ERROR: No samples in {gnn_dir}")
        return

    n_params = min(uni_s.shape[1], gnn_s.shape[1], 20)
    print(f"Uniform samples: {uni_s.shape}, GNN samples: {gnn_s.shape}")

    # Compute metrics for GNN-active parameters
    print("\n=== GNN Prior Effect (Active Edges) ===")
    for k, tidx in enumerate(ACTIVE_THETA_IDX):
        if tidx >= n_params:
            continue
        ol = compute_overlap(uni_s, gnn_s, tidx)
        wr = compute_width_ratio(uni_s, gnn_s, tidx)
        uni_std = uni_s[:, tidx].std()
        gnn_std = gnn_s[:, tidx].std()
        gnn_mu = gnn_pred[k] if gnn_pred is not None else float('nan')
        print(f"  {EDGE_NAMES[k]:8s} (θ[{tidx:2d}]): OL={ol:.3f}, "
              f"σ_uni={uni_std:.3f}, σ_gnn={gnn_std:.3f}, "
              f"ratio={wr:.2f}×, GNN μ={gnn_mu:.3f}")

    # === Main Figure: 5×4 grid showing all 20 params ===
    c_uni = '#7B1FA2'   # purple
    c_gnn = '#00838F'   # teal
    c_pred = '#FF6F00'  # amber

    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(5, 4, figure=fig, hspace=0.50, wspace=0.35)

    for idx in range(n_params):
        row, col = idx // 4, idx % 4
        ax = fig.add_subplot(gs[row, col])

        plot_kde(ax, uni_s[:, idx], c_uni, 'Uniform prior')
        plot_kde(ax, gnn_s[:, idx], c_gnn, 'GNN prior')

        # Mark GNN prediction for active edges
        if idx in ACTIVE_THETA_IDX and gnn_pred is not None:
            k = ACTIVE_THETA_IDX.index(idx)
            ax.axvline(gnn_pred[k], color=c_pred, lw=2.5, ls='--', alpha=0.8,
                       label=f'GNN pred={gnn_pred[k]:.2f}')

        # Mark MAP estimates
        if uni_m is not None and idx < len(uni_m):
            ax.axvline(uni_m[idx], color=c_uni, lw=1.0, ls=':', alpha=0.5)
        if gnn_m is not None and idx < len(gnn_m):
            ax.axvline(gnn_m[idx], color=c_gnn, lw=1.0, ls=':', alpha=0.5)

        # Overlap annotation
        ol = compute_overlap(uni_s, gnn_s, idx)
        is_active = idx in ACTIVE_THETA_IDX
        olc = '#00838F' if is_active else '#616161'
        star = ' *' if is_active else ''
        ax.set_title(f'{PARAM_NAMES[idx]}{star}  OL={ol:.2f}', fontsize=8.5, color=olc,
                     fontweight='bold' if is_active else 'normal')
        ax.tick_params(labelsize=7)
        ax.set_yticks([])

        if row == 0 and col == 0:
            ax.legend(fontsize=6, loc='upper right')

    fig.suptitle('Fig. 23: GNN-Informed Prior Effect on TMCMC Posterior (Dysbiotic HOBIC)',
                 fontsize=13, fontweight='bold', y=0.98)
    fig.text(0.5, 0.01,
             '* = GNN-predicted interaction parameter  |  '
             'OL = Bhattacharyya overlap  |  '
             'Dashed = GNN prediction  |  Dotted = MAP estimate',
             ha='center', fontsize=8, style='italic', color='#424242')

    out_path = PAPER_FIG_DIR / 'fig23_gnn_prior_effect.png'
    fig.savefig(str(out_path), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\nSaved: {out_path}")

    # === Focused Figure: Only 5 GNN-active edges ===
    fig2, axes = plt.subplots(1, 5, figsize=(18, 3.5))

    for k, tidx in enumerate(ACTIVE_THETA_IDX):
        if tidx >= n_params:
            continue
        ax = axes[k]
        plot_kde(ax, uni_s[:, tidx], c_uni, 'Uniform prior', alpha_fill=0.20)
        plot_kde(ax, gnn_s[:, tidx], c_gnn, 'GNN prior', alpha_fill=0.20)

        if gnn_pred is not None:
            ax.axvline(gnn_pred[k], color=c_pred, lw=2.5, ls='--', alpha=0.8,
                       label=f'GNN pred')

        ol = compute_overlap(uni_s, gnn_s, tidx)
        wr = compute_width_ratio(uni_s, gnn_s, tidx)
        ax.set_title(f'{EDGE_NAMES[k]}\nOL={ol:.2f}, σ ratio={wr:.1f}×', fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_yticks([])
        if k == 0:
            ax.legend(fontsize=7)

    fig2.suptitle('GNN Prior Effect on Interaction Parameters (DH condition)',
                  fontsize=11, fontweight='bold')
    fig2.tight_layout(rect=[0, 0, 1, 0.92])

    out_path2 = PAPER_FIG_DIR / 'fig23_gnn_prior_edges.png'
    fig2.savefig(str(out_path2), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig2)
    print(f"Saved: {out_path2}")

    # === Summary statistics ===
    print("\n=== Summary ===")
    overlaps = [compute_overlap(uni_s, gnn_s, i) for i in range(n_params)]
    active_ols = [overlaps[i] for i in ACTIVE_THETA_IDX if i < n_params]
    non_active = [overlaps[i] for i in range(n_params) if i not in ACTIVE_THETA_IDX]

    print(f"  Mean overlap (all 20): {np.mean(overlaps):.3f}")
    print(f"  Mean overlap (5 active): {np.mean(active_ols):.3f}")
    print(f"  Mean overlap (15 non-active): {np.mean(non_active):.3f}")

    # Width ratios for active edges
    active_wr = [compute_width_ratio(uni_s, gnn_s, i) for i in ACTIVE_THETA_IDX if i < n_params]
    print(f"  Mean σ ratio (5 active): {np.mean(active_wr):.2f}×")
    print(f"  Max σ ratio (5 active): {np.max(active_wr):.2f}×")


if __name__ == '__main__':
    main()
