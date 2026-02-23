#!/usr/bin/env python3
"""
Compare Nishioka (reduced) vs Standard parameter estimation results.

This script compares:
1. Parameter estimates (MAP, Mean)
2. Fit quality (RMSE, MAE per species)
3. Posterior uncertainty
4. Computational efficiency

Usage:
    python compare_nishioka_standard.py <nishioka_run_dir> <standard_run_dir> [--output-dir OUTPUT]

Example:
    python compare_nishioka_standard.py \
        _runs/nishioka_v1_20260205_120000 \
        _runs/improved_v1_20260205_002904 \
        --output-dir comparison_nishioka_vs_standard
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Nishioka locked indices
LOCKED_INDICES = [6, 12, 13, 16, 17]

# Parameter names for the 20-parameter model
PARAM_NAMES = [
    'a11', 'a12', 'a13', 'b1', 'b2',      # 0-4
    'a22', 'a23', 'a24', 'b3', 'b4',      # 5-9
    'a31', 'a32', 'a33', 'a34', 'a35',    # 10-14
    'b5', 'a41', 'a42', 'a43', 'a44'      # 15-19
]

SPECIES_NAMES = ['S. oralis', 'A. naeslundii', 'Veillonella', 'F. nucleatum', 'P. gingivalis']


def load_run_data(run_dir):
    """Load all relevant data from a run directory."""
    run_dir = Path(run_dir)
    data = {'run_dir': str(run_dir)}

    # Load config
    config_path = run_dir / 'config.json'
    if config_path.exists():
        with open(config_path) as f:
            data['config'] = json.load(f)

    # Load theta MAP
    map_path = run_dir / 'theta_MAP.json'
    if map_path.exists():
        with open(map_path) as f:
            data['theta_MAP'] = json.load(f)

    # Load theta Mean
    mean_path = run_dir / 'theta_MEAN.json'
    if mean_path.exists():
        with open(mean_path) as f:
            data['theta_MEAN'] = json.load(f)

    # Load fit metrics
    metrics_path = run_dir / 'fit_metrics.json'
    if metrics_path.exists():
        with open(metrics_path) as f:
            data['fit_metrics'] = json.load(f)

    # Load posterior samples
    samples_path = run_dir / 'posterior_samples.csv'
    if samples_path.exists():
        data['samples'] = pd.read_csv(samples_path)

    # Load timing/metrics
    run_metrics_path = run_dir / 'metrics.json'
    if run_metrics_path.exists():
        with open(run_metrics_path) as f:
            data['run_metrics'] = json.load(f)

    return data


def compare_parameters(nishioka_data, standard_data, output_dir):
    """Compare parameter estimates between runs."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Get theta values
    nishioka_map = np.array(nishioka_data.get('theta_MAP', {}).get('theta', [0]*20))
    standard_map = np.array(standard_data.get('theta_MAP', {}).get('theta', [0]*20))
    nishioka_mean = np.array(nishioka_data.get('theta_MEAN', {}).get('theta', [0]*20))
    standard_mean = np.array(standard_data.get('theta_MEAN', {}).get('theta', [0]*20))

    # Ensure arrays are correct length
    if len(nishioka_map) < 20:
        nishioka_map = np.pad(nishioka_map, (0, 20 - len(nishioka_map)))
    if len(standard_map) < 20:
        standard_map = np.pad(standard_map, (0, 20 - len(standard_map)))
    if len(nishioka_mean) < 20:
        nishioka_mean = np.pad(nishioka_mean, (0, 20 - len(nishioka_mean)))
    if len(standard_mean) < 20:
        standard_mean = np.pad(standard_mean, (0, 20 - len(standard_mean)))

    x = np.arange(20)
    width = 0.35

    # Plot 1: MAP comparison
    ax1 = axes[0, 0]
    colors_nishioka = ['gray' if i in LOCKED_INDICES else 'steelblue' for i in range(20)]
    colors_standard = ['orange' for _ in range(20)]

    bars1 = ax1.bar(x - width/2, nishioka_map, width, label='Nishioka (MAP)', color=colors_nishioka, edgecolor='black', alpha=0.8)
    bars2 = ax1.bar(x + width/2, standard_map, width, label='Standard (MAP)', color='orange', alpha=0.7)

    ax1.set_xlabel('Parameter Index')
    ax1.set_ylabel('Value')
    ax1.set_title('MAP Estimates Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(PARAM_NAMES, rotation=45, ha='right', fontsize=8)
    ax1.legend()
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    # Highlight locked indices
    for idx in LOCKED_INDICES:
        ax1.axvline(x=idx, color='red', linestyle='--', alpha=0.3, linewidth=1)

    # Plot 2: Mean comparison
    ax2 = axes[0, 1]
    ax2.bar(x - width/2, nishioka_mean, width, label='Nishioka (Mean)', color=colors_nishioka, edgecolor='black', alpha=0.8)
    ax2.bar(x + width/2, standard_mean, width, label='Standard (Mean)', color='orange', alpha=0.7)

    ax2.set_xlabel('Parameter Index')
    ax2.set_ylabel('Value')
    ax2.set_title('Posterior Mean Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(PARAM_NAMES, rotation=45, ha='right', fontsize=8)
    ax2.legend()
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    for idx in LOCKED_INDICES:
        ax2.axvline(x=idx, color='red', linestyle='--', alpha=0.3, linewidth=1)

    # Plot 3: Difference (Standard - Nishioka)
    ax3 = axes[1, 0]
    diff_map = standard_map - nishioka_map
    diff_mean = standard_mean - nishioka_mean

    ax3.bar(x - width/2, diff_map, width, label='MAP Difference', color='steelblue', alpha=0.7)
    ax3.bar(x + width/2, diff_mean, width, label='Mean Difference', color='coral', alpha=0.7)

    ax3.set_xlabel('Parameter Index')
    ax3.set_ylabel('Standard - Nishioka')
    ax3.set_title('Parameter Difference (Standard - Nishioka)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(PARAM_NAMES, rotation=45, ha='right', fontsize=8)
    ax3.legend()
    ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    for idx in LOCKED_INDICES:
        ax3.axvline(x=idx, color='red', linestyle='--', alpha=0.3, linewidth=1)

    # Plot 4: Locked vs Active parameter magnitudes
    ax4 = axes[1, 1]

    active_indices = [i for i in range(20) if i not in LOCKED_INDICES]

    # Standard model values at locked indices
    std_locked = [standard_map[i] for i in LOCKED_INDICES]
    std_active = [standard_map[i] for i in active_indices]

    ax4.boxplot([std_locked, std_active], labels=['Locked (Standard)', 'Active (Standard)'])
    ax4.scatter([1]*len(std_locked), std_locked, alpha=0.5, color='red', label='Locked params in Standard')
    ax4.scatter([2]*len(std_active), std_active, alpha=0.5, color='blue', label='Active params')

    ax4.set_ylabel('Parameter Value')
    ax4.set_title('Standard Model: Locked vs Active Parameters')
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / 'parameter_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'nishioka_map': nishioka_map.tolist(),
        'standard_map': standard_map.tolist(),
        'diff_map': diff_map.tolist(),
        'std_locked_values': std_locked
    }


def compare_fit_metrics(nishioka_data, standard_data, output_dir):
    """Compare fit quality between runs."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    metrics_n = nishioka_data.get('fit_metrics', {})
    metrics_s = standard_data.get('fit_metrics', {})

    # Extract per-species RMSE
    rmse_n = metrics_n.get('rmse_per_species', [0]*5)
    rmse_s = metrics_s.get('rmse_per_species', [0]*5)
    mae_n = metrics_n.get('mae_per_species', [0]*5)
    mae_s = metrics_s.get('mae_per_species', [0]*5)

    x = np.arange(5)
    width = 0.35

    # RMSE comparison
    ax1 = axes[0]
    ax1.bar(x - width/2, rmse_n, width, label='Nishioka', color='steelblue', alpha=0.8)
    ax1.bar(x + width/2, rmse_s, width, label='Standard', color='orange', alpha=0.8)
    ax1.set_xlabel('Species')
    ax1.set_ylabel('RMSE')
    ax1.set_title('RMSE per Species')
    ax1.set_xticks(x)
    ax1.set_xticklabels(SPECIES_NAMES, rotation=30, ha='right')
    ax1.legend()

    # MAE comparison
    ax2 = axes[1]
    ax2.bar(x - width/2, mae_n, width, label='Nishioka', color='steelblue', alpha=0.8)
    ax2.bar(x + width/2, mae_s, width, label='Standard', color='orange', alpha=0.8)
    ax2.set_xlabel('Species')
    ax2.set_ylabel('MAE')
    ax2.set_title('MAE per Species')
    ax2.set_xticks(x)
    ax2.set_xticklabels(SPECIES_NAMES, rotation=30, ha='right')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'fit_metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Compute summary
    total_rmse_n = metrics_n.get('total_rmse', np.sqrt(np.mean(np.array(rmse_n)**2)))
    total_rmse_s = metrics_s.get('total_rmse', np.sqrt(np.mean(np.array(rmse_s)**2)))

    return {
        'nishioka_rmse': rmse_n,
        'standard_rmse': rmse_s,
        'nishioka_total_rmse': total_rmse_n,
        'standard_total_rmse': total_rmse_s,
        'improvement': (total_rmse_s - total_rmse_n) / total_rmse_s * 100 if total_rmse_s > 0 else 0
    }


def compare_posterior_uncertainty(nishioka_data, standard_data, output_dir):
    """Compare posterior uncertainty (std) for each parameter."""
    samples_n = nishioka_data.get('samples')
    samples_s = standard_data.get('samples')

    if samples_n is None or samples_s is None:
        print("Warning: Posterior samples not available for one or both runs")
        return {}

    # Compute std for each parameter
    std_n = samples_n.std().values[:20] if len(samples_n.columns) >= 20 else np.zeros(20)
    std_s = samples_s.std().values[:20] if len(samples_s.columns) >= 20 else np.zeros(20)

    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(20)
    width = 0.35

    ax.bar(x - width/2, std_n, width, label='Nishioka', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, std_s, width, label='Standard', color='orange', alpha=0.8)

    ax.set_xlabel('Parameter Index')
    ax.set_ylabel('Posterior Std')
    ax.set_title('Posterior Uncertainty Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(PARAM_NAMES, rotation=45, ha='right', fontsize=8)
    ax.legend()

    # Highlight locked indices
    for idx in LOCKED_INDICES:
        ax.axvline(x=idx, color='red', linestyle='--', alpha=0.3, linewidth=1)

    plt.tight_layout()
    plt.savefig(output_dir / 'posterior_uncertainty_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'nishioka_std': std_n.tolist(),
        'standard_std': std_s.tolist(),
        'mean_std_nishioka': np.mean(std_n),
        'mean_std_standard': np.mean(std_s)
    }


def generate_summary_table(nishioka_data, standard_data, param_results, fit_results, output_dir):
    """Generate a summary comparison table."""

    # Get configs
    config_n = nishioka_data.get('config', {})
    config_s = standard_data.get('config', {})

    # Build summary
    summary = {
        'Metric': [
            'Run Type',
            'Free Parameters',
            'Locked Parameters',
            'N Particles',
            'N Stages',
            'Total RMSE (MAP)',
            'Total MAE (MAP)',
            'RMSE Improvement (%)',
            'Species 1 RMSE',
            'Species 2 RMSE',
            'Species 3 RMSE',
        ],
        'Nishioka': [
            'Reduced (Nishioka)',
            15,
            5,
            config_n.get('n_particles', 'N/A'),
            config_n.get('n_stages', 'N/A'),
            f"{fit_results.get('nishioka_total_rmse', 'N/A'):.4f}" if isinstance(fit_results.get('nishioka_total_rmse'), (int, float)) else 'N/A',
            'N/A',
            '-',
            f"{fit_results.get('nishioka_rmse', [0])[0]:.4f}" if fit_results.get('nishioka_rmse') else 'N/A',
            f"{fit_results.get('nishioka_rmse', [0,0])[1]:.4f}" if len(fit_results.get('nishioka_rmse', [])) > 1 else 'N/A',
            f"{fit_results.get('nishioka_rmse', [0,0,0])[2]:.4f}" if len(fit_results.get('nishioka_rmse', [])) > 2 else 'N/A',
        ],
        'Standard': [
            'Full (20 params)',
            20,
            0,
            config_s.get('n_particles', 'N/A'),
            config_s.get('n_stages', 'N/A'),
            f"{fit_results.get('standard_total_rmse', 'N/A'):.4f}" if isinstance(fit_results.get('standard_total_rmse'), (int, float)) else 'N/A',
            'N/A',
            f"{fit_results.get('improvement', 0):.1f}%",
            f"{fit_results.get('standard_rmse', [0])[0]:.4f}" if fit_results.get('standard_rmse') else 'N/A',
            f"{fit_results.get('standard_rmse', [0,0])[1]:.4f}" if len(fit_results.get('standard_rmse', [])) > 1 else 'N/A',
            f"{fit_results.get('standard_rmse', [0,0,0])[2]:.4f}" if len(fit_results.get('standard_rmse', [])) > 2 else 'N/A',
        ]
    }

    df = pd.DataFrame(summary)

    # Save as CSV
    df.to_csv(output_dir / 'comparison_summary.csv', index=False)

    # Create visual table
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Color header
    for j in range(len(df.columns)):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', weight='bold')

    plt.title('Nishioka vs Standard: Comparison Summary', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(output_dir / 'comparison_summary_table.png', dpi=150, bbox_inches='tight')
    plt.close()

    return df


def main():
    parser = argparse.ArgumentParser(description='Compare Nishioka vs Standard estimation results')
    parser.add_argument('nishioka_dir', help='Path to Nishioka (reduced) run directory')
    parser.add_argument('standard_dir', help='Path to Standard run directory')
    parser.add_argument('--output-dir', default='comparison_nishioka_vs_standard',
                        help='Output directory for comparison results')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Nishioka results from: {args.nishioka_dir}")
    nishioka_data = load_run_data(args.nishioka_dir)

    print(f"Loading Standard results from: {args.standard_dir}")
    standard_data = load_run_data(args.standard_dir)

    print("\n=== Comparing Parameter Estimates ===")
    param_results = compare_parameters(nishioka_data, standard_data, output_dir)

    # Check if locked params in standard are near zero
    std_locked = param_results.get('std_locked_values', [])
    if std_locked:
        print(f"Standard model values at locked indices: {std_locked}")
        print(f"Mean absolute value at locked indices: {np.mean(np.abs(std_locked)):.4f}")

    print("\n=== Comparing Fit Metrics ===")
    fit_results = compare_fit_metrics(nishioka_data, standard_data, output_dir)
    print(f"Nishioka Total RMSE: {fit_results.get('nishioka_total_rmse', 'N/A')}")
    print(f"Standard Total RMSE: {fit_results.get('standard_total_rmse', 'N/A')}")
    print(f"Improvement: {fit_results.get('improvement', 0):.1f}%")

    print("\n=== Comparing Posterior Uncertainty ===")
    uncertainty_results = compare_posterior_uncertainty(nishioka_data, standard_data, output_dir)

    print("\n=== Generating Summary ===")
    summary_df = generate_summary_table(nishioka_data, standard_data, param_results, fit_results, output_dir)
    print(summary_df.to_string(index=False))

    # Save all results to JSON
    all_results = {
        'nishioka_dir': args.nishioka_dir,
        'standard_dir': args.standard_dir,
        'parameter_comparison': param_results,
        'fit_comparison': fit_results,
        'uncertainty_comparison': uncertainty_results
    }

    with open(output_dir / 'comparison_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n=== Results saved to: {output_dir} ===")
    print("Files generated:")
    print("  - parameter_comparison.png")
    print("  - fit_metrics_comparison.png")
    print("  - posterior_uncertainty_comparison.png")
    print("  - comparison_summary.csv")
    print("  - comparison_summary_table.png")
    print("  - comparison_results.json")


if __name__ == '__main__':
    main()
