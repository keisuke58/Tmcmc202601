#!/usr/bin/env python3
"""
Enhanced Bayesian Biofilm Analysis - Complete Data Saving Version
==================================================================

This version saves ALL data including:
1. Input synthetic data (d_M1, d_M2, d_M3)
2. True parameters used for data generation
3. Raw posterior samples (for later analysis)
4. Convergence diagnostics
5. Optimization history
6. Summary statistics

Author: Enhanced implementation based on Fritsch et al. paper
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, norm
import pandas as pd
import os
from dataclasses import dataclass
from typing import List, Tuple
import warnings
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# Set seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'figure.dpi': 120,
    'lines.linewidth': 2,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.facecolor': 'white'
})

output_folder = "bayesian_results"
os.makedirs(output_folder, exist_ok=True)

# =============================================================================
# DATA STRUCTURES
# =============================================================================
@dataclass
class ConvergenceDiagnostics:
    r_hat: np.ndarray
    ess: np.ndarray
    acceptance_rate: float
    is_converged: bool

@dataclass
class ModelValidation:
    aic: float
    bic: float
    rmse: float
    mae: float
    r_squared: float

@dataclass
class BayesianResults:
    map_estimate: np.ndarray
    posterior_samples: np.ndarray
    posterior_mean: np.ndarray
    posterior_std: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    convergence: ConvergenceDiagnostics
    validation: ModelValidation
    param_names: List[str]
    loss_history: List[float]

# =============================================================================
# DATA GENERATION (Simulated - for demonstration)
# =============================================================================
def generate_synthetic_data_and_results():
    """
    Generate synthetic data and posterior samples.
    In real computation, this would be replaced by actual optimization.
    
    Returns all data needed for saving.
    """
    
    # True parameters from paper (Case II)
    TRUE_M1 = np.array([0.8, 2.0, 1.0, 0.1, 0.2])
    TRUE_M2 = np.array([1.5, 1.0, 2.0, 0.3, 0.4])
    TRUE_M3 = np.array([2.0, 1.0, 2.0, 1.0])
    
    # Parameter names
    param_names_M1 = ['a11', 'a12', 'a22', 'b1', 'b2']
    param_names_M2 = ['a33', 'a34', 'a44', 'b3', 'b4']
    param_names_M3 = ['a13', 'a14', 'a23', 'a24']
    
    # =========================================================================
    # Generate Input Data (Synthetic experimental data)
    # =========================================================================
    # Time points
    n_timepoints = 10
    t = np.linspace(0, 1, n_timepoints)
    
    # M1 data: Species 1 & 2 growth curves
    d_M1 = np.column_stack([
        0.25 + 0.65 * (1 - np.exp(-4*t)) + np.random.normal(0, 0.01, n_timepoints),
        0.30 + 0.55 * (1 - np.exp(-3*t)) + np.random.normal(0, 0.01, n_timepoints)
    ])
    
    # M2 data: Species 3 & 4 with antibiotic effect
    d_M2 = np.column_stack([
        0.20 + 0.60 * (1 - np.exp(-3*t)) * np.exp(-0.5*t) + np.random.normal(0, 0.01, n_timepoints),
        0.20 + 0.50 * (1 - np.exp(-2*t)) * np.exp(-0.8*t) + np.random.normal(0, 0.01, n_timepoints)
    ])
    
    # M3 data: All 4 species interaction
    d_M3 = np.column_stack([
        0.02 + 0.18 * (1 - np.exp(-5*t)) + np.random.normal(0, 0.005, n_timepoints),
        0.02 + 0.20 * (1 - np.exp(-4*t)) + np.random.normal(0, 0.005, n_timepoints),
        0.02 + 0.22 * (1 - np.exp(-3*t)) + np.random.normal(0, 0.005, n_timepoints),
        0.02 + 0.19 * (1 - np.exp(-4.5*t)) + np.random.normal(0, 0.005, n_timepoints)
    ])
    
    # =========================================================================
    # Generate Posterior Samples (Simulated MCMC output)
    # =========================================================================
    n_samples = 1000  # Number of posterior samples
    
    def gen_posterior(true_vals, uncertainty_scale=0.05):
        """Generate posterior samples around true values"""
        samples = np.zeros((n_samples, len(true_vals)))
        for i, tv in enumerate(true_vals):
            std = max(0.02, tv * uncertainty_scale)
            samples[:, i] = np.random.normal(tv, std, n_samples)
        return samples
    
    def add_correlations(samples, corr_pairs, corr_strength=0.7):
        """Add realistic correlations between parameters"""
        for i, j in corr_pairs:
            noise = np.random.normal(0, 0.1, n_samples)
            samples[:, j] = corr_strength * samples[:, i] + (1-corr_strength) * samples[:, j] + noise * 0.05
        return samples
    
    # M1 Posterior samples
    samples_M1 = gen_posterior(TRUE_M1, 0.03)
    samples_M1 = add_correlations(samples_M1, [(1, 3), (1, 4), (0, 2)])
    
    # M2 Posterior samples
    samples_M2 = gen_posterior(TRUE_M2, 0.04)
    samples_M2 = add_correlations(samples_M2, [(0, 2), (3, 4)])
    
    # M3 Posterior samples
    samples_M3 = gen_posterior(TRUE_M3, 0.03)
    samples_M3 = add_correlations(samples_M3, [(0, 2), (1, 3)])
    
    # =========================================================================
    # Create BayesianResults objects
    # =========================================================================
    def create_results(samples, param_names, aic_base, r_hat_vals, ess_vals, acc_rate):
        return BayesianResults(
            map_estimate=np.mean(samples, axis=0),
            posterior_samples=samples,
            posterior_mean=np.mean(samples, axis=0),
            posterior_std=np.std(samples, axis=0),
            ci_lower=np.percentile(samples, 2.5, axis=0),
            ci_upper=np.percentile(samples, 97.5, axis=0),
            convergence=ConvergenceDiagnostics(
                r_hat=np.array(r_hat_vals),
                ess=np.array(ess_vals),
                acceptance_rate=acc_rate,
                is_converged=True
            ),
            validation=ModelValidation(
                aic=aic_base, bic=aic_base + 12.8, 
                rmse=0.004, mae=0.003, r_squared=0.987
            ),
            param_names=param_names,
            loss_history=list(np.exp(-np.linspace(0, 3, 100)) * 500 + 50 + np.random.randn(100)*5)
        )
    
    results_M1 = create_results(
        samples_M1, param_names_M1, 125.4,
        [1.02, 1.01, 1.03, 1.02, 1.01], [850, 920, 780, 890, 910], 0.28
    )
    
    results_M2 = create_results(
        samples_M2, param_names_M2, 142.1,
        [1.03, 1.02, 1.04, 1.02, 1.08], [720, 850, 680, 790, 450], 0.25
    )
    
    results_M3 = create_results(
        samples_M3, param_names_M3, 98.5,
        [1.01, 1.02, 1.01, 1.03], [920, 880, 950, 820], 0.31
    )
    
    # Generate fit data
    fit_M1 = np.column_stack([
        0.25 + 0.65 * (1 - np.exp(-4*t)),
        0.30 + 0.55 * (1 - np.exp(-3*t))
    ])
    
    fit_M2 = np.column_stack([
        0.20 + 0.60 * (1 - np.exp(-3*t)) * np.exp(-0.5*t),
        0.20 + 0.50 * (1 - np.exp(-2*t)) * np.exp(-0.8*t)
    ])
    
    fit_M3 = np.column_stack([
        0.02 + 0.18 * (1 - np.exp(-5*t)),
        0.02 + 0.20 * (1 - np.exp(-4*t)),
        0.02 + 0.22 * (1 - np.exp(-3*t)),
        0.02 + 0.19 * (1 - np.exp(-4.5*t))
    ])
    
    # Validation data (antibiotic shock at t=0.5)
    t_val = np.linspace(0, 1, 150)
    val_data = np.zeros((150, 4))
    for i in range(4):
        growth = 0.02 + 0.18 * (1 - np.exp(-5 * t_val)) * (1 + 0.1 * (i - 1.5))
        decay_rate = 2 + i * 0.5
        val_data[:, i] = np.where(t_val > 0.5, 
                                   growth * np.exp(-decay_rate * (t_val - 0.5)),
                                   growth)
    
    return {
        # True parameters
        'TRUE_M1': TRUE_M1,
        'TRUE_M2': TRUE_M2,
        'TRUE_M3': TRUE_M3,
        # Parameter names
        'param_names_M1': param_names_M1,
        'param_names_M2': param_names_M2,
        'param_names_M3': param_names_M3,
        # Input data
        'd_M1': d_M1,
        'd_M2': d_M2,
        'd_M3': d_M3,
        't': t,
        # Results
        'results_M1': results_M1,
        'results_M2': results_M2,
        'results_M3': results_M3,
        # Fit data
        'fit_M1': fit_M1,
        'fit_M2': fit_M2,
        'fit_M3': fit_M3,
        # Validation
        't_val': t_val,
        'val_data': val_data,
        # Metadata
        'n_samples': n_samples,
        'n_timepoints': n_timepoints,
        'random_seed': RANDOM_SEED
    }

# =============================================================================
# VISUALIZATION FUNCTIONS (Same as before)
# =============================================================================
def plot_posterior_with_ci(results, true_values, param_names, filename, title):
    """Plot posterior with 95% confidence intervals"""
    n_params = len(param_names)
    fig, axes = plt.subplots(n_params, n_params, figsize=(13, 13))
    
    samples = results.posterior_samples
    
    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]
            
            if i == j:
                ax.hist(samples[:, i], bins=35, color='#4472C4', 
                       alpha=0.75, density=True, edgecolor='white', linewidth=0.5)
                ax.axvline(true_values[i], color='#C00000', linestyle='--', 
                          linewidth=2.5, label='True', zorder=5)
                ax.axvline(results.posterior_mean[i], color='#2E7D32', 
                          linestyle='-', linewidth=2, label='Mean', zorder=5)
                ax.axvspan(results.ci_lower[i], results.ci_upper[i], 
                          alpha=0.25, color='#FF8C00', label='95% CI')
                ax.axvline(results.ci_lower[i], color='#FF8C00', linestyle=':', linewidth=1.5)
                ax.axvline(results.ci_upper[i], color='#FF8C00', linestyle=':', linewidth=1.5)
                if i == 0:
                    ax.legend(fontsize=7, loc='upper right')
                    
            elif i > j:
                ax.scatter(samples[:, j], samples[:, i], alpha=0.15, s=2, c='#4472C4')
                ax.axvline(true_values[j], color='#C00000', linestyle='--', alpha=0.6, linewidth=1)
                ax.axhline(true_values[i], color='#C00000', linestyle='--', alpha=0.6, linewidth=1)
                ax.scatter([results.posterior_mean[j]], [results.posterior_mean[i]], 
                          color='#2E7D32', s=80, marker='x', linewidths=2, zorder=5)
            else:
                corr, _ = pearsonr(samples[:, i], samples[:, j])
                if abs(corr) > 0.7:
                    bg_color = '#FF6B6B' if corr > 0 else '#6B8EFF'
                elif abs(corr) > 0.4:
                    bg_color = '#FFB6B6' if corr > 0 else '#B6C4FF'
                else:
                    bg_color = '#F0F0F0'
                ax.set_facecolor(bg_color)
                ax.text(0.5, 0.5, f'ρ={corr:.3f}', ha='center', va='center', 
                       fontsize=10, fontweight='bold', transform=ax.transAxes)
            
            if i == n_params - 1:
                ax.set_xlabel(param_names[j], fontsize=10)
            else:
                ax.set_xticklabels([])
            
            if j == 0 and i != j:
                ax.set_ylabel(param_names[i], fontsize=10)
            elif j != 0:
                ax.set_yticklabels([])
    
    plt.suptitle(title, fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, filename), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {filename}")


def plot_convergence_diagnostics(results, param_names, filename, title):
    """Plot comprehensive convergence diagnostics"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # R-hat
    ax1 = axes[0, 0]
    x = np.arange(len(param_names))
    bars = ax1.bar(x, results.convergence.r_hat, color='#4472C4', edgecolor='white', linewidth=1.5)
    ax1.axhline(1.1, color='#C00000', linestyle='--', linewidth=2, label='Threshold (1.1)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(param_names, fontsize=10)
    ax1.set_ylabel('R-hat', fontsize=11)
    ax1.set_title('Gelman-Rubin Statistic (R̂)', fontweight='bold', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.set_ylim(0.95, 1.2)
    for bar, rhat in zip(bars, results.convergence.r_hat):
        if rhat > 1.1:
            bar.set_color('#C00000')
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{rhat:.3f}', ha='center', fontsize=9)
    
    # ESS
    ax2 = axes[0, 1]
    bars = ax2.bar(x, results.convergence.ess, color='#2E7D32', edgecolor='white', linewidth=1.5)
    ax2.axhline(100, color='#C00000', linestyle='--', linewidth=2, label='Threshold (100)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(param_names, fontsize=10)
    ax2.set_ylabel('ESS', fontsize=11)
    ax2.set_title('Effective Sample Size', fontweight='bold', fontsize=12)
    ax2.legend(fontsize=9)
    for bar, ess in zip(bars, results.convergence.ess):
        if ess < 100:
            bar.set_color('#C00000')
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15, f'{ess:.0f}', ha='center', fontsize=9)
    
    # Loss trace
    ax3 = axes[1, 0]
    iterations = np.arange(len(results.loss_history))
    ax3.plot(iterations, results.loss_history, color='#4472C4', linewidth=1, alpha=0.8)
    ax3.fill_between(iterations, results.loss_history, alpha=0.2, color='#4472C4')
    ax3.set_xlabel('Iteration', fontsize=11)
    ax3.set_ylabel('Loss', fontsize=11)
    ax3.set_title('Optimization Trace', fontweight='bold', fontsize=12)
    
    # Summary table
    ax4 = axes[1, 1]
    ax4.axis('off')
    conv = results.convergence
    val = results.validation
    summary_data = [
        ['Convergence Metric', 'Value', 'Status'],
        ['R-hat (max)', f'{np.max(conv.r_hat):.4f}', '✓' if np.max(conv.r_hat) < 1.1 else '✗'],
        ['ESS (min)', f'{np.min(conv.ess):.0f}', '✓' if np.min(conv.ess) > 100 else '✗'],
        ['Acceptance Rate', f'{100*conv.acceptance_rate:.1f}%', 
         '✓' if 0.15 < conv.acceptance_rate < 0.50 else '✗'],
        ['', '', ''],
        ['Validation Metric', 'Value', ''],
        ['AIC', f'{val.aic:.2f}', ''],
        ['BIC', f'{val.bic:.2f}', ''],
        ['RMSE', f'{val.rmse:.6f}', ''],
        ['R²', f'{val.r_squared:.4f}', ''],
        ['', '', ''],
        ['Overall Status', '', '✓ CONVERGED' if conv.is_converged else '✗ CHECK']
    ]
    table = ax4.table(cellText=summary_data, loc='center', cellLoc='center', colWidths=[0.45, 0.3, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.6)
    for i in range(3):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
        table[(5, i)].set_facecolor('#2E7D32')
        table[(5, i)].set_text_props(color='white', fontweight='bold')
    table[(11, 0)].set_facecolor('#FFE4B5')
    table[(11, 1)].set_facecolor('#FFE4B5')
    table[(11, 2)].set_facecolor('#90EE90' if conv.is_converged else '#FFB6C1')
    table[(11, 2)].set_text_props(fontweight='bold')
    ax4.set_title('Diagnostics Summary', fontweight='bold', fontsize=12, pad=20)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, filename), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {filename}")


def plot_model_fit_with_bands(t, data, fit, species_indices, title, filename):
    """Plot model fit with uncertainty bands"""
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    fig, ax = plt.subplots(figsize=(11, 6))
    
    for i, sp_idx in enumerate(species_indices):
        color = colors[sp_idx - 1]
        std = np.abs(fit[:, i]) * 0.05 + 0.005
        lower = fit[:, i] - 1.96 * std
        upper = fit[:, i] + 1.96 * std
        ax.scatter(t, data[:, i], color=color, alpha=0.7, s=60, edgecolors='white', linewidth=1,
                  label=f'Data Species {sp_idx}', zorder=4)
        ax.plot(t, fit[:, i], '-', color=color, linewidth=2.5, label=f'Fit Species {sp_idx}', zorder=3)
        ax.fill_between(t, lower, upper, color=color, alpha=0.2, zorder=2)
    
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel("Normalized Time $t$", fontsize=11)
    ax.set_ylabel("Living Biomass $\\overline{\\Phi}(t)$", fontsize=11)
    ax.set_xlim(0, 1)
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, filename), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {filename}")


def plot_parameters_comparison(estimated, true_values, ci_lower, ci_upper, labels, filename):
    """Enhanced parameter comparison with CI"""
    fig, ax = plt.subplots(figsize=(16, 7))
    x = np.arange(len(true_values))
    width = 0.35
    
    bars1 = ax.bar(x + width/2, true_values, width, label='True Mean', 
                  color='#FF8C00', alpha=0.85, edgecolor='white', linewidth=1.5)
    errors = np.array([estimated - ci_lower, ci_upper - estimated])
    bars2 = ax.bar(x - width/2, estimated, width, label='Posterior Mean',
                  color='#4472C4', alpha=0.85, edgecolor='white', linewidth=1.5,
                  yerr=errors, capsize=4, error_kw={'linewidth': 1.5, 'color': 'black'})
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Parameter Value', fontsize=12)
    ax.set_title('Parameter Estimation with 95% Confidence Intervals', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.axvline(x=9.5, color='gray', linestyle='--', alpha=0.6, linewidth=1.5)
    ymax = max(np.max(true_values), np.max(ci_upper)) * 1.12
    ax.text(4.5, ymax, "Interaction Parameters (A)", ha='center', fontsize=11, fontstyle='italic')
    ax.text(11.5, ymax, "Sensitivity (B)", ha='center', fontsize=11, fontstyle='italic')
    ax.set_ylim(0, ymax * 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, filename), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {filename}")


def plot_validation(t_val, val_data, filename):
    """Plot validation with antibiotic shock"""
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    ax.axvspan(0.0, 0.5, color='#4472C4', alpha=0.06, label='Updating Phase')
    ax.axvspan(0.5, 1.0, color='#C00000', alpha=0.06, label='Prediction Phase')
    
    for i in range(4):
        std = val_data[:, i] * 0.08 + 0.003
        ax.plot(t_val, val_data[:, i], color=colors[i], linewidth=2.5, label=f'Species {i+1}', zorder=3)
        ax.fill_between(t_val, val_data[:, i] - 1.96*std, val_data[:, i] + 1.96*std, 
                       color=colors[i], alpha=0.15, zorder=2)
    
    ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2.5, label='Antibiotics ON', zorder=4)
    ax.set_title('Validation: Antibiotic Shock at t=0.5', fontsize=14, fontweight='bold')
    ax.set_xlabel("Normalized Time $t$", fontsize=12)
    ax.set_ylabel("Living Biomass $\\overline{\\Phi}(t)$", fontsize=12)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, None)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, filename), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {filename}")


# =============================================================================
# MAIN EXECUTION WITH COMPLETE DATA SAVING
# =============================================================================
def main():
    print("="*75)
    print("  BAYESIAN BIOFILM ANALYSIS - COMPLETE DATA SAVING VERSION")
    print("="*75)
    print(f"  Output directory: {output_folder}/")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*75)
    
    # =========================================================================
    # STEP 1: Generate/Load Data
    # =========================================================================
    print("\n[STEP 1] Generating synthetic data and results...")
    data = generate_synthetic_data_and_results()
    print("  ✓ Data generation complete")
    
    # Unpack for convenience
    TRUE_M1 = data['TRUE_M1']
    TRUE_M2 = data['TRUE_M2']
    TRUE_M3 = data['TRUE_M3']
    param_names_M1 = data['param_names_M1']
    param_names_M2 = data['param_names_M2']
    param_names_M3 = data['param_names_M3']
    d_M1 = data['d_M1']
    d_M2 = data['d_M2']
    d_M3 = data['d_M3']
    t = data['t']
    results_M1 = data['results_M1']
    results_M2 = data['results_M2']
    results_M3 = data['results_M3']
    fit_M1 = data['fit_M1']
    fit_M2 = data['fit_M2']
    fit_M3 = data['fit_M3']
    t_val = data['t_val']
    val_data = data['val_data']
    
    # =========================================================================
    # STEP 2: Generate Visualizations
    # =========================================================================
    print("\n[STEP 2] Generating visualizations...")
    
    # M1
    print("\n  --- M1 (Species 1 & 2) ---")
    plot_posterior_with_ci(results_M1, TRUE_M1, 
                          ['$a_{11}$', '$a_{12}$', '$a_{22}$', '$b_1$', '$b_2$'],
                          "fig08_M1_posterior.png", "M1 Posterior Distribution")
    plot_convergence_diagnostics(results_M1, param_names_M1, "fig08b_M1_convergence.png", "M1 Convergence")
    plot_model_fit_with_bands(t, d_M1, fit_M1, [1, 2], "M1 Model Fit", "fig09_M1_fit.png")
    
    # M2
    print("\n  --- M2 (Species 3 & 4) ---")
    plot_posterior_with_ci(results_M2, TRUE_M2,
                          ['$a_{33}$', '$a_{34}$', '$a_{44}$', '$b_3$', '$b_4$'],
                          "fig10_M2_posterior.png", "M2 Posterior Distribution")
    plot_convergence_diagnostics(results_M2, param_names_M2, "fig10b_M2_convergence.png", "M2 Convergence")
    plot_model_fit_with_bands(t, d_M2, fit_M2, [3, 4], "M2 Model Fit", "fig11_M2_fit.png")
    
    # M3
    print("\n  --- M3 (Cross-Interactions) ---")
    plot_posterior_with_ci(results_M3, TRUE_M3,
                          ['$a_{13}$', '$a_{14}$', '$a_{23}$', '$a_{24}$'],
                          "fig12_M3_posterior.png", "M3 Posterior Distribution")
    plot_convergence_diagnostics(results_M3, param_names_M3, "fig12b_M3_convergence.png", "M3 Convergence")
    plot_model_fit_with_bands(t, d_M3, fit_M3, [1, 2, 3, 4], "M3 Model Fit", "fig13_M3_fit.png")
    
    # Combined
    print("\n  --- Combined Results ---")
    raw_est = np.concatenate([results_M1.posterior_mean, results_M2.posterior_mean, results_M3.posterior_mean])
    raw_true = np.concatenate([TRUE_M1, TRUE_M2, TRUE_M3])
    raw_ci_lower = np.concatenate([results_M1.ci_lower, results_M2.ci_lower, results_M3.ci_lower])
    raw_ci_upper = np.concatenate([results_M1.ci_upper, results_M2.ci_upper, results_M3.ci_upper])
    raw_labels = np.array(["a11", "a12", "a22", "b1", "b2", "a33", "a34", "a44", "b3", "b4", "a13", "a14", "a23", "a24"])
    new_order = [0, 1, 10, 11, 2, 12, 13, 5, 6, 7, 3, 4, 8, 9]
    
    plot_parameters_comparison(raw_est[new_order], raw_true[new_order], 
                              raw_ci_lower[new_order], raw_ci_upper[new_order],
                              raw_labels[new_order], "fig14_parameters_comparison.png")
    
    # Validation
    print("\n  --- Validation ---")
    plot_validation(t_val, val_data, "fig15_validation.png")
    
    # =========================================================================
    # STEP 3: SAVE ALL DATA (COMPLETE BACKUP)
    # =========================================================================
    print("\n" + "="*75)
    print("[STEP 3] SAVING ALL DATA (COMPLETE BACKUP)")
    print("="*75)
    
    # ---------------------------------------------------------
    # 3.1 Summary Statistics (CSV)
    # ---------------------------------------------------------
    df_summary = pd.DataFrame({
        "Parameter": raw_labels[new_order],
        "True": raw_true[new_order],
        "Estimated": np.round(raw_est[new_order], 6),
        "Std": np.round(np.concatenate([results_M1.posterior_std, results_M2.posterior_std, 
                                        results_M3.posterior_std])[new_order], 6),
        "CI_Lower_95": np.round(raw_ci_lower[new_order], 6),
        "CI_Upper_95": np.round(raw_ci_upper[new_order], 6),
        "Error": np.round(raw_est[new_order] - raw_true[new_order], 6),
        "Error_Percent": np.round(100 * (raw_est[new_order] - raw_true[new_order]) / (raw_true[new_order] + 1e-10), 2)
    })
    df_summary.to_csv(os.path.join(output_folder, "estimation_summary.csv"), index=False)
    print(f"  [1/7] ✓ Summary: estimation_summary.csv")
    
    # ---------------------------------------------------------
    # 3.2 Raw Posterior Samples (CSV + NPZ) - MOST IMPORTANT
    # ---------------------------------------------------------
    # Combine all samples
    raw_samples = np.hstack([
        results_M1.posterior_samples,
        results_M2.posterior_samples,
        results_M3.posterior_samples
    ])
    
    all_param_names = (
        ['m1_a11', 'm1_a12', 'm1_a22', 'm1_b1', 'm1_b2'] +
        ['m2_a33', 'm2_a34', 'm2_a44', 'm2_b3', 'm2_b4'] +
        ['m3_a13', 'm3_a14', 'm3_a23', 'm3_a24']
    )
    
    df_samples = pd.DataFrame(raw_samples, columns=all_param_names)
    df_samples.to_csv(os.path.join(output_folder, "posterior_samples_raw.csv"), index=False)
    print(f"  [2/7] ✓ Posterior samples (CSV): posterior_samples_raw.csv (Shape: {raw_samples.shape})")
    
    # Also save as NPZ for faster loading
    np.savez_compressed(
        os.path.join(output_folder, "posterior_samples_raw.npz"),
        samples_M1=results_M1.posterior_samples,
        samples_M2=results_M2.posterior_samples,
        samples_M3=results_M3.posterior_samples,
        param_names_M1=param_names_M1,
        param_names_M2=param_names_M2,
        param_names_M3=param_names_M3
    )
    print(f"  [2/7] ✓ Posterior samples (NPZ): posterior_samples_raw.npz")
    
    # ---------------------------------------------------------
    # 3.3 Input Data (Synthetic Experimental Data)
    # ---------------------------------------------------------
    # M1
    df_d_M1 = pd.DataFrame(d_M1, columns=['Species1', 'Species2'])
    df_d_M1['time'] = t
    df_d_M1.to_csv(os.path.join(output_folder, "input_data_M1.csv"), index=False)
    
    # M2
    df_d_M2 = pd.DataFrame(d_M2, columns=['Species3', 'Species4'])
    df_d_M2['time'] = t
    df_d_M2.to_csv(os.path.join(output_folder, "input_data_M2.csv"), index=False)
    
    # M3
    df_d_M3 = pd.DataFrame(d_M3, columns=['Species1', 'Species2', 'Species3', 'Species4'])
    df_d_M3['time'] = t
    df_d_M3.to_csv(os.path.join(output_folder, "input_data_M3.csv"), index=False)
    
    print(f"  [3/7] ✓ Input data: input_data_M1.csv, input_data_M2.csv, input_data_M3.csv")
    
    # ---------------------------------------------------------
    # 3.4 True Parameters
    # ---------------------------------------------------------
    df_true = pd.DataFrame({
        'Model': ['M1']*5 + ['M2']*5 + ['M3']*4,
        'Parameter': param_names_M1 + param_names_M2 + param_names_M3,
        'True_Value': list(TRUE_M1) + list(TRUE_M2) + list(TRUE_M3)
    })
    df_true.to_csv(os.path.join(output_folder, "true_parameters.csv"), index=False)
    print(f"  [4/7] ✓ True parameters: true_parameters.csv")
    
    # ---------------------------------------------------------
    # 3.5 Convergence Diagnostics
    # ---------------------------------------------------------
    conv_data = []
    for name, r, ess in zip(param_names_M1, results_M1.convergence.r_hat, results_M1.convergence.ess):
        conv_data.append({'Model': 'M1', 'Param': name, 'R_hat': r, 'ESS': ess, 
                         'AcceptRate': results_M1.convergence.acceptance_rate,
                         'Converged': results_M1.convergence.is_converged})
    for name, r, ess in zip(param_names_M2, results_M2.convergence.r_hat, results_M2.convergence.ess):
        conv_data.append({'Model': 'M2', 'Param': name, 'R_hat': r, 'ESS': ess,
                         'AcceptRate': results_M2.convergence.acceptance_rate,
                         'Converged': results_M2.convergence.is_converged})
    for name, r, ess in zip(param_names_M3, results_M3.convergence.r_hat, results_M3.convergence.ess):
        conv_data.append({'Model': 'M3', 'Param': name, 'R_hat': r, 'ESS': ess,
                         'AcceptRate': results_M3.convergence.acceptance_rate,
                         'Converged': results_M3.convergence.is_converged})
    
    pd.DataFrame(conv_data).to_csv(os.path.join(output_folder, "convergence_metrics.csv"), index=False)
    print(f"  [5/7] ✓ Convergence metrics: convergence_metrics.csv")
    
    # ---------------------------------------------------------
    # 3.6 Optimization History
    # ---------------------------------------------------------
    max_len = max(len(results_M1.loss_history), len(results_M2.loss_history), len(results_M3.loss_history))
    
    def pad_list(lst, length):
        return lst + [np.nan] * (length - len(lst))
    
    df_loss = pd.DataFrame({
        'Iteration': range(max_len),
        'Loss_M1': pad_list(results_M1.loss_history, max_len),
        'Loss_M2': pad_list(results_M2.loss_history, max_len),
        'Loss_M3': pad_list(results_M3.loss_history, max_len)
    })
    df_loss.to_csv(os.path.join(output_folder, "optimization_history.csv"), index=False)
    print(f"  [6/7] ✓ Optimization history: optimization_history.csv")
    
    # ---------------------------------------------------------
    # 3.7 Validation Data
    # ---------------------------------------------------------
    df_val = pd.DataFrame(val_data, columns=['Species1', 'Species2', 'Species3', 'Species4'])
    df_val['time'] = t_val
    df_val.to_csv(os.path.join(output_folder, "validation_prediction.csv"), index=False)
    print(f"  [7/7] ✓ Validation data: validation_prediction.csv")
    
    # ---------------------------------------------------------
    # 3.8 Metadata (JSON)
    # ---------------------------------------------------------
    metadata = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'random_seed': RANDOM_SEED,
        'n_posterior_samples': data['n_samples'],
        'n_timepoints': data['n_timepoints'],
        'true_parameters': {
            'M1': list(TRUE_M1),
            'M2': list(TRUE_M2),
            'M3': list(TRUE_M3)
        },
        'parameter_names': {
            'M1': param_names_M1,
            'M2': param_names_M2,
            'M3': param_names_M3
        },
        'convergence_summary': {
            'M1': {'converged': results_M1.convergence.is_converged, 
                   'max_rhat': float(np.max(results_M1.convergence.r_hat)),
                   'min_ess': float(np.min(results_M1.convergence.ess))},
            'M2': {'converged': results_M2.convergence.is_converged,
                   'max_rhat': float(np.max(results_M2.convergence.r_hat)),
                   'min_ess': float(np.min(results_M2.convergence.ess))},
            'M3': {'converged': results_M3.convergence.is_converged,
                   'max_rhat': float(np.max(results_M3.convergence.r_hat)),
                   'min_ess': float(np.min(results_M3.convergence.ess))}
        }
    }
    
    with open(os.path.join(output_folder, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  [+]   ✓ Metadata: metadata.json")
    
    # ---------------------------------------------------------
    # 3.9 Complete Archive (NPZ - everything in one file)
    # ---------------------------------------------------------
    np.savez_compressed(
        os.path.join(output_folder, "complete_archive.npz"),
        # True parameters
        TRUE_M1=TRUE_M1, TRUE_M2=TRUE_M2, TRUE_M3=TRUE_M3,
        # Input data
        d_M1=d_M1, d_M2=d_M2, d_M3=d_M3, t=t,
        # Posterior samples
        samples_M1=results_M1.posterior_samples,
        samples_M2=results_M2.posterior_samples,
        samples_M3=results_M3.posterior_samples,
        # Posterior statistics
        mean_M1=results_M1.posterior_mean, mean_M2=results_M2.posterior_mean, mean_M3=results_M3.posterior_mean,
        std_M1=results_M1.posterior_std, std_M2=results_M2.posterior_std, std_M3=results_M3.posterior_std,
        ci_lower_M1=results_M1.ci_lower, ci_lower_M2=results_M2.ci_lower, ci_lower_M3=results_M3.ci_lower,
        ci_upper_M1=results_M1.ci_upper, ci_upper_M2=results_M2.ci_upper, ci_upper_M3=results_M3.ci_upper,
        # Convergence
        r_hat_M1=results_M1.convergence.r_hat, r_hat_M2=results_M2.convergence.r_hat, r_hat_M3=results_M3.convergence.r_hat,
        ess_M1=results_M1.convergence.ess, ess_M2=results_M2.convergence.ess, ess_M3=results_M3.convergence.ess,
        # Validation
        t_val=t_val, val_data=val_data,
        # Fit data
        fit_M1=fit_M1, fit_M2=fit_M2, fit_M3=fit_M3
    )
    print(f"  [+]   ✓ Complete archive: complete_archive.npz")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "="*75)
    print("  RESULTS SUMMARY")
    print("="*75)
    print(df_summary.to_string(index=False))
    
    print("\n" + "="*75)
    print("  ALL DATA SAVED SUCCESSFULLY!")
    print("="*75)
    print(f"\n  Output directory: {output_folder}/")
    print("\n  Saved files:")
    for f in sorted(os.listdir(output_folder)):
        size = os.path.getsize(os.path.join(output_folder, f))
        size_str = f"{size/1024:.1f} KB" if size > 1024 else f"{size} B"
        print(f"    • {f:<40} ({size_str})")
    
    print("\n" + "-"*75)
    print("  Data Types Saved:")
    print("    1. ✓ Summary statistics (estimation_summary.csv)")
    print("    2. ✓ Raw posterior samples (posterior_samples_raw.csv/.npz)")
    print("    3. ✓ Input experimental data (input_data_M1/M2/M3.csv)")
    print("    4. ✓ True parameters (true_parameters.csv)")
    print("    5. ✓ Convergence diagnostics (convergence_metrics.csv)")
    print("    6. ✓ Optimization history (optimization_history.csv)")
    print("    7. ✓ Validation predictions (validation_prediction.csv)")
    print("    8. ✓ Metadata (metadata.json)")
    print("    9. ✓ Complete archive (complete_archive.npz)")
    print("-"*75)
    print("\n  To reload all data later:")
    print("    data = np.load('complete_archive.npz')")
    print("    samples_M1 = data['samples_M1']")
    print("-"*75)

if __name__ == "__main__":
    main()
