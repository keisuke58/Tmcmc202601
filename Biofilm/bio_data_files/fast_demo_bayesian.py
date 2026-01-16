#!/usr/bin/env python3
"""
Enhanced Bayesian Biofilm Analysis - Fast Demo Version
=======================================================

This is a faster demonstration that shows all key enhancements:
1. Confidence Interval Visualization (95% CI)
2. Convergence Diagnostics (R-hat, ESS)
3. Model Validation Metrics (AIC, BIC, RMSE)
4. Sensitivity Analysis (Tornado plot)
5. Summary Report
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, norm
import pandas as pd
import os
from dataclasses import dataclass
from typing import List, Tuple
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'figure.dpi': 120,
    'lines.linewidth': 2,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.facecolor': 'white'
})

output_folder = "figures_enhanced"
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
# SIMULATED RESULTS (Based on Paper Case II)
# =============================================================================
def generate_simulated_results():
    """
    Generate simulated posterior samples based on paper's results.
    This demonstrates all visualization features without expensive computation.
    """
    
    # True parameters from paper
    TRUE_M1 = np.array([0.8, 2.0, 1.0, 0.1, 0.2])
    TRUE_M2 = np.array([1.5, 1.0, 2.0, 0.3, 0.4])
    TRUE_M3 = np.array([2.0, 1.0, 2.0, 1.0])
    
    n_samples = 1000
    
    # Generate posterior samples around true values with realistic uncertainty
    def gen_posterior(true_vals, uncertainty_scale=0.05):
        samples = np.zeros((n_samples, len(true_vals)))
        for i, tv in enumerate(true_vals):
            std = max(0.02, tv * uncertainty_scale)
            samples[:, i] = np.random.normal(tv, std, n_samples)
        return samples
    
    # Add correlations (as observed in paper)
    def add_correlations(samples, corr_pairs, corr_strength=0.7):
        for i, j in corr_pairs:
            noise = np.random.normal(0, 0.1, n_samples)
            samples[:, j] = corr_strength * samples[:, i] + (1-corr_strength) * samples[:, j] + noise * 0.05
        return samples
    
    # M1 Results
    samples_M1 = gen_posterior(TRUE_M1, 0.03)
    samples_M1 = add_correlations(samples_M1, [(1, 3), (1, 4), (0, 2)])
    
    results_M1 = BayesianResults(
        map_estimate=np.mean(samples_M1, axis=0),
        posterior_samples=samples_M1,
        posterior_mean=np.mean(samples_M1, axis=0),
        posterior_std=np.std(samples_M1, axis=0),
        ci_lower=np.percentile(samples_M1, 2.5, axis=0),
        ci_upper=np.percentile(samples_M1, 97.5, axis=0),
        convergence=ConvergenceDiagnostics(
            r_hat=np.array([1.02, 1.01, 1.03, 1.02, 1.01]),
            ess=np.array([850, 920, 780, 890, 910]),
            acceptance_rate=0.28,
            is_converged=True
        ),
        validation=ModelValidation(
            aic=125.4, bic=138.2, rmse=0.0042, mae=0.0031, r_squared=0.987
        ),
        param_names=['a11', 'a12', 'a22', 'b1', 'b2'],
        loss_history=list(np.exp(-np.linspace(0, 3, 100)) * 500 + 50 + np.random.randn(100)*5)
    )
    
    # M2 Results
    samples_M2 = gen_posterior(TRUE_M2, 0.04)
    samples_M2 = add_correlations(samples_M2, [(0, 2), (3, 4)])
    
    results_M2 = BayesianResults(
        map_estimate=np.mean(samples_M2, axis=0),
        posterior_samples=samples_M2,
        posterior_mean=np.mean(samples_M2, axis=0),
        posterior_std=np.std(samples_M2, axis=0),
        ci_lower=np.percentile(samples_M2, 2.5, axis=0),
        ci_upper=np.percentile(samples_M2, 97.5, axis=0),
        convergence=ConvergenceDiagnostics(
            r_hat=np.array([1.03, 1.02, 1.04, 1.02, 1.08]),
            ess=np.array([720, 850, 680, 790, 450]),
            acceptance_rate=0.25,
            is_converged=True
        ),
        validation=ModelValidation(
            aic=142.1, bic=155.8, rmse=0.0058, mae=0.0045, r_squared=0.972
        ),
        param_names=['a33', 'a34', 'a44', 'b3', 'b4'],
        loss_history=list(np.exp(-np.linspace(0, 2.5, 100)) * 600 + 70 + np.random.randn(100)*8)
    )
    
    # M3 Results
    samples_M3 = gen_posterior(TRUE_M3, 0.03)
    samples_M3 = add_correlations(samples_M3, [(0, 2), (1, 3)])
    
    results_M3 = BayesianResults(
        map_estimate=np.mean(samples_M3, axis=0),
        posterior_samples=samples_M3,
        posterior_mean=np.mean(samples_M3, axis=0),
        posterior_std=np.std(samples_M3, axis=0),
        ci_lower=np.percentile(samples_M3, 2.5, axis=0),
        ci_upper=np.percentile(samples_M3, 97.5, axis=0),
        convergence=ConvergenceDiagnostics(
            r_hat=np.array([1.01, 1.02, 1.01, 1.03]),
            ess=np.array([920, 880, 950, 820]),
            acceptance_rate=0.31,
            is_converged=True
        ),
        validation=ModelValidation(
            aic=98.5, bic=108.3, rmse=0.0035, mae=0.0028, r_squared=0.991
        ),
        param_names=['a13', 'a14', 'a23', 'a24'],
        loss_history=list(np.exp(-np.linspace(0, 3.5, 100)) * 400 + 40 + np.random.randn(100)*4)
    )
    
    return results_M1, results_M2, results_M3, TRUE_M1, TRUE_M2, TRUE_M3

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================
def plot_posterior_with_ci(results: BayesianResults, 
                           true_values: np.ndarray,
                           param_names: List[str],
                           filename: str,
                           title: str):
    """Plot posterior with 95% confidence intervals - Paper Fig 3, 8, 10, 12 style"""
    n_params = len(param_names)
    fig, axes = plt.subplots(n_params, n_params, figsize=(13, 13))
    
    samples = results.posterior_samples
    
    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]
            
            if i == j:
                # Diagonal: Histogram with CI
                ax.hist(samples[:, i], bins=35, color='#4472C4', 
                       alpha=0.75, density=True, edgecolor='white', linewidth=0.5)
                ax.axvline(true_values[i], color='#C00000', linestyle='--', 
                          linewidth=2.5, label='True', zorder=5)
                ax.axvline(results.posterior_mean[i], color='#2E7D32', 
                          linestyle='-', linewidth=2, label='Mean', zorder=5)
                
                # 95% CI shading
                ax.axvspan(results.ci_lower[i], results.ci_upper[i], 
                          alpha=0.25, color='#FF8C00', label='95% CI')
                ax.axvline(results.ci_lower[i], color='#FF8C00', 
                          linestyle=':', linewidth=1.5)
                ax.axvline(results.ci_upper[i], color='#FF8C00', 
                          linestyle=':', linewidth=1.5)
                
                if i == 0:
                    ax.legend(fontsize=7, loc='upper right')
                    
            elif i > j:
                # Lower triangle: Scatter
                ax.scatter(samples[:, j], samples[:, i], alpha=0.15, s=2, c='#4472C4')
                ax.axvline(true_values[j], color='#C00000', linestyle='--', alpha=0.6, linewidth=1)
                ax.axhline(true_values[i], color='#C00000', linestyle='--', alpha=0.6, linewidth=1)
                ax.scatter([results.posterior_mean[j]], [results.posterior_mean[i]], 
                          color='#2E7D32', s=80, marker='x', linewidths=2, zorder=5)
                
            else:
                # Upper triangle: Correlation
                corr, _ = pearsonr(samples[:, i], samples[:, j])
                # Color based on correlation
                if abs(corr) > 0.7:
                    bg_color = '#FF6B6B' if corr > 0 else '#6B8EFF'
                elif abs(corr) > 0.4:
                    bg_color = '#FFB6B6' if corr > 0 else '#B6C4FF'
                else:
                    bg_color = '#F0F0F0'
                ax.set_facecolor(bg_color)
                ax.text(0.5, 0.5, f'ρ={corr:.3f}', 
                       ha='center', va='center', fontsize=10,
                       fontweight='bold', transform=ax.transAxes)
            
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
    plt.savefig(os.path.join(output_folder, filename), dpi=150, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close()
    print(f"  ✓ Saved: {filename}")


def plot_convergence_diagnostics(results: BayesianResults, 
                                 param_names: List[str],
                                 filename: str,
                                 title: str):
    """Plot comprehensive convergence diagnostics"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. R-hat values
    ax1 = axes[0, 0]
    x = np.arange(len(param_names))
    bars = ax1.bar(x, results.convergence.r_hat, color='#4472C4', 
                  edgecolor='white', linewidth=1.5)
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
        color = 'white' if rhat > 1.05 else 'black'
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{rhat:.3f}', ha='center', fontsize=9, color='black')
    
    # 2. ESS values
    ax2 = axes[0, 1]
    bars = ax2.bar(x, results.convergence.ess, color='#2E7D32', 
                  edgecolor='white', linewidth=1.5)
    ax2.axhline(100, color='#C00000', linestyle='--', linewidth=2, label='Threshold (100)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(param_names, fontsize=10)
    ax2.set_ylabel('ESS', fontsize=11)
    ax2.set_title('Effective Sample Size', fontweight='bold', fontsize=12)
    ax2.legend(fontsize=9)
    
    for bar, ess in zip(bars, results.convergence.ess):
        if ess < 100:
            bar.set_color('#C00000')
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
                f'{ess:.0f}', ha='center', fontsize=9)
    
    # 3. Loss trace
    ax3 = axes[1, 0]
    iterations = np.arange(len(results.loss_history))
    ax3.plot(iterations, results.loss_history, color='#4472C4', linewidth=1, alpha=0.8)
    ax3.fill_between(iterations, results.loss_history, alpha=0.2, color='#4472C4')
    ax3.set_xlabel('Iteration', fontsize=11)
    ax3.set_ylabel('Loss (Neg. Log-Likelihood)', fontsize=11)
    ax3.set_title('Optimization Trace', fontweight='bold', fontsize=12)
    
    # Add moving average
    window = 10
    if len(results.loss_history) > window:
        ma = np.convolve(results.loss_history, np.ones(window)/window, mode='valid')
        ax3.plot(np.arange(window-1, len(results.loss_history)), ma, 
                color='#C00000', linewidth=2, label='Moving Avg.')
        ax3.legend(fontsize=9)
    
    # 4. Summary table
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
    
    table = ax4.table(cellText=summary_data, loc='center', cellLoc='center',
                     colWidths=[0.45, 0.3, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.6)
    
    # Style header and sections
    for i in range(3):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
        table[(5, i)].set_facecolor('#2E7D32')
        table[(5, i)].set_text_props(color='white', fontweight='bold')
    
    # Status row
    table[(11, 0)].set_facecolor('#FFE4B5')
    table[(11, 1)].set_facecolor('#FFE4B5')
    table[(11, 2)].set_facecolor('#90EE90' if conv.is_converged else '#FFB6C1')
    table[(11, 2)].set_text_props(fontweight='bold')
    
    ax4.set_title('Diagnostics Summary', fontweight='bold', fontsize=12, pad=20)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, filename), dpi=150, bbox_inches='tight',
               facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {filename}")


def plot_model_fit_with_bands(t: np.ndarray, data: np.ndarray, 
                              fit: np.ndarray, species_indices: List[int],
                              title: str, filename: str):
    """Plot model fit with uncertainty bands"""
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    fig, ax = plt.subplots(figsize=(11, 6))
    
    for i, sp_idx in enumerate(species_indices):
        color = colors[sp_idx - 1]
        
        # Generate synthetic uncertainty band
        std = np.abs(fit[:, i]) * 0.05 + 0.005
        lower = fit[:, i] - 1.96 * std
        upper = fit[:, i] + 1.96 * std
        
        # Data points
        ax.scatter(t, data[:, i], color=color, alpha=0.7, s=60, 
                  edgecolors='white', linewidth=1,
                  label=f'Data Species {sp_idx}', zorder=4)
        
        # Fit line
        ax.plot(t, fit[:, i], '-', color=color, linewidth=2.5,
               label=f'Fit Species {sp_idx}', zorder=3)
        
        # 95% CI band
        ax.fill_between(t, lower, upper, color=color, alpha=0.2, zorder=2)
    
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel("Normalized Time $t$", fontsize=11)
    ax.set_ylabel("Living Biomass $\\overline{\\Phi}(t)$", fontsize=11)
    ax.set_xlim(0, 1)
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, filename), dpi=150, bbox_inches='tight',
               facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {filename}")


def plot_parameters_comparison_enhanced(estimated: np.ndarray,
                                        true_values: np.ndarray,
                                        ci_lower: np.ndarray,
                                        ci_upper: np.ndarray,
                                        labels: np.ndarray,
                                        filename: str):
    """Enhanced parameter comparison - Paper Fig 14 style with CI"""
    fig, ax = plt.subplots(figsize=(16, 7))
    
    x = np.arange(len(true_values))
    width = 0.35
    
    # True values
    bars1 = ax.bar(x + width/2, true_values, width, label='True Mean', 
                  color='#FF8C00', alpha=0.85, edgecolor='white', linewidth=1.5)
    
    # Estimated with error bars
    errors = np.array([estimated - ci_lower, ci_upper - estimated])
    bars2 = ax.bar(x - width/2, estimated, width, label='Posterior Mean',
                  color='#4472C4', alpha=0.85, edgecolor='white', linewidth=1.5,
                  yerr=errors, capsize=4, error_kw={'linewidth': 1.5, 'color': 'black'})
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Parameter Value', fontsize=12)
    ax.set_title('Parameter Estimation with 95% Confidence Intervals', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    
    # Add divider between A and B parameters
    ax.axvline(x=9.5, color='gray', linestyle='--', alpha=0.6, linewidth=1.5)
    ymax = max(np.max(true_values), np.max(ci_upper)) * 1.12
    ax.text(4.5, ymax, "Interaction Parameters (A)", ha='center', fontsize=11, 
           fontstyle='italic')
    ax.text(11.5, ymax, "Sensitivity (B)", ha='center', fontsize=11,
           fontstyle='italic')
    
    ax.set_ylim(0, ymax * 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, filename), dpi=150, bbox_inches='tight',
               facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {filename}")


def plot_validation_antibiotic_shock(filename: str):
    """Plot validation with time-dependent antibiotics - Paper Fig 15 style"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Generate simulated validation data
    t = np.linspace(0, 1, 150)
    
    # Phase regions
    ax.axvspan(0.0, 0.5, color='#4472C4', alpha=0.06, label='Updating Phase')
    ax.axvspan(0.5, 1.0, color='#C00000', alpha=0.06, label='Prediction Phase')
    
    # Simulated trajectories
    for i in range(4):
        # Growth phase (t < 0.5)
        growth = 0.02 + 0.18 * (1 - np.exp(-5 * t)) * (1 + 0.1 * (i - 1.5))
        
        # Antibiotic effect (t > 0.5)
        decay_rate = 2 + i * 0.5
        decay = np.where(t > 0.5, 
                        growth * np.exp(-decay_rate * (t - 0.5)),
                        growth)
        
        # Add uncertainty band
        std = decay * 0.08 + 0.003
        
        ax.plot(t, decay, color=colors[i], linewidth=2.5, 
               label=f'Species {i+1}', zorder=3)
        ax.fill_between(t, decay - 1.96*std, decay + 1.96*std, 
                       color=colors[i], alpha=0.15, zorder=2)
    
    ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2.5,
              label='Antibiotics ON', zorder=4)
    
    ax.set_title('Validation: Antibiotic Shock at t=0.5', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel("Normalized Time $t$", fontsize=12)
    ax.set_ylabel("Living Biomass $\\overline{\\Phi}(t)$", fontsize=12)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, None)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, filename), dpi=150, bbox_inches='tight',
               facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {filename}")


def plot_sensitivity_tornado(param_names: List[str], filename: str, title: str):
    """Tornado plot for sensitivity analysis"""
    n = len(param_names)
    
    # Simulated Sobol indices
    first_order = np.random.uniform(0.05, 0.35, n)
    first_order = first_order / first_order.sum() * 0.8  # Normalize
    
    total_order = first_order + np.random.uniform(0.02, 0.15, n)
    total_order = np.clip(total_order, 0, 1)
    
    # Sort by total order
    order = np.argsort(total_order)[::-1]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # First-order
    ax1 = axes[0]
    y = np.arange(n)
    ax1.barh(y, first_order[order], color='#4472C4', edgecolor='white', height=0.7)
    ax1.set_yticks(y)
    ax1.set_yticklabels([param_names[i] for i in order])
    ax1.set_xlabel('First-Order Index ($S_i$)', fontsize=11)
    ax1.set_title('Main Effects', fontweight='bold', fontsize=12)
    ax1.set_xlim(0, 0.5)
    ax1.invert_yaxis()
    
    for i, (idx, val) in enumerate(zip(order, first_order[order])):
        ax1.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=9)
    
    # Total-order
    ax2 = axes[1]
    ax2.barh(y, total_order[order], color='#2E7D32', edgecolor='white', height=0.7)
    ax2.set_yticks(y)
    ax2.set_yticklabels([param_names[i] for i in order])
    ax2.set_xlabel('Total-Order Index ($S_{Ti}$)', fontsize=11)
    ax2.set_title('Total Effects (incl. Interactions)', fontweight='bold', fontsize=12)
    ax2.set_xlim(0, 0.5)
    ax2.invert_yaxis()
    
    for i, (idx, val) in enumerate(zip(order, total_order[order])):
        ax2.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=9)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, filename), dpi=150, bbox_inches='tight',
               facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {filename}")


def create_summary_report(results_M1, results_M2, results_M3,
                         TRUE_M1, TRUE_M2, TRUE_M3, filename: str):
    """Create comprehensive summary report"""
    fig = plt.figure(figsize=(18, 14))
    
    fig.suptitle('Bayesian Model Updating - Comprehensive Summary Report', 
                fontsize=16, fontweight='bold', y=0.98)
    
    gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.25,
                         height_ratios=[1.2, 1, 1, 0.8])
    
    # Row 1: Parameter estimates for each model
    for col, (results, true_vals, title, pnames) in enumerate([
        (results_M1, TRUE_M1, 'M1: Species 1 & 2', ['a11', 'a12', 'a22', 'b1', 'b2']),
        (results_M2, TRUE_M2, 'M2: Species 3 & 4', ['a33', 'a34', 'a44', 'b3', 'b4']),
        (results_M3, TRUE_M3, 'M3: Cross-Interactions', ['a13', 'a14', 'a23', 'a24'])
    ]):
        ax = fig.add_subplot(gs[0, col])
        x = np.arange(len(true_vals))
        width = 0.35
        
        ax.bar(x - width/2, true_vals, width, label='True', color='#FF8C00', alpha=0.8)
        
        errors = np.array([results.posterior_mean - results.ci_lower,
                          results.ci_upper - results.posterior_mean])
        ax.bar(x + width/2, results.posterior_mean, width, label='Posterior',
              color='#4472C4', alpha=0.8, yerr=errors, capsize=3)
        
        ax.set_xticks(x)
        ax.set_xticklabels(pnames, fontsize=9)
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.legend(fontsize=8, loc='upper right')
        ax.set_ylabel('Value', fontsize=10)
    
    # Row 2: Convergence metrics
    ax_rhat = fig.add_subplot(gs[1, 0])
    ax_ess = fig.add_subplot(gs[1, 1])
    ax_accept = fig.add_subplot(gs[1, 2])
    
    # R-hat comparison
    all_rhats = [results_M1.convergence.r_hat, 
                 results_M2.convergence.r_hat,
                 results_M3.convergence.r_hat]
    all_labels = ['M1', 'M2', 'M3']
    
    for i, (rhat, label) in enumerate(zip(all_rhats, all_labels)):
        ax_rhat.bar(np.arange(len(rhat)) + i*0.25, rhat, 0.25, 
                   label=label, alpha=0.8)
    ax_rhat.axhline(1.1, color='#C00000', linestyle='--', linewidth=2)
    ax_rhat.set_title('R-hat by Model', fontweight='bold', fontsize=11)
    ax_rhat.legend(fontsize=8)
    ax_rhat.set_ylim(0.95, 1.15)
    
    # ESS comparison (min per model)
    min_ess = [np.min(r.convergence.ess) for r in [results_M1, results_M2, results_M3]]
    bars = ax_ess.bar(all_labels, min_ess, color=['#4472C4', '#FF8C00', '#2E7D32'], 
                     alpha=0.8, edgecolor='white')
    ax_ess.axhline(100, color='#C00000', linestyle='--', linewidth=2)
    ax_ess.set_title('Minimum ESS by Model', fontweight='bold', fontsize=11)
    ax_ess.set_ylabel('ESS')
    for bar, val in zip(bars, min_ess):
        ax_ess.text(bar.get_x() + bar.get_width()/2, val + 20, 
                   f'{val:.0f}', ha='center', fontsize=10)
    
    # Acceptance rate
    acc_rates = [r.convergence.acceptance_rate for r in [results_M1, results_M2, results_M3]]
    bars = ax_accept.bar(all_labels, [r*100 for r in acc_rates], 
                        color=['#4472C4', '#FF8C00', '#2E7D32'], alpha=0.8, edgecolor='white')
    ax_accept.axhline(15, color='#C00000', linestyle='--', linewidth=1.5, alpha=0.7)
    ax_accept.axhline(50, color='#C00000', linestyle='--', linewidth=1.5, alpha=0.7)
    ax_accept.axhspan(15, 50, alpha=0.1, color='green', label='Optimal Range')
    ax_accept.set_title('Acceptance Rate', fontweight='bold', fontsize=11)
    ax_accept.set_ylabel('Rate (%)')
    ax_accept.legend(fontsize=8, loc='upper right')
    for bar, val in zip(bars, acc_rates):
        ax_accept.text(bar.get_x() + bar.get_width()/2, val*100 + 2, 
                      f'{val*100:.1f}%', ha='center', fontsize=10)
    
    # Row 3: Model validation metrics
    ax_val = fig.add_subplot(gs[2, :])
    ax_val.axis('off')
    
    val_data = [
        ['Metric', 'M1', 'M2', 'M3', 'Interpretation'],
        ['AIC', f'{results_M1.validation.aic:.1f}', 
         f'{results_M2.validation.aic:.1f}', 
         f'{results_M3.validation.aic:.1f}', 'Lower is better'],
        ['BIC', f'{results_M1.validation.bic:.1f}', 
         f'{results_M2.validation.bic:.1f}', 
         f'{results_M3.validation.bic:.1f}', 'Lower is better'],
        ['RMSE', f'{results_M1.validation.rmse:.5f}', 
         f'{results_M2.validation.rmse:.5f}', 
         f'{results_M3.validation.rmse:.5f}', 'Lower is better'],
        ['R²', f'{results_M1.validation.r_squared:.4f}', 
         f'{results_M2.validation.r_squared:.4f}', 
         f'{results_M3.validation.r_squared:.4f}', 'Higher is better (≤1)'],
        ['MAE', f'{results_M1.validation.mae:.5f}', 
         f'{results_M2.validation.mae:.5f}', 
         f'{results_M3.validation.mae:.5f}', 'Lower is better'],
    ]
    
    table = ax_val.table(cellText=val_data, loc='center', cellLoc='center',
                        colWidths=[0.18, 0.15, 0.15, 0.15, 0.30])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.15, 1.7)
    
    for i in range(5):
        table[(0, i)].set_facecolor('#2E7D32')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    ax_val.set_title('Model Validation Metrics', fontweight='bold', fontsize=12, pad=15)
    
    # Row 4: Overall status
    ax_status = fig.add_subplot(gs[3, :])
    ax_status.axis('off')
    
    # Check overall convergence
    all_converged = all(r.convergence.is_converged for r in [results_M1, results_M2, results_M3])
    
    status_text = "✓ ALL MODELS CONVERGED - READY FOR PREDICTION" if all_converged else \
                  "⚠ CHECK CONVERGENCE - SOME DIAGNOSTICS FAILED"
    status_color = '#90EE90' if all_converged else '#FFB6C1'
    
    ax_status.add_patch(plt.Rectangle((0.1, 0.2), 0.8, 0.6, 
                                       facecolor=status_color, edgecolor='black',
                                       linewidth=2, transform=ax_status.transAxes))
    ax_status.text(0.5, 0.5, status_text, transform=ax_status.transAxes,
                  fontsize=14, fontweight='bold', ha='center', va='center')
    
    plt.savefig(os.path.join(output_folder, filename), dpi=150, bbox_inches='tight',
               facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {filename}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    print("="*75)
    print("  ENHANCED BAYESIAN UPDATING - VISUALIZATION DEMO")
    print("  Confidence Intervals | Convergence | Validation | Sensitivity")
    print("="*75)
    
    # Generate simulated results
    print("\n[1] Generating simulated posterior samples...")
    results_M1, results_M2, results_M3, TRUE_M1, TRUE_M2, TRUE_M3 = generate_simulated_results()
    print("  ✓ Results generated")
    
    # Generate fit data
    t = np.linspace(0, 1, 10)
    
    # M1 fit data
    d_M1 = np.column_stack([
        0.25 + 0.65 * (1 - np.exp(-4*t)) + np.random.normal(0, 0.01, 10),
        0.30 + 0.55 * (1 - np.exp(-3*t)) + np.random.normal(0, 0.01, 10)
    ])
    fit_M1 = np.column_stack([
        0.25 + 0.65 * (1 - np.exp(-4*t)),
        0.30 + 0.55 * (1 - np.exp(-3*t))
    ])
    
    # M2 fit data
    d_M2 = np.column_stack([
        0.20 + 0.60 * (1 - np.exp(-3*t)) * np.exp(-0.5*t) + np.random.normal(0, 0.01, 10),
        0.20 + 0.50 * (1 - np.exp(-2*t)) * np.exp(-0.8*t) + np.random.normal(0, 0.01, 10)
    ])
    fit_M2 = np.column_stack([
        0.20 + 0.60 * (1 - np.exp(-3*t)) * np.exp(-0.5*t),
        0.20 + 0.50 * (1 - np.exp(-2*t)) * np.exp(-0.8*t)
    ])
    
    # M3 fit data
    d_M3 = np.column_stack([
        0.02 + 0.18 * (1 - np.exp(-5*t)) + np.random.normal(0, 0.005, 10),
        0.02 + 0.20 * (1 - np.exp(-4*t)) + np.random.normal(0, 0.005, 10),
        0.02 + 0.22 * (1 - np.exp(-3*t)) + np.random.normal(0, 0.005, 10),
        0.02 + 0.19 * (1 - np.exp(-4.5*t)) + np.random.normal(0, 0.005, 10)
    ])
    fit_M3 = np.column_stack([
        0.02 + 0.18 * (1 - np.exp(-5*t)),
        0.02 + 0.20 * (1 - np.exp(-4*t)),
        0.02 + 0.22 * (1 - np.exp(-3*t)),
        0.02 + 0.19 * (1 - np.exp(-4.5*t))
    ])
    
    # ==========================================================================
    # GENERATE ALL VISUALIZATIONS
    # ==========================================================================
    print("\n[2] Generating visualizations...")
    
    # M1 Posterior with CI
    print("\n  --- M1 (Species 1 & 2) ---")
    plot_posterior_with_ci(
        results_M1, TRUE_M1, 
        ['$a_{11}$', '$a_{12}$', '$a_{22}$', '$b_1$', '$b_2$'],
        "fig08_M1_posterior_with_CI.png",
        "M1 Posterior Distribution with 95% Confidence Intervals"
    )
    
    plot_convergence_diagnostics(
        results_M1, ['a11', 'a12', 'a22', 'b1', 'b2'],
        "fig08b_M1_convergence.png",
        "M1 Convergence Diagnostics"
    )
    
    plot_model_fit_with_bands(
        t, d_M1, fit_M1, [1, 2],
        "Fig 9: M1 Fit with 95% Confidence Bands",
        "fig09_M1_fit_with_CI.png"
    )
    
    # M2 Posterior with CI
    print("\n  --- M2 (Species 3 & 4) ---")
    plot_posterior_with_ci(
        results_M2, TRUE_M2,
        ['$a_{33}$', '$a_{34}$', '$a_{44}$', '$b_3$', '$b_4$'],
        "fig10_M2_posterior_with_CI.png",
        "M2 Posterior Distribution with 95% Confidence Intervals"
    )
    
    plot_convergence_diagnostics(
        results_M2, ['a33', 'a34', 'a44', 'b3', 'b4'],
        "fig10b_M2_convergence.png",
        "M2 Convergence Diagnostics"
    )
    
    plot_model_fit_with_bands(
        t, d_M2, fit_M2, [3, 4],
        "Fig 11: M2 Fit with 95% Confidence Bands",
        "fig11_M2_fit_with_CI.png"
    )
    
    # M3 Posterior with CI
    print("\n  --- M3 (Cross-Interactions) ---")
    plot_posterior_with_ci(
        results_M3, TRUE_M3,
        ['$a_{13}$', '$a_{14}$', '$a_{23}$', '$a_{24}$'],
        "fig12_M3_posterior_with_CI.png",
        "M3 Posterior Distribution with 95% Confidence Intervals"
    )
    
    plot_convergence_diagnostics(
        results_M3, ['a13', 'a14', 'a23', 'a24'],
        "fig12b_M3_convergence.png",
        "M3 Convergence Diagnostics"
    )
    
    plot_model_fit_with_bands(
        t, d_M3, fit_M3, [1, 2, 3, 4],
        "Fig 13: M3 Fit with 95% Confidence Bands (All Species)",
        "fig13_M3_fit_with_CI.png"
    )
    
    # Combined results
    print("\n  --- Combined Results ---")
    
    # Combine all parameters for comparison plot
    raw_est = np.concatenate([results_M1.posterior_mean, results_M2.posterior_mean, 
                              results_M3.posterior_mean])
    raw_true = np.concatenate([TRUE_M1, TRUE_M2, TRUE_M3])
    raw_ci_lower = np.concatenate([results_M1.ci_lower, results_M2.ci_lower, 
                                   results_M3.ci_lower])
    raw_ci_upper = np.concatenate([results_M1.ci_upper, results_M2.ci_upper,
                                   results_M3.ci_upper])
    raw_labels = np.array(["a11", "a12", "a22", "b1", "b2",
                          "a33", "a34", "a44", "b3", "b4",
                          "a13", "a14", "a23", "a24"])
    
    new_order = [0, 1, 10, 11, 2, 12, 13, 5, 6, 7, 3, 4, 8, 9]
    
    plot_parameters_comparison_enhanced(
        raw_est[new_order], raw_true[new_order],
        raw_ci_lower[new_order], raw_ci_upper[new_order],
        raw_labels[new_order],
        "fig14_parameters_comparison_enhanced.png"
    )
    
    # Validation plot
    print("\n  --- Validation ---")
    plot_validation_antibiotic_shock("fig15_validation_antibiotic_shock.png")
    
    # Sensitivity analysis
    print("\n  --- Sensitivity Analysis ---")
    plot_sensitivity_tornado(
        ['a11', 'a12', 'a22', 'b1', 'b2'],
        "fig16_sensitivity_M1.png",
        "M1 Sensitivity Analysis (Sobol Indices)"
    )
    
    # Summary report
    print("\n  --- Summary Report ---")
    create_summary_report(
        results_M1, results_M2, results_M3,
        TRUE_M1, TRUE_M2, TRUE_M3,
        "fig17_summary_report.png"
    )
    
    # ==========================================================================
    # SAVE RESULTS TABLE
    # ==========================================================================
    print("\n[3] Saving results table...")
    
    df = pd.DataFrame({
        "Parameter": raw_labels[new_order],
        "True": raw_true[new_order],
        "Estimated": np.round(raw_est[new_order], 4),
        "95% CI Lower": np.round(raw_ci_lower[new_order], 4),
        "95% CI Upper": np.round(raw_ci_upper[new_order], 4),
        "Error": np.round(raw_est[new_order] - raw_true[new_order], 4),
        "Error %": np.round(100 * (raw_est[new_order] - raw_true[new_order]) / 
                           (raw_true[new_order] + 1e-10), 2)
    })
    
    df.to_csv(os.path.join(output_folder, "estimation_results.csv"), index=False)
    print(f"  ✓ Saved: estimation_results.csv")
    
    print("\n" + "="*75)
    print("  RESULTS TABLE")
    print("="*75)
    print(df.to_string(index=False))
    
    # ==========================================================================
    # FINAL SUMMARY
    # ==========================================================================
    print("\n" + "="*75)
    print("  ALL ANALYSES COMPLETED SUCCESSFULLY")
    print("="*75)
    print(f"\n  Output directory: {output_folder}/")
    print("\n  Generated files:")
    for f in sorted(os.listdir(output_folder)):
        print(f"    • {f}")
    
    print("\n" + "-"*75)
    print("  Key Enhancements Demonstrated:")
    print("    1. ✓ 95% Confidence Intervals on all estimates")
    print("    2. ✓ Convergence Diagnostics (R-hat, ESS, Acceptance Rate)")
    print("    3. ✓ Model Validation Metrics (AIC, BIC, RMSE, R²)")
    print("    4. ✓ Sensitivity Analysis (Sobol Indices)")
    print("    5. ✓ Comprehensive Summary Report")
    print("-"*75)

if __name__ == "__main__":
    main()
