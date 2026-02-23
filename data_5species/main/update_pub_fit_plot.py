#!/usr/bin/env python3
"""
Update Publication Fit Comparison Plot (Uncertainty Quantification).
Requirements:
1. Specific color scheme (Yellow for V. dispar in Commensal).
2. Shared Y-axis across subplots.
3. X-axis starting at 0.
4. Output to specific analysis directory.
"""

import argparse
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Project path resolution
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_5SPECIES_ROOT = Path(__file__).parent.parent

sys.path.insert(0, str(DATA_5SPECIES_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tmcmc" / "program2602"))
sys.path.insert(0, str(PROJECT_ROOT / "tmcmc"))
sys.path.insert(0, str(PROJECT_ROOT))

# Set seaborn style
sns.set_context("paper", font_scale=1.5)
sns.set_style("whitegrid")

# Color mapping
COLORS_COMMENSAL = [
    '#1f77b4', # S0: Blue
    '#2ca02c', # S1: Green
    '#bcbd22', # S2: Yellow (using 'tab:olive' for better visibility)
    '#9467bd', # S3: Purple
    '#d62728'  # S4: Red
]

COLORS_DYSBIOTIC = [
    '#1f77b4', # S0: Blue
    '#2ca02c', # S1: Green
    '#ff7f0e', # S2: Orange
    '#9467bd', # S3: Purple
    '#d62728'  # S4: Red
]

SPECIES_NAMES = [
    "S. oralis", 
    "A. naeslundii", 
    "V. dispar", 
    "F. nucleatum", 
    "P. gingivalis"
]

def get_colors(condition: str) -> List[str]:
    """Get color palette based on condition."""
    if "Dysbiotic" in condition:
        return COLORS_DYSBIOTIC
    return COLORS_COMMENSAL

def load_data(run_dir: Path) -> Tuple[Any, Any, Any, Any, str]:
    """Load UQ data, history, and experimental data."""
    analysis_path = run_dir / "analysis"
    
    # Load UQ results
    uq_data = np.load(analysis_path / "uncertainty_quantification.npz")
    
    # Load raw history for spaghetti if available
    history_path = analysis_path / "raw_history_subset.npy"
    raw_history = np.load(history_path) if history_path.exists() else None
    
    # Load observed data
    data_points = np.load(run_dir / "data.npy")
    t_days = np.load(run_dir / "t_days.npy")
    
    # Load config for condition
    with open(run_dir / "config.json", 'r') as f:
        config = json.load(f)
    condition = config.get("condition", "Commensal")
    
    return uq_data, raw_history, data_points, t_days, condition

def plot_publication_fit(
    run_dir: Path, 
    output_path: Path
):
    """Generate a publication-quality fit comparison plot with UQ."""
    uq_data, raw_history, data_points, t_days, condition = load_data(run_dir)
    time = uq_data['time']
    
    # Map simulation time to Days
    t_min, t_max = time.min(), time.max()
    day_min, day_max = t_days.min(), t_days.max()
    
    if t_max > t_min:
        sim_days = day_min + (time - t_min) / (t_max - t_min) * (day_max - day_min)
    else:
        sim_days = time
        
    current_colors = get_colors(condition)
    
    # Share Y axis across all subplots
    fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharex=True, sharey=True)
    
    for i in range(5):
        ax = axes[i]
        species = f"S{i+1}" # UQ data uses 1-based indexing S1..S5
        name = SPECIES_NAMES[i]
        color = current_colors[i]
        
        # 1. Spaghetti Plot (Thin lines)
        if raw_history is not None:
            # raw_history: (n_samples, n_steps, n_species)
            # Limit to first 50 samples for clarity
            for sample_idx in range(min(50, raw_history.shape[0])):
                # Check if phibar needs calculation or is already phibar
                # Usually raw_history is phibar if from analyze_results.py
                # But let's verify dimensions. 
                # analyze_results.py saves `phibar_samples` as `raw_history_subset.npy`
                # So it is (n_samples, n_steps, 5)
                ax.plot(sim_days, raw_history[sample_idx, :, i], 
                       color=color, alpha=0.1, linewidth=0.5)
        
        # 2. Median Line
        if f"{species}_p50" in uq_data:
            ax.plot(sim_days, uq_data[f"{species}_p50"], 
                   color=color, linewidth=2.5, label='Median')
            
            # 3. 95% CI Band
            ax.fill_between(sim_days, 
                           uq_data[f"{species}_p2.5"], 
                           uq_data[f"{species}_p97.5"],
                           color=color, alpha=0.2, label='95% CI')
        
        # 4. Experimental Data
        ax.errorbar(t_days, data_points[:, i], 
                   yerr=0.05, # Assumed error bar if not provided
                   fmt='o', color='black', ecolor='black', 
                   capsize=3, markersize=6, zorder=10, label='Experiment')
        
        # Styling
        ax.set_title(name, fontsize=14, fontweight='bold', color=color)
        ax.set_xlabel("Time (Days)", fontsize=12)
        
        if i == 0:
            ax.set_ylabel("Relative Abundance", fontsize=12)
            
        ax.set_xticks(t_days)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Limits
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        
        if i == 0:
            # De-duplicate legend labels
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=10, frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Generated: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate updated UQ fit comparison plot")
    parser.add_argument("run_dir", type=str, help="Path to run directory")
    parser.add_argument("output_file", type=str, help="Full path to output file")
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from {run_dir}...")
    plot_publication_fit(run_dir, output_path)

if __name__ == "__main__":
    main()
