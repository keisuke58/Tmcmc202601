#!/usr/bin/env python3
"""
Generate publication-quality plots (MAP Fit and Posterior Band) using the Nishioka color scheme.
Supports both Commensal (Yellow S2) and Dysbiotic (Orange S2) conditions.
"""

import argparse
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Project path resolution
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_5SPECIES_ROOT = Path(__file__).parent.parent

sys.path.insert(0, str(DATA_5SPECIES_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tmcmc" / "program2602"))
sys.path.insert(0, str(PROJECT_ROOT / "tmcmc"))
sys.path.insert(0, str(PROJECT_ROOT))

# Try importing the solver
try:
    from improved_5species_jit import BiofilmNewtonSolver5S
except ImportError:
    sys.path.append(str(PROJECT_ROOT / "tmcmc" / "program2602"))
    from improved_5species_jit import BiofilmNewtonSolver5S

# Set seaborn style
sns.set_context("paper", font_scale=1.5)
sns.set_style("whitegrid")

# Color mapping
COLORS_COMMENSAL = [
    '#1f77b4', # S0: Blue
    '#2ca02c', # S1: Green
    '#bcbd22', # S2: Yellow (using 'tab:olive' as yellow is too bright)
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

def load_run_data(run_dir: Path) -> Dict[str, Any]:
    """Load necessary data for simulation."""
    data = {}
    
    with open(run_dir / "config.json", 'r') as f:
        data["config"] = json.load(f)
    
    if (run_dir / "theta_MAP.json").exists():
        with open(run_dir / "theta_MAP.json", 'r') as f:
            data["theta_MAP"] = json.load(f)
    
    if (run_dir / "samples.npy").exists():
        data["samples"] = np.load(run_dir / "samples.npy")
        
    data["data_points"] = np.load(run_dir / "data.npy")
    data["t_days"] = np.load(run_dir / "t_days.npy")
    
    return data

def run_simulation(config: Dict, theta_full: np.ndarray) -> tuple:
    """Run single simulation."""
    phi_init = config["phi_init"]
    if isinstance(phi_init, list):
        phi_init = np.array(phi_init)
        
    solver = BiofilmNewtonSolver5S(
        dt=config["dt"],
        maxtimestep=config["maxtimestep"],
        c_const=config["c_const"],
        alpha_const=config["alpha_const"],
        phi_init=phi_init
    )
    
    t_arr, g_arr = solver.solve(theta_full)
    
    phi = g_arr[:, 0:5]
    psi = g_arr[:, 6:11]
    phibar = phi * psi
    
    return t_arr, phibar

def plot_map_fit(
    t_arr: np.ndarray, 
    phibar_map: np.ndarray, 
    data: Dict[str, Any], 
    output_path: Path
):
    """Generate a publication-quality 5-panel MAP fit plot."""
    t_days = data["t_days"]
    data_points = data["data_points"]
    condition = data["config"].get("condition", "Commensal")
    current_colors = get_colors(condition)
    
    # Map simulation time to Days
    t_min, t_max = t_arr.min(), t_arr.max()
    day_min, day_max = t_days.min(), t_days.max()
    
    if t_max > t_min:
        t_plot = day_min + (t_arr - t_min) / (t_max - t_min) * (day_max - day_min)
    else:
        t_plot = t_arr
        
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharex=True, sharey=False)
    
    for i in range(5):
        ax = axes[i]
        species_name = SPECIES_NAMES[i]
        color = current_colors[i]
        
        # 1. Plot Simulation Line (MAP)
        ax.plot(t_plot, phibar_map[:, i], color=color, linewidth=3, label='MAP Model')
        
        # 2. Plot Experimental Data
        ax.scatter(t_days, data_points[:, i], color=color, s=100, edgecolor='black', zorder=10, label='Experiment')
        
        # Styling
        ax.set_title(species_name, fontsize=14, fontweight='bold', color=color)
        ax.set_xlabel("Time (Days)", fontsize=12)
        if i == 0:
            ax.set_ylabel("Abundance (φ̄)", fontsize=12)
            
        ax.set_xticks(t_days)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        if i == 0:
            ax.legend(loc='upper left', fontsize=10, frameon=True)

    plt.tight_layout()
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    logger.info(f"Generated MAP plot: {output_path.with_suffix('.png')}")

def plot_posterior_band(
    t_arr: np.ndarray,
    phibar_samples: np.ndarray,
    data: Dict[str, Any],
    output_path: Path
):
    """Generate posterior predictive band plot (single panel with all species)."""
    t_days = data["t_days"]
    data_points = data["data_points"]
    condition = data["config"].get("condition", "Commensal")
    current_colors = get_colors(condition)
    
    # Calculate quantiles
    q05 = np.nanpercentile(phibar_samples, 5, axis=0)
    q50 = np.nanpercentile(phibar_samples, 50, axis=0)
    q95 = np.nanpercentile(phibar_samples, 95, axis=0)
    
    # Map simulation time to Days
    t_min, t_max = t_arr.min(), t_arr.max()
    day_min, day_max = t_days.min(), t_days.max()
    
    if t_max > t_min:
        t_plot = day_min + (t_arr - t_min) / (t_max - t_min) * (day_max - day_min)
    else:
        t_plot = t_arr
        
    plt.figure(figsize=(10, 6))
    
    for i in range(5):
        color = current_colors[i]
        
        # Plot band
        plt.fill_between(t_plot, q05[:, i], q95[:, i], alpha=0.25, color=color, label=f"{SPECIES_NAMES[i]} 90% CI")
        # Plot median
        plt.plot(t_plot, q50[:, i], linewidth=2, color=color)
        
        # Plot data
        plt.scatter(t_days, data_points[:, i], s=40, edgecolor='black', facecolor=color, alpha=0.9, zorder=10)

    plt.xlabel("Time (Days)", fontsize=14)
    plt.ylabel("Abundance (φ̄)", fontsize=14)
    plt.title(f"Posterior Predictive Band - {condition}", fontsize=14)
    plt.xticks(t_days, fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Custom legend to avoid duplication if multiple elements share label
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=10, loc='upper left', bbox_to_anchor=(1.05, 1))
    
    plt.tight_layout()
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    logger.info(f"Generated Posterior Band plot: {output_path.with_suffix('.png')}")

def main():
    parser = argparse.ArgumentParser(description="Generate Nishioka-style publication plots")
    parser.add_argument("run_dirs", nargs='+', type=str, help="List of run directories")
    args = parser.parse_args()
    
    for run_dir_str in args.run_dirs:
        run_dir = Path(run_dir_str)
        if not run_dir.exists():
            logger.warning(f"Directory not found: {run_dir}")
            continue
            
        logger.info(f"Processing {run_dir}...")
        
        # Create output directory for figures if it doesn't exist
        output_dir = run_dir / "figures"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        data = load_run_data(run_dir)
        
        # 1. MAP Fit Plot
        if "theta_MAP" in data:
            logger.info("Running MAP simulation...")
            theta_full = np.array(data["theta_MAP"]["theta_full"])
            t_arr, phibar_map = run_simulation(data["config"], theta_full)
            
            plot_map_fit(t_arr, phibar_map, data, output_dir / "nishioka_MAP_Fit_with_data")
        else:
            logger.warning("No theta_MAP.json found, skipping MAP plot.")
            
        # 2. Posterior Band Plot
        if "samples" in data:
            logger.info("Running Posterior simulations...")
            samples = data["samples"]
            # Thin samples if too many (e.g. max 100 samples)
            n_samples = samples.shape[0]
            if n_samples > 100:
                indices = np.random.choice(n_samples, 100, replace=False)
                samples_subset = samples[indices]
            else:
                samples_subset = samples
                
            phibar_samples = []
            t_arr_ref = None
            
            for i, theta_sample in enumerate(samples_subset):
                t_arr, phibar = run_simulation(data["config"], theta_sample)
                phibar_samples.append(phibar)
                if t_arr_ref is None:
                    t_arr_ref = t_arr
                    
            phibar_samples = np.array(phibar_samples)
            
            plot_posterior_band(t_arr_ref, phibar_samples, data, output_dir / "nishioka_PosteriorBand")
        else:
            logger.warning("No samples.npy found, skipping Posterior Band plot.")

if __name__ == "__main__":
    main()
