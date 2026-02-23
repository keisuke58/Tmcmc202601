#!/usr/bin/env python3
"""
Generate publication-quality MAP estimate comparison plots.
This script focuses on visualizing the 'best fit' (MAP) parameter set against experimental data.
"""

import argparse
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List

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
    # Fallback for different directory structures
    sys.path.append(str(PROJECT_ROOT / "tmcmc" / "program2602"))
    from improved_5species_jit import BiofilmNewtonSolver5S

# Set seaborn style for publication
sns.set_context("paper", font_scale=1.5)
sns.set_style("whitegrid")

# Standard Socransky Colors
COLORS = [
    '#1f77b4', # S1: Blue (Health)
    '#2ca02c', # S2: Green (Early)
    '#bcbd22', # S3: Yellow (V. dispar) - Commensal Default
    '#9467bd', # S4: Purple (Bridge)
    '#d62728'  # S5: Red (Red Complex)
]

def get_colors(condition: str) -> List[str]:
    """Get color palette based on condition."""
    colors = list(COLORS)
    # If Dysbiotic, S3 is Orange (V. parvula)
    if "Dysbiotic" in condition:
        colors[2] = '#ff7f0e' # Orange
    return colors

SPECIES_NAMES = [
    "S. oralis", 
    "A. naeslundii", 
    "V. dispar", 
    "F. nucleatum", 
    "P. gingivalis"
]

def load_run_data(run_dir: Path) -> Dict[str, Any]:
    """Load necessary data for MAP simulation."""
    data = {}
    
    # Load JSONs
    with open(run_dir / "config.json", 'r') as f:
        data["config"] = json.load(f)
    
    with open(run_dir / "theta_MAP.json", 'r') as f:
        data["theta_MAP"] = json.load(f)
        
    # Load NPYs
    data["data_points"] = np.load(run_dir / "data.npy")
    data["t_days"] = np.load(run_dir / "t_days.npy")
    
    return data

def run_map_simulation(data: Dict[str, Any]) -> tuple:
    """Run simulation with MAP parameters."""
    config = data["config"]
    theta_full = np.array(data["theta_MAP"]["theta_full"])
    
    # Initial conditions
    phi_init = config["phi_init"]
    if isinstance(phi_init, list):
        phi_init = np.array(phi_init)
        
    Kp1 = config.get("Kp1", 1e-4)
    K_hill = config.get("K_hill", 0.0)
    n_hill = config.get("n_hill", 2.0)
    
    # Solver Setup
    solver = BiofilmNewtonSolver5S(
        dt=config["dt"],
        maxtimestep=config["maxtimestep"],
        c_const=config["c_const"],
        alpha_const=config["alpha_const"],
        phi_init=phi_init,
        Kp1=Kp1,
        K_hill=K_hill,
        n_hill=n_hill,
    )
    
    # Run simulation
    # solve returns t_arr, g_arr
    # g_arr: [phi1..5, phi0, psi1..5, gamma]
    t_arr, g_arr = solver.solve(theta_full)
    
    # Extract phibar = phi * psi
    phi = g_arr[:, 0:5]
    psi = g_arr[:, 6:11]
    phibar_map = phi * psi
    
    return t_arr, phibar_map

def save_plot_with_preview(fig, output_path):
    """Save plot in high res (300dpi) and preview (100dpi) formats."""
    # High resolution
    fig.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix('.pdf'), format='pdf', bbox_inches="tight")
    
    # Preview (max 2000px width)
    # 100 dpi is usually safe for typical figure sizes (up to 20 inches)
    fig.savefig(output_path.with_name(output_path.stem + "_preview.png"), dpi=100, bbox_inches="tight")
    print(f"Generated: {output_path.with_suffix('.png')} (+preview)")
    print(f"Generated: {output_path.with_suffix('.pdf')}")

def plot_map_fit(
    t_arr: np.ndarray, 
    phibar_map: np.ndarray, 
    data: Dict[str, Any], 
    output_path: Path
):
    """Generate a publication-quality 5-panel plot."""
    t_days = data["t_days"]
    data_points = data["data_points"]
    condition = data["config"].get("condition", "Commensal") # Default to Commensal if not found
    
    # Get colors based on condition
    current_colors = get_colors(condition)
    
    # Map simulation time to Days
    t_min, t_max = t_arr.min(), t_arr.max()
    day_min, day_max = t_days.min(), t_days.max()
    
    # Linear mapping of time
    if t_max > t_min:
        t_plot = day_min + (t_arr - t_min) / (t_max - t_min) * (day_max - day_min)
    else:
        t_plot = t_arr
        
    # Create figure: 1 row, 5 columns
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
        ax.set_title(species_name, fontsize=14, fontweight='bold')
        ax.set_xlabel("Time (Days)", fontsize=12)
        if i == 0:
            ax.set_ylabel("Abundance (φ̄)", fontsize=12)
            
        # Customize ticks
        ax.set_xticks(t_days)
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        # Grid
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Legend (only for first plot to save space, or inside)
        if i == 0:
            ax.legend(loc='upper left', fontsize=10, frameon=True)

    plt.tight_layout()
    save_plot_with_preview(fig, output_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate MAP fit comparison plot")
    parser.add_argument("run_dir", type=str, help="Path to run directory")
    parser.add_argument("output_dir", type=str, help="Path to output directory")
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from {run_dir}...")
    data = load_run_data(run_dir)
    
    print("Running MAP simulation...")
    t_arr, phibar_map = run_map_simulation(data)
    
    output_path = output_dir / "pub_map_fit_comparison"
    print("Generating plots...")
    plot_map_fit(t_arr, phibar_map, data, output_path)

if __name__ == "__main__":
    main()
