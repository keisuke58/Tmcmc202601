#!/usr/bin/env python3
"""
Update MAP estimate comparison plots with specific requirements:
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
    sys.path.append(str(PROJECT_ROOT / "tmcmc" / "program2602"))
    from improved_5species_jit import BiofilmNewtonSolver5S

# Set seaborn style
sns.set_context("paper", font_scale=1.5)
sns.set_style("whitegrid")

# Color mapping based on SPECIES_MAP and SPECIES_MAP_COMMENSAL
COLORS_COMMENSAL = [
    '#1f77b4', # S0: Blue
    '#2ca02c', # S1: Green
    '#bcbd22', # S2: Yellow (using 'tab:olive' for better visibility on white)
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
    """Load necessary data for MAP simulation."""
    data = {}
    
    with open(run_dir / "config.json", 'r') as f:
        data["config"] = json.load(f)
    
    with open(run_dir / "theta_MAP.json", 'r') as f:
        data["theta_MAP"] = json.load(f)
        
    data["data_points"] = np.load(run_dir / "data.npy")
    data["t_days"] = np.load(run_dir / "t_days.npy")
    
    return data

def run_map_simulation(data: Dict[str, Any]) -> tuple:
    """Run simulation with MAP parameters."""
    config = data["config"]
    theta_full = np.array(data["theta_MAP"]["theta_full"])
    
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
    phibar_map = phi * psi
    
    return t_arr, phibar_map

def plot_map_fit(
    t_arr: np.ndarray, 
    phibar_map: np.ndarray, 
    data: Dict[str, Any], 
    output_path: Path
):
    """Generate a publication-quality 5-panel plot with shared axes."""
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
        
    # Share Y axis across all subplots
    fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharex=True, sharey=True)
    
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
        
        # Only set Y-label for the first plot
        if i == 0:
            ax.set_ylabel("Relative Abundance", fontsize=12)
            
        # Customize ticks
        ax.set_xticks(t_days)
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        # Grid
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Set X-axis limit to start at 0
        ax.set_xlim(left=0)
        
        # Set Y-axis limit to start at 0
        ax.set_ylim(bottom=0)
        
        # Legend (only for first plot)
        if i == 0:
            ax.legend(loc='upper right', fontsize=10, frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Generated: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate updated MAP fit comparison plot")
    parser.add_argument("run_dir", type=str, help="Path to run directory")
    parser.add_argument("output_file", type=str, help="Full path to output file")
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from {run_dir}...")
    data = load_run_data(run_dir)
    
    print("Running MAP simulation...")
    t_arr, phibar_map = run_map_simulation(data)
    
    print(f"Generating plots to {output_path}...")
    plot_map_fit(t_arr, phibar_map, data, output_path)

if __name__ == "__main__":
    main()
