#!/usr/bin/env python3
"""
Generate a combined 2x2 summary plot for all 4 conditions:
Row 1: Commensal (Static, HOBIC)
Row 2: Dysbiotic (Static, HOBIC)

Usage:
    python generate_combined_summary.py \
        --cs /path/to/Commensal_Static_Run \
        --ch /path/to/Commensal_HOBIC_Run \
        --ds /path/to/Dysbiotic_Static_Run \
        --dh /path/to/Dysbiotic_HOBIC_Run \
        --output combined_summary.png
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

# Import data loader from estimate script
try:
    import estimate_commensal_static
except ImportError:
    logger.warning("Could not import estimate_commensal_static. Data loading fallback may fail.")

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

import pickle

def load_run_data(run_dir: Path) -> Dict[str, Any]:
    """Load necessary data for simulation."""
    if not run_dir.exists():
        logger.error(f"Run directory not found: {run_dir}")
        return None
        
    data = {}
    try:
        if (run_dir / "config.json").exists():
            with open(run_dir / "config.json", 'r') as f:
                data["config"] = json.load(f)
        else:
            # Try to load config from checkpoint metadata if config.json missing
            if (run_dir / "checkpoint_metadata.json").exists():
                 with open(run_dir / "checkpoint_metadata.json", 'r') as f:
                    data["config"] = json.load(f)
        
        if (run_dir / "theta_MAP.json").exists():
            with open(run_dir / "theta_MAP.json", 'r') as f:
                data["theta_MAP"] = json.load(f)
        
        # Priority 1: Finished samples
        if (run_dir / "samples.npy").exists():
            data["samples"] = np.load(run_dir / "samples.npy")
            logger.info(f"Loaded final samples from {run_dir}")
        # Priority 2: Checkpoint samples (for ongoing runs)
        elif (run_dir / "tmcmc_checkpoint.pkl").exists():
            try:
                with open(run_dir / "tmcmc_checkpoint.pkl", 'rb') as f:
                    ckpt = pickle.load(f)
                # Checkpoint chains are list of arrays, need to concatenate
                # ckpt["chains"] is a list of arrays (one per chain or stage?) 
                # In estimate_reduced_nishioka, chains is a list of arrays, usually we want the last stage samples
                # Actually in TMCMC, the "chains" at end of stage are the samples for that stage.
                # Let's assume ckpt["chains"] is a list of arrays, we stack them.
                chains = ckpt.get("chains", [])
                if isinstance(chains, list) and len(chains) > 0:
                    data["samples"] = np.concatenate(chains, axis=0)
                    logger.info(f"Loaded checkpoint samples (Stage {ckpt.get('stage')}) from {run_dir}")
                else:
                     logger.warning(f"Checkpoint found but no chains in {run_dir}")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint from {run_dir}: {e}")
            
        # Try loading data.npy, if not found, load from experimental data
        if (run_dir / "data.npy").exists() and (run_dir / "t_days.npy").exists():
            data["data_points"] = np.load(run_dir / "data.npy")
            data["t_days"] = np.load(run_dir / "t_days.npy")
        else:
            logger.info(f"data.npy not found in {run_dir}, loading from experimental source...")
            config = data["config"]
            condition = config.get("condition", "Commensal")
            cultivation = config.get("cultivation", "Static")
            start_day = config.get("start_from_day", 1)
            normalize = config.get("normalize_data", False)
            
            exp_data, t_days, _, phi_init_exp, _ = estimate_commensal_static.load_experimental_data(
                data_dir=DATA_5SPECIES_ROOT,
                condition=condition,
                cultivation=cultivation,
                start_from_day=start_day,
                normalize=normalize
            )
            data["data_points"] = exp_data
            data["t_days"] = t_days
            data["phi_init_exp"] = phi_init_exp

        return data
    except Exception as e:
        logger.error(f"Error loading data from {run_dir}: {e}")
        return None

def run_simulation(config: Dict, theta_full: np.ndarray, phi_init_override: Optional[np.ndarray] = None) -> tuple:
    """Run single simulation."""
    if phi_init_override is not None:
        phi_init = phi_init_override
    else:
        phi_init = config["phi_init"]
        if isinstance(phi_init, list):
            phi_init = np.array(phi_init)
        
    solver = BiofilmNewtonSolver5S(
        dt=config["dt"],
        maxtimestep=config["maxtimestep"],
        c_const=config["c_const"],
        alpha_const=config.get("alpha_const", 0.0),
        phi_init=phi_init
    )
    
    t_days = config.get("t_days_list", [1, 3, 6, 10, 15, 21])
    sim_steps = int(max(t_days) / config["dt"]) + 100
    
    t_arr, phi_res, _, _ = solver.solve(
        theta_full,
        steps=sim_steps
    )
    
    # Calculate relative abundance (phi_bar)
    # Avoid division by zero
    sum_phi = np.sum(phi_res, axis=1, keepdims=True)
    sum_phi[sum_phi < 1e-9] = 1.0
    phibar = phi_res / sum_phi
    
    return t_arr, phibar

def plot_subplot(ax, run_dir: Path, title: str, condition: str):
    """Plot a single condition on the given axes."""
    data = load_run_data(run_dir)
    if data is None:
        ax.text(0.5, 0.5, "Data Not Found", ha='center', va='center')
        return

    colors = get_colors(condition)
    t_days = data["t_days"]
    data_points = data["data_points"]
    
    # 1. Plot Posterior Band (if samples exist)
    if "samples" in data:
        samples = data["samples"]
        # Thin samples
        n_samples = samples.shape[0]
        if n_samples > 100:
            indices = np.random.choice(n_samples, 100, replace=False)
            samples_subset = samples[indices]
        else:
            samples_subset = samples
            
        phibar_samples = []
        t_arr_ref = None
        
        # Check for phi_init override
        phi_init_override = data.get("phi_init_exp", None)
        
        for theta_sample in samples_subset:
            t_arr, phibar = run_simulation(data["config"], theta_sample, phi_init_override=phi_init_override)
            phibar_samples.append(phibar)
            if t_arr_ref is None:
                t_arr_ref = t_arr
                
        phibar_samples = np.array(phibar_samples)
        
        # Calculate quantiles
        q05 = np.nanpercentile(phibar_samples, 5, axis=0)
        q50 = np.nanpercentile(phibar_samples, 50, axis=0)
        q95 = np.nanpercentile(phibar_samples, 95, axis=0)
        
        # Plot bands and median
        for i in range(5): # 5 species
            color = colors[i]
            # Convert time to days
            t_plot = t_arr_ref * data["config"]["dt"]
            
            # Limit plot range to max data day + 1
            max_day = max(t_days) + 1
            mask = t_plot <= max_day
            
            ax.fill_between(t_plot[mask], q05[mask, i], q95[mask, i], color=color, alpha=0.2)
            ax.plot(t_plot[mask], q50[mask, i], color=color, linewidth=2)

    # 2. Plot Data Points
    for i in range(5):
        color = colors[i]
        ax.scatter(t_days, data_points[:, i], s=40, edgecolor='black', facecolor=color, alpha=0.9, zorder=10)

    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel("Time (Days)", fontsize=12)
    ax.set_ylabel("Abundance (φ̄)", fontsize=12)
    ax.set_xticks(t_days)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

def main():
    parser = argparse.ArgumentParser(description="Generate Combined 4-Panel Summary Plot")
    parser.add_argument("--cs", type=str, required=True, help="Path to Commensal Static run")
    parser.add_argument("--ch", type=str, required=True, help="Path to Commensal HOBIC run")
    parser.add_argument("--ds", type=str, required=True, help="Path to Dysbiotic Static run")
    parser.add_argument("--dh", type=str, required=True, help="Path to Dysbiotic HOBIC run")
    parser.add_argument("--output", type=str, default="combined_summary.png", help="Output filename")
    
    args = parser.parse_args()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Row 1: Commensal
    plot_subplot(axes[0, 0], Path(args.cs), "Commensal / Static", "Commensal")
    plot_subplot(axes[0, 1], Path(args.ch), "Commensal / HOBIC", "Commensal")
    
    # Row 2: Dysbiotic
    plot_subplot(axes[1, 0], Path(args.ds), "Dysbiotic / Static", "Dysbiotic")
    plot_subplot(axes[1, 1], Path(args.dh), "Dysbiotic / HOBIC", "Dysbiotic")
    
    # Add shared legend
    # Create dummy lines for legend
    legend_elements = []
    for i, name in enumerate(SPECIES_NAMES):
        # Use Commensal colors for legend (Green/Blue/Purple/Red are same, only S2 differs)
        # We can note the S2 difference
        color = COLORS_COMMENSAL[i]
        label = name
        if i == 2: # V. dispar
            label = f"{name} (Yel/Org)"
        
        line = plt.Line2D([0], [0], color=color, lw=4, label=label)
        legend_elements.append(line)
        
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=5, fontsize=14)
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    logger.info(f"Generated combined plot: {args.output}")

if __name__ == "__main__":
    main()
