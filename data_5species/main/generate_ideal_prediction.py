#!/usr/bin/env python3
"""
Generate an 'Ideal Outcome' visualization.
This script creates a mock-up of what the final simulation results 
SHOULD look like if the parameter estimation is 100% successful.
It uses interpolation of experimental data to simulate a 'Perfect Model Fit'.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.interpolate import make_interp_spline

# Configuration
DATA_DIR = Path("../experiment_data")
OUTPUT_DIR = Path("../experiment_fig")
CSV_FILE = DATA_DIR / "species_distribution_data.csv"
OUTPUT_FILE = OUTPUT_DIR / "Ideal_Outcome_Prediction.png"

# Colors matching the simulation script
COLORS_COMMENSAL = {
    'Blue': '#1f77b4',      # S0: S. oralis
    'Green': '#2ca02c',     # S1: A. naeslundii
    'Yellow': '#bcbd22',    # S2: V. dispar (Commensal)
    'Purple': '#9467bd',    # S3: F. nucleatum
    'Red': '#d62728'        # S4: P. gingivalis
}

COLORS_DYSBIOTIC = {
    'Blue': '#1f77b4',      # S0
    'Green': '#2ca02c',     # S1
    'Orange': '#ff7f0e',    # S2: V. parvula (Dysbiotic)
    'Purple': '#9467bd',    # S3
    'Red': '#d62728'        # S4
}

SPECIES_ORDER = ["Blue", "Green", "Yellow", "Orange", "Purple", "Red"]

def smooth_curve(x, y, x_new):
    """Generate a smooth curve using spline interpolation."""
    if len(x) < 2:
        return np.zeros_like(x_new)
    # k=2 for quadratic spline (prevents wild oscillations of k=3)
    try:
        spl = make_interp_spline(x, y, k=2) 
        y_smooth = spl(x_new)
        # Clip to valid range [0, 1]
        y_smooth = np.clip(y_smooth, 0.0, 1.0)
        return y_smooth
    except:
        return np.interp(x_new, x, y)

def plot_ideal_subplot(ax, df, condition, cultivation, title):
    """Plot a single condition simulating a perfect fit."""
    subset = df[(df['condition'] == condition) & (df['cultivation'] == cultivation)]
    
    # Choose color palette
    if condition == "Dysbiotic":
        palette = COLORS_DYSBIOTIC
    else:
        palette = COLORS_COMMENSAL
        
    # Time points for smooth curve
    t_smooth = np.linspace(1, 21, 100)
    
    # Plot each species
    for color_name in SPECIES_ORDER:
        if color_name not in palette:
            continue
            
        species_data = subset[subset['species'] == color_name].sort_values('day')
        
        if species_data.empty:
            continue
            
        days = species_data['day'].values
        medians = species_data['median'].values / 100.0 # Convert % to fraction
        
        # 1. Plot the "Ideal Model Prediction" (Smooth Line)
        # This represents the posterior mean of a perfect model
        y_smooth = smooth_curve(days, medians, t_smooth)
        
        c = palette[color_name]
        
        # Draw the "Prediction" line
        ax.plot(t_smooth, y_smooth, color=c, linewidth=2.5, alpha=0.9)
        
        # 2. Draw "Ideal Confidence Interval" (Shaded area)
        # Simulate a tight, confident uncertainty band (+/- 5%)
        y_upper = np.clip(y_smooth + 0.05, 0, 1)
        y_lower = np.clip(y_smooth - 0.05, 0, 1)
        ax.fill_between(t_smooth, y_lower, y_upper, color=c, alpha=0.15)
        
        # 3. Plot the "Experimental Data" (Dots)
        ax.scatter(days, medians, color=c, s=50, edgecolor='white', zorder=10)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Time (Days)")
    ax.set_ylabel("Abundance")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([1, 3, 6, 10, 15, 21])
    ax.grid(True, alpha=0.3)

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not CSV_FILE.exists():
        print("Data file not found.")
        return

    df = pd.read_csv(CSV_FILE)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Commensal Static
    plot_ideal_subplot(axes[0, 0], df, "Commensal", "Static", "Commensal / Static (Ideal)")
    
    # 2. Commensal HOBIC
    plot_ideal_subplot(axes[0, 1], df, "Commensal", "HOBIC", "Commensal / HOBIC (Ideal)")
    
    # 3. Dysbiotic Static
    plot_ideal_subplot(axes[1, 0], df, "Dysbiotic", "Static", "Dysbiotic / Static (Ideal)")
    
    # 4. Dysbiotic HOBIC
    plot_ideal_subplot(axes[1, 1], df, "Dysbiotic", "HOBIC", "Dysbiotic / HOBIC (Ideal)")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300)
    print(f"Generated Ideal Outcome Visualization: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
