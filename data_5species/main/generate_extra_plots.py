#!/usr/bin/env python3
"""
Generate additional plots for the 5-species model analysis.
1. Parameter Correlation Heatmap
2. Parameter Boxplots (Posterior Distributions)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

# Parameter Labels based on nishioka_algorithm_ja.tex
PARAM_LABELS = {
    0: "a11 (S.o self)",
    1: "a12 (S.o-A.n)",
    2: "a22 (A.n self)",
    3: "b1 (S.o)",
    4: "b2 (A.n)",
    5: "a33 (Vei self)",
    6: "a34 (Vei-F.n)",
    7: "a44 (F.n self)",
    8: "b3 (Vei)",
    9: "b4 (F.n)",
    10: "a13 (S.o-Vei)",
    11: "a14 (S.o-F.n)",
    12: "a23 (A.n-Vei)",
    13: "a24 (A.n-F.n)",
    14: "a55 (P.g self)",
    15: "b5 (P.g)",
    16: "a15 (S.o-P.g)",
    17: "a25 (A.n-P.g)",
    18: "a35 (Vei-P.g)",
    19: "a45 (F.n-P.g)"
}

SHORT_LABELS = [
    "a11", "a12", "a22", "b1", "b2",
    "a33", "a34", "a44", "b3", "b4",
    "a13", "a14", "a23", "a24",
    "a55", "b5", "a15", "a25", "a35", "a45"
]

def load_samples(run_dir: Path):
    samples_path = run_dir / "samples.npy"
    if not samples_path.exists():
        raise FileNotFoundError(f"samples.npy not found in {run_dir}")
    return np.load(samples_path)

def plot_correlation_heatmap(df, output_path):
    plt.figure(figsize=(14, 12))
    corr = df.corr()
    
    # Mask upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(
        corr, 
        mask=mask,
        cmap='coolwarm', 
        vmax=1, 
        vmin=-1, 
        center=0,
        annot=True, 
        fmt='.2f',
        square=True, 
        linewidths=.5, 
        cbar_kws={"shrink": .5}
    )
    plt.title('Parameter Correlation Matrix (Posterior)', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved correlation heatmap to {output_path}")
    plt.close()

def plot_boxplots(df, output_path):
    plt.figure(figsize=(16, 8))
    
    # Melt dataframe for seaborn boxplot
    df_melted = df.melt(var_name='Parameter', value_name='Value')
    
    sns.boxplot(x='Parameter', y='Value', data=df_melted, color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.title('Posterior Parameter Distributions', fontsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved boxplots to {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate extra plots")
    parser.add_argument("--run-dir", type=str, required=True)
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    output_dir = run_dir / "analysis"
    output_dir.mkdir(exist_ok=True)
    
    samples = load_samples(run_dir)
    
    # Create DataFrame
    # Use active indices logic if needed, but here we assume full theta (20 dims)
    # If samples are compressed, we might need to map them.
    # Usually samples.npy contains only active parameters? 
    # Let's check dimensions.
    
    n_samples, n_dims = samples.shape
    print(f"Loaded samples: {n_samples} samples, {n_dims} dimensions")
    
    if n_dims == 20:
        labels = SHORT_LABELS
    else:
        # Fallback if samples are subset
        # We need to know which indices are active.
        # But for now, let's assume 20.
        labels = [f"p{i}" for i in range(n_dims)]
        
    df = pd.DataFrame(samples, columns=labels)
    
    # Generate plots
    plot_correlation_heatmap(df, output_dir / "param_correlation_heatmap.png")
    plot_boxplots(df, output_dir / "param_boxplots.png")

if __name__ == "__main__":
    main()
