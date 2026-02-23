
import os
import json
import glob
import pandas as pd
import numpy as np
from pathlib import Path

def analyze_sweeps(base_dir="Tmcmc202601/_sweeps"):
    results = []
    
    # Find all config.json files to identify sweep directories
    config_files = glob.glob(os.path.join(base_dir, "*", "config.json"))
    
    print(f"Found {len(config_files)} sweep directories.")
    
    for config_file in config_files:
        dir_path = os.path.dirname(config_file)
        dir_name = os.path.basename(dir_path)
        
        # Load config
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"Error loading config for {dir_name}: {e}")
            continue
            
        k_hill = config.get("K_hill", "N/A")
        n_hill = config.get("n_hill", "N/A")
        
        # Load metrics
        metrics_file = os.path.join(dir_path, "fit_metrics.json")
        if not os.path.exists(metrics_file):
            print(f"Metrics not found for {dir_name} (Run might be incomplete)")
            results.append({
                "directory": dir_name,
                "K_hill": k_hill,
                "n_hill": n_hill,
                "status": "Running/Failed",
                "rmse_mean": np.nan,
                "log_evidence": np.nan,
                "ess_min": np.nan,
                "rhat_max": np.nan
            })
            continue
            
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
        except Exception as e:
            print(f"Error loading metrics for {dir_name}: {e}")
            continue
            
        # Extract key metrics
        rmse = metrics.get("rmse_mean", np.nan)
        # Some versions might save log_evidence, others might not
        log_evidence = metrics.get("log_evidence", np.nan)
        
        # Diagnostics
        ess_min = metrics.get("diagnostics", {}).get("ess_min", np.nan)
        rhat_max = metrics.get("diagnostics", {}).get("rhat_max", np.nan)
        
        results.append({
            "directory": dir_name,
            "K_hill": k_hill,
            "n_hill": n_hill,
            "status": "Completed",
            "rmse_mean": rmse,
            "log_evidence": log_evidence,
            "ess_min": ess_min,
            "rhat_max": rhat_max
        })
    
    if not results:
        print("No results found.")
        return
        
    df = pd.DataFrame(results)
    
    # Sort by RMSE (lower is better)
    if "rmse_mean" in df.columns:
        df = df.sort_values("rmse_mean")
        
    print("\n" + "="*80)
    print("BRIDGE ORGANISM HYPOTHESIS - SWEEP RESULTS")
    print("="*80)
    
    # Format for display
    display_cols = ["directory", "K_hill", "n_hill", "status", "rmse_mean", "rhat_max", "ess_min"]
    if "log_evidence" in df.columns and not df["log_evidence"].isna().all():
        display_cols.append("log_evidence")
        
    print(df[display_cols].to_string(index=False))
    print("="*80)
    
    # Best run recommendation
    completed_runs = df[df["status"] == "Completed"]
    if not completed_runs.empty:
        best_run = completed_runs.iloc[0]
        print(f"\nBest run based on RMSE: {best_run['directory']}")
        print(f"Parameters: K={best_run['K_hill']}, n={best_run['n_hill']}")
        print(f"RMSE: {best_run['rmse_mean']:.6f}")
        
        if best_run['rhat_max'] > 1.1:
            print("WARNING: Max R-hat > 1.1. Convergence might be poor.")
            
if __name__ == "__main__":
    analyze_sweeps()
