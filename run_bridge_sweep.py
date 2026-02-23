#!/usr/bin/env python3
import subprocess
import json
import pandas as pd
from pathlib import Path
import sys

def run_sweep():
    # Parameters to sweep
    K_values = [0.05]
    n_values = [4.0]

    # Fixed parameters — mild-weight test run
    n_particles = 150
    n_stages = 8
    c_const = 25.0
    kp1 = 1e-5
    condition = "Dysbiotic"
    cultivation = "HOBIC"
    n_jobs = 4  # Run multiple instances in parallel, so reduce jobs per instance
    
    processes = []
    
    print(f"Starting parameter sweep for {condition} {cultivation}...")
    print(f"K_hill: {K_values}")
    print(f"n_hill: {n_values}")
    
    # Launch all runs
    for K in K_values:
        for n in n_values:
            run_name = f"K{K}_n{n}"
            output_dir = Path(f"_sweeps/{run_name}")
            
            print(f"Launching {run_name} (K={K}, n={n})...")
            
            cmd = [
                "python", "data_5species/main/estimate_reduced_nishioka.py",
                "--condition", condition,
                "--cultivation", cultivation,
                "--n-particles", str(n_particles),
                "--n-stages", str(n_stages),
                "--K-hill", str(K),
                "--n-hill", str(n),
                "--lambda-pg", "2.0",
                "--lambda-late", "1.5",
                "--n-late", "2",
                "--c-const", str(c_const),
                "--kp1", str(kp1),
                "--n-jobs", str(n_jobs),
                "--output-dir", str(output_dir),
                "--no-notify-slack"
            ]
            
            # Use Popen for parallel execution
            p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            processes.append((run_name, p, output_dir, K, n))

    print(f"\nLaunched {len(processes)} processes. Waiting for completion...")
    
    # Wait for completion and collect results
    results = []
    for run_name, p, output_dir, K, n in processes:
        stdout, stderr = p.communicate()
        
        if p.returncode == 0:
            print(f"✓ {run_name} finished successfully")
            
            # Load metrics
            metrics_file = output_dir / "fit_metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                
                res = {
                    "K_hill": K,
                    "n_hill": n,
                    "logL": metrics.get("logL_max", float('nan')),
                    "rmse_total": metrics.get("rmse_total", float('nan')),
                }
                
                if "rmse_per_species" in metrics:
                     if isinstance(metrics["rmse_per_species"], list):
                         if len(metrics["rmse_per_species"]) > 4:
                             res["rmse_s4"] = metrics["rmse_per_species"][4]
                     elif isinstance(metrics["rmse_per_species"], dict):
                         res["rmse_s4"] = metrics["rmse_per_species"].get("Species 4", float('nan'))
                
                results.append(res)
                print(f"  -> LogL: {res['logL']:.2f}, RMSE: {res['rmse_total']:.4f}")
            else:
                print(f"  -> Failed to find metrics file at {metrics_file}")
        else:
            print(f"✗ {run_name} failed with exit code {p.returncode}")
            if stderr:
                print(f"  Error: {stderr.decode('utf-8')[:500]}...")

    # Save summary

    # Save summary
    if results:
        df = pd.DataFrame(results)
        print("\nSweep Results Summary:")
        print(df.sort_values("rmse_total"))
        
        df.to_csv("_sweeps/summary.csv", index=False)
        print("\nSummary saved to _sweeps/summary.csv")
    else:
        print("\nNo results collected.")

if __name__ == "__main__":
    run_sweep()
