import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_metrics(result_dir):
    csv_path = os.path.join(result_dir, 'dy_metrics.csv')
    if not os.path.exists(csv_path):
        return None
    return pd.read_csv(csv_path).iloc[0].to_dict()

def compare_models():
    # Define the models and their corresponding result directories
    models = {
        '1-Param Scaling': '_tmcmc_results_dy_k_st03_gpu3',
        '8-Param Multi': '_tmcmc_results_dy_multi_st02_gpu2',
        'Hybrid Ratkowsky': '_tmcmc_results_dy_ratkowsky_st01_gpu1'
    }
    
    base_dir = os.path.dirname(__file__)
    results = []
    
    for name, dir_name in models.items():
        full_dir = os.path.join(base_dir, dir_name)
        metrics = load_metrics(full_dir)
        if metrics:
            results.append({
                'Model': name,
                'RMSE': float(metrics['rmse_mean']),
                'MAE': float(metrics['mae_mean']),
                'R2': float(metrics['r2_mean']),
                'WRMSE': float(metrics['wrmse_mean']),
                'LogLikelihood': float(metrics['logL_map'])
            })
        else:
            print(f"Metrics not yet available for {name} ({dir_name})")
            
    if not results:
        print("No results found yet. Waiting for server computations to finish.")
        return
        
    df = pd.DataFrame(results)
    
    # Calculate AIC/BIC assuming N=24 observations (from raw data)
    N = 24
    k_params = {'1-Param Scaling': 1, '8-Param Multi': 8, 'Hybrid Ratkowsky': 5}
    
    df['k'] = df['Model'].map(k_params)
    df['AIC'] = 2 * df['k'] - 2 * df['LogLikelihood']
    df['BIC'] = np.log(N) * df['k'] - 2 * df['LogLikelihood']
    
    print("\n=== Dynamic Model Comparison ===")
    print(df[['Model', 'RMSE', 'R2', 'AIC', 'BIC']].to_string(index=False))
    
    # Save to CSV
    out_path = os.path.join(base_dir, 'dy_model_comparison.csv')
    df.to_csv(out_path, index=False)
    print(f"\nSaved comparison to {out_path}")
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot RMSE
    axes[0].bar(df['Model'], df['RMSE'], color=['#457b9d', '#1d3557', '#e63946'])
    axes[0].set_ylabel('RMSE (log10 CFU/g)')
    axes[0].set_title('Prediction Error (Lower is better)')
    axes[0].tick_params(axis='x', rotation=15)
    
    # Plot AIC
    axes[1].bar(df['Model'], df['AIC'], color=['#457b9d', '#1d3557', '#e63946'])
    axes[1].set_ylabel('AIC')
    axes[1].set_title('Akaike Information Criterion (Lower is better)')
    axes[1].tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    plot_path = os.path.join(base_dir, 'figures', 'dy_model_comparison.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300)
    print(f"Saved plot to {plot_path}")

if __name__ == '__main__':
    compare_models()
