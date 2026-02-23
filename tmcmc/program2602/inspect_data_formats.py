
import numpy as np
import pandas as pd
import os

# Paths
npy_path = "tmcmc/data_5species/g_arr.npy"
csv_vol_path = "data_5species/biofilm_boxplot_data.csv"
csv_sp_path = "data_5species/species_distribution_data.csv"

print("--- TMCMC Synthetic Data (g_arr.npy) ---")
if os.path.exists(npy_path):
    g_arr = np.load(npy_path)
    print(f"Shape: {g_arr.shape}")
    # g = [phi1..5, phi0, psi1..5, gamma]
    
    phi = g_arr[:, 0:5]
    psi = g_arr[:, 6:11]
    gamma = g_arr[:, 11]
    
    print(f"Gamma (Thickness/Volume) Range: {gamma.min():.4f} - {gamma.max():.4f}")
    print(f"Phi (Species Ratio) Range:      {phi.min():.4f} - {phi.max():.4f}")
    print(f"Psi (Potential?) Range:         {psi.min():.4f} - {psi.max():.4f}")
    
    # Calculate potential "Absolute Volume" candidates
    abs_vol_gamma = phi * gamma[:, np.newaxis]
    abs_vol_psi = phi * psi
    
    print(f"Phi * Gamma Range:              {abs_vol_gamma.min():.4f} - {abs_vol_gamma.max():.4f}")
    print(f"Phi * Psi Range:                {abs_vol_psi.min():.4f} - {abs_vol_psi.max():.4f}")
else:
    print(f"File not found: {npy_path}")

print("\n--- Experimental Data (CSVs) ---")
if os.path.exists(csv_vol_path):
    df_vol = pd.read_csv(csv_vol_path)
    print("Total Volume Data:")
    print(df_vol['median'].describe())
else:
    print(f"File not found: {csv_vol_path}")

if os.path.exists(csv_sp_path):
    df_sp = pd.read_csv(csv_sp_path)
    print("\nSpecies Ratio Data (%):")
    print(df_sp['median'].describe())
    
    # Check calculated absolute volume
    # Note: notebook did: total_volume * (species_ratio / 100)
    # We can approximate the range
    avg_total_vol = df_vol['median'].mean()
    avg_ratio = df_sp['median'].mean()
    print(f"\nApprox Absolute Volume (Mean): {avg_total_vol * avg_ratio / 100:.4f}")

