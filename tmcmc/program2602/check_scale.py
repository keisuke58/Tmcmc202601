import pandas as pd
import numpy as np
from pathlib import Path

CSV_SOURCE_DIR = Path("/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species")
vol_file = CSV_SOURCE_DIR / "biofilm_boxplot_data.csv"
sp_file = CSV_SOURCE_DIR / "species_distribution_data.csv"

if not vol_file.exists() or not sp_file.exists():
    print("CSV files not found")
    exit(1)

df_vol = pd.read_csv(vol_file)
df_sp = pd.read_csv(sp_file)

# Clean and Rename
df_vol = df_vol[["condition", "cultivation", "day", "median"]].rename(
    columns={"median": "total_volume"}
)
df_sp = df_sp[["condition", "cultivation", "species", "day", "median"]].rename(
    columns={"median": "species_ratio"}
)

# Merge
df_merged = pd.merge(df_sp, df_vol, on=["condition", "cultivation", "day"], how="left")

# Calculate Absolute Volume
df_merged["absolute_volume"] = df_merged["total_volume"] * (df_merged["species_ratio"] / 100.0)

print("Absolute Volume Stats:")
print(df_merged["absolute_volume"].describe())

# Check per species
print("\nPer Species Max:")
print(df_merged.groupby("species")["absolute_volume"].max())

# Check conditions
print("\nConditions:")
print(df_merged["condition"].unique())

# Suggest Scale Factor
max_vol = df_merged["absolute_volume"].max()
target_sim_val = 30.0  # Assumption for gamma
scale_factor = target_sim_val / max_vol
print(f"\nMax Abs Vol: {max_vol}")
print(f"Target Sim Val: {target_sim_val}")
print(f"Suggested Scale Factor: {scale_factor}")
