"""
Standalone build_replicate_sigma for testing (avoids evaluator.py import chain).
This is a copy of core.evaluator.build_replicate_sigma — keep in sync.
"""

import numpy as np


def build_replicate_sigma(
    replicate_csv: str,
    condition: str,
    cultivation: str,
    days: list,
    species_map: dict,
    n_species: int = 5,
    min_sigma: float = 0.05,
    rare_threshold: float = 0.05,
    max_info_ratio: float = 10.0,
) -> np.ndarray:
    import pandas as pd

    df = pd.read_csv(replicate_csv)
    mask = (df["condition"] == condition) & (df["cultivation"] == cultivation)
    dc = df[mask]

    n_t = len(days)
    sigma_mat = np.full((n_t, n_species), min_sigma, dtype=np.float64)

    for i, day in enumerate(days):
        for sp_name, sp_idx in species_map.items():
            vals = dc[(dc["species"] == sp_name) & (dc["day"] == day)]["distribution_pct"].values
            if len(vals) >= 3:
                iqr = np.percentile(vals, 75) - np.percentile(vals, 25)
                sigma_iqr = iqr / 1.35 / 100.0
                sigma_val = max(sigma_iqr, min_sigma)
                sigma_mat[i, sp_idx] = sigma_val

    # Cap information imbalance per timepoint
    if max_info_ratio > 0:
        for i in range(n_t):
            median_sig = np.median(sigma_mat[i, :])
            sigma_floor_info = median_sig / np.sqrt(max_info_ratio)
            sigma_mat[i, :] = np.maximum(sigma_mat[i, :], sigma_floor_info)

    return sigma_mat
