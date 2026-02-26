import numpy as np
import pandas as pd
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = Path(__file__).parent.parent / "experiment_data"
OUTPUT_DIR = Path(__file__).parent.parent / "processed_data"

SPECIES_MAP_GENERAL = {
    "Blue": 0,  # S. oralis
    "Green": 1,  # A. naeslundii
    "Yellow": 2,  # V. dispar
    "Orange": 2,  # V. parvula (Dysbiotic strain of Veillonella)
    "Purple": 3,  # F. nucleatum
    "Red": 4,  # P. gingivalis
}

SPECIES_NAMES = ["S.oralis", "A.naeslundii", "Veillonella", "F.nucleatum", "P.gingivalis"]


def load_and_process_data(condition, cultivation):
    logger.info(f"Processing {condition} {cultivation}...")

    # 1. Load Total Volume (Boxplot data)
    boxplot_file = DATA_DIR / f"boxplot_{condition}_{cultivation}.csv"
    if not boxplot_file.exists():
        # Fallback
        boxplot_file = DATA_DIR / "biofilm_boxplot_data.csv"

    if not boxplot_file.exists():
        logger.error(f"Boxplot data not found for {condition} {cultivation}")
        return None

    df_box = pd.read_csv(boxplot_file)
    if "condition" in df_box.columns:
        df_box = df_box[(df_box["condition"] == condition) & (df_box["cultivation"] == cultivation)]

    # 2. Load Species Distribution
    species_file = DATA_DIR / "species_distribution_data.csv"
    if not species_file.exists():
        logger.error("Species distribution data not found")
        return None

    df_species = pd.read_csv(species_file)
    df_species = df_species[
        (df_species["condition"] == condition) & (df_species["cultivation"] == cultivation)
    ]

    # 3. Process Data
    days = sorted(df_box["day"].unique())
    n_timepoints = len(days)
    n_species = 5

    data_abs = np.zeros((n_timepoints, n_species))
    data_norm = np.zeros((n_timepoints, n_species))
    total_volumes = np.zeros(n_timepoints)

    # Header for CSV
    cols = ["Day"] + [f"Vol_{name}" for name in SPECIES_NAMES]
    cols_norm = ["Day"] + [f"Frac_{name}" for name in SPECIES_NAMES]

    records_abs = []
    records_norm = []

    for i, day in enumerate(days):
        # Total Volume
        day_vol_data = df_box[df_box["day"] == day]
        if len(day_vol_data) == 0:
            continue

        total_vol = day_vol_data["median"].values[0]
        total_volumes[i] = total_vol

        # Species Ratios
        day_species_data = df_species[df_species["day"] == day]

        row_abs = [day] + [0.0] * 5
        row_norm = [day] + [0.0] * 5

        current_sum_frac = 0.0

        for _, row in day_species_data.iterrows():
            sp_color = row["species"]
            if sp_color in SPECIES_MAP_GENERAL:
                idx = SPECIES_MAP_GENERAL[sp_color]
                percentage = row["median"]
                fraction = percentage / 100.0

                # Absolute Volume = Total * Fraction
                abs_vol = total_vol * fraction

                data_abs[i, idx] = abs_vol
                row_abs[idx + 1] = abs_vol

                # Store raw fraction for normalization later
                data_norm[i, idx] = fraction
                current_sum_frac += fraction

        # Normalize fractions to sum to 1.0 (Plan A)
        if current_sum_frac > 0:
            data_norm[i, :] /= current_sum_frac
            for idx in range(5):
                row_norm[idx + 1] = data_norm[i, idx]

        records_abs.append(row_abs)
        records_norm.append(row_norm)

    # Convert to DataFrames
    df_out_abs = pd.DataFrame(records_abs, columns=cols)
    df_out_norm = pd.DataFrame(records_norm, columns=cols_norm)

    return df_out_abs, df_out_norm


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    conditions = [
        ("Commensal", "Static"),
        ("Commensal", "HOBIC"),
        ("Dysbiotic", "Static"),
        ("Dysbiotic", "HOBIC"),
    ]

    for cond, cult in conditions:
        result = load_and_process_data(cond, cult)
        if result:
            df_abs, df_norm = result

            # Save Plan B (Absolute)
            file_abs = OUTPUT_DIR / f"target_data_{cond}_{cult}_absolute.csv"
            df_abs.to_csv(file_abs, index=False)
            logger.info(f"Saved Plan B (Absolute): {file_abs}")

            # Save Plan A (Normalized)
            file_norm = OUTPUT_DIR / f"target_data_{cond}_{cult}_normalized.csv"
            df_norm.to_csv(file_norm, index=False)
            logger.info(f"Saved Plan A (Normalized): {file_norm}")


if __name__ == "__main__":
    main()
