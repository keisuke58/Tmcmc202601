import json
import numpy as np
from pathlib import Path
from scipy.stats import gaussian_kde
import time

# Configuration
RUN_A_DIR = Path("/home/nishioka/IKM_Hiwi/Tmcmc202601/tmcmc/tmcmc/_runs/parallel_fixed_M1M2M3_20260126_210657_MEAN")
OUTPUT_DIR = Path("/home/nishioka/IKM_Hiwi/Tmcmc202601/tmcmc/tmcmc/_runs/parallel_fixed_M1M2M3_20260126_210657")
TRACE_FILE = RUN_A_DIR / "trace_M2.npy"

def main():
    print(f"Checking for {TRACE_FILE}...")
    if not TRACE_FILE.exists():
        print("trace_M2.npy not found yet. M2 might still be running.")
        return

    print(f"Loading samples from {TRACE_FILE}...")
    samples = np.load(TRACE_FILE)
    print(f"Samples shape: {samples.shape}")
    
    # 1. Compute Mean
    mean_vec = np.mean(samples, axis=0)
    print(f"Computed Mean: {mean_vec}")
    
    # 2. Compute MAP (using KDE on samples)
    print("Computing MAP using KDE...")
    try:
        kde = gaussian_kde(samples.T)
        # Evaluate density on the samples themselves to find the highest density point
        # This is a robust approximation for the mode
        log_densities = kde.logpdf(samples.T)
        idx_map = np.argmax(log_densities)
        map_vec = samples[idx_map]
        print(f"Computed MAP (KDE peak): {map_vec}")
    except Exception as e:
        print(f"KDE computation failed: {e}. Falling back to Mean for MAP.")
        map_vec = mean_vec

    # Save JSONs
    # MAP
    map_data = {
        "model": "M2",
        "theta_sub": map_vec.tolist(),
        "note": "Extracted from trace_M2.npy using KDE"
    }
    with open(OUTPUT_DIR / "theta_MAP_M2.json", "w") as f:
        json.dump(map_data, f, indent=4)
    print(f"Saved {OUTPUT_DIR / 'theta_MAP_M2.json'}")

    # MEAN
    mean_data = {
        "model": "M2",
        "theta_sub": mean_vec.tolist(),
        "note": "Extracted from trace_M2.npy (Mean)"
    }
    with open(OUTPUT_DIR / "theta_MEAN_M2.json", "w") as f:
        json.dump(mean_data, f, indent=4)
    print(f"Saved {OUTPUT_DIR / 'theta_MEAN_M2.json'}")

if __name__ == "__main__":
    main()
