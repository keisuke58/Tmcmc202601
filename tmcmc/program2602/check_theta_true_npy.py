
import numpy as np
from pathlib import Path

data_dir = Path("/home/nishioka/IKM_Hiwi/Tmcmc202601/tmcmc/data_5species")
try:
    theta_true = np.load(data_dir / "theta_true.npy")
    print("Loaded theta_true from npy:")
    print(theta_true)
    print("\nShape:", theta_true.shape)
    
    # Check against hardcoded values
    hardcoded = np.array([
        0.8, 2.0, 1.0, 0.1, 0.2,
        1.5, 1.0, 2.0, 0.3, 0.4,
        2.0, 1.0, 2.0, 1.0,
        1.2, 0.25,
        1.0, 1.0, 1.0, 1.0
    ])
    
    if np.allclose(theta_true, hardcoded):
        print("\nMatches hardcoded values!")
    else:
        print("\nDOES NOT MATCH hardcoded values!")
        print("Difference:", theta_true - hardcoded)
        
except Exception as e:
    print(f"Error loading theta_true.npy: {e}")

try:
    g_arr = np.load(data_dir / "g_arr.npy")
    print(f"\ng_arr exists. Shape: {g_arr.shape}")
except Exception as e:
    print(f"Error loading g_arr.npy: {e}")
