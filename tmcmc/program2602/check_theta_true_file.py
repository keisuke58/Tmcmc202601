import numpy as np
import os

data_dir = "data_5species"
theta_true_path = os.path.join(data_dir, "theta_true.npy")

if os.path.exists(theta_true_path):
    theta_true = np.load(theta_true_path)
    print("Loaded theta_true from file:")
    for i, val in enumerate(theta_true):
        print(f"Index {i}: {val}")
else:
    print(f"File not found: {theta_true_path}")
