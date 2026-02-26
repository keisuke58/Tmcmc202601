import numpy as np
import json
import os
from pathlib import Path

# Setup
RUN_DIR = Path(
    "/home/nishioka/IKM_Hiwi/Tmcmc202601/tmcmc/tmcmc/_runs/parallel_fixed_M1M2M3_20260126_210657"
)

# Ground Truth Values
TRUE_M1 = np.array([0.8, 2.0, 1.0, 0.1, 0.2])
TRUE_M2 = np.array([1.5, 1.0, 2.0, 0.3, 0.4])
TRUE_M3 = np.array([2.0, 1.0, 2.0, 1.0])

labels_M1 = ["a11", "a12", "a22", "b1", "b2"]
labels_M2 = ["a33", "a34", "a44", "b3", "b4"]
labels_M3 = ["a13", "a14", "a23", "a24"]


def load_map(model_name):
    filename = RUN_DIR / f"theta_MAP_{model_name}.json"
    if not filename.exists():
        return None
    with open(filename, "r") as f:
        data = json.load(f)
    if isinstance(data, dict) and "theta_sub" in data:
        return np.array(data["theta_sub"])
    elif isinstance(data, list):
        return np.array(data)
    else:
        return np.array(data)


est_M1 = load_map("M1")
est_M2 = load_map("M2")
est_M3 = load_map("M3")

print(f"{'Model':<6} | {'Param':<5} | {'True':<8} | {'Est':<8} | {'Rel Err(%)':<10}")
print("-" * 50)


def print_rows(model, labels, true_vals, est_vals):
    if est_vals is None:
        print(f"{model:<6} | Data not found")
        return

    for i, label in enumerate(labels):
        t = true_vals[i]
        e = est_vals[i]
        err = abs(t - e)
        rel_err = (err / t) * 100
        print(f"{model:<6} | {label:<5} | {t:<8.4f} | {e:<8.4f} | {rel_err:<10.2f}")


print_rows("M1", labels_M1, TRUE_M1, est_M1)
print("-" * 50)
print_rows("M2", labels_M2, TRUE_M2, est_M2)
print("-" * 50)
print_rows("M3", labels_M3, TRUE_M3, est_M3)
