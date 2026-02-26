import numpy as np
from pathlib import Path

base_dir = Path("/home/nishioka/IKM_Hiwi/Tmcmc202601/tmcmc")
data_path = base_dir / "_runs/manual_M2_high_precision/data.npy"
g_arr_path = base_dir / "data_5species/g_arr.npy"

if data_path.exists():
    data = np.load(data_path)
    print("Data.npy first row (t=0):")
    print(data[0])
    print("Data.npy shape:", data.shape)
else:
    print("Data.npy not found")

if g_arr_path.exists():
    g_arr = np.load(g_arr_path)
    print("g_arr.npy first row (t=0):")
    print(g_arr[0])
    # Calculate phi * psi for t=0
    phi = g_arr[0, 0:5]
    psi = g_arr[0, 6:11]
    print("phi * psi at t=0:")
    print(phi * psi)
else:
    print("g_arr.npy not found")
