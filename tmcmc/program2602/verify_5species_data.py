import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


def verify_data():
    data_dir = "/home/nishioka/IKM_Hiwi/Tmcmc202601/tmcmc/data_5species"
    t_arr = np.load(os.path.join(data_dir, "t_arr.npy"))
    g_arr = np.load(os.path.join(data_dir, "g_arr.npy"))
    theta_true = np.load(os.path.join(data_dir, "theta_true.npy"))

    # g_arr shape: (time, 12)
    # 0-4: phi1-phi5
    # 5: phi0
    # 6-10: psi1-psi5
    # 11: gamma

    # 1. Conservation Law Check (Sum of Volume Fractions)
    # sum(phi_i) + phi_0 should be 1.0
    phi_sum = np.sum(g_arr[:, 0:5], axis=1) + g_arr[:, 5]
    error = phi_sum - 1.0

    plt.figure(figsize=(10, 6))
    plt.plot(t_arr, error, label="Sum($\phi_i$) - 1.0")
    plt.title("Conservation Law Check: $\sum \phi_i + \phi_0 = 1$", fontsize=14)
    plt.xlabel("Time [s]")
    plt.ylabel("Error")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "verify_conservation.png"), dpi=300)
    print("Saved verify_conservation.png")

    max_err = np.max(np.abs(error))
    print(f"Max conservation error: {max_err:.2e}")

    # 2. Pairwise Interactions (Phase Plane)
    # S1 vs S2, S3 vs S4, S1 vs S5, etc.
    plt.figure(figsize=(12, 10))

    pairs = [(0, 1, "S1 vs S2"), (2, 3, "S3 vs S4"), (0, 4, "S1 vs S5"), (2, 4, "S3 vs S5")]

    for idx, (i, j, title) in enumerate(pairs):
        plt.subplot(2, 2, idx + 1)
        plt.plot(g_arr[:, i], g_arr[:, j], "b-", linewidth=1.5)
        plt.plot(g_arr[0, i], g_arr[0, j], "go", label="Start")  # Start
        plt.plot(g_arr[-1, i], g_arr[-1, j], "ro", label="End")  # End
        plt.xlabel(f"Species {i+1} ($\phi_{i+1}$)")
        plt.ylabel(f"Species {j+1} ($\phi_{j+1}$)")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "verify_phase_planes.png"), dpi=300)
    print("Saved verify_phase_planes.png")

    # 3. Parameter Table Image
    # Create a nice table of parameters
    param_names = [
        "a11",
        "a12",
        "a22",
        "b1",
        "b2",  # M1
        "a33",
        "a34",
        "a44",
        "b3",
        "b4",  # M2
        "a13",
        "a14",
        "a23",
        "a24",  # M3
        "a55",
        "b5",  # M4
        "a15",
        "a25",
        "a35",
        "a45",  # M5
    ]

    # Categorize
    categories = (
        ["M1 (S1-S2)"] * 5
        + ["M2 (S3-S4)"] * 5
        + ["M3 (Cross)"] * 4
        + ["M4 (S5 Self)"] * 2
        + ["M5 (S5 Cross)"] * 4
    )

    df_params = pd.DataFrame(
        {"Parameter": param_names, "Value": theta_true, "Category": categories}
    )

    # Plot table
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.axis("tight")
    ax.axis("off")

    table_data = []
    for row in df_params.itertuples():
        table_data.append([row.Category, row.Parameter, f"{row.Value:.4f}"])

    table = ax.table(
        cellText=table_data,
        colLabels=["Category", "Parameter", "Value"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)

    plt.title("True Parameters for 5-Species Synthetic Data", fontsize=16, y=0.98)
    plt.savefig(os.path.join(data_dir, "verify_parameters_table.png"), dpi=300, bbox_inches="tight")
    print("Saved verify_parameters_table.png")

    # 4. Total Biomass Growth
    total_biomass = np.sum(g_arr[:, 0:5], axis=1)
    plt.figure(figsize=(10, 6))
    plt.plot(t_arr, total_biomass, "k-", linewidth=2, label="Total Biomass ($\sum \phi_{1..5}$)")
    plt.plot(t_arr, g_arr[:, 5], "b--", label="Solvent ($\phi_0$)")
    plt.title("Total Biomass vs Solvent", fontsize=14)
    plt.xlabel("Time [s]")
    plt.ylabel("Volume Fraction")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "verify_total_biomass.png"), dpi=300)
    print("Saved verify_total_biomass.png")

    # Check if biomass is growing
    growth = total_biomass[-1] - total_biomass[0]
    print(f"Total biomass change: {growth:.4f}")
    if growth <= 0:
        print("WARNING: Total biomass is not growing or decaying!")


if __name__ == "__main__":
    verify_data()
