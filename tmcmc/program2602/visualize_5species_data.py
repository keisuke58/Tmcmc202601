
import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_data():
    data_dir = "/home/nishioka/IKM_Hiwi/Tmcmc202601/tmcmc/data_5species"
    t_arr = np.load(os.path.join(data_dir, "t_arr.npy"))
    g_arr = np.load(os.path.join(data_dir, "g_arr.npy"))
    theta_true = np.load(os.path.join(data_dir, "theta_true.npy"))

    # g_arr shape: (time, 12)
    # 0-4: phi1-phi5
    # 5: phi0
    # 6-10: psi1-psi5
    # 11: gamma

    plt.figure(figsize=(12, 10))

    # Plot 1: Volume Fractions (Phi)
    plt.subplot(2, 1, 1)
    colors = ['b', 'g', 'r', 'c', 'm']
    labels = [f'Species {i+1} ($\phi_{i+1}$)' for i in range(5)]
    
    for i in range(5):
        plt.plot(t_arr, g_arr[:, i], label=labels[i], color=colors[i], linewidth=2)
    
    plt.plot(t_arr, g_arr[:, 5], label='Solvent ($\phi_0$)', color='k', linestyle='--', alpha=0.7)
    
    plt.title("Volume Fractions ($\phi$) Dynamics - 5 Species Model", fontsize=14)
    plt.ylabel("Volume Fraction [-]", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='center right')

    # Plot 2: Nutrients (Psi) & Gamma
    plt.subplot(2, 1, 2)
    for i in range(5):
        plt.plot(t_arr, g_arr[:, 6+i], label=f'Nutrient {i+1} ($\psi_{i+1}$)', color=colors[i], linestyle=':', linewidth=2)
    
    # Scale gamma for visibility if needed, but usually it's small or comparable.
    plt.plot(t_arr, g_arr[:, 11], label='Gamma ($\gamma$)', color='gray', linestyle='-.')

    plt.title("Nutrient ($\psi$) and Growth Potential ($\gamma$) Dynamics", fontsize=14)
    plt.xlabel("Time [s]", fontsize=12)
    plt.ylabel("Concentration / Value", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='center right')

    plt.tight_layout()
    output_path = os.path.join(data_dir, "5species_dynamics.png")
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

    # Create Markdown Info
    md_path = os.path.join(data_dir, "README.md")
    
    md_content = f"""# 5-Species Model Synthetic Data

## Overview
This dataset was synthetically generated to compensate for the missing 5th species data in the original paper.
It uses an extended version of the continuum biofilm model (`improved_5species_jit.py`) with hypothetical but plausible parameters for the 5th species.

## Files
- `t_arr.npy`: Time steps array (Shape: {t_arr.shape})
- `g_arr.npy`: State vector array (Shape: {g_arr.shape})
  - Indices 0-4: $\phi_1$ to $\phi_5$ (Volume fractions)
  - Index 5: $\phi_0$ (Solvent)
  - Indices 6-10: $\psi_1$ to $\psi_5$ (Nutrients)
  - Index 11: $\gamma$ (Growth potential)
- `theta_true.npy`: True parameters used for generation (Shape: {theta_true.shape})
- `5species_dynamics.png`: Visualization of the dynamics

## Parameters Used (True Theta)
The parameter vector $\\theta$ consists of 20 elements:

| Index | Name | Value | Block | Description |
|---|---|---|---|---|
| 0 | a11 | {theta_true[0]:.2f} | M1 | S1 Self-interaction |
| 1 | a12 | {theta_true[1]:.2f} | M1 | S1-S2 Interaction |
| 2 | a22 | {theta_true[2]:.2f} | M1 | S2 Self-interaction |
| 3 | b1 | {theta_true[3]:.2f} | M1 | S1 Decay |
| 4 | b2 | {theta_true[4]:.2f} | M1 | S2 Decay |
| 5 | a33 | {theta_true[5]:.2f} | M2 | S3 Self-interaction |
| 6 | a34 | {theta_true[6]:.2f} | M2 | S3-S4 Interaction |
| 7 | a44 | {theta_true[7]:.2f} | M2 | S4 Self-interaction |
| 8 | b3 | {theta_true[8]:.2f} | M2 | S3 Decay |
| 9 | b4 | {theta_true[9]:.2f} | M2 | S4 Decay |
| 10 | a13 | {theta_true[10]:.2f} | M3 | S1-S3 Interaction |
| 11 | a14 | {theta_true[11]:.2f} | M3 | S1-S4 Interaction |
| 12 | a23 | {theta_true[12]:.2f} | M3 | S2-S3 Interaction |
| 13 | a24 | {theta_true[13]:.2f} | M3 | S2-S4 Interaction |
| 14 | a55 | {theta_true[14]:.2f} | M4 | S5 Self-interaction (**Hypothetical**) |
| 15 | b5 | {theta_true[15]:.2f} | M4 | S5 Decay (**Hypothetical**) |
| 16 | a15 | {theta_true[16]:.2f} | M5 | S1-S5 Interaction (**Hypothetical**) |
| 17 | a25 | {theta_true[17]:.2f} | M5 | S2-S5 Interaction (**Hypothetical**) |
| 18 | a35 | {theta_true[18]:.2f} | M5 | S3-S5 Interaction (**Hypothetical**) |
| 19 | a45 | {theta_true[19]:.2f} | M5 | S4-S5 Interaction (**Hypothetical**) |

## Simulation Details
- **Time steps**: {len(t_arr)} (dt=1e-5)
- **Initial Condition**:
  - $\phi_i = 0.2$ for all active species
  - $\psi_i = 0.999$
  - $\gamma = 0.0$
- **Solver**: Newton-Raphson with Numba JIT acceleration (`improved_5species_jit.py`)
"""
    
    with open(md_path, "w") as f:
        f.write(md_content)
    print(f"Markdown info saved to {md_path}")

if __name__ == "__main__":
    visualize_data()
