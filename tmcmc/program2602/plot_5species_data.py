
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_data():
    # data_dir = "data_5species"
    data_dir = os.path.join(os.path.dirname(__file__), "data_5species")
    t_arr = np.load(os.path.join(data_dir, "t_arr.npy"))
    g_arr = np.load(os.path.join(data_dir, "g_arr.npy"))
    
    # Normalize time to [0, 1]
    t_norm = t_arr / t_arr[-1]
    
    # g_arr shape: (T, 12)
    # 0-4: Phi1-5
    # 5: Phi0
    # 6-10: Psi1-5
    # 11: Gamma
    
    plt.figure(figsize=(10, 6))
    colors = ['b', 'g', 'r', 'c', 'm']
    labels = [f'Species {i+1}' for i in range(5)]
    
    # Plot Phi 1-5 only (Phi0 removed)
    for i in range(5):
        plt.plot(t_norm, g_arr[:, i], label=labels[i], color=colors[i], linewidth=2)
    
    # Removed Phi0 plot
    # plt.plot(t_arr, g_arr[:, 5], label='Phi0', color='k', linestyle='--', linewidth=2)
    
    plt.xlabel('Normalized Time')
    plt.ylabel('Volume Fraction')
    plt.title('5-Species Biofilm Dynamics')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust y-limit to focus on species growth (since Phi0 is gone, max is likely < 1.0)
    # But let's keep it somewhat standard or auto.
    # User said "make it easier to see". Since sum(Phi1..5) is small initially (0.1) and grows,
    # auto-scaling or 0-1 is fine. Let's try 0 to max value + margin.
    max_val = np.max(g_arr[:, 0:5])
    plt.ylim(0, max_val * 1.2)
    
    output_path = os.path.join(data_dir, "5species_dynamics_normalized.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Plot saved to {output_path}")

    # --- New Plot: With Phi0 ---
    plt.figure(figsize=(10, 6))
    
    # Plot Phi 1-5
    for i in range(5):
        plt.plot(t_norm, g_arr[:, i], label=labels[i], color=colors[i], linewidth=2)
    
    # Plot Phi0
    plt.plot(t_norm, g_arr[:, 5], label='Phi0', color='k', linestyle='--', linewidth=2)
    
    plt.xlabel('Normalized Time')
    plt.ylabel('Volume Fraction')
    plt.title('5-Species Biofilm Dynamics (with Phi0)')
    plt.legend(loc='upper right') # Phi0 is usually high, so legend might be better elsewhere
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 1.05) # Full range
    
    output_path_phi0 = os.path.join(data_dir, "5species_dynamics_with_phi0.png")
    plt.savefig(output_path_phi0, dpi=300)
    plt.close()
    print(f"Plot saved to {output_path_phi0}")
    # ---------------------------
    
    # Check initial values
    print("Initial values (t=0):")
    for i in range(5):
        print(f"Phi{i+1}: {g_arr[0, i]:.4f}")

if __name__ == "__main__":
    plot_data()
