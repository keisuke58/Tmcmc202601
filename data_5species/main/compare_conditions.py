#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_conditions.py

Compares two TMCMC runs (typically Commensal vs Dysbiotic) to visualize the "Surge".
Generates overlay plots for key species (S5 P. gingivalis, S4 F. nucleatum).

Usage:
    python3 data_5species/main/compare_conditions.py --commensal _runs/RunA --dysbiotic _runs/RunB
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from tmcmc.program2602.improved_5species_jit import BiofilmNewtonSolver5S
except ImportError:
    sys.path.insert(0, str(project_root / "tmcmc" / "program2602"))
    from improved_5species_jit import BiofilmNewtonSolver5S


def parse_args():
    parser = argparse.ArgumentParser(description="Compare Commensal vs Dysbiotic runs")
    parser.add_argument(
        "--commensal", type=str, required=True, help="Path to Commensal run directory"
    )
    parser.add_argument(
        "--dysbiotic", type=str, required=True, help="Path to Dysbiotic run directory"
    )
    parser.add_argument(
        "--output-dir", type=str, default="comparison_plots", help="Output directory"
    )
    return parser.parse_args()


def load_run_data(run_dir):
    path = Path(run_dir)
    map_file = path / "theta_MAP.json"
    if not map_file.exists():
        raise FileNotFoundError(f"theta_MAP.json not found in {run_dir}")

    with open(map_file, "r") as f:
        map_data = json.load(f)

    return map_data


def simulate_run(map_data, label):
    # Setup solver (defaults)
    solver = BiofilmNewtonSolver5S(dt=0.01, maxtimestep=2500, phi_init=0.01, use_numba=True)

    theta_full = np.array(map_data["theta_full"])
    t_arr, g_arr = solver.solve(theta_full)

    # Calculate relative abundance
    phi_abs = g_arr[:, 0:5]
    phi_total = np.sum(phi_abs, axis=1)
    phi_total[phi_total < 1e-9] = 1e-9
    phi_rel = phi_abs / phi_total[:, None]

    return t_arr, phi_rel


def plot_comparison(t_c, phi_c, t_d, phi_d, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    species_names = ["S. oralis", "A. naeslundii", "V. dispar", "F. nucleatum", "P. gingivalis"]
    colors = ["blue", "cyan", "green", "purple", "red"]

    # 1. Target Species (S5: Red) Comparison
    plt.figure(figsize=(8, 6))

    # S5 Commensal
    plt.plot(t_c, phi_c[:, 4], label="Commensal (S5)", color="red", linestyle="--", alpha=0.7)
    # S5 Dysbiotic
    plt.plot(t_d, phi_d[:, 4], label="Dysbiotic (S5)", color="red", linestyle="-", linewidth=2.5)

    plt.title("P. gingivalis (S5) Surge Verification", fontsize=14)
    plt.xlabel("Time (steps)", fontsize=12)
    plt.ylabel("Relative Abundance", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.ylim(0, 0.4)  # Adjust based on expected range

    plt.savefig(output_path / "comparison_S5_surge.png", dpi=300)
    plt.close()

    # 2. Helper Species (S4: Purple) Comparison
    plt.figure(figsize=(8, 6))
    plt.plot(t_c, phi_c[:, 3], label="Commensal (S4)", color="purple", linestyle="--", alpha=0.7)
    plt.plot(t_d, phi_d[:, 3], label="Dysbiotic (S4)", color="purple", linestyle="-", linewidth=2.5)

    plt.title("F. nucleatum (S4) Response", fontsize=14)
    plt.xlabel("Time (steps)", fontsize=12)
    plt.ylabel("Relative Abundance", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)

    plt.savefig(output_path / "comparison_S4_response.png", dpi=300)
    plt.close()

    print(f"Comparison plots saved to {output_path}")


def main():
    args = parse_args()

    print("Loading Commensal run...")
    map_c = load_run_data(args.commensal)

    print("Loading Dysbiotic run...")
    map_d = load_run_data(args.dysbiotic)

    print("Simulating Commensal MAP...")
    t_c, phi_c = simulate_run(map_c, "Commensal")

    print("Simulating Dysbiotic MAP...")
    t_d, phi_d = simulate_run(map_d, "Dysbiotic")

    print("Generating Comparison Plots...")
    plot_comparison(t_c, phi_c, t_d, phi_d, args.output_dir)


if __name__ == "__main__":
    main()
