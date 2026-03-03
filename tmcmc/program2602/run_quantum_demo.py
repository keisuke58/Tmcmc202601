# -*- coding: utf-8 -*-
"""
run_quantum_demo.py - Demonstration of Quantum Surrogate Model

This script demonstrates the usage of the QuantumBiofilmSurrogate class
to approximate the behavior of the biofilm simulation.
"""

import numpy as np
import logging
import sys
import os
import matplotlib.pyplot as plt

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from quantum_surrogate import QuantumBiofilmSurrogate

try:
    from improved_5species_jit import BiofilmNewtonSolver5S

    HAS_SOLVER = True
except ImportError:
    HAS_SOLVER = False
    print("Warning: BiofilmNewtonSolver5S not found. Running in standalone mode.")


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    print("==================================================")
    print("   Quantum Algorithm Integration Demo")
    print("   Target: Biofilm Simulation Surrogate Model")
    print("==================================================")

    # 1. Initialize Quantum Surrogate
    print("\n[1] Initializing Quantum Surrogate Model (Qiskit Aer)...")
    n_qubits = 6  # Use more qubits for better expressivity
    n_layers = 2
    surrogate = QuantumBiofilmSurrogate(n_qubits=n_qubits, n_layers=n_layers)

    # Load trained weights if available
    weights_file = "quantum_weights.npy"
    if os.path.exists(weights_file):
        print(f"    Loading trained weights from {weights_file}...")
        try:
            surrogate.load_weights(weights_file)
        except Exception as e:
            print(f"    Failed to load weights: {e}")
            print("    Using random weights.")
    else:
        print("    No trained weights found. Using random weights.")

    print(f"    - Qubits: {n_qubits}")
    print(f"    - Layers: {n_layers}")
    print(f"    - Circuit Depth: {surrogate._circuit.depth()}")

    # 2. Define a test parameter set (theta)
    # 20 parameters in total for the full model
    theta_true = np.array(
        [
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,  # a11..a15 (dummy)
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,  # b1..b5
        ]
    )

    print("\n[2] Running Prediction...")

    # Classical Simulation (if available)
    if HAS_SOLVER:
        print("    Running Classical Simulation (BiofilmNewtonSolver5S)...")
        # solver = BiofilmNewtonSolver5S(...)
        # For this demo, we skip actual heavy computation as we don't have data files
        # and just simulate a dummy value or use a random one
        classical_output = 0.5 + 0.1 * np.sin(np.sum(theta_true))
        print(f"    Classical Result (Simulated): {classical_output:.4f}")
    else:
        classical_output = 0.5

    # Quantum Prediction
    print("    Running Quantum Surrogate Prediction...")
    # Normalize theta to [0, 2pi] for rotation gates
    # Simple scaling for demo
    theta_scaled = theta_true * np.pi
    quantum_output = surrogate.predict(theta_scaled)

    print(f"    Quantum Result: {quantum_output:.4f}")

    # 3. Visualization (Mockup)
    print("\n[3] Comparison")
    diff = abs(classical_output - quantum_output)
    print(f"    Difference: {diff:.4f}")
    print("\nNote: The quantum model is untrained. In a real scenario,")
    print("      we would train the VQC parameters to minimize this difference.")

    # Generate a simple plot of Quantum Landscape
    print("\n[4] Generating Quantum Landscape Scan...")
    scan_points = 20
    x_vals = np.linspace(0, 2 * np.pi, scan_points)
    y_vals = []

    # Vary one parameter
    base_theta = theta_scaled.copy()
    for x in x_vals:
        base_theta[0] = x
        y_vals.append(surrogate.predict(base_theta))

    try:
        plt.figure(figsize=(10, 6))
        plt.plot(x_vals, y_vals, "b-o", label="Quantum Output")
        plt.title("Quantum Surrogate Landscape (Parameter 0 Variation)")
        plt.xlabel("Parameter 0 (radians)")
        plt.ylabel("Expectation Value <Z>")
        plt.grid(True)
        plt.legend()
        output_file = "quantum_landscape_demo.png"
        plt.savefig(output_file)
        print(f"    Plot saved to {output_file}")
    except Exception as e:
        print(f"    Plotting failed: {e}")


if __name__ == "__main__":
    main()
