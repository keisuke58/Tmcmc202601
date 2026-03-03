# -*- coding: utf-8 -*-
"""
train_and_evaluate.py - Quantum Surrogate Training & Evaluation Script

1. Generates synthetic dataset using Classical Solver (BiofilmNewtonSolver5S).
2. Trains the Quantum Surrogate (QuantumBiofilmSurrogate).
3. Evaluates accuracy (MSE, R2) and visualizes results.
"""

import numpy as np
import logging
import sys
import os
import matplotlib.pyplot as plt
import time

# Ensure we can import from current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from improved_5species_jit import BiofilmNewtonSolver5S
from quantum_surrogate import QuantumBiofilmSurrogate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_data(n_samples=20):
    """
    Generate synthetic (Input: Params -> Output: Biomass/Metabolite) pairs
    using the expensive classical solver.
    """
    logger.info(f"Generating {n_samples} samples using Classical Solver...")
    solver = BiofilmNewtonSolver5S()

    X = []
    y = []

    start_time = time.time()

    for i in range(n_samples):
        # Random parameters (normalized around some mean)
        # 5 species growth rates + 1 gamma
        # range: [0.1, 0.5]
        params = np.random.uniform(0.1, 0.5, 6)

        # Solve ODE
        t, res = solver.solve(params)

        # Extract a scalar metric: Final Biomass of Species 1 (index 0)
        # res shape: (time, 12) -> 5 phi, 1 phi0, 5 psi, 1 gamma
        # Let's take the final value of phi_1 (index 0)
        metric = res[-1, 0]

        X.append(params)
        y.append(metric)

        if (i + 1) % 5 == 0:
            logger.info(f"Generated {i+1}/{n_samples}")

    end_time = time.time()
    logger.info(f"Data Generation took {end_time - start_time:.2f}s")

    return np.array(X), np.array(y)


def main():
    # 1. Generate Data
    # Small dataset for demo. In production, use 100+ samples.
    N_TRAIN = 20
    N_TEST = 10

    X_raw, y_raw = generate_data(N_TRAIN + N_TEST)

    # Preprocessing (Normalization)
    # Quantum inputs should be in [-pi, pi], ideally [-1, 1] or [0, pi] for angle encoding.
    # Our params are [0.1, 0.5].
    # Let's map [0.1, 0.5] to [0, pi]
    def normalize_X(val):
        return (val - 0.1) / 0.4 * np.pi

    X_norm = normalize_X(X_raw)

    # Target normalization (StandardScaler or MinMax)
    # Quantum output (Z expectation) is in [-1, 1].
    # y_raw is likely in [0, 1] (volume fraction).
    # Map [min, max] -> [-0.9, 0.9] to stay within linear region of sigmoid/expectation
    y_min = np.min(y_raw)
    y_max = np.max(y_raw)

    def normalize_y(val):
        return 1.8 * (val - y_min) / (y_max - y_min) - 0.9

    def denormalize_y(val):
        return (val + 0.9) / 1.8 * (y_max - y_min) + y_min

    y_norm = normalize_y(y_raw)

    # Split
    X_train, X_test = X_norm[:N_TRAIN], X_norm[N_TRAIN:]
    y_train, y_test = y_norm[:N_TRAIN], y_norm[N_TRAIN:]

    # 2. Initialize Model
    # 6 qubits (for 6 input params), 4 layers
    model = QuantumBiofilmSurrogate(n_qubits=6, n_layers=4, method="COBYLA")

    # 3. Train
    # COBYLA is robust but slow-ish. max_iter=100
    logger.info("Training Model...")
    # Increase iterations to improve accuracy
    model.train(X_train, y_train, max_iter=800)

    # Save weights
    model.save_weights("quantum_weights.json")

    # 4. Evaluate
    logger.info("Evaluating on Test Set...")
    y_pred_norm = np.array([model.predict(x) for x in X_test])

    # Metrics
    mse = np.mean((y_pred_norm - y_test) ** 2)
    logger.info(f"Test MSE (Normalized): {mse:.6f}")

    # R2 Score
    y_test_phys = denormalize_y(y_test)
    y_pred_phys = denormalize_y(y_pred_norm)

    ss_res = np.sum((y_test_phys - y_pred_phys) ** 2)
    ss_tot = np.sum((y_test_phys - np.mean(y_test_phys)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    logger.info(f"Test R2 Score: {r2:.4f}")

    # Plot
    try:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(model.training_history)
        plt.title("Training Loss")
        plt.xlabel("Iteration")
        plt.ylabel("MSE Loss")

        plt.subplot(1, 2, 2)
        plt.scatter(y_test_phys, y_pred_phys, color="blue", label="Test Data")
        plt.plot([y_min, y_max], [y_min, y_max], "r--", label="Ideal")
        plt.title(f"Prediction (R2={r2:.2f})")
        plt.xlabel("True Value")
        plt.ylabel("Predicted Value")
        plt.legend()

        plt.tight_layout()
        plt.savefig("quantum_accuracy_report.png")
        logger.info("Plot saved to quantum_accuracy_report.png")
    except Exception as e:
        logger.warning(f"Plotting failed: {e}")


if __name__ == "__main__":
    main()
