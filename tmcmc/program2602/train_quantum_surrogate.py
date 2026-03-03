# -*- coding: utf-8 -*-
"""
train_quantum_surrogate.py - Training Script for Quantum Surrogate Model

This script trains the QuantumBiofilmSurrogate model using a synthetic dataset
that mimics the complexity of the biofilm simulation (non-linear, multi-dimensional).
It saves the trained weights and visualizes the training progress.
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from quantum_surrogate import QuantumBiofilmSurrogate


def target_function(theta):
    """
    Synthetic target function to mimic biofilm simulation output.
    Input: theta (shape: (N, D))
    Output: y (shape: (N,)) in range [-1, 1]
    """
    # A combination of sin/cos and interactions
    # We use only the first 4 dimensions for the core interaction
    t0 = theta[:, 0]
    t1 = theta[:, 1]
    t2 = theta[:, 2]
    t3 = theta[:, 3]

    # Non-linear interaction
    val = np.sin(t0 * np.pi) * np.cos(t1 * np.pi) + 0.5 * np.sin(t2 * t3 * 2 * np.pi)

    # Normalize roughly to [-1, 1]
    return np.clip(val, -1.0, 1.0)


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    print("==================================================")
    print("   Quantum Surrogate Model Training Demo")
    print("==================================================")

    # 1. Prepare Data
    n_train = 50
    n_test = 20
    n_features = 10  # Simulate 20 parameters, but only use first few for demo

    print(f"[1] Generating synthetic dataset (Train: {n_train}, Test: {n_test})...")
    X_train = np.random.rand(n_train, n_features)
    y_train = target_function(X_train)

    X_test = np.random.rand(n_test, n_features)
    y_test = target_function(X_test)

    # 2. Initialize Model
    print("\n[2] Initializing Quantum Surrogate...")
    n_qubits = 6
    n_layers = 3  # Increased layers for better expressivity
    model = QuantumBiofilmSurrogate(n_qubits=n_qubits, n_layers=n_layers)

    # 3. Train
    print("\n[3] Starting Training (COBYLA)...")

    # We want to capture loss history, but scipy.optimize.minimize doesn't support callback for COBYLA easily
    # in a way that returns the loss. We'll wrap the loss function inside the class if we modified it,
    # but here we'll just rely on the final result and pre/post evaluation.

    # Evaluate before training
    y_pred_pre = np.array([model.predict(x) for x in X_test])
    mse_pre = np.mean((y_pred_pre - y_test) ** 2)
    print(f"    Pre-training MSE (Test): {mse_pre:.6f}")

    # Train
    max_iter = 200
    model.train(X_train, y_train, max_iter=max_iter, method="COBYLA")

    # Save weights
    weights_file = "quantum_weights.npy"
    model.save_weights(weights_file)

    # 4. Evaluate
    print("\n[4] Evaluation...")
    y_pred_post = np.array([model.predict(x) for x in X_test])
    mse_post = np.mean((y_pred_post - y_test) ** 2)
    print(f"    Post-training MSE (Test): {mse_post:.6f}")
    print(f"    Improvement: {mse_pre - mse_post:.6f}")

    # 5. Visualization
    print("\n[5] Visualizing Results...")
    plt.figure(figsize=(12, 5))

    # Plot 1: Prediction vs True (Test Set)
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred_pre, alpha=0.5, label="Pre-training", color="gray")
    plt.scatter(y_test, y_pred_post, alpha=0.7, label="Post-training", color="blue")
    plt.plot([-1, 1], [-1, 1], "r--", label="Ideal")
    plt.xlabel("True Value")
    plt.ylabel("Predicted Value")
    plt.title("Prediction Accuracy (Test Set)")
    plt.legend()
    plt.grid(True)

    # Plot 2: Residuals
    plt.subplot(1, 2, 2)
    plt.hist(y_pred_post - y_test, bins=10, color="blue", alpha=0.7)
    plt.xlabel("Residual (Pred - True)")
    plt.ylabel("Count")
    plt.title("Residual Distribution")
    plt.grid(True)

    output_img = "quantum_training_result.png"
    plt.savefig(output_img)
    print(f"    Plot saved to {output_img}")

    print("\nDone!")


if __name__ == "__main__":
    main()
