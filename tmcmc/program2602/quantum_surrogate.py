# -*- coding: utf-8 -*-
"""
quantum_surrogate.py - Quantum Surrogate Model for Biofilm Simulation

This module implements a Variational Quantum Circuit (VQC) using Qiskit
to act as a surrogate model for the expensive biofilm simulation.

It maps input parameters (theta) to output quantities (e.g., biofilm thickness)
using a parameterized quantum circuit.
"""

import numpy as np
import logging
import os
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp

logger = logging.getLogger(__name__)


class QuantumBiofilmSurrogate:
    """
    A quantum surrogate model that uses a parameterized quantum circuit (PQC)
    to approximate the biofilm simulation.
    """

    def __init__(self, n_qubits=4, n_layers=2):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.simulator = AerSimulator()

        # Define circuit parameters
        self._circuit = None
        self._input_params = None
        self._weight_params = None
        self._observable = None

        # Store trained weights
        self.trained_weights = None

        self._build_circuit()

    def _build_circuit(self):
        """Builds the variational quantum circuit with Input and Weight layers."""
        qc = QuantumCircuit(self.n_qubits)

        # 1. Input Encoding Layer (Data Re-uploading could be used, but simple encoding here)
        # Parameter vector for input features (theta from TMCMC)
        self._input_params = ParameterVector("theta", self.n_qubits)

        # 2. Variational Layer (Trainable Weights)
        # We use a hardware-efficient ansatz: Ry rotations + CNOT entanglement
        num_weights = self.n_qubits * self.n_layers
        self._weight_params = ParameterVector("weights", num_weights)

        # --- Construct Circuit ---

        # A. Input Encoding (State Preparation)
        # We map input theta components to rotation angles on qubits
        for q in range(self.n_qubits):
            qc.ry(self._input_params[q], q)
            # Maybe add Rz for more expressivity if inputs are complex?
            # For now, simple Ry(theta) is enough for real-valued inputs.

        # B. Variational Layers (The "Neural Network" part)
        weight_idx = 0
        for l in range(self.n_layers):
            # Entanglement Layer
            for q in range(self.n_qubits - 1):
                qc.cx(q, q + 1)
            # Boundary entanglement (Ring topology)
            if self.n_qubits > 1:
                qc.cx(self.n_qubits - 1, 0)

            # Rotation Layer (Trainable)
            for q in range(self.n_qubits):
                if weight_idx < len(self._weight_params):
                    qc.ry(self._weight_params[weight_idx], q)
                    weight_idx += 1

        self._circuit = qc

        # Define a simple observable (e.g., Z on the first qubit)
        # We want the output to be in range [-1, 1] usually, or [0, 1]
        self._observable = SparsePauliOp.from_list([("Z" * self.n_qubits, 1)])

        # Initialize weights randomly
        self.trained_weights = np.random.uniform(0, 2 * np.pi, len(self._weight_params))

    def predict(self, theta, weights=None):
        """
        Predicts the simulation output for a given set of parameters theta.

        Args:
            theta (np.ndarray): Input parameters (1D array).
            weights (np.ndarray, optional): Weights for the variational layer.
                                          If None, uses self.trained_weights.

        Returns:
            float: Predicted value.
        """
        if weights is None:
            weights = self.trained_weights

        # Normalize/Scale input theta to fit into circuit parameters
        # We assume theta is already pre-processed or we take the first n_qubits
        flat_theta = theta.flatten()

        # Simple Input Mapping: Take first n_qubits
        if len(flat_theta) >= len(self._input_params):
            input_vals = flat_theta[: len(self._input_params)]
        else:
            # Pad with zeros
            input_vals = np.pad(flat_theta, (0, len(self._input_params) - len(flat_theta)))

        # Combine inputs and weights
        # Order in assign_parameters depends on how they were created.
        # Qiskit's assign_parameters by name or list.
        # Since we used ParameterVectors, we can assign them dictionary-style or concatenated list.
        # But here we need to be careful about order.

        # Let's use a dictionary for safety
        param_dict = {}
        for i, p in enumerate(self._input_params):
            param_dict[p] = input_vals[i]
        for i, p in enumerate(self._weight_params):
            param_dict[p] = weights[i]

        # Simple measurement sampling
        bound_qc = self._circuit.assign_parameters(param_dict)
        bound_qc.measure_all()
        result = self.simulator.run(bound_qc, shots=1024).result()
        counts = result.get_counts()

        # Calculate expectation value from counts (Z operator on all qubits)
        # Parity: even '1's -> +1, odd '1's -> -1
        exp_val = 0
        total_shots = sum(counts.values())
        for state, count in counts.items():
            parity = state.count("1")
            sign = 1 if parity % 2 == 0 else -1
            exp_val += sign * count

        return exp_val / total_shots

    def train(self, X_train, y_train, max_iter=100, method="COBYLA"):
        """
        Train the quantum circuit parameters using a classical optimizer.

        Args:
            X_train (np.ndarray): Training input data (n_samples, n_features).
            y_train (np.ndarray): Training target data (n_samples,).
            max_iter (int): Maximum number of iterations.
            method (str): Optimization method (default: 'COBYLA').

        Returns:
            dict: Optimization result.
        """
        from scipy.optimize import minimize

        logger.info(f"Starting training with {len(X_train)} samples using {method}...")

        # Loss function: Mean Squared Error
        def loss_function(weights):
            total_loss = 0
            for i in range(len(X_train)):
                # Predict
                y_pred = self.predict(X_train[i], weights)
                # MSE
                total_loss += (y_pred - y_train[i]) ** 2
            return total_loss / len(X_train)

        # Initial weights
        initial_weights = self.trained_weights.copy()

        # Optimization
        result = minimize(
            loss_function,
            initial_weights,
            method=method,
            options={"maxiter": max_iter, "disp": True},
        )

        if result.success:
            logger.info("Training successful!")
        else:
            logger.warning(f"Training finished with message: {result.message}")

        self.trained_weights = result.x
        logger.info(f"Final Loss: {result.fun:.6f}")

        return result

    def save_weights(self, filepath):
        """Save trained weights to a file."""
        np.save(filepath, self.trained_weights)
        logger.info(f"Weights saved to {filepath}")

    def load_weights(self, filepath):
        """Load weights from a file."""
        if os.path.exists(filepath):
            self.trained_weights = np.load(filepath)
            logger.info(f"Weights loaded from {filepath}")
        else:
            logger.error(f"File not found: {filepath}")


if __name__ == "__main__":
    # Simple test
    logging.basicConfig(level=logging.INFO)
    surrogate = QuantumBiofilmSurrogate(n_qubits=4, n_layers=2)

    # Generate dummy training data
    # Target function: y = sin(theta[0] + theta[1]) (scaled to [-1, 1])
    n_samples = 20
    X_train = np.random.uniform(0, np.pi, (n_samples, 10))
    y_train = np.sin(X_train[:, 0] + X_train[:, 1])

    print("Initial Prediction (Random Weights):")
    print(f"Target: {y_train[0]:.4f}, Pred: {surrogate.predict(X_train[0]):.4f}")

    # Train
    surrogate.train(X_train, y_train, max_iter=50)

    print("Post-Training Prediction:")
    print(f"Target: {y_train[0]:.4f}, Pred: {surrogate.predict(X_train[0]):.4f}")
