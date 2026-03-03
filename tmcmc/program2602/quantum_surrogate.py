# -*- coding: utf-8 -*-
"""
quantum_surrogate.py - Variational Quantum Surrogate Model for Biofilm Simulation

This class implements a Quantum Neural Network (QNN) surrogate for the TMCMC pipeline.
Key Features:
- Replaces expensive ODE solving with fast quantum circuit inference.
- Uses Hardware-Efficient Ansatz (Ry + Rz + CNOT Ring) for expressivity.
- Supports training (fit) via classical optimization (COBYLA/L-BFGS-B).
- Saves/Loads trained weights for persistence.

Author: Keisuke Nishioka (AI Assistant)
"""

import numpy as np
import logging
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
import os
import json
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class QuantumBiofilmSurrogate:
    def __init__(self, n_qubits=6, n_layers=3, method="COBYLA"):
        """
        Initialize the Quantum Surrogate Model.

        Args:
            n_qubits (int): Number of qubits (determines input dimension capacity).
            n_layers (int): Depth of the variational circuit (determines expressivity).
            method (str): Optimization method for training (e.g., 'COBYLA', 'L-BFGS-B').
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.method = method
        self.training_history = []

        # Build the circuit
        self._build_circuit()

        # Initialize simulator
        # 'statevector' is exact and fast for small qubit counts (up to ~20)
        self.backend = AerSimulator(method="statevector")

        # Initialize weights randomly [-pi, pi]
        self.trained_weights = np.random.uniform(-np.pi, np.pi, len(self._weight_params))

        logger.info(f"Quantum Surrogate Initialized: {n_qubits} qubits, {n_layers} layers.")
        logger.info(f"Total Trainable Parameters: {len(self.trained_weights)}")

    def _build_circuit(self):
        """
        Builds the Variational Quantum Circuit (VQC).
        Structure:
        - Input Encoding: Ry(x) + Rz(x) for complex amplitude mapping
        - Ansatz: Hardware Efficient (Ry + CNOT Ring) + RealAmplitudes-like blocks
        """
        qc = QuantumCircuit(self.n_qubits)

        # --- Parameters ---
        self._input_params = ParameterVector("theta", self.n_qubits)

        # Enhanced Ansatz: More weights per layer
        # 2 rotations per qubit per layer (Ry, Rz) + Entanglement
        num_weights_per_layer = 2 * self.n_qubits
        total_weights = num_weights_per_layer * self.n_layers
        self._weight_params = ParameterVector("weights", total_weights)

        # --- Circuit Construction ---

        # 1. Input Encoding (Data Loading)
        # Using Angle Encoding with both Ry and Rz to map input to Bloch sphere surface
        for q in range(self.n_qubits):
            qc.ry(self._input_params[q], q)
            qc.rz(self._input_params[q], q)  # Adds phase information

        # 2. Variational Layers
        weight_idx = 0
        for l in range(self.n_layers):
            # A. Entanglement Layer (Ring Topology + Nearest Neighbor)
            # EfficientSU2 style entanglement
            for q in range(self.n_qubits):
                qc.cx(q, (q + 1) % self.n_qubits)

            # B. Rotation Layer (Variational Parameters)
            # Ry + Rz allows arbitrary single-qubit rotation
            for q in range(self.n_qubits):
                if weight_idx < len(self._weight_params):
                    qc.ry(self._weight_params[weight_idx], q)
                    weight_idx += 1
                if weight_idx < len(self._weight_params):
                    qc.rz(self._weight_params[weight_idx], q)
                    weight_idx += 1

        self._circuit = qc

        # Observable: Z measurement on all qubits (Sum of Z)
        # Averaging Z operators is standard for regression
        # For multi-output, we would measure specific qubits for specific outputs
        self._observable = SparsePauliOp(
            ["I" * self.n_qubits]
        )  # Start with Identity (will be replaced)

        # Create Sum(Z_i) operator
        ops = []
        for i in range(self.n_qubits):
            # 'I' * (n-1-i) + 'Z' + 'I' * i  (Qiskit Little Endian)
            # Actually SparsePauliOp.from_list handles this easier
            # Let's just measure Z on qubit 0 for scalar output for now,
            # or Average Z across all qubits for robustness.
            # Robust: Average Magnetization
            pauli_str = ["I"] * self.n_qubits
            pauli_str[self.n_qubits - 1 - i] = "Z"
            ops.append("".join(pauli_str))

        self._observable = SparsePauliOp(ops, coeffs=[1.0 / self.n_qubits] * self.n_qubits)

    def predict(self, theta, weights=None):
        """
        Run inference on the quantum circuit.
        Args:
            theta (array-like): Input parameters (normalized to roughly [-pi, pi]).
            weights (array-like, optional): Variational weights. Uses self.trained_weights if None.
        Returns:
            float: Predicted value (expectation value of Observable).
        """
        theta = np.asarray(theta)

        # Handle batch input if necessary (but for TMCMC we usually do one by one or small batches)
        # For this simple implementation, we assume single sample input
        if theta.ndim > 1:
            return np.array([self.predict(t, weights) for t in theta])

        # Truncate or Pad theta to match n_qubits
        if len(theta) > self.n_qubits:
            theta_eff = theta[: self.n_qubits]
        else:
            theta_eff = np.pad(theta, (0, self.n_qubits - len(theta)))

        current_weights = weights if weights is not None else self.trained_weights

        # Bind parameters
        # We need to map Parameter objects to values
        # order: input_params then weight_params
        param_dict = {}
        for i, p in enumerate(self._input_params):
            param_dict[p] = theta_eff[i]
        for i, p in enumerate(self._weight_params):
            param_dict[p] = current_weights[i]

        bound_qc = self._circuit.assign_parameters(param_dict)

        # Correct usage: save_expectation_value(operator, qubits)
        # qubits argument should be the list of qubits the operator acts on.
        # Since our operator is Z*n_qubits acting on all qubits, we pass list(range(n_qubits))
        bound_qc.save_expectation_value(self._observable, list(range(self.n_qubits)))

        # Transpile is usually needed for real backends, but Aer handles basic gates well.
        # Run simulation
        result = self.backend.run(bound_qc).result()
        exp_val = result.data()["expectation_value"]

        return np.real(exp_val)  # Return real part (expectation of Hermitian is real)

    def train(self, X_train, y_train, max_iter=100):
        """
        Train the quantum surrogate using classical optimization.
        Args:
            X_train (array-like): Training inputs (N_samples, N_features).
            y_train (array-like): Training targets (N_samples,).
            max_iter (int): Maximum optimization iterations.
        """
        from scipy.optimize import minimize

        logger.info(f"Starting Quantum Training with {len(X_train)} samples...")
        start_time = time.time()

        # Loss function (Mean Squared Error)
        def loss_function(weights):
            total_loss = 0
            # Note: This simple loop is slow for large datasets.
            # In production, we would parallelize or use Qiskit's Estimator primitive with batching.
            for i in range(len(X_train)):
                y_pred = self.predict(X_train[i], weights)
                total_loss += (y_pred - y_train[i]) ** 2

            loss = total_loss / len(X_train)
            self.training_history.append(loss)
            return loss

        initial_weights = self.trained_weights.copy()

        # Run optimization
        result = minimize(
            loss_function,
            initial_weights,
            method=self.method,
            options={"maxiter": max_iter, "disp": True},
        )

        end_time = time.time()
        duration = end_time - start_time

        if result.success:
            logger.info(f"Training successful! Final Loss: {result.fun:.6f}")
            logger.info(f"Time: {duration:.2f}s")
        else:
            logger.warning(f"Training finished with status: {result.message}")

        self.trained_weights = result.x
        return result

    def save_weights(self, filepath="quantum_weights.json"):
        """Save trained weights to a JSON file."""
        data = {
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "weights": self.trained_weights.tolist(),
            "history": self.training_history,
        }
        with open(filepath, "w") as f:
            json.dump(data, f)
        logger.info(f"Weights saved to {filepath}")

    def load_weights(self, filepath="quantum_weights.json"):
        """Load weights from a JSON file."""
        if not os.path.exists(filepath):
            logger.error(f"Weight file not found: {filepath}")
            return False

        with open(filepath, "r") as f:
            data = json.load(f)

        if data["n_qubits"] != self.n_qubits or data["n_layers"] != self.n_layers:
            logger.warning("Model architecture mismatch in loaded weights!")

        self.trained_weights = np.array(data["weights"])
        self.training_history = data.get("history", [])
        logger.info(f"Weights loaded from {filepath}")
        return True
