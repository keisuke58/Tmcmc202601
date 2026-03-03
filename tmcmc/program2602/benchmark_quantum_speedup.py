# -*- coding: utf-8 -*-
"""
benchmark_quantum_speedup.py - Benchmark Script for Quantum vs Classical

Measures execution time of:
1. Classical Solver (BiofilmNewtonSolver5S) - Solving stiff ODEs
2. Quantum Surrogate (QuantumBiofilmSurrogate) - Circuit inference

Calculates Speedup Factor.
"""

import time
import numpy as np
import logging
import sys
import os

# Ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from improved_5species_jit import BiofilmNewtonSolver5S
from quantum_surrogate import QuantumBiofilmSurrogate

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("=== Starting Benchmark: Quantum vs Classical ===")

    # 1. Setup
    n_classical_samples = 5
    n_quantum_samples = 1000

    params = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.01])  # 6 params

    # 2. Classical Benchmark
    solver = BiofilmNewtonSolver5S()
    logger.info(f"Running Classical Solver ({n_classical_samples} samples)...")

    start_c = time.time()
    for _ in range(n_classical_samples):
        solver.solve(params)
    end_c = time.time()

    avg_c = (end_c - start_c) / n_classical_samples
    logger.info(f"Classical Avg Time: {avg_c:.4f} s/sample")

    # 3. Quantum Benchmark
    # Initialize with same architecture
    q_model = QuantumBiofilmSurrogate(n_qubits=6, n_layers=4)
    # Use random weights (performance is independent of weight values)

    logger.info(f"Running Quantum Surrogate ({n_quantum_samples} samples)...")

    start_q = time.time()
    for _ in range(n_quantum_samples):
        q_model.predict(params)
    end_q = time.time()

    avg_q = (end_q - start_q) / n_quantum_samples
    logger.info(f"Quantum Avg Time:   {avg_q:.6f} s/sample")

    # 4. Result
    speedup = avg_c / avg_q
    logger.info(f"=== RESULT: Quantum Speedup Factor: {speedup:.1f}x ===")


if __name__ == "__main__":
    main()
