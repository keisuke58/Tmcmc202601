#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quantum_kernel_pca.py - Quantum Kernel PCA for TMCMC Posterior Analysis

Compares classical RBF kernel vs quantum (ZZFeatureMap) kernel
for visualizing the structure of posterior samples across 4 biofilm conditions.

趣味プロジェクト - 論文には入れない
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
import time
import logging

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# --- Config ---
RUNS_DIR = Path(__file__).resolve().parent.parent.parent / "data_5species" / "_runs"
CONDITIONS = {
    "CS": "commensal_static_posterior",
    "CH": "commensal_hobic_posterior",
    "DH": "dh_baseline",
    "DS": "dysbiotic_static_posterior",
}
COLORS = {"CS": "#2196F3", "CH": "#4CAF50", "DH": "#FF9800", "DS": "#F44336"}
MARKERS = {"CS": "o", "CH": "s", "DH": "^", "DS": "D"}

N_SUBSAMPLE = 60  # per condition (total 240 → 240×240 kernel)
N_QUBITS = 8  # ZZFeatureMap qubits (input PCA-reduced to this)
ZZ_REPS = 2  # ZZFeatureMap repetitions
SEED = 42


def load_posteriors():
    """Load posterior samples from all 4 conditions."""
    data = {}
    for label, dirname in CONDITIONS.items():
        fpath = RUNS_DIR / dirname / "samples.npy"
        if not fpath.exists():
            logger.warning(f"Not found: {fpath}")
            continue
        samples = np.load(fpath)
        logger.info(f"{label}: {samples.shape}")
        data[label] = samples
    return data


def build_zz_feature_map(n_qubits, reps=2):
    """
    Build a ZZFeatureMap circuit with parameter placeholders.
    φ(x) = U_ZZ(x)|0⟩

    Structure per rep:
      H^⊗n → P(2*x_i) on each qubit → CNOT+P(2*x_i*x_j)+CNOT on pairs
    """
    qc = QuantumCircuit(n_qubits)

    for _ in range(reps):
        # Hadamard layer
        for q in range(n_qubits):
            qc.h(q)

        # Single-qubit phase: P(2*x_i) — placeholder, will bind later
        # ZZ entangling: P(2*x_i*x_j) via CNOT-P-CNOT
        # We'll build this as a function that takes x and returns a bound circuit

    return qc  # skeleton — actual binding done in compute_kernel_entry


def quantum_feature_circuit(x, n_qubits, reps=2):
    """
    Create a bound ZZFeatureMap circuit for input vector x.
    Follows Havlicek et al. (2019) / Qiskit's ZZFeatureMap.
    """
    qc = QuantumCircuit(n_qubits)

    for r in range(reps):
        # Hadamard
        for q in range(n_qubits):
            qc.h(q)

        # Single-qubit Z-rotation: exp(i * x_i * Z)
        for q in range(n_qubits):
            qc.p(2.0 * x[q], q)

        # Two-qubit ZZ entanglement (linear connectivity)
        for q in range(n_qubits - 1):
            phi = 2.0 * (np.pi - x[q]) * (np.pi - x[q + 1])
            qc.cx(q, q + 1)
            qc.p(phi, q + 1)
            qc.cx(q, q + 1)

    return qc


def compute_quantum_kernel_entry(x_i, x_j, n_qubits, reps, backend):
    """
    Compute K(x_i, x_j) = |⟨φ(x_i)|φ(x_j)⟩|²

    This is done by: U†(x_i) · U(x_j) |0⟩ and measuring prob of |0...0⟩
    """
    # Build U(x_j)
    qc_j = quantum_feature_circuit(x_j, n_qubits, reps)
    # Build U†(x_i) = inverse of U(x_i)
    qc_i_inv = quantum_feature_circuit(x_i, n_qubits, reps).inverse()

    # Combined circuit: U†(x_i) · U(x_j)
    qc = qc_j.compose(qc_i_inv)

    # Save statevector to get |⟨0|U†(x_i)U(x_j)|0⟩|²
    qc.save_statevector()
    result = backend.run(qc).result()
    sv = result.get_statevector()

    # Probability of |0...0⟩ state
    prob_zero = np.abs(sv.data[0]) ** 2
    return prob_zero


def compute_quantum_kernel_matrix(X, n_qubits, reps=2):
    """
    Compute the full quantum kernel matrix for dataset X.
    X: (N, n_qubits) array
    """
    N = len(X)
    K = np.eye(N)  # K(x,x) = 1
    backend = AerSimulator(method="statevector")

    total_pairs = N * (N - 1) // 2
    logger.info(f"Computing quantum kernel: {N} samples, {total_pairs} pairs...")

    t0 = time.time()
    done = 0
    for i in range(N):
        for j in range(i + 1, N):
            k_ij = compute_quantum_kernel_entry(X[i], X[j], n_qubits, reps, backend)
            K[i, j] = k_ij
            K[j, i] = k_ij
            done += 1
            if done % 2000 == 0:
                elapsed = time.time() - t0
                eta = elapsed / done * (total_pairs - done)
                logger.info(f"  {done}/{total_pairs} ({100*done/total_pairs:.0f}%) ETA {eta:.0f}s")

    elapsed = time.time() - t0
    logger.info(
        f"Quantum kernel computed in {elapsed:.1f}s ({elapsed/total_pairs*1000:.2f}ms/pair)"
    )
    return K


def plot_results(embed_classical, embed_quantum, labels, output_path):
    """Side-by-side plot of classical vs quantum kernel PCA."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, embed, title in [
        (axes[0], embed_classical, "Classical RBF Kernel PCA"),
        (axes[1], embed_quantum, "Quantum ZZ Kernel PCA"),
    ]:
        for label in CONDITIONS:
            mask = np.array(labels) == label
            ax.scatter(
                embed[mask, 0],
                embed[mask, 1],
                c=COLORS[label],
                marker=MARKERS[label],
                s=30,
                alpha=0.6,
                label=label,
                edgecolors="k",
                linewidths=0.3,
            )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Posterior Structure: 4 Biofilm Conditions ({N_SUBSAMPLE}/cond, {N_QUBITS}q ZZFeatureMap)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {output_path}")
    plt.close()


def plot_kernel_matrices(K_classical, K_quantum, labels, output_path):
    """Visualize kernel matrices side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, K, title in [
        (axes[0], K_classical, "Classical RBF Kernel"),
        (axes[1], K_quantum, "Quantum ZZ Kernel"),
    ]:
        im = ax.imshow(K, cmap="viridis", aspect="equal")
        ax.set_title(title, fontsize=12, fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Add condition boundaries
        n = N_SUBSAMPLE
        for i, label in enumerate(CONDITIONS):
            mid = i * n + n // 2
            ax.text(
                -3,
                mid,
                label,
                fontsize=9,
                fontweight="bold",
                color=COLORS[label],
                ha="right",
                va="center",
            )
        for i in range(1, 4):
            ax.axhline(i * n - 0.5, color="white", linewidth=1, alpha=0.7)
            ax.axvline(i * n - 0.5, color="white", linewidth=1, alpha=0.7)

    plt.suptitle("Kernel Matrix Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {output_path}")
    plt.close()


def compute_kernel_alignment(K1, K2):
    """Kernel alignment score (Cristianini 2002): cosine similarity in kernel space."""
    K1c = K1 - np.mean(K1)
    K2c = K2 - np.mean(K2)
    return np.sum(K1c * K2c) / (np.linalg.norm(K1c, "fro") * np.linalg.norm(K2c, "fro"))


def compute_cluster_separation(embed, labels):
    """
    Fisher's discriminant ratio: between-class variance / within-class variance.
    Higher = better separation.
    """
    unique = list(CONDITIONS.keys())
    overall_mean = np.mean(embed, axis=0)

    between = 0.0
    within = 0.0
    for label in unique:
        mask = np.array(labels) == label
        cluster = embed[mask]
        n_k = len(cluster)
        mean_k = np.mean(cluster, axis=0)
        between += n_k * np.sum((mean_k - overall_mean) ** 2)
        within += np.sum((cluster - mean_k) ** 2)

    return between / within if within > 0 else float("inf")


def main():
    np.random.seed(SEED)

    # 1. Load data
    logger.info("=== Quantum Kernel PCA: Posterior Analysis ===")
    posteriors = load_posteriors()
    if len(posteriors) < 4:
        logger.error("Need all 4 conditions. Aborting.")
        return

    # 2. Subsample + combine
    X_list, labels = [], []
    for label, samples in posteriors.items():
        idx = np.random.choice(len(samples), size=min(N_SUBSAMPLE, len(samples)), replace=False)
        X_list.append(samples[idx])
        labels.extend([label] * len(idx))

    X_all = np.vstack(X_list)  # (240, 20)
    logger.info(f"Combined: {X_all.shape}")

    # 3. Standardize + PCA reduce to N_QUBITS dimensions
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    pca = PCA(n_components=N_QUBITS)
    X_pca = pca.fit_transform(X_scaled)
    logger.info(f"PCA variance explained: {pca.explained_variance_ratio_.sum():.2%}")

    # Rescale to [0, 2π] for quantum feature map
    X_min = X_pca.min(axis=0)
    X_max = X_pca.max(axis=0)
    X_quantum = (X_pca - X_min) / (X_max - X_min + 1e-10) * 2 * np.pi

    # 4. Classical RBF kernel
    logger.info("Computing classical RBF kernel...")
    gamma = 1.0 / (N_QUBITS * X_pca.var())
    K_classical = rbf_kernel(X_pca, gamma=gamma)

    # 5. Quantum kernel
    K_quantum = compute_quantum_kernel_matrix(X_quantum, N_QUBITS, reps=ZZ_REPS)

    # 6. Kernel PCA (2D embedding)
    logger.info("Running Kernel PCA...")
    kpca_classical = KernelPCA(n_components=2, kernel="precomputed")
    embed_classical = kpca_classical.fit_transform(K_classical)

    kpca_quantum = KernelPCA(n_components=2, kernel="precomputed")
    embed_quantum = kpca_quantum.fit_transform(K_quantum)

    # 7. Metrics
    alignment = compute_kernel_alignment(K_classical, K_quantum)
    sep_classical = compute_cluster_separation(embed_classical, labels)
    sep_quantum = compute_cluster_separation(embed_quantum, labels)

    logger.info(f"Kernel alignment (classical vs quantum): {alignment:.4f}")
    logger.info(f"Fisher separation — Classical: {sep_classical:.3f}, Quantum: {sep_quantum:.3f}")
    logger.info(f"Quantum / Classical separation ratio: {sep_quantum/sep_classical:.2f}x")

    # 8. Plots
    out_dir = Path(__file__).resolve().parent.parent.parent
    plot_results(
        embed_classical, embed_quantum, labels, out_dir / "quantum_kernel_pca_comparison.png"
    )
    plot_kernel_matrices(K_classical, K_quantum, labels, out_dir / "quantum_kernel_matrices.png")

    # 9. Summary
    print("\n" + "=" * 60)
    print("  Quantum Kernel PCA — Summary")
    print("=" * 60)
    print(f"  Samples:    {N_SUBSAMPLE}/condition × 4 = {len(X_all)}")
    print(f"  Qubits:     {N_QUBITS}")
    print(f"  ZZ reps:    {ZZ_REPS}")
    print(f"  PCA var:    {pca.explained_variance_ratio_.sum():.2%}")
    print(f"  Kernel alignment:     {alignment:.4f}")
    print(f"  Fisher (Classical):   {sep_classical:.3f}")
    print(f"  Fisher (Quantum):     {sep_quantum:.3f}")
    print(f"  Quantum advantage:    {sep_quantum/sep_classical:.2f}x")
    print("=" * 60)


if __name__ == "__main__":
    main()
