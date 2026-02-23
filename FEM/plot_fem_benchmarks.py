#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_mean(path):
    path = Path(path)
    phi = np.load(path / "snapshots_phi.npy")
    t = np.load(path / "snapshots_t.npy")
    mean_phi = phi.mean(axis=(2, 3))
    return t, mean_phi


def plot_split_scheme(out_dir):
    t_lie, m_lie = load_mean("_benchmarks/2d_lie")
    t_str, m_str = load_mean("_benchmarks/2d_strang")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["S.o", "A.n", "Veillonella", "F.n", "P.g"]
    colors = ["#1f77b4", "#2ca02c", "#bcbd22", "#9467bd", "#d62728"]
    for i in range(5):
        ax.plot(t_lie, m_lie[:, i], color=colors[i], linestyle="-", label=labels[i] + " (Lie)")
        ax.plot(t_str, m_str[:, i], color=colors[i], linestyle="--", label=labels[i] + " (Strang)")
    ax.set_xlabel("Model time")
    ax.set_ylabel("Mean volume fraction")
    ax.set_title("2D FEM: Lie vs Strang splitting (mean φ)")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fem_benchmark_split_scheme.png", dpi=300)
    plt.close(fig)


def plot_anisotropy(out_dir):
    t_iso, m_iso = load_mean("_benchmarks/2d_iso")
    t_aniso, m_aniso = load_mean("_benchmarks/2d_aniso")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t_iso, m_iso[:, 4], color="#d62728", linestyle="-", label="Pg mean (isotropic)")
    ax.plot(t_aniso, m_aniso[:, 4], color="#d62728", linestyle="--", label="Pg mean (anisotropic)")
    ax.set_xlabel("Model time")
    ax.set_ylabel("Mean volume fraction (P.g)")
    ax.set_title("2D FEM: isotropic vs anisotropic diffusion (P.g)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fem_benchmark_anisotropic.png", dpi=300)
    plt.close(fig)


def main():
    out_dir = "_benchmarks/plots"
    plot_split_scheme(out_dir)
    plot_anisotropy(out_dir)


if __name__ == "__main__":
    main()

