#!/usr/bin/env python3
"""
Evaluate VAE for Project E: compare VAE-sampled θ with TMCMC posterior.

Usage:
    python eval.py --checkpoint data/checkpoints/best.pt
    python eval.py --checkpoint data/checkpoints/best.pt --plot
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from load_posterior_data import load_all
from vae_model import VAE

PROJECT_E_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_E_DIR / "data"
CKPT_DIR = DATA_DIR / "checkpoints"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(CKPT_DIR / "best.pt"))
    parser.add_argument("--n-samples", type=int, default=100, help="Samples per condition for VAE")
    parser.add_argument("--plot", action="store_true", help="Generate comparison plots")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        print("Run: python train.py --epochs 500")
        return

    ckpt = torch.load(ckpt_path, map_location=device)
    latent_dim = ckpt.get("latent_dim", 16)
    model = VAE(latent_dim=latent_dim).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Load data
    data = load_all()
    y_all = data["y_obs"]
    theta_true = data["theta"]
    labels = data["condition_labels"]

    # Load norm params if available
    norm_path = DATA_DIR / "theta_norm_params.npz"
    if norm_path.exists():
        npz = np.load(norm_path)
        theta_min = torch.from_numpy(npz["theta_min"]).float().to(device)
        theta_max = torch.from_numpy(npz["theta_max"]).float().to(device)
    else:
        theta_min = torch.from_numpy(theta_true.min(axis=0)).float().to(device)
        theta_max = torch.from_numpy(theta_true.max(axis=0)).float().to(device)

    # Get unique y per condition (first sample of each)
    cond_names = []
    y_unique = []
    seen = set()
    for i, label in enumerate(labels):
        if label not in seen:
            seen.add(label)
            cond_names.append(label)
            y_unique.append(y_all[i])

    y_unique = np.array(y_unique)
    y_t = torch.from_numpy(y_unique).float().to(device)

    # VAE sample
    with torch.no_grad():
        theta_vae = model.sample(y_t, n_samples=args.n_samples)
    # theta_vae: (n_samples, n_conditions, 20)
    theta_vae = theta_vae.cpu().numpy()

    # Denormalize
    theta_range = theta_max.cpu().numpy() - theta_min.cpu().numpy()
    theta_range[theta_range < 1e-6] = 1.0
    theta_vae_denorm = theta_vae * theta_range + theta_min.cpu().numpy()

    # Compare: mean per condition
    print("\nCondition-wise comparison (mean ± std):")
    print("-" * 60)
    for c, cond in enumerate(cond_names):
        idx_true = [i for i, l in enumerate(labels) if l == cond]
        theta_t = theta_true[idx_true]
        theta_v = theta_vae_denorm[:, c, :]
        mean_true = theta_t.mean(axis=0)
        mean_vae = theta_v.mean(axis=0)
        mae = np.abs(mean_true - mean_vae).mean()
        print(f"  {cond}: MAE(mean) = {mae:.4f}")
        print(f"    True mean[0:5]: {mean_true[:5].round(3)}")
        print(f"    VAE  mean[0:5]: {mean_vae[:5].round(3)}")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
            for c, cond in enumerate(cond_names):
                ax = axes[c]
                idx_true = [i for i, l in enumerate(labels) if l == cond]
                theta_t = theta_true[idx_true]
                theta_v = theta_vae_denorm[:, c, :]
                ax.scatter(theta_t[:, 0], theta_t[:, 1], alpha=0.3, s=8, label="TMCMC")
                ax.scatter(theta_v[:, 0], theta_v[:, 1], alpha=0.3, s=8, label="VAE")
                ax.set_xlabel("a11")
                ax.set_ylabel("a12")
                ax.set_title(cond)
                ax.legend()
            fig.suptitle("Project E: TMCMC vs VAE posterior (a11, a12)")
            fig.tight_layout()
            out_path = PROJECT_E_DIR / "data" / "eval_comparison.png"
            fig.savefig(out_path, dpi=150)
            print(f"\nSaved plot: {out_path}")
        except ImportError:
            print("matplotlib not available, skipping plot")


if __name__ == "__main__":
    main()
