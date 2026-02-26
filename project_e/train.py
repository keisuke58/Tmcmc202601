#!/usr/bin/env python3
"""
Train VAE for Project E (Phase 1: posterior compression, Phase 2: + synthetic data).

Usage:
    python train.py --epochs 500 --latent-dim 16
    python train.py --epochs 200 --synthetic data/synthetic_all_N20000.npz
    python train.py --epochs 300 --posterior --synthetic data/synthetic_all_N10000.npz
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from load_posterior_data import load_all
from vae_model import VAE, vae_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_E_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_E_DIR / "data"
CKPT_DIR = DATA_DIR / "checkpoints"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--beta", type=float, default=1.0, help="KL weight in VAE loss")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument(
        "--posterior",
        action="store_true",
        default=True,
        help="Use TMCMC posterior data (default: True)",
    )
    parser.add_argument("--no-posterior", action="store_true", help="Skip posterior data")
    parser.add_argument(
        "--synthetic", type=str, default=None, help="Path to synthetic .npz (y_obs, theta)"
    )
    args = parser.parse_args()
    if args.no_posterior:
        args.posterior = False

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    y_list, theta_list = [], []

    # Load TMCMC posterior
    if args.posterior:
        logger.info("Loading TMCMC posterior data...")
        data = load_all()
        y_list.append(data["y_obs"])
        theta_list.append(data["theta"])

    # Load synthetic (Phase 2)
    if args.synthetic:
        syn_path = Path(args.synthetic)
        if not syn_path.is_absolute():
            syn_path = PROJECT_E_DIR / syn_path
        if syn_path.exists():
            npz = np.load(syn_path)
            y_syn = npz["y_obs"]
            theta_syn = npz["theta"]
            if y_syn.ndim == 3:
                y_syn = y_syn.reshape(y_syn.shape[0], -1)
            y_list.append(y_syn)
            theta_list.append(theta_syn)
            logger.info(f"Loaded synthetic: y {y_syn.shape}, theta {theta_syn.shape}")
        else:
            logger.warning(f"Synthetic file not found: {syn_path}")

    if not y_list or not theta_list:
        raise ValueError("No data: use --posterior and/or --synthetic")
    y = torch.from_numpy(np.concatenate(y_list, axis=0)).float()
    theta = torch.from_numpy(np.concatenate(theta_list, axis=0)).float()

    # Normalize theta to [0,1] for stable training (optional; theta is already ~[0,5])
    theta_min, theta_max = theta.min(dim=0).values, theta.max(dim=0).values
    theta_range = theta_max - theta_min
    theta_range[theta_range < 1e-6] = 1.0
    theta_norm = (theta - theta_min) / theta_range

    dataset = TensorDataset(y, theta_norm)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # Save normalization params for inference
    np.savez(
        DATA_DIR / "theta_norm_params.npz",
        theta_min=theta_min.numpy(),
        theta_max=theta_max.numpy(),
    )

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, epoch_recon, epoch_kl = 0.0, 0.0, 0.0
        n_batches = 0
        for y_b, theta_b in loader:
            y_b, theta_b = y_b.to(device), theta_b.to(device)
            optimizer.zero_grad()
            theta_recon, mu, logvar, _ = model(y_b)
            loss, d = vae_loss(theta_b, theta_recon, mu, logvar, beta=args.beta)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_recon += d["recon"]
            epoch_kl += d["kl"]
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        avg_recon = epoch_recon / n_batches
        avg_kl = epoch_kl / n_batches

        if epoch % 50 == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch}/{args.epochs}  loss={avg_loss:.4f}  recon={avg_recon:.4f}  kl={avg_kl:.4f}"
            )

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "loss": avg_loss,
                    "latent_dim": args.latent_dim,
                },
                CKPT_DIR / "best.pt",
            )

        if args.save_every > 0 and epoch % args.save_every == 0:
            torch.save(model.state_dict(), CKPT_DIR / f"epoch_{epoch}.pt")

    logger.info(f"Training done. Best loss: {best_loss:.4f}")
    logger.info(f"Checkpoint: {CKPT_DIR / 'best.pt'}")


if __name__ == "__main__":
    main()
