#!/usr/bin/env python3
"""
VAE model for Project E: approximate p(θ|y_obs).

Encoder: y_obs (30) -> z (latent)
Decoder: z -> θ (20)

Uses reparameterization trick for training.
"""

import torch
import torch.nn as nn

# Input: y_obs flattened (6*5=30), Output: theta (20)
Y_DIM = 30
THETA_DIM = 20


class Encoder(nn.Module):
    """Encode y_obs -> (z_mean, z_logvar)."""

    def __init__(self, latent_dim: int = 16, hidden: list[int] = [64, 32]):
        super().__init__()
        layers = []
        prev = Y_DIM
        for h in hidden:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h)])
            prev = h
        self.fc = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.fc(x)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """Decode z -> theta."""

    def __init__(self, latent_dim: int = 16, hidden: list[int] = [32, 64]):
        super().__init__()
        layers = []
        prev = latent_dim
        for h in hidden:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h)])
            prev = h
        layers.append(nn.Linear(prev, THETA_DIM))
        self.fc = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z)


class VAE(nn.Module):
    """VAE: y_obs -> z -> theta."""

    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(y)
        z = self.reparameterize(mu, logvar)
        theta_recon = self.decoder(z)
        return theta_recon, mu, logvar, z

    def sample(self, y: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """Sample theta given y_obs (for inference)."""
        self.eval()
        with torch.no_grad():
            mu, logvar = self.encoder(y)
            samples = []
            for _ in range(n_samples):
                z = self.reparameterize(mu, logvar)
                theta = self.decoder(z)
                samples.append(theta)
            return torch.stack(samples, dim=0)


def vae_loss(
    theta_true: torch.Tensor,
    theta_recon: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """
    VAE loss: reconstruction + beta * KL.

    Returns
    -------
    loss : torch.Tensor
    dict : {recon_loss, kl_loss}
    """
    recon_loss = nn.functional.mse_loss(theta_recon, theta_true, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss = recon_loss + beta * kl_loss
    return loss, {"recon": recon_loss.item(), "kl": kl_loss.item()}
