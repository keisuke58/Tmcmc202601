"""
GNN model for parameter prediction (Project B, Issue #39).

v1: GCN → 5 active edges (a_ij point estimates)
v2: GCN → all 20 params with heteroscedastic (μ, σ) output
MLP baseline: same I/O without graph structure (ablation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv, global_mean_pool
    from torch_geometric.data import Batch

    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False


class InteractionGNN(nn.Module):
    """
    v1: GCN + residual + LayerNorm → flatten 5 nodes → MLP → 5 a_ij.
    Input: node features (5 nodes x 3 features)
    Output: 5 scalars (a_ij for active edges)
    """

    def __init__(self, in_dim=3, hidden=64, out_dim=5, n_layers=3, dropout=0.1):
        super().__init__()
        if not PYG_AVAILABLE:
            raise ImportError("torch_geometric required")

        self.input_proj = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(GCNConv(hidden, hidden))
            self.norms.append(nn.LayerNorm(hidden))

        self.dropout = dropout
        self.fc = nn.Sequential(
            nn.Linear(hidden * 5, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, edge_index)
            h = norm(h)
            h = F.gelu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = x + h  # residual connection

        # Each graph has exactly 5 nodes → reshape directly
        batch_size = batch.max().item() + 1
        h = x.view(batch_size, -1)  # (B, 5*hidden)
        return self.fc(h)


# ---------------------------------------------------------------------------
# v2: heteroscedastic GNN for all 20 parameters
# ---------------------------------------------------------------------------

LOG_SIGMA_MIN = -5.0
LOG_SIGMA_MAX = 2.0


class InteractionGNNv2(nn.Module):
    """
    GCN → predict (μ, log_σ) for all n_params parameters.

    Output shape: (B, n_params, 2) where [:,:,0]=μ, [:,:,1]=log_σ.
    Heteroscedastic: each parameter has its own learned uncertainty.
    """

    def __init__(self, in_dim=3, hidden=64, n_params=20, n_layers=3, dropout=0.1):
        super().__init__()
        if not PYG_AVAILABLE:
            raise ImportError("torch_geometric required")

        self.n_params = n_params
        self.input_proj = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(GCNConv(hidden, hidden))
            self.norms.append(nn.LayerNorm(hidden))

        self.dropout = dropout
        self.fc_shared = nn.Sequential(
            nn.Linear(hidden * 5, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
        )
        # Separate heads for μ and log_σ
        self.head_mu = nn.Linear(hidden, n_params)
        self.head_log_sigma = nn.Linear(hidden, n_params)

        # Initialize log_sigma bias to moderate uncertainty
        nn.init.constant_(self.head_log_sigma.bias, -1.0)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, edge_index)
            h = norm(h)
            h = F.gelu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = x + h

        batch_size = batch.max().item() + 1
        h = x.view(batch_size, -1)
        shared = self.fc_shared(h)

        mu = self.head_mu(shared)  # (B, n_params)
        log_sigma = self.head_log_sigma(shared)  # (B, n_params)
        log_sigma = log_sigma.clamp(LOG_SIGMA_MIN, LOG_SIGMA_MAX)

        return torch.stack([mu, log_sigma], dim=-1)  # (B, n_params, 2)


# ---------------------------------------------------------------------------
# MLP baseline (ablation: no graph structure)
# ---------------------------------------------------------------------------


class InteractionMLP(nn.Module):
    """
    MLP baseline: flatten 5×3=15 → Dense → (μ, log_σ) for n_params.
    Same input/output contract as InteractionGNNv2 but without message passing.
    """

    def __init__(self, in_dim=15, hidden=64, n_params=20, n_layers=3, dropout=0.1):
        super().__init__()
        self.n_params = n_params
        layers = [nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(n_layers - 1):
            layers.extend(
                [
                    nn.Linear(hidden, hidden),
                    nn.LayerNorm(hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
        self.trunk = nn.Sequential(*layers)
        self.head_mu = nn.Linear(hidden, n_params)
        self.head_log_sigma = nn.Linear(hidden, n_params)
        nn.init.constant_(self.head_log_sigma.bias, -1.0)

    def forward(self, data):
        """Accept PyG data for API compat; just flatten node features."""
        x = data.x  # (B*5, 3)
        batch = data.batch
        batch_size = batch.max().item() + 1
        x_flat = x.view(batch_size, -1)  # (B, 15)
        h = self.trunk(x_flat)
        mu = self.head_mu(h)
        log_sigma = self.head_log_sigma(h)
        log_sigma = log_sigma.clamp(LOG_SIGMA_MIN, LOG_SIGMA_MAX)
        return torch.stack([mu, log_sigma], dim=-1)  # (B, n_params, 2)


def heteroscedastic_nll(pred, target, mask=None):
    """Negative log-likelihood for heteroscedastic Gaussian.

    Args:
        pred: (B, n_params, 2) where [:,:,0]=μ, [:,:,1]=log_σ
        target: (B, n_params)
        mask: (B, n_params) bool, True=compute loss (optional)

    Returns:
        scalar loss (mean over valid entries)
    """
    mu = pred[:, :, 0]
    log_sigma = pred[:, :, 1]
    # NLL = log_σ + 0.5 * (y - μ)² / σ²
    nll = log_sigma + 0.5 * ((target - mu) / log_sigma.exp()) ** 2

    if mask is not None:
        nll = nll[mask]
    return nll.mean()
