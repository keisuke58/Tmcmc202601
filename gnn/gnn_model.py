"""
GNN model for a_ij edge regression (Project B, Issue #39).

Architecture: GCN layers + edge MLP to predict a_ij for active edges.
"""

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
    GCN + residual + LayerNorm → flatten 5 nodes → MLP → 5 a_ij.
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
