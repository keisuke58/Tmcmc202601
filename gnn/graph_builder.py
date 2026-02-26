"""
Graph builder for GNN: composition → PyTorch Geometric Data.

Converts (phi_mean, phi_std, phi_final, a_ij) into PyG Data format.
- Nodes: 5 species
- Node features: [phi_mean, phi_std, phi_final] per species
- Edges: complete graph (all pairs) for message passing; we predict a_ij on active edges
- Edge labels: a_ij for 5 active edges (theta[1],10,11,18,19)
"""

import numpy as np

try:
    import torch
    from torch_geometric.data import Data

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Active edges: (i, j) -> theta index
# (0,1)/(1,0): theta[1], (0,2)/(2,0): theta[10], (0,3)/(3,0): theta[11]
# (2,4)/(4,2): theta[18], (3,4)/(4,3): theta[19]
ACTIVE_EDGES = [(0, 1), (0, 2), (0, 3), (2, 4), (3, 4)]
ACTIVE_THETA_IDX = [1, 10, 11, 18, 19]


def composition_to_node_features(
    phi_mean: np.ndarray,
    phi_std: np.ndarray,
    phi_final: np.ndarray,
) -> np.ndarray:
    """Build node features: (5, 3) per species [mean, std, final]."""
    return np.stack([phi_mean, phi_std, phi_final], axis=1).astype(np.float32)


def build_edge_index_and_labels(a_ij_active: np.ndarray) -> tuple:
    """
    Build complete graph edge_index and edge labels for active edges.

    Returns:
        edge_index: (2, E) for message passing (all 10 pairs)
        edge_labels: (5,) for active edges only
        mask: (E,) boolean mask for which edges have labels
    """
    # Complete graph: all 10 undirected edges
    edge_list = []
    for i in range(5):
        for j in range(i + 1, 5):
            edge_list.append((i, j))
            edge_list.append((j, i))

    edge_index = np.array(edge_list, dtype=np.int64).T  # (2, 20)

    # Map active edges to a_ij values
    edge_to_theta = {
        (0, 1): 0,
        (1, 0): 0,
        (0, 2): 1,
        (2, 0): 1,
        (0, 3): 2,
        (3, 0): 2,
        (2, 4): 3,
        (4, 2): 3,
        (3, 4): 4,
        (4, 3): 4,
    }
    edge_labels = np.zeros(20, dtype=np.float32)
    mask = np.zeros(20, dtype=bool)
    for k, (i, j) in enumerate(zip(edge_index[0], edge_index[1])):
        idx = edge_to_theta.get((int(i), int(j)))
        if idx is not None:
            edge_labels[k] = a_ij_active[idx]
            mask[k] = True
    return edge_index, edge_labels, mask


def build_pyg_data(
    phi_mean: np.ndarray,
    phi_std: np.ndarray,
    phi_final: np.ndarray,
    a_ij_active: np.ndarray,
):
    """Build a single PyG Data object."""
    if not TORCH_AVAILABLE:
        raise ImportError("torch and torch_geometric required. pip install torch torch-geometric")

    x = np.stack([phi_mean, phi_std, phi_final], axis=1).astype(np.float32)
    edge_index, edge_labels, mask = build_edge_index_and_labels(a_ij_active)

    return Data(
        x=torch.from_numpy(x),
        edge_index=torch.from_numpy(edge_index),
        edge_attr=torch.from_numpy(edge_labels).unsqueeze(1),
        edge_mask=torch.from_numpy(mask),
        a_ij_active=torch.from_numpy(a_ij_active).float(),
    )


def dataset_to_pyg_list(data: dict) -> list:
    """Convert full dataset dict to list of PyG Data objects."""
    n = len(data["theta"])
    phi_mean = data["phi_mean"]
    phi_std = data["phi_std"]
    phi_final = data["phi_final"]
    a_ij_active = data["a_ij_active"]

    out = []
    for i in range(n):
        d = build_pyg_data(
            phi_mean[i],
            phi_std[i],
            phi_final[i],
            a_ij_active[i],
        )
        out.append(d)
    return out
