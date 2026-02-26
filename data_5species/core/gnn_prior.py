"""
GNN-informed prior for TMCMC (Project B, Issue #39).

Wraps a trained InteractionGNN model to provide:
  - log_density(theta): log prior probability for MCMC acceptance
  - sample(rng, bounds): sample theta from GNN-guided distribution
  - predict_aij(phi_stats): predict a_ij from composition features

Integration points in tmcmc.py:
  1. log_prior() closure → add GNN density term
  2. Particle initialization → GNN-guided sampling
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Active edges matching graph_builder.py
ACTIVE_THETA_IDX = [1, 10, 11, 18, 19]

# Try importing torch (optional dependency)
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class GNNPrior:
    """GNN-based informative prior for TMCMC.

    Replaces flat uniform prior with a learned distribution:
      p(theta) ∝ U(bounds) × N(theta_active | mu_gnn, sigma_gnn²)

    where mu_gnn = GNN(phi_features) and sigma_gnn controls trust level.

    Parameters
    ----------
    model_path : str or Path
        Path to trained InteractionGNN checkpoint (.pt file)
    sigma : float
        Standard deviation of Gaussian prior around GNN prediction.
        Smaller = tighter prior (more trust in GNN). Default = 1.0.
    weight : float
        Weight of GNN prior in log-posterior. 0 = ignore GNN, 1 = full weight.
    condition_phi : np.ndarray, optional
        Composition features (5,3) for the target condition.
        If None, GNN prior is disabled (falls back to uniform).
    locked_indices : list of int, optional
        Parameter indices locked to 0 (condition-specific).
    hidden : int
        Hidden dimension of the GNN model (must match checkpoint).
    n_layers : int
        Number of GCN layers (must match checkpoint).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        sigma: float = 1.0,
        weight: float = 1.0,
        condition_phi: Optional[np.ndarray] = None,
        locked_indices: Optional[List[int]] = None,
        hidden: int = 128,
        n_layers: int = 4,
    ):
        self.sigma = sigma
        self.weight = weight
        self.locked_indices = set(locked_indices or [])
        self._mu_gnn = None  # predicted a_ij means
        self._model = None

        if model_path is not None and TORCH_AVAILABLE:
            self._load_model(model_path, hidden, n_layers)
            if condition_phi is not None:
                self._mu_gnn = self._predict(condition_phi)
                logger.info(f"GNN prior initialized: mu={self._mu_gnn}, sigma={sigma}")

    def _load_model(self, model_path: str, hidden: int, n_layers: int):
        """Load trained InteractionGNN from checkpoint."""
        import sys

        gnn_dir = Path(__file__).resolve().parent.parent.parent / "gnn"
        sys.path.insert(0, str(gnn_dir))
        from gnn_model import InteractionGNN

        self._model = InteractionGNN(in_dim=3, hidden=hidden, out_dim=5, n_layers=n_layers)
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        self._model.load_state_dict(state)
        self._model.eval()
        logger.info(f"Loaded GNN model from {model_path}")

    def _predict(self, phi_features: np.ndarray) -> np.ndarray:
        """Predict a_ij from composition features (5, 3) or (5,).

        Args:
            phi_features: Either (5,3) array [phi_mean, phi_std, phi_final]
                         or (5,) array [phi] which gets expanded to (5,3).

        Returns:
            (5,) predicted a_ij for active edges
        """
        if self._model is None:
            raise RuntimeError("GNN model not loaded")

        import sys

        gnn_dir = Path(__file__).resolve().parent.parent.parent / "gnn"
        sys.path.insert(0, str(gnn_dir))
        from graph_builder import build_pyg_data

        if phi_features.ndim == 1 and phi_features.shape[0] == 5:
            # Expand single composition to (mean, std=0.1, final=same)
            phi_mean = phi_features
            phi_std = np.full(5, 0.1)
            phi_final = phi_features
        elif phi_features.shape == (5, 3):
            phi_mean = phi_features[:, 0]
            phi_std = phi_features[:, 1]
            phi_final = phi_features[:, 2]
        else:
            raise ValueError(f"phi_features must be (5,) or (5,3), got {phi_features.shape}")

        dummy_aij = np.zeros(5, dtype=np.float32)
        data = build_pyg_data(phi_mean, phi_std, phi_final, dummy_aij)
        data.batch = torch.zeros(5, dtype=torch.long)

        with torch.no_grad():
            pred = self._model(data)

        return pred.squeeze().numpy()

    def predict_aij(self, phi: np.ndarray) -> np.ndarray:
        """Public API: predict a_ij from composition. Returns (5,) array."""
        return self._predict(phi)

    def set_condition(self, phi_features: np.ndarray):
        """Update the GNN prediction for a new condition."""
        if self._model is not None:
            self._mu_gnn = self._predict(phi_features)

    def log_density(self, theta: np.ndarray) -> float:
        """Compute log prior density for theta vector.

        Returns 0.0 if GNN prior is disabled (uniform prior).
        Otherwise returns -0.5 * sum((theta_active - mu_gnn)² / sigma²).
        """
        if self._mu_gnn is None or self.weight == 0.0:
            return 0.0

        theta_active = theta[ACTIVE_THETA_IDX]
        diff = theta_active - self._mu_gnn
        log_p = -0.5 * np.sum((diff / self.sigma) ** 2)
        return self.weight * log_p

    def sample(
        self,
        rng: np.random.Generator,
        bounds: List[Tuple[float, float]],
    ) -> np.ndarray:
        """Sample theta from GNN-guided prior within bounds.

        For active edges: sample from N(mu_gnn, sigma²), clipped to bounds.
        For other parameters: uniform within bounds.
        Locked parameters: 0.
        """
        theta = np.zeros(len(bounds))

        for i, (lo, hi) in enumerate(bounds):
            if i in self.locked_indices or abs(hi - lo) < 1e-12:
                theta[i] = lo
                continue

            if self._mu_gnn is not None and i in ACTIVE_THETA_IDX:
                # GNN-guided: normal centered on prediction
                aij_idx = ACTIVE_THETA_IDX.index(i)
                mu = self._mu_gnn[aij_idx]
                theta[i] = np.clip(rng.normal(mu, self.sigma), lo, hi)
            else:
                # Uniform for non-active parameters
                theta[i] = rng.uniform(lo, hi)

        return theta

    @classmethod
    def load(
        cls,
        checkpoint: str = "gnn/data/checkpoints/best.pt",
        condition: str = "Dysbiotic_HOBIC",
        sigma: float = 1.0,
        weight: float = 1.0,
        phi_features: Optional[np.ndarray] = None,
        locked_indices: Optional[List[int]] = None,
        hidden: int = 128,
        n_layers: int = 4,
    ) -> "GNNPrior":
        """Factory: load GNN prior with condition-specific settings.

        If phi_features is None, tries to load from _runs/ MAP data.
        """
        project_root = Path(__file__).resolve().parent.parent.parent

        if locked_indices is None:
            bounds_path = project_root / "data_5species" / "model_config" / "prior_bounds.json"
            if bounds_path.exists():
                with open(bounds_path) as f:
                    cfg = json.load(f)
                strategy = cfg["strategies"].get(condition, {})
                locked_indices = strategy.get("locks", [])

        ckpt_path = project_root / checkpoint
        if not ckpt_path.exists():
            logger.warning(f"GNN checkpoint not found: {ckpt_path}. Using uniform prior.")
            return cls(sigma=sigma, weight=weight, locked_indices=locked_indices)

        return cls(
            model_path=str(ckpt_path),
            sigma=sigma,
            weight=weight,
            condition_phi=phi_features,
            locked_indices=locked_indices,
            hidden=hidden,
            n_layers=n_layers,
        )

    @classmethod
    def from_json(
        cls,
        json_path: str,
        sigma: float = 1.0,
        weight: float = 1.0,
        locked_indices: Optional[List[int]] = None,
        condition: Optional[str] = None,
    ) -> "GNNPrior":
        """Load GNN prior from JSON.

        Supports two formats:
        1. HMP format: {"gnn_prior_center": {"1": v, ...}, "gnn_prior_std": {...}}
        2. predict_for_tmcmc format: {"Condition_Name": {"a_ij_pred": [...], ...}}
        """
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"GNN prior JSON not found: {path}")

        with open(path) as f:
            data = json.load(f)

        # Format 2: predict_for_tmcmc.py output (condition-keyed dict)
        if condition and condition in data:
            cond_data = data[condition]
            mu_gnn = np.array(cond_data["a_ij_pred"], dtype=np.float64)
            # Apply locked flags
            if "a_ij_free" in cond_data:
                for k, free in enumerate(cond_data["a_ij_free"]):
                    if not free:
                        mu_gnn[k] = 0.0
            obj = cls(
                model_path=None,
                sigma=sigma,
                weight=weight,
                condition_phi=None,
                locked_indices=locked_indices or [],
            )
            obj._mu_gnn = mu_gnn
            logger.info(f"GNN prior from JSON [{condition}]: mu={mu_gnn}, sigma={sigma}")
            return obj

        # Format 1: HMP pipeline output
        center = data.get("gnn_prior_center", {})
        std_map = data.get("gnn_prior_std", {})
        if center:
            mu_gnn = np.array([float(center.get(str(i), 0.0)) for i in ACTIVE_THETA_IDX])
            sigma_vals = [float(std_map.get(str(i), sigma)) for i in ACTIVE_THETA_IDX]
            sigma_use = min(sigma_vals) if sigma_vals else sigma
        else:
            # Try auto-detect condition from keys
            cond_keys = [k for k in data if isinstance(data[k], dict) and "a_ij_pred" in data[k]]
            if cond_keys:
                cond_data = data[cond_keys[0]]
                mu_gnn = np.array(cond_data["a_ij_pred"], dtype=np.float64)
                sigma_use = sigma
                logger.warning(f"No condition specified, using first: {cond_keys[0]}")
            else:
                raise ValueError(f"Cannot parse GNN prior JSON: {path}")

        obj = cls(
            model_path=None,
            sigma=sigma_use,
            weight=weight,
            condition_phi=None,
            locked_indices=locked_indices or [],
        )
        obj._mu_gnn = mu_gnn
        logger.info(f"GNN prior from JSON: mu={mu_gnn}, sigma={sigma_use}")
        return obj
