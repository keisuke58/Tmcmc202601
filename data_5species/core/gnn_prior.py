"""
GNN-informed prior for TMCMC (Project B, Issue #39).

v1: 5 active edges with fixed σ
v2: all 20 params with per-param adaptive σ (heteroscedastic)

Wraps a trained InteractionGNN/v2 model to provide:
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
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Active edges matching graph_builder.py (v1 subset)
ACTIVE_THETA_IDX = [1, 10, 11, 18, 19]

# Try importing torch (optional dependency)
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class GNNPrior:
    """GNN-based informative prior for TMCMC.

    v1 mode (backward compat):
      p(theta) ∝ U(bounds) × N(theta_active | mu_gnn, sigma²)
      Fixed sigma for all 5 active edges.

    v2 mode:
      p(theta) ∝ U(bounds) × Π_i N(theta_i | mu_i, sigma_i²)
      Per-parameter adaptive sigma from heteroscedastic model.
      Applies to all non-locked parameters.

    Parameters
    ----------
    model_path : str or Path
        Path to trained checkpoint (.pt file)
    sigma : float
        Global sigma override (v1 mode, or fallback for v2).
    weight : float
        Weight of GNN prior in log-posterior. 0 = ignore GNN.
    condition_phi : np.ndarray, optional
        Composition features (5,3) for the target condition.
    locked_indices : list of int, optional
        Parameter indices locked to 0 (condition-specific).
    hidden : int
        Hidden dimension of the GNN model.
    n_layers : int
        Number of GCN layers.
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
        version: str = "v1",
    ):
        self.sigma = sigma
        self.weight = weight
        self.locked_indices = set(locked_indices or [])
        self.version = version

        # v1: (5,) means for active edges
        self._mu_gnn = None

        # v2: (n_params,) means and sigmas for all predicted params
        self._mu_all = None  # (20,) predicted means
        self._sigma_all = None  # (20,) predicted sigmas
        self._predicted_mask = None  # (20,) bool: which params have predictions

        self._model = None

        if model_path is not None and TORCH_AVAILABLE:
            self._load_model(model_path, hidden, n_layers)
            if condition_phi is not None:
                if self.version == "v2":
                    self._predict_v2(condition_phi)
                else:
                    self._mu_gnn = self._predict_v1(condition_phi)
                    logger.info(f"GNN prior v1: mu={self._mu_gnn}, sigma={sigma}")

    def _load_model(self, model_path: str, hidden: int, n_layers: int):
        """Load trained model from checkpoint."""
        import sys

        gnn_dir = Path(__file__).resolve().parent.parent.parent / "gnn"
        sys.path.insert(0, str(gnn_dir))

        ckpt = torch.load(model_path, map_location="cpu", weights_only=True)

        # New format: dict with metadata
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
            saved_version = ckpt.get("model_version", "v1")
            model_type = ckpt.get("model_type", "gnn")
            n_params = ckpt.get("n_params", 20)
            hidden = ckpt.get("hidden", hidden)
            n_layers = ckpt.get("n_layers", n_layers)
            dropout = ckpt.get("dropout", 0.2)
            self.version = saved_version

            if saved_version == "v2":
                if model_type == "mlp":
                    from gnn_model import InteractionMLP

                    self._model = InteractionMLP(
                        in_dim=15,
                        hidden=hidden,
                        n_params=n_params,
                        n_layers=n_layers,
                        dropout=dropout,
                    )
                else:
                    from gnn_model import InteractionGNNv2

                    self._model = InteractionGNNv2(
                        in_dim=3,
                        hidden=hidden,
                        n_params=n_params,
                        n_layers=n_layers,
                        dropout=dropout,
                    )
            else:
                from gnn_model import InteractionGNN

                self._model = InteractionGNN(
                    in_dim=3,
                    hidden=hidden,
                    out_dim=5,
                    n_layers=n_layers,
                    dropout=dropout,
                )
        else:
            # Old format: raw state dict (v1)
            state_dict = ckpt
            from gnn_model import InteractionGNN

            self._model = InteractionGNN(in_dim=3, hidden=hidden, out_dim=5, n_layers=n_layers)

        self._model.load_state_dict(state_dict)
        self._model.eval()
        logger.info(f"Loaded GNN model ({self.version}) from {model_path}")

    def _build_pyg_input(self, phi_features: np.ndarray):
        """Build PyG data from phi features."""
        import sys

        gnn_dir = Path(__file__).resolve().parent.parent.parent / "gnn"
        sys.path.insert(0, str(gnn_dir))
        from graph_builder import build_pyg_data

        if phi_features.ndim == 1 and phi_features.shape[0] == 5:
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
        return data

    def _predict_v1(self, phi_features: np.ndarray) -> np.ndarray:
        """Predict 5 a_ij from composition features (v1)."""
        if self._model is None:
            raise RuntimeError("GNN model not loaded")
        data = self._build_pyg_input(phi_features)
        with torch.no_grad():
            pred = self._model(data)
        return pred.squeeze().numpy()

    def _predict_v2(self, phi_features: np.ndarray):
        """Predict all params with uncertainty (v2)."""
        if self._model is None:
            raise RuntimeError("GNN model not loaded")
        data = self._build_pyg_input(phi_features)
        with torch.no_grad():
            pred = self._model(data)  # (1, n_params, 2)
        pred = pred.squeeze(0).numpy()  # (n_params, 2)

        n_params = pred.shape[0]
        self._mu_all = pred[:, 0]
        self._sigma_all = np.exp(pred[:, 1])

        # Build mask: predict all non-locked params
        self._predicted_mask = np.ones(n_params, dtype=bool)
        for i in self.locked_indices:
            if i < n_params:
                self._predicted_mask[i] = False

        # Also set v1 compat
        self._mu_gnn = self._mu_all[ACTIVE_THETA_IDX]

        logger.info(
            f"GNN prior v2: {self._predicted_mask.sum()} predicted params, "
            f"mean σ={self._sigma_all[self._predicted_mask].mean():.4f}"
        )

    def predict_aij(self, phi: np.ndarray) -> np.ndarray:
        """Public API: predict a_ij from composition. Returns (5,) array."""
        return self._predict_v1(phi)

    def set_condition(self, phi_features: np.ndarray):
        """Update prediction for a new condition."""
        if self._model is not None:
            if self.version == "v2":
                self._predict_v2(phi_features)
            else:
                self._mu_gnn = self._predict_v1(phi_features)

    def log_density(self, theta: np.ndarray) -> float:
        """Compute log prior density for theta vector.

        v1: Gaussian on 5 active edges with fixed sigma.
        v2: Gaussian on all predicted params with per-param sigma.
        """
        if self.weight == 0.0:
            return 0.0

        if self.version == "v2" and self._mu_all is not None:
            # v2: per-param heteroscedastic prior
            log_p = 0.0
            for i in range(len(self._mu_all)):
                if not self._predicted_mask[i]:
                    continue
                if i >= len(theta):
                    continue
                diff = theta[i] - self._mu_all[i]
                s = self._sigma_all[i]
                log_p -= 0.5 * (diff / s) ** 2
            return self.weight * log_p

        # v1 fallback
        if self._mu_gnn is None:
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

        v1: Normal on active edges, uniform elsewhere.
        v2: Normal with per-param sigma on all predicted params.
        """
        theta = np.zeros(len(bounds))

        for i, (lo, hi) in enumerate(bounds):
            if i in self.locked_indices or abs(hi - lo) < 1e-12:
                theta[i] = lo
                continue

            if self.version == "v2" and self._mu_all is not None:
                if i < len(self._mu_all) and self._predicted_mask[i]:
                    mu = self._mu_all[i]
                    s = self._sigma_all[i]
                    theta[i] = np.clip(rng.normal(mu, s), lo, hi)
                else:
                    theta[i] = rng.uniform(lo, hi)
            elif self._mu_gnn is not None and i in ACTIVE_THETA_IDX:
                aij_idx = ACTIVE_THETA_IDX.index(i)
                mu = self._mu_gnn[aij_idx]
                theta[i] = np.clip(rng.normal(mu, self.sigma), lo, hi)
            else:
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

        Auto-detects v1/v2 from checkpoint metadata.
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
        """Load GNN prior from pre-computed JSON.

        Supports formats:
        1. v1 HMP: {"gnn_prior_center": {"1": v, ...}, "gnn_prior_std": {...}}
        2. v1 predict_for_tmcmc: {"Condition": {"a_ij_pred": [...], ...}}
        3. v2 predict_for_tmcmc: {"Condition": {"theta_mu": [...], "theta_sigma": [...], ...}}
        """
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"GNN prior JSON not found: {path}")

        with open(path) as f:
            data = json.load(f)

        # Format 3 (v2): condition-keyed with theta_mu/theta_sigma
        if condition and condition in data:
            cond_data = data[condition]

            if "theta_mu" in cond_data:
                # v2 format
                obj = cls(
                    model_path=None,
                    sigma=sigma,
                    weight=weight,
                    condition_phi=None,
                    locked_indices=locked_indices or [],
                    version="v2",
                )
                mu = np.array(cond_data["theta_mu"], dtype=np.float64)
                sig = np.array(cond_data["theta_sigma"], dtype=np.float64)
                n_params = len(mu)
                obj._mu_all = mu
                obj._sigma_all = sig
                obj._predicted_mask = np.ones(n_params, dtype=bool)
                for i in locked_indices or []:
                    if i < n_params:
                        obj._predicted_mask[i] = False
                # Also set v1 compat
                obj._mu_gnn = mu[ACTIVE_THETA_IDX] if n_params >= 20 else None
                logger.info(
                    f"GNN prior v2 from JSON [{condition}]: "
                    f"{obj._predicted_mask.sum()} params, mean σ={sig[obj._predicted_mask].mean():.4f}"
                )
                return obj

            # Format 2 (v1): a_ij_pred
            if "a_ij_pred" in cond_data:
                mu_gnn = np.array(cond_data["a_ij_pred"], dtype=np.float64)
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
                logger.info(f"GNN prior v1 from JSON [{condition}]: mu={mu_gnn}, sigma={sigma}")
                return obj

        # Format 1: HMP pipeline output
        center = data.get("gnn_prior_center", {})
        std_map = data.get("gnn_prior_std", {})
        if center:
            mu_gnn = np.array([float(center.get(str(i), 0.0)) for i in ACTIVE_THETA_IDX])
            sigma_vals = [float(std_map.get(str(i), sigma)) for i in ACTIVE_THETA_IDX]
            sigma_use = min(sigma_vals) if sigma_vals else sigma
        else:
            # Auto-detect condition
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
