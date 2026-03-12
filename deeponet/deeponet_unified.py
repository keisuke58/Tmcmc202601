#!/usr/bin/env python3
"""
A4: Unified DeepONet — single model for all 4 conditions.

Architecture identical to deeponet_hamilton.py but with theta_dim=24
(20 ODE params + 4-dim condition one-hot encoding).

Usage:
  # Generate unified data
  python generate_unified_data.py --max-per-condition 10000

  # Train
  python deeponet_unified.py train --data data/train_Unified_N40000.npz

  # Evaluate
  python deeponet_unified.py eval --checkpoint checkpoints_unified/best.eqx \
      --data data/train_Unified_N40000.npz

  # Predict for a specific condition
  python deeponet_unified.py predict --checkpoint checkpoints_unified/best.eqx \
      --condition Dysbiotic_HOBIC
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Re-use all model classes and training code from deeponet_hamilton
sys.path.insert(0, str(Path(__file__).parent))
from deeponet_hamilton import (
    DeepONet,
    load_data,
    train,
    evaluate,
)

CONDITION_ONEHOT = {
    "Commensal_Static": np.array([1, 0, 0, 0], dtype=np.float32),
    "Commensal_HOBIC": np.array([0, 1, 0, 0], dtype=np.float32),
    "Dysbiotic_Static": np.array([0, 0, 1, 0], dtype=np.float32),
    "Dysbiotic_HOBIC": np.array([0, 0, 0, 1], dtype=np.float32),
}


class UnifiedDeepONetSurrogate:
    """
    Wraps unified DeepONet (theta_dim=24) as a condition-aware surrogate.

    Usage:
        surrogate = UnifiedDeepONetSurrogate(checkpoint, norm_stats, condition)
        phi = surrogate.predict(theta_20d)  # (T, 5)
    """

    def __init__(
        self,
        checkpoint: str,
        norm_stats_path: str,
        condition: str,
        n_species: int = 5,
        p: int = 64,
        hidden: int = 128,
        n_layers: int = 3,
        n_time_out: int = 100,
    ):
        import jax
        import jax.numpy as jnp
        import equinox as eqx

        self.condition = condition
        self.onehot = CONDITION_ONEHOT[condition]

        # Auto-detect arch from config.json
        config_path = Path(checkpoint).parent / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
            p = cfg.get("p", p)
            hidden = cfg.get("hidden", hidden)
            n_layers = cfg.get("n_layers", n_layers)

        # Load model with theta_dim=24
        key = jax.random.PRNGKey(0)
        self.model = DeepONet(
            theta_dim=24,
            n_species=n_species,
            p=p,
            hidden=hidden,
            n_layers=n_layers,
            key=key,
        )
        self.model = eqx.tree_deserialise_leaves(checkpoint, self.model)

        # Load normalization (24-dim bounds)
        stats = np.load(norm_stats_path)
        self.theta_lo = stats["theta_lo"]  # (24,)
        self.theta_width = stats["theta_width"]  # (24,)
        self.t_min = float(stats["t_min"])
        self.t_max = float(stats["t_max"])

        # Time grid
        self.n_time = n_time_out
        self.t_norm = jnp.linspace(0, 1, n_time_out, dtype=jnp.float32)

        # JIT
        self._predict_jit = jax.jit(self._predict_single)

        # Warmup
        dummy = jnp.zeros(24, dtype=jnp.float32)
        _ = self._predict_jit(dummy)

    def _predict_single(self, theta_norm_24):
        import jax.numpy as jnp

        return self.model.predict_trajectory(theta_norm_24, self.t_norm)

    def predict(self, theta_raw_20: np.ndarray) -> np.ndarray:
        """
        Predict species trajectory from raw 20-dim θ (auto-appends condition).

        Args:
            theta_raw_20: (20,) raw parameter vector

        Returns:
            phi_pred: (n_time, 5) predicted species fractions
        """
        import jax.numpy as jnp

        # Extend to 24-dim: append condition one-hot
        theta_24 = np.concatenate([theta_raw_20, self.onehot])

        # Normalize
        theta_norm = (theta_24 - self.theta_lo) / self.theta_width
        theta_norm = jnp.array(theta_norm, dtype=jnp.float32)

        phi = self._predict_jit(theta_norm)
        return np.array(phi)

    def predict_at_indices(self, theta_raw_20: np.ndarray, idx_sparse: np.ndarray) -> np.ndarray:
        """Predict and extract at sparse observation indices."""
        phi_full = self.predict(theta_raw_20)
        return phi_full[idx_sparse]


def main():
    parser = argparse.ArgumentParser(description="A4: Unified DeepONet (4 conditions → 1 model)")
    sub = parser.add_subparsers(dest="command")

    # Train
    p_train = sub.add_parser("train")
    p_train.add_argument("--data", required=True, help="Unified .npz data path")
    p_train.add_argument("--epochs", type=int, default=500)
    p_train.add_argument("--batch-size", type=int, default=256)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--p", type=int, default=64)
    p_train.add_argument("--hidden", type=int, default=128)
    p_train.add_argument("--n-layers", type=int, default=3)
    p_train.add_argument("--checkpoint-dir", default="checkpoints_unified")

    # Eval
    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--checkpoint", required=True)
    p_eval.add_argument("--data", required=True)

    # Predict
    p_pred = sub.add_parser("predict")
    p_pred.add_argument("--checkpoint", required=True)
    p_pred.add_argument("--norm-stats", required=True)
    p_pred.add_argument("--condition", required=True, choices=list(CONDITION_ONEHOT.keys()))

    args = parser.parse_args()

    if args.command == "train":
        print("=" * 60)
        print("  A4: Training Unified DeepONet (theta_dim=24)")
        print("=" * 60)
        model, stats = train(
            data_path=args.data,
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            p=args.p,
            hidden=args.hidden,
            n_layers=args.n_layers,
            checkpoint_dir=args.checkpoint_dir,
        )
        print("\nUnified model trained. theta_dim=24 (20 params + 4 condition one-hot)")

    elif args.command == "eval":
        # Auto-detect config
        config_path = Path(args.checkpoint).parent / "config.json"
        p, hidden, n_layers = 64, 128, 3
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
            p = cfg.get("p", p)
            hidden = cfg.get("hidden", hidden)
            n_layers = cfg.get("n_layers", n_layers)
        evaluate(args.checkpoint, args.data, p=p, hidden=hidden, n_layers=n_layers)

    elif args.command == "predict":
        # Demo prediction for a specific condition
        norm_stats = args.norm_stats
        surrogate = UnifiedDeepONetSurrogate(args.checkpoint, norm_stats, args.condition)

        # Random theta from prior
        bounds_path = (
            Path(__file__).parent.parent / "data_5species" / "model_config" / "prior_bounds.json"
        )
        import json as json_mod

        with open(bounds_path) as f:
            prior = json_mod.load(f)
        theta_20 = np.array(
            [0.5 * (b["low"] + b["high"]) for b in prior["parameters"][:20]], dtype=np.float32
        )

        t0 = time.time()
        phi = surrogate.predict(theta_20)
        dt = time.time() - t0

        print(f"\nCondition: {args.condition}")
        print(f"Prediction time: {dt*1000:.1f} ms")
        print(f"Output shape: {phi.shape}")
        print(f"Species at t=0: {phi[0]}")
        print(f"Species at t=T: {phi[-1]}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
