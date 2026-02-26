#!/usr/bin/env python3
"""
Pre-compute GNN/MLP predictions for TMCMC prior.

v1: Saves 5 a_ij predictions per condition.
v2: Saves all 20 param predictions (μ, σ) per condition.

Saves JSON to be loaded by TMCMC (which runs in JAX env without torch).

Usage:
    python predict_for_tmcmc.py
    python predict_for_tmcmc.py --checkpoint data/checkpoints/best_gnn_v2.pt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from graph_builder import build_pyg_data, ACTIVE_THETA_IDX

RUNS_DIR = PROJECT_ROOT / "data_5species" / "_runs"
CONDITION_DIRS = {
    "Commensal_Static": "commensal_static_posterior",
    "Commensal_HOBIC": "commensal_hobic_posterior",
    "Dysbiotic_Static": "dysbiotic_static_posterior",
    "Dysbiotic_HOBIC": "dysbiotic_hobic_1000p",
}

PARAM_NAMES_SHORT = [
    "mu_So",
    "a_So_An",
    "mu_An",
    "b_So",
    "b_An",
    "mu_Vd",
    "a_An_Vd",
    "a_Vd_An",
    "b_Vd",
    "a_An_Fn",
    "a_So_Vd",
    "a_So_Fn",
    "mu_Fn",
    "a_Fn_An",
    "a_Vd_Fn",
    "b_Fn",
    "mu_Pg",
    "a_Fn_Vd",
    "a_Vd_Pg",
    "a_Fn_Pg",
]

EDGE_NAMES = ["a01(So→An)", "a02(So→Vd)", "a03(So→Fn)", "a24(Vd→Pg)", "a34(Fn→Pg)"]


def load_real_data(condition):
    """Load observed data (6 timepoints × 5 species)."""
    run_name = CONDITION_DIRS[condition]
    data_path = RUNS_DIR / run_name / "data.npy"
    if not data_path.exists():
        raise FileNotFoundError(f"No data.npy at {data_path}")
    return np.load(str(data_path))


def load_prior_bounds(condition):
    """Load prior bounds."""
    cfg_path = PROJECT_ROOT / "data_5species" / "model_config" / "prior_bounds.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    strategy = cfg["strategies"].get(condition, {})
    locks = set(strategy.get("locks", []))
    custom = strategy.get("bounds", {})
    bounds = np.zeros((20, 2))
    for i in range(20):
        if i in locks:
            bounds[i] = [0.0, 0.0]
        elif str(i) in custom:
            bounds[i] = custom[str(i)]
        else:
            bounds[i] = cfg["default_bounds"]
    return bounds, locks


def load_model(ckpt_path, device="cpu"):
    """Load model from checkpoint, auto-detecting version."""
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=True)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        version = ckpt.get("model_version", "v1")
        model_type = ckpt.get("model_type", "gnn")
        n_params = ckpt.get("n_params", 20)
        hidden = ckpt.get("hidden", 128)
        n_layers = ckpt.get("n_layers", 4)
        dropout = ckpt.get("dropout", 0.2)

        if version == "v2":
            if model_type == "mlp":
                from gnn_model import InteractionMLP

                model = InteractionMLP(
                    in_dim=15,
                    hidden=hidden,
                    n_params=n_params,
                    n_layers=n_layers,
                    dropout=dropout,
                )
            else:
                from gnn_model import InteractionGNNv2

                model = InteractionGNNv2(
                    in_dim=3,
                    hidden=hidden,
                    n_params=n_params,
                    n_layers=n_layers,
                    dropout=dropout,
                )
        else:
            from gnn_model import InteractionGNN

            model = InteractionGNN(
                in_dim=3,
                hidden=hidden,
                out_dim=5,
                n_layers=n_layers,
                dropout=dropout,
            )
    else:
        # Old format: v1 raw state dict
        state_dict = ckpt
        version = "v1"
        model_type = "gnn"
        from gnn_model import InteractionGNN

        model = InteractionGNN(in_dim=3, hidden=128, out_dim=5, n_layers=4, dropout=0.2)

    model.load_state_dict(state_dict)
    model.eval()
    return model, version, model_type


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    base = Path(__file__).parent

    # Auto-detect: prefer v2 checkpoint, fall back to v1
    if args.checkpoint:
        ckpt = Path(args.checkpoint)
    else:
        v2_ckpt = base / "data" / "checkpoints" / "best_gnn_v2.pt"
        v1_ckpt = base / "data" / "checkpoints" / "best.pt"
        ckpt = v2_ckpt if v2_ckpt.exists() else v1_ckpt

    out_path = Path(args.output) if args.output else base / "data" / "gnn_prior_predictions.json"

    model, version, model_type = load_model(ckpt)
    print(f"Loaded {model_type} model ({version}): {ckpt}")

    results = {}
    for cond, dirname in CONDITION_DIRS.items():
        try:
            obs = load_real_data(cond)  # (6, 5)
            phi_mean = obs.mean(axis=0).astype(np.float32)
            phi_std = obs.std(axis=0).astype(np.float32)
            phi_final = obs[-1].astype(np.float32)
        except FileNotFoundError:
            print(f"  SKIP {cond}: no data.npy")
            continue

        # Build graph and predict
        data = build_pyg_data(phi_mean, phi_std, phi_final, np.zeros(5, dtype=np.float32))
        data.batch = torch.zeros(5, dtype=torch.long)
        bounds, locks = load_prior_bounds(cond)

        with torch.no_grad():
            raw_pred = model(data)

        if version == "v2":
            pred_np = raw_pred[0].numpy()  # (n_params, 2)
            mu = pred_np[:, 0]
            sigma = np.exp(pred_np[:, 1])
            n_params = len(mu)

            # Clip mu to bounds, mark locked
            free_flags = []
            for i in range(n_params):
                lo, hi = bounds[i] if i < len(bounds) else (-1.0, 1.0)
                if i in locks or abs(hi - lo) < 1e-12:
                    mu[i] = lo
                    sigma[i] = 1e6  # effectively uniform
                    free_flags.append(False)
                else:
                    mu[i] = np.clip(mu[i], lo, hi)
                    free_flags.append(True)

            results[cond] = {
                "theta_mu": mu.tolist(),
                "theta_sigma": sigma.tolist(),
                "theta_free": free_flags,
                "param_names": PARAM_NAMES_SHORT[:n_params],
                "model_version": "v2",
                "model_type": model_type,
                # v1 compat fields
                "a_ij_pred": mu[ACTIVE_THETA_IDX].tolist(),
                "a_ij_free": [free_flags[i] for i in ACTIVE_THETA_IDX],
                "active_theta_idx": ACTIVE_THETA_IDX,
                "edge_names": EDGE_NAMES,
                "phi_mean": phi_mean.tolist(),
                "phi_std": phi_std.tolist(),
                "phi_final": phi_final.tolist(),
            }

            # Print summary
            n_free = sum(free_flags)
            print(f"  {cond} (v2, {n_free} free params):")
            for i in range(n_params):
                if free_flags[i]:
                    name = PARAM_NAMES_SHORT[i] if i < len(PARAM_NAMES_SHORT) else f"θ[{i}]"
                    print(f"    {name:>12s}: μ={mu[i]:+.3f}, σ={sigma[i]:.3f}")

        else:
            pred = raw_pred[0].numpy()  # (5,)
            free_flags = []
            for k, tidx in enumerate(ACTIVE_THETA_IDX):
                lo, hi = bounds[tidx]
                if abs(hi - lo) < 1e-12:
                    pred[k] = lo
                    free_flags.append(False)
                else:
                    pred[k] = np.clip(pred[k], lo, hi)
                    free_flags.append(True)

            results[cond] = {
                "a_ij_pred": pred.tolist(),
                "a_ij_free": free_flags,
                "active_theta_idx": ACTIVE_THETA_IDX,
                "edge_names": EDGE_NAMES,
                "model_version": "v1",
                "phi_mean": phi_mean.tolist(),
                "phi_std": phi_std.tolist(),
                "phi_final": phi_final.tolist(),
            }
            print(f"  {cond} (v1): a_ij = {[f'{v:.3f}' for v in pred]}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
