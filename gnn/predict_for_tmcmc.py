#!/usr/bin/env python3
"""
Pre-compute GNN a_ij predictions for TMCMC prior.

Saves JSON with predicted a_ij for each condition, to be loaded by
gradient_tmcmc_nuts.py (which runs in JAX env without torch).

Usage:
    python predict_for_tmcmc.py
    python predict_for_tmcmc.py --checkpoint data/checkpoints/best.pt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from gnn_model import InteractionGNN
from graph_builder import build_pyg_data, ACTIVE_THETA_IDX

RUNS_DIR = PROJECT_ROOT / "data_5species" / "_runs"
CONDITION_DIRS = {
    "Commensal_Static": "commensal_static_posterior",
    "Commensal_HOBIC": "commensal_hobic_posterior",
    "Dysbiotic_Static": "dysbiotic_static_posterior",
    "Dysbiotic_HOBIC": "dysbiotic_hobic_1000p",
}


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
    return bounds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    base = Path(__file__).parent
    ckpt = Path(args.checkpoint) if args.checkpoint else base / "data" / "checkpoints" / "best.pt"
    out_path = Path(args.output) if args.output else base / "data" / "gnn_prior_predictions.json"

    # Load model
    model = InteractionGNN(
        in_dim=3, hidden=args.hidden, out_dim=5, n_layers=args.layers, dropout=args.dropout
    )
    model.load_state_dict(torch.load(str(ckpt), map_location="cpu", weights_only=True))
    model.eval()
    print(f"Loaded GNN: {ckpt}")

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

        with torch.no_grad():
            pred = model(data)[0].numpy()

        # Clip to valid bounds; set locked params to their locked value
        bounds = load_prior_bounds(cond)
        free_flags = []
        for k, tidx in enumerate(ACTIVE_THETA_IDX):
            lo, hi = bounds[tidx]
            if abs(hi - lo) < 1e-12:
                pred[k] = lo  # locked
                free_flags.append(False)
            else:
                pred[k] = np.clip(pred[k], lo, hi)
                free_flags.append(True)

        results[cond] = {
            "a_ij_pred": pred.tolist(),
            "a_ij_free": free_flags,
            "active_theta_idx": ACTIVE_THETA_IDX,
            "edge_names": ["a01(So→An)", "a02(So→Vd)", "a03(So→Fn)", "a24(Vd→Pg)", "a34(Fn→Pg)"],
            "phi_mean": phi_mean.tolist(),
            "phi_std": phi_std.tolist(),
            "phi_final": phi_final.tolist(),
        }

        print(f"  {cond}: a_ij = {[f'{v:.3f}' for v in pred]}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
