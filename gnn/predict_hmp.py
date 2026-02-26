#!/usr/bin/env python3
"""
Phase 2: HMP 組成データ → GNN → a_ij 予測 → TMCMC prior.

Usage:
  # HMP CSV から予測 (R script 出力を想定)
  python predict_hmp.py --input data/hmp_oral/species_abundance.csv --checkpoint data/checkpoints/best.pt

  # prior_bounds 用 JSON 出力
  python predict_hmp.py --input data/hmp_oral/species_abundance.csv --output-prior data/hmp_oral/gnn_prior.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from graph_builder import build_pyg_data
from gnn_model import InteractionGNN

try:
    from torch_geometric.data import Batch

    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False

# theta index for active edges
ACTIVE_THETA_IDX = [1, 10, 11, 18, 19]
EDGE_NAMES = ["a01(So→An)", "a02(So→Vd)", "a03(So→Fn)", "a24(Vd→Pg)", "a34(Fn→Pg)"]


def load_hmp_abundance(csv_path: str) -> np.ndarray:
    """Load species_abundance.csv from R script. Returns (N, 5) in order [So, An, Vd, Fn, Pg]."""
    try:
        import pandas as pd

        df = pd.read_csv(csv_path, index_col=0)
        cols = ["S_oralis", "A_naeslundii", "V_dispar", "F_nucleatum", "P_gingivalis"]
        data = df[[c for c in cols if c in df.columns]].values
    except ImportError:
        data = np.genfromtxt(csv_path, delimiter=",", skip_header=1, usecols=range(1, 6))
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return np.asarray(data, dtype=np.float32)


def hmp_to_gnn_input(phi: np.ndarray) -> tuple:
    """
    HMP 単一時点組成 → GNN 入力形式.
    phi: (N, 5) volume fraction (sum=1 per row)
    Returns: phi_mean, phi_std, phi_final (all Nx5)
    """
    phi = np.asarray(phi, dtype=np.float32)
    if phi.ndim == 1:
        phi = phi.reshape(1, -1)
    # 単一時点: mean=final=phi, std=0 (小さい正数で除算安定)
    phi_mean = phi
    phi_std = np.full_like(phi, 1e-6)
    phi_final = phi
    return phi_mean, phi_std, phi_final


def predict_aij(model, phi_mean, phi_std, phi_final, device="cpu"):
    """GNN で a_ij を予測。Returns (N, 5)."""
    data_list = []
    for i in range(len(phi_mean)):
        d = build_pyg_data(phi_mean[i], phi_std[i], phi_final[i], np.zeros(5))
        d.a_ij_active = torch.zeros(5)
        data_list.append(d)
    batch = Batch.from_data_list(data_list)
    with torch.no_grad():
        pred = model(batch.to(device))
    return pred.cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="species_abundance.csv path")
    parser.add_argument("--checkpoint", type=str, default="data/checkpoints/best.pt")
    parser.add_argument("--output-prior", type=str, default=None, help="prior_bounds JSON output")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    args = parser.parse_args()

    base = Path(__file__).parent
    input_path = base / args.input if not Path(args.input).is_absolute() else Path(args.input)
    ckpt_path = (
        base / args.checkpoint if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    )

    if not input_path.exists():
        print(
            f"Error: {input_path} not found. Run: Rscript scripts/extract_hmp_oral.R data/hmp_oral"
        )
        return 1

    phi = load_hmp_abundance(str(input_path))
    phi_mean, phi_std, phi_final = hmp_to_gnn_input(phi)

    device = torch.device("cpu")
    model = InteractionGNN(
        in_dim=3, hidden=args.hidden, out_dim=5, n_layers=args.layers, dropout=args.dropout
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    pred = predict_aij(model, phi_mean, phi_std, phi_final, device)

    # サンプル平均を prior の中心として使用
    aij_mean = np.mean(pred, axis=0)
    aij_std = np.std(pred, axis=0)
    # 学習データの範囲外はクリップ (prior bounds に合わせる)
    aij_mean = np.clip(aij_mean, -3.0, 5.0)

    print(f"Loaded {len(phi)} HMP oral samples from {input_path}")
    print("\nGNN predicted a_ij (mean over samples):")
    for j, name in enumerate(EDGE_NAMES):
        print(f"  {name}: {aij_mean[j]:.4f} ± {aij_std[j]:.4f}")
    print(f"\nPrior center (for TMCMC): theta indices {ACTIVE_THETA_IDX}")
    print(f"  aij_mean = {aij_mean.tolist()}")

    if args.output_prior:
        out_path = (
            base / args.output_prior
            if not Path(args.output_prior).is_absolute()
            else Path(args.output_prior)
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        prior = {
            "gnn_prior_center": {str(ACTIVE_THETA_IDX[j]): float(aij_mean[j]) for j in range(5)},
            "gnn_prior_std": {
                str(ACTIVE_THETA_IDX[j]): float(max(aij_std[j], 0.5)) for j in range(5)
            },
            "note": "GNN predictions clipped to [-3, 5]. Use as prior center for TMCMC.",
            "n_samples": int(len(phi)),
        }
        with open(out_path, "w") as f:
            json.dump(prior, f, indent=2)
        print(f"\nSaved prior to {out_path}")

    return 0


if __name__ == "__main__":
    exit(main())
