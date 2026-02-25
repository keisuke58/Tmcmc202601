#!/usr/bin/env python3
"""
Train GNN for a_ij prediction (Project B, Issue #39).

Usage:
  python train.py --data data/train_gnn_N1000.npz --epochs 1000
  python train.py eval --checkpoint data/checkpoints/best.pt --data data/train_gnn_N1000.npz
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import random_split

from graph_builder import build_pyg_data, dataset_to_pyg_list, ACTIVE_EDGES
from gnn_model import InteractionGNN

try:
    from torch_geometric.data import Batch
    from torch_geometric.loader import DataLoader as PyGDataLoader
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False

EDGE_NAMES = ["a01(So→An)", "a02(So→Vd)", "a03(So→Fn)", "a24(Vd→Pg)", "a34(Fn→Pg)"]


def load_dataset(path):
    data = np.load(path)
    return {k: data[k] for k in data.files}


def train_epoch(model, loader, optimizer, device, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    n_batch = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        target = batch.a_ij_active.to(device).view(batch.num_graphs, 5)
        loss = torch.nn.functional.mse_loss(pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
        n_batch += 1
    return total_loss / max(n_batch, 1)


def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    n_batch = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            target = batch.a_ij_active.to(device).view(batch.num_graphs, 5)
            loss = torch.nn.functional.mse_loss(pred, target)
            total_loss += loss.item()
            n_batch += 1
    return total_loss / max(n_batch, 1)


def main_train(args):
    if not PYG_AVAILABLE:
        print("pip install torch torch-geometric")
        return

    data = load_dataset(args.data)
    pyg_list = dataset_to_pyg_list(data)
    n = len(pyg_list)
    n_val = int(n * 0.15)
    n_train = n - n_val
    train_list, val_list = torch.utils.data.random_split(
        pyg_list, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_list = [pyg_list[i] for i in train_list.indices]
    val_list = [pyg_list[i] for i in val_list.indices]

    train_loader = PyGDataLoader(train_list, batch_size=args.batch_size, shuffle=True)
    val_loader = PyGDataLoader(val_list, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InteractionGNN(
        in_dim=3, hidden=args.hidden, out_dim=5, n_layers=args.layers, dropout=args.dropout
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    out_dir = Path(args.checkpoint).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    patience_counter = 0

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
        scheduler.step()

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), args.checkpoint)
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch+1:4d}  train={train_loss:.6f}  val={val_loss:.6f}  best={best_val:.6f}  lr={lr_now:.2e}", flush=True)

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1} (patience={args.patience})")
            break

    print(f"\nBest val MSE: {best_val:.6f}")
    print(f"Saved to {args.checkpoint}")


def main_eval(args):
    data = load_dataset(args.data)
    device = torch.device("cpu")
    model = InteractionGNN(
        in_dim=3, hidden=args.hidden, out_dim=5, n_layers=args.layers, dropout=args.dropout
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()

    pyg_list = dataset_to_pyg_list(data)
    batch = Batch.from_data_list(pyg_list)
    with torch.no_grad():
        pred = model(batch)
    target = batch.a_ij_active.reshape(-1, 5)

    mse = torch.nn.functional.mse_loss(pred, target).item()
    mae = torch.nn.functional.l1_loss(pred, target).item()

    # Per-output metrics
    print(f"Overall  MSE={mse:.4f}  MAE={mae:.4f}")
    print(f"{'Edge':<16s} {'MSE':>8s} {'MAE':>8s} {'R²':>8s} {'pred_std':>9s} {'tgt_std':>9s}")
    print("-" * 60)
    for j in range(5):
        p, t = pred[:, j], target[:, j]
        mse_j = ((p - t) ** 2).mean().item()
        mae_j = (p - t).abs().mean().item()
        ss_res = ((t - p) ** 2).sum().item()
        ss_tot = ((t - t.mean()) ** 2).sum().item()
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        print(f"{EDGE_NAMES[j]:<16s} {mse_j:8.4f} {mae_j:8.4f} {r2:8.4f} {p.std().item():9.4f} {t.std().item():9.4f}")

    # Show worst / best predictions
    errs = ((pred - target) ** 2).sum(dim=1)
    best_idx = errs.argmin().item()
    worst_idx = errs.argmax().item()
    print(f"\nBest sample  #{best_idx}: pred={pred[best_idx].numpy().round(3)}, tgt={target[best_idx].numpy().round(3)}")
    print(f"Worst sample #{worst_idx}: pred={pred[worst_idx].numpy().round(3)}, tgt={target[worst_idx].numpy().round(3)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/train_gnn_N10000.npz")
    parser.add_argument("--checkpoint", type=str, default="data/checkpoints/best.pt")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("mode", nargs="?", default="train", choices=["train", "eval"])
    args = parser.parse_args()

    base = Path(__file__).parent
    args.data = str(base / args.data) if not Path(args.data).is_absolute() else args.data
    args.checkpoint = str(base / args.checkpoint) if not Path(args.checkpoint).is_absolute() else args.checkpoint

    if args.mode == "eval":
        main_eval(args)
    else:
        main_train(args)


if __name__ == "__main__":
    main()
