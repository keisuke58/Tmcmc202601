#!/usr/bin/env python3
"""
Train GNN / MLP for parameter prediction (Project B, Issue #39).

v1 mode (original): GCN → 5 a_ij (MSE loss)
v2 mode: GCN → all 20 params with heteroscedastic (μ, σ) (NLL loss)
mlp mode: MLP ablation with same v2 output

Usage:
  # v1 (backward compatible)
  python train.py --data data/train_gnn_N1000.npz --epochs 1000

  # v2: 20-param heteroscedastic GNN
  python train.py --model-version v2 --data data/train_gnn_N10000.npz

  # MLP ablation
  python train.py --model-version v2 --model-type mlp --data data/train_gnn_N10000.npz

  # Eval
  python train.py eval --model-version v2 --data data/train_gnn_N10000.npz
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from graph_builder import dataset_to_pyg_list
from gnn_model import InteractionGNN, InteractionGNNv2, InteractionMLP, heteroscedastic_nll

try:
    from torch_geometric.data import Batch
    from torch_geometric.loader import DataLoader as PyGDataLoader

    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False

EDGE_NAMES = ["a01(So→An)", "a02(So→Vd)", "a03(So→Fn)", "a24(Vd→Pg)", "a34(Fn→Pg)"]

PARAM_NAMES_SHORT = [
    "μ_So",
    "a_So→An",
    "μ_An",
    "b_So",
    "b_An",
    "μ_Vd",
    "a_An→Vd",
    "a_Vd→An",
    "b_Vd",
    "a_An→Fn",
    "a_So→Vd",
    "a_So→Fn",
    "μ_Fn",
    "a_Fn→An",
    "a_Vd→Fn",
    "b_Fn",
    "μ_Pg",
    "a_Fn→Vd",
    "a_Vd→Pg",
    "a_Fn→Pg",
]


def load_dataset(path):
    data = np.load(path)
    return {k: data[k] for k in data.files}


def build_model(args, device):
    """Build model based on args."""
    if args.model_version == "v1":
        model = InteractionGNN(
            in_dim=3,
            hidden=args.hidden,
            out_dim=5,
            n_layers=args.layers,
            dropout=args.dropout,
        )
    elif args.model_type == "mlp":
        model = InteractionMLP(
            in_dim=15,
            hidden=args.hidden,
            n_params=args.n_params,
            n_layers=args.layers,
            dropout=args.dropout,
        )
    else:
        model = InteractionGNNv2(
            in_dim=3,
            hidden=args.hidden,
            n_params=args.n_params,
            n_layers=args.layers,
            dropout=args.dropout,
        )
    return model.to(device)


# ---------------------------------------------------------------------------
# v1 training (backward compat)
# ---------------------------------------------------------------------------


def train_epoch_v1(model, loader, optimizer, device, grad_clip=1.0):
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


def validate_v1(model, loader, device):
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


# ---------------------------------------------------------------------------
# v2 training (heteroscedastic, all 20 params)
# ---------------------------------------------------------------------------


def train_epoch_v2(model, loader, optimizer, device, grad_clip=1.0, lock_mask=None):
    model.train()
    total_loss = 0.0
    n_batch = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)  # (B, n_params, 2)
        target = batch.theta_all.to(device).view(batch.num_graphs, -1)  # (B, 20)
        # Build mask: exclude locked params (all zeros in target)
        if lock_mask is not None:
            mask = lock_mask.unsqueeze(0).expand_as(target)
        else:
            mask = None
        loss = heteroscedastic_nll(pred, target, mask=mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
        n_batch += 1
    return total_loss / max(n_batch, 1)


def validate_v2(model, loader, device, lock_mask=None):
    model.eval()
    total_loss = 0.0
    n_batch = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            target = batch.theta_all.to(device).view(batch.num_graphs, -1)
            if lock_mask is not None:
                mask = lock_mask.unsqueeze(0).expand_as(target)
            else:
                mask = None
            loss = heteroscedastic_nll(pred, target, mask=mask)
            total_loss += loss.item()
            n_batch += 1
    return total_loss / max(n_batch, 1)


def detect_lock_mask(data_dict, n_params=20):
    """Detect locked parameters from training data (all-zero columns)."""
    theta = data_dict["theta"]  # (N, 20)
    # A param is locked if all samples have the same value (std ≈ 0)
    stds = theta.std(axis=0)
    mask = torch.tensor(stds > 1e-8, dtype=torch.bool)
    locked = [i for i in range(n_params) if not mask[i]]
    if locked:
        print(f"  Detected locked params: {locked}")
    return mask


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def main_train(args):
    if not PYG_AVAILABLE:
        print("pip install torch torch-geometric")
        return

    data = load_dataset(args.data)
    is_v2 = args.model_version == "v2"
    pyg_list = dataset_to_pyg_list(data, include_theta_all=is_v2)
    n = len(pyg_list)
    n_val = int(n * 0.15)
    n_train = n - n_val
    train_list, val_list = torch.utils.data.random_split(
        pyg_list,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_list = [pyg_list[i] for i in train_list.indices]
    val_list = [pyg_list[i] for i in val_list.indices]

    train_loader = PyGDataLoader(train_list, batch_size=args.batch_size, shuffle=True)
    val_loader = PyGDataLoader(val_list, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args, device)
    print(
        f"Model: {model.__class__.__name__}, params: {sum(p.numel() for p in model.parameters()):,}"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # Detect locked params for v2
    lock_mask = None
    if is_v2:
        lock_mask = detect_lock_mask(data, args.n_params).to(device)

    out_dir = Path(args.checkpoint).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    patience_counter = 0

    loss_name = "NLL" if is_v2 else "MSE"

    for epoch in range(args.epochs):
        if is_v2:
            train_loss = train_epoch_v2(model, train_loader, optimizer, device, lock_mask=lock_mask)
            val_loss = validate_v2(model, val_loader, device, lock_mask=lock_mask)
        else:
            train_loss = train_epoch_v1(model, train_loader, optimizer, device)
            val_loss = validate_v1(model, val_loader, device)
        scheduler.step()

        if val_loss < best_val:
            best_val = val_loss
            # Save model + metadata
            save_dict = {
                "model_state_dict": model.state_dict(),
                "model_version": args.model_version,
                "model_type": args.model_type,
                "n_params": args.n_params,
                "hidden": args.hidden,
                "n_layers": args.layers,
                "dropout": args.dropout,
            }
            torch.save(save_dict, args.checkpoint)
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch+1:4d}  train_{loss_name}={train_loss:.6f}  "
                f"val_{loss_name}={val_loss:.6f}  best={best_val:.6f}  lr={lr_now:.2e}",
                flush=True,
            )

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1} (patience={args.patience})")
            break

    print(f"\nBest val {loss_name}: {best_val:.6f}")
    print(f"Saved to {args.checkpoint}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def main_eval(args):
    data = load_dataset(args.data)
    device = torch.device("cpu")

    is_v2 = args.model_version == "v2"

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)

    # Handle both old (state_dict only) and new (dict with metadata) formats
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        saved_version = ckpt.get("model_version", "v1")
        saved_type = ckpt.get("model_type", "gnn")
        # Always use checkpoint metadata for version/type (override CLI defaults)
        args.model_version = saved_version
        is_v2 = saved_version == "v2"
        if args.model_type == "auto" or args.model_type != saved_type:
            args.model_type = saved_type
        args.n_params = ckpt.get("n_params", args.n_params)
        args.hidden = ckpt.get("hidden", args.hidden)
        args.layers = ckpt.get("n_layers", args.layers)
        args.dropout = ckpt.get("dropout", args.dropout)
    else:
        state_dict = ckpt

    model = build_model(args, device)
    model.load_state_dict(state_dict)
    model.eval()

    pyg_list = dataset_to_pyg_list(data, include_theta_all=is_v2)

    if is_v2:
        _eval_v2(model, pyg_list, data, device)
    else:
        _eval_v1(model, pyg_list, device)


def _eval_v1(model, pyg_list, device):
    """Evaluate v1 model (5 a_ij, MSE)."""
    batch = Batch.from_data_list(pyg_list)
    with torch.no_grad():
        pred = model(batch)
    target = batch.a_ij_active.reshape(-1, 5)

    mse = torch.nn.functional.mse_loss(pred, target).item()
    mae = torch.nn.functional.l1_loss(pred, target).item()

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
        print(
            f"{EDGE_NAMES[j]:<16s} {mse_j:8.4f} {mae_j:8.4f} {r2:8.4f} "
            f"{p.std().item():9.4f} {t.std().item():9.4f}"
        )


def _eval_v2(model, pyg_list, data_dict, device):
    """Evaluate v2 model (20 params, heteroscedastic)."""
    batch = Batch.from_data_list(pyg_list)
    with torch.no_grad():
        pred = model(batch)  # (N, n_params, 2)

    target = batch.theta_all.reshape(-1, pred.shape[1])  # (N, n_params)
    mu = pred[:, :, 0]
    log_sigma = pred[:, :, 1]
    sigma = log_sigma.exp()

    n_params = pred.shape[1]

    # Detect locked params
    stds = target.std(dim=0)
    locked = set(i for i in range(n_params) if stds[i] < 1e-8)

    # Overall metrics (non-locked only)
    free_mask = torch.tensor([i not in locked for i in range(n_params)])
    mu_free = mu[:, free_mask]
    tgt_free = target[:, free_mask]
    overall_mse = ((mu_free - tgt_free) ** 2).mean().item()
    overall_mae = (mu_free - tgt_free).abs().mean().item()

    print(f"Overall (free params)  MSE={overall_mse:.4f}  MAE={overall_mae:.4f}")
    print(f"Locked params: {sorted(locked)}")
    print()
    print(
        f"{'Param':<14s} {'MSE':>8s} {'MAE':>8s} {'R²':>8s} {'pred_σ':>8s} {'tgt_std':>8s} {'calib':>8s}"
    )
    print("-" * 72)

    for j in range(n_params):
        if j in locked:
            continue
        p, t, s = mu[:, j], target[:, j], sigma[:, j]
        mse_j = ((p - t) ** 2).mean().item()
        mae_j = (p - t).abs().mean().item()
        ss_res = ((t - p) ** 2).sum().item()
        ss_tot = ((t - t.mean()) ** 2).sum().item()
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        pred_sigma_mean = s.mean().item()
        tgt_std = t.std().item()
        # Calibration: fraction of targets within predicted ±2σ
        within_2s = ((t - p).abs() < 2 * s).float().mean().item()
        name = PARAM_NAMES_SHORT[j] if j < len(PARAM_NAMES_SHORT) else f"θ[{j}]"
        print(
            f"{name:<14s} {mse_j:8.4f} {mae_j:8.4f} {r2:8.4f} "
            f"{pred_sigma_mean:8.4f} {tgt_std:8.4f} {within_2s:8.2%}"
        )

    # Summary by category
    mu_idx = [0, 2, 5, 12, 16]
    b_idx = [3, 4, 8, 15, 9]  # note: θ[9] varies by interpretation
    a_active = [1, 10, 11, 18, 19]
    a_other = [i for i in range(n_params) if i not in mu_idx + b_idx + a_active and i not in locked]

    for cat_name, idxs in [
        ("Growth μ", mu_idx),
        ("Yield b", b_idx),
        ("Active a_ij", a_active),
        ("Other a_ij", a_other),
    ]:
        idxs_valid = [i for i in idxs if i not in locked and i < n_params]
        if not idxs_valid:
            continue
        cat_mse = ((mu[:, idxs_valid] - target[:, idxs_valid]) ** 2).mean().item()
        cat_sigma = sigma[:, idxs_valid].mean().item()
        print(f"\n  {cat_name}: MSE={cat_mse:.4f}, mean predicted σ={cat_sigma:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/train_gnn_N10000.npz")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mlflow", action="store_true", help="Log to MLflow")
    parser.add_argument(
        "--model-version",
        type=str,
        default="v1",
        choices=["v1", "v2", "auto"],
        help="v1=5 a_ij MSE, v2=20 params heteroscedastic",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="gnn",
        choices=["gnn", "mlp", "auto"],
        help="gnn=GCN, mlp=MLP ablation",
    )
    parser.add_argument(
        "--n-params", type=int, default=20, help="Number of params to predict (v2 only)"
    )
    parser.add_argument("mode", nargs="?", default="train", choices=["train", "eval"])
    args = parser.parse_args()

    base = Path(__file__).parent
    args.data = str(base / args.data) if not Path(args.data).is_absolute() else args.data

    # Default checkpoint path
    if args.checkpoint is None:
        if args.model_version == "v2":
            suffix = "mlp" if args.model_type == "mlp" else "gnn_v2"
            args.checkpoint = str(base / f"data/checkpoints/best_{suffix}.pt")
        else:
            args.checkpoint = str(base / "data/checkpoints/best.pt")
    elif not Path(args.checkpoint).is_absolute():
        args.checkpoint = str(base / args.checkpoint)

    if args.mode == "eval":
        main_eval(args)
    else:
        main_train(args)


if __name__ == "__main__":
    main()
