#!/usr/bin/env python3
"""Fig 23: GNN Prior Effect on TMCMC Posterior (paper quality).

Extended version:
  Panel A: 5×4 grid showing all 20 params (uniform vs GNN v2 prior)
  Panel B: GNN vs MLP ablation (per-param R² and σ comparison)
  Panel C: Predicted σ heatmap (heteroscedastic confidence)

Compares:
  - Uniform prior TMCMC (dh_baseline)
  - GNN-informed prior TMCMC (gnn_test_dh or gnn_v2_dh)
  - GNN/MLP predicted values (vertical lines / bars)
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import gaussian_kde

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
PAPER_FIG_DIR = PROJECT_ROOT / "FEM" / "figures" / "paper_final"
PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR = PROJECT_ROOT / "data_5species" / "_runs"

PARAM_NAMES = [
    r"$\mu_{\mathrm{So}}$",
    r"$a_{\mathrm{So \to An}}$",
    r"$\mu_{\mathrm{An}}$",
    r"$b_{\mathrm{So}}$",
    r"$b_{\mathrm{An}}$",
    r"$\mu_{\mathrm{Vd}}$",
    r"$a_{\mathrm{An \to Vd}}$",
    r"$a_{\mathrm{Vd \to An}}$",
    r"$b_{\mathrm{Vd}}$",
    r"$a_{\mathrm{An \to Fn}}$",
    r"$a_{\mathrm{So \to Vd}}$",
    r"$a_{\mathrm{So \to Fn}}$",
    r"$\mu_{\mathrm{Fn}}$",
    r"$a_{\mathrm{Fn \to An}}$",
    r"$a_{\mathrm{Vd \to Fn}}$",
    r"$b_{\mathrm{Fn}}$",
    r"$\mu_{\mathrm{Pg}}$",
    r"$a_{\mathrm{Fn \to Vd}}^{\mathrm{inv}}$",
    r"$a_{\mathrm{Vd \to Pg}}$",
    r"$a_{\mathrm{Fn \to Pg}}$",
]

PARAM_SHORT = [
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

# Parameter categories
MU_IDX = [0, 2, 5, 12, 16]
B_IDX = [3, 4, 8, 15, 9]
ACTIVE_THETA_IDX = [1, 10, 11, 18, 19]
OTHER_A_IDX = [6, 7, 13, 14, 17]

EDGE_NAMES = ["So→An", "So→Vd", "So→Fn", "Vd→Pg", "Fn→Pg"]

CATEGORY_COLORS = {
    "μ": "#1976D2",  # blue
    "b": "#388E3C",  # green
    "a_active": "#E65100",  # orange
    "a_other": "#7B1FA2",  # purple
}


def get_param_category(idx):
    if idx in MU_IDX:
        return "μ"
    if idx in B_IDX:
        return "b"
    if idx in ACTIVE_THETA_IDX:
        return "a_active"
    return "a_other"


def load_samples_and_map(run_dir):
    """Load posterior samples and MAP estimate."""
    samples_path = run_dir / "samples.npy"
    map_path = run_dir / "theta_MAP.json"
    logL_path = run_dir / "logL.npy"
    samples, theta_map, logL = None, None, None
    if samples_path.exists():
        samples = np.load(str(samples_path))
    if logL_path.exists():
        logL = np.load(str(logL_path))
    if map_path.exists():
        with open(map_path) as f:
            data = json.load(f)
        theta_map = (
            np.array(data["theta_full"])
            if isinstance(data, dict) and "theta_full" in data
            else np.array(data)
        )
    return samples, theta_map, logL


def load_gnn_predictions(version="v2"):
    """Load pre-computed GNN predictions."""
    pred_path = SCRIPT_DIR / "data" / "gnn_prior_predictions.json"
    if not pred_path.exists():
        return None
    with open(pred_path) as f:
        data = json.load(f)
    if "Dysbiotic_HOBIC" not in data:
        return None
    return data["Dysbiotic_HOBIC"]


def compute_overlap(s1, s2, idx):
    """Bhattacharyya coefficient approximation."""
    d1, d2 = s1[:, idx], s2[:, idx]
    m1, std1, m2, std2 = d1.mean(), d1.std(), d2.mean(), d2.std()
    if std1 < 1e-8 or std2 < 1e-8:
        return 0.0
    bd = 0.25 * ((m1 - m2) ** 2 / (std1**2 + std2**2)) + 0.5 * np.log(
        0.5 * (std1 / std2 + std2 / std1)
    )
    return float(np.exp(-bd))


def compute_width_ratio(s1, s2, idx):
    """Posterior width ratio (uniform/GNN)."""
    std1 = s1[:, idx].std()
    std2 = s2[:, idx].std()
    if std2 < 1e-8:
        return float("inf")
    return std1 / std2


def plot_kde(ax, data, color, label, alpha_fill=0.25):
    if len(data) < 5 or data.std() < 1e-10:
        ax.axvline(data.mean(), color=color, lw=2, label=label)
        return
    try:
        kde = gaussian_kde(data, bw_method="silverman")
        xg = np.linspace(data.min() - 0.15 * data.ptp(), data.max() + 0.15 * data.ptp(), 300)
        y = kde(xg)
        ax.plot(xg, y, color=color, lw=1.5, label=label)
        ax.fill_between(xg, y, alpha=alpha_fill, color=color)
    except Exception:
        ax.hist(data, bins=30, alpha=0.4, density=True, color=color, label=label)


def main():
    # Load predictions
    gnn_data = load_gnn_predictions()
    is_v2 = gnn_data is not None and "theta_mu" in gnn_data
    gnn_mu = None
    gnn_sigma = None

    if is_v2:
        gnn_mu = np.array(gnn_data["theta_mu"])
        gnn_sigma = np.array(gnn_data["theta_sigma"])
        gnn_free = gnn_data.get("theta_free", [True] * len(gnn_mu))
        print(f"v2 predictions loaded: {sum(gnn_free)} free params")
    elif gnn_data is not None:
        gnn_pred_5 = np.array(gnn_data["a_ij_pred"])
        print(f"v1 predictions: {gnn_pred_5}")

    # Load posterior samples
    uniform_dir = RUNS_DIR / "dh_baseline"
    # Try v2 run first, fall back to v1
    gnn_dir_v2 = RUNS_DIR / "gnn_v2_dh"
    gnn_dir_v1 = RUNS_DIR / "gnn_test_dh"
    gnn_dir = gnn_dir_v2 if gnn_dir_v2.exists() else gnn_dir_v1

    uni_s, uni_m, _ = load_samples_and_map(uniform_dir)
    gnn_s, gnn_m, _ = load_samples_and_map(gnn_dir)

    if uni_s is None:
        print(f"ERROR: No samples in {uniform_dir}")
        return
    if gnn_s is None:
        print(f"WARNING: No GNN posterior samples in {gnn_dir}")
        print("  Generating prediction-only figures.")
        gnn_s = None

    n_params = min(uni_s.shape[1], 20)
    if gnn_s is not None:
        n_params = min(n_params, gnn_s.shape[1])
    print(f"Uniform samples: {uni_s.shape}")
    if gnn_s is not None:
        print(f"GNN samples: {gnn_s.shape}")

    # ===================================================================
    # Panel A: 5×4 grid — all 20 params posterior comparison
    # ===================================================================
    c_uni = "#7B1FA2"  # purple
    c_gnn = "#00838F"  # teal
    c_pred = "#FF6F00"  # amber

    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(5, 4, figure=fig, hspace=0.50, wspace=0.35)

    for idx in range(n_params):
        row, col = idx // 4, idx % 4
        ax = fig.add_subplot(gs[row, col])

        plot_kde(ax, uni_s[:, idx], c_uni, "Uniform prior")
        if gnn_s is not None:
            plot_kde(ax, gnn_s[:, idx], c_gnn, "GNN prior")

        # Mark prediction
        if is_v2 and gnn_mu is not None and idx < len(gnn_mu):
            if gnn_free[idx]:
                ax.axvline(
                    gnn_mu[idx],
                    color=c_pred,
                    lw=2.0,
                    ls="--",
                    alpha=0.8,
                    label=f"pred={gnn_mu[idx]:.2f}",
                )
                # Show ±σ band
                lo = gnn_mu[idx] - gnn_sigma[idx]
                hi = gnn_mu[idx] + gnn_sigma[idx]
                ax.axvspan(lo, hi, alpha=0.08, color=c_pred)
        elif not is_v2 and idx in ACTIVE_THETA_IDX and gnn_data is not None:
            k = ACTIVE_THETA_IDX.index(idx)
            ax.axvline(
                gnn_pred_5[k],
                color=c_pred,
                lw=2.5,
                ls="--",
                alpha=0.8,
                label=f"GNN={gnn_pred_5[k]:.2f}",
            )

        # MAP estimates
        if uni_m is not None and idx < len(uni_m):
            ax.axvline(uni_m[idx], color=c_uni, lw=1.0, ls=":", alpha=0.5)
        if gnn_m is not None and idx < len(gnn_m):
            ax.axvline(gnn_m[idx], color=c_gnn, lw=1.0, ls=":", alpha=0.5)

        # Title with overlap and category color
        cat = get_param_category(idx)
        cat_color = CATEGORY_COLORS.get(cat, "#616161")

        if gnn_s is not None:
            ol = compute_overlap(uni_s, gnn_s, idx)
            wr = compute_width_ratio(uni_s, gnn_s, idx)
            title_extra = f"  OL={ol:.2f}"
            if is_v2 and gnn_mu is not None and gnn_free[idx]:
                title_extra += f" σ̂={gnn_sigma[idx]:.2f}"
        else:
            title_extra = ""
            if is_v2 and gnn_mu is not None and idx < len(gnn_mu) and gnn_free[idx]:
                title_extra = f"  σ̂={gnn_sigma[idx]:.2f}"

        ax.set_title(
            f"{PARAM_NAMES[idx]}{title_extra}",
            fontsize=8,
            color=cat_color,
            fontweight="bold",
        )
        ax.tick_params(labelsize=7)
        ax.set_yticks([])

        if row == 0 and col == 0:
            ax.legend(fontsize=6, loc="upper right")

    v_label = "v2 (20-param heteroscedastic)" if is_v2 else "v1 (5 active edges)"
    fig.suptitle(
        f"Fig. 23: GNN-Informed Prior Effect on TMCMC Posterior — {v_label}",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    legend_text = (
        "Blue=μ  Green=b  Orange=active a_ij  Purple=other a_ij  |  "
        "Dashed=prediction  Shaded=±σ  Dotted=MAP"
    )
    fig.text(0.5, 0.01, legend_text, ha="center", fontsize=8, style="italic", color="#424242")

    out_path = PAPER_FIG_DIR / "Fig23_gnn_prior_effect.png"
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {out_path}")

    # ===================================================================
    # Panel B: Predicted σ by parameter (v2 only)
    # ===================================================================
    if is_v2 and gnn_mu is not None:
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

        # Left: predicted σ bar chart by parameter
        free_idx = [i for i in range(len(gnn_mu)) if gnn_free[i]]
        names = [PARAM_SHORT[i] for i in free_idx]
        sigmas = [gnn_sigma[i] for i in free_idx]
        colors = [CATEGORY_COLORS[get_param_category(i)] for i in free_idx]

        ax1.barh(range(len(free_idx)), sigmas, color=colors, edgecolor="white", linewidth=0.5)
        ax1.set_yticks(range(len(free_idx)))
        ax1.set_yticklabels(names, fontsize=8)
        ax1.set_xlabel("Predicted σ (heteroscedastic)", fontsize=10)
        ax1.set_title("GNN v2: Per-Parameter Uncertainty", fontsize=11, fontweight="bold")
        ax1.invert_yaxis()

        # Right: width ratio (if posterior samples available)
        if gnn_s is not None:
            ratios = [compute_width_ratio(uni_s, gnn_s, i) for i in free_idx]
            ax2.barh(range(len(free_idx)), ratios, color=colors, edgecolor="white", linewidth=0.5)
            ax2.set_yticks(range(len(free_idx)))
            ax2.set_yticklabels(names, fontsize=8)
            ax2.set_xlabel("Posterior Width Ratio (Uniform/GNN)", fontsize=10)
            ax2.set_title("Posterior Contraction per Parameter", fontsize=11, fontweight="bold")
            ax2.axvline(1.0, color="gray", ls="--", lw=0.8)
            ax2.invert_yaxis()
        else:
            ax2.text(
                0.5,
                0.5,
                "No GNN posterior samples\navailable for comparison",
                ha="center",
                va="center",
                transform=ax2.transAxes,
                fontsize=12,
            )
            ax2.set_axis_off()

        fig2.tight_layout()
        out_path2 = PAPER_FIG_DIR / "Fig23_gnn_v2_sigma_analysis.png"
        fig2.savefig(str(out_path2), dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig2)
        print(f"Saved: {out_path2}")

    # ===================================================================
    # Panel C: GNN vs MLP ablation (if both checkpoints exist)
    # ===================================================================
    gnn_v2_ckpt = SCRIPT_DIR / "data" / "checkpoints" / "best_gnn_v2.pt"
    mlp_ckpt = SCRIPT_DIR / "data" / "checkpoints" / "best_mlp.pt"
    data_path = SCRIPT_DIR / "data" / "train_gnn_N10000.npz"

    if gnn_v2_ckpt.exists() and mlp_ckpt.exists() and data_path.exists():
        _plot_ablation(gnn_v2_ckpt, mlp_ckpt, data_path)
    else:
        missing = []
        if not gnn_v2_ckpt.exists():
            missing.append("GNN v2 checkpoint")
        if not mlp_ckpt.exists():
            missing.append("MLP checkpoint")
        if not data_path.exists():
            missing.append("Training data")
        print(f"\nSkipping GNN vs MLP ablation (missing: {', '.join(missing)})")

    # ===================================================================
    # Summary statistics
    # ===================================================================
    if gnn_s is not None:
        print("\n=== Summary ===")
        overlaps = [compute_overlap(uni_s, gnn_s, i) for i in range(n_params)]

        for cat_name, idxs in [
            ("Growth μ", MU_IDX),
            ("Yield b", B_IDX),
            ("Active a_ij", ACTIVE_THETA_IDX),
            ("Other a_ij", OTHER_A_IDX),
        ]:
            valid = [i for i in idxs if i < n_params]
            if not valid:
                continue
            cat_ol = np.mean([overlaps[i] for i in valid])
            cat_wr = [compute_width_ratio(uni_s, gnn_s, i) for i in valid]
            print(f"  {cat_name:12s}: mean OL={cat_ol:.3f}, mean σ ratio={np.mean(cat_wr):.2f}×")

        all_wr = [compute_width_ratio(uni_s, gnn_s, i) for i in range(n_params)]
        print(
            f"  {'All':12s}: mean OL={np.mean(overlaps):.3f}, mean σ ratio={np.mean(all_wr):.2f}×"
        )


def _plot_ablation(gnn_ckpt, mlp_ckpt, data_path):
    """Plot GNN vs MLP ablation comparison."""
    import torch
    import sys

    sys.path.insert(0, str(SCRIPT_DIR))
    from gnn_model import InteractionGNNv2, InteractionMLP
    from graph_builder import dataset_to_pyg_list

    data = np.load(str(data_path))
    data_dict = {k: data[k] for k in data.files}
    pyg_list = dataset_to_pyg_list(data_dict, include_theta_all=True)

    from torch_geometric.data import Batch

    batch = Batch.from_data_list(pyg_list)
    target = batch.theta_all.reshape(-1, 20)

    # Detect locked
    stds = target.std(dim=0)
    locked = set(i for i in range(20) if stds[i] < 1e-8)

    results = {}
    for name, ckpt_path in [("GCN", gnn_ckpt), ("MLP", mlp_ckpt)]:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        state_dict = ckpt["model_state_dict"]
        model_type = ckpt.get("model_type", "gnn")
        n_params = ckpt.get("n_params", 20)
        hidden = ckpt.get("hidden", 128)
        n_layers = ckpt.get("n_layers", 4)
        dropout = ckpt.get("dropout", 0.2)

        if model_type == "mlp":
            model = InteractionMLP(
                in_dim=15, hidden=hidden, n_params=n_params, n_layers=n_layers, dropout=dropout
            )
        else:
            model = InteractionGNNv2(
                in_dim=3, hidden=hidden, n_params=n_params, n_layers=n_layers, dropout=dropout
            )
        model.load_state_dict(state_dict)
        model.eval()

        with torch.no_grad():
            pred = model(batch)  # (N, 20, 2)
        mu = pred[:, :, 0]
        log_sigma = pred[:, :, 1]
        sigma = log_sigma.exp()

        # Per-param R² and mean σ
        r2_list = []
        sigma_list = []
        for j in range(20):
            if j in locked:
                r2_list.append(np.nan)
                sigma_list.append(np.nan)
                continue
            p, t = mu[:, j], target[:, j]
            ss_res = ((t - p) ** 2).sum().item()
            ss_tot = ((t - t.mean()) ** 2).sum().item()
            r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
            r2_list.append(r2)
            sigma_list.append(sigma[:, j].mean().item())

        results[name] = {"r2": r2_list, "sigma": sigma_list}

    # Plot
    free_idx = [i for i in range(20) if i not in locked]
    names = [PARAM_SHORT[i] for i in free_idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # R² comparison
    x = np.arange(len(free_idx))
    w = 0.35
    r2_gnn = [results["GCN"]["r2"][i] for i in free_idx]
    r2_mlp = [results["MLP"]["r2"][i] for i in free_idx]
    ax1.barh(x - w / 2, r2_gnn, w, label="GCN", color="#00838F", edgecolor="white")
    ax1.barh(x + w / 2, r2_mlp, w, label="MLP", color="#E65100", edgecolor="white")
    ax1.set_yticks(x)
    ax1.set_yticklabels(names, fontsize=8)
    ax1.set_xlabel("R² (higher = better)", fontsize=10)
    ax1.set_title("GCN vs MLP: Per-Parameter R²", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.axvline(0.0, color="gray", ls="--", lw=0.5)
    ax1.invert_yaxis()

    # σ comparison
    sig_gnn = [results["GCN"]["sigma"][i] for i in free_idx]
    sig_mlp = [results["MLP"]["sigma"][i] for i in free_idx]
    ax2.barh(x - w / 2, sig_gnn, w, label="GCN", color="#00838F", edgecolor="white")
    ax2.barh(x + w / 2, sig_mlp, w, label="MLP", color="#E65100", edgecolor="white")
    ax2.set_yticks(x)
    ax2.set_yticklabels(names, fontsize=8)
    ax2.set_xlabel("Mean Predicted σ (lower = more confident)", fontsize=10)
    ax2.set_title("GCN vs MLP: Heteroscedastic Uncertainty", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.invert_yaxis()

    fig.suptitle("Fig. 23b: GCN vs MLP Ablation Study", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    out_path = PAPER_FIG_DIR / "Fig23b_gnn_vs_mlp_ablation.png"
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved ablation: {out_path}")

    # Print summary
    print("\n=== GCN vs MLP Ablation ===")
    r2_diff = np.array(r2_gnn) - np.array(r2_mlp)
    print(f"  Mean R² — GCN: {np.nanmean(r2_gnn):.4f}, MLP: {np.nanmean(r2_mlp):.4f}")
    print(f"  Mean σ  — GCN: {np.nanmean(sig_gnn):.4f}, MLP: {np.nanmean(sig_mlp):.4f}")
    wins_gnn = sum(1 for d in r2_diff if d > 0.01)
    wins_mlp = sum(1 for d in r2_diff if d < -0.01)
    ties = len(r2_diff) - wins_gnn - wins_mlp
    print(f"  R² wins: GCN={wins_gnn}, MLP={wins_mlp}, tie={ties}")

    # Per-category summary
    for cat_name, idxs in [
        ("Growth μ", MU_IDX),
        ("Yield b", B_IDX),
        ("Active a_ij", ACTIVE_THETA_IDX),
        ("Other a_ij", OTHER_A_IDX),
    ]:
        valid = [i for i in idxs if i not in locked]
        if not valid:
            continue
        r2_g = np.nanmean([results["GCN"]["r2"][i] for i in valid])
        r2_m = np.nanmean([results["MLP"]["r2"][i] for i in valid])
        print(f"  {cat_name:12s}: GCN R²={r2_g:.3f}, MLP R²={r2_m:.3f}, Δ={r2_g-r2_m:+.3f}")


if __name__ == "__main__":
    main()
