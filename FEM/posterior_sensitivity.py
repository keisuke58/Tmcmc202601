#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_theta(run_dir: Path) -> np.ndarray:
    theta_map_path = run_dir / "theta_MAP.npy"
    if theta_map_path.exists():
        return np.load(theta_map_path)
    json_path = run_dir / "theta_MAP.json"
    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if "theta_full" in data:
            return np.array(data["theta_full"], dtype=float)
        if "theta_sub" in data:
            return np.array(data["theta_sub"], dtype=float)
        raise KeyError("theta_MAP.json does not contain theta_full or theta_sub")
    raise SystemExit(f"theta_MAP not found in {run_dir}")


def compute_depth_center(phi_pg: np.ndarray, x: np.ndarray) -> float:
    prof = phi_pg.mean(axis=(1, 2))
    prof_sum = prof.sum()
    if prof_sum <= 0.0:
        return float(x.mean())
    w = prof / prof_sum
    return float(np.sum(w * x))


def summarize_sample(
    run_dir: Path, fem_base: Path, sample_idx: int, target: str
) -> tuple[np.ndarray, float]:
    fem_dirs = sorted([p for p in fem_base.glob("sample_*") if p.is_dir()])
    if not fem_dirs:
        raise SystemExit(f"No sample_* directories found under {fem_base}")
    fem_dir = fem_dirs[sample_idx % len(fem_dirs)]

    theta_path = fem_dir / "theta.npy"
    if theta_path.exists():
        theta_full = np.load(theta_path)
    else:
        theta_full = load_theta(run_dir)

    if target == "depth":
        depth_pg_path = fem_dir / "depth_pg.npy"
        if depth_pg_path.exists():
            depth_arr = np.load(depth_pg_path)
            value = float(depth_arr[-1])
        else:
            phi = np.load(fem_dir / "snapshots_phi.npy")
            x = np.load(fem_dir / "mesh_x.npy")
            phi_pg = phi[-1, 4]
            value = compute_depth_center(phi_pg, x)
    else:
        stress_path = fem_dir / "stress.json"
        if not stress_path.exists():
            raise SystemExit(f"stress.json not found in {fem_dir}")
        with stress_path.open("r", encoding="utf-8") as f:
            stress = json.load(f)
        if target == "stress_substrate":
            value = float(stress["substrate_smises"])
        else:
            value = float(stress["surface_smises"])

    return theta_full, value


def plot_scatter(params: np.ndarray, values: np.ndarray, out_path: Path, y_label: str) -> None:
    n_param = params.shape[1]
    ncols = 5
    nrows = int(np.ceil(n_param / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 2.4 * nrows), sharex=False, sharey=True)
    axes = axes.ravel()
    for i in range(n_param):
        ax = axes[i]
        ax.scatter(params[:, i], values, s=8, alpha=0.4)
        ax.set_xlabel(f"theta[{i}]", fontsize=8)
        if i % ncols == 0:
            ax.set_ylabel(y_label, fontsize=8)
        ax.grid(True, alpha=0.3)
    for j in range(n_param, len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Posterior sensitivity: theta vs scalar observable")
    ap.add_argument(
        "--run-dir",
        required=True,
        help="TMCMC run directory containing samples.npy and theta_MAP.json/npy",
    )
    ap.add_argument(
        "--fem-base", required=True, help="Base directory with per-sample outputs (sample_XXXX)"
    )
    ap.add_argument("--n-samples", type=int, default=40, help="Number of samples to use")
    ap.add_argument(
        "--target",
        choices=["depth", "stress_substrate", "stress_surface"],
        default="depth",
        help="Observable to plot on y-axis",
    )
    ap.add_argument(
        "--out-dir", default="_posterior_sensitivity", help="Output directory for plots"
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    fem_base = Path(args.fem_base)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    theta_template = load_theta(run_dir)
    n_param = len(theta_template)
    theta_list = []
    value_list = []
    if args.target == "depth":
        for k in range(args.n_samples):
            theta_k, value_k = summarize_sample(run_dir, fem_base, k, args.target)
            theta_list.append(theta_k)
            value_list.append(value_k)
    else:
        fem_dirs = sorted([p for p in fem_base.glob("sample_*") if p.is_dir()])
        if not fem_dirs:
            raise SystemExit(f"No sample_* directories found under {fem_base}")
        for fem_dir in fem_dirs:
            stress_path = fem_dir / "stress.json"
            if not stress_path.exists():
                continue
            theta_path = fem_dir / "theta.npy"
            if theta_path.exists():
                theta_full = np.load(theta_path)
            else:
                theta_full = load_theta(run_dir)
            with stress_path.open("r", encoding="utf-8") as f:
                stress = json.load(f)
            if args.target == "stress_substrate":
                value_k = float(stress["substrate_smises"])
            else:
                value_k = float(stress["surface_smises"])
            theta_list.append(theta_full)
            value_list.append(value_k)
    theta_arr = np.stack(theta_list, axis=0).reshape(len(theta_list), n_param)
    value_arr = np.array(value_list)

    if args.target == "depth":
        out_path = out_dir / "theta_vs_depth_scatter.png"
        y_label = "Pg depth"
    elif args.target == "stress_substrate":
        out_path = out_dir / "theta_vs_stress_substrate_scatter.png"
        y_label = "Substrate S_Mises [Pa]"
    else:
        out_path = out_dir / "theta_vs_stress_surface_scatter.png"
        y_label = "Surface S_Mises [Pa]"

    plot_scatter(theta_arr, value_arr, out_path, y_label)


if __name__ == "__main__":
    main()
