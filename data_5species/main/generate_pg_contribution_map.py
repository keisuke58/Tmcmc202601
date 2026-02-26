#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_5SPECIES_DIR = SCRIPT_DIR.parent
TMCMC_ROOT = DATA_5SPECIES_DIR.parent
PROGRAM_DIR = TMCMC_ROOT / "tmcmc" / "program2602"

sys.path.insert(0, str(DATA_5SPECIES_DIR))
sys.path.insert(0, str(TMCMC_ROOT))
sys.path.insert(0, str(PROGRAM_DIR))

from improved_5species_jit import BiofilmNewtonSolver5S


def model_time_to_days(t_arr, t_days):
    t_min, t_max = t_arr.min(), t_arr.max()
    d_min, d_max = t_days.min(), t_days.max()
    if t_max > t_min:
        return d_min + (t_arr - t_min) / (t_max - t_min) * (d_max - d_min)
    return t_arr.copy()


def load_run_and_simulate(run_dir: Path):
    with open(run_dir / "config.json") as f:
        config = json.load(f)
    with open(run_dir / "theta_MAP.json") as f:
        theta_map = json.load(f)
    data_points = np.load(run_dir / "data.npy")
    t_days = np.load(run_dir / "t_days.npy")

    if isinstance(theta_map, dict):
        theta_full = np.array(theta_map["theta_full"], dtype=float)
    else:
        theta_full = np.array(theta_map, dtype=float)

    phi_init = config.get("phi_init", 0.2)
    if isinstance(phi_init, list):
        phi_init = np.array(phi_init, dtype=float)

    Kp1 = config.get("Kp1", 1e-4)
    K_hill = config.get("K_hill", 0.0)
    n_hill = config.get("n_hill", 2.0)

    solver = BiofilmNewtonSolver5S(
        dt=config["dt"],
        maxtimestep=config["maxtimestep"],
        c_const=config["c_const"],
        alpha_const=config["alpha_const"],
        phi_init=phi_init,
        Kp1=Kp1,
        K_hill=K_hill,
        n_hill=n_hill,
    )

    t_arr, g_arr = solver.solve(theta_full)
    phi = g_arr[:, 0:5]
    psi = g_arr[:, 6:11]
    phibar = phi * psi

    t_model_days = model_time_to_days(t_arr, t_days)
    idx_map = [np.argmin(np.abs(t_model_days - d)) for d in t_days]
    phibar_obs = phibar[idx_map, :]

    metadata = config.get("metadata", {})
    condition = str(metadata.get("condition", "Unknown"))
    cultivation = str(metadata.get("cultivation", "Unknown"))
    label = f"{condition} {cultivation}"

    return (
        config,
        theta_full,
        data_points,
        t_days,
        phibar_obs,
        label,
    )


def compute_pg_rmse_late(data_pg, model_pg, n_late=2):
    data_pg = np.asarray(data_pg, dtype=float)
    model_pg = np.asarray(model_pg, dtype=float)
    if data_pg.shape != model_pg.shape:
        raise ValueError(f"Shape mismatch: data_pg {data_pg.shape}, model_pg {model_pg.shape}")
    if n_late <= 0 or n_late > len(data_pg):
        n_late = len(data_pg)
    idx_start = len(data_pg) - n_late
    d = data_pg[idx_start:]
    m = model_pg[idx_start:]
    return float(np.sqrt(np.mean((d - m) ** 2)))


def compute_species_contributions(theta_full, data_points, t_days, config, n_late=2):
    phi_init = config.get("phi_init", 0.2)
    if isinstance(phi_init, list):
        phi_init = np.array(phi_init, dtype=float)

    Kp1 = config.get("Kp1", 1e-4)
    K_hill = config.get("K_hill", 0.0)
    n_hill = config.get("n_hill", 2.0)

    solver = BiofilmNewtonSolver5S(
        dt=config["dt"],
        maxtimestep=config["maxtimestep"],
        c_const=config["c_const"],
        alpha_const=config["alpha_const"],
        phi_init=phi_init,
        Kp1=Kp1,
        K_hill=K_hill,
        n_hill=n_hill,
    )

    t_arr, g_arr = solver.solve(theta_full)
    phi = g_arr[:, 0:5]
    psi = g_arr[:, 6:11]
    phibar = phi * psi
    t_days_model = model_time_to_days(t_arr, t_days)
    idx_map = [np.argmin(np.abs(t_days_model - d)) for d in t_days]
    phibar_obs = phibar[idx_map, :]

    data_pg = data_points[:, 4]
    model_pg = phibar_obs[:, 4]
    rmse_base = compute_pg_rmse_late(data_pg, model_pg, n_late=n_late)

    contrib = {}
    species_indices = {
        "So": 16,
        "An": 17,
        "Vd": 18,
        "Fn": 19,
    }

    for sp_short, idx in species_indices.items():
        theta_mod = theta_full.copy()
        theta_mod[idx] = 0.0

        t_mod, g_mod = solver.solve(theta_mod)
        phi_mod = g_mod[:, 0:5]
        psi_mod = g_mod[:, 6:11]
        phibar_mod = phi_mod * psi_mod
        t_mod_days = model_time_to_days(t_mod, t_days)
        idx_mod = [np.argmin(np.abs(t_mod_days - d)) for d in t_days]
        phibar_mod_obs = phibar_mod[idx_mod, :]
        model_pg_mod = phibar_mod_obs[:, 4]
        rmse_mod = compute_pg_rmse_late(data_pg, model_pg_mod, n_late=n_late)
        contrib[sp_short] = rmse_mod - rmse_base

    return rmse_base, contrib


def plot_contribution_heatmap(labels, species_order, contrib_matrix, out_path: Path):
    contrib_arr = np.asarray(contrib_matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 3.5))

    vmin = np.min(contrib_arr)
    vmax = np.max(contrib_arr)
    if vmin == vmax:
        vmax = vmin + 1e-6
    im = ax.imshow(contrib_arr, cmap="Reds", vmin=0.0, vmax=vmax)

    ax.set_xticks(np.arange(len(species_order)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(species_order)
    ax.set_yticklabels(labels)

    for i in range(contrib_arr.shape[0]):
        for j in range(contrib_arr.shape[1]):
            val = contrib_arr[i, j]
            ax.text(
                j,
                i,
                f"{val:.3f}",
                ha="center",
                va="center",
                color="black" if val < vmax * 0.6 else "white",
                fontsize=9,
            )

    ax.set_xlabel("Species (driver of P.g)")
    ax.set_ylabel("Condition / Cultivation")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\Delta$RMSE (late P.g, removal)", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_pg_contribution_map.py <run_dir1> [<run_dir2> ...]")
        sys.exit(1)

    run_dirs = [Path(p).resolve() for p in sys.argv[1:]]
    for rd in run_dirs:
        if not rd.exists():
            print(f"Error: run directory {rd} does not exist")
            sys.exit(1)

    labels = []
    contrib_rows = []
    species_order = ["So", "An", "Vd", "Fn"]

    for rd in run_dirs:
        config, theta_full, data_points, t_days, _, label = load_run_and_simulate(rd)
        rmse_base, contrib = compute_species_contributions(
            theta_full, data_points, t_days, config, n_late=2
        )
        labels.append(label)
        row = [contrib[s] for s in species_order]
        contrib_rows.append(row)
        print(f"{label}: base RMSE_late={rmse_base:.4f}, contributions={contrib}")

    first_run_dir = run_dirs[0]
    out_path = first_run_dir.parent / "Fig_pg_contribution_map_HOBIC_vs_Static.png"
    plot_contribution_heatmap(labels, species_order, contrib_rows, out_path)
    print(f"Saved contribution map to {out_path}")


if __name__ == "__main__":
    main()
