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

from core.nishioka_model import PgingivalisSurgeMLP
from improved_5species_jit import BiofilmNewtonSolver5S

HOBIC_FEATURE_NAMES = [
    "phi_S",
    "phi_A",
    "phi_V",
    "phi_F",
    "phi_Pg",
    "t_norm",
    "t_norm2",
    "Pg_prev",
    "dPg",
    "comm",
    "vei_ratio",
    "fn_ratio",
]


def model_time_to_days(t_arr, t_days):
    t_min, t_max = t_arr.min(), t_arr.max()
    d_min, d_max = t_days.min(), t_days.max()
    if t_max > t_min:
        return d_min + (t_arr - t_min) / (t_max - t_min) * (d_max - d_min)
    return t_arr.copy()


def load_run_data(run_dir: Path):
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

    phi_init = np.array(config["phi_init"], dtype=float)
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

    return config, data_points, t_days, phibar_obs


def build_dataset(run_dirs):
    X_list = []
    y_list = []
    for run_dir in run_dirs:
        config, data_points, t_days, phibar_obs = load_run_data(run_dir)
        metadata = config.get("metadata", {})
        cultivation = str(metadata.get("cultivation", ""))
        cultivation_lower = cultivation.lower()
        is_static = 1.0 if "static" in cultivation_lower else 0.0

        t0 = float(t_days[0])
        t1 = float(t_days[-1])
        denom = t1 - t0 if t1 > t0 else 1.0

        for k, day in enumerate(t_days):
            t_norm = (float(day) - t0) / denom
            phi = phibar_obs[k, :]
            x = [
                float(phi[0]),
                float(phi[1]),
                float(phi[2]),
                float(phi[3]),
                float(phi[4]),
                t_norm,
                is_static,
            ]
            resid = float(data_points[k, 4] - phi[4])
            X_list.append(x)
            y_list.append(resid)

    X = np.array(X_list, dtype=float)
    y = np.array(y_list, dtype=float)
    return X, y


def forward_batch(mlp: PgingivalisSurgeMLP, X):
    x = np.asarray(X, dtype=float)
    h = np.tanh(x @ mlp.W1 + mlp.b1)
    z2 = h @ mlp.W2 + mlp.b2
    y = np.tanh(z2)
    y = mlp.output_scale * y
    return y, h, z2


def train_mlp(mlp: PgingivalisSurgeMLP, X, y, epochs=2000, lr=5e-2):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    N = X.shape[0]
    for _ in range(epochs):
        y_pred, h, z2 = forward_batch(mlp, X)
        y_pred_flat = y_pred.reshape(-1)
        diff = y_pred_flat - y
        grad_y = (2.0 / N) * diff
        tanh_z2 = np.tanh(z2)
        dy_dz2 = mlp.output_scale * (1.0 - tanh_z2 ** 2)
        grad_z2 = grad_y.reshape(-1, 1) * dy_dz2
        grad_W2 = h.T @ grad_z2
        grad_b2 = grad_z2.sum(axis=0)
        dh = grad_z2 @ mlp.W2.T
        dz1 = dh * (1.0 - h ** 2)
        grad_W1 = X.T @ dz1
        grad_b1 = dz1.sum(axis=0)
        mlp.W1 -= lr * grad_W1
        mlp.b1 -= lr * grad_b1
        mlp.W2 -= lr * grad_W2
        mlp.b2 -= lr * grad_b2


def generate_correction_plot(run_dir: Path, mlp: PgingivalisSurgeMLP):
    config, data_points, t_days, phibar_obs = load_run_data(run_dir)
    metadata = config.get("metadata", {})
    condition = str(metadata.get("condition", ""))
    cultivation = str(metadata.get("cultivation", ""))
    cultivation_lower = cultivation.lower()
    is_static = 1.0 if "static" in cultivation_lower else 0.0

    t0 = float(t_days[0])
    t1 = float(t_days[-1])
    denom = t1 - t0 if t1 > t0 else 1.0

    X_run = []
    for k, day in enumerate(t_days):
        t_norm = (float(day) - t0) / denom
        phi = phibar_obs[k, :]
        x = [
            float(phi[0]),
            float(phi[1]),
            float(phi[2]),
            float(phi[3]),
            float(phi[4]),
            t_norm,
            is_static,
        ]
        X_run.append(x)

    X_run = np.array(X_run, dtype=float)
    resid_pred = np.asarray(mlp(X_run), dtype=float)

    pg_data = data_points[:, 4]
    pg_base = phibar_obs[:, 4]
    pg_corr = pg_base + resid_pred

    rmse_base = float(np.sqrt(np.mean((pg_base - pg_data) ** 2)))
    rmse_corr = float(np.sqrt(np.mean((pg_corr - pg_data) ** 2)))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t_days, pg_data, "ko", label="Data")
    ax.plot(t_days, pg_base, "-o", color="tab:blue", label=f"Model base (RMSE={rmse_base:.3f})")
    ax.plot(t_days, pg_corr, "-o", color="tab:red", label=f"Model+ML (RMSE={rmse_corr:.3f})")
    ax.set_xlabel("Day")
    ax.set_ylabel("P.g phibar")
    title = f"{condition} {cultivation} P.g ML correction"
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    condition_clean = condition.replace(" ", "_")
    cultivation_clean = cultivation.replace(" ", "_")
    out_name = f"Fig_A11_Pg_ml_correction_{condition_clean}_{cultivation_clean}.png"
    out_path = run_dir / out_name
    fig.savefig(out_path, dpi=300, bbox_inches="tight")


def build_hobic_strong_dataset(run_dir: Path):
    config, data_points, t_days, phibar_obs = load_run_data(run_dir)
    metadata = config.get("metadata", {})
    condition = str(metadata.get("condition", ""))
    cultivation = str(metadata.get("cultivation", ""))
    t0 = float(t_days[0])
    t1 = float(t_days[-1])
    denom = t1 - t0 if t1 > t0 else 1.0
    X_list = []
    y_list = []
    for k, day in enumerate(t_days):
        phi = phibar_obs[k, :]
        pg = float(phi[4])
        if k == 0:
            pg_prev = pg
        else:
            pg_prev = float(phibar_obs[k - 1, 4])
        dpg = pg - pg_prev
        comm = float(phi[0] + phi[1] + phi[2] + phi[3])
        eps = 1e-8
        vei_ratio = float(phi[2]) / (comm + eps)
        fn_ratio = float(phi[3]) / (comm + eps)
        t_norm = (float(day) - t0) / denom
        t_norm2 = t_norm * t_norm
        x = [
            float(phi[0]),
            float(phi[1]),
            float(phi[2]),
            float(phi[3]),
            pg,
            t_norm,
            t_norm2,
            pg_prev,
            dpg,
            comm,
            vei_ratio,
            fn_ratio,
        ]
        resid = float(data_points[k, 4] - pg)
        X_list.append(x)
        y_list.append(resid)
    X = np.array(X_list, dtype=float)
    y = np.array(y_list, dtype=float)
    w = np.ones_like(y, dtype=float)
    for k, day in enumerate(t_days):
        d = float(day)
        if abs(d - 15.0) < 1e-3:
            w[k] = 3.0
        if abs(d - 21.0) < 1e-3:
            w[k] = 5.0
    if np.allclose(w, 1.0):
        if len(w) >= 2:
            w[-2] = 3.0
            w[-1] = 5.0
    return X, y, w, t_days, phibar_obs, data_points, condition, cultivation


def train_mlp_weighted(mlp: PgingivalisSurgeMLP, X, y, w, epochs=4000, lr=5e-2, l2=0.0):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    y_vec = y.reshape(-1, 1)
    w_vec = w.reshape(-1, 1)
    s = float(w_vec.sum())
    for _ in range(epochs):
        y_pred, h, z2 = forward_batch(mlp, X)
        y_pred_vec = y_pred.reshape(-1, 1)
        diff = y_pred_vec - y_vec
        grad_y = (2.0 / s) * w_vec * diff
        tanh_z2 = np.tanh(z2)
        dy_dz2 = mlp.output_scale * (1.0 - tanh_z2 ** 2)
        grad_z2 = grad_y * dy_dz2
        grad_W2 = h.T @ grad_z2
        grad_b2 = grad_z2.sum(axis=0)
        dh = grad_z2 @ mlp.W2.T
        dz1 = dh * (1.0 - h ** 2)
        grad_W1 = X.T @ dz1
        grad_b1 = dz1.sum(axis=0)
        if l2 > 0.0:
            grad_W2 += 2.0 * l2 * mlp.W2
            grad_W1 += 2.0 * l2 * mlp.W1
        mlp.W1 -= lr * grad_W1
        mlp.b1 -= lr * grad_b1
        mlp.W2 -= lr * grad_W2
        mlp.b2 -= lr * grad_b2


def generate_correction_plot_hobic_strong(
    run_dir: Path,
    mlp: PgingivalisSurgeMLP,
    X,
    t_days,
    phibar_obs,
    data_points,
    condition: str,
    cultivation: str,
):
    X_run = np.asarray(X, dtype=float)
    resid_pred = np.asarray(mlp(X_run), dtype=float)
    pg_data = data_points[:, 4]
    pg_base = phibar_obs[:, 4]
    pg_corr = pg_base + resid_pred
    rmse_base = float(np.sqrt(np.mean((pg_base - pg_data) ** 2)))
    rmse_corr = float(np.sqrt(np.mean((pg_corr - pg_data) ** 2)))
    rmse_base_late = float(np.sqrt(np.mean((pg_base[-2:] - pg_data[-2:]) ** 2)))
    rmse_corr_late = float(np.sqrt(np.mean((pg_corr[-2:] - pg_data[-2:]) ** 2)))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t_days, pg_data, "ko", label="Data")
    ax.plot(t_days, pg_base, "-o", color="tab:blue", label=f"Model base (RMSE={rmse_base:.3f})")
    ax.plot(
        t_days,
        pg_corr,
        "-o",
        color="tab:purple",
        label=f"Model+ML HOBIC strong (RMSE={rmse_corr:.3f})",
    )
    ax.set_xlabel("Day")
    ax.set_ylabel("P.g phibar")
    title = f"{condition} {cultivation} P.g ML correction HOBIC strong"
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    condition_clean = condition.replace(" ", "_")
    cultivation_clean = cultivation.replace(" ", "_")
    out_name = f"Fig_A12_Pg_ml_correction_{condition_clean}_{cultivation_clean}_HOBIC_strong.png"
    out_path = run_dir / out_name
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    resid_base = pg_data - pg_base
    resid_corr = pg_data - pg_corr
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    ax2.plot(t_days, resid_base, "-o", color="tab:blue", label="Data - Model base")
    ax2.plot(t_days, resid_corr, "-o", color="tab:red", label="Data - Model+ML HOBIC strong")
    ax2.set_xlabel("Day")
    ax2.set_ylabel("Residual (data - model)")
    title2 = f"{condition} {cultivation} P.g residuals HOBIC strong"
    ax2.set_title(title2)
    ax2.legend()
    fig2.tight_layout()
    out_name2 = f"Fig_A13_Pg_residual_{condition_clean}_{cultivation_clean}_HOBIC_strong.png"
    out_path2 = run_dir / out_name2
    fig2.savefig(out_path2, dpi=300, bbox_inches="tight")
    print(
        f"HOBIC strong: RMSE_all_base={rmse_base:.4f}, RMSE_all_corr={rmse_corr:.4f}, "
        f"RMSE_late2_base={rmse_base_late:.4f}, RMSE_late2_corr={rmse_corr_late:.4f}"
    )


def run_hobic_m5_scan(run_dir: Path):
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
    phi_init = np.array(config["phi_init"], dtype=float)
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
    def simulate(theta_vec):
        t_arr, g_arr = solver.solve(theta_vec)
        phi = g_arr[:, 0:5]
        psi = g_arr[:, 6:11]
        phibar = phi * psi
        t_model_days = model_time_to_days(t_arr, t_days)
        idx_map = [np.argmin(np.abs(t_model_days - d)) for d in t_days]
        phibar_obs = phibar[idx_map, :]
        return phibar_obs
    phibar_base = simulate(theta_full)
    pg_data = data_points[:, 4]
    pg_base = phibar_base[:, 4]
    theta_b5_up = theta_full.copy()
    theta_b5_up[15] *= 1.5
    phibar_b5 = simulate(theta_b5_up)
    pg_b5 = phibar_b5[:, 4]
    theta_m5_up = theta_full.copy()
    theta_m5_up[18] *= 1.5
    theta_m5_up[19] *= 1.5
    phibar_m5 = simulate(theta_m5_up)
    pg_m5 = phibar_m5[:, 4]
    theta_b5_m5_up = theta_full.copy()
    theta_b5_m5_up[15] *= 1.5
    theta_b5_m5_up[18] *= 1.5
    theta_b5_m5_up[19] *= 1.5
    phibar_b5_m5 = simulate(theta_b5_m5_up)
    pg_b5_m5 = phibar_b5_m5[:, 4]
    def rmse_all(y):
        return float(np.sqrt(np.mean((y - pg_data) ** 2)))
    def rmse_late(y):
        return float(np.sqrt(np.mean((y[-2:] - pg_data[-2:]) ** 2)))
    print("Base   RMSE_all", rmse_all(pg_base), "RMSE_late2", rmse_late(pg_base), "Pg_final", pg_base[-1])
    print("b5x1.5 RMSE_all", rmse_all(pg_b5), "RMSE_late2", rmse_late(pg_b5), "Pg_final", pg_b5[-1])
    print("M5x1.5 RMSE_all", rmse_all(pg_m5), "RMSE_late2", rmse_late(pg_m5), "Pg_final", pg_m5[-1])
    print(
        "b5&M5x1.5 RMSE_all",
        rmse_all(pg_b5_m5),
        "RMSE_late2",
        rmse_late(pg_b5_m5),
        "Pg_final",
        pg_b5_m5[-1],
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t_days, pg_data, "ko", label="Data")
    ax.plot(t_days, pg_base, "-o", color="black", label="Base")
    ax.plot(t_days, pg_b5, "-o", color="tab:blue", label="b5 x1.5")
    ax.plot(t_days, pg_m5, "-o", color="tab:green", label="M5 x1.5")
    ax.plot(t_days, pg_b5_m5, "-o", color="tab:red", label="b5&M5 x1.5")
    ax.set_xlabel("Day")
    ax.set_ylabel("P.g phibar")
    ax.set_title("Dysbiotic HOBIC Pg M4/M5 sensitivity")
    ax.legend()
    fig.tight_layout()
    cond = config.get("metadata", {}).get("condition", "Unknown")
    cult = config.get("metadata", {}).get("cultivation", "Unknown")
    cond_clean = str(cond).replace(" ", "_")
    cult_clean = str(cult).replace(" ", "_")
    out_name = f"Fig_A14_Pg_m5_scan_{cond_clean}_{cult_clean}.png"
    out_path = run_dir / out_name
    fig.savefig(out_path, dpi=300, bbox_inches="tight")


def run_hobic_sensitivity(run_dir: Path):
    X_h, y_h, w_h, t_days_h, phibar_h, data_h, cond_h, cult_h = build_hobic_strong_dataset(run_dir)
    mlp_h = PgingivalisSurgeMLP(input_dim=X_h.shape[1], hidden_dim=12, output_scale=0.4, seed=0)
    train_mlp_weighted(mlp_h, X_h, y_h, w_h, epochs=4000, lr=5e-2, l2=1e-4)
    idx = len(t_days_h) - 1
    x0 = X_h[idx].astype(float)
    base = float(mlp_h(x0))
    grads = []
    for j in range(len(x0)):
        step = 0.05 * max(abs(x0[j]), 1e-3)
        x1 = x0.copy()
        x1[j] += step
        y1 = float(mlp_h(x1))
        g = (y1 - base) / step
        grads.append(g)
    grads = np.array(grads, dtype=float)
    for name, g in zip(HOBIC_FEATURE_NAMES, grads):
        print(f"sensitivity[{name}] = {g:.4f}")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(grads)), grads)
    ax.set_xticks(range(len(grads)))
    ax.set_xticklabels(HOBIC_FEATURE_NAMES, rotation=45, ha="right")
    ax.set_ylabel("d(ML_resid)/d(feature)")
    ax.set_title("Dysbiotic HOBIC Pg ML sensitivity at final day")
    fig.tight_layout()
    cond_clean = str(cond_h).replace(" ", "_")
    cult_clean = str(cult_h).replace(" ", "_")
    out_name = f"Fig_A15_Pg_ml_sensitivity_{cond_clean}_{cult_clean}.png"
    out_path = run_dir / out_name
    fig.savefig(out_path, dpi=300, bbox_inches="tight")


def main():
    args = sys.argv[1:]
    if not args:
        print("Usage: python apply_pg_surge_ml_correction.py <run_dir1> [<run_dir2> ...]")
        sys.exit(1)

    mode = "joint"
    if args[0] == "--hobic-strong":
        mode = "hobic_strong"
        args = args[1:]
    elif args[0] == "--hobic-m5-scan":
        mode = "hobic_m5_scan"
        args = args[1:]
    elif args[0] == "--hobic-sensitivity":
        mode = "hobic_sensitivity"
        args = args[1:]
    if not args:
        print("Usage: python apply_pg_surge_ml_correction.py <run_dir1> [<run_dir2> ...]")
        sys.exit(1)

    run_dirs = [Path(p).resolve() for p in args]
    for rd in run_dirs:
        if not rd.exists():
            print(f"Error: run directory {rd} does not exist")
            sys.exit(1)

    if mode == "joint":
        X, y = build_dataset(run_dirs)
        mlp = PgingivalisSurgeMLP(input_dim=7, hidden_dim=8, output_scale=0.2, seed=0)
        train_mlp(mlp, X, y, epochs=4000, lr=5e-2)
        for rd in run_dirs:
            generate_correction_plot(rd, mlp)
    elif mode == "hobic_strong":
        run_dir = run_dirs[0]
        X_h, y_h, w_h, t_days_h, phibar_h, data_h, cond_h, cult_h = build_hobic_strong_dataset(
            run_dir
        )
        mlp_h = PgingivalisSurgeMLP(input_dim=X_h.shape[1], hidden_dim=12, output_scale=0.4, seed=0)
        train_mlp_weighted(mlp_h, X_h, y_h, w_h, epochs=4000, lr=5e-2, l2=1e-4)
        generate_correction_plot_hobic_strong(
            run_dir,
            mlp_h,
            X_h,
            t_days_h,
            phibar_h,
            data_h,
            cond_h,
            cult_h,
        )
    elif mode == "hobic_m5_scan":
        run_dir = run_dirs[0]
        run_hobic_m5_scan(run_dir)
    else:
        run_dir = run_dirs[0]
        run_hobic_sensitivity(run_dir)


if __name__ == "__main__":
    main()
