#!/usr/bin/env python3
"""
Evaluate DeepONet accuracy at θ_MAP for all 4 conditions.

Loads θ_MAP from TMCMC runs, compares ODE vs DeepONet output (φ trajectory).
Output: table of MSE, MAE, Rel.Err per condition and per species.
"""

import sys
import json
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "tmcmc" / "program2602"))
sys.path.insert(0, str(PROJECT_ROOT / "tmcmc"))

# Condition mappings
CONDITION_RUNS = {
    "Commensal_Static": "commensal_static",
    "Commensal_HOBIC": "commensal_hobic",
    "Dysbiotic_Static": "dysbiotic_static",
    "Dysbiotic_HOBIC": "dh_baseline",
}

CONDITION_CHECKPOINTS = {
    "Commensal_Static": "checkpoints_Commensal_Static",
    "Commensal_HOBIC": "checkpoints_Commensal_HOBIC",
    "Dysbiotic_Static": "checkpoints_DS_v2",
    "Dysbiotic_HOBIC": "checkpoints_Dysbiotic_HOBIC_50k",
}

# All available checkpoints per condition (for comparison)
ALL_CHECKPOINTS = {
    "Commensal_Static": [
        "checkpoints_Commensal_Static",
        "checkpoints_CS_v2",
        "checkpoints_CS_50k",
    ],
    "Commensal_HOBIC": [
        "checkpoints_Commensal_HOBIC",
        "checkpoints_CH_v2",
        "checkpoints_CH_50k",
    ],
    "Dysbiotic_Static": [
        "checkpoints_Dysbiotic_Static",
        "checkpoints_DS_v2",
    ],
    "Dysbiotic_HOBIC": [
        "checkpoints_Dysbiotic_HOBIC",
        "checkpoints_Dysbiotic_HOBIC_50k",
    ],
}

SPECIES = ["So", "An", "Vd", "Fn", "Pg"]


def load_theta_map(condition: str) -> np.ndarray:
    """Load θ_MAP from TMCMC runs. Returns (20,) or None."""
    dirname = CONDITION_RUNS.get(condition)
    if dirname is None:
        return None
    base = PROJECT_ROOT / "data_5species" / "_runs"
    for pattern in [base / dirname / "theta_MAP.json", base / dirname / "posterior" / "theta_MAP.json"]:
        if pattern.exists():
            with open(pattern) as f:
                data = json.load(f)
            if isinstance(data, dict):
                return np.array(data["theta_full"], dtype=np.float64)
            return np.array(data, dtype=np.float64)
    return None


def run_ode(theta: np.ndarray):
    """Run ODE solver, return (t, phi) or None."""
    from improved_5species_jit import BiofilmNewtonSolver5S

    solver = BiofilmNewtonSolver5S(maxtimestep=500, dt=1e-5)
    try:
        t_arr, g_arr = solver.run_deterministic(theta)
        phi = g_arr[:, :5]
        if np.any(~np.isfinite(phi)):
            return None
        return t_arr, phi
    except Exception:
        return None


def load_deeponet_and_predict(condition: str, theta: np.ndarray):
    """Load DeepONet, predict trajectory. Returns (n_time, 5) or None."""
    try:
        import jax.numpy as jnp
        import equinox as eqx
        from deeponet_hamilton import DeepONet

        ckpt_dir = SCRIPT_DIR / CONDITION_CHECKPOINTS.get(condition, "checkpoints")
        ckpt_path = ckpt_dir / "best.eqx"
        stats_path = ckpt_dir / "norm_stats.npz"
        if not ckpt_path.exists() or not stats_path.exists():
            return None

        config_path = ckpt_dir / "config.json"
        p, hidden, n_layers = 64, 128, 3
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
            p = cfg.get("p", p)
            hidden = cfg.get("hidden", hidden)
            n_layers = cfg.get("n_layers", n_layers)

        key = __import__("jax").random.PRNGKey(0)
        model = DeepONet(theta_dim=20, n_species=5, p=p, hidden=hidden, n_layers=n_layers, key=key)
        model = eqx.tree_deserialise_leaves(str(ckpt_path), model)

        stats = np.load(str(stats_path))
        theta_lo = np.array(stats["theta_lo"], dtype=np.float32)
        theta_width = np.array(stats["theta_width"], dtype=np.float32)
        theta_width[theta_width < 1e-12] = 1.0

        t_norm = np.linspace(0, 1, 100, dtype=np.float32)
        theta_norm = (theta - theta_lo) / theta_width
        theta_jnp = jnp.array(theta_norm, dtype=jnp.float32)
        t_jnp = jnp.array(t_norm, dtype=jnp.float32)

        phi_pred = model.predict_trajectory(theta_jnp, t_jnp)
        phi_pred = np.array(phi_pred)
        phi_pred = np.clip(phi_pred, 0.0, 1.0)
        return phi_pred
    except Exception as e:
        print(f"  DeepONet load/predict failed: {e}")
        return None


def compare_trajectories(phi_ode, phi_don, t_ode):
    """
    Compare ODE and DeepONet trajectories.
    ODE has variable t; DeepONet has 100 points t in [0,1].
    Interpolate ODE to DeepONet's grid.
    """
    n_don, _ = phi_don.shape
    t_don = np.linspace(0, 1, n_don)
    t_ode_norm = (t_ode - t_ode.min()) / (t_ode.max() - t_ode.min() + 1e-12)

    phi_ode_interp = np.zeros_like(phi_don)
    for j in range(5):
        phi_ode_interp[:, j] = np.interp(t_don, t_ode_norm, phi_ode[:, j])

    mse = np.mean((phi_ode_interp - phi_don) ** 2)
    mae = np.mean(np.abs(phi_ode_interp - phi_don))
    denom = np.mean(np.abs(phi_ode_interp)) + 1e-10
    rel_err = np.mean(np.abs(phi_ode_interp - phi_don)) / denom * 100

    return {
        "mse": mse,
        "mae": mae,
        "rel_err_pct": rel_err,
        "mse_per_species": np.mean((phi_ode_interp - phi_don) ** 2, axis=0),
        "mae_per_species": np.mean(np.abs(phi_ode_interp - phi_don), axis=0),
    }


def main():
    print("=" * 60)
    print("DeepONet MAP θ Accuracy Evaluation")
    print("=" * 60)

    runs_base = PROJECT_ROOT / "data_5species" / "_runs"
    results = []

    for cond in ["Commensal_Static", "Commensal_HOBIC", "Dysbiotic_Static", "Dysbiotic_HOBIC"]:
        print(f"\n--- {cond} ---")
        theta = load_theta_map(cond)
        if theta is None:
            print(f"  [SKIP] θ_MAP not found")
            results.append({"condition": cond, "status": "no_theta"})
            continue

        # ODE
        ode_out = run_ode(theta)
        if ode_out is None:
            print(f"  [SKIP] ODE solver failed")
            results.append({"condition": cond, "status": "ode_failed"})
            continue
        t_ode, phi_ode = ode_out

        # DeepONet
        phi_don = load_deeponet_and_predict(cond, theta)
        if phi_don is None:
            print(f"  [SKIP] DeepONet failed")
            results.append({"condition": cond, "status": "deeponet_failed"})
            continue

        # Compare
        metrics = compare_trajectories(phi_ode, phi_don, t_ode)
        results.append({"condition": cond, "status": "ok", **metrics})

        print(f"  MSE:      {metrics['mse']:.2e}")
        print(f"  MAE:      {metrics['mae']:.4f}")
        print(f"  Rel.Err:  {metrics['rel_err_pct']:.2f}%")
        for i, sp in enumerate(SPECIES):
            print(f"    {sp}: MSE={metrics['mse_per_species'][i]:.2e}, MAE={metrics['mae_per_species'][i]:.4f}")

    # Summary table
    print("\n" + "=" * 60)
    print("Summary Table")
    print("=" * 60)
    print(f"{'Condition':<22} {'Status':<12} {'MSE':>12} {'MAE':>8} {'Rel.Err%':>10}")
    print("-" * 64)
    for r in results:
        if r.get("status") == "ok":
            print(f"{r['condition']:<22} {'OK':<12} {r['mse']:>12.2e} {r['mae']:>8.4f} {r['rel_err_pct']:>10.2f}")
        else:
            print(f"{r['condition']:<22} {r['status']:<12}")

    # Save
    out_path = SCRIPT_DIR / "map_accuracy_results.json"
    save_dict = []
    for r in results:
        s = {"condition": r["condition"], "status": r["status"]}
        if r.get("status") == "ok":
            s["mse"] = float(r["mse"])
            s["mae"] = float(r["mae"])
            s["rel_err_pct"] = float(r["rel_err_pct"])
        save_dict.append(s)
    with open(out_path, "w") as f:
        json.dump(save_dict, f, indent=2)
    print(f"\nSaved: {out_path}")


def main_compare_all():
    """Compare all available checkpoints per condition."""
    print("=" * 80)
    print("DeepONet MAP θ Accuracy: ALL Checkpoints Comparison")
    print("=" * 80)

    all_results = []

    for cond in ["Commensal_Static", "Commensal_HOBIC", "Dysbiotic_Static", "Dysbiotic_HOBIC"]:
        theta = load_theta_map(cond)
        if theta is None:
            print(f"\n--- {cond}: θ_MAP not found ---")
            continue

        ode_out = run_ode(theta)
        if ode_out is None:
            print(f"\n--- {cond}: ODE solver failed ---")
            continue
        t_ode, phi_ode = ode_out

        print(f"\n--- {cond} ---")
        ckpts = ALL_CHECKPOINTS.get(cond, [])
        for ckpt_name in ckpts:
            ckpt_dir = SCRIPT_DIR / ckpt_name
            if not (ckpt_dir / "best.eqx").exists():
                print(f"  {ckpt_name:<35} [NOT FOUND]")
                continue

            # Temporarily override checkpoint
            old = CONDITION_CHECKPOINTS.get(cond)
            CONDITION_CHECKPOINTS[cond] = ckpt_name
            phi_don = load_deeponet_and_predict(cond, theta)
            CONDITION_CHECKPOINTS[cond] = old

            if phi_don is None:
                print(f"  {ckpt_name:<35} [PREDICT FAILED]")
                continue

            metrics = compare_trajectories(phi_ode, phi_don, t_ode)
            print(f"  {ckpt_name:<35} MSE={metrics['mse']:.2e}  MAE={metrics['mae']:.4f}  Rel={metrics['rel_err_pct']:.1f}%")
            all_results.append({
                "condition": cond,
                "checkpoint": ckpt_name,
                "mse": float(metrics["mse"]),
                "mae": float(metrics["mae"]),
                "rel_err_pct": float(metrics["rel_err_pct"]),
            })

    # Save
    out_path = SCRIPT_DIR / "map_accuracy_all_checkpoints.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Print best per condition
    print("\n" + "=" * 80)
    print("Best Checkpoint per Condition")
    print("=" * 80)
    for cond in ["Commensal_Static", "Commensal_HOBIC", "Dysbiotic_Static", "Dysbiotic_HOBIC"]:
        cond_results = [r for r in all_results if r["condition"] == cond]
        if cond_results:
            best = min(cond_results, key=lambda r: r["rel_err_pct"])
            print(f"  {cond:<22} → {best['checkpoint']:<35} Rel={best['rel_err_pct']:.1f}%")


if __name__ == "__main__":
    import sys
    if "--compare-all" in sys.argv:
        main_compare_all()
    else:
        main()
