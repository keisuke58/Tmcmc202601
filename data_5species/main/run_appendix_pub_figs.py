#!/usr/bin/env python3
"""
Appendix figure generator: uses the SAME scripts as the paper figures.

Converts JAX TMCMC run directories to the CPU-solver compatible format,
then calls generate_pub_map_plot.py and generate_pub_plots.py exactly as
they are called for the main paper figures.

Usage:
    python run_appendix_pub_figs.py --output-dir figures_appendix
    python run_appendix_pub_figs.py --output-dir figures_appendix \
        --run-CH /path/to/ch --run-CS /path/to/cs \
        --run-DH /path/to/dh --run-DS /path/to/ds
"""

import argparse
import json
import subprocess
import sys
import shutil
import tempfile
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = DATA_DIR.parent.parent

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(DATA_DIR))

from estimate_reduced_nishioka import convert_days_to_model_time, load_experimental_data


def jax_theta_to_cpu_theta(jax_theta: np.ndarray) -> np.ndarray:
    """
    Remap 20-param JAX theta to CPU BiofilmNewtonSolver5S theta format.

    JAX format (hamilton_ode_jax.py theta_to_matrices):
      [0..2]  A[0,0] A[0,1] A[1,1]         (So-An symmetric block)
      [3..5]  A[2,2] A[2,3] A[3,3]         (Vd-Fn symmetric block)
      [6..9]  A[0,2] A[0,3] A[1,2] A[1,3]  (cross terms)
      [10]    A[4,4]                         (Pg self)
      [11..14] A[0,4] A[1,4] A[2,4] A[3,4] (Pg cross)
      [15..19] unused phantom params         (b always 0 in JAX)

    CPU format (improved_5species_jit.py theta_to_matrices):
      [0..2]  A[0,0] A[0,1] A[1,1]
      [3,4]   b[0]   b[1]            <- growth rates (zero; alpha_const=0)
      [5..7]  A[2,2] A[2,3] A[3,3]
      [8,9]   b[2]   b[3]            <- growth rates (zero)
      [10..13] A[0,2] A[0,3] A[1,2] A[1,3]
      [14]    A[4,4]
      [15]    b[4]                   <- growth rate (zero)
      [16..19] A[0,4] A[1,4] A[2,4] A[3,4]
    """
    cpu = np.zeros(20, dtype=np.float64)
    cpu[0] = jax_theta[0]  # A[0,0]
    cpu[1] = jax_theta[1]  # A[0,1] = A[1,0]
    cpu[2] = jax_theta[2]  # A[1,1]
    cpu[3] = 0.0  # b[0] = 0  (JAX has b=0, alpha_const=0 → no effect)
    cpu[4] = 0.0  # b[1] = 0
    cpu[5] = jax_theta[3]  # A[2,2]
    cpu[6] = jax_theta[4]  # A[2,3] = A[3,2]
    cpu[7] = jax_theta[5]  # A[3,3]
    cpu[8] = 0.0  # b[2] = 0
    cpu[9] = 0.0  # b[3] = 0
    cpu[10] = jax_theta[6]  # A[0,2] = A[2,0]
    cpu[11] = jax_theta[7]  # A[0,3] = A[3,0]
    cpu[12] = jax_theta[8]  # A[1,2] = A[2,1]
    cpu[13] = jax_theta[9]  # A[1,3] = A[3,1]
    cpu[14] = jax_theta[10]  # A[4,4]
    cpu[15] = 0.0  # b[4] = 0
    cpu[16] = jax_theta[11]  # A[0,4] = A[4,0]
    cpu[17] = jax_theta[12]  # A[1,4] = A[4,1]
    cpu[18] = jax_theta[13]  # A[2,4] = A[4,2]
    cpu[19] = jax_theta[14]  # A[3,4] = A[4,3]
    return cpu


def best_jax_run(condition: str, cultivation: str, runs_dir: Path = None) -> Path:
    """Find the JAX run dir with highest logL for this condition."""
    if runs_dir is None:
        runs_dir = SCRIPT_DIR / "_runs"
    tag = f"jax_ode_nuts_{condition}_{cultivation}_"
    candidates = [
        p
        for p in runs_dir.iterdir()
        if p.is_dir()
        and tag in p.name
        and (p / "theta_MAP.json").exists()
        and (p / "logL.npy").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No completed JAX run for {condition}_{cultivation} in {runs_dir}")
    return max(candidates, key=lambda p: np.load(p / "logL.npy").max())


def prepare_compatible_dir(
    jax_run_dir: Path,
    dest_dir: Path,
    condition: str,
    cultivation: str,
    K_hill: float = 0.05,
    n_hill: float = 4.0,
) -> Path:
    """
    Create a CPU-solver compatible run dir from a JAX run dir.
    Returns the dest_dir path.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    # --- Load JAX theta and convert ---
    with open(jax_run_dir / "theta_MAP.json") as f:
        jax_d = json.load(f)
    jax_theta = np.array([jax_d[str(i)] for i in range(20)], dtype=np.float64)
    cpu_theta = jax_theta_to_cpu_theta(jax_theta)

    theta_map = {
        "theta_sub": cpu_theta.tolist(),
        "theta_full": cpu_theta.tolist(),
        "active_indices": list(range(20)),
    }
    with open(dest_dir / "theta_MAP.json", "w") as f:
        json.dump(theta_map, f, indent=2)

    # --- Load experimental data ---
    data, t_days, sigma_obs_est, phi_init_exp, metadata = load_experimental_data(
        DATA_DIR, condition, cultivation, normalize=True
    )
    phi_init = np.clip(phi_init_exp, 0.01, 0.99)
    phi_init = phi_init / phi_init.sum()

    np.save(dest_dir / "data.npy", data)
    np.save(dest_dir / "t_days.npy", t_days)

    # --- Build config.json ---
    dt = 1e-4
    n_steps = 2500
    t_model, idx_sparse = convert_days_to_model_time(t_days, dt, n_steps)
    config = {
        "condition": condition,
        "cultivation": cultivation,
        "dt": dt,
        "maxtimestep": n_steps,
        "c_const": 25.0,
        "Kp1": 1e-4,
        "K_hill": K_hill,
        "n_hill": n_hill,
        "alpha_const": 0.0,
        "phi_init": phi_init.tolist(),
        "phi_init_is_array": True,
        "use_exp_init": True,
        "sigma_obs": float(np.mean(sigma_obs_est)),
        "metadata": {
            "condition": condition,
            "cultivation": cultivation,
            "days": t_days.tolist(),
            "n_timepoints": int(len(t_days)),
            "idx_sparse": idx_sparse.tolist(),
            "source": str(jax_run_dir),
        },
    }
    with open(dest_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    return dest_dir


PYTHON = sys.executable
PUB_MAP = str(SCRIPT_DIR / "generate_pub_map_plot.py")


def run_pub_scripts(compat_dir: Path, out_dir: Path):
    """
    Call generate_pub_map_plot.py (fitting figure) and
    plot_interaction_heatmap() from generate_pub_plots.py (A matrix).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Fitting figure (same script as paper) ---
    print("  Running generate_pub_map_plot.py ...")
    result = subprocess.run(
        [PYTHON, PUB_MAP, str(compat_dir), str(out_dir)], capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  [WARN] map_fit failed:\n{result.stderr[-2000:]}")
    else:
        print(result.stdout.strip())

    # --- Interaction heatmap (direct import to avoid UQ dependency) ---
    print("  Running plot_interaction_heatmap() from generate_pub_plots.py ...")
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "generate_pub_plots", str(SCRIPT_DIR / "generate_pub_plots.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.plot_interaction_heatmap(str(compat_dir), str(out_dir))
        print("  Heatmap generated.")
    except Exception as e:
        print(f"  [WARN] heatmap failed: {e}")


CONDITIONS = [
    ("Commensal", "HOBIC", "CH"),
    ("Commensal", "Static", "CS"),
    ("Dysbiotic", "HOBIC", "DH"),
    ("Dysbiotic", "Static", "DS"),
]


def main():
    parser = argparse.ArgumentParser(
        description="Generate appendix figures using the paper's existing fig scripts"
    )
    parser.add_argument(
        "--output-dir", default="figures_appendix", help="Final output directory for figures"
    )
    parser.add_argument(
        "--runs-dir",
        default=None,
        help="Directory with jax_ode_nuts_* run dirs (default: _runs next to this script)",
    )
    parser.add_argument("--run-CH", default=None, help="Explicit JAX run dir for Commensal HOBIC")
    parser.add_argument("--run-CS", default=None, help="Explicit JAX run dir for Commensal Static")
    parser.add_argument("--run-DH", default=None, help="Explicit JAX run dir for Dysbiotic HOBIC")
    parser.add_argument("--run-DS", default=None, help="Explicit JAX run dir for Dysbiotic Static")
    parser.add_argument(
        "--keep-tmp",
        action="store_true",
        help="Keep temporary compatible run dirs after generation",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs_dir = Path(args.runs_dir) if args.runs_dir else None
    explicit = {"CH": args.run_CH, "CS": args.run_CS, "DH": args.run_DH, "DS": args.run_DS}

    tmp_root = Path(tempfile.mkdtemp(prefix="jax_compat_"))
    print(f"Temp dir: {tmp_root}")

    for condition, cultivation, key in CONDITIONS:
        print(f"\n=== {key} ({condition} {cultivation}) ===")

        # Resolve JAX run dir
        if explicit[key]:
            jax_dir = Path(explicit[key])
        else:
            jax_dir = best_jax_run(condition, cultivation, runs_dir)
        logL_best = np.load(jax_dir / "logL.npy").max()
        print(f"  JAX run: {jax_dir.name}  (logL={logL_best:.3f})")

        # Prepare compatible dir
        compat_dir = tmp_root / f"{condition}_{cultivation}"
        prepare_compatible_dir(jax_dir, compat_dir, condition, cultivation)

        # Per-condition output
        cond_out = output_dir / f"{condition}_{cultivation}"
        cond_out.mkdir(exist_ok=True)
        run_pub_scripts(compat_dir, cond_out)

    if not args.keep_tmp:
        shutil.rmtree(tmp_root, ignore_errors=True)

    print(f"\nDone. Figures in: {output_dir}/")
    for condition, cultivation, key in CONDITIONS:
        d = output_dir / f"{condition}_{cultivation}"
        files = list(d.glob("*.png")) + list(d.glob("*.pdf"))
        for f in sorted(files):
            print(f"  {f}")


if __name__ == "__main__":
    main()
