#!/usr/bin/env python3
"""
estimate_siddiqui_6sp_jax.py — 6-species TMCMC on Siddiqui 2021 data.

Uses hamilton_ode_jax_6sp (27 params) + tmcmc_nuts_engine (vmap GPU).

Usage:
    python estimate_siddiqui_6sp_jax.py --dataset planktonic --n-particles 500 --device gpu
    python estimate_siddiqui_6sp_jax.py --dataset adherent --n-particles 10000 --device gpu
"""
from __future__ import annotations
import argparse, json, logging, os, sys
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent

# Device selection before JAX import
def _parse_device_early():
    for i, a in enumerate(sys.argv):
        if a == "--device" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return "auto"

_dev = _parse_device_early()
if _dev == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
elif _dev in ("auto", "gpu"):
    if "JAX_PLATFORMS" not in os.environ:
        os.environ["JAX_PLATFORMS"] = "cuda"
    try:
        import jax_cuda12_plugin  # noqa: F401
    except ImportError:
        pass

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(SCRIPT_DIR))
from hamilton_ode_jax_6sp import simulate_0d_6sp, N_PARAMS, N_PARAMS_BASE
from tmcmc_nuts_engine import tmcmc_engine

N_SP = 6
SP_NAMES = ["So", "An", "Aa", "Vp", "Fn", "Pg"]


def load_data_json(path, start_from_idx=1):
    """Load data, using first timepoint as IC and fitting from start_from_idx onward.

    Default: start_from_idx=1 means Day 0.25 is IC, fit Day 1+.
    This avoids the psi transient issue where phibar collapses in early steps.
    """
    with open(path) as f:
        d = json.load(f)
    data_all = np.array(d["data_frac"], dtype=np.float64)
    t_days_all = np.array(d["t_days"], dtype=np.float64)
    sigma = d.get("sigma_obs", 0.05)
    # IC from first timepoint
    phi_init = data_all[0].copy()
    phi_init = np.clip(phi_init, 0.005, 0.99)
    phi_init = phi_init / phi_init.sum()
    # Fit data from start_from_idx onward
    data = data_all[start_from_idx:]
    t_days = t_days_all[start_from_idx:]
    return data, t_days, sigma, phi_init


def convert_days_to_idx(t_days, dt, n_steps):
    t_max = n_steps * dt
    scale = (t_max * 0.95) / t_days.max()
    idx = np.round(t_days * scale / dt).astype(int)
    return np.clip(idx, 0, n_steps)


def make_sigma_schedule(t_days, base_sigma=0.10):
    """Per-timepoint sigma: early digitized bars have higher uncertainty.
    Base sigma raised to 0.10 to avoid overflow in TMCMC weights."""
    sigma = np.zeros(len(t_days))
    for i, d in enumerate(t_days):
        if d <= 0.25:
            sigma[i] = base_sigma * 2.0   # 0.20
        elif d <= 1.0:
            sigma[i] = base_sigma * 1.5   # 0.15
        elif d <= 3.0:
            sigma[i] = base_sigma * 1.2   # 0.12
        elif d <= 7.0:
            sigma[i] = base_sigma * 1.0   # 0.10
        elif d <= 14.0:
            sigma[i] = base_sigma * 0.9   # 0.09
        else:
            sigma[i] = base_sigma * 0.8   # 0.08
    return sigma


def make_log_likelihood_6sp(data, idx_sparse, sigma_obs, phi_init,
                             dt=5e-5, n_steps=5000, K_hill=0.15, n_hill=2.0,
                             c_const=25.0, lambda_pg=3.0, lambda_late=3.0, n_late=2,
                             sigma_schedule=None):
    obs = jnp.array(data, dtype=jnp.float64)
    phi_init_jax = jnp.array(phi_init, dtype=jnp.float64)
    idx = jnp.array(idx_sparse, dtype=jnp.int32)
    n_obs = data.shape[0]

    weights = jnp.ones((n_obs, N_SP))
    weights = weights.at[:, 5].set(lambda_pg)  # Pg = index 5
    # Fn (bridging species) gets mild emphasis
    weights = weights.at[:, 4].set(1.5)  # Fn = index 4
    # Late timepoints: mild emphasis only (avoid overflow)
    if n_late > 0 and n_obs > n_late:
        weights = weights.at[-n_late:, :].set(weights[-n_late:, :] * 1.5)

    # Per-timepoint sigma (n_obs,) broadcast to (n_obs, N_SP)
    if sigma_schedule is not None:
        sigma_arr = jnp.array(sigma_schedule, dtype=jnp.float64)[:, jnp.newaxis]
    else:
        sigma_arr = jnp.full((n_obs, 1), sigma_obs, dtype=jnp.float64)

    def log_likelihood(theta):
        # theta[27] = K_hill when 28 params (auto-detected in simulate_0d_6sp)
        phi_traj = simulate_0d_6sp(
            theta, n_steps=n_steps, dt=dt, phi_init=phi_init_jax,
            K_hill=0.15, n_hill=n_hill, c_const=c_const,  # K_hill kwarg is overridden by theta[27]
        )
        phi_pred = phi_traj[idx, :]
        phi_pred = jnp.clip(phi_pred, 1e-10, 1.0 - 1e-10)
        phi_sum = jnp.sum(phi_pred, axis=1, keepdims=True)
        phi_pred = phi_pred / jnp.maximum(phi_sum, 1e-12)
        residual = obs - phi_pred
        return -0.5 * jnp.sum(weights * (residual / sigma_arr) ** 2)

    return log_likelihood


def get_prior_bounds_6sp(use_heine_prior=False):
    """Prior bounds for 27 params.

    If use_heine_prior=True, uses DH MAP ± margin for the 15 shared A params
    and 5 shared b params, with wide bounds only for Aa-related params.

    5sp→6sp parameter mapping:
      5sp theta[0:20] = [a11,a12,a22,b1,b2, a33,a34,a44,b3,b4,
                          a13,a14,a23,a24, a55,b5, a15,a25,a35,a45]
      Species: 0=So,1=An,2=Vd/Vp,3=Fn,4=Pg

      6sp theta[0:27] = [a11,a12,a22,a13_Aa,a23_Aa,a33_Aa,  (So-An-Aa block)
                          a14_Vp,a24_Vp,a34_Aa_Vp,a44_Vp,    (Vp col)
                          a15_Fn,a25_Fn,a35_Aa_Fn,a45_Vp_Fn,a55_Fn,  (Fn col)
                          a16_Pg,a26_Pg,a36_Aa_Pg,a46_Vp_Pg,a56_Fn_Pg,a66_Pg,  (Pg col)
                          b1,b2,b3_Aa,b4_Vp,b5_Fn,b6_Pg]
    """
    # DH Phase 2 MAP (5 species, latest run)
    DH_MAP_5SP = np.array([
        1.674, 0.756, 0.012, 1.068, 2.233,   # a11,a12,a22,b1,b2
        -0.297, -0.769, 4.450, 5.133, 2.064,  # a33,a34,a44,b3,b4
        -2.728, 0.382, 0.458, 0.450,          # a13,a14,a23,a24
        3.434, 0.231,                          # a55,b5
        0.007, -0.494, -0.404, 5.631,         # a15,a25,a35,a45
    ])

    # 5sp index → 6sp index mapping (only for shared params)
    # 5sp: So=0, An=1, Vd=2, Fn=3, Pg=4
    # 6sp: So=0, An=1, Aa=2, Vp=3, Fn=4, Pg=5
    # Vd(5sp) maps to Vp(6sp), Fn(5sp index 3) maps to Fn(6sp index 4), Pg(5sp 4)→Pg(6sp 5)
    MAP_5TO6 = {
        # 5sp_idx: (6sp_idx, 5sp_MAP_value)
        0: (0, DH_MAP_5SP[0]),    # a(So-So)
        1: (1, DH_MAP_5SP[1]),    # a(So-An)
        2: (2, DH_MAP_5SP[2]),    # a(An-An)  → 6sp a22 (note: 6sp idx 2 is a(An-An) not a(An-Aa)!)
        # Wait - 6sp idx mapping is different. Let me be explicit:
        # 6sp: 0=a(So,So), 1=a(So,An), 2=a(An,An), 3=a(So,Aa), 4=a(An,Aa), 5=a(Aa,Aa)
        #       6=a(So,Vp), 7=a(An,Vp), 8=a(Aa,Vp), 9=a(Vp,Vp)
        #      10=a(So,Fn),11=a(An,Fn),12=a(Aa,Fn),13=a(Vp,Fn),14=a(Fn,Fn)
        #      15=a(So,Pg),16=a(An,Pg),17=a(Aa,Pg),18=a(Vp,Pg),19=a(Fn,Pg),20=a(Pg,Pg)
        #      21=b(So),22=b(An),23=b(Aa),24=b(Vp),25=b(Fn),26=b(Pg)
    }
    # 5sp indices → 6sp indices (excluding Aa-related)
    SHARED_MAP = {
        # 5sp → 6sp for A params
        0: 0,   # a(So,So)→a(So,So)
        1: 1,   # a(So,An)→a(So,An)
        2: 2,   # a(An,An)→a(An,An)
        5: 9,   # a(Vd,Vd)→a(Vp,Vp)
        6: 13,  # a(Vd,Fn)→a(Vp,Fn)
        7: 14,  # a(Fn,Fn)→a(Fn,Fn)
        10: 6,  # a(So,Vd)→a(So,Vp)
        11: 10, # a(So,Fn)→a(So,Fn)
        12: 7,  # a(An,Vd)→a(An,Vp)
        13: 11, # a(An,Fn)→a(An,Fn)
        14: 20, # a(Pg,Pg)→a(Pg,Pg)
        16: 15, # a(So,Pg)→a(So,Pg)
        17: 16, # a(An,Pg)→a(An,Pg)
        18: 18, # a(Vd,Pg)→a(Vp,Pg)
        19: 19, # a(Fn,Pg)→a(Fn,Pg)
        # b params
        3: 21,  # b(So)→b(So)
        4: 22,  # b(An)→b(An)
        8: 24,  # b(Vd)→b(Vp)
        9: 25,  # b(Fn)→b(Fn)
        15: 26, # b(Pg)→b(Pg)
    }

    bounds = np.zeros((N_PARAMS, 2))

    if use_heine_prior:
        # Aa-related indices (wide prior)
        aa_indices = {3, 4, 5, 8, 12, 17, 23}  # a(So,Aa),a(An,Aa),a(Aa,Aa),a(Aa,Vp),a(Aa,Fn),a(Aa,Pg),b(Aa)

        for i in range(N_PARAMS):
            if i in aa_indices:
                # Aa-related: wide prior
                if i == 23:  # b(Aa)
                    bounds[i] = [0.0, 4.0]
                else:
                    bounds[i] = [-2.0, 3.0]
            else:
                # Check if this 6sp index has a 5sp MAP value
                found = False
                for sp5_idx, sp6_idx in SHARED_MAP.items():
                    if sp6_idx == i:
                        map_val = DH_MAP_5SP[sp5_idx]
                        margin = 3.0  # ± 3.0 around 5sp MAP
                        lo = map_val - margin
                        hi = map_val + margin
                        if i >= 21:  # growth rates must be >= 0
                            lo = max(lo, 0.0)
                        bounds[i] = [lo, hi]
                        found = True
                        break
                if not found:
                    # Unmapped (shouldn't happen, but fallback)
                    bounds[i] = [-2.0, 4.0] if i < 21 else [0.0, 6.0]
    else:
        for i in range(21):
            bounds[i] = [-2.0, 4.0]
        for i in range(21, 27):
            bounds[i] = [0.0, 6.0]

    # theta[27] = K_hill (free)
    bounds[27] = [0.01, 0.50]  # K_hill range from sensitivity analysis

    return bounds


def main():
    parser = argparse.ArgumentParser(description="6-species TMCMC on Siddiqui 2021")
    parser.add_argument("--dataset", default="planktonic", choices=["planktonic", "adherent", "both"])
    parser.add_argument("--n-particles", type=int, default=5000)
    parser.add_argument("--max-stages", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-mutation-steps", type=int, default=20)
    parser.add_argument("--sigma-scale", type=float, default=3.0)
    parser.add_argument("--lambda-pg", type=float, default=3.0)
    parser.add_argument("--lambda-late", type=float, default=3.0)
    parser.add_argument("--K-hill", type=float, default=0.15)
    parser.add_argument("--n-hill", type=float, default=2.0)
    parser.add_argument("--dt", type=float, default=1e-4)
    parser.add_argument("--n-steps", type=int, default=2500)
    parser.add_argument("--per-timepoint-sigma", action="store_true",
                        help="Use per-timepoint sigma schedule (higher σ for early digitized points)")
    parser.add_argument("--heine-prior", action="store_true", default=True,
                        help="Use Heine 2025 DH MAP as informative prior (default: on)")
    parser.add_argument("--no-heine-prior", dest="heine_prior", action="store_false",
                        help="Disable Heine informative prior, use wide bounds")
    parser.add_argument("--max-delta-beta", type=float, default=0.1,
                        help="Cap beta increment per stage (default: 0.1, forces ≥10 stages)")
    parser.add_argument("--mutation", default="rw", choices=["rw", "hmc", "nuts"])
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    args = parser.parse_args()

    if args.quick:
        args.n_particles = 50
        args.n_steps = 500
        logger.info("Quick mode: 50p, 500 steps")

    devs = jax.devices()
    logger.info(f"JAX devices: {[str(d) for d in devs]}")

    datasets = ["planktonic", "adherent"] if args.dataset == "both" else [args.dataset]

    for ds_name in datasets:
        logger.info(f"\n{'='*60}")
        logger.info(f"6-species TMCMC: Siddiqui 2021 — {ds_name}")
        logger.info(f"{'='*60}")

        json_path = SCRIPT_DIR / f"siddiqui2021_{ds_name}_6sp.json"
        data, t_days, sigma_default, phi_init = load_data_json(json_path)
        sigma_obs = sigma_default * args.sigma_scale

        # Per-timepoint sigma schedule
        sigma_sched = None
        if args.per_timepoint_sigma:
            sigma_sched = make_sigma_schedule(t_days, base_sigma=sigma_default)
            logger.info(f"Data: {data.shape}, per-timepoint σ: {[f'{s:.3f}' for s in sigma_sched]}")
        else:
            logger.info(f"Data: {data.shape}, sigma_obs={sigma_obs:.4f}")

        logger.info(f"phi_init: {' '.join(f'{SP_NAMES[j]}={phi_init[j]:.3f}' for j in range(N_SP))}")

        for i, d in enumerate(t_days):
            logger.info(f"  Day {d:5.2f}: {' '.join(f'{SP_NAMES[j]}={data[i,j]:.3f}' for j in range(N_SP))}")

        idx_sparse = convert_days_to_idx(t_days, args.dt, args.n_steps)
        logger.info(f"idx_sparse: {idx_sparse}")

        log_likelihood = make_log_likelihood_6sp(
            data=data, idx_sparse=idx_sparse, sigma_obs=sigma_obs,
            phi_init=phi_init, dt=args.dt, n_steps=args.n_steps,
            K_hill=args.K_hill, n_hill=args.n_hill,
            lambda_pg=args.lambda_pg, lambda_late=args.lambda_late,
            sigma_schedule=sigma_sched,
        )

        prior_bounds = get_prior_bounds_6sp(use_heine_prior=args.heine_prior)
        if args.heine_prior:
            logger.info("Using Heine DH MAP as informative prior (shared params ± 3.0)")
        prior_bounds = np.array(prior_bounds, dtype=np.float32)

        logger.info("JIT warmup (6sp)...")
        _ = jax.jit(log_likelihood)(jnp.zeros(N_PARAMS, dtype=jnp.float64))
        if args.mutation in ("hmc", "nuts"):
            _ = jax.jit(jax.value_and_grad(log_likelihood))(jnp.zeros(N_PARAMS, dtype=jnp.float64))
        logger.info("Warmup OK")

        logger.info(f"Running {args.mutation.upper()}-TMCMC ({args.n_particles}p, {N_PARAMS} params)...")
        result = tmcmc_engine(
            log_likelihood, prior_bounds,
            mutation=args.mutation,
            n_particles=args.n_particles,
            max_stages=args.max_stages,
            seed=args.seed,
            nuts_max_depth=6,
            n_mutation_steps=args.n_mutation_steps,
            max_delta_beta=args.max_delta_beta,
        )

        theta_MAP = result["theta_MAP"]
        logger.info(
            f"Done: {result['n_stages']} stages, "
            f"time={result['total_time']:.1f}s, "
            f"accept={np.mean(result['accept_rates']):.2f}, "
            f"max logL={result['log_likelihoods'].max():.1f}"
        )

        # Save
        default_out = SCRIPT_DIR / "_runs" / f"siddiqui_6sp_{ds_name}_{args.n_particles}p_best"
        out_dir = Path(args.output_dir) if args.output_dir else default_out
        out_dir.mkdir(parents=True, exist_ok=True)

        np.save(out_dir / "samples.npy", result["samples"])
        np.save(out_dir / "logL.npy", result["log_likelihoods"])
        np.save(out_dir / "betas.npy", result["betas"])
        with open(out_dir / "theta_MAP.json", "w") as f:
            json.dump({f"theta_{i}": float(v) for i, v in enumerate(theta_MAP)}, f, indent=2)

        config = {
            "paper": "Siddiqui et al. 2021 (PMC8828709)",
            "dataset": ds_name,
            "n_species": N_SP,
            "n_params": N_PARAMS,
            "n_particles": args.n_particles,
            "mutation": args.mutation,
            "n_mutation_steps": args.n_mutation_steps,
            "sigma_obs": sigma_obs,
            "sigma_schedule": sigma_sched.tolist() if sigma_sched is not None else None,
            "n_steps": args.n_steps,
            "dt": args.dt,
            "K_hill": args.K_hill,
            "n_hill": args.n_hill,
            "prior_a_ij": [-2.0, 4.0],
            "prior_mu": [0.0, 6.0],
            "lambda_pg": args.lambda_pg,
            "lambda_fn": 1.5,
            "n_stages": result["n_stages"],
            "total_time_s": result["total_time"],
            "max_logL": float(result["log_likelihoods"].max()),
            "strategy": "best_v2: K_hill=0.15, dt=5e-5, n_steps=5000, wider prior, per-tp sigma, Fn weight",
        }
        with open(out_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Print MAP
        param_names = [
            "a(So-So)", "a(So-An)", "a(An-An)", "a(So-Aa)", "a(An-Aa)", "a(Aa-Aa)",
            "a(So-Vp)", "a(An-Vp)", "a(Aa-Vp)", "a(Vp-Vp)",
            "a(So-Fn)", "a(An-Fn)", "a(Aa-Fn)", "a(Vp-Fn)", "a(Fn-Fn)",
            "a(So-Pg)", "a(An-Pg)", "a(Aa-Pg)", "a(Vp-Pg)", "a(Fn-Pg)", "a(Pg-Pg)",
            "μ(So)", "μ(An)", "μ(Aa)", "μ(Vp)", "μ(Fn)", "μ(Pg)",
            "K_hill",
        ]
        logger.info("\n--- MAP theta (27 params) ---")
        for i, (name, val) in enumerate(zip(param_names, theta_MAP)):
            logger.info(f"  {i:2d} {name:12s} = {val:+.4f}")

        logger.info(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
