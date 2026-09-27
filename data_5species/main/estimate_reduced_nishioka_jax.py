#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
estimate_reduced_nishioka_jax.py — TMCMC with JAX ODE + NUTS.

Uses pure JAX Hamilton ODE (hamilton_ode_jax) instead of DeepONet.
Enables NUTS mutation with exact gradients, no surrogate approximation.

Usage:
    cd data_5species/main
    python estimate_reduced_nishioka_jax.py --condition Dysbiotic --cultivation HOBIC \\
        --n-particles 200 --use-exp-init

Requires: jax, jaxlib (e.g. conda env klempt_fem2)

  PYTHON=$HOME/.pyenv/versions/miniconda3-latest/envs/klempt_fem2/bin/python
  $PYTHON estimate_reduced_nishioka_jax.py --condition Dysbiotic --cultivation HOBIC ...
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

# Project root for imports
SCRIPT_DIR = Path(__file__).resolve().parent
MAIN_DIR = SCRIPT_DIR
DATA_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = DATA_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# Device selection MUST be before jax import
def _parse_device_early():
    for i, a in enumerate(sys.argv):
        if a == "--device" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return "auto"


_device_early = _parse_device_early()
if _device_early == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
elif _device_early in ("auto", "gpu"):
    # CUDA を優先: JAX_PLATFORMS が未設定なら cuda を明示（ROCM 誤検出回避）
    if "JAX_PLATFORMS" not in os.environ:
        os.environ["JAX_PLATFORMS"] = "cuda"
    # jax-cuda12-plugin を JAX より先にロード（CUDA 初期化を確実に）
    try:
        import jax_cuda12_plugin  # noqa: F401
    except ImportError:
        pass  # CPU-only jaxlib の場合は無視

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Local imports
from hamilton_ode_jax import simulate_0d

# Import data loading and bounds from main estimator
from estimate_reduced_nishioka import (
    convert_days_to_model_time,
    load_experimental_data,
)
from core.nishioka_model import get_condition_bounds

from tmcmc_nuts_engine import tmcmc_engine

# eHOMD/Dieckow SF1 sign constraints (global theta index, sign, weight)
# Same as SignPrior._CONSTRAINTS_EHOMD in core/evaluator.py
_EHOMD_CONSTRAINTS = [
    (1, +1, 2.0),  # A[So,An]
    (6, +1, 2.0),  # A[Vd,Fn]
    (10, +1, 4.0),  # A[So,Vd]
    (11, +1, 2.0),  # A[So,Fn]
    (12, +1, 2.0),  # A[An,Vd]
    (13, +1, 2.0),  # A[An,Fn]
    (16, -1, 1.0),  # A[So,Pg]
    (17, +1, 1.0),  # A[An,Pg]
    (18, +1, 3.0),  # A[Vd,Pg]
    (19, +1, 2.0),  # A[Fn,Pg]
]


def make_log_likelihood_jax_ode(
    data: np.ndarray,
    t_days: np.ndarray,
    idx_sparse: np.ndarray,
    sigma_obs: float,
    phi_init: np.ndarray,
    dt: float = 1e-4,
    n_steps: int = 2500,
    K_hill: float = 0.05,
    n_hill: float = 2.0,
    c_const: float = 25.0,
    lambda_pg: float = 1.0,
    lambda_late: float = 1.0,
    n_late: int = 2,
    sign_prior: bool = False,
    sign_lambda: float = 0.1,
    lambda_bc: float = 0.0,
):
    """
    Build JAX-differentiable log-likelihood using Hamilton ODE.

    Returns
    -------
    log_likelihood : callable(theta) -> scalar
    """
    obs = jnp.array(data, dtype=jnp.float64)
    phi_init_jax = jnp.array(phi_init, dtype=jnp.float64)
    idx = jnp.array(idx_sparse, dtype=jnp.int32)
    n_obs, n_species = data.shape

    # Weights: lambda_pg for Pg (species 4), lambda_late for last n_late timepoints
    weights = jnp.ones((n_obs, n_species))
    weights = weights.at[:, 4].set(lambda_pg)
    if n_late > 0:
        late_slice = slice(-n_late, None)
        weights = weights.at[late_slice, :].set(weights[late_slice, :] * lambda_late)

    def log_likelihood(theta):
        phi_traj = simulate_0d(
            theta,
            n_steps=n_steps,
            dt=dt,
            phi_init=phi_init_jax,
            K_hill=K_hill,
            n_hill=n_hill,
            c_const=c_const,
        )
        # Sample at observation indices
        phi_pred = phi_traj[idx, :]  # (n_obs, 5)
        phi_pred = jnp.clip(phi_pred, 1e-10, 1.0 - 1e-10)
        # Normalize to fractions
        phi_sum = jnp.sum(phi_pred, axis=1, keepdims=True)
        phi_pred = phi_pred / jnp.maximum(phi_sum, 1e-12)
        residual = obs - phi_pred
        logL = -0.5 * jnp.sum(weights * (residual / sigma_obs) ** 2)

        # eHOMD sign prior: logL -= lam * weight * max(0, -sign*theta)^2
        if sign_prior:
            for idx_c, sign, weight in _EHOMD_CONSTRAINTS:
                violation = jnp.maximum(0.0, -sign * theta[idx_c])
                logL -= sign_lambda * weight * violation * violation

        # Bray-Curtis dissimilarity penalty
        if lambda_bc > 0.0:
            q_norm = obs / jnp.maximum(jnp.sum(obs, axis=1, keepdims=True), 1e-12)
            bc_per_t = 1.0 - jnp.sum(jnp.minimum(phi_pred, q_norm), axis=1)
            logL -= lambda_bc * jnp.mean(bc_per_t)

        return logL

    return log_likelihood


def load_prior_bounds(condition: str, cultivation: str) -> np.ndarray:
    """Get prior bounds as (20, 2) array."""
    bounds, _ = get_condition_bounds(condition, cultivation)
    return np.array(bounds[:20], dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description="TMCMC with JAX ODE + NUTS (no DeepONet)")
    parser.add_argument("--condition", default="Dysbiotic")
    parser.add_argument("--cultivation", default="HOBIC")
    parser.add_argument("--n-particles", type=int, default=200)
    parser.add_argument("--max-stages", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-exp-init", action="store_true")
    parser.add_argument("--start-from-day", type=int, default=1)
    parser.add_argument("--lambda-pg", type=float, default=5.0)
    parser.add_argument("--lambda-late", type=float, default=3.0)
    parser.add_argument("--sigma-scale", type=float, default=1.0)
    parser.add_argument(
        "--estimate-b",
        action="store_true",
        help=(
            "b (theta[3,4,8,9,15]) も推定する。既定では 0 に固定して探索から外す。"
            "論文 Sec.2 のとおり抗生物質が無い実験では alpha*=0 であり、"
            "b は alpha を掛けられて動力学から完全に消えるため（hamilton_ode_jax.py: "
            "t2 = b_diag[i] * alpha / Eta[i] * psi）、尤度に一切入らない。"
            "推定すると 20 次元中 5 次元が尤度勾配ゼロのまま提案共分散に混ざる。"
        ),
    )
    parser.add_argument(
        "--prior-scale",
        type=float,
        default=0.0,
        help=(
            "0 より大きいと A の15成分に平均0・標準偏差この値の正規事前分布を置く"
            "（弱情報事前分布）。0 なら従来どおり箱一様。"
            "尤度に縮退方向があり箱の端で止まる成分があるため導入した: "
            "DH の a35 (Vei-Pg) は事前分布を [-4,8] から [-15,20] に広げても "
            "新しい境界に張り付き chi が 0.6233 -> 0.5104 と改善し続ける。"
            "有限の最適値を持たないので箱一様では事前分布の選択が結果を決めてしまう。"
        ),
    )
    parser.add_argument("--K-hill", type=float, default=0.05)
    parser.add_argument("--n-hill", type=float, default=2.0)
    parser.add_argument("--dt", type=float, default=1e-4)
    parser.add_argument("--n-steps", type=int, default=2500)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--mutation", default="nuts", choices=["rw", "hmc", "nuts"])
    parser.add_argument(
        "--n-mutation-steps", type=int, default=1, help="Mutation steps per particle per stage"
    )
    parser.add_argument(
        "--external-data",
        type=str,
        default=None,
        help="Path to JSON with external data (e.g. Sanz-Martin). "
        "Keys: t_days, data (n_obs x 5), phi_init (5,), sigma_obs",
    )
    parser.add_argument(
        "--wide-prior",
        action="store_true",
        help="Use wide prior bounds [-1,3] instead of condition-specific",
    )
    parser.add_argument("--quick", action="store_true", help="Test run: 50p, 500 steps")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Ultra-short benchmark: 20p, 500 steps, 3 stages (~1min each)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="Device: auto (use GPU if available), cpu, gpu",
    )
    parser.add_argument(
        "--sign-prior", action="store_true", help="Enable eHOMD/Dieckow SF1 metabolic sign prior"
    )
    parser.add_argument(
        "--sign-lambda", type=float, default=0.1, help="Sign prior penalty weight (default: 0.1)"
    )
    parser.add_argument(
        "--bc-lambda",
        type=float,
        default=0.0,
        help="Bray-Curtis dissimilarity penalty weight (0=disabled, recommend 5-20)",
    )
    parser.add_argument(
        "--no-polish",
        action="store_true",
        help="Skip L-BFGS-B MAP polish step after TMCMC",
    )
    parser.add_argument(
        "--polish-top-k",
        type=int,
        default=2,
        help="Number of top particles to use as L-BFGS-B starting points (default: 2)",
    )
    args = parser.parse_args()

    if args.quick:
        args.n_particles = 50
        args.n_steps = 500
        logger.info("Quick mode: n_particles=50, n_steps=500")
    if args.benchmark:
        args.n_particles = 20
        args.n_steps = 500
        args.max_stages = 3
        logger.info("Benchmark mode: n_particles=20, n_steps=500, max_stages=3")

    devs = jax.devices()
    has_gpu = any("cuda" in str(d).lower() or "gpu" in str(d).lower() for d in devs)
    logger.info(f"JAX devices: {[str(d) for d in devs]} (--device={args.device})")
    if args.device == "gpu" and not has_gpu:
        raise RuntimeError(
            "GPU が利用できません。以下を確認してください:\n"
            "  1. pip install jax[cuda12] または jax-cuda12-plugin が入っているか\n"
            "  2. nvidia-smi で GPU が認識されているか\n"
            '  3. python -c "import jax; print(jax.devices())" でデバイス確認\n'
            "  4. LD_LIBRARY_PATH がシステム CUDA を指しており pip の nvidia-* と競合していないか"
        )

    if args.external_data:
        logger.info(f"Loading external data from {args.external_data}")
        with open(args.external_data) as f:
            ext = json.load(f)
        data = np.array(ext["data"], dtype=np.float64)
        t_days = np.array(ext["t_days"], dtype=np.float64)
        sigma_obs = ext.get("sigma_obs", 0.05) * args.sigma_scale
        phi_init = np.array(ext["phi_init"], dtype=np.float64)
        phi_init = np.clip(phi_init, 0.01, 0.99)
        phi_init = phi_init / phi_init.sum()
        logger.info(f"External data: {data.shape}, sigma_obs={np.mean(sigma_obs):.4f}")
    else:
        logger.info("Loading experimental data...")
        data, t_days, sigma_obs_est, phi_init_exp, metadata = load_experimental_data(
            DATA_DIR,
            args.condition,
            args.cultivation,
            args.start_from_day,
            normalize=True,
        )
        sigma_obs = sigma_obs_est * args.sigma_scale
        phi_init = phi_init_exp if args.use_exp_init else np.full(5, 0.2)
        if args.use_exp_init:
            total = phi_init.sum()
            if total > 0:
                phi_init = phi_init / total
            phi_init = np.clip(phi_init, 0.01, 0.99)
    logger.info(f"Data: {data.shape}, sigma_obs={np.mean(sigma_obs):.4f}")

    t_model, idx_sparse = convert_days_to_model_time(t_days, args.dt, args.n_steps, day_scale=None)
    idx_sparse = np.clip(idx_sparse, 0, args.n_steps)
    logger.info(f"idx_sparse: {idx_sparse}")

    log_likelihood = make_log_likelihood_jax_ode(
        data=data,
        t_days=t_days,
        idx_sparse=idx_sparse,
        sigma_obs=sigma_obs,
        phi_init=phi_init,
        dt=args.dt,
        n_steps=args.n_steps,
        K_hill=args.K_hill,
        n_hill=args.n_hill,
        lambda_pg=args.lambda_pg,
        lambda_late=args.lambda_late,
        sign_prior=args.sign_prior,
        sign_lambda=args.sign_lambda,
        lambda_bc=args.bc_lambda,
    )
    if args.sign_prior:
        logger.info(
            f"Sign prior enabled (eHOMD, lam={args.sign_lambda}, {len(_EHOMD_CONSTRAINTS)} constraints)"
        )
    if args.bc_lambda > 0:
        logger.info(f"Bray-Curtis penalty enabled (lam={args.bc_lambda})")

    if args.wide_prior or args.external_data:
        prior_bounds = np.zeros((20, 2), dtype=np.float64)
        for i in range(20):
            prior_bounds[i] = [-1.0, 3.0]
        for i in [3, 4, 8, 9, 15]:  # growth rates b_i
            prior_bounds[i] = [0.0, 5.0]
        logger.info("Using wide prior bounds [-1,3] / b:[0,5]")
    else:
        prior_bounds = load_prior_bounds(args.condition, args.cultivation)

    # b (theta[3,4,8,9,15]) は alpha*=0 では動力学に入らない（論文 Sec.2）。
    # 下限=上限にすると tmcmc_engine の free_mask がこの次元を除外し、
    # 粒子は 0 に固定されたまま提案されない（tmcmc_nuts_engine.py:414-420）。
    B_DIMS = [3, 4, 8, 9, 15]
    if not args.estimate_b:
        for i in B_DIMS:
            prior_bounds[i] = [0.0, 0.0]
        logger.info(
            "b (theta[3,4,8,9,15]) を 0 に固定。alpha*=0 で動力学に入らないため。"
            "探索次元 20 -> 15"
        )
    else:
        logger.warning(
            "--estimate-b: b も推定する。alpha*=0 では尤度に入らないので "
            "事後は事前分布のままになる"
        )

    prior_bounds = np.array(prior_bounds, dtype=np.float32)

    # 弱情報事前分布。A の15成分に N(0, prior_scale^2) を置く。
    # 箱は support として残す（prior_bounds の外は engine 側で棄却される）。
    log_prior_fn = None
    prior_sample_fn = None
    if args.prior_scale > 0.0:
        _free = np.array([i for i in range(20) if i not in B_DIMS], dtype=np.int32)
        _sd = float(args.prior_scale)
        _lo = np.asarray(prior_bounds[:, 0], dtype=np.float64)
        _hi = np.asarray(prior_bounds[:, 1], dtype=np.float64)
        _free_j = jnp.array(_free)

        def _log_prior(theta):
            x = theta[_free_j]
            return -0.5 * jnp.sum((x / _sd) ** 2)

        def _sample_prior(rng, n):
            # N(0, sd^2) を箱で truncate して棄却法で引く
            out = np.zeros((n, 20), dtype=np.float64)
            for i in _free:
                lo_i, hi_i = _lo[i], _hi[i]
                col = np.empty(n)
                filled = 0
                for _ in range(200):
                    cand = rng.normal(0.0, _sd, size=max(n - filled, 1) * 2)
                    cand = cand[(cand >= lo_i) & (cand <= hi_i)]
                    take = min(len(cand), n - filled)
                    if take > 0:
                        col[filled : filled + take] = cand[:take]
                        filled += take
                    if filled >= n:
                        break
                if filled < n:  # 箱が事前分布に対して極端に狭い場合の保険
                    col[filled:] = rng.uniform(lo_i, hi_i, n - filled)
                out[:, i] = col
            return out

        log_prior_fn = _log_prior
        prior_sample_fn = _sample_prior
        logger.info(
            f"弱情報事前分布: A の15成分に N(0, {_sd}^2)。箱 "
            f"[{_lo[_free].min():.1f}, {_hi[_free].max():.1f}] は support として残す"
        )

    logger.info("JIT warmup (forward pass)...")
    _ = jax.jit(log_likelihood)(jnp.zeros(20, dtype=jnp.float64))
    logger.info("Warmup OK (forward). Grad warmup deferred to first mutation step.")

    logger.info(f"Running NUTS-TMCMC ({args.n_particles} particles)...")
    result = tmcmc_engine(
        log_likelihood,
        prior_bounds,
        log_prior_fn=log_prior_fn,
        prior_sample_fn=prior_sample_fn,
        mutation=args.mutation,
        n_particles=args.n_particles,
        max_stages=args.max_stages,
        seed=args.seed,
        nuts_max_depth=6,
        n_mutation_steps=args.n_mutation_steps,
    )

    theta_MAP = result["theta_MAP"]
    logL_tmcmc = float(result["log_likelihoods"].max())
    logger.info(
        f"Done: {result['n_stages']} stages, "
        f"total_time={result['total_time']:.1f}s, "
        f"accept={np.mean(result['accept_rates']):.2f}, "
        f"max logL={logL_tmcmc:.1f}"
    )

    # MAP polish: L-BFGS-B refines MAP starting from top-K TMCMC particles.
    # Uses scipy finite-difference gradient (avoids 30+ min JAX gradient JIT).
    if not args.no_polish and not args.quick and not args.benchmark:
        logger.info(f"MAP polishing (L-BFGS-B + FD grad, top-{args.polish_top_k} starts)...")
        from scipy.optimize import minimize as scipy_minimize

        log_likelihood_jit = jax.jit(log_likelihood)
        # Warmup already done above (forward pass only)

        def _neg_logL(theta_np):
            return float(-log_likelihood_jit(jnp.array(theta_np, dtype=jnp.float64)))

        bounds_list = [(float(prior_bounds[i, 0]), float(prior_bounds[i, 1])) for i in range(20)]
        top_k = min(args.polish_top_k, args.n_particles)
        top_idx = np.argsort(result["log_likelihoods"])[-top_k:]
        best_logL = logL_tmcmc
        best_theta = theta_MAP.copy()

        for rank, idx in enumerate(top_idx[::-1]):
            theta0 = result["samples"][idx].astype(np.float64)
            try:
                # jac="3-point": scipy computes numerical gradient (3-point FD, ~60 evals/step).
                # Each eval ~5-10ms on GPU → ~30-60 min total for 500 iters is OK post-TMCMC.
                # Much faster than waiting for JAX gradient JIT compilation (30+ min).
                opt = scipy_minimize(
                    _neg_logL,
                    theta0,
                    method="L-BFGS-B",
                    jac="3-point",
                    bounds=bounds_list,
                    options={"maxiter": 50, "ftol": 1e-9},
                )
                candidate_logL = -float(opt.fun)
                if candidate_logL > best_logL and np.all(np.isfinite(opt.x)):
                    best_logL = candidate_logL
                    best_theta = opt.x.copy()
                    logger.info(
                        f"  Polish start #{rank+1}: logL {logL_tmcmc:.2f} → {candidate_logL:.2f} "
                        f"(+{candidate_logL - logL_tmcmc:.3f}) ✓"
                    )
                else:
                    logger.info(f"  Polish start #{rank+1}: {-opt.fun:.2f} (no improvement)")
            except Exception as e:
                logger.warning(f"  Polish start #{rank+1} failed: {e}")

        if best_logL > logL_tmcmc:
            logger.info(f"MAP polish improved logL by {best_logL - logL_tmcmc:.3f}")
            theta_MAP = best_theta
        else:
            logger.info("MAP polish: no improvement over TMCMC MAP")

    from datetime import datetime

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_out = MAIN_DIR / "_runs" / f"jax_ode_nuts_{args.condition}_{args.cultivation}_{ts}"
    out_dir = Path(args.output_dir) if args.output_dir else default_out
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "samples.npy", result["samples"])
    np.save(out_dir / "logL.npy", result["log_likelihoods"])
    with open(out_dir / "theta_MAP.json", "w") as f:
        json.dump({str(i): float(v) for i, v in enumerate(theta_MAP)}, f, indent=2)
    with open(out_dir / "config.json", "w") as f:
        # 結果を決める設定はすべて記録する。これが欠けていたために、ある run で
        # Hill ゲートが有効だったかを成果物から確認できなかった。
        json.dump(
            {
                "condition": args.condition,
                "cultivation": args.cultivation,
                "n_particles": args.n_particles,
                "max_stages": args.max_stages,
                "mutation": args.mutation,
                "seed": args.seed,
                "sigma_obs": float(np.mean(sigma_obs)),
                "sigma_scale": args.sigma_scale,
                # --- 前進モデル ---
                "dt": args.dt,
                "n_steps": args.n_steps,
                "K_hill": args.K_hill,
                "n_hill": args.n_hill,
                "alpha_const": 0.0,  # 抗生物質なし。b を動力学から消す（論文 Sec.2）
                # --- 尤度の重み（論文の式には無い。必ず記録する） ---
                "lambda_pg": args.lambda_pg,
                "lambda_late": args.lambda_late,
                "sign_prior": args.sign_prior,
                "sign_lambda": args.sign_lambda,
                "bc_lambda": args.bc_lambda,
                # --- 探索次元 ---
                "estimate_b": args.estimate_b,
                "prior_scale": args.prior_scale,
                "n_free_dims": int((np.abs(prior_bounds[:, 1] - prior_bounds[:, 0]) > 1e-12).sum()),
                "prior_bounds": np.asarray(prior_bounds, dtype=float).tolist(),
                # --- 初期条件・データの扱い ---
                "use_exp_init": args.use_exp_init,
                "start_from_day": args.start_from_day,
                "data_normalized": True,  # load_experimental_data(normalize=True)
                "external_data": args.external_data,
                "wide_prior": args.wide_prior,
            },
            f,
            indent=2,
        )

    logger.info(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()
