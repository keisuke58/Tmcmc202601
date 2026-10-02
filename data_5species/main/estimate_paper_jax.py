#!/usr/bin/env python3
"""
estimate_paper_jax.py — 論文パイプライン（2段階推定）の estimator（固定版）。

論文の MAP（2026-03-20, commit ba1c285）を作った estimate_reduced_nishioka_jax.py を
そのまま取り出し、その後に見つかった不具合の修正と、ゲート OFF の再推定に要る
オプションだけを足したもの。論文の4条件の RMSE（0.119/0.104/0.033/0.087）は
この系統のコードで再現できる（tools/repro_paper_rmse_ba1c285.py）。

なぜ別ファイルにしたか:
  estimate_reduced_nishioka_jax.py は 2026-03-22 に2段階推定（多チャネル尤度・ψ固定・
  ウォームスタート・事前分布の絞り込み）を失い、2026-03-30 には前進モデルだけが
  15次元になって θ の並びが食い違った（θ[15..19] が読まれない）。さらに前進モデル・
  エンジン・ローダーが同名の別ファイルで、sys.path の順で暗黙に選ばれていた。
  ここでは固有名のモジュールだけを明示的に読み込み、起動時に照合する。

ba1c285 からの変更点:
  - 前進モデルは hamilton_ode_jax_paper（K_hill<=0 を真のゲート OFF にした版）
  - エンジンは tmcmc_engine_paper（evidence 補正・事前分布の配線を直した版）
  - 事前分布の箱は model_config/prior_bounds_paper_20d.json（ba1c285 時点の 20 次元版）
  - --estimate-b（既定 OFF）: b=θ[3,4,8,9,15] は alpha=0 で尤度に入らないので 0 固定
  - --prior-scale, --box: 識別性の検証用
  - 起動時の感度チェック: 自由な θ がすべて尤度を動かすことを確かめ、動かない次元が
    あれば止まる（2026-09 の「a35 が尤度に入っていなかった」事故の再発防止）
  - config.json に実効値を全部記録（DH の lambda_ch3 自動調整後の値、git commit など）
  - 主結果（samples/logL/theta_MAP/config）は RMSE 計算より前に保存する

(以下は元の説明)
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
        import jax_cuda12_plugin
    except ImportError:
        pass  # CPU-only jaxlib の場合は無視

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Local imports — 論文パイプラインは固有名のモジュールだけを使う（同名ファイルの取り違え防止）
import hamilton_ode_jax_paper as _H
import tmcmc_engine_paper as _E

for _mod, _name in ((_H, "hamilton_ode_jax_paper.py"), (_E, "tmcmc_engine_paper.py")):
    if Path(_mod.__file__).resolve() != (SCRIPT_DIR / _name).resolve():
        raise ImportError(f"{_name} が想定外の場所から読まれた: {_mod.__file__}")
simulate_0d = _H.simulate_0d
simulate_0d_full = _H.simulate_0d_full

# Import data loading and bounds from main estimator
from estimate_reduced_nishioka import (
    convert_days_to_model_time,
    load_experimental_data,
    load_multichannel_data,
)

tmcmc_engine = _E.tmcmc_engine

THETA_NAMES = [
    "a11",
    "a12",
    "a22",
    "b1",
    "b2",
    "a33",
    "a34",
    "a44",
    "b3",
    "b4",
    "a13",
    "a14",
    "a23",
    "a24",
    "a55",
    "b5",
    "a15",
    "a25",
    "a35",
    "a45",
]
B_DIMS = [3, 4, 8, 9, 15]
PAPER_BOUNDS_JSON = DATA_DIR / "model_config" / "prior_bounds_paper_20d.json"


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
    # Multi-channel arguments
    data_total: np.ndarray | None = None,
    sigma_obs_total: float | np.ndarray | None = None,
    data_viability: np.ndarray | None = None,
    sigma_obs_viability: float = 0.10,
    data_pH: np.ndarray | None = None,
    idx_pH: np.ndarray | None = None,
    sigma_obs_pH: float = 0.15,
    lambda_ch: dict | None = None,
    # Robust likelihood
    use_student_t: bool = False,
    student_t_nu: float = 5.0,
    # Rare species downweighting
    lambda_rare: float = 1.0,
    rare_threshold: float = 0.05,
    # Fix ψ to experimental viability
    psi_fixed: np.ndarray | None = None,
):
    """
    Build JAX-differentiable log-likelihood using Hamilton ODE.

    Supports multi-channel likelihood:
      Ch1: viable species (φ·ψ) — primary
      Ch2: total species (φ) — qRT-PCR without PMA
      Ch3: membrane integrity (viability) — LIVE/DEAD
      Ch5: pH time series — HOBIC only

    Returns
    -------
    log_likelihood : callable(theta) -> scalar
    """
    if lambda_ch is None:
        lambda_ch = {1: 1.0, 2: 0.5, 3: 2.0, 5: 0.3}

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
    # Downweight rare species: if mean abundance < rare_threshold, apply lambda_rare
    if lambda_rare < 1.0:
        mean_abundance = jnp.mean(obs, axis=0)  # (n_species,)
        rare_mask = (mean_abundance < rare_threshold).astype(jnp.float64)
        rare_weight = rare_mask * lambda_rare + (1.0 - rare_mask) * 1.0
        weights = weights * rare_weight[jnp.newaxis, :]

    # Pre-convert multichannel data to JAX arrays
    _has_total = data_total is not None
    _has_viab = data_viability is not None
    _has_pH = data_pH is not None and idx_pH is not None

    if _has_total:
        obs_total = jnp.array(data_total, dtype=jnp.float64)
        if sigma_obs_total is None:
            sigma_obs_total = 0.08
        sig_total = (
            jnp.array(sigma_obs_total, dtype=jnp.float64)
            if hasattr(sigma_obs_total, "__len__")
            else jnp.float64(sigma_obs_total)
        )
    if _has_viab:
        obs_viab = jnp.array(data_viability, dtype=jnp.float64)
        sig_viab = jnp.float64(sigma_obs_viability)
    if _has_pH:
        obs_pH = jnp.array(data_pH, dtype=jnp.float64)
        idx_ph = jnp.array(idx_pH, dtype=jnp.int32)
        sig_pH = jnp.float64(sigma_obs_pH)

    lw1 = lambda_ch.get(1, 1.0)
    lw2 = lambda_ch.get(2, 0.5)
    lw3 = lambda_ch.get(3, 2.0)
    lw5 = lambda_ch.get(5, 0.3)

    # Fixed ψ mode: interpolate experimental viability to ODE timesteps
    _fix_psi = psi_fixed is not None
    if _fix_psi:
        _psi_at_obs = jnp.array(psi_fixed, dtype=jnp.float64)  # (n_obs,)
        # Disable viability and pH channels (need g_traj which requires full solver)
        _has_viab = False
        _has_pH = False

    # Student-t log-density: log p(x | mu, sigma, nu)
    # More robust to outliers than Gaussian (heavier tails)
    _use_t = use_student_t
    _nu = jnp.float64(student_t_nu)

    def _log_student_t(residual_scaled_sq):
        """Log-density of Student-t (up to constant), given (r/sigma)^2."""
        return -0.5 * (_nu + 1) * jnp.log(1.0 + residual_scaled_sq / _nu)

    # Use full state solver when viability/pH channels are needed
    _need_full = (_has_viab or _has_pH) and not _fix_psi

    def log_likelihood(theta):
        if _need_full:
            g_traj = simulate_0d_full(
                theta,
                n_steps=n_steps,
                dt=dt,
                phi_init=phi_init_jax,
                K_hill=K_hill,
                n_hill=n_hill,
                c_const=c_const,
            )
            phi_traj = g_traj[:, 0:5]
        else:
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

        if _fix_psi:
            # φ̄ = φ × ψ_exp (ψ is uniform across species, from experiment)
            psi_broadcast = _psi_at_obs[:, jnp.newaxis]  # (n_obs, 1)
            phi_pred = phi_pred * psi_broadcast

        # Normalize to fractions
        phi_sum = jnp.sum(phi_pred, axis=1, keepdims=True)
        phi_pred = phi_pred / jnp.maximum(phi_sum, 1e-12)

        # Ch1: viable species (primary)
        residual = obs - phi_pred
        r_sq = (residual / sigma_obs) ** 2
        if _use_t:
            # Student-t: weights multiply the log-density, NOT the argument
            # (multiplying the argument changes effective nu per observation)
            logL = lw1 * jnp.sum(weights * _log_student_t(r_sq))
        else:
            logL = lw1 * (-0.5 * jnp.sum(weights * r_sq))

        # Ch2: total species distribution (φ without ψ weighting)
        if _has_total:
            residual_total = obs_total - phi_pred
            logL_total = -0.5 * jnp.sum((residual_total / sig_total) ** 2)
            logL = logL + lw2 * logL_total

        # Ch3: membrane integrity / viability
        if _has_viab:
            phi_at_obs = g_traj[idx, 0:5]
            psi_at_obs = g_traj[idx, 6:11]
            phi_psi = phi_at_obs * psi_at_obs
            viab_pred = jnp.sum(phi_psi, axis=1) / jnp.maximum(jnp.sum(phi_at_obs, axis=1), 1e-12)
            residual_viab = obs_viab - viab_pred
            logL_viab = -0.5 * jnp.sum((residual_viab / sig_viab) ** 2)
            logL = logL + lw3 * logL_viab

        # Ch5: pH (HOBIC only)
        if _has_pH:
            phi_at_pH = g_traj[idx_ph, 0:5]
            psi_at_pH = g_traj[idx_ph, 6:11]
            phibar = phi_at_pH * psi_at_pH
            phibar_sum = jnp.maximum(jnp.sum(phibar, axis=1), 1e-12)
            phi_bar_So = phibar[:, 0] / phibar_sum
            phi_bar_Vd = phibar[:, 2] / phibar_sum
            pH_pred = 7.5 - 0.74 * phi_bar_So - 0.48 * phi_bar_Vd
            residual_pH = obs_pH - pH_pred
            logL_pH = -0.5 * jnp.sum((residual_pH / sig_pH) ** 2)
            logL = logL + lw5 * logL_pH

        return logL

    return log_likelihood


def load_prior_bounds(condition: str, cultivation: str) -> np.ndarray:
    """論文時点（ba1c285）の 20 次元の箱を (20, 2) で返す。

    core.nishioka_model.get_condition_bounds と同じ規則（既定値 → locks → 個別 bounds）。
    model_config/prior_bounds.json は 2026-03-30 に 15 次元用へ並べ替えられているので使わない。
    """
    with open(PAPER_BOUNDS_JSON) as f:
        cfg = json.load(f)
    bounds = [tuple(cfg.get("default_bounds", [-1.0, 1.0]))] * 20
    strat = cfg["strategies"][f"{condition}_{cultivation}"]
    locked = list(strat.get("locks", []))
    for i in locked:
        bounds[i] = (0.0, 0.0)
    for k, v in strat.get("bounds", {}).items():
        bounds[int(k)] = tuple(v)
    for i in locked:
        bounds[i] = (0.0, 0.0)
    return np.array(bounds, dtype=np.float64)


def check_sensitivity(log_likelihood, prior_bounds: np.ndarray, tol: float = 1e-9) -> None:
    """自由な θ の各成分が尤度を動かすことを確かめる。動かない成分があれば止める。

    2026-09 に「推定していると思っていた a35/a45 が前進モデルに読まれていなかった」事故が
    起きた（θ の並びの食い違い）。尤度に効かない次元は事前分布のまま漂うだけで、実行は
    正常終了するので気づけない。箱の中点から各自由次元を箱幅の 1/4 だけ動かし、尤度が
    変わらない次元を列挙して RuntimeError にする。
    """
    free = np.abs(prior_bounds[:, 1] - prior_bounds[:, 0]) > 1e-12
    free_idx = np.where(free)[0]
    base = np.where(free, 0.5 * (prior_bounds[:, 0] + prior_bounds[:, 1]), prior_bounds[:, 0])
    pert = np.tile(base, (len(free_idx) + 1, 1))
    for k, i in enumerate(free_idx):
        pert[k + 1, i] = base[i] + 0.25 * (prior_bounds[i, 1] - prior_bounds[i, 0])
    ll = np.array(jax.jit(jax.vmap(log_likelihood))(jnp.array(pert)))
    if not np.all(np.isfinite(ll)):
        raise RuntimeError(f"感度チェック: 尤度が有限でない {ll}")
    dead = [THETA_NAMES[i] for k, i in enumerate(free_idx) if abs(ll[k + 1] - ll[0]) < tol]
    if dead:
        raise RuntimeError(
            f"感度チェック: 自由次元 {dead} が尤度を動かさない。θ の並びかモデルを確認"
        )


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
    parser.add_argument(
        "--lambda-rare",
        type=float,
        default=0.1,
        help="Weight for rare species (mean < 5%%). Default 0.1 = downweight 10x",
    )
    parser.add_argument("--sigma-scale", type=float, default=1.0)
    parser.add_argument(
        "--replicate-sigma",
        action="store_true",
        help="Use per-timepoint×per-species replicate sigma matrix (matches CPU)",
    )
    parser.add_argument("--K-hill", type=float, default=0.05)
    parser.add_argument("--n-hill", type=float, default=4.0)
    parser.add_argument("--dt", type=float, default=1e-4)
    parser.add_argument("--n-steps", type=int, default=2500)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--mutation", default="nuts", choices=["rw", "hmc", "nuts"])
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
    # Multi-channel likelihood
    parser.add_argument(
        "--multichannel",
        action="store_true",
        help="Enable multi-channel likelihood: viable + total + viability + pH",
    )
    parser.add_argument("--lambda-ch1", type=float, default=1.0, help="Weight for viable channel")
    parser.add_argument("--lambda-ch2", type=float, default=0.5, help="Weight for total species")
    parser.add_argument("--lambda-ch3", type=float, default=2.0, help="Weight for viability")
    parser.add_argument("--lambda-ch5", type=float, default=0.3, help="Weight for pH (HOBIC only)")
    parser.add_argument(
        "--n-mutation-steps", type=int, default=1, help="RW mutation steps per stage"
    )
    # Accuracy improvements
    parser.add_argument(
        "--use-de-mc", action="store_true", help="Enable DE-MC proposals (alternating with RW)"
    )
    parser.add_argument(
        "--de-mc-gamma",
        type=float,
        default=2.38,
        help="DE-MC scale factor (default 2.38, ter Braak 2006)",
    )
    parser.add_argument(
        "--use-student-t", action="store_true", help="Use Student-t likelihood (robust to outliers)"
    )
    parser.add_argument(
        "--student-t-nu",
        type=float,
        default=5.0,
        help="Student-t degrees of freedom (lower = heavier tails, default 5)",
    )
    parser.add_argument(
        "--resample-method",
        default="systematic",
        choices=["systematic", "multinomial"],
        help="Resampling method (systematic = lower variance)",
    )
    parser.add_argument(
        "--max-info-ratio",
        type=float,
        default=10.0,
        help="Max information ratio for replicate sigma (caps An dominance, default 10)",
    )
    # Flow-enhanced SMC (Gabrie et al. 2022)
    parser.add_argument("--use-flow", action="store_true", help="Enable NF proposal (RealNVP)")
    parser.add_argument("--flow-n-layers", type=int, default=6)
    parser.add_argument("--flow-hidden-dim", type=int, default=64)
    parser.add_argument("--flow-n-epochs", type=int, default=200)
    parser.add_argument("--flow-lr", type=float, default=1e-3)
    parser.add_argument("--flow-start-beta", type=float, default=0.1)
    parser.add_argument("--flow-mix-ratio", type=float, default=0.8)
    # Waste-free SMC (Dau & Chopin 2022)
    parser.add_argument("--waste-free", action="store_true", help="Enable waste-free SMC")
    # Warm-start from previous run
    parser.add_argument(
        "--init-from-dir",
        type=str,
        default=None,
        help="Directory of previous run to warm-start from (loads MAP + samples)",
    )
    parser.add_argument(
        "--posterior-prior-nsigma",
        type=float,
        default=None,
        help="Auto-narrow bounds to MAP ± N*σ from previous posterior (requires --init-from-dir)",
    )
    parser.add_argument(
        "--override-bounds",
        type=str,
        default=None,
        help="Override prior bounds: 'idx:lo:hi,idx:lo:hi,...' (e.g. '18:0.5:1.5,19:0.5:2.0')",
    )
    # Fix ψ to experimental viability (focus on A matrix estimation)
    parser.add_argument(
        "--fix-psi",
        action="store_true",
        help="Fix ψ_i to experimental viability data (uniform across species). "
        "Disables viability channel (Ch3) and uses φ-only ODE (faster). "
        "Goal: focus likelihood on A matrix estimation.",
    )
    parser.add_argument(
        "--estimate-b",
        action="store_true",
        help="b=θ[3,4,8,9,15] も推定する。既定では 0 固定（alpha=0 で尤度に入らないため）",
    )
    parser.add_argument(
        "--prior-scale",
        type=float,
        default=0.0,
        help="0 より大きいと A の自由成分に N(0, prior_scale^2) を置く（識別性の検証用）",
    )
    parser.add_argument(
        "--box",
        type=float,
        nargs=2,
        default=None,
        metavar=("LO", "HI"),
        help="A の自由成分の箱を [LO, HI] で一括上書き（識別性の検証用）。b は対象外",
    )
    parser.add_argument(
        "--skip-sensitivity-check",
        action="store_true",
        help="起動時の感度チェック（全自由次元が尤度を動かすか）を省く。通常は使わない",
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

    logger.info("Loading experimental data...")
    data, t_days, sigma_obs_est, phi_init_exp, metadata = load_experimental_data(
        DATA_DIR,
        args.condition,
        args.cultivation,
        args.start_from_day,
        normalize=True,
        use_exp_init=args.use_exp_init,
    )
    if args.replicate_sigma:
        from estimate_reduced_nishioka import build_replicate_sigma

        replicate_csv = DATA_DIR / "fig3_species_distribution_replicates.csv"
        if not replicate_csv.exists():
            replicate_csv = (
                DATA_DIR.parent / "experiment_data" / "fig3_species_distribution_replicates.csv"
            )
        _rep_species_map = {
            "S. oralis": 0,
            "A. naeslundii": 1,
            "V. dispar": 2,
            "V. parvula": 2,
            "F. nucleatum": 3,
            "P. gingivalis_20709": 4,
            "P. gingivalis_W83": 4,
        }
        sigma_obs = (
            build_replicate_sigma(
                replicate_csv=str(replicate_csv),
                condition=args.condition,
                cultivation=args.cultivation,
                days=metadata["days"],
                species_map=_rep_species_map,
                n_species=data.shape[1],
                min_sigma=0.05,
                max_info_ratio=getattr(args, "max_info_ratio", 10.0),
            )
            * args.sigma_scale
        )
        logger.info(f"Replicate sigma: shape={sigma_obs.shape}, mean={np.mean(sigma_obs):.4f}")
    else:
        sigma_obs = sigma_obs_est * args.sigma_scale
    logger.info(f"Data: {data.shape}, sigma_obs mean={np.mean(sigma_obs):.4f}")

    _t_model, idx_sparse = convert_days_to_model_time(t_days, args.dt, args.n_steps, day_scale=None)
    idx_sparse = np.clip(idx_sparse, 0, args.n_steps)
    logger.info(f"idx_sparse: {idx_sparse}")

    phi_init = phi_init_exp if args.use_exp_init else np.full(5, 0.2)
    if args.use_exp_init:
        total = phi_init.sum()
        if total > 0:
            phi_init = phi_init / total
        phi_init = np.clip(phi_init, 0.001, 0.99)

    # Multi-channel data loading
    mc_kwargs = {}
    if args.multichannel:
        mc_data = load_multichannel_data(
            data_dir=DATA_DIR / "experiment_data",
            condition=args.condition,
            cultivation=args.cultivation,
            days_filter=t_days.tolist(),
        )
        # Adaptive channel weights: DH needs stronger viability signal for Pg
        # HOBIC uses pH channel; Static doesn't have pH data
        lambda_ch = {
            1: args.lambda_ch1,
            2: args.lambda_ch2,
            3: args.lambda_ch3,
            5: args.lambda_ch5,
        }
        # Auto-tune: Dysbiotic HOBIC benefits from higher viability weight
        # (Pg viability drops significantly, key diagnostic signal)
        # only if user didn't override
        if args.condition == "Dysbiotic" and args.cultivation == "HOBIC" and args.lambda_ch3 == 2.0:
            lambda_ch[3] = 3.0
            logger.info("Auto-tuned: lambda_ch3=3.0 for DH (Pg viability signal)")
        # Commensal conditions: rare Pg/Fn have minimal viability signal,
        # reduce Ch3 weight to avoid noise amplification
        if args.condition == "Commensal" and args.lambda_ch3 == 2.0:  # only if not overridden
            lambda_ch[3] = 1.5
            logger.info(
                f"Auto-tuned: lambda_ch3=1.5 for {args.condition} (less Pg/Fn viability info)"
            )
        mc_kwargs = {
            "data_total": mc_data.get("data_total"),
            "sigma_obs_total": mc_data.get("sigma_obs_total"),
            "data_viability": mc_data.get("data_viability"),
            "sigma_obs_viability": mc_data.get("sigma_obs_viability", 0.10),
            "lambda_ch": lambda_ch,
        }
        # pH channel (HOBIC only)
        if mc_data.get("data_pH") is not None:
            t_pH_days = mc_data["t_pH_days"]
            _, idx_pH = convert_days_to_model_time(t_pH_days, args.dt, args.n_steps, day_scale=None)
            idx_pH = np.clip(idx_pH, 0, args.n_steps)
            mc_kwargs["data_pH"] = mc_data["data_pH"]
            mc_kwargs["idx_pH"] = idx_pH
            mc_kwargs["sigma_obs_pH"] = mc_data.get("sigma_obs_pH", 0.15)
        logger.info(
            f"Multi-channel: total={'yes' if mc_data.get('data_total') is not None else 'no'}, "
            f"viability={'yes' if mc_data.get('data_viability') is not None else 'no'}, "
            f"pH={'yes' if mc_data.get('data_pH') is not None else 'no'}"
        )

    # --fix-psi: load experimental viability and fix ψ_i = viab(t) for all species
    psi_fixed_arr = None
    if args.fix_psi:
        import pandas as pd

        viab_file = DATA_DIR / "experiment_data" / "fig2_membrane_distribution.csv"
        viab_df = pd.read_csv(viab_file)
        viab_cond = viab_df[
            (viab_df["condition"] == args.condition) & (viab_df["cultivation"] == args.cultivation)
        ].sort_values("day")
        viab_days = viab_cond["day"].values.astype(float)
        viab_intact = viab_cond["intact_membrane_pct"].values / 100.0

        # Interpolate viability to observation days (t_days)
        psi_fixed_arr = np.interp(t_days, viab_days, viab_intact)
        psi_fixed_arr = np.clip(psi_fixed_arr, 0.01, 0.99)
        logger.info(
            f"[fix-psi] ψ fixed to exp viability: {dict(zip(t_days, psi_fixed_arr.round(3)))}"
        )
        # Disable viability channel (redundant when ψ is fixed)
        mc_kwargs.pop("data_viability", None)
        mc_kwargs.pop("sigma_obs_viability", None)
        mc_kwargs["lambda_ch"] = mc_kwargs.get("lambda_ch", {})
        if mc_kwargs["lambda_ch"]:
            mc_kwargs["lambda_ch"][3] = 0.0

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
        use_student_t=args.use_student_t,
        student_t_nu=args.student_t_nu,
        lambda_rare=args.lambda_rare,
        psi_fixed=psi_fixed_arr,
        **mc_kwargs,
    )

    prior_bounds = load_prior_bounds(args.condition, args.cultivation)
    prior_bounds = np.array(prior_bounds, dtype=np.float64)  # float64 for consistency

    # --- Override bounds (from CLI) ---
    if args.override_bounds:
        free_mask_ovr = np.abs(prior_bounds[:, 1] - prior_bounds[:, 0]) > 1e-12
        for spec in args.override_bounds.split(","):
            idx_s, lo_s, hi_s = spec.split(":")
            idx_o = int(idx_s)
            if not free_mask_ovr[idx_o]:
                logger.warning(f"Cannot override locked index {idx_o}, skipping")
                continue
            prior_bounds[idx_o] = [float(lo_s), float(hi_s)]
            logger.info(f"Override bound [{idx_o}] -> [{lo_s}, {hi_s}]")

    # --- 識別性検証用の箱の一括上書き ---
    if args.box is not None:
        lo_b, hi_b = float(args.box[0]), float(args.box[1])
        for i in range(20):
            if i not in B_DIMS and abs(prior_bounds[i, 1] - prior_bounds[i, 0]) > 1e-12:
                prior_bounds[i] = [lo_b, hi_b]
        logger.info(f"--box: A の自由成分の箱を [{lo_b}, {hi_b}] に上書き")

    # --- b を 0 固定（下限=上限にするとエンジンの free_mask から外れる） ---
    if not args.estimate_b:
        for i in B_DIMS:
            prior_bounds[i] = [0.0, 0.0]
        logger.info("b=θ[3,4,8,9,15] を 0 に固定（alpha=0 で尤度に入らない）。探索 20→15 次元")

    free_idx = np.where(np.abs(prior_bounds[:, 1] - prior_bounds[:, 0]) > 1e-12)[0]

    # --- 弱情報事前分布（任意） ---
    log_prior_fn = None
    sample_prior = None
    if args.prior_scale > 0.0:
        _sd = float(args.prior_scale)
        _free_j = jnp.array(free_idx)

        def log_prior_fn(theta):
            return -0.5 * jnp.sum((theta[_free_j] / _sd) ** 2)

        def sample_prior(rng, n):
            out = np.tile(prior_bounds[:, 0], (n, 1)).astype(np.float64)
            for i in free_idx:
                lo_i, hi_i = prior_bounds[i]
                col = np.empty(0)
                for _ in range(1000):
                    cand = rng.normal(0.0, _sd, size=2 * n)
                    col = np.concatenate([col, cand[(cand >= lo_i) & (cand <= hi_i)]])
                    if col.size >= n:
                        break
                if col.size < n:
                    raise RuntimeError(f"θ[{i}] の箱 [{lo_i}, {hi_i}] で N(0,{_sd}^2) が引けない")
                out[:, i] = col[:n]
            return out

        logger.info(f"弱情報事前分布 N(0, {_sd}^2) を自由 {len(free_idx)} 成分に置く")

    # --- Warm-start from previous run ---
    init_particles = None
    if args.init_from_dir:
        prev_dir = Path(args.init_from_dir)
        prev_samples = np.load(prev_dir / "samples.npy")
        prev_map_file = prev_dir / "theta_MAP.json"
        prev_map = None
        if prev_map_file.exists():
            with open(prev_map_file) as f:
                _mj = json.load(f)
            prev_map = np.array([_mj[str(i)] for i in range(20)])
        logger.info(f"Warm-start: loaded {prev_samples.shape[0]} samples from {prev_dir.name}")

        # Posterior-informed prior narrowing
        if args.posterior_prior_nsigma is not None and prev_samples.shape[0] > 10:
            nsig = args.posterior_prior_nsigma
            post_mean = prev_samples.mean(axis=0)
            post_std = prev_samples.std(axis=0)
            free_mask_pp = np.abs(prior_bounds[:, 1] - prior_bounds[:, 0]) > 1e-12
            n_narrowed = 0
            for i in range(20):
                if not free_mask_pp[i] or post_std[i] < 1e-10:
                    continue
                new_lo = max(prior_bounds[i, 0], post_mean[i] - nsig * post_std[i])
                new_hi = min(prior_bounds[i, 1], post_mean[i] + nsig * post_std[i])
                if new_hi - new_lo > 0.01:  # safety: don't collapse
                    old_w = prior_bounds[i, 1] - prior_bounds[i, 0]
                    prior_bounds[i] = [new_lo, new_hi]
                    n_narrowed += 1
                    logger.info(
                        f"  [{i}] {old_w:.3f} -> {new_hi - new_lo:.3f} "
                        f"(μ={post_mean[i]:.3f}, σ={post_std[i]:.3f})"
                    )
            logger.info(f"Posterior prior: narrowed {n_narrowed} params to MAP ± {nsig}σ")

        # Build init_particles: 50% from MAP neighborhood, 50% from prior
        rng_init = np.random.default_rng(args.seed + 1000)
        init_particles = np.zeros((args.n_particles, 20), dtype=np.float64)
        n_warm = args.n_particles // 2
        free_mask_w = np.abs(prior_bounds[:, 1] - prior_bounds[:, 0]) > 1e-12

        if prev_map is not None and prev_samples.shape[0] >= 10:
            # Sample from previous posterior with small perturbation
            post_cov = np.cov(prev_samples.T)
            post_cov = np.atleast_2d(post_cov)
            # Shrink covariance for tighter initialization
            post_cov *= 0.25  # 0.5σ perturbation
            post_cov += np.eye(20) * 1e-10
            warm_samples = rng_init.multivariate_normal(prev_map, post_cov, size=n_warm)
        else:
            # Fallback: resample from prev_samples
            idx_resample = rng_init.choice(len(prev_samples), n_warm, replace=True)
            warm_samples = prev_samples[idx_resample]

        init_particles[:n_warm] = warm_samples
        # Remaining from prior（弱情報事前分布があればそこから、無ければ箱一様）
        if sample_prior is not None:
            init_particles[n_warm:] = sample_prior(rng_init, args.n_particles - n_warm)
        for i in range(20):
            lo, hi = prior_bounds[i]
            if abs(hi - lo) < 1e-12:
                init_particles[:, i] = lo
            elif sample_prior is None:
                init_particles[n_warm:, i] = rng_init.uniform(lo, hi, args.n_particles - n_warm)
        # Clamp all to bounds
        for i in range(20):
            init_particles[:, i] = np.clip(
                init_particles[:, i], prior_bounds[i, 0], prior_bounds[i, 1]
            )
        logger.info(f"Init particles: {n_warm} warm-start + {args.n_particles - n_warm} prior")

    logger.info("JIT warmup...")
    _ = jax.jit(log_likelihood)(jnp.zeros(20, dtype=jnp.float64))
    _ = jax.jit(jax.value_and_grad(log_likelihood))(jnp.zeros(20, dtype=jnp.float64))
    logger.info("Warmup OK")

    # --- 起動時の感度チェック: 自由な θ がすべて尤度を動かすか ---
    # 2026-09 に「推定していると思っていた a35/a45 が前進モデルに読まれていなかった」
    # 事故が起きた（θ の並びの食い違い）。尤度に効かない次元は事前分布のまま漂うだけで、
    # 実行は正常終了するので気づけない。ここで機械的に止める。
    if not args.skip_sensitivity_check:
        check_sensitivity(log_likelihood, prior_bounds)
        logger.info(f"感度チェック OK: 自由 {len(free_idx)} 次元すべてが尤度を動かす")

    logger.info(f"Running {args.mutation.upper()}-TMCMC ({args.n_particles} particles)...")
    result = tmcmc_engine(
        log_likelihood,
        prior_bounds,
        mutation=args.mutation,
        n_particles=args.n_particles,
        max_stages=args.max_stages,
        seed=args.seed,
        nuts_max_depth=6,
        n_mutation_steps=args.n_mutation_steps,
        use_de_mc=args.use_de_mc,
        de_mc_gamma=args.de_mc_gamma,
        resample_method=args.resample_method,
        use_flow=args.use_flow,
        flow_n_layers=args.flow_n_layers,
        flow_hidden_dim=args.flow_hidden_dim,
        flow_n_epochs=args.flow_n_epochs,
        flow_lr=args.flow_lr,
        flow_start_beta=args.flow_start_beta,
        flow_mix_ratio=args.flow_mix_ratio,
        waste_free=args.waste_free,
        init_particles=init_particles,
        log_prior_fn=log_prior_fn,
        prior_sample_fn=sample_prior if init_particles is None else None,
    )

    theta_MAP = result["theta_MAP"]
    log_evidence = result.get("log_evidence", None)

    # 事前分布があるときの MAP は「事後最大」。engine の theta_MAP は argmax(logL)（最尤点）
    theta_MAP_post = None
    if log_prior_fn is not None:
        _lp = np.array(jax.vmap(log_prior_fn)(jnp.array(result["samples"])))
        theta_MAP_post = result["samples"][int(np.argmax(result["log_likelihoods"] + _lp))]

    import subprocess
    from datetime import datetime

    ts = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    default_out = MAIN_DIR / "_runs" / f"paper_{args.condition}_{args.cultivation}_{ts}"
    out_dir = Path(args.output_dir) if args.output_dir else default_out
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        git_head = subprocess.run(
            ["git", "-C", str(SCRIPT_DIR), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        git_dirty = bool(
            subprocess.run(
                ["git", "-C", str(SCRIPT_DIR), "status", "--porcelain", "--", str(SCRIPT_DIR)],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        git_head, git_dirty = "unknown", None

    # 主結果を先に保存（後処理で落ちても事後は残る）
    np.save(out_dir / "samples.npy", result["samples"])
    np.save(out_dir / "logL.npy", result["log_likelihoods"])
    with open(out_dir / "theta_MAP.json", "w") as f:
        json.dump({str(i): float(v) for i, v in enumerate(theta_MAP)}, f, indent=2)
    if theta_MAP_post is not None:
        with open(out_dir / "theta_MAP_posterior.json", "w") as f:
            json.dump({str(i): float(v) for i, v in enumerate(theta_MAP_post)}, f, indent=2)
    run_record = {
        "pipeline": "estimate_paper_jax.py (ba1c285 系・論文パイプライン)",
        "git_head": git_head,
        "git_dirty_main_dir": git_dirty,
        "modules": {"forward": _H.__file__, "engine": _E.__file__},
        "argv": sys.argv,
        "args": vars(args),
        "theta_names": THETA_NAMES,
        "free_dims": [int(i) for i in free_idx],
        "prior_bounds_final": prior_bounds.tolist(),
        "phi_init": np.asarray(phi_init).tolist(),
        "sigma_obs": np.asarray(sigma_obs).tolist(),
        "lambda_ch_effective": {str(k): v for k, v in mc_kwargs.get("lambda_ch", {}).items()},
        "psi_fixed": None if psi_fixed_arr is None else np.asarray(psi_fixed_arr).tolist(),
        "n_stages": result["n_stages"],
        # beta<1 で打ち切られた run は事後でもエビデンスでもない（max_stages 不足）
        "beta_final": (
            float(np.asarray(result["betas"])[-1]) if len(result.get("betas", [])) else None
        ),
        "total_time_s": result["total_time"],
        "mean_accept": float(np.mean(result["accept_rates"])),
        "max_logL": float(result["log_likelihoods"].max()),
        "log_evidence": float(log_evidence) if log_evidence is not None else None,
    }
    if run_record["beta_final"] is not None and run_record["beta_final"] < 1.0 - 1e-9:
        logger.warning(f"beta={run_record['beta_final']:.4f} で打ち切り。事後として使えない")
    with open(out_dir / "run_record.json", "w") as f:
        json.dump(run_record, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"主結果を保存: {out_dir}")
    logger.info(
        f"Done: {result['n_stages']} stages, "
        f"total_time={result['total_time']:.1f}s, "
        f"accept={np.mean(result['accept_rates']):.2f}, "
        f"max logL={result['log_likelihoods'].max():.1f}"
        + (f", log_evidence={log_evidence:.2f}" if log_evidence is not None else "")
    )

    # Compute RMSE for MAP estimate (multichannel-aware)
    theta_MAP_jax = jnp.array(theta_MAP, dtype=jnp.float64)
    mc_rmse = {}
    if args.multichannel:
        g_traj_map = simulate_0d_full(
            theta_MAP_jax,
            n_steps=args.n_steps,
            dt=args.dt,
            phi_init=jnp.array(phi_init, dtype=jnp.float64),
            K_hill=args.K_hill,
            n_hill=args.n_hill,
        )
        phi_traj_map = g_traj_map[:, 0:5]
    else:
        phi_traj_map = simulate_0d(
            theta_MAP_jax,
            n_steps=args.n_steps,
            dt=args.dt,
            phi_init=jnp.array(phi_init, dtype=jnp.float64),
            K_hill=args.K_hill,
            n_hill=args.n_hill,
        )
        g_traj_map = None
    phi_pred_map = np.array(phi_traj_map[idx_sparse, :])
    phi_pred_map = np.clip(phi_pred_map, 1e-10, 1.0 - 1e-10)
    phi_pred_map = phi_pred_map / phi_pred_map.sum(axis=1, keepdims=True)
    rmse = np.sqrt(np.mean((data - phi_pred_map) ** 2))
    mae = np.mean(np.abs(data - phi_pred_map))
    r2_vals = []
    for sp in range(data.shape[1]):
        ss_res = np.sum((data[:, sp] - phi_pred_map[:, sp]) ** 2)
        ss_tot = np.sum((data[:, sp] - np.mean(data[:, sp])) ** 2)
        r2_vals.append(1.0 - ss_res / max(ss_tot, 1e-12))
    mc_rmse["ch1_species"] = {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": [float(v) for v in r2_vals],
    }
    logger.info(
        f"Ch1 (species): RMSE={rmse:.4f}, MAE={mae:.4f}, R²={np.mean(r2_vals):.3f} "
        f"(per-sp: {[f'{v:.3f}' for v in r2_vals]})"
    )
    # Ch2: total species (φ without viability weighting)
    if args.multichannel and mc_kwargs.get("data_total") is not None and g_traj_map is not None:
        phi_at_obs = np.array(g_traj_map[idx_sparse, 0:5])
        phi_total_pred = phi_at_obs / np.maximum(phi_at_obs.sum(axis=1, keepdims=True), 1e-12)
        data_total = mc_kwargs["data_total"]
        rmse_total = np.sqrt(np.mean((data_total - phi_total_pred) ** 2))
        mc_rmse["ch2_total"] = {"rmse": float(rmse_total)}
        logger.info(f"Ch2 (total species): RMSE={rmse_total:.4f}")
    # Ch3: viability
    if args.multichannel and mc_kwargs.get("data_viability") is not None and g_traj_map is not None:
        g_at_obs = np.array(g_traj_map[idx_sparse, :])
        phi_at = g_at_obs[:, 0:5]
        psi_at = g_at_obs[:, 6:11]
        viab_pred = np.sum(phi_at * psi_at, axis=1) / np.maximum(np.sum(phi_at, axis=1), 1e-12)
        data_viab = mc_kwargs["data_viability"]
        rmse_viab = np.sqrt(np.mean((data_viab - viab_pred) ** 2))
        r2_viab = 1.0 - np.sum((data_viab - viab_pred) ** 2) / max(
            np.sum((data_viab - np.mean(data_viab)) ** 2), 1e-12
        )
        mc_rmse["ch3_viability"] = {"rmse": float(rmse_viab), "r2": float(r2_viab)}
        logger.info(f"Ch3 (viability): RMSE={rmse_viab:.4f}, R²={r2_viab:.3f}")
    # Ch5: pH
    if args.multichannel and mc_kwargs.get("data_pH") is not None and g_traj_map is not None:
        idx_ph = mc_kwargs["idx_pH"]
        g_pH = np.array(g_traj_map[idx_ph, :])
        phi_pH = g_pH[:, 0:5]
        psi_pH = g_pH[:, 6:11]
        phibar = phi_pH * psi_pH
        phibar_sum = np.maximum(np.sum(phibar, axis=1), 1e-12)
        phi_So = phibar[:, 0] / phibar_sum
        phi_Vd = phibar[:, 2] / phibar_sum
        pH_pred = 7.5 - 0.74 * phi_So - 0.48 * phi_Vd
        data_pH = mc_kwargs["data_pH"]
        rmse_pH = np.sqrt(np.mean((data_pH - pH_pred) ** 2))
        r2_pH = 1.0 - np.sum((data_pH - pH_pred) ** 2) / max(
            np.sum((data_pH - np.mean(data_pH)) ** 2), 1e-12
        )
        mc_rmse["ch5_pH"] = {"rmse": float(rmse_pH), "r2": float(r2_pH)}
        logger.info(f"Ch5 (pH): RMSE={rmse_pH:.4f}, R²={r2_pH:.3f}")

    with open(out_dir / "config.json", "w") as f:
        json.dump(
            {
                **run_record,
                "condition": args.condition,
                "cultivation": args.cultivation,
                "n_particles": args.n_particles,
                "n_mutation_steps": args.n_mutation_steps,
                "mutation": args.mutation,
                "sigma_obs": (
                    float(np.mean(sigma_obs)) if hasattr(sigma_obs, "__len__") else float(sigma_obs)
                ),
                "device": str(jax.devices()[0]),
                "multichannel": args.multichannel,
                "use_de_mc": args.use_de_mc,
                "use_student_t": args.use_student_t,
                "student_t_nu": args.student_t_nu,
                "resample_method": args.resample_method,
                "n_hill": args.n_hill,
                "K_hill": args.K_hill,
                "replicate_sigma": args.replicate_sigma,
                "max_info_ratio": args.max_info_ratio,
                "lambda_rare": args.lambda_rare,
                "lambda_ch": {
                    "1": args.lambda_ch1,
                    "2": args.lambda_ch2,
                    "3": args.lambda_ch3,
                    "5": args.lambda_ch5,
                },
                "n_stages": result["n_stages"],
                "total_time_s": result["total_time"],
                "mean_accept": float(np.mean(result["accept_rates"])),
                "max_logL": float(result["log_likelihoods"].max()),
                "log_evidence": float(log_evidence) if log_evidence is not None else None,
                "cv_weights_final": (
                    float(result["cv_weights"][-1]) if result.get("cv_weights") else None
                ),
                "rmse": float(rmse),
                "mae": float(mae),
                "r2_mean": float(np.mean(r2_vals)),
                "r2_per_species": [float(v) for v in r2_vals],
                "multichannel_rmse": mc_rmse,
            },
            f,
            indent=2,
        )

    logger.info(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()
