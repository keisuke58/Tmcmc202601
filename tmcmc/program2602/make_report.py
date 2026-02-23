#!/usr/bin/env python3
"""
Generate a single REPORT.md from a tmcmc run directory.

Usage:
  python tmcmc/make_report.py --run-dir tmcmc/_runs/<run_id>
  python tmcmc/make_report.py --runs-root tmcmc/_runs --run-id <run_id>

Outputs:
  tmcmc/_runs/<run_id>/REPORT.md
  tmcmc/_runs/<run_id>/report_assets/*.png   (best-effort; requires matplotlib)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from config import REPORT_THRESHOLDS, setup_logging
try:
    import pbox
except ImportError:
    # If run as module tmcmc.make_report
    from tmcmc import pbox

logger = logging.getLogger(__name__)

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore


# -----------------------------
# Thresholds (Pass/Warn/Fail)
# -----------------------------
FAIL_ESS_MIN = REPORT_THRESHOLDS.fail_ess_min
FAIL_ACCEPT_RATE_MEAN = REPORT_THRESHOLDS.fail_accept_rate_mean
FAIL_ROM_ERROR_MAX = REPORT_THRESHOLDS.fail_rom_error_max
WARN_ESS_MIN = REPORT_THRESHOLDS.warn_ess_min
WARN_ROM_ERROR_MAX = REPORT_THRESHOLDS.warn_rom_error_max
MODELS = tuple(REPORT_THRESHOLDS.models)


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if math.isfinite(v):
            return v
        return None
    except Exception:
        return None


def _fmt(x: Any, ndigits: int = 4) -> str:
    v = _safe_float(x)
    if v is None:
        return "missing"
    return f"{v:.{ndigits}g}"


def _read_json(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not path.exists():
        return None, "missing"
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            return obj, None
        return None, "invalid_json"
    except Exception as e:  # pragma: no cover
        return None, f"error: {type(e).__name__}: {e}"


def _read_csv_rows(path: Path) -> Tuple[List[Dict[str, str]], Optional[str]]:
    if not path.exists():
        return [], "missing"
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = [dict(r) for r in reader]
        return rows, None
    except Exception as e:  # pragma: no cover
        return [], f"error: {type(e).__name__}: {e}"


def _read_csv_df(path: Path):
    if pd is None:
        return None, "pandas_missing"
    if not path.exists():
        return None, "missing"
    try:
        return pd.read_csv(path), None
    except Exception as e:  # pragma: no cover
        return None, f"error: {type(e).__name__}: {e}"


def _summ_stats(values: Iterable[float]) -> Dict[str, Optional[float]]:
    # Be robust to strings / NaN / None (keep report generation non-fatal).
    xs: List[float] = []
    for v in values:
        fv = _safe_float(v)
        if fv is None:
            continue
        xs.append(fv)
    if not xs:
        return {"last": None, "min": None, "max": None, "mean": None}
    return {
        "last": xs[-1],
        "min": min(xs),
        "max": max(xs),
        "mean": float(statistics.fmean(xs)),
    }


def _load_npz_data(run_dir: Path) -> Dict[str, Any]:
    """
    Load data from results npz (diagnostics, samples, truth).
    """
    out: Dict[str, Any] = {}
    npz_path = run_dir / "results_MAP_linearization.npz"
    if not npz_path.exists():
        return out
    try:
        with np.load(npz_path, allow_pickle=True) as z:
            # Global true
            if "theta_true" in z:
                out["theta_true"] = z["theta_true"]

            for m in MODELS:
                # Diagnostics
                k = f"diagnostics_{m}"
                if k in z and z[k].dtype == object:
                    d = z[k].item()
                    if isinstance(d, dict):
                        out[m] = d  # legacy key: "M1" -> diagnostics dict
                        out[k] = d  # explicit key
                
                # Samples
                sk = f"samples_{m}"
                if sk in z:
                    out[sk] = z[sk]
                
                # MAP
                mk = f"MAP_{m}"
                if mk in z:
                    out[mk] = z[mk]
    except Exception:
        pass
    return out


def _models_for_run(config: Optional[Dict[str, Any]], metrics: Optional[Dict[str, Any]]) -> Tuple[str, ...]:
    """
    Determine which models to report on for this run.

    Priority (first non-empty wins):
    - metrics["models_ran"] (explicit)
    - metrics["requested_models"] (explicit)
    - config["models"] (run configuration)
    - default MODELS from REPORT_THRESHOLDS
    """
    for src in (metrics or {}, config or {}):
        for k in ("models_ran", "requested_models", "models"):
            v = src.get(k)
            if isinstance(v, str):
                xs = [s.strip() for s in v.split(",") if s.strip()]
            elif isinstance(v, (list, tuple)):
                xs = [str(s) for s in v if str(s)]
            else:
                xs = []
            if xs:
                return tuple(xs)
    return MODELS


def _try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore

        return plt, None
    except Exception as e:  # pragma: no cover
        return None, f"{type(e).__name__}: {e}"


@dataclass
class ModelKeyMetrics:
    rmse_total: Optional[float]
    mae_total: Optional[float]
    max_abs: Optional[float]
    rom_error_last: Optional[float]
    rom_error_max: Optional[float]
    rom_error_mean: Optional[float]
    ess_min: Optional[float]
    ess_mean: Optional[float]
    accept_rate_mean: Optional[float]
    accept_rate_min: Optional[float]
    accept_rate_last: Optional[float]
    beta_final: Optional[float]
    beta_stages: Optional[int]
    theta0_updates: Optional[int]
    theta0_step_norm_max: Optional[float]


def _collect_model_metrics(
    run_dir: Path,
    model: str,
    fit_metrics: Optional[Dict[str, Any]],
    npz_diag: Optional[Dict[str, Any]],
) -> ModelKeyMetrics:
    # (A) Fit metrics (MAP)
    rmse_total = _safe_float(((fit_metrics or {}).get("fit_MAP") or {}).get("rmse_total"))
    mae_total = _safe_float(((fit_metrics or {}).get("fit_MAP") or {}).get("mae_total"))
    max_abs = _safe_float(((fit_metrics or {}).get("fit_MAP") or {}).get("max_abs"))

    # (B) ROM error stats from diagnostics_tables
    diag_dir = run_dir / "diagnostics_tables"
    rom_path = diag_dir / f"{model}_rom_error.csv"
    rom_df, _ = _read_csv_df(rom_path)
    if rom_df is not None and "rom_error" in rom_df.columns:
        rom_vals = [float(x) for x in rom_df["rom_error"].to_list() if _safe_float(x) is not None]
        rom_stats = _summ_stats(rom_vals)
    else:
        rom_rows, _ = _read_csv_rows(rom_path)
        rom_vals = [_safe_float(r.get("rom_error")) for r in rom_rows]
        rom_stats = _summ_stats([v for v in rom_vals if v is not None])

    # (C) Acceptance rate stats
    acc_path = diag_dir / f"{model}_acceptance_rate.csv"
    acc_df, _ = _read_csv_df(acc_path)
    if acc_df is not None and "accept_rate" in acc_df.columns:
        acc_vals = [float(x) for x in acc_df["accept_rate"].to_list() if _safe_float(x) is not None]
        acc_stats = _summ_stats(acc_vals)
    else:
        acc_rows, _ = _read_csv_rows(acc_path)
        acc_vals = [_safe_float(r.get("accept_rate")) for r in acc_rows]
        acc_stats = _summ_stats([v for v in acc_vals if v is not None])

    # (D) Beta schedule
    beta_path = diag_dir / f"{model}_beta_schedule.csv"
    beta_df, _ = _read_csv_df(beta_path)
    if beta_df is not None and "beta" in beta_df.columns:
        beta_vals = [float(x) for x in beta_df["beta"].to_list() if _safe_float(x) is not None]
    else:
        beta_rows, _ = _read_csv_rows(beta_path)
        beta_vals = [_safe_float(r.get("beta")) for r in beta_rows]
        beta_vals = [v for v in beta_vals if v is not None]

    beta_stats = _summ_stats(beta_vals)

    # (D2) Theta0 update summary
    theta0_path = diag_dir / f"{model}_theta0_history.csv"
    theta0_updates = None
    theta0_step_norm_max = None
    theta0_df, _ = _read_csv_df(theta0_path)
    if theta0_df is not None and "step_norm" in theta0_df.columns:
        # step_norm is empty for update=0
        try:
            step = pd.to_numeric(theta0_df["step_norm"], errors="coerce") if pd is not None else theta0_df["step_norm"]
            step = np.asarray(step, dtype=float)
            step = step[np.isfinite(step)]
            theta0_updates = int(step.size) if step.size > 0 else 0
            theta0_step_norm_max = float(np.max(step)) if step.size > 0 else None
        except Exception:
            theta0_updates = None
            theta0_step_norm_max = None
    else:
        rows, _ = _read_csv_rows(theta0_path)
        steps = [_safe_float(r.get("step_norm")) for r in rows]
        steps = [v for v in steps if v is not None]
        theta0_updates = len(steps) if steps else 0
        theta0_step_norm_max = max(steps) if steps else None

    # (E) ESS from npz diagnostics
    ess_min = None
    ess_mean = None
    if npz_diag and isinstance(npz_diag, dict):
        ess_ref = npz_diag.get("ESS_reference")
        try:
            if ess_ref is not None:
                arr = np.asarray(ess_ref, dtype=float).reshape(-1)
                arr = arr[np.isfinite(arr)]
                if arr.size > 0:
                    ess_min = float(np.min(arr))
                    ess_mean = float(np.mean(arr))
        except Exception:
            pass

    return ModelKeyMetrics(
        rmse_total=rmse_total,
        mae_total=mae_total,
        max_abs=max_abs,
        rom_error_last=rom_stats["last"],
        rom_error_max=rom_stats["max"],
        rom_error_mean=rom_stats["mean"],
        ess_min=ess_min,
        ess_mean=ess_mean,
        accept_rate_mean=acc_stats["mean"],
        accept_rate_min=acc_stats["min"],
        accept_rate_last=acc_stats["last"],
        beta_final=beta_stats["last"],
        beta_stages=(len(beta_vals) - 1) if beta_vals else None,
        theta0_updates=theta0_updates,
        theta0_step_norm_max=theta0_step_norm_max,
    )


def _overall_status(per_model: Dict[str, ModelKeyMetrics]) -> Tuple[str, List[str]]:
    reasons: List[str] = []

    # Aggregate: worst-case across models
    ess_min_all = [m.ess_min for m in per_model.values() if m.ess_min is not None]
    acc_mean_all = [m.accept_rate_mean for m in per_model.values() if m.accept_rate_mean is not None]
    rom_last_all = [m.rom_error_last for m in per_model.values() if m.rom_error_last is not None]

    ess_min = min(ess_min_all) if ess_min_all else None
    acc_mean = min(acc_mean_all) if acc_mean_all else None
    rom_last = max(rom_last_all) if rom_last_all else None

    if ess_min is None:
        reasons.append("ESS_min missing (using WARN by default for ESS gate)")
    if acc_mean is None:
        reasons.append("accept_rate_mean missing (using WARN by default for acceptance gate)")
    if rom_last is None:
        reasons.append("rom_error_final missing (using WARN by default for ROM gate)")

    # FAIL checks (only when metric is available)
    fail = False
    if ess_min is not None and ess_min < FAIL_ESS_MIN:
        fail = True
        reasons.append(f"FAIL: ESS_min {ess_min:.3g} < {FAIL_ESS_MIN}")
    if acc_mean is not None and acc_mean < FAIL_ACCEPT_RATE_MEAN:
        fail = True
        reasons.append(f"FAIL: accept_rate_mean {acc_mean:.3g} < {FAIL_ACCEPT_RATE_MEAN}")
    if rom_last is not None and rom_last > FAIL_ROM_ERROR_MAX:
        fail = True
        reasons.append(f"FAIL: rom_error_final {rom_last:.3g} > {FAIL_ROM_ERROR_MAX}")
    if fail:
        return "FAIL", reasons

    # WARN checks
    warn = False
    if ess_min is not None and ess_min < WARN_ESS_MIN:
        warn = True
        reasons.append(f"WARN: ESS_min {ess_min:.3g} < {WARN_ESS_MIN}")
    if rom_last is not None and rom_last > WARN_ROM_ERROR_MAX:
        warn = True
        reasons.append(f"WARN: rom_error_final {rom_last:.3g} > {WARN_ROM_ERROR_MAX}")

    if warn or reasons:
        return "WARN", reasons
    return "PASS", reasons


def _pick_best_model(per_model: Dict[str, ModelKeyMetrics]) -> Tuple[Optional[str], str]:
    """
    Select best model by lowest RMSE (MAP fit). Fallbacks if missing.
    """
    candidates: List[Tuple[float, str]] = []
    for m, km in per_model.items():
        if km.rmse_total is not None:
            candidates.append((km.rmse_total, m))
    if candidates:
        candidates.sort()
        return candidates[0][1], "lowest rmse_total (MAP)"

    # fallback: lowest MAE
    candidates2: List[Tuple[float, str]] = []
    for m, km in per_model.items():
        if km.mae_total is not None:
            candidates2.append((km.mae_total, m))
    if candidates2:
        candidates2.sort()
        return candidates2[0][1], "lowest mae_total (MAP)"

    return None, "missing fit_metrics"


def _plot_pbox_assets(
    run_dir: Path,
    per_model: Dict[str, ModelKeyMetrics],
    npz_data: Dict[str, Any],
    config: Optional[Dict[str, Any]],
) -> Tuple[List[str], List[str]]:
    """
    Generate p-box plots using pbox.py.
    """
    assets_dir = run_dir / "report_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    
    generated = []
    warnings = []
    
    theta_true = npz_data.get("theta_true")
    
    # Prior bounds from config or default (hardcoded default for now matching case2)
    # Ideally should be in config, but we can fallback.
    default_bounds = [(0.0, 3.0)] * 14
    
    for m in per_model.keys():
        samples = npz_data.get(f"samples_{m}")
        if samples is None:
            # warnings.append(f"{m}: samples missing in npz, cannot plot p-box")
            continue
            
        try:
            # Active indices for this model
            # Need to get param_names and active indices from config?
            # We can use placeholder names if config missing, or try to get from config json.
            model_cfg = (config or {}).get("tmcmc", {}).get(m) # wrong structure
            # Access MODEL_CONFIGS via config if available? No, config.json has flat structure mostly.
            # Best effort: use generic names or fixed length
            
            # Use pbox utility
            bounds = pbox.compute_pbox_bounds(samples)
            
            # Param names: create dummy if needed
            n_params = samples.shape[1]
            param_names = [f"p{i}" for i in range(n_params)]
            # If we know model, we can map active indices.
            # But here we just plot the sampled parameters (the active ones).
            if m == "M1":
                param_names = ["a11", "a12", "a22", "b1", "b2"]
            elif m == "M2":
                param_names = ["a33", "a34", "a44", "b3", "b4"]
            elif m == "M3":
                param_names = ["a13", "a14", "a23", "a24"]
            
            # True values slice
            # Needs active indices mapping.
            # M1: [0:5], M2: [5:10], M3: [10:14]
            theta_true_sub = None
            if theta_true is not None:
                if m == "M1":
                   theta_true_sub = theta_true[0:5]
                elif m == "M2":
                   theta_true_sub = theta_true[5:10]
                elif m == "M3":
                   theta_true_sub = theta_true[10:14]
            
            # Prior bounds slice
            # Assuming uniform prior [0,3] for all.
            prior_bounds = [(0.0, 3.0)] * n_params
            
            fname = f"{m}_pbox.png"
            pbox.plot_pbox_comparison(
                pbox_posterior=bounds,
                prior_bounds=prior_bounds,
                param_names=param_names,
                theta_true=theta_true_sub,
                theta_map=npz_data.get(f"MAP_{m}"),
                filename=str(assets_dir / fname),
                title=f"{m} p-box (min-max posterior)",
            )
            generated.append(str((Path("report_assets") / fname).as_posix()))
        except Exception as e:
            warnings.append(f"{m} p-box plot failed: {e}")
            
    return generated, warnings


def _plot_timeseries_assets(
    run_dir: Path,
    per_model: Dict[str, ModelKeyMetrics],
    npz_diags: Dict[str, Any],
) -> Tuple[List[str], List[str]]:
    """
    Create report_assets/*.png (best-effort). Returns (generated_paths, warnings).
    Paths are relative to run_dir (POSIX).
    """
    assets_dir = run_dir / "report_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    plt, plt_err = _try_import_matplotlib()
    if plt is None:
        return [], [f"matplotlib missing; skipping report_assets plots ({plt_err})"]

    generated: List[str] = []
    warnings: List[str] = []

    diag_dir = run_dir / "diagnostics_tables"

    def save(fig_name: str):
        p = assets_dir / fig_name
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        generated.append(str((Path("report_assets") / fig_name).as_posix()))

    # Acceptance rate timeseries
    for m in per_model.keys():
        df, err = _read_csv_df(diag_dir / f"{m}_acceptance_rate.csv")
        if df is None:
            warnings.append(f"{m}: acceptance_rate.csv missing ({err})")
            continue
        if not {"stage", "accept_rate", "chain"}.issubset(set(df.columns)):
            warnings.append(f"{m}: acceptance_rate.csv columns unexpected: {list(df.columns)}")
            continue
        plt.figure(figsize=(7, 3))
        for chain_id, g in df.groupby("chain"):
            plt.plot(g["stage"], g["accept_rate"], marker="o", label=f"chain {chain_id}", linewidth=1.5)
        plt.ylim(0.0, 1.0)
        plt.xlabel("stage")
        plt.ylabel("accept_rate")
        plt.title(f"{m} acceptance rate")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        save(f"{m}_acceptance_rate.png")

    # ROM error timeseries
    for m in per_model.keys():
        df, err = _read_csv_df(diag_dir / f"{m}_rom_error.csv")
        if df is None:
            warnings.append(f"{m}: rom_error.csv missing ({err})")
            continue
        if not {"update", "rom_error", "chain"}.issubset(set(df.columns)):
            warnings.append(f"{m}: rom_error.csv columns unexpected: {list(df.columns)}")
            continue
        plt.figure(figsize=(7, 3))
        for chain_id, g in df.groupby("chain"):
            plt.plot(g["update"], g["rom_error"], marker="o", label=f"chain {chain_id} (post)", linewidth=1.5)
            if "rom_error_pre" in g.columns:
                plt.plot(
                    g["update"],
                    g["rom_error_pre"],
                    marker="x",
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.6,
                    label=f"chain {chain_id} (pre)",
                )
        plt.axhline(WARN_ROM_ERROR_MAX, linestyle="--", linewidth=1, color="orange", alpha=0.7, label="WARN")
        plt.axhline(FAIL_ROM_ERROR_MAX, linestyle="--", linewidth=1, color="red", alpha=0.7, label="FAIL")
        plt.xlabel("linearization update")
        plt.ylabel("rom_error")
        plt.title(f"{m} ROM error at updates")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        save(f"{m}_rom_error.png")

    # Theta0 step norm (if available)
    for m in per_model.keys():
        df, err = _read_csv_df(diag_dir / f"{m}_theta0_history.csv")
        if df is None:
            warnings.append(f"{m}: theta0_history.csv missing ({err})")
            continue
        if not {"update", "step_norm", "chain"}.issubset(set(df.columns)):
            warnings.append(f"{m}: theta0_history.csv columns unexpected: {list(df.columns)}")
            continue
        # step_norm may contain empty values
        df2 = df.copy()
        df2["step_norm"] = pd.to_numeric(df2["step_norm"], errors="coerce") if pd is not None else df2["step_norm"]
        plt.figure(figsize=(7, 3))
        n_lines = 0
        for chain_id, g in df2.groupby("chain"):
            gg = g.dropna(subset=["step_norm"])
            if gg.empty:
                continue
            plt.plot(gg["update"], gg["step_norm"], marker="o", label=f"chain {chain_id}", linewidth=1.5)
            n_lines += 1
        plt.xlabel("linearization update")
        plt.ylabel("||Δθ0||")
        plt.title(f"{m} theta0 step norm")
        plt.grid(True, alpha=0.3)
        if n_lines > 0:
            plt.legend(fontsize=8)
        save(f"{m}_theta0_step_norm.png")

    # ESS reference (from npz) plot
    for m in per_model.keys():
        d = npz_diags.get(m)
        if not isinstance(d, dict) or d.get("ESS_reference") is None:
            warnings.append(f"{m}: ESS_reference missing in results npz")
            continue
        try:
            ess = np.asarray(d.get("ESS_reference"), dtype=float).reshape(-1)
            ess = ess[np.isfinite(ess)]
            if ess.size == 0:
                warnings.append(f"{m}: ESS_reference empty")
                continue
            plt.figure(figsize=(7, 3))
            plt.plot(range(len(ess)), ess, marker="o", linewidth=1.5)
            plt.axhline(WARN_ESS_MIN, linestyle="--", linewidth=1, color="orange", alpha=0.7, label="WARN")
            plt.axhline(FAIL_ESS_MIN, linestyle="--", linewidth=1, color="red", alpha=0.7, label="FAIL")
            plt.xlabel("stage index (reference)")
            plt.ylabel("ESS_reference")
            plt.title(f"{m} ESS_reference")
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=8)
            save(f"{m}_ESS_reference.png")
        except Exception as e:  # pragma: no cover
            warnings.append(f"{m}: ESS plot error: {type(e).__name__}: {e}")

    return generated, warnings


def _md_link(path_posix: str, label: Optional[str] = None) -> str:
    label = label or path_posix
    return f"[`{label}`]({path_posix})"


def _render_report(
    run_dir: Path,
    config: Optional[Dict[str, Any]],
    metrics: Optional[Dict[str, Any]],
    fit_by_model: Dict[str, Optional[Dict[str, Any]]],
    per_model: Dict[str, ModelKeyMetrics],
    status: str,
    status_reasons: List[str],
    figures: List[str],
    assets: List[str],
    assets_warnings: List[str],
    best_model: Optional[str],
    best_reason: str,
    models: Tuple[str, ...],
) -> str:
    run_id = (config or {}).get("run_id") or run_dir.name
    mode = (config or {}).get("mode") or (metrics or {}).get("mode") or "missing"
    start_time = (config or {}).get("start_time") or "missing"
    end_time = (metrics or {}).get("end_time") or "missing"
    seed = ((config or {}).get("seeds") or {}).get("base_seed")
    tmcmc_cfg = (config or {}).get("tmcmc") or {}
    exp_cfg = (config or {}).get("experiment") or {}

    # Key metrics table rows
    def row(m: str) -> List[str]:
        km = per_model[m]
        return [
            m,
            _fmt(km.rmse_total, 4),
            _fmt(km.mae_total, 4),
            _fmt(km.max_abs, 4),
            _fmt(km.rom_error_last, 4),
            _fmt(km.ess_min, 4),
            _fmt(km.accept_rate_mean, 4),
            _fmt(km.beta_final, 4),
            str(km.beta_stages if km.beta_stages is not None else "missing"),
        ]

    key_headers = [
        "Model",
        "RMSE_total (MAP)",
        "MAE_total (MAP)",
        "max_abs (MAP)",
        "rom_error_final",
        "ESS_min",
        "accept_rate_mean",
        "beta_final",
        "beta_stages",
    ]
    key_rows = [row(m) for m in models]

    # Model comparison table (fit_metrics)
    comp_headers = ["Model", "rmse_total(MAP)", "mae_total(MAP)", "rom_error_MAP_vs_FOM", "rom_error_MEAN_vs_FOM"]
    comp_rows: List[List[str]] = []
    for m in models:
        fm = fit_by_model.get(m) or {}
        fit_map = fm.get("fit_MAP") or {}
        comp_rows.append(
            [
                m,
                _fmt(fit_map.get("rmse_total"), 4),
                _fmt(fit_map.get("mae_total"), 4),
                _fmt(fm.get("rom_error_MAP_vs_FOM"), 4),
                _fmt(fm.get("rom_error_MEAN_vs_FOM"), 4),
            ]
        )

    # Diagnostics bullets
    diag_lines: List[str] = []
    for m in models:
        km = per_model[m]
        diag_lines.append(
            f"- **{m}**: beta_final={_fmt(km.beta_final)}, stages={km.beta_stages if km.beta_stages is not None else 'missing'}, "
            f"accept_mean={_fmt(km.accept_rate_mean)} (min={_fmt(km.accept_rate_min)}, last={_fmt(km.accept_rate_last)}), "
            f"ESS_min={_fmt(km.ess_min)} (mean={_fmt(km.ess_mean)}), "
            f"rom_final={_fmt(km.rom_error_last)} (max={_fmt(km.rom_error_max)}, mean={_fmt(km.rom_error_mean)}), "
            f"theta0_updates={km.theta0_updates if km.theta0_updates is not None else 'missing'} (max_step={_fmt(km.theta0_step_norm_max)})"
        )
    if assets_warnings:
        diag_lines.append("- **report_assets warnings**:")
        for w in assets_warnings:
            diag_lines.append(f"  - {w}")

    # Figures list
    fig_lines: List[str] = []
    if assets:
        fig_lines.append("- **report_assets (auto-generated)**:")
        for p in assets:
            fig_lines.append(f"  - {_md_link(p)}")
    fig_lines.append("- **run figures (from FIGURES_MANIFEST.json)**:")
    if figures:
        for fn in figures:
            fig_lines.append(f"  - {_md_link(str((Path('figures')/fn).as_posix()))}")
    else:
        fig_lines.append("  - missing")

    # Next actions (templated)
    next_lines: List[str] = []
    next_lines.append(f"- **Overall status**: {status}")
    if status_reasons:
        next_lines.append("- **Reasons**:")
        for r in status_reasons:
            next_lines.append(f"  - {r}")

    # Fail-driven suggestions (use aggregated worst-case)
    ess_min_all = [per_model[m].ess_min for m in models if per_model[m].ess_min is not None]
    acc_mean_all = [per_model[m].accept_rate_mean for m in models if per_model[m].accept_rate_mean is not None]
    rom_last_all = [per_model[m].rom_error_last for m in models if per_model[m].rom_error_last is not None]
    ess_min = min(ess_min_all) if ess_min_all else None
    acc_mean = min(acc_mean_all) if acc_mean_all else None
    rom_last = max(rom_last_all) if rom_last_all else None

    if ess_min is not None and ess_min < WARN_ESS_MIN:
        next_lines.append("- **If ESS is low**:")
        next_lines.append("  - Increase `--n-particles` (first), then consider more `--n-stages`.")
        next_lines.append("  - Consider loosening likelihood (e.g., larger `--sigma-obs`) for exploration, then tighten.")
    if any((per_model[m].beta_final is not None and per_model[m].beta_final < 0.999) for m in models):
        next_lines.append("- **If beta does not reach 1.0 (not fully tempered to posterior)**:")
        next_lines.append("  - Increase `--n-stages` and/or adjust ESS target so β can progress to 1.0.")
    if acc_mean is not None and acc_mean < 0.1:
        next_lines.append("- **If acceptance rate is low**:")
        next_lines.append("  - Reduce proposal scale (or lower mutation scale inside TMCMC) and/or increase mutation steps.")
        next_lines.append("  - Check for numerical issues (NaN/Inf) by using `--debug-level ERROR` or `VERBOSE`.")
    if rom_last is not None and rom_last > WARN_ROM_ERROR_MAX:
        next_lines.append("- **If ROM error is high**:")
        next_lines.append("  - Enable/strengthen linearization updates (observation-based update) and ensure updates actually happen.")
        next_lines.append("  - Increase `--cov-rel` only if it represents uncertainty; otherwise reduce step sizes / update frequency.")

    # Best model
    best_line = f"{best_model} ({best_reason})" if best_model else f"missing ({best_reason})"

    md = []
    md.append(f"# REPORT — {run_id}")
    md.append("")
    md.append("## 1) Run summary")
    md.append("")
    md.append(f"- **run_id**: `{run_id}`")
    md.append(f"- **mode**: `{mode}`")
    md.append(f"- **start_time**: `{start_time}`")
    md.append(f"- **end_time**: `{end_time}`")
    md.append(f"- **seed**: `{seed if seed is not None else 'missing'}`")
    md.append(f"- **tmcmc**: n_particles={tmcmc_cfg.get('n_particles','missing')}, n_stages={tmcmc_cfg.get('n_stages','missing')}, n_mutation_steps={tmcmc_cfg.get('n_mutation_steps','missing')}, n_chains={tmcmc_cfg.get('n_chains','missing')}")
    md.append(f"- **experiment**: n_data={exp_cfg.get('n_data','missing')}, sigma_obs={exp_cfg.get('sigma_obs','missing')}, cov_rel={exp_cfg.get('cov_rel','missing')}")
    md.append("")
    md.append("## 2) Pass/Warn/Fail 判定")
    md.append("")
    md.append(f"- **status**: **{status}**")
    if status_reasons:
        md.append("- **details**:")
        for r in status_reasons:
            md.append(f"  - {r}")
    md.append("")
    md.append("## 3) Key metrics 表")
    md.append("")
    md.append("| " + " | ".join(key_headers) + " |")
    md.append("| " + " | ".join(["---"] * len(key_headers)) + " |")
    for r in key_rows:
        md.append("| " + " | ".join(r) + " |")
    md.append("")
    md.append("## 4) Model comparison")
    md.append("")
    md.append(f"- **best model**: **{best_line}**")
    md.append("")
    md.append("| " + " | ".join(comp_headers) + " |")
    md.append("| " + " | ".join(["---"] * len(comp_headers)) + " |")
    for r in comp_rows:
        md.append("| " + " | ".join(r) + " |")
    md.append("")
    md.append("## 5) Diagnostics")
    md.append("")
    md.extend(diag_lines if diag_lines else ["- missing"])
    md.append("")
    md.append("## 6) Figures")
    md.append("")
    md.extend(fig_lines)
    md.append("")
    md.append("## 7) Next actions")
    md.append("")
    md.extend(next_lines)
    md.append("")
    md.append("---")
    md.append(f"_Generated by `tmcmc/make_report.py` at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_")
    md.append("")
    return "\n".join(md)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate REPORT.md for a tmcmc run directory",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-dir", type=str, default=None, help="Run directory path (tmcmc/_runs/<run_id>)")
    g.add_argument("--run-id", type=str, default=None, help="Run id under --runs-root")
    p.add_argument("--runs-root", type=str, default="tmcmc/_runs", help="Root directory containing runs")
    return p.parse_args(argv)


def main() -> int:
    setup_logging("INFO")
    args = parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = Path(args.runs_root) / str(args.run_id)

    run_dir = run_dir.resolve()
    if not run_dir.exists():
        raise SystemExit(f"Run dir not found: {run_dir}")

    # Persist report generation logs under the run directory.
    setup_logging("INFO", log_path=run_dir / "report.log")

    # Load core artifacts (best-effort)
    config, config_err = _read_json(run_dir / "config.json")
    metrics, metrics_err = _read_json(run_dir / "metrics.json")
    manifest, manifest_err = _read_json(run_dir / "figures" / "FIGURES_MANIFEST.json")

    models = _models_for_run(config, metrics)

    fit_by_model: Dict[str, Optional[Dict[str, Any]]] = {}
    fit_errs: Dict[str, str] = {}
    for m in models:
        fm, err = _read_json(run_dir / f"fit_metrics_{m}.json")
        fit_by_model[m] = fm
        if err:
            fit_errs[m] = err

    # Optional diagnostics from npz (for ESS_reference)
    npz_data = _load_npz_data(run_dir)

    # Aggregate model metrics
    per_model: Dict[str, ModelKeyMetrics] = {}
    for m in models:
        per_model[m] = _collect_model_metrics(run_dir, m, fit_by_model.get(m), npz_data.get(m))

    status, reasons = _overall_status(per_model)
    if config_err:
        reasons.append(f"config.json: {config_err}")
    if metrics_err:
        reasons.append(f"metrics.json: {metrics_err}")
    if manifest_err:
        reasons.append(f"figures/FIGURES_MANIFEST.json: {manifest_err}")
    for m, err in fit_errs.items():
        reasons.append(f"fit_metrics_{m}.json: {err}")

    best_model, best_reason = _pick_best_model(per_model)

    figures = []
    if isinstance(manifest, dict):
        figs = manifest.get("figures")
        if isinstance(figs, list):
            figures = [str(x) for x in figs if isinstance(x, (str, Path))]

    # Assets
    assets_ts, ts_warnings = _plot_timeseries_assets(run_dir, per_model, npz_data)
    assets_pbox, pbox_warnings = _plot_pbox_assets(run_dir, per_model, npz_data, config)
    
    assets = assets_ts + assets_pbox
    assets_warnings = ts_warnings + pbox_warnings

    md = _render_report(
        run_dir=run_dir,
        config=config,
        metrics=metrics,
        fit_by_model=fit_by_model,
        per_model=per_model,
        status=status,
        status_reasons=reasons,
        figures=figures,
        assets=assets,
        assets_warnings=assets_warnings,
        best_model=best_model,
        best_reason=best_reason,
        models=models,
    )

    out_path = run_dir / "REPORT.md"
    out_path.write_text(md, encoding="utf-8")

    logger.info("REPORT.md generated: %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

