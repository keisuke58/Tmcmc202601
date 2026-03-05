#!/usr/bin/env python3
"""
run_validation.py — DI→E 実験検証パイプラインの実行

合成データでフィッティングを検証し、結果を図とレポートに出力する。
実データが得られた場合は data/di_e_pairs.csv に配置して実行。

Usage
-----
  # 合成データで検証（デフォルト）
  python -m experimental_verification_DI.run_validation

  # 実データでフィット
  python -m experimental_verification_DI.run_validation --data data/di_e_pairs.csv

  # 出力先指定
  python -m experimental_verification_DI.run_validation --out-dir _validation_output
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

# Tmcmc202601 を path に追加
_SCRIPT_DIR = Path(__file__).resolve().parent
_TMCMC_ROOT = _SCRIPT_DIR.parent
if str(_TMCMC_ROOT) not in sys.path:
    sys.path.insert(0, str(_TMCMC_ROOT))

from experimental_verification_DI.data_loader import (
    load_experimental_data,
    save_experimental_data,
)
from experimental_verification_DI.fit_constitutive import fit_E_DI, predict_E_DI
from experimental_verification_DI.synthetic_data import (
    generate_synthetic_data,
    generate_condition_aware_data,
)
from experimental_verification_DI.experiment_design import run_design_estimate
from experimental_verification_DI.literature_data import get_literature_arrays

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_synthetic_validation(out_dir: Path) -> dict:
    """合成データでフィット検証を実行。"""
    logger.info("Generating synthetic (DI, E) data...")
    data = generate_synthetic_data(n_samples=40, noise_frac=0.12, seed=42)

    logger.info("Fitting E(DI) constitutive law...")
    result, report = fit_E_DI(
        data["di"],
        data["E"],
        E_err=data.get("E_err"),
    )

    # 真値との比較
    pt = data["params_true"]
    logger.info("Fit result:")
    logger.info("  e_max:   %.1f (true %.1f)", result.e_max, pt["e_max"])
    logger.info("  e_min:   %.1f (true %.1f)", result.e_min, pt["e_min"])
    logger.info("  di_scale: %.4f (true %.4f)", result.di_scale, pt["di_scale"])
    logger.info("  exponent: %.2f (true %.2f)", result.exponent, pt["exponent"])
    logger.info("  RMSE: %.1f Pa, R²=%.4f", result.rmse, result.r_squared)

    # 図の生成（matplotlib が使える場合）
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        di_plot = np.linspace(0, 1, 200)
        E_model = predict_E_DI(
            di_plot, result.e_max, result.e_min, result.di_scale, result.exponent
        )
        E_true_curve = predict_E_DI(
            di_plot, pt["e_max"], pt["e_min"], pt["di_scale"], pt["exponent"]
        )

        ax.plot(di_plot, E_model / 1e3, "b-", lw=2, label="Fitted E(DI)")
        ax.plot(di_plot, E_true_curve / 1e3, "k--", lw=1, alpha=0.7, label="True (synthetic)")
        ax.errorbar(
            data["di"],
            data["E"] / 1e3,
            yerr=(data["E_err"] / 1e3) if "E_err" in data else None,
            fmt="o",
            capsize=2,
            label="Synthetic data",
        )
        # 文献データオーバーレイ
        di_lit, e_lit, _ = get_literature_arrays()
        if len(di_lit) > 0:
            ax.scatter(
                di_lit,
                e_lit,
                marker="s",
                s=60,
                c="gray",
                alpha=0.7,
                label="Literature (Pattem, Gloag)",
            )
        ax.set_xlabel("Dysbiosis Index (DI)")
        ax.set_ylabel("Young modulus E [kPa]")
        ax.set_title("DI→E validation: synthetic data fit")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        fig.tight_layout()
        fig_path = out_dir / "validation_fit.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        logger.info("Saved figure: %s", fig_path)
    except ImportError:
        logger.warning("matplotlib not available, skipping figure")

    # レポート保存
    report_dict = {
        "fit": {
            "e_max": result.e_max,
            "e_min": result.e_min,
            "di_scale": result.di_scale,
            "exponent": result.exponent,
            "rmse": result.rmse,
            "r_squared": result.r_squared,
            "n_samples": result.n_samples,
        },
        "true": pt,
        "relative_error": {
            "e_max": abs(result.e_max - pt["e_max"]) / pt["e_max"],
            "e_min": abs(result.e_min - pt["e_min"]) / pt["e_min"],
        },
    }
    report_path = out_dir / "validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    logger.info("Saved report: %s", report_path)

    # 合成データを CSV に保存（フォーマット例として）
    csv_path = out_dir / "synthetic_di_e_data.csv"
    save_experimental_data(
        csv_path,
        data["di"],
        data["E"],
        E_err=data.get("E_err"),
    )
    logger.info("Saved synthetic data: %s", csv_path)

    return report_dict


def run_real_data_fit(data_path: Path, out_dir: Path) -> dict:
    """実データでフィット。"""
    logger.info("Loading experimental data from %s", data_path)
    data = load_experimental_data(data_path)

    result, report = fit_E_DI(
        data["di"],
        data["E"],
        E_err=data.get("E_err"),
    )

    logger.info("Fit result:")
    logger.info("  e_max=%.1f Pa, e_min=%.1f Pa", result.e_max, result.e_min)
    logger.info("  di_scale=%.4f, exponent=%.2f", result.di_scale, result.exponent)
    logger.info("  RMSE=%.1f Pa, R²=%.4f", result.rmse, result.r_squared)

    report_dict = {
        "fit": {
            "e_max": result.e_max,
            "e_min": result.e_min,
            "di_scale": result.di_scale,
            "exponent": result.exponent,
            "rmse": result.rmse,
            "r_squared": result.r_squared,
            "n_samples": result.n_samples,
        },
    }
    report_path = out_dir / "fit_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        di_plot = np.linspace(0, 1, 200)
        E_model = predict_E_DI(
            di_plot, result.e_max, result.e_min, result.di_scale, result.exponent
        )
        ax.plot(di_plot, E_model / 1e3, "b-", lw=2, label="Fitted E(DI)")
        ax.errorbar(
            data["di"],
            data["E"] / 1e3,
            yerr=(data["E_err"] / 1e3) if "E_err" in data else None,
            fmt="o",
            capsize=2,
            label="Experimental data",
        )
        di_lit, e_lit, _ = get_literature_arrays()
        if len(di_lit) > 0:
            ax.scatter(di_lit, e_lit, marker="s", s=60, c="gray", alpha=0.7, label="Literature")
        ax.set_xlabel("Dysbiosis Index (DI)")
        ax.set_ylabel("Young modulus E [kPa]")
        ax.set_title("DI→E: experimental validation")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "experimental_fit.png", dpi=150)
        plt.close(fig)
    except ImportError:
        pass

    return report_dict


def main() -> int:
    parser = argparse.ArgumentParser(description="DI→E experimental validation pipeline")
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to experimental (DI,E) CSV. If not set, use synthetic data.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_SCRIPT_DIR / "_validation_output",
        help="Output directory",
    )
    parser.add_argument(
        "--condition-aware",
        action="store_true",
        help="Use condition-aware synthetic data (4 conditions)",
    )
    parser.add_argument(
        "--design",
        action="store_true",
        help="Run sample size estimation (experiment design)",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.design:
        run_design_estimate(args.out_dir)
        return 0

    if args.data is not None and args.data.exists():
        run_real_data_fit(args.data, args.out_dir)
    else:
        if args.condition_aware:
            data = generate_condition_aware_data(n_per_condition=8, noise_frac=0.15, seed=42)
            csv_path = args.out_dir / "synthetic_4conditions.csv"
            save_experimental_data(
                csv_path,
                data["di"],
                data["E"],
                E_err=data["E_err"],
                condition=data["condition"],
                sample_id=data["sample_id"],
            )
            logger.info("Saved 4-condition synthetic data: %s", csv_path)
            # 4 条件データでもフィット実行
            run_real_data_fit(csv_path, args.out_dir)
        else:
            run_synthetic_validation(args.out_dir)

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
