#!/usr/bin/env python3
"""
export_fitted_params.py — フィット結果を FEM パイプライン用にエクスポート

検証で得た e_max, e_min, di_scale, exponent を JSON で出力。
FEM/material_models や multiscale パイプラインで読み込み可能にする。

Usage
-----
  python -m experimental_verification_DI.export_fitted_params [--report path/to/report.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_TMCMC_ROOT = _SCRIPT_DIR.parent
if str(_TMCMC_ROOT) not in sys.path:
    sys.path.insert(0, str(_TMCMC_ROOT))


def export_from_report(report_path: Path, out_path: Path | None = None) -> dict:
    """
    validation_report.json からパラメータを読み、FEM 用形式でエクスポート。

    Returns
    -------
    dict
        e_max_pa, e_min_pa, di_scale, exponent, source
    """
    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)
    fit = report.get("fit", report)
    params = {
        "e_max_pa": fit["e_max"],
        "e_min_pa": fit["e_min"],
        "di_scale": fit["di_scale"],
        "exponent": fit["exponent"],
        "source": str(report_path),
        "rmse": fit.get("rmse"),
        "r_squared": fit.get("r_squared"),
    }
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2)
    return params


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=_SCRIPT_DIR / "_validation_output" / "validation_report.json",
        help="Path to validation_report.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path (default: same dir as report, fitted_E_DI_params.json)",
    )
    args = parser.parse_args()

    if not args.report.exists():
        print(f"Report not found: {args.report}", file=sys.stderr)
        print("Run: python -m experimental_verification_DI.run_validation", file=sys.stderr)
        return 1

    out_path = args.out
    if out_path is None:
        out_path = args.report.parent / "fitted_E_DI_params.json"

    params = export_from_report(args.report, out_path)
    print(f"Exported to {out_path}:")
    print(json.dumps(params, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
