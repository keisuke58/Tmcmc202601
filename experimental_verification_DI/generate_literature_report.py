#!/usr/bin/env python3
"""
generate_literature_report.py — 文献との整合性レポート生成

モデル E(DI) 曲線と文献データ（Pattem, Gloag）の整合性を定量的に評価。
論文の「実験的裏付け」セクション用の表・図を出力。

Usage
-----
  python -m experimental_verification_DI.generate_literature_report
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_TMCMC_ROOT = _SCRIPT_DIR.parent
for p in [_TMCMC_ROOT, _TMCMC_ROOT.parent]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from experimental_verification_DI.fit_constitutive import predict_E_DI
from experimental_verification_DI.literature_data import get_literature_points


def main() -> int:
    e_max, e_min = 909.0, 32.0
    di_scale, exponent = 1.0, 2.0

    points = get_literature_points()
    out_dir = _SCRIPT_DIR / "_validation_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in points:
        if p.di_approx is None:
            continue
        E_model = predict_E_DI(np.array([p.di_approx]), e_max, e_min, di_scale, exponent)[0]
        E_model_kPa = E_model / 1e3
        residual = (E_model_kPa - p.E_kPa) / p.E_kPa * 100 if p.E_kPa > 0 else 0
        rows.append(
            {
                "label": p.label,
                "source": p.source,
                "di_approx": p.di_approx,
                "E_lit_kPa": p.E_kPa,
                "E_model_kPa": float(E_model_kPa),
                "residual_pct": float(residual),
            }
        )

    report = {
        "model_params": {
            "e_max_Pa": e_max,
            "e_min_Pa": e_min,
            "di_scale": di_scale,
            "exponent": exponent,
        },
        "literature_comparison": rows,
        "summary": {
            "n_points": len(rows),
            "mean_abs_residual_pct": float(np.mean([abs(r["residual_pct"]) for r in rows])),
            "conclusion": "Literature (Pattem, Gloag) reports different biofilm systems and protocols; absolute E values differ. Our model captures the qualitative trend (diverse→stiffer, dysbiotic→softer). Direct validation requires same-sample 16S+AFM data.",
        },
    }

    out_json = out_dir / "literature_consistency_report.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved: {out_json}")

    # Markdown report
    md_lines = [
        "# Literature Consistency Report",
        "",
        "## Model Parameters",
        f"- E_max = {e_max} Pa, E_min = {e_min} Pa",
        f"- di_scale = {di_scale}, exponent = {exponent}",
        "",
        "## Comparison with Literature",
        "",
        "| Source | DI (approx) | E_lit [kPa] | E_model [kPa] | Residual [%] |",
        "|--------|-------------|-------------|---------------|---------------|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['label']} | {r['di_approx']:.2f} | {r['E_lit_kPa']:.2f} | {r['E_model_kPa']:.2f} | {r['residual_pct']:+.0f} |"
        )
    md_lines.extend(
        [
            "",
            "## Summary",
            f"- Mean |residual| = {report['summary']['mean_abs_residual_pct']:.0f}%",
            f"- {report['summary']['conclusion']}",
            "",
            "## Next Step",
            "Direct experimental validation: 16S rRNA + AFM on same sample. See EXPERIMENTAL_PROTOCOL.md.",
        ]
    )

    out_md = out_dir / "literature_consistency_report.md"
    with open(out_md, "w") as f:
        f.write("\n".join(md_lines))
    print(f"Saved: {out_md}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
