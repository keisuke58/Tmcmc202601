#!/usr/bin/env python3
"""
plot_E_DI_with_literature.py — E(DI) 曲線 + 文献データ（論文 Fig 11 風）

モデル曲線と Pattem 2018/2021, Gloag 2019 の文献値を重ねてプロット。
論文用のスタンドアロン図生成。

Usage
-----
  python -m experimental_verification_DI.plot_E_DI_with_literature [--out fig_E_DI_literature.png]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_TMCMC_ROOT = _SCRIPT_DIR.parent
if str(_TMCMC_ROOT) not in sys.path:
    sys.path.insert(0, str(_TMCMC_ROOT))

from experimental_verification_DI.fit_constitutive import predict_E_DI
from experimental_verification_DI.literature_data import get_literature_arrays


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path, default=_SCRIPT_DIR / "_validation_output" / "fig_E_DI_literature.png"
    )
    parser.add_argument("--e-max", type=float, default=909.0, help="E_max [Pa]")
    parser.add_argument("--e-min", type=float, default=32.0, help="E_min [Pa]")
    parser.add_argument("--di-scale", type=float, default=1.0, help="DI scale")
    parser.add_argument("--exponent", type=float, default=2.0, help="Exponent n")
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required", file=sys.stderr)
        return 1

    di_plot = np.linspace(0, 1, 300)
    E_model = predict_E_DI(di_plot, args.e_max, args.e_min, args.di_scale, args.exponent)
    di_lit, e_lit, labels = get_literature_arrays()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(di_plot, E_model / 1e3, "b-", lw=2.5, label=r"$E = E_{\max}(1-r)^n + E_{\min}\,r$")
    ax.scatter(
        di_lit, e_lit, marker="s", s=80, c="gray", alpha=0.8, label="Literature (Pattem, Gloag)"
    )
    for i, lbl in enumerate(labels):
        ax.annotate(
            lbl, (di_lit[i], e_lit[i]), xytext=(5, 5), textcoords="offset points", fontsize=8
        )
    ax.set_xlabel("Dysbiosis Index (DI)")
    ax.set_ylabel("Young modulus E [kPa]")
    ax.set_title("DI-dependent elastic modulus with literature overlay")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, None)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"Saved: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
