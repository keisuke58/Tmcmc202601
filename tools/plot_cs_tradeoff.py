#!/usr/bin/env python3
"""CS: 生きた割合の重み λψ を変えたときの組成の誤差と生きた割合の誤差（補足 S13 の図、修論 図 3.3）。

値は 2026-10-13 の決定論的な最適化（生データ・2 通路、handoff 13p の表）。2026-10-19: 元のスクリプトが残っていなかったので
表から描き直す（右端の「λψ = 1.5」のラベルが枠で切れていたのを直した）。

    python3 tools/plot_cs_tradeoff.py --out docs/revision/EML_submission/figures/raw/cs_tradeoff
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# λψ, 生きた菌の組成, 全菌の組成, 生きた割合（RMSE）
ROWS = [(1.5, 0.092, 0.039, 0.231), (3, 0.103, 0.100, 0.174), (6, 0.120, 0.120, 0.132),
        (12, 0.144, 0.141, 0.074), (24, 0.158, 0.149, 0.051)]

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True, help="拡張子なし（.pdf と .png を書く）")
a = ap.parse_args()

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eml_style  # noqa: E402

eml_style.apply()  # 本文と同じ Latin Modern（usetex、2026-10-19）
plt.rcParams.update({"font.size": 8})
lam = [r[0] for r in ROWS]
via = [r[1] for r in ROWS]
tot = [r[2] for r in ROWS]
live = [r[3] for r in ROWS]
fig, ax = plt.subplots(figsize=(3.4, 2.4))
ax.plot(live, via, "o-", color="#1f6fa8", ms=3.5, lw=1.0, label="viable composition")
ax.plot(live, tot, "s--", color="0.4", ms=3.0, lw=0.9, label="total-count composition")
for lv, v, l in zip(live, via, lam):
    ax.annotate(rf"$\lambda_\psi={l:g}$", (lv, v), xytext=(0, 5), textcoords="offset points",
                ha="center", va="bottom", fontsize=6.5)
ax.set_xlim(0.035, 0.25)
ax.set_ylim(0.03, 0.175)
ax.set_xlabel("RMSE live fraction")
ax.set_ylabel("RMSE composition")
ax.grid(alpha=0.25, lw=0.5)
ax.legend(loc="upper right", fontsize=6.5, frameon=False)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"{a.out}.{ext}", dpi=200)
print("->", a.out)
