"""
improve_ph_model.py
===================
pH 予測モデル改善:  So+Vd (現行) vs So+An+Fn+Pg (最適)
90点 (0.5日解像度, HOBIC CH+DH) で LOO-CV 評価。
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "experiment_data"
OUT_DIR = SCRIPT_DIR / "paper_comprehensive_figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df_ph = pd.read_csv(DATA_DIR / "fig4A_pH_timeseries.csv")
t_all = df_ph["time_days"].values
ph_ch_all = df_ph["pH_commensal"].values
ph_dh_all = df_ph["pH_dysbiotic"].values

df_sp = pd.read_csv(DATA_DIR / "fig3_species_distribution_summary.csv")
CMAP = {
    "S. oralis": "So",
    "A. naeslundii": "An",
    "V. dispar": "Vd",
    "V. parvula": "Vd",
    "F. nucleatum": "Fn",
    "P. gingivalis_20709": "Pg",
    "P. gingivalis_W83": "Pg",
}


def get_interp(cond, cult, sp):
    sub = df_sp[(df_sp.condition == cond) & (df_sp.cultivation == cult)]
    sub = sub[sub.species.map(CMAP) == sp]
    if len(sub) == 0:
        return lambda t: np.zeros(len(np.asarray(t)))
    t_r = sub["day"].values
    f_r = sub["median"].values / 100.0
    return lambda t, t_r=t_r, f_r=f_r: np.interp(
        np.asarray(t), t_r, f_r, left=f_r[0], right=f_r[-1]
    )


# Use t ≥ 1 (species data available from day 1)
mask = t_all >= 1.0
t_vec = t_all[mask]
ph_ch = ph_ch_all[mask]
ph_dh = ph_dh_all[mask]

SPECIES = ["So", "An", "Vd", "Fn", "Pg"]
sp_ch = {sp: get_interp("Commensal", "HOBIC", sp)(t_vec) for sp in SPECIES}
sp_dh = {sp: get_interp("Dysbiotic", "HOBIC", sp)(t_vec) for sp in SPECIES}
all_sp = {sp: np.concatenate([sp_ch[sp], sp_dh[sp]]) for sp in SPECIES}
all_t = np.concatenate([t_vec, t_vec])
all_ph = np.concatenate([ph_ch, ph_dh])
N = len(all_ph)


# ---------------------------------------------------------------------------
# LOO-CV helper
# ---------------------------------------------------------------------------
def loo_cv(X, y):
    n = len(y)
    preds = np.zeros(n)
    for i in range(n):
        idx = np.arange(n) != i
        b, *_ = np.linalg.lstsq(X[idx], y[idx], rcond=None)
        preds[i] = X[i] @ b
    rmse = np.sqrt(np.mean((preds - y) ** 2))
    r2 = pearsonr(y, preds)[0] ** 2
    b_full, *_ = np.linalg.lstsq(X, y, rcond=None)
    return rmse, r2, preds, b_full


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
MODELS = {
    "A: So+Vd (current)": ["So", "Vd"],
    "B: So+Fn": ["So", "Fn"],
    "C: So+Fn+Pg": ["So", "Fn", "Pg"],
    "D: So+An+Fn+Pg (best)": ["So", "An", "Fn", "Pg"],
}

print(f"{'Model':<28} {'LOO-RMSE':>9} {'LOO-R²':>7}  β")
print("=" * 80)
results = {}
for name, flist in MODELS.items():
    X = np.column_stack([np.ones(N)] + [all_sp[f] for f in flist])
    rmse, r2, preds, beta = loo_cv(X, all_ph)
    results[name] = dict(flist=flist, rmse=rmse, r2=r2, preds=preds, beta=beta)
    bstr = "  ".join([f"β{f}={v:+.3f}" for f, v in zip(["0"] + flist, beta)])
    print(f"{name:<28} {rmse:9.4f} {r2:7.4f}  {bstr}")

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
fig.suptitle("pH Model Improvement: Replacing φ_Vd with φ_Fn/φ_Pg", fontsize=13, fontweight="bold")

COLORS = {
    "A: So+Vd (current)": "#9E9E9E",
    "B: So+Fn": "#FF9800",
    "C: So+Fn+Pg": "#4CAF50",
    "D: So+An+Fn+Pg (best)": "#E91E63",
}
LINES = {
    "A: So+Vd (current)": (0, (3, 3)),
    "B: So+Fn": "--",
    "C: So+Fn+Pg": "-.",
    "D: So+An+Fn+Pg (best)": "-",
}
n_half = len(t_vec)

# (0,0) CH time series
ax = axes[0, 0]
ax.plot(t_all, ph_ch_all, "k-", lw=2, alpha=0.8, label="Experiment")
for name, r in results.items():
    ax.plot(
        t_vec,
        r["preds"][:n_half],
        ls=LINES[name],
        color=COLORS[name],
        lw=1.5,
        label=f"{name[:15]}…",
    )
ax.set(xlabel="Day", ylabel="pH", title="Commensal HOBIC")
ax.set_ylim(6.0, 7.0)
ax.legend(fontsize=7)

# (0,1) DH time series
ax = axes[0, 1]
ax.plot(t_all, ph_dh_all, "k-", lw=2, alpha=0.8, label="Experiment")
for name, r in results.items():
    ax.plot(
        t_vec,
        r["preds"][n_half:],
        ls=LINES[name],
        color=COLORS[name],
        lw=1.5,
        label=f"{name[:15]}…",
    )
ax.set(xlabel="Day", ylabel="pH", title="Dysbiotic HOBIC")
ax.set_ylim(6.0, 7.2)
ax.legend(fontsize=7)

# (0,2) scatter: best vs current
ax = axes[0, 2]
for name, marker in [("A: So+Vd (current)", "s"), ("D: So+An+Fn+Pg (best)", "o")]:
    r = results[name]
    ax.scatter(
        all_ph,
        r["preds"],
        color=COLORS[name],
        s=20,
        alpha=0.6,
        marker=marker,
        label=f"{name} (R²={r['r2']:.3f})",
    )
lim = [6.1, 7.2]
ax.plot(lim, lim, "k--", lw=0.8, alpha=0.5)
ax.set(
    xlabel="Measured pH",
    ylabel="Predicted pH",
    title="LOO Predictions vs Measured",
    xlim=lim,
    ylim=lim,
)
ax.legend(fontsize=8)

# (1,0) LOO-RMSE bar
ax = axes[1, 0]
names_short = ["A\n(So+Vd)", "B\n(So+Fn)", "C\n(So+Fn+Pg)", "D\n(So+An+Fn+Pg)"]
rmse_vals = [results[k]["rmse"] for k in MODELS]
bars = ax.bar(range(4), rmse_vals, color=list(COLORS.values()), edgecolor="k", lw=0.5)
for i, v in enumerate(rmse_vals):
    ax.text(i, v + 0.0008, f"{v:.4f}", ha="center", fontsize=8, fontweight="bold")
ax.set_xticks(range(4))
ax.set_xticklabels(names_short, fontsize=8)
ax.set(ylabel="LOO-RMSE (pH)", title="LOO-RMSE by Model")
ax.set_ylim(0, max(rmse_vals) * 1.25)

# (1,1) LOO-R² bar
ax = axes[1, 1]
r2_vals = [results[k]["r2"] for k in MODELS]
bars = ax.bar(range(4), r2_vals, color=list(COLORS.values()), edgecolor="k", lw=0.5)
for i, v in enumerate(r2_vals):
    ax.text(i, v + 0.003, f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")
ax.set_xticks(range(4))
ax.set_xticklabels(names_short, fontsize=8)
ax.set(ylabel="LOO-R²", title="LOO-R² by Model")
ax.set_ylim(0, 1.05)

# (1,2) Correlation table
ax = axes[1, 2]
ax.axis("off")
corrs = {sp: np.corrcoef(all_sp[sp], all_ph)[0, 1] for sp in SPECIES}
best = results["D: So+An+Fn+Pg (best)"]
b = best["beta"]

table_lines = [
    "Species → pH correlation:",
    "",
]
for sp in SPECIES:
    bar = "█" * int(abs(corrs[sp]) * 12)
    sign = "+" if corrs[sp] > 0 else "-"
    table_lines.append(f"  φ_{sp:<2}: {sign}{abs(corrs[sp]):.3f}  {bar}")

table_lines += [
    "",
    "Best model (So+An+Fn+Pg):",
    f"  β₀   = {b[0]:+.3f}",
    f"  β_So = {b[1]:+.3f}  (acid↑)",
    f"  β_An = {b[2]:+.3f}  (succinate↑)",
    f"  β_Fn = {b[3]:+.3f}  (butyrate↑)",
    f"  β_Pg = {b[4]:+.3f}  (gingipain↑)",
    "",
    f"  LOO-RMSE = {best['rmse']:.4f}",
    f"  LOO-R²   = {best['r2']:.4f}",
    "",
    "vs current (So+Vd):",
    f"  LOO-RMSE = {results['A: So+Vd (current)']['rmse']:.4f}",
    f"  LOO-R²   = {results['A: So+Vd (current)']['r2']:.4f}",
    "",
    "→ Vd (Veillonella) r=−0.076",
    "  ≈ zero. Fn/Pg drive pH rise",
    "  in dysbiosis, not Vp.",
]
ax.text(
    0.03,
    0.98,
    "\n".join(table_lines),
    transform=ax.transAxes,
    fontsize=7.5,
    va="top",
    fontfamily="monospace",
    bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.9),
)

plt.tight_layout()
out = OUT_DIR / "improve_ph_model.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out}")


if __name__ == "__main__":
    pass
