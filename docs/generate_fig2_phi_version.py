#!/usr/bin/env python3
"""
generate_fig2_phi_version.py — φ_i 版 Fig 2 (publication quality)
=================================================================
Transposed layout (5 rows=species, 4 cols=conditions) + 箱ひげ図 + CI bands。
既存 regenerate_all_figures.py と同一スタイル。

JAX ODE 計算結果はキャッシュ (_cache/*.npz) を使用。
キャッシュがない場合は subprocess で JAX を実行して自動生成。

Usage:
    python generate_fig2_phi_version.py            # キャッシュから高速プロット
    python generate_fig2_phi_version.py --recompute # キャッシュ再生成
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ============================================================
# Paths
# ============================================================
ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "data_5species" / "main" / "_runs"
FIG_DIR = ROOT / "docs" / "figures"
FIG_DIR2 = ROOT / "docs" / "paper_figures"
CACHE_DIR = FIG_DIR2 / "_cache"
REP_CSV = ROOT / "data_5species" / "experiment_data" / "fig3_species_distribution_replicates.csv"

for d in [FIG_DIR, FIG_DIR2, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# Style (identical to regenerate_all_figures.py)
# ============================================================
STYLE = {
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "legend.fontsize": 7.5,
    "legend.framealpha": 0.9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "lines.linewidth": 1.2,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.2,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
}

# ============================================================
# Constants
# ============================================================
SPECIES_ITALIC = [
    r"$\it{S.\ oralis}$",
    r"$\it{A.\ naeslundii}$",
    r"$\it{V.\ dispar}$",
    r"$\it{F.\ nucleatum}$",
    r"$\it{P.\ gingivalis}$",
]
# Pg strain differs by condition
PG_STRAIN = {
    "CS": r"$\it{P.\ gingivalis}$ ATCC 20709",
    "CH": r"$\it{P.\ gingivalis}$ ATCC 20709",
    "DS": r"$\it{P.\ gingivalis}$ W83",
    "DH": r"$\it{P.\ gingivalis}$ W83",
}
SP_COLORS = {
    "C": ["#2166AC", "#1B7837", "#DAA520", "#7B3294", "#E07070"],  # Pg: light red (avirulent 20709)
    "D": ["#2166AC", "#1B7837", "#FF8C00", "#7B3294", "#8B0000"],  # Pg: dark red (virulent W83)
}
CONDS = ["CS", "CH", "DS", "DH"]
COND_LABELS = {
    "CS": r"$\bf{Commensal}$" + "\n" + r"$\bf{Static}$",
    "CH": r"$\bf{Commensal}$" + "\n" + r"$\bf{HOBIC}$",
    "DS": r"$\bf{Dysbiotic}$" + "\n" + r"$\bf{Static}$",
    "DH": r"$\bf{Dysbiotic}$" + "\n" + r"$\bf{HOBIC}$",
}
CK_MAP = {
    "CS": ("Commensal", "Static"),
    "CH": ("Commensal", "HOBIC"),
    "DS": ("Dysbiotic", "Static"),
    "DH": ("Dysbiotic", "HOBIC"),
}
P2_DIRS = {
    "CS": "jax_ode_nuts_Commensal_Static_20260320_043505",
    "CH": "jax_ode_nuts_Commensal_HOBIC_20260320_043812",
    "DS": "jax_ode_nuts_Dysbiotic_Static_20260320_044847",
    "DH": "jax_ode_nuts_Dysbiotic_HOBIC_20260320_052844",
}
REP_SP_MAP = {
    "S. oralis": 0,
    "A. naeslundii": 1,
    "V. dispar": 2,
    "V. parvula": 2,
    "F. nucleatum": 3,
    "P. gingivalis_20709": 4,
    "P. gingivalis_W83": 4,
}
DAYS_ALL = [1, 3, 6, 10, 15, 21]

# ============================================================
# JAX worker script (embedded, runs in subprocess)
# ============================================================
WORKER_SCRIPT = r"""
import sys, os, json
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
import jax; import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.expanduser("~/IKM_Hiwi/Tmcmc202601/data_5species/main"))
sys.path.insert(0, os.path.expanduser("~/IKM_Hiwi/Tmcmc202601/data_5species"))
from hamilton_ode_jax import simulate_0d_full, simulate_0d
from estimate_reduced_nishioka import load_experimental_data
from pathlib import Path

args = json.load(open(sys.argv[1]))
ck = args["ck"]; cond = args["cond"]; cult = args["cult"]
run_dir = Path(args["run_dir"]); out_path = args["out_path"]

ROOT = Path.home() / "IKM_Hiwi/Tmcmc202601"
data_dir = ROOT / "data_5species"
data, t_days, _, phi_init, _ = load_experimental_data(data_dir, cond, cult, start_from_day=3)
data_norm = data / np.maximum(data.sum(axis=1, keepdims=True), 1e-12)
ic = np.clip(phi_init, 1e-6, None); ic = ic / ic.sum()
ic_jax = jnp.array(ic, dtype=jnp.float64)

data1, _, _, _, _ = load_experimental_data(data_dir, cond, cult, start_from_day=1)
d1n = data1 / np.maximum(data1.sum(axis=1, keepdims=True), 1e-12)

mj = json.load(open(run_dir / "theta_MAP.json"))
theta = jnp.array([mj[str(i)] for i in range(20)], dtype=jnp.float64)
n_steps = 2500; dt = 1e-4

g = np.array(simulate_0d_full(theta, n_steps=n_steps, dt=dt, phi_init=ic_jax,
    K_hill=0.05, n_hill=4.0, c_const=25.0))
phi_map = g[:, 0:5]; phibar_map = g[:, 0:5] * g[:, 6:11]

samples = np.load(run_dir / "samples.npy")
n_sub = min(100, len(samples))
rng = np.random.default_rng(42)
idx_sub = rng.choice(len(samples), n_sub, replace=False)
trajs = np.zeros((n_sub, n_steps+1, 5))
for k, si in enumerate(idx_sub):
    try:
        trajs[k] = np.array(simulate_0d(
            jnp.array(samples[si], dtype=jnp.float64),
            n_steps=n_steps, dt=dt, phi_init=ic_jax,
            K_hill=0.05, n_hill=4.0, c_const=25.0))
    except:
        trajs[k] = phi_map

np.savez(out_path, phi_map=phi_map, phibar_map=phibar_map,
         trajs=trajs, data_norm=data_norm, d1n=d1n[0], t_days=t_days)
print("OK")
"""


# ============================================================
# Cache management
# ============================================================
def ensure_cache(recompute=False):
    """Ensure all 4 conditions have cached computation results."""
    worker_path = CACHE_DIR / "_worker.py"
    worker_path.write_text(WORKER_SCRIPT)

    for ck in CONDS:
        cache_file = CACHE_DIR / f"fig2_phi_{ck}.npz"
        if cache_file.exists() and not recompute:
            print(f"  {ck}: using cache")
            continue

        cond, cult = CK_MAP[ck]
        af = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(
            {
                "ck": ck,
                "cond": cond,
                "cult": cult,
                "run_dir": str(RUNS / P2_DIRS[ck]),
                "out_path": str(cache_file),
            },
            af,
        )
        af.close()

        print(f"  {ck}: computing via JAX subprocess...")
        r = subprocess.run(
            [sys.executable, str(worker_path), af.name],
            capture_output=True,
            text=True,
            timeout=600,
        )
        os.unlink(af.name)
        if r.returncode != 0:
            print(f"  {ck}: FAILED\n{r.stderr[-300:]}")
        else:
            print(f"  {ck}: cached")

    worker_path.unlink(missing_ok=True)


def load_cache(ck):
    return np.load(CACHE_DIR / f"fig2_phi_{ck}.npz")


# ============================================================
# Plot
# ============================================================
def generate_fig2_phi_transposed():
    """Transposed layout: 5 rows (species) × 4 cols (conditions). Publication quality."""
    # Larger fonts for publication
    pub_style = STYLE.copy()
    pub_style.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 8.5,
        }
    )
    plt.rcParams.update(pub_style)

    rep_df = pd.read_csv(REP_CSV)
    model_days = np.linspace(1, 21, 2501)  # Day 1 start (IC is Day 1)
    exp_ticks = [1, 3, 6, 10, 15, 21]

    fig, axes = plt.subplots(5, 4, figsize=(7.5, 9.5), sharex=True, sharey=True)

    # Collect RMSE/logL for subtitle row
    metrics = {}

    for col_idx, ck in enumerate(CONDS):
        d = load_cache(ck)
        phi_map = d["phi_map"]
        phibar_map = d["phibar_map"]
        trajs = d["trajs"]
        data_norm = d["data_norm"]
        d1n = d["d1n"]
        t_days = d["t_days"]
        cfg = json.load(open(RUNS / P2_DIRS[ck] / "config.json"))
        rmse = cfg.get("rmse", 0)
        logL = cfg.get("max_logL", 0)
        metrics[ck] = (rmse, logL)
        cond, cult = CK_MAP[ck]

        colors = SP_COLORS["C"] if ck.startswith("C") else SP_COLORS["D"]
        cond_rep = rep_df[(rep_df["condition"] == cond) & (rep_df["cultivation"] == cult)]

        q10 = np.percentile(trajs, 10, axis=0)
        q25 = np.percentile(trajs, 25, axis=0)
        q75 = np.percentile(trajs, 75, axis=0)
        q90 = np.percentile(trajs, 90, axis=0)

        all_days = np.concatenate([[1], t_days])
        all_data = np.vstack([d1n[np.newaxis, :], data_norm])

        for row_idx in range(5):
            ax = axes[row_idx, col_idx]

            # Boxplots from replicates
            sp_names = [k for k, v in REP_SP_MAP.items() if v == row_idx]
            bd, bp = [], []
            for day in DAYS_ALL:
                vals = (
                    cond_rep[(cond_rep["day"] == day) & (cond_rep["species"].isin(sp_names))][
                        "distribution_pct"
                    ].values
                    / 100.0
                )
                if len(vals) > 0:
                    bd.append(vals)
                    bp.append(day)
            if bd:
                bx = ax.boxplot(
                    bd,
                    positions=bp,
                    widths=1.8,
                    patch_artist=True,
                    showfliers=False,
                    manage_ticks=False,
                    zorder=1,
                    medianprops=dict(color="k", linewidth=0.8),
                    whiskerprops=dict(color="gray", linewidth=0.5),
                    capprops=dict(color="gray", linewidth=0.5),
                    boxprops=dict(linewidth=0.4),
                )
                for p in bx["boxes"]:
                    p.set_facecolor(colors[row_idx])
                    p.set_alpha(0.15)
                    p.set_edgecolor(colors[row_idx])

            # CI bands
            ax.fill_between(
                model_days,
                q10[:, row_idx],
                q90[:, row_idx],
                color=colors[row_idx],
                alpha=0.10,
                zorder=1,
            )
            ax.fill_between(
                model_days,
                q25[:, row_idx],
                q75[:, row_idx],
                color=colors[row_idx],
                alpha=0.25,
                zorder=1,
            )

            # MAP φ (solid)
            ax.plot(model_days, phi_map[:, row_idx], color=colors[row_idx], lw=1.5, zorder=4)
            # MAP φ̄ (dashed)
            ax.plot(
                model_days,
                phibar_map[:, row_idx],
                color=colors[row_idx],
                lw=1.0,
                ls="--",
                alpha=0.5,
                zorder=3,
            )

            # Data points
            ax.scatter(
                all_days,
                all_data[:, row_idx],
                s=20,
                color=colors[row_idx],
                edgecolors="k",
                linewidths=0.4,
                zorder=5,
            )

            ax.set_xlim(0, 22.5)
            ax.set_ylim(-0.02, 1.02)
            ax.set_xticks(exp_ticks)
            ax.grid(True)

            # Column title: condition name + RMSE only
            if row_idx == 0:
                title = COND_LABELS[ck]
                ax.set_title(
                    title + "\n" + r"$\mathrm{RMSE} = " + f"{rmse:.3f}" + r"$",
                    fontsize=11,
                    pad=6,
                )

            # Row label: species name on left + φ unit
            if col_idx == 0:
                sp_label = SPECIES_ITALIC[row_idx]
                # Pg row: show strain per condition (handled per-cell below)
                if row_idx < 4:
                    ax.set_ylabel(
                        sp_label + r"$\quad \varphi_i\;[-]$",
                        fontsize=10,
                    )
                else:
                    ax.set_ylabel(r"$\varphi_i\;[-]$", fontsize=10)

            # Pg row (row_idx=4): show strain name inside panel
            if row_idx == 4:
                strain = PG_STRAIN[ck]
                ax.text(
                    0.5,
                    0.92,
                    strain,
                    transform=ax.transAxes,
                    fontsize=7.5,
                    ha="center",
                    va="top",
                    fontstyle="italic",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8),
                )

            # X-axis label on bottom row
            if row_idx == 4:
                ax.set_xlabel("Day", fontsize=12)

    # Top legend (above figure, horizontal)
    _lc = "#2166AC"  # legend color (blue, more visible than gray)
    handles = [
        Line2D([0], [0], color=_lc, lw=1.5, label=r"MAP $\varphi_i$"),
        Line2D(
            [0],
            [0],
            color=_lc,
            lw=1.0,
            ls="--",
            alpha=0.6,
            label=r"MAP $\bar{\varphi}_i = \varphi_i\psi_i$",
        ),
        Patch(facecolor=_lc, alpha=0.25, label="50% CI"),
        Patch(facecolor=_lc, alpha=0.12, label="80% CI"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=_lc,
            markeredgecolor="k",
            markersize=5,
            label="Exp. median",
        ),
        Patch(facecolor=_lc, alpha=0.15, edgecolor=_lc, linewidth=0.5, label="Replicate IQR"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=6,
        fontsize=8.5,
        framealpha=0.95,
        handlelength=1.5,
        handletextpad=0.5,
        borderpad=0.4,
        columnspacing=1.0,
        bbox_to_anchor=(0.5, 1.0),
    )

    fig.tight_layout(h_pad=0.4, w_pad=0.3, rect=[0, 0, 1, 0.965])

    for ext in ["png", "pdf"]:
        for d in [FIG_DIR, FIG_DIR2]:
            out = d / f"paper_fig2_phi_transposed.{ext}"
            fig.savefig(out)
            print(f"Saved: {out}")
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recompute", action="store_true", help="Force recompute cache (default: use existing)"
    )
    args = parser.parse_args()

    print("Ensuring cache...")
    ensure_cache(recompute=args.recompute)

    print("Generating figure...")
    generate_fig2_phi_transposed()
    print("Done.")


if __name__ == "__main__":
    main()
