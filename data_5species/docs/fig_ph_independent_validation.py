"""
fig_ph_independent_validation.py
=================================
Publication-quality figure: independent pH validation.

Pipeline (end-to-end):
  1. Fit pH regression on experimental species fracs (Fig 3A medians)
       pH = β₀ + β_So·φ_So + β_An·φ_An + β_Fn·φ_Fn + β_Pg·φ_Pg
  2. Run 10000p JAX-NUTS posterior samples through Hamilton ODE
       → posterior φ_i(t) trajectories (with Day-1 exp. fracs as IC)
  3. Apply pH regression to predicted φ → posterior pH CI band
  4. Compare to experimental Fig 4A pH (completely independent of TMCMC)

Output: paper_comprehensive_figs/fig_ph_validation.{pdf,png}
"""

from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# thesis_style — lmodern 9pt
sys.path.insert(0, str(Path.home() / "IKM_Hiwi/nife"))
import thesis_style as _ts

# ── Paths ──────────────────────────────────────────────────────────────────
REPO_DIR = Path(__file__).resolve().parent.parent  # data_5species/
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = REPO_DIR / "experiment_data"
OUT_DIR = SCRIPT_DIR / "paper_comprehensive_figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLAB_DIR = REPO_DIR.parent / "colab_package"
NUTS_RUNS_DIR = REPO_DIR / "main" / "_runs"

CH_RUN = "jax_ode_nuts_Commensal_HOBIC_20260320_043812"
DH_RUN = "jax_ode_nuts_Dysbiotic_HOBIC_20260320_052844"

sys.path.insert(0, str(COLAB_DIR))
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

# ── Constants ──────────────────────────────────────────────────────────────
SPECIES = ["So", "An", "Vei", "Fn", "Pg"]
IDX_SPARSE = np.array([113, 339, 679, 1131, 1696, 2375])
EXP_DAYS = np.array([1.0, 3.0, 6.0, 10.0, 15.0, 21.0])

N_SAMPLES = 500
BATCH_SIZE = 100  # vmap chunk to avoid OOM
SEED = 42
CI_LO, CI_HI = 5.0, 95.0

SP_COL = {"So": "#1565C0", "An": "#388E3C", "Vei": "#F57C00", "Fn": "#7B1FA2", "Pg": "#C62828"}
SP_LAB = {
    "So": "S. oralis",
    "An": "A. naeslundii",
    "Vei": "V. parvula",
    "Fn": "F. nucleatum",
    "Pg": "P. gingivalis",
}
# Colors for conditions
COL_CH = "#1565C0"
COL_DH = "#C62828"

# ── Load experimental data ──────────────────────────────────────────────────


def load_ph_exp():
    df = pd.read_csv(DATA_DIR / "fig4A_pH_timeseries.csv")
    return df["time_days"].values, df["pH_commensal"].values, df["pH_dysbiotic"].values


def load_species_medians_hobic():
    """Return {sp: array(6,)} for CH and DH at EXP_DAYS."""
    df = pd.read_csv(DATA_DIR / "fig3_species_distribution_summary.csv")
    CMAP = {
        "S. oralis": "So",
        "A. naeslundii": "An",
        "V. dispar": "Vei",
        "V. parvula": "Vei",
        "F. nucleatum": "Fn",
        "P. gingivalis_20709": "Pg",
        "P. gingivalis_W83": "Pg",
    }

    def get(cond, sp):
        sub = df[(df.condition == cond) & (df.cultivation == "HOBIC")]
        sub = sub[sub.species.map(CMAP) == sp]
        if len(sub) == 0:
            return np.zeros(len(EXP_DAYS))
        return np.interp(
            EXP_DAYS,
            sub["day"].values,
            sub["median"].values / 100.0,
            left=sub["median"].values[0] / 100.0,
            right=sub["median"].values[-1] / 100.0,
        )

    ch = {sp: get("Commensal", sp) for sp in SPECIES}
    dh = {sp: get("Dysbiotic", sp) for sp in SPECIES}
    return ch, dh


def phi_init_from_day1(cond):
    """Day-1 experimental fracs normalized to sum=1 (used as ODE IC)."""
    ch, dh = load_species_medians_hobic()
    fracs = ch if cond == "CH" else dh
    v = np.array([fracs[sp][0] for sp in SPECIES])
    v = np.clip(v, 1e-4, 1.0)
    return v / v.sum()


# ── Fit pH regression ───────────────────────────────────────────────────────


def _loo_fit(X, y):
    """LOO-CV. Returns (beta_full, r2_loo, rmse_loo, preds_loo)."""
    n = len(y)
    preds = np.zeros(n)
    for i in range(n):
        idx = np.arange(n) != i
        b, *_ = np.linalg.lstsq(X[idx], y[idx], rcond=None)
        preds[i] = X[i] @ b
    r2 = pearsonr(y, preds)[0] ** 2
    rmse = np.sqrt(np.mean((preds - y) ** 2))
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta, r2, rmse, preds


def fit_ph_regression():
    """
    Fit 4-species (So+An+Fn+Pg) AND 5-species (So+An+Vei+Fn+Pg) models.
    Returns dict with keys '4sp' and '5sp', each containing
    (beta, r2_loo, rmse_loo).
    """
    t_ph, ph_com, ph_dys = load_ph_exp()
    mask = t_ph >= 1.0
    t_vec = t_ph[mask]
    ph_ch = ph_com[mask]
    ph_dh = ph_dys[mask]

    ch_raw, dh_raw = load_species_medians_hobic()

    def interp_cont(sp_dict, t):
        return {
            sp: np.interp(t, EXP_DAYS, sp_dict[sp], left=sp_dict[sp][0], right=sp_dict[sp][-1])
            for sp in SPECIES
        }

    ch_c = interp_cont(ch_raw, t_vec)
    dh_c = interp_cont(dh_raw, t_vec)
    y = np.concatenate([ph_ch, ph_dh])

    results = {}
    for name, cols in [
        ("4sp", ["So", "An", "Fn", "Pg"]),
        ("5sp", ["So", "An", "Vei", "Fn", "Pg"]),
    ]:
        X = np.vstack(
            [
                np.column_stack([np.ones(len(t_vec))] + [ch_c[s] for s in cols]),
                np.column_stack([np.ones(len(t_vec))] + [dh_c[s] for s in cols]),
            ]
        )
        beta, r2, rmse, _ = _loo_fit(X, y)
        results[name] = (beta, r2, rmse)
        bstr = "  ".join(f"β_{s}={v:+.3f}" for s, v in zip(["0"] + cols, beta))
        print(f"  [{name}] LOO R²={r2:.4f}, RMSE={rmse:.4f}  |  {bstr}")
    return results


def ph_from_phi(phi_mat, beta, use_vei=False):
    """
    phi_mat: (..., 5) — species order: So An Vei Fn Pg
    use_vei=False → 4-species model (So+An+Fn+Pg), beta has 5 elements
    use_vei=True  → 5-species model (So+An+Vei+Fn+Pg), beta has 6 elements
    """
    if use_vei:
        return (
            beta[0]
            + beta[1] * phi_mat[..., 0]  # So
            + beta[2] * phi_mat[..., 1]  # An
            + beta[3] * phi_mat[..., 2]  # Vei
            + beta[4] * phi_mat[..., 3]  # Fn
            + beta[5] * phi_mat[..., 4]
        )  # Pg
    return (
        beta[0]
        + beta[1] * phi_mat[..., 0]  # So
        + beta[2] * phi_mat[..., 1]  # An
        + beta[3] * phi_mat[..., 3]  # Fn
        + beta[4] * phi_mat[..., 4]
    )  # Pg


# ── Run JAX ODE posterior ensemble ─────────────────────────────────────────


def run_posterior_jax(run_name, phi_ic, n_samples=N_SAMPLES):
    """
    Returns phi_post: (n_valid, 6, 5) — species fracs at EXP_DAYS.
    Uses jax.vmap over theta batch for speed (~18s per 100 samples).
    """
    import jax, time
    import jax.numpy as jnp
    from functools import partial
    from hamilton_ode_jax import simulate_0d_full  # colab 20-param

    samples_all = np.load(NUTS_RUNS_DIR / run_name / "samples.npy")
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(samples_all), size=n_samples, replace=False)
    samples = samples_all[idx]

    phi_init_jax = jnp.array(phi_ic, dtype=jnp.float64)

    def sim_single(theta):
        return simulate_0d_full(
            theta,
            n_steps=2500,
            dt=1e-4,
            phi_init=phi_init_jax,
            K_hill=0.05,
            n_hill=4.0,
            c_const=25.0,
        )

    sim_batch_jit = jax.jit(jax.vmap(sim_single))

    phi_post = np.zeros((n_samples, 6, 5))
    n_chunks = int(np.ceil(n_samples / BATCH_SIZE))
    t0 = time.time()

    for chunk in range(n_chunks):
        sl = slice(chunk * BATCH_SIZE, min((chunk + 1) * BATCH_SIZE, n_samples))
        batch = jnp.array(samples[sl], dtype=jnp.float64)
        g_batch = np.array(sim_batch_jit(batch))  # (B, 2501, 12)
        for di, t_idx in enumerate(IDX_SPARSE):
            phi_post[sl, di, :] = np.clip(g_batch[:, t_idx, :5], 0.0, None)
        print(
            f"    {run_name}: {sl.stop}/{n_samples}  ({time.time()-t0:.0f}s)", end="\r", flush=True
        )

    print(f"\n    Done ({run_name}). Total: {time.time()-t0:.1f}s")
    # Normalize each day's fractions to sum=1
    s = phi_post.sum(axis=2, keepdims=True)
    phi_post /= np.where(s > 1e-12, s, 1.0)
    valid = ~np.any(np.isnan(phi_post.reshape(n_samples, -1)), axis=1)
    return phi_post[valid]


# ── Figure helpers ──────────────────────────────────────────────────────────


def ci(arr_nd, axis=0):
    lo = np.nanpercentile(arr_nd, CI_LO, axis=axis)
    med = np.nanmedian(arr_nd, axis=axis)
    hi = np.nanpercentile(arr_nd, CI_HI, axis=axis)
    return lo, med, hi


def panel_species(ax, phi_post, sp_med_exp, title):
    sp_idx = {sp: i for i, sp in enumerate(SPECIES)}
    for sp in SPECIES:
        c = SP_COL[sp]
        idx = sp_idx[sp]
        lo, med, hi = ci(phi_post[:, :, idx])
        ax.fill_between(EXP_DAYS, lo * 100, hi * 100, color=c, alpha=0.18)
        ax.plot(EXP_DAYS, med * 100, "-", color=c, lw=1.6, label=SP_LAB[sp])
        if sp_med_exp[sp].max() > 0.5:
            ax.scatter(EXP_DAYS, sp_med_exp[sp] * 100, color=c, s=22, zorder=5)
    ax.set(
        xlabel="Day", ylabel="Species fraction (%)", title=title, xlim=[-0.5, 22], ylim=[-2, 108]
    )
    ax.legend(fontsize=6.5, loc="upper right", ncol=1, framealpha=0.85)


def panel_ph_series(ax, phi_post, ph_exp_full_t, ph_exp_full, color, title):
    ph_mat = ph_from_phi(phi_post.reshape(-1, 6, 5), beta_glob)  # (n, 6)
    lo, med, hi = ci(ph_mat)
    ax.plot(ph_exp_full_t, ph_exp_full, "k-", lw=2.2, alpha=0.85, label="Experiment (Fig 4A)")
    ax.fill_between(EXP_DAYS, lo, hi, color=color, alpha=0.22, label=f"TMCMC pH pred. (90% CI)")
    ax.plot(EXP_DAYS, med, "s--", color=color, lw=1.6, ms=5, label="TMCMC median")
    ax.set(xlabel="Day", ylabel="pH", title=title, xlim=[-0.5, 22], ylim=[5.8, 7.8])
    ax.legend(fontsize=7.5)


beta_glob = None  # 4-species beta, set in main
beta5_glob = None  # 5-species beta, set in main


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    global beta_glob, beta5_glob

    print("=" * 62)
    print("Independent pH Validation — Publication Figure")
    print("=" * 62)

    # 1. Fit regression (4-sp and 5-sp)
    print("\n[1] Fitting pH regression (4-sp: So+An+Fn+Pg  /  5-sp: +Vei)...")
    reg = fit_ph_regression()
    beta, r2_loo, rmse_loo = reg["4sp"]
    beta5, r2_loo5, rmse_loo5 = reg["5sp"]
    beta_glob = beta
    beta5_glob = beta5

    # 2. Load experimental data
    t_ph, ph_com_full, ph_dys_full = load_ph_exp()
    ch_exp, dh_exp = load_species_medians_hobic()
    ph_exp_ch = np.interp(EXP_DAYS, t_ph, ph_com_full)
    ph_exp_dh = np.interp(EXP_DAYS, t_ph, ph_dys_full)

    # 3. Run Hamilton ODE posterior (JAX, 10000p NUTS runs)
    print(f"\n[2] Running JAX Hamilton ODE ({N_SAMPLES} samples × 2 conditions)...")
    phi_ic_ch = phi_init_from_day1("CH")
    phi_ic_dh = phi_init_from_day1("DH")
    print(f"  CH IC: {dict(zip(SPECIES, phi_ic_ch.round(3)))}")
    print(f"  DH IC: {dict(zip(SPECIES, phi_ic_dh.round(3)))}")

    print("  CH:")
    phi_ch = run_posterior_jax(CH_RUN, phi_ic_ch, N_SAMPLES)
    print("  DH:")
    phi_dh = run_posterior_jax(DH_RUN, phi_ic_dh, N_SAMPLES)
    print(f"\n  Valid: CH={len(phi_ch)}, DH={len(phi_dh)}")

    # Median species fracs for reporting
    sp_idx_map = {sp: i for i, sp in enumerate(SPECIES)}
    print("\n  TMCMC median φ_Fn, φ_Pg (DH, should rise over time):")
    for sp in ["Fn", "Pg"]:
        print(f"    {sp}: {np.nanmedian(phi_dh[:,:,sp_idx_map[sp]], axis=0).round(3)}")

    # 4. Validation metrics at EXP_DAYS (4-sp and 5-sp)
    def val_metrics(phi_c, phi_d, b, vei=False):
        pm_c = ph_from_phi(phi_c, b, use_vei=vei)  # (n, 6)
        pm_d = ph_from_phi(phi_d, b, use_vei=vei)
        mc = np.nanmedian(pm_c, axis=0)
        md = np.nanmedian(pm_d, axis=0)
        ae = np.concatenate([ph_exp_ch, ph_exp_dh])
        ap = np.concatenate([mc, md])
        r2 = pearsonr(ae, ap)[0] ** 2
        rmse = np.sqrt(np.mean((ap - ae) ** 2))
        return pm_c, pm_d, mc, md, r2, rmse

    ph_pred_ch, ph_pred_dh, ph_med_ch, ph_med_dh, r2_val, rmse_val = val_metrics(
        phi_ch, phi_dh, beta, vei=False
    )
    ph_pred_ch5, ph_pred_dh5, ph_med_ch5, ph_med_dh5, r2_val5, rmse_val5 = val_metrics(
        phi_ch, phi_dh, beta5, vei=True
    )
    rmse_d = np.sqrt(np.mean(((ph_med_dh - ph_med_ch) - (ph_exp_dh - ph_exp_ch)) ** 2))
    rmse_d5 = np.sqrt(np.mean(((ph_med_dh5 - ph_med_ch5) - (ph_exp_dh - ph_exp_ch)) ** 2))

    print(f"\n{'='*62}")
    print(
        f"  4-sp LOO R²={r2_loo:.4f} RMSE={rmse_loo:.4f} | "
        f"indep R²={r2_val:.4f}  RMSE={rmse_val:.4f}"
    )
    print(
        f"  5-sp LOO R²={r2_loo5:.4f} RMSE={rmse_loo5:.4f} | "
        f"indep R²={r2_val5:.4f}  RMSE={rmse_val5:.4f}"
    )
    print(f"{'='*62}")

    # 5. Figure — publication quality (5-sp only, 2 panels, Times-style serif)
    print("\n[3] Generating publication figure...")

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    _ts.use(width_frac=1.0, aspect=0.50)  # thesis_style: lmodern 9pt

    # ── 5-sp posteriors ────────────────────────────────────────────────────
    ph5_ch = ph_from_phi(phi_ch.reshape(-1, 6, 5), beta5, use_vei=True)
    ph5_dh = ph_from_phi(phi_dh.reshape(-1, 6, 5), beta5, use_vei=True)
    lo_ch, med_ch, hi_ch = ci(ph5_ch)
    lo_dh, med_dh, hi_dh = ci(ph5_dh)

    def despine(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    def panel_label(ax, lbl):
        ax.text(
            -0.13,
            1.05,
            lbl,
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            va="top",
            ha="left",
            fontfamily="serif",
        )

    # ── Layout ────────────────────────────────────────────────────────────
    figsize = _ts.use(width_frac=1.0, aspect=0.50)
    fig = plt.figure(figsize=figsize)
    gs_main = gridspec.GridSpec(
        1,
        2,
        width_ratios=[1.25, 0.85],
        wspace=0.38,
        left=0.09,
        right=0.97,
        top=0.93,
        bottom=0.13,
    )

    # In-figure title + method note intentionally removed for publication:
    # the LaTeX \caption is the figure title and the method/equation belongs
    # in the caption/text. (This also drops the stale "NUTS" label — the run
    # is TMCMC.) Panel headings (set_title) are kept as (a)/(b) descriptors.

    # ═══════════════════════════════════════════════════════════════════════
    # Panel (a) — pH time series
    # ═══════════════════════════════════════════════════════════════════════
    ax_a = fig.add_subplot(gs_main[0, 0])

    ax_a.plot(t_ph, ph_com_full, "-", color=COL_CH, lw=2.0, alpha=0.85)
    ax_a.plot(t_ph, ph_dys_full, "-", color=COL_DH, lw=2.0, alpha=0.85)

    ax_a.fill_between(EXP_DAYS, lo_ch, hi_ch, color=COL_CH, alpha=0.20, lw=0)
    ax_a.fill_between(EXP_DAYS, lo_dh, hi_dh, color=COL_DH, alpha=0.20, lw=0)

    ax_a.plot(EXP_DAYS, med_ch, "o--", color=COL_CH, lw=1.5, ms=5.5, mec="white", mew=0.5)
    ax_a.plot(EXP_DAYS, med_dh, "s--", color=COL_DH, lw=1.5, ms=5.5, mec="white", mew=0.5)

    ax_a.set_xlabel("Day")
    ax_a.set_ylabel("pH")
    ax_a.set_xlim(-0.5, 22)
    ax_a.set_ylim(6.0, 7.75)
    ax_a.set_title("Commensal vs. Dysbiotic HOBIC", pad=5)

    leg_a = [
        Line2D([0], [0], color=COL_CH, lw=2.0, label="Commensal — measured"),
        Line2D([0], [0], color=COL_DH, lw=2.0, label="Dysbiotic — measured"),
        Line2D(
            [0],
            [0],
            color=COL_CH,
            lw=1.5,
            ls="--",
            marker="o",
            ms=5,
            mec="white",
            label=r"Commensal — predicted (90\% CI)",
        ),
        Line2D(
            [0],
            [0],
            color=COL_DH,
            lw=1.5,
            ls="--",
            marker="s",
            ms=5,
            mec="white",
            label=r"Dysbiotic — predicted (90\% CI)",
        ),
        Patch(fc=COL_CH, alpha=0.30, ec="none", label=""),
        Patch(fc=COL_DH, alpha=0.30, ec="none", label=""),
    ]
    # Compact 2-column legend: measured | predicted
    ax_a.legend(
        handles=leg_a[:4],
        loc="upper left",
        framealpha=0.92,
        ncol=1,
        handlelength=2.2,
        labelspacing=0.45,
        borderpad=0.7,
        handletextpad=0.6,
        edgecolor="#CCCCCC",
    )
    despine(ax_a)
    panel_label(ax_a, "a")

    # ═══════════════════════════════════════════════════════════════════════
    # Panel (b) — Scatter
    # ═══════════════════════════════════════════════════════════════════════
    ax_b = fig.add_subplot(gs_main[0, 1])

    for di in range(6):
        ax_b.plot(
            [ph_exp_ch[di]] * 2,
            [lo_ch[di], hi_ch[di]],
            "-",
            color=COL_CH,
            lw=1.2,
            alpha=0.50,
            solid_capstyle="round",
        )
        ax_b.plot(
            [ph_exp_dh[di]] * 2,
            [lo_dh[di], hi_dh[di]],
            "-",
            color=COL_DH,
            lw=1.2,
            alpha=0.50,
            solid_capstyle="round",
        )

    ax_b.scatter(
        ph_exp_ch,
        med_ch,
        color=COL_CH,
        s=65,
        zorder=5,
        marker="o",
        edgecolors="white",
        linewidths=0.5,
        label="Commensal HOBIC",
    )
    ax_b.scatter(
        ph_exp_dh,
        med_dh,
        color=COL_DH,
        s=65,
        zorder=5,
        marker="s",
        edgecolors="white",
        linewidths=0.5,
        label="Dysbiotic HOBIC",
    )

    lim = [6.05, 7.55]
    ax_b.plot(lim, lim, "--", color="#9CA3AF", lw=0.9, zorder=0)
    ax_b.set_xlabel("Measured pH")
    ax_b.set_ylabel("Predicted pH (TMCMC median)")
    ax_b.set_xlim(lim)
    ax_b.set_ylim(lim)
    ax_b.set_aspect("equal")
    ax_b.set_title("Predicted vs. Measured", pad=5)
    ax_b.legend(
        loc="upper left", framealpha=0.92, handletextpad=0.5, borderpad=0.7, edgecolor="#CCCCCC"
    )

    # Metrics annotation — inside plot, bottom-right
    ax_b.text(
        0.97,
        0.06,
        f"$R^2 = {r2_val5:.3f}$\n" f"RMSE $= {rmse_val5:.3f}$",
        transform=ax_b.transAxes,
        fontsize=12,
        va="bottom",
        ha="right",
        color="#111827",
        bbox=dict(boxstyle="round,pad=0.40", fc="white", ec="#CCCCCC", lw=0.8),
    )
    despine(ax_b)
    panel_label(ax_b, "b")

    for fmt in ("pdf", "png"):
        out = OUT_DIR / f"fig_ph_validation.{fmt}"
        fig.savefig(out, dpi=600, bbox_inches="tight")
        print(f"  Saved: {out}")
    plt.close(fig)
    plt.rcdefaults()


if __name__ == "__main__":
    os.chdir(REPO_DIR.parent)
    main()
