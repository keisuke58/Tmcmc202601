"""
compare_ph_methods.py
=====================
3つの pH 予測手法を Heine 2025 Fig 4A データに対して定量評価・比較する。

Method A (Simple):  pH = β₀ + β₁·φ_So + β₂·φ_Vei  (OLS, 実験 species fractions)
Method B (dFBA):    Monod kinetics ODE + 経験的 pH = pH₀ - k·A_net (文献 stoichiometry)
Method C (FBA):     Method B の stoichiometry を AGORA pFBA で置換

評価指標:
  - RMSE at HOBIC experimental timepoints (days 1,3,6,10,15,21)
  - R² (12点 CH+DH 合算)
  - ΔpH(DH-CH) の時系列再現: 「なぜ DH の pH が高いか」を捉えられるか
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "experiment_data"
OUT_DIR = SCRIPT_DIR / "paper_comprehensive_figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXP_DAYS = np.array([1.0, 3.0, 6.0, 10.0, 15.0, 21.0])

# ---------------------------------------------------------------------------
# Load experimental data
# ---------------------------------------------------------------------------


def load_ph_exp():
    """pH time series from Fig 4A."""
    df = pd.read_csv(DATA_DIR / "fig4A_pH_timeseries.csv")
    return df["time_days"].values, df["pH_commensal"].values, df["pH_dysbiotic"].values


def load_species_fracs():
    """Median species fractions from Fig 3A (HOBIC only)."""
    df = pd.read_csv(DATA_DIR / "fig3_species_distribution_summary.csv")
    cmap = {
        "S. oralis": "So",
        "A. naeslundii": "An",
        "V. dispar": "Vd",
        "V. parvula": "Vd",
        "F. nucleatum": "Fn",
        "P. gingivalis_20709": "Pg",
        "P. gingivalis_W83": "Pg",
    }
    result = {}
    for cond in ["Commensal", "Dysbiotic"]:
        sub = df[(df["condition"] == cond) & (df["cultivation"] == "HOBIC")]
        sp_data: dict[str, list] = {}
        for _, row in sub.iterrows():
            sp = cmap.get(row["species"], row["species"])
            sp_data.setdefault(sp, [[], []])
            sp_data[sp][0].append(float(row["day"]))
            sp_data[sp][1].append(float(row["median"]) / 100.0)
        result[cond] = {k: (np.array(v[0]), np.array(v[1])) for k, v in sp_data.items()}
    return result


def interp_at_days(t_arr, ph_arr, days):
    return np.interp(days, t_arr, ph_arr)


# ---------------------------------------------------------------------------
# Method A: Simple linear regression  pH = β₀ + β₁·φ_So + β₂·φ_Vd
# ---------------------------------------------------------------------------


def method_a(sp_fracs):
    """Fit OLS on HOBIC data (CH+DH, 6 timepoints each = 12 points)."""
    t_ph, ph_com, ph_dys = load_ph_exp()

    ph_exp_ch = interp_at_days(t_ph, ph_com, EXP_DAYS)
    ph_exp_dh = interp_at_days(t_ph, ph_dys, EXP_DAYS)

    def get_frac(cond, sp, days):
        t_arr, f_arr = sp_fracs[cond].get(sp, (np.array([0.0, 21.0]), np.array([0.0, 0.0])))
        return np.interp(days, t_arr, f_arr)

    so_ch = get_frac("Commensal", "So", EXP_DAYS)
    vd_ch = get_frac("Commensal", "Vd", EXP_DAYS)
    so_dh = get_frac("Dysbiotic", "So", EXP_DAYS)
    vd_dh = get_frac("Dysbiotic", "Vd", EXP_DAYS)

    so_all = np.concatenate([so_ch, so_dh])
    vd_all = np.concatenate([vd_ch, vd_dh])
    ph_all = np.concatenate([ph_exp_ch, ph_exp_dh])

    X = np.column_stack([np.ones(len(ph_all)), so_all, vd_all])
    beta, _, _, _ = np.linalg.lstsq(X, ph_all, rcond=None)

    ph_pred_ch = beta[0] + beta[1] * so_ch + beta[2] * vd_ch
    ph_pred_dh = beta[0] + beta[1] * so_dh + beta[2] * vd_dh

    # continuous prediction on daily grid
    t_cont = np.linspace(0, 21, 300)
    so_ch_c = np.interp(
        t_cont,
        sp_fracs["Commensal"].get("So", (np.array([0.0, 21.0]), np.array([0.0, 0.0])))[0],
        sp_fracs["Commensal"].get("So", (np.array([0.0, 21.0]), np.array([0.0, 0.0])))[1],
    )
    vd_ch_c = np.interp(
        t_cont,
        sp_fracs["Commensal"].get("Vd", (np.array([0.0, 21.0]), np.array([0.0, 0.0])))[0],
        sp_fracs["Commensal"].get("Vd", (np.array([0.0, 21.0]), np.array([0.0, 0.0])))[1],
    )
    so_dh_c = np.interp(
        t_cont,
        sp_fracs["Dysbiotic"].get("So", (np.array([0.0, 21.0]), np.array([0.0, 0.0])))[0],
        sp_fracs["Dysbiotic"].get("So", (np.array([0.0, 21.0]), np.array([0.0, 0.0])))[1],
    )
    vd_dh_c = np.interp(
        t_cont,
        sp_fracs["Dysbiotic"].get("Vd", (np.array([0.0, 21.0]), np.array([0.0, 0.0])))[0],
        sp_fracs["Dysbiotic"].get("Vd", (np.array([0.0, 21.0]), np.array([0.0, 0.0])))[1],
    )

    ph_cont_ch = beta[0] + beta[1] * so_ch_c + beta[2] * vd_ch_c
    ph_cont_dh = beta[0] + beta[1] * so_dh_c + beta[2] * vd_dh_c

    return dict(
        beta=beta,
        ph_pred_ch=ph_pred_ch,  # at EXP_DAYS
        ph_pred_dh=ph_pred_dh,
        t_cont=t_cont,
        ph_cont_ch=ph_cont_ch,  # continuous
        ph_cont_dh=ph_cont_dh,
        ph_exp_ch=ph_exp_ch,
        ph_exp_dh=ph_exp_dh,
    )


# ---------------------------------------------------------------------------
# Method B/C core: Monod dFBA
# ---------------------------------------------------------------------------
SPECIES = ["So", "An", "Vp", "Fn", "Pg"]
METS = ["glc", "lac", "suc", "pro", "ace", "but", "for"]
N_MET = len(METS)
MET_IDX = {m: i for i, m in enumerate(METS)}

ACID_WEIGHTS = {"lac": 1.00, "ace": 0.80, "pro": 0.70, "but": 0.60, "for": 1.00, "suc": 0.50}
PH_INITIAL = 7.5
X_SCALE = 1.5e-3
D_HOBIC = 0.004

MEDIA_FULL = {"glc": 5.5, "lac": 0.0, "suc": 0.05}
MEDIA_DILU = {"glc": 2.8, "lac": 0.0, "suc": 0.03}

MONOD_LIT: dict[str, dict] = {
    "So": {
        "uptake": {"glc": (8.0, 0.05, 0.10)},
        "secretion_per_glc": {"lac": 1.8, "ace": 0.3, "for": 0.3, "suc": 0.1},
    },
    "An": {
        "uptake": {"glc": (6.0, 0.08, 0.08)},
        "secretion_per_glc": {"lac": 1.2, "suc": 0.3, "ace": 0.2, "for": 0.2},
    },
    "Vp": {
        "uptake": {"lac": (2.2, 0.15, 0.07), "suc": (1.0, 0.10, 0.06)},
        "secretion_per_lac": {"pro": 0.33, "ace": 0.33},
        "secretion_per_suc": {"pro": 0.50, "ace": 0.20},
    },
    "Fn": {
        "uptake": {"glc": (4.0, 0.12, 0.09), "lac": (3.0, 0.18, 0.07)},
        "secretion_per_glc": {"but": 0.6, "pro": 0.2, "ace": 0.2},
        "secretion_per_lac": {"but": 0.4, "ace": 0.2},
    },
    "Pg": {"uptake": {"suc": (3.0, 0.08, 0.10)}, "secretion_per_suc": {"but": 0.5, "ace": 0.3}},
}

MONOD_FBA: dict[str, dict] = {
    # So: FBA → 0 per glc (grows on amino acids in AGORA); keep literature
    "So": MONOD_LIT["So"],
    # An: FBA shows mixed-acid (no lac, ace+for+suc) — use FBA stoichiometry
    "An": {
        "uptake": {"glc": (6.0, 0.08, 0.08)},
        "secretion_per_glc": {"suc": 0.35, "ace": 0.39, "for": 0.53},
    },
    # Vp: keep literature (AGORA amino-acid inflation)
    "Vp": MONOD_LIT["Vp"],
    # Fn: FBA (capped): pro=2.77, ace+for=3.0 each
    "Fn": {
        "uptake": {"glc": (4.0, 0.12, 0.09), "lac": (3.0, 0.18, 0.07)},
        "secretion_per_glc": {"pro": 2.0, "ace": 1.5, "for": 1.5},
        "secretion_per_lac": {"pro": 1.0, "ace": 0.5},
    },
    # Pg: FBA (capped): but+ace+pro+for all large
    "Pg": {
        "uptake": {"suc": (3.0, 0.08, 0.10)},
        "secretion_per_suc": {"but": 1.5, "ace": 1.5, "pro": 0.8, "for": 0.8},
    },
}

INIT_FRACS = {
    "Commensal": {"So": 0.698, "An": 0.057, "Vp": 0.099, "Fn": 0.093, "Pg": 0.005},
    "Dysbiotic": {"So": 0.031, "An": 0.013, "Vp": 0.949, "Fn": 0.005, "Pg": 0.007},
}


def build_interps_hobic(cond, sp_fracs_raw):
    f0 = INIT_FRACS[cond]
    interps = {}
    for sp in SPECIES:
        sp_key = "Vd" if sp == "Vp" else sp
        raw = sp_fracs_raw[cond].get(sp_key, None)
        if raw is not None:
            t_raw, f_raw = raw
            pairs = sorted(zip(t_raw, f_raw))
            days = [0.0] + [p[0] for p in pairs]
            frac = [f0.get(sp, 0.0)] + [p[1] for p in pairs]
            interps[sp] = interp1d(
                days, frac, kind="linear", bounds_error=False, fill_value=(frac[0], frac[-1])
            )
        else:
            _v = f0.get(sp, 0.0)
            interps[sp] = lambda t, v=_v: np.full(np.shape(t) or (1,), v)
    return interps


def make_ode(interps, monod):
    def ode(t_h, C):
        C = np.maximum(C, 0.0)
        t_d = t_h / 24.0
        phi = np.array([float(interps[sp](t_d)) for sp in SPECIES])
        phi = np.maximum(phi, 0.0)
        s = phi.sum()
        if s > 1e-15:
            phi /= s
        X = phi * X_SCALE
        Cd = {m: C[MET_IDX[m]] for m in METS}
        dC = np.zeros(N_MET)
        feed = MEDIA_FULL if t_h < 24.0 else MEDIA_DILU
        for i, sp in enumerate(SPECIES):
            p = monod[sp]
            bm = X[i]
            q = {}
            for sub, (qmax, Km, _) in p["uptake"].items():
                q[sub] = qmax * Cd[sub] / (Km + Cd[sub] + 1e-15)
            for sub, qv in q.items():
                if sub in MET_IDX:
                    dC[MET_IDX[sub]] -= qv * bm
                for prod, stoich in p.get(f"secretion_per_{sub}", {}).items():
                    if prod in MET_IDX:
                        dC[MET_IDX[prod]] += stoich * qv * bm
        for m in METS:
            dC[MET_IDX[m]] += D_HOBIC * (feed.get(m, 0.0) - C[MET_IDX[m]])
        return dC

    return ode


def acid_load(C_time):
    A = np.zeros(len(C_time))
    for m, w in ACID_WEIGHTS.items():
        if m in MET_IDX:
            A += w * C_time[:, MET_IDX[m]]
    return A


def run_dfba(cond, monod, sp_fracs_raw, k_pH, t_days=21.0):
    interps = build_interps_hobic(cond, sp_fracs_raw)
    ode = make_ode(interps, monod)
    C0 = np.array([MEDIA_FULL.get(m, 0.0) for m in METS])
    t_eval = np.linspace(0.0, t_days * 24.0, 600)
    sol = solve_ivp(
        ode, [0.0, t_days * 24.0], C0, t_eval=t_eval, method="RK45", rtol=1e-6, atol=1e-9
    )
    t_d = sol.t / 24.0
    C = np.maximum(sol.y.T, 0.0)
    pH = np.clip(PH_INITIAL - k_pH * acid_load(C), 3.0, 9.0)
    return t_d, pH


def calibrate_k(monod, sp_fracs_raw):
    """k_pH such that CH HOBIC plateau pH (days 10-21) ≈ 6.30."""
    t_d, C_ch = _run_raw("Commensal", monod, sp_fracs_raw)
    mask = t_d >= 10.0
    A_ss = acid_load(C_ch)[mask].mean()
    return (PH_INITIAL - 6.30) / A_ss


def _run_raw(cond, monod, sp_fracs_raw):
    interps = build_interps_hobic(cond, sp_fracs_raw)
    ode = make_ode(interps, monod)
    C0 = np.array([MEDIA_FULL.get(m, 0.0) for m in METS])
    t_eval = np.linspace(0.0, 21.0 * 24.0, 600)
    sol = solve_ivp(ode, [0.0, 21.0 * 24.0], C0, t_eval=t_eval, method="RK45", rtol=1e-6, atol=1e-9)
    return sol.t / 24.0, np.maximum(sol.y.T, 0.0)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(ph_pred_ch, ph_pred_dh, ph_exp_ch, ph_exp_dh):
    """RMSE, R², ΔpH accuracy."""
    pred = np.concatenate([ph_pred_ch, ph_pred_dh])
    obs = np.concatenate([ph_exp_ch, ph_exp_dh])
    rmse = np.sqrt(np.mean((pred - obs) ** 2))
    r, _ = pearsonr(obs, pred)
    r2 = r**2

    delta_pred = ph_pred_dh - ph_pred_ch
    delta_obs = ph_exp_dh - ph_exp_ch
    rmse_delta = np.sqrt(np.mean((delta_pred - delta_obs) ** 2))

    return dict(
        rmse=rmse,
        r2=r2,
        rmse_delta=rmse_delta,
        rmse_ch=np.sqrt(np.mean((ph_pred_ch - ph_exp_ch) ** 2)),
        rmse_dh=np.sqrt(np.mean((ph_pred_dh - ph_exp_dh) ** 2)),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("Loading experimental data...")
    t_ph, ph_com, ph_dys = load_ph_exp()
    sp_fracs = load_species_fracs()

    ph_exp_ch = interp_at_days(t_ph, ph_com, EXP_DAYS)
    ph_exp_dh = interp_at_days(t_ph, ph_dys, EXP_DAYS)

    # ------------------------------------------------------------------
    # Method A: Simple regression
    # ------------------------------------------------------------------
    print("\n[A] Simple regression (So + Vei)...")
    res_a = method_a(sp_fracs)
    met_a = compute_metrics(res_a["ph_pred_ch"], res_a["ph_pred_dh"], ph_exp_ch, ph_exp_dh)
    print(f"   β = {res_a['beta']}")
    print(f"   RMSE={met_a['rmse']:.4f}, R²={met_a['r2']:.4f}, RMSE_ΔpH={met_a['rmse_delta']:.4f}")

    # ------------------------------------------------------------------
    # Method B: dFBA-Monod (literature stoichiometry)
    # ------------------------------------------------------------------
    print("\n[B] dFBA Monod (literature stoichiometry)...")
    k_B = calibrate_k(MONOD_LIT, sp_fracs)
    print(f"   k_pH = {k_B:.4f}")
    t_ch_B, pH_ch_B = run_dfba("Commensal", MONOD_LIT, sp_fracs, k_B)
    t_dh_B, pH_dh_B = run_dfba("Dysbiotic", MONOD_LIT, sp_fracs, k_B)
    ph_pred_ch_B = np.interp(EXP_DAYS, t_ch_B, pH_ch_B)
    ph_pred_dh_B = np.interp(EXP_DAYS, t_dh_B, pH_dh_B)
    met_b = compute_metrics(ph_pred_ch_B, ph_pred_dh_B, ph_exp_ch, ph_exp_dh)
    print(f"   RMSE={met_b['rmse']:.4f}, R²={met_b['r2']:.4f}, RMSE_ΔpH={met_b['rmse_delta']:.4f}")

    # ------------------------------------------------------------------
    # Method C: dFBA-Monod (AGORA FBA-informed stoichiometry)
    # ------------------------------------------------------------------
    print("\n[C] dFBA Monod (AGORA FBA-informed)...")
    k_C = calibrate_k(MONOD_FBA, sp_fracs)
    print(f"   k_pH = {k_C:.4f}")
    t_ch_C, pH_ch_C = run_dfba("Commensal", MONOD_FBA, sp_fracs, k_C)
    t_dh_C, pH_dh_C = run_dfba("Dysbiotic", MONOD_FBA, sp_fracs, k_C)
    ph_pred_ch_C = np.interp(EXP_DAYS, t_ch_C, pH_ch_C)
    ph_pred_dh_C = np.interp(EXP_DAYS, t_dh_C, pH_dh_C)
    met_c = compute_metrics(ph_pred_ch_C, ph_pred_dh_C, ph_exp_ch, ph_exp_dh)
    print(f"   RMSE={met_c['rmse']:.4f}, R²={met_c['r2']:.4f}, RMSE_ΔpH={met_c['rmse_delta']:.4f}")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 68)
    print(f"{'Method':<30} {'RMSE':>6}  {'R²':>6}  {'RMSE_ΔpH':>9}  {'RMSE_CH':>8}  {'RMSE_DH':>8}")
    print("-" * 68)
    for name, m in [
        ("A: Simple (So+Vei)", met_a),
        ("B: dFBA-Monod (lit.)", met_b),
        ("C: dFBA-Monod (FBA)", met_c),
    ]:
        print(
            f"{name:<30} {m['rmse']:6.4f}  {m['r2']:6.4f}  {m['rmse_delta']:9.4f}  "
            f"{m['rmse_ch']:8.4f}  {m['rmse_dh']:8.4f}"
        )
    print("=" * 68)

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        "pH Prediction Comparison: 3 Methods vs Heine 2025 Fig 4A", fontsize=13, fontweight="bold"
    )
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.32)

    COLOR_A = "#4CAF50"  # green
    COLOR_B = "#FF9800"  # orange
    COLOR_C = "#9C27B0"  # purple
    COL_CH = "#1565C0"  # dark blue
    COL_DH = "#B71C1C"  # dark red

    # ---- (top-left) CH pH time series ----
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(t_ph, ph_com, "-", color=COL_CH, lw=2, alpha=0.8, label="Experiment (CH)")
    ax0.plot(res_a["t_cont"], res_a["ph_cont_ch"], "--", color=COLOR_A, lw=1.5, label="A: Simple")
    ax0.plot(t_ch_B, pH_ch_B, "-.", color=COLOR_B, lw=1.5, label="B: dFBA-Lit")
    ax0.plot(t_ch_C, pH_ch_C, ":", color=COLOR_C, lw=1.5, label="C: dFBA-FBA")
    ax0.scatter(EXP_DAYS, ph_exp_ch, c=COL_CH, s=30, zorder=5)
    ax0.set(xlabel="Day", ylabel="pH", title="Commensal HOBIC")
    ax0.set_ylim(5.8, 8.0)
    ax0.legend(fontsize=7)

    # ---- (top-center) DH pH time series ----
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(t_ph, ph_dys, "-", color=COL_DH, lw=2, alpha=0.8, label="Experiment (DH)")
    ax1.plot(res_a["t_cont"], res_a["ph_cont_dh"], "--", color=COLOR_A, lw=1.5, label="A: Simple")
    ax1.plot(t_dh_B, pH_dh_B, "-.", color=COLOR_B, lw=1.5, label="B: dFBA-Lit")
    ax1.plot(t_dh_C, pH_dh_C, ":", color=COLOR_C, lw=1.5, label="C: dFBA-FBA")
    ax1.scatter(EXP_DAYS, ph_exp_dh, c=COL_DH, s=30, zorder=5)
    ax1.set(xlabel="Day", ylabel="pH", title="Dysbiotic HOBIC")
    ax1.set_ylim(5.8, 8.0)
    ax1.legend(fontsize=7)

    # ---- (top-right) ΔpH(DH-CH) time series ----
    ax2 = fig.add_subplot(gs[0, 2])
    delta_exp = np.interp(EXP_DAYS, t_ph, ph_dys) - np.interp(EXP_DAYS, t_ph, ph_com)
    ax2.plot(EXP_DAYS, delta_exp, "-o", color="k", lw=2, ms=6, label="Experiment", zorder=6)
    delta_A = res_a["ph_pred_dh"] - res_a["ph_pred_ch"]
    delta_B = ph_pred_dh_B - ph_pred_ch_B
    delta_C = ph_pred_dh_C - ph_pred_ch_C
    ax2.plot(EXP_DAYS, delta_A, "--s", color=COLOR_A, lw=1.5, ms=5, label="A: Simple")
    ax2.plot(EXP_DAYS, delta_B, "-.^", color=COLOR_B, lw=1.5, ms=5, label="B: dFBA-Lit")
    ax2.plot(EXP_DAYS, delta_C, ":D", color=COLOR_C, lw=1.5, ms=5, label="C: dFBA-FBA")
    ax2.axhline(0, color="gray", lw=0.8, ls="--")
    ax2.set(xlabel="Day", ylabel="ΔpH (DH − CH)", title="Condition Discrimination: ΔpH")
    ax2.legend(fontsize=7)

    # ---- (bottom-left) Predicted vs Measured scatter ----
    ax3 = fig.add_subplot(gs[1, 0])
    all_obs = np.concatenate([ph_exp_ch, ph_exp_dh])
    for pred, color, label in [
        (
            np.concatenate([res_a["ph_pred_ch"], res_a["ph_pred_dh"]]),
            COLOR_A,
            f"A (R²={met_a['r2']:.3f})",
        ),
        (np.concatenate([ph_pred_ch_B, ph_pred_dh_B]), COLOR_B, f"B (R²={met_b['r2']:.3f})"),
        (np.concatenate([ph_pred_ch_C, ph_pred_dh_C]), COLOR_C, f"C (R²={met_c['r2']:.3f})"),
    ]:
        ax3.scatter(all_obs, pred, color=color, s=30, alpha=0.7, label=label)
    lim = [5.9, 7.4]
    ax3.plot(lim, lim, "k--", lw=0.8, alpha=0.5)
    ax3.set(
        xlabel="Measured pH",
        ylabel="Predicted pH",
        title="Predicted vs Measured (CH+DH)",
        xlim=lim,
        ylim=lim,
    )
    ax3.legend(fontsize=8)

    # ---- (bottom-center) RMSE bar chart ----
    ax4 = fig.add_subplot(gs[1, 1])
    methods = ["A: Simple\n(So+Vei)", "B: dFBA\n(Lit.)", "C: dFBA\n(FBA)"]
    colors = [COLOR_A, COLOR_B, COLOR_C]
    rmse_all = [met_a["rmse"], met_b["rmse"], met_c["rmse"]]
    rmse_ch = [met_a["rmse_ch"], met_b["rmse_ch"], met_c["rmse_ch"]]
    rmse_dh = [met_a["rmse_dh"], met_b["rmse_dh"], met_c["rmse_dh"]]
    x = np.arange(3)
    w = 0.25
    ax4.bar(x - w, rmse_ch, w, label="CH", color=[c + "80" for c in colors])
    ax4.bar(x, rmse_dh, w, label="DH", color=colors)
    ax4.bar(
        x + w, rmse_all, w, label="All", color=[c + "CC" for c in colors], edgecolor="k", lw=0.5
    )
    ax4.set_xticks(x)
    ax4.set_xticklabels(methods, fontsize=8)
    ax4.set(ylabel="RMSE (pH units)", title="RMSE by Method")
    ax4.legend(fontsize=8)
    for xi, v in zip(x + w, rmse_all):
        ax4.text(xi, v + 0.003, f"{v:.3f}", ha="center", fontsize=7)

    # ---- (bottom-right) RMSE_ΔpH + summary text ----
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.bar(
        range(3),
        [met_a["rmse_delta"], met_b["rmse_delta"], met_c["rmse_delta"]],
        color=colors,
        edgecolor="k",
        lw=0.5,
    )
    ax5.set_xticks(range(3))
    ax5.set_xticklabels(methods, fontsize=8)
    ax5.set(ylabel="RMSE of ΔpH (DH−CH)", title="ΔpH Discrimination Error")
    for xi, v in enumerate([met_a["rmse_delta"], met_b["rmse_delta"], met_c["rmse_delta"]]):
        ax5.text(xi, v + 0.003, f"{v:.3f}", ha="center", fontsize=8)

    # Inset text summary
    summary = (
        "Model summary (N=12 points: 6days×2cond)\n"
        f"A: pH={res_a['beta'][0]:.2f}{res_a['beta'][1]:+.2f}·φ_So{res_a['beta'][2]:+.2f}·φ_Vd\n"
        f"   RMSE={met_a['rmse']:.4f}, R²={met_a['r2']:.3f}\n"
        f"B: Monod ODE, lit. stoichiometry\n"
        f"   RMSE={met_b['rmse']:.4f}, R²={met_b['r2']:.3f}\n"
        f"C: Monod ODE, AGORA FBA-informed\n"
        f"   RMSE={met_c['rmse']:.4f}, R²={met_c['r2']:.3f}\n"
        f"\nΔpH discrimination RMSE:\n"
        f"  A={met_a['rmse_delta']:.4f}, B={met_b['rmse_delta']:.4f}, C={met_c['rmse_delta']:.4f}"
    )
    ax5.text(
        1.05,
        1.0,
        summary,
        transform=ax5.transAxes,
        fontsize=7,
        va="top",
        ha="left",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.8),
    )

    out = OUT_DIR / "compare_ph_methods.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out}")

    # JSON summary
    import json

    summary_dict = {
        "method_A_simple": {**met_a, "beta": res_a["beta"].tolist()},
        "method_B_dfba_lit": met_b,
        "method_C_dfba_fba": met_c,
    }
    json_out = OUT_DIR / "compare_ph_methods.json"
    with open(json_out, "w") as f:
        json.dump(summary_dict, f, indent=2)
    print(f"Saved: {json_out}")


if __name__ == "__main__":
    main()
