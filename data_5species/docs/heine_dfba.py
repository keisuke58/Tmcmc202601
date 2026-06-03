"""
heine_dfba.py
=============
Hybrid Monod dFBA for Heine 2025 5-species oral biofilm.
Applies nife-intern COMETS/Monod stoichiometric framework to 4 Heine conditions.

Strategy (hybrid):
  Species fractions φ_i(t): interpolated from experimental medians (Fig 3A)
  Metabolite dynamics:  dC_j/dt = Σ_i S_ij · q_i(C) · X_i + D·(C_feed - C)
  pH prediction:        iterative charge balance from organic acids + BHI buffer

This separates the species-competition problem (handled by TMCMC/Hamilton ODE)
from the metabolic/pH mechanism question (tested here via stoichiometry).

Saved figures:
  heine_dfba_species.png   — experimental species fractions (dots) + interp lines
  heine_dfba_pH.png        — pH comparison (HOBIC): dFBA vs Fig 4A experiment
  heine_dfba_metabolites.png — metabolite time courses (HOBIC)
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import pandas as pd
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

# ---------------------------------------------------------------------------
# Monod kinetic parameters (from nife/comets/oral_biofilm.py)
# uptake: {substrate: (q_max [mmol/gDW/h], Km [mM], Y [gDW/mmol])}
# ---------------------------------------------------------------------------
MONOD: dict[str, dict] = {
    "So": {
        "uptake": {"glc": (8.0, 0.05, 0.10)},
        "secretion_per_glc": {"lac": 1.8, "ace": 0.3, "for": 0.3, "suc": 0.1},
    },
    "An": {
        "uptake": {"glc": (6.0, 0.08, 0.08)},
        "secretion_per_glc": {"lac": 1.2, "suc": 0.3, "ace": 0.2, "for": 0.2},
    },
    "Vp": {
        # Obligate lactate fermenter; also utilises succinate
        # 3 lac → 1 pro + 1 ace + CO2  (Rogosa 1964)
        "uptake": {"lac": (2.2, 0.15, 0.07), "suc": (1.0, 0.10, 0.06)},
        "secretion_per_lac": {"pro": 0.33, "ace": 0.33},
        "secretion_per_suc": {"pro": 0.50, "ace": 0.20},
    },
    "Fn": {
        "uptake": {"glc": (4.0, 0.12, 0.09), "lac": (3.0, 0.18, 0.07)},
        "secretion_per_glc": {"but": 0.6, "pro": 0.2, "ace": 0.2},
        "secretion_per_lac": {"but": 0.4, "ace": 0.2},
    },
    "Pg": {
        # Pg uses succinate + hemin; hemin modelled as always-available (BHI)
        "uptake": {"suc": (3.0, 0.08, 0.10)},
        "secretion_per_suc": {"but": 0.5, "ace": 0.3},
    },
}

SPECIES = ["So", "An", "Vp", "Fn", "Pg"]
METS = ["glc", "lac", "suc", "pro", "ace", "but", "for"]
N_SP = len(SPECIES)
N_MET = len(METS)
MET_IDX = {m: i for i, m in enumerate(METS)}

# Empirical pH model:  pH(t) = PH_INITIAL - k_pH × A_net(t)
# Acid weights reflect relative proton-donating capacity (Stryer 2002).
# k_pH auto-calibrated in main() to match Commensal HOBIC plateau pH ~ 6.3 (Fig 4A).
ACID_WEIGHTS: dict[str, float] = {
    "lac": 1.00,
    "ace": 0.80,
    "pro": 0.70,
    "but": 0.60,
    "for": 1.00,
    "suc": 0.50,
}
PH_INITIAL = 7.5

# Effective biomass density [g/mL] for 100% colonization (dense biofilm surface layer)
# Calibrated so that steady-state [lac]~3 mM in CH (needed for pH ~6.3)
X_SCALE = 1.5e-3

# BHI medium (full vs 1:2 diluted from day 1)
MEDIA_FULL = {"glc": 5.5, "lac": 0.0, "suc": 0.05, "pro": 0.0, "ace": 0.0, "but": 0.0, "for": 0.0}
MEDIA_DILU = {"glc": 2.8, "lac": 0.0, "suc": 0.03, "pro": 0.0, "ace": 0.0, "but": 0.0, "for": 0.0}

# Metabolite washout/dilution rate [h⁻¹]
# HOBIC: 100 µl/min over ~2 mL chamber → D_flow≈3 h⁻¹, but biofilm matrix traps acids
# Effective D calibrated to give steady-state lactate ≈ 3-5 mM (matching pH ~6.3)
DILUTION_MET = {
    "Commensal_HOBIC": 0.004,
    "Commensal_Static": 0.001,
    "Dysbiotic_HOBIC": 0.004,
    "Dysbiotic_Static": 0.001,
}

# Day-1 initial species fractions (from Heine Fig 3A medians)
INIT_FRACS = {
    "Commensal_HOBIC": {"So": 0.698, "An": 0.057, "Vp": 0.099, "Fn": 0.093, "Pg": 0.005},
    "Commensal_Static": {"So": 0.180, "An": 0.005, "Vp": 0.830, "Fn": 0.005, "Pg": 0.005},
    "Dysbiotic_HOBIC": {"So": 0.031, "An": 0.013, "Vp": 0.949, "Fn": 0.005, "Pg": 0.007},
    "Dysbiotic_Static": {"So": 0.020, "An": 0.010, "Vp": 0.500, "Fn": 0.200, "Pg": 0.250},
}

# ---------------------------------------------------------------------------
# Experimental species interpolators
# ---------------------------------------------------------------------------


def build_species_interpolators(condition: str, cultivation: str) -> dict:
    """Interpolate experimental median species fractions from fig3 summary."""
    df = pd.read_csv(DATA_DIR / "fig3_species_distribution_summary.csv")
    color_map = {
        "S. oralis": "So",
        "A. naeslundii": "An",
        "V. dispar": "Vp",
        "V. parvula": "Vp",
        "F. nucleatum": "Fn",
        "P. gingivalis_20709": "Pg",
        "P. gingivalis_W83": "Pg",
    }
    subset = df[(df["condition"] == condition) & (df["cultivation"] == cultivation)]

    sp_data: dict[str, tuple[list, list]] = {}
    for _, row in subset.iterrows():
        sp = color_map.get(row["species"], row["species"])
        if sp not in sp_data:
            sp_data[sp] = ([], [])
        sp_data[sp][0].append(float(row["day"]))
        sp_data[sp][1].append(float(row["median"]) / 100.0)

    cond_key = f"{condition}_{cultivation}"
    f0 = INIT_FRACS[cond_key]

    interps = {}
    for sp in SPECIES:
        if sp in sp_data:
            days_raw, frac_raw = sp_data[sp]
            pairs = sorted(zip(days_raw, frac_raw))
            days = [0.0] + [p[0] for p in pairs]
            frac = [f0[sp]] + [p[1] for p in pairs]
            interps[sp] = interp1d(
                days, frac, kind="linear", bounds_error=False, fill_value=(frac[0], frac[-1])
            )
        else:
            _f = f0[sp]
            interps[sp] = lambda t, _v=_f: np.full(np.shape(t) or (1,), _v, dtype=float)

    return interps


# ---------------------------------------------------------------------------
# Metabolite ODE
# ---------------------------------------------------------------------------


def make_met_ode(cond_key: str, interps: dict):
    D = DILUTION_MET[cond_key]

    def ode(t_h: float, C: np.ndarray) -> np.ndarray:
        C = np.maximum(C, 0.0)
        t_d = t_h / 24.0

        # species fractions from experiment
        phi = np.array([float(interps[sp](t_d)) for sp in SPECIES])
        phi = np.maximum(phi, 0.0)
        s = phi.sum()
        if s > 1e-15:
            phi /= s
        X = phi * X_SCALE

        Cdict = {m: C[MET_IDX[m]] for m in METS}
        dC = np.zeros(N_MET)
        feed = MEDIA_FULL if t_h < 24.0 else MEDIA_DILU

        for i, sp in enumerate(SPECIES):
            p = MONOD[sp]
            bm = X[i]

            q: dict[str, float] = {}
            for sub, (q_max, Km, _Y) in p["uptake"].items():
                q[sub] = q_max * Cdict[sub] / (Km + Cdict[sub] + 1e-15)

            for sub, q_val in q.items():
                if sub in MET_IDX:
                    dC[MET_IDX[sub]] -= q_val * bm

                sec_key = f"secretion_per_{sub}"
                if sec_key in p:
                    for prod, stoich in p[sec_key].items():
                        if prod in MET_IDX:
                            dC[MET_IDX[prod]] += stoich * q_val * bm

        # Washout + fresh medium supply
        for m in METS:
            dC[MET_IDX[m]] += D * (feed.get(m, 0.0) - C[MET_IDX[m]])

        return dC

    return ode


# ---------------------------------------------------------------------------
# pH from organic acid concentrations — iterative charge balance
# ---------------------------------------------------------------------------


def compute_acid_load(C_time: np.ndarray) -> np.ndarray:
    """Weighted sum of organic acid concentrations [mM]. Shape: (T,)"""
    A = np.zeros(len(C_time))
    for m, w in ACID_WEIGHTS.items():
        if m in MET_IDX:
            A += w * C_time[:, MET_IDX[m]]
    return A


def compute_pH_series(C_time: np.ndarray, k_pH: float) -> np.ndarray:
    """
    Empirical linear pH model:
      pH(t) = PH_INITIAL - k_pH × A_net(t)
    k_pH is calibrated externally (see main) to match Commensal HOBIC plateau.
    """
    return np.clip(PH_INITIAL - k_pH * compute_acid_load(C_time), 3.0, 9.0)


# ---------------------------------------------------------------------------
# Run one condition
# ---------------------------------------------------------------------------


def run_condition(condition: str, cultivation: str, k_pH: float = 1.0, t_days: float = 21.0):
    cond_key = f"{condition}_{cultivation}"
    interps = build_species_interpolators(condition, cultivation)
    ode = make_met_ode(cond_key, interps)

    C0 = np.array([MEDIA_FULL.get(m, 0.0) for m in METS])
    t_end_h = t_days * 24.0
    t_eval = np.linspace(0.0, t_end_h, 600)

    sol = solve_ivp(ode, [0.0, t_end_h], C0, t_eval=t_eval, method="RK45", rtol=1e-6, atol=1e-9)

    t_d = sol.t / 24.0
    C = np.maximum(sol.y.T, 0.0)
    pH = compute_pH_series(C, k_pH)

    # Species fractions for plotting (from interpolators)
    phi_mat = np.zeros((len(t_d), N_SP))
    for i, sp in enumerate(SPECIES):
        phi_mat[:, i] = np.clip([float(interps[sp](d)) for d in t_d], 0.0, None)
    phi_mat /= phi_mat.sum(axis=1, keepdims=True) + 1e-15

    return t_d, phi_mat, C, pH


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
SPECIES_COLORS = {
    "So": "#2196F3",
    "An": "#4CAF50",
    "Vp": "#FF9800",
    "Fn": "#9C27B0",
    "Pg": "#F44336",
}
SPECIES_LABELS = {
    "So": "S. oralis",
    "An": "A. naeslundii",
    "Vp": "V. dispar/parvula",
    "Fn": "F. nucleatum",
    "Pg": "P. gingivalis",
}
CONDITIONS = [
    ("Commensal", "HOBIC"),
    ("Commensal", "Static"),
    ("Dysbiotic", "HOBIC"),
    ("Dysbiotic", "Static"),
]
MET_COLORS = {
    "glc": "#795548",
    "lac": "#2196F3",
    "suc": "#4CAF50",
    "pro": "#FF9800",
    "ace": "#9C27B0",
    "but": "#F44336",
    "for": "#607D8B",
}


def load_pH_data():
    df = pd.read_csv(DATA_DIR / "fig4A_pH_timeseries.csv")
    return df["time_days"].values, df["pH_commensal"].values, df["pH_dysbiotic"].values


def load_species_exp(condition: str, cultivation: str) -> dict[str, tuple]:
    df = pd.read_csv(DATA_DIR / "fig3_species_distribution_summary.csv")
    color_map = {
        "S. oralis": "So",
        "A. naeslundii": "An",
        "V. dispar": "Vp",
        "V. parvula": "Vp",
        "F. nucleatum": "Fn",
        "P. gingivalis_20709": "Pg",
        "P. gingivalis_W83": "Pg",
    }
    subset = df[(df["condition"] == condition) & (df["cultivation"] == cultivation)]
    result: dict[str, tuple] = {}
    for _, row in subset.iterrows():
        sp = color_map.get(row["species"], row["species"])
        if sp not in result:
            result[sp] = ([], [])
        result[sp][0].append(float(row["day"]))
        result[sp][1].append(float(row["median"]))
    return {k: (np.array(v[0]), np.array(v[1])) for k, v in result.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("Running Heine hybrid dFBA simulations...")

    # Phase 1: run Commensal HOBIC WITHOUT pH calibration to compute k_pH
    t_CH, _, C_CH_raw, _ = run_condition("Commensal", "HOBIC", k_pH=1.0)
    # Use plateau (day 10-21) mean acid load to calibrate k_pH
    mask = t_CH >= 10.0
    A_CH_ss = compute_acid_load(C_CH_raw)[mask].mean()
    pH_CH_exp_plateau = 6.30  # Heine Fig 4A plateau value [Commensal HOBIC day 10-21]
    k_pH = (PH_INITIAL - pH_CH_exp_plateau) / A_CH_ss
    print(f"  k_pH calibrated = {k_pH:.4f} pH/mM (A_net_CH = {A_CH_ss:.3f} mM)")

    # Phase 2: run all conditions with calibrated k_pH
    results = {}
    for cond, cult in CONDITIONS:
        key = f"{cond}_{cult}"
        print(f"  {key}...", end=" ", flush=True)
        results[key] = run_condition(cond, cult, k_pH=k_pH)
        print("done")

    t_pH_exp, pH_comm_exp, pH_dys_exp = load_pH_data()

    # ---- Fig 1: 4-panel species fractions ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    axes = axes.flatten()
    titles = {
        "Commensal_HOBIC": "Commensal HOBIC",
        "Commensal_Static": "Commensal Static",
        "Dysbiotic_HOBIC": "Dysbiotic HOBIC",
        "Dysbiotic_Static": "Dysbiotic Static",
    }

    for ax, (cond, cult) in zip(axes, CONDITIONS):
        key = f"{cond}_{cult}"
        t_sim, phi_sim, _C, _pH = results[key]

        # Experimental scatter
        exp = load_species_exp(cond, cult)
        for sp, (days_e, med_e) in exp.items():
            ax.scatter(
                days_e,
                med_e,
                color=SPECIES_COLORS.get(sp, "gray"),
                s=35,
                zorder=5,
                marker="o",
                alpha=0.8,
            )

        # Interpolated lines
        for i, sp in enumerate(SPECIES):
            ax.plot(
                t_sim, phi_sim[:, i] * 100, color=SPECIES_COLORS[sp], lw=2, label=SPECIES_LABELS[sp]
            )

        ax.set_title(titles[key], fontsize=11, fontweight="bold")
        ax.set_xlabel("Day")
        ax.set_ylabel("Species fraction (%)")
        ax.set_xlim(0, 21)
        ax.set_ylim(-2, 105)
        ax.set_xticks([0, 3, 6, 10, 15, 21])
        ax.grid(alpha=0.3)

    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(
        "Heine 2025: Species Fractions\n(lines = exp. median interpolation, dots = observed medians)",
        fontsize=12,
    )
    out1 = OUT_DIR / "heine_dfba_species.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out1}")

    # ---- Fig 2: pH HOBIC ----
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(t_pH_exp, pH_comm_exp, "b-", lw=2, alpha=0.7, label="Commensal HOBIC (exp)")
    ax.plot(t_pH_exp, pH_dys_exp, "r-", lw=2, alpha=0.7, label="Dysbiotic HOBIC (exp)")

    t_ch, _, C_ch, pH_ch = results["Commensal_HOBIC"]
    t_dh, _, C_dh, pH_dh = results["Dysbiotic_HOBIC"]
    ax.plot(t_ch, pH_ch, "b--", lw=2, label="Commensal HOBIC (dFBA)")
    ax.plot(t_dh, pH_dh, "r--", lw=2, label="Dysbiotic HOBIC (dFBA)")
    ax.axhline(7.4, color="gray", ls=":", lw=1, alpha=0.4)

    ax.set_xlabel("Day", fontsize=12)
    ax.set_ylabel("pH", fontsize=12)
    ax.set_title(
        "Heine 2025 HOBIC: pH dynamics\nMonod dFBA (S·v stoichiometry) vs experiment", fontsize=12
    )
    ax.set_xlim(0, 21)
    ax.set_ylim(5.8, 7.8)
    ax.set_xticks([0, 3, 6, 10, 15, 21])
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    out2 = OUT_DIR / "heine_dfba_pH.png"
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out2}")

    # ---- Fig 3: Metabolite time courses (HOBIC) ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, (cond, _) in zip(axes, [("Commensal", "HOBIC"), ("Dysbiotic", "HOBIC")]):
        key = f"{cond}_HOBIC"
        t_sim, _, C_sim, _ = results[key]
        for m in METS:
            ax.plot(t_sim, C_sim[:, MET_IDX[m]], color=MET_COLORS[m], lw=2, label=m.upper())
        ax.set_title(f"{cond} HOBIC — Metabolites", fontsize=11, fontweight="bold")
        ax.set_xlabel("Day")
        ax.set_ylabel("Concentration (mM)")
        ax.set_xlim(0, 21)
        ax.set_xticks([0, 3, 6, 10, 15, 21])
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.3)
    fig.suptitle("Heine dFBA: Metabolite Dynamics (HOBIC)", fontsize=12)
    out3 = OUT_DIR / "heine_dfba_metabolites.png"
    fig.savefig(out3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out3}")

    # ---- Summary stats ----
    print("\n=== pH day 21 (HOBIC) ===")
    print(f"  Commensal dFBA: {pH_ch[-1]:.3f}   exp: {pH_comm_exp[-1]:.3f}")
    print(f"  Dysbiotic  dFBA: {pH_dh[-1]:.3f}   exp: {pH_dys_exp[-1]:.3f}")
    print(
        f"  ΔpH (Dys-Comm) dFBA: {pH_dh[-1]-pH_ch[-1]:.3f}   exp: {pH_dys_exp[-1]-pH_comm_exp[-1]:.3f}"
    )

    print("\n=== Steady-state metabolites [mM] day 21, Commensal vs Dysbiotic HOBIC ===")
    print(f"{'Met':>5}  {'Comm':>8}  {'Dys':>8}  {'Ratio(D/C)':>10}")
    for m in METS:
        c_c = C_ch[-1, MET_IDX[m]]
        c_d = C_dh[-1, MET_IDX[m]]
        ratio = c_d / (c_c + 1e-9)
        print(f"{m:>5}  {c_c:8.4f}  {c_d:8.4f}  {ratio:10.3f}")


if __name__ == "__main__":
    main()
