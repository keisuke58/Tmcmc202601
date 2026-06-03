"""
heine_fba.py
============
AGORA GEM-informed dFBA for Heine 2025 5-species oral biofilm.

Workflow:
  1. Load AGORA GEMs → run pFBA to extract secretion stoichiometry ratios
  2. Replace assumed Monod stoichiometry with FBA-derived ratios where reliable
  3. Run Monod-rate dFBA with updated stoichiometry
  4. Compare: Monod-assumed vs FBA-informed vs experiment (pH, metabolites)

Key finding from AGORA analysis:
  - So/An: AGORA GEM predicts mixed-acid fermentation (acetate + formate), NOT primarily lactate
    → Consistent with low-glucose mixed-acid pathway (Streptococcus at limited carbon)
  - Vp: AGORA propionate pathway from succinate confirmed; unreliable stoichiometry ratios
    → Keep literature values (3 lac → 1 pro + 1 ace, Rogosa 1964)
  - Fn/Pg: Butyrate + propionate confirmed; FBA ratios inflated due to amino acid catabolism
    → Use normalized FBA ratios capped at physiological range

Limitation: AGORA oral bacteria models require VMH dietary inputs (amino acids) for growth.
Without these, some species cannot grow. With them, amino acid catabolism inflates fluxes.
FBA stoichiometry is used QUALITATIVELY (which metabolites are produced) rather than
QUANTITATIVELY (exact ratios).
"""

from __future__ import annotations

import warnings, sys
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "experiment_data"
OUT_DIR = SCRIPT_DIR / "paper_comprehensive_figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
AGORA_DIR = Path("/home/nishioka/IKM_Hiwi/nife/comets/agora_gems")

GEM_FILES = {
    "So": AGORA_DIR / "Streptococcus_oralis_Uo5.xml",
    "An": AGORA_DIR / "Actinomyces_naeslundii_str_Howell_279.xml",
    "Vp": AGORA_DIR / "Veillonella_parvula_Te3_DSM_2008.xml",
    "Fn": AGORA_DIR / "Fusobacterium_nucleatum_subsp_nucleatum_ATCC_25586.xml",
    "Pg": AGORA_DIR / "Porphyromonas_gingivalis_W83.xml",
}

SPECIES = ["So", "An", "Vp", "Fn", "Pg"]
METS = ["glc", "lac", "suc", "pro", "ace", "but", "for"]
N_SP = len(SPECIES)
N_MET = len(METS)
MET_IDX = {m: i for i, m in enumerate(METS)}

MET_TO_EX = {
    "glc": "EX_glc_D(e)",
    "lac": "EX_lac_L(e)",
    "suc": "EX_succ(e)",
    "pro": "EX_ppa(e)",
    "ace": "EX_ac(e)",
    "but": "EX_but(e)",
    "for": "EX_for(e)",
    "h2s": "EX_h2s(e)",
}

MONOD_UPTAKE = {
    "So": {"glc": (8.0, 0.05)},
    "An": {"glc": (6.0, 0.08)},
    "Vp": {"lac": (2.2, 0.15), "suc": (1.0, 0.10)},
    "Fn": {"glc": (4.0, 0.12), "lac": (3.0, 0.18)},
    "Pg": {"suc": (3.0, 0.08)},
}

# ---------------------------------------------------------------------------
# AGORA base medium (required for GEM feasibility — trace amino acids + minerals)
# ---------------------------------------------------------------------------
BASE_MEDIUM_EX = {
    "EX_nh4(e)": 100.0,
    "EX_pi(e)": 100.0,
    "EX_h2o(e)": 1000.0,
    "EX_co2(e)": 100.0,
    "EX_h(e)": 100.0,
    "EX_so4(e)": 10.0,
    "EX_ca2(e)": 5.0,
    "EX_mg2(e)": 5.0,
    "EX_k(e)": 10.0,
    "EX_fe2(e)": 1.0,
    "EX_fe3(e)": 1.0,
    "EX_mn2(e)": 1.0,
    "EX_zn2(e)": 1.0,
    "EX_cu2(e)": 0.5,
    "EX_cobalt2(e)": 0.5,
    "EX_cl(e)": 10.0,
    "EX_thm(e)": 1.0,
    "EX_ribflv(e)": 1.0,
    "EX_nac(e)": 1.0,
    "EX_pnto_R(e)": 1.0,
    "EX_fol(e)": 1.0,
    "EX_btn(e)": 1.0,
    "EX_pydx(e)": 1.0,
    "EX_2dmmq8(e)": 0.1,
    "EX_mqn7(e)": 0.1,
    "EX_mqn8(e)": 0.1,
    "EX_q8(e)": 0.1,
    "EX_ade(e)": 0.1,
    "EX_csn(e)": 0.1,
    "EX_cytd(e)": 0.1,
    "EX_dad_2(e)": 0.1,
    "EX_dcyt(e)": 0.1,
    "EX_dgsn(e)": 0.1,
    "EX_gsn(e)": 0.1,
    "EX_gua(e)": 0.1,
    "EX_hxan(e)": 0.1,
    "EX_ins(e)": 0.1,
    "EX_uri(e)": 0.1,
    "EX_ala_L(e)": 10.0,
    "EX_arg_L(e)": 10.0,
    "EX_cys_L(e)": 5.0,
    "EX_gln_L(e)": 10.0,
    "EX_glu_L(e)": 10.0,
    "EX_his_L(e)": 5.0,
    "EX_ile_L(e)": 10.0,
    "EX_leu_L(e)": 10.0,
    "EX_lys_L(e)": 10.0,
    "EX_met_L(e)": 5.0,
    "EX_phe_L(e)": 5.0,
    "EX_pro_L(e)": 10.0,
    "EX_ser_L(e)": 10.0,
    "EX_thr_L(e)": 10.0,
    "EX_trp_L(e)": 2.0,
    "EX_tyr_L(e)": 5.0,
    "EX_val_L(e)": 10.0,
    "EX_alagln(e)": 1.0,
    "EX_alahis(e)": 1.0,
    "EX_alathr(e)": 1.0,
    "EX_cgly(e)": 1.0,
    "EX_glyasn(e)": 1.0,
    "EX_glycys(e)": 1.0,
    "EX_glygln(e)": 1.0,
    "EX_glyleu(e)": 1.0,
    "EX_glymet(e)": 1.0,
    "EX_glytyr(e)": 1.0,
    "EX_metala(e)": 1.0,
    "EX_pheme(e)": 1.0,
    "EX_sheme(e)": 0.5,
    "EX_26dap_M(e)": 0.1,
    "EX_3mop(e)": 0.1,
    "EX_4hbz(e)": 0.1,
    "EX_acgam(e)": 0.1,
    "EX_chtbs(e)": 0.1,
    "EX_ddca(e)": 0.1,
    "EX_gam(e)": 0.1,
    "EX_glyc3p(e)": 0.1,
    "EX_gthrd(e)": 0.1,
    "EX_ocdca(e)": 0.1,
    "EX_orn(e)": 0.1,
    "EX_phpyr(e)": 0.1,
    "EX_spmd(e)": 0.1,
    "EX_ttdca(e)": 0.1,
    "EX_fru(e)": 0.1,
    "EX_pullulan1200(e)": 0.1,
    "EX_stys(e)": 0.1,
    "EX_o2(e)": 0.0,
}


def load_model(sp: str):
    import cobra

    m = cobra.io.read_sbml_model(str(GEM_FILES[sp]))
    for r in m.exchanges:
        r.lower_bound = 0.0
        r.upper_bound = 1000.0
    for ex_id, lb in BASE_MEDIUM_EX.items():
        if ex_id in m.reactions:
            m.reactions.get_by_id(ex_id).lower_bound = -lb
    return m


# ---------------------------------------------------------------------------
# Extract FBA secretion stoichiometry
# ---------------------------------------------------------------------------


def extract_fba_stoichiometry(
    test_glc: float = 2.8, test_lac: float = 2.5, test_suc: float = 0.05
) -> dict[str, dict[str, float]]:
    """
    Run pFBA for each species at given substrate concentrations.
    Return normalized secretion ratios: {sp: {met: mmol_secreted/mmol_primary_consumed}}
    """
    from cobra.flux_analysis import pfba

    test_conc = {"glc": test_glc, "lac": test_lac, "suc": test_suc}
    stoich: dict[str, dict[str, float]] = {}

    for sp in SPECIES:
        m = load_model(sp)
        for sub, (q_max, Km) in MONOD_UPTAKE[sp].items():
            c = test_conc.get(sub, 0.0)
            q_lim = q_max * c / (Km + c + 1e-15)
            ex_id = MET_TO_EX[sub]
            if ex_id in m.reactions:
                m.reactions.get_by_id(ex_id).lower_bound = -q_lim

        sol_fba = m.optimize()
        if sol_fba.status != "optimal" or (sol_fba.objective_value or 0) < 1e-8:
            stoich[sp] = {}
            continue

        try:
            sol = pfba(m)
        except Exception:
            sol = sol_fba

        primary = list(MONOD_UPTAKE[sp].keys())[0]
        prim_ex = MET_TO_EX[primary]
        uptake_rate = abs(sol.fluxes.get(prim_ex, 0))
        if uptake_rate < 1e-9:
            stoich[sp] = {}
            continue

        sp_stoich = {}
        for met, ex_id in MET_TO_EX.items():
            if met == primary:
                continue
            if ex_id in m.reactions:
                v = sol.fluxes.get(ex_id, 0.0)
                if v > 0.001:  # secretion
                    raw_ratio = v / uptake_rate
                    # Cap at physiological maximum: no product can exceed 3× substrate input
                    capped = min(raw_ratio, 3.0)
                    if capped > 0.005:
                        sp_stoich[met] = capped

        stoich[sp] = sp_stoich

    return stoich


# ---------------------------------------------------------------------------
# Build MONOD dict with FBA-corrected stoichiometry
# ---------------------------------------------------------------------------


def build_fba_monod(fba_stoich: dict) -> dict:
    """
    Merge FBA stoichiometry with literature-based Monod parameters.
    Strategy:
      - Keep Monod uptake rates (q_max, Km) from calibrated literature values
      - Replace secretion ratios with FBA-derived values where available & reliable
      - For Vp: always use literature (FBA Vp stoichiometry is inflated)
      - Cap all secretion ratios at physiological maximum (3.0)
    """
    # Literature-based Monod (fallback)
    LIT_MONOD = {
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
        "Pg": {
            "uptake": {"suc": (3.0, 0.08, 0.10)},
            "secretion_per_suc": {"but": 0.5, "ace": 0.3},
        },
    }

    merged = {}
    for sp in SPECIES:
        p = dict(LIT_MONOD[sp])
        merged[sp] = p

        sp_fba = fba_stoich.get(sp, {})
        if not sp_fba or sp == "Vp":
            continue  # Vp: literature only; others: no FBA result → keep literature

        primary = list(MONOD_UPTAKE[sp].keys())[0]
        sec_key = f"secretion_per_{primary}"

        # Start from FBA ratios, but only for organic acid metabolites
        fba_sec = {
            m: v
            for m, v in sp_fba.items()
            if m in ["lac", "ace", "pro", "but", "for", "suc"] and v > 0.01
        }

        if fba_sec:
            merged[sp] = dict(p)
            merged[sp][sec_key] = fba_sec
            print(
                f"  {sp}: updated stoichiometry from FBA "
                f"({', '.join(f'{m}={v:.2f}' for m, v in fba_sec.items())})"
            )
        else:
            print(f"  {sp}: keeping literature stoichiometry")

    return merged


# ---------------------------------------------------------------------------
# ODE and simulation (same structure as heine_dfba.py)
# ---------------------------------------------------------------------------
X_SCALE = 1.5e-3
DILUTION_MET = {
    "Commensal_HOBIC": 0.004,
    "Commensal_Static": 0.001,
    "Dysbiotic_HOBIC": 0.004,
    "Dysbiotic_Static": 0.001,
}
MEDIA_FULL = {"glc": 5.5, "lac": 0.0, "suc": 0.05, "pro": 0.0, "ace": 0.0, "but": 0.0, "for": 0.0}
MEDIA_DILU = {"glc": 2.8, "lac": 0.0, "suc": 0.03, "pro": 0.0, "ace": 0.0, "but": 0.0, "for": 0.0}
INIT_FRACS = {
    "Commensal_HOBIC": {"So": 0.698, "An": 0.057, "Vp": 0.099, "Fn": 0.093, "Pg": 0.005},
    "Commensal_Static": {"So": 0.180, "An": 0.005, "Vp": 0.830, "Fn": 0.005, "Pg": 0.005},
    "Dysbiotic_HOBIC": {"So": 0.031, "An": 0.013, "Vp": 0.949, "Fn": 0.005, "Pg": 0.007},
    "Dysbiotic_Static": {"So": 0.020, "An": 0.010, "Vp": 0.500, "Fn": 0.200, "Pg": 0.250},
}


def build_interps(condition: str, cultivation: str) -> dict:
    df = pd.read_csv(DATA_DIR / "fig3_species_distribution_summary.csv")
    cm = {
        "S. oralis": "So",
        "A. naeslundii": "An",
        "V. dispar": "Vp",
        "V. parvula": "Vp",
        "F. nucleatum": "Fn",
        "P. gingivalis_20709": "Pg",
        "P. gingivalis_W83": "Pg",
    }
    sub = df[(df["condition"] == condition) & (df["cultivation"] == cultivation)]
    sp_d: dict = {}
    for _, row in sub.iterrows():
        sp = cm.get(row["species"], row["species"])
        if sp not in sp_d:
            sp_d[sp] = ([], [])
        sp_d[sp][0].append(float(row["day"]))
        sp_d[sp][1].append(float(row["median"]) / 100.0)
    f0 = INIT_FRACS[f"{condition}_{cultivation}"]
    out = {}
    for sp in SPECIES:
        if sp in sp_d:
            pairs = sorted(zip(sp_d[sp][0], sp_d[sp][1]))
            days = [0.0] + [p[0] for p in pairs]
            frac = [f0[sp]] + [p[1] for p in pairs]
            out[sp] = interp1d(
                days, frac, kind="linear", bounds_error=False, fill_value=(frac[0], frac[-1])
            )
        else:
            _f = f0[sp]
            out[sp] = lambda t, _v=_f: float(_v)
    return out


def make_ode(cond_key: str, interps: dict, monod: dict):
    D = DILUTION_MET[cond_key]

    def ode(t_h, C):
        C = np.maximum(C, 0.0)
        t_d = t_h / 24.0
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
            p = monod[sp]
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
        for m in METS:
            dC[MET_IDX[m]] += D * (feed.get(m, 0.0) - C[MET_IDX[m]])
        return dC

    return ode


ACID_WEIGHTS = {"lac": 1.00, "ace": 0.80, "pro": 0.70, "but": 0.60, "for": 1.00, "suc": 0.50}
PH_INITIAL = 7.5


def acid_load(C_time):
    A = np.zeros(len(C_time))
    for m, w in ACID_WEIGHTS.items():
        if m in MET_IDX:
            A += w * C_time[:, MET_IDX[m]]
    return A


def simulate(condition, cultivation, monod, k_pH=1.0):
    cond_key = f"{condition}_{cultivation}"
    interps = build_interps(condition, cultivation)
    ode = make_ode(cond_key, interps, monod)
    C0 = np.array([MEDIA_FULL.get(m, 0.0) for m in METS])
    t_eval = np.linspace(0.0, 21.0 * 24, 500)
    sol = solve_ivp(ode, [0.0, 21.0 * 24], C0, t_eval=t_eval, method="RK45", rtol=1e-6, atol=1e-9)
    t_d = sol.t / 24.0
    C = np.maximum(sol.y.T, 0.0)
    pH = np.clip(PH_INITIAL - k_pH * acid_load(C), 3.0, 9.0)
    phi_mat = np.zeros((len(t_d), N_SP))
    for i, sp in enumerate(SPECIES):
        phi_mat[:, i] = np.clip([float(interps[sp](d)) for d in t_d], 0.0, None)
    phi_mat /= phi_mat.sum(axis=1, keepdims=True) + 1e-15
    return t_d, phi_mat, C, pH


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
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
SP_COLORS = {"So": "#2196F3", "An": "#4CAF50", "Vp": "#FF9800", "Fn": "#9C27B0", "Pg": "#F44336"}
SP_LABELS = {
    "So": "S. oralis",
    "An": "A. naeslundii",
    "Vp": "V. dispar/parvula",
    "Fn": "F. nucleatum",
    "Pg": "P. gingivalis",
}


def load_pH_data():
    df = pd.read_csv(DATA_DIR / "fig4A_pH_timeseries.csv")
    return df["time_days"].values, df["pH_commensal"].values, df["pH_dysbiotic"].values


def build_lit_monod():
    return {
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


def calibrate_k_pH(monod, target_pH=6.30, target_cond=("Commensal", "HOBIC")):
    t, _, C, _ = simulate(*target_cond, monod, k_pH=1.0)
    A_ss = acid_load(C)[t >= 10.0].mean()
    return (PH_INITIAL - target_pH) / A_ss if A_ss > 1e-6 else 1.0


def main():
    print("=" * 65)
    print("Heine FBA-dFBA: AGORA GEM-informed stoichiometry")
    print("=" * 65)

    # ---- 1. Extract FBA stoichiometry from AGORA ----
    print("\n[1] Extracting FBA stoichiometry from AGORA GEMs...")
    fba_stoich = extract_fba_stoichiometry()

    print("\n  Raw FBA secretion ratios (per mmol primary substrate consumed):")
    print(
        f"  {'Species':>6} | {'primary':<6} | {'product':>5} | {'FBA ratio':>10} | {'Lit. ratio':>10}"
    )
    lit_ratios = {
        "So": ("glc", {"lac": 1.8, "ace": 0.3, "for": 0.3}),
        "An": ("glc", {"lac": 1.2, "suc": 0.3, "ace": 0.2, "for": 0.2}),
        "Vp": ("lac", {"pro": 0.33, "ace": 0.33}),
        "Fn": ("glc", {"but": 0.6, "pro": 0.2, "ace": 0.2}),
        "Pg": ("suc", {"but": 0.5, "pro": 0.0, "ace": 0.3}),
    }
    for sp in SPECIES:
        prim, lit = lit_ratios[sp]
        sp_fba = fba_stoich.get(sp, {})
        mets_to_show = set(lit.keys()) | set(sp_fba.keys())
        for m in sorted(mets_to_show):
            fba_v = sp_fba.get(m, 0.0)
            lit_v = lit.get(m, 0.0)
            if fba_v > 0.001 or lit_v > 0.001:
                print(f"  {sp:>6} | {prim:<6} | {m:>5} | {fba_v:10.3f} | {lit_v:10.3f}")

    # ---- 2. Build merged FBA+literature MONOD ----
    print("\n[2] Building FBA-informed Monod parameters...")
    fba_monod = build_fba_monod(fba_stoich)
    lit_monod = build_lit_monod()

    # ---- 3. Calibrate k_pH for both models ----
    print("\n[3] Calibrating pH model...")
    k_lit = calibrate_k_pH(lit_monod)
    k_fba = calibrate_k_pH(fba_monod)
    print(f"   k_pH (literature Monod): {k_lit:.4f}")
    print(f"   k_pH (FBA-informed):     {k_fba:.4f}")

    # ---- 4. Run all conditions ----
    print("\n[4] Running simulations (literature Monod vs FBA-informed)...")
    res_lit = {}
    res_fba = {}
    for cond, cult in CONDITIONS:
        key = f"{cond}_{cult}"
        print(f"   {key}...", end=" ", flush=True)
        res_lit[key] = simulate(cond, cult, lit_monod, k_pH=k_lit)
        res_fba[key] = simulate(cond, cult, fba_monod, k_pH=k_fba)
        print("done")

    t_pH_exp, pH_comm_exp, pH_dys_exp = load_pH_data()

    # ---- 5. Summary stats ----
    print("\n=== pH at day 21 (HOBIC) ===")
    print(
        f"{'Model':<22} {'CH':>8}  {'DH':>8}  {'ΔpH':>8}  {'exp_CH':>8}  {'exp_DH':>8}  {'exp_ΔpH':>8}"
    )
    for label, res in [("Literature Monod", res_lit), ("FBA-informed", res_fba)]:
        t_ch, _, C_ch, pH_ch = res["Commensal_HOBIC"]
        t_dh, _, C_dh, pH_dh = res["Dysbiotic_HOBIC"]
        print(
            f"{label:<22} {pH_ch[-1]:8.3f}  {pH_dh[-1]:8.3f}  {pH_dh[-1]-pH_ch[-1]:8.3f}"
            f"  {pH_comm_exp[-1]:8.3f}  {pH_dys_exp[-1]:8.3f}  "
            f"{pH_dys_exp[-1]-pH_comm_exp[-1]:8.3f}"
        )

    print("\n=== Day-21 key metabolites, CH vs DH HOBIC ===")
    print(
        f"  {'Met':>5} | {'CH_lit':>8} {'DH_lit':>8} | {'CH_fba':>8} {'DH_fba':>8} | {'ratio_lit':>9} {'ratio_fba':>9}"
    )
    for m in METS:
        c_lit = res_lit["Commensal_HOBIC"][2][-1, MET_IDX[m]]
        d_lit = res_lit["Dysbiotic_HOBIC"][2][-1, MET_IDX[m]]
        c_fba = res_fba["Commensal_HOBIC"][2][-1, MET_IDX[m]]
        d_fba = res_fba["Dysbiotic_HOBIC"][2][-1, MET_IDX[m]]
        r_lit = d_lit / (c_lit + 1e-9)
        r_fba = d_fba / (c_fba + 1e-9)
        print(
            f"  {m:>5} | {c_lit:8.4f} {d_lit:8.4f} | {c_fba:8.4f} {d_fba:8.4f} | {r_lit:9.3f} {r_fba:9.3f}"
        )

    # ---- 6. Figures ----
    # Fig A: pH comparison
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, (title, res) in zip(axes, [("Literature Monod", res_lit), ("FBA-informed", res_fba)]):
        t_ch, _, _, pH_ch = res["Commensal_HOBIC"]
        t_dh, _, _, pH_dh = res["Dysbiotic_HOBIC"]
        ax.plot(t_pH_exp, pH_comm_exp, "b-", lw=2, alpha=0.7, label="Commensal exp")
        ax.plot(t_pH_exp, pH_dys_exp, "r-", lw=2, alpha=0.7, label="Dysbiotic exp")
        ax.plot(t_ch, pH_ch, "b--", lw=2, label="Commensal model")
        ax.plot(t_dh, pH_dh, "r--", lw=2, label="Dysbiotic model")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Day")
        ax.set_ylabel("pH")
        ax.set_xlim(0, 21)
        ax.set_ylim(5.5, 8.0)
        ax.set_xticks([0, 3, 6, 10, 15, 21])
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.axhline(7.5, color="gray", ls=":", lw=1, alpha=0.4)
    fig.suptitle("pH: Literature Monod vs AGORA FBA-informed stoichiometry", fontsize=12)
    out1 = OUT_DIR / "heine_fba_vs_monod_pH.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out1}")

    # Fig B: Metabolite comparison CH HOBIC
    fig, axes = plt.subplots(2, N_MET, figsize=(16, 7), constrained_layout=True)
    for row_i, (label, res) in enumerate(
        [("Literature Monod", res_lit), ("FBA-informed", res_fba)]
    ):
        t_ch, _, C_ch, _ = res["Commensal_HOBIC"]
        t_dh, _, C_dh, _ = res["Dysbiotic_HOBIC"]
        for j, m in enumerate(METS):
            ax = axes[row_i, j]
            ax.plot(t_ch, C_ch[:, j], color=MET_COLORS[m], lw=2, label="CH")
            ax.plot(t_dh, C_dh[:, j], color=MET_COLORS[m], lw=2, ls="--", alpha=0.7, label="DH")
            ax.set_title(f"{m.upper()}\n{label[:10]}", fontsize=7)
            ax.set_xlim(0, 21)
            ax.set_xticks([0, 10, 21])
            if j == 0:
                ax.set_ylabel("mM", fontsize=8)
            if row_i == 0 and j == 0:
                ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
    fig.suptitle("HOBIC Metabolites: Literature vs FBA-informed (CH=solid, DH=dash)", fontsize=10)
    out2 = OUT_DIR / "heine_fba_vs_monod_mets.png"
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out2}")

    # Fig C: FBA stoichiometry comparison bar chart
    fig, axes = plt.subplots(1, len(SPECIES), figsize=(14, 4), constrained_layout=True)
    for ax, sp in zip(axes, SPECIES):
        prim, lit = lit_ratios[sp]
        sp_fba = fba_stoich.get(sp, {})
        mets_all = sorted(set(lit.keys()) | set(sp_fba.keys()))
        x = np.arange(len(mets_all))
        lit_vals = [lit.get(m, 0.0) for m in mets_all]
        fba_vals = [min(sp_fba.get(m, 0.0), 3.0) for m in mets_all]
        ax.bar(x - 0.2, lit_vals, 0.35, label="Literature", color="steelblue", alpha=0.8)
        ax.bar(x + 0.2, fba_vals, 0.35, label="AGORA FBA", color="coral", alpha=0.8)
        ax.set_title(f"{sp}", fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(mets_all, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(f"mmol / mmol {prim}", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        if sp == "So":
            ax.legend(fontsize=7)
    fig.suptitle(
        "Secretion stoichiometry: Literature vs AGORA FBA\n"
        "(AGORA caps at 3.0; FBA values reflect full medium incl. amino acid catabolism)",
        fontsize=10,
    )
    out3 = OUT_DIR / "heine_fba_stoichiometry.png"
    fig.savefig(out3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out3}")

    print("\nDone.")


if __name__ == "__main__":
    main()
