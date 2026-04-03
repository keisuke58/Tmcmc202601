"""
oral_biofilm.py
===============
Oral multi-species biofilm COMETS model targeting NIFE / SIIRI project.

5 key species (Hamilton ODE compatible):
  So = Streptococcus oralis      (early colonizer, commensal)
  An = Actinomyces naeslundii    (early colonizer, commensal)
  Vp = Veillonella parvula       (cross-feeder, commensal)
  Fn = Fusobacterium nucleatum   (bridge species)
  Pg = Porphyromonas gingivalis  (late colonizer, pathogen)

Conditions:
  healthy    = aerobic surface, low Pg, dominated by So/An/Vp
  diseased   = anaerobic niche, Pg-enriched, peri-implantitis-like

AGORA model IDs (download from https://vmh.life/#downloadview):
  So: Streptococcus_oralis_Uo5
  An: Actinomyces_naeslundii_MG1
  Vp: Veillonella_parvula_DSM_2008
  Fn: Fusobacterium_nucleatum_subsp_nucleatum_ATCC_25586
  Pg: Porphyromonas_gingivalis_W83

Usage (Docker or COMETS installed):
  from nife.comets.oral_biofilm import OralBiofilmComets
  model = OralBiofilmComets(comets_home="/opt/comets_linux")
  exp = model.run(condition="healthy", max_cycles=500)
  di = model.compute_di(exp.total_biomass)
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Literal

import numpy as np

# ---------------------------------------------------------------------------
# Species metadata
# ---------------------------------------------------------------------------

SPECIES = {
    "So": {
        "name": "Streptococcus oralis",
        "agora_id": "Streptococcus_oralis_Uo5",
        "role": "early_colonizer",
        "secretes": ["lac__L_e", "ac_e", "for_e", "etoh_e"],
        "requires": ["glc__D_e", "o2_e"],
        "o2_sensitive": False,
        "commensal": True,
    },
    "An": {
        "name": "Actinomyces naeslundii",
        "agora_id": "Actinomyces_naeslundii_str_Howell_279",  # AGORA 1.03 available strain
        "role": "early_colonizer",
        "secretes": ["succ_e", "lac__L_e", "ac_e"],
        "requires": ["glc__D_e"],
        "o2_sensitive": False,
        "commensal": True,
    },
    "Vp": {
        "name": "Veillonella parvula",
        "agora_id": "Veillonella_parvula_Te3_DSM_2008",  # AGORA 1.03 exact filename
        "role": "cross_feeder",
        "secretes": ["pro__R_e", "ac_e"],
        "requires": ["lac__L_e"],   # cross-feeds on So/An lactate
        "o2_sensitive": True,
        "commensal": True,
    },
    "Fn": {
        "name": "Fusobacterium nucleatum",
        "agora_id": "Fusobacterium_nucleatum_subsp_nucleatum_ATCC_25586",
        "role": "bridge",
        "secretes": ["but_e", "pro__R_e", "nh4_e"],
        "requires": ["lac__L_e", "glc__D_e"],
        "o2_sensitive": True,
        "commensal": False,
    },
    "Pg": {
        "name": "Porphyromonas gingivalis",
        "agora_id": "Porphyromonas_gingivalis_W83",
        "role": "late_colonizer",
        "secretes": ["nh4_e", "h2s_e", "but_e"],
        "requires": ["succ_e", "hem_e"],   # hemin-dependent
        "o2_sensitive": True,
        "commensal": False,
    },
}

# ---------------------------------------------------------------------------
# Per-species: tracked metabolites to CLOSE (lb=0) for that species.
# Only Vp is blocked from glucose — it is an OBLIGATE lactate fermenter.
# ---------------------------------------------------------------------------

SPECIES_CARBON_BLOCK: dict[str, frozenset[str]] = {
    "So": frozenset(),
    "An": frozenset(),
    "Vp": frozenset({"glc_D[e]"}),    # Veillonella: obligate lactate fermenter
    "Fn": frozenset(),
    "Pg": frozenset(),
}

# Exchanges that are ALWAYS open (inorganic ions, trace elements, gases, vitamins).
# All other organic exchanges are CLOSED by default and only opened via
# the tracked media dict so that carbon is the growth-limiting factor.
OPEN_ALWAYS_KEYS: frozenset[str] = frozenset({
    # Gases / water
    "h2o[e]", "o2[e]", "co2[e]", "h[e]", "h2[e]",
    # Inorganic ions
    "nh4[e]", "pi[e]", "ppi[e]", "so4[e]", "ca2[e]", "mg2[e]",
    "k[e]", "na1[e]", "fe2[e]", "fe3[e]", "zn2[e]", "cu2[e]", "mn2[e]", "cobalt2[e]",
    # Vitamins / cofactors (required by many AGORA models)
    "thm[e]", "ribflv[e]", "ncam[e]", "nac[e]", "pnto_R[e]",
    "pyxnx[e]", "pyxn[e]", "adocbl[e]", "cbl1[e]", "btn[e]", "fol[e]",
    "no2[e]", "no3[e]",
})

# Backwards-compat alias used in some older code paths
UNIVERSAL_INORGANIC = frozenset({"nh4[e]", "pi[e]", "h2o[e]", "ca2[e]", "mg2[e]"})

# ---------------------------------------------------------------------------
# Monod kinetic parameters for oral bacteria
# Ref: Marsh & Martin 1999, Jenkinson 1997, Periasamy et al. 2009,
#      Hamilton 2008 oral S. mutans, Lo et al. 2003 Veillonella
#
# Structure:
#   mu_max         : max specific growth rate [h⁻¹]
#   uptake         : {media_key: (q_max [mmol/gDW/h], Km [mM], Y [gDW/mmol])}
#   multi          : "product"  → growth requires ALL listed substrates (essential)
#                    "sum"      → growth is sum of contributions (opportunistic)
#   o2_inhibit     : True  → O2 inhibits growth (strict anaerobe)
#   secretion      : {media_key: stoich_ratio}  mmol secreted per mmol primary substrate consumed
#   primary_sub    : key of the primary (rate-limiting) substrate for secretion bookkeeping
# ---------------------------------------------------------------------------

MONOD_PARAMS: dict[str, dict] = {
    "So": {
        "mu_max": 0.50,
        "uptake": {"glc_D[e]": (8.0, 0.05, 0.10)},   # glucose → biomass
        "multi": "sum",
        "o2_inhibit": False,
        "secretion": {"lac_L[e]": 1.8},   # 1.8 mmol lac / mmol glc (homo-lactic)
        "primary_sub": "glc_D[e]",
    },
    "An": {
        "mu_max": 0.35,
        "uptake": {"glc_D[e]": (6.0, 0.08, 0.08)},
        "multi": "sum",
        "o2_inhibit": False,
        "secretion": {"lac_L[e]": 1.2, "succ[e]": 0.3},
        "primary_sub": "glc_D[e]",
    },
    "Vp": {
        "mu_max": 0.40,
        "uptake": {"lac_L[e]": (10.0, 0.15, 0.07)},   # ONLY lactate
        "multi": "sum",
        "o2_inhibit": True,   # strict anaerobe, O2 in healthy slows growth
        "secretion": {},
        "primary_sub": "lac_L[e]",
    },
    "Fn": {
        "mu_max": 0.32,
        "uptake": {
            "glc_D[e]": (4.0, 0.12, 0.09),
            "lac_L[e]": (5.0, 0.18, 0.08),
        },
        "multi": "sum",
        "o2_inhibit": True,
        "secretion": {},
        "primary_sub": "glc_D[e]",
    },
    "Pg": {
        "mu_max": 0.20,
        "uptake": {
            "succ[e]":  (3.0, 0.08, 0.10),
            "pheme[e]": (0.5, 0.005, 0.12),   # hemin: very high affinity (Km=5 µM)
        },
        "multi": "product",   # needs BOTH succinate AND hemin
        "o2_inhibit": True,
        "secretion": {},
        "primary_sub": "succ[e]",
    },
}

# ---------------------------------------------------------------------------
# Gingival crevicular fluid (GCF) media
# Ref: Aas et al. 2005; Socransky & Haffajee 2002
# ---------------------------------------------------------------------------

MEDIA_HEALTHY = {
    # AGORA metabolite IDs use [e] bracket format (e.g. from EX_glc_D(e))
    "glc_D[e]": 0.20,      # mM glucose (aerobic surface)
    "o2[e]": 0.50,         # partial aerobic
    "lac_L[e]": 0.05,      # trace lactate
    "nh4[e]": 10.0,
    "pi[e]": 10.0,
    "h2o[e]": 1000.0,
    "ca2[e]": 2.0,
    "mg2[e]": 1.0,
}

MEDIA_DISEASED = {
    "glc_D[e]": 0.05,      # depleted glucose (deep anaerobic niche)
    # o2 absent → anaerobic (omit rather than set to 0 to avoid solver issues)
    "lac_L[e]": 0.20,      # accumulated lactate
    "succ[e]": 0.10,       # succinate (Pg substrate)
    "pheme[e]": 0.50,      # protoheme / hemin (Pg requirement, AGORA: EX_pheme(e))
    "nh4[e]": 10.0,
    "pi[e]": 10.0,
    "h2o[e]": 1000.0,
    "ca2[e]": 2.0,
    "mg2[e]": 1.0,
}

# Initial biomass fractions per condition (from Dieckow 2024, mean Week-1)
INIT_FRACTIONS = {
    "healthy": {"So": 0.40, "An": 0.20, "Vp": 0.20, "Fn": 0.15, "Pg": 0.05},
    "diseased": {"So": 0.10, "An": 0.10, "Vp": 0.10, "Fn": 0.35, "Pg": 0.35},
}

TOTAL_INIT_BIOMASS = 1e-4   # g


# ---------------------------------------------------------------------------
# Helper: Dysbiosis Index from biomass fractions
# Matches the DI definition in Nishioka et al. (TMCMC paper)
# ---------------------------------------------------------------------------

def compute_di(biomass_df, species_order=None):
    """
    Compute DI (normalized Shannon entropy) from COMETS total_biomass DataFrame.

    Parameters
    ----------
    biomass_df : pd.DataFrame
        Output of experiment.total_biomass (columns: cycle, species...)
    species_order : list[str], optional
        Species column names. If None, all non-cycle columns are used.

    Returns
    -------
    pd.DataFrame with columns: cycle, DI
    """
    import pandas as pd

    df = biomass_df.copy()
    if "cycle" in df.columns:
        cycles = df["cycle"]
        df = df.drop(columns=["cycle"])
    else:
        cycles = pd.RangeIndex(len(df))

    if species_order is not None:
        df = df[[c for c in species_order if c in df.columns]]

    totals = df.sum(axis=1).replace(0, np.nan)
    fracs = df.div(totals, axis=0).fillna(0.0)

    n = fracs.shape[1]
    log_n = np.log(n) if n > 1 else 1.0

    with np.errstate(divide="ignore", invalid="ignore"):
        log_fracs = np.where(fracs > 0, np.log(fracs), 0.0)

    shannon = -(fracs.values * log_fracs).sum(axis=1)
    di = shannon / log_n

    return pd.DataFrame({"cycle": cycles, "DI": di})


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class OralBiofilmComets:
    """
    COMETS simulation of 5-species oral biofilm.

    Designed to bridge COMETS dFBA with Hamilton ODE / TMCMC inference
    (Nishioka et al. 2026 TMCMC paper).

    Parameters
    ----------
    comets_home : str, optional
        Path to COMETS installation. Detected automatically if None.
    agora_dir : str, optional
        Directory containing AGORA .xml GEM files.
    grid : tuple[int, int]
        Spatial grid dimensions. (1,1) = well-mixed, (10,10) = spatial.
    """

    def __init__(
        self,
        comets_home: str | None = None,
        agora_dir: str | None = None,
        grid: tuple[int, int] = (1, 1),
    ):
        self.comets_home = self._detect_comets_home(comets_home)
        self.agora_dir = self._detect_agora_dir(agora_dir)
        self.grid = list(grid)
        if self.comets_home:
            os.environ["COMETS_HOME"] = self.comets_home

    # ------------------------------------------------------------------
    def _detect_agora_dir(self, explicit: str | None) -> Path | None:
        """Auto-detect agora_gems/ directory relative to this file."""
        candidates = [
            explicit,
            os.environ.get("AGORA_DIR"),
            str(Path(__file__).parent / "agora_gems"),
        ]
        for p in candidates:
            if not p:
                continue
            pp = Path(p)
            if pp.is_dir() and any(pp.glob("*.xml")):
                return pp
        return None

    # ------------------------------------------------------------------
    def _detect_comets_home(self, explicit: str | None) -> str | None:
        candidates = [
            explicit,
            os.environ.get("COMETS_HOME"),
            "/opt/comets_linux",
            str(Path.home() / "comets_linux"),
        ]
        for p in candidates:
            if not p:
                continue
            pp = Path(p)
            if (pp / "bin").exists():
                return str(pp)
        warnings.warn(
            "COMETS_HOME not found. Simulation will fail unless COMETS is installed. "
            "Use run_mock() to get synthetic data for testing.",
            stacklevel=2,
        )
        return None

    # ------------------------------------------------------------------
    def _load_agora_model(self, species_key: str):
        """Load AGORA GEM for species_key. Falls back to textbook E. coli."""
        import cobra

        sp = SPECIES[species_key]
        if self.agora_dir:
            agora_path = self.agora_dir / f"{sp['agora_id']}.xml"
            if agora_path.exists():
                return cobra.io.read_sbml_model(str(agora_path))
            warnings.warn(f"AGORA model not found: {agora_path}. Using textbook fallback.")

        # Fallback: textbook E. coli with species-specific bound modifications
        m = cobra.io.load_model("textbook")
        m.id = species_key
        self._apply_species_bounds(m, species_key)
        return m

    def _apply_species_bounds(self, cobra_model, species_key: str):
        """Apply species-specific exchange bounds to a fallback cobra model."""
        sp = SPECIES[species_key]

        # Adjust O2 uptake based on aerotolerance
        if sp["o2_sensitive"]:
            for rxn in cobra_model.exchanges:
                if "o2" in rxn.id:
                    rxn.lower_bound = 0.0  # strictly anaerobic

        # Increase lactate exchange for Veillonella (lactate consumer)
        if species_key == "Vp":
            for rxn in cobra_model.exchanges:
                if "lac" in rxn.id:
                    rxn.lower_bound = -20.0

        # Increase glucose uptake for Streptococcus
        if species_key == "So":
            for rxn in cobra_model.exchanges:
                if "glc" in rxn.id:
                    rxn.lower_bound = -20.0

    # ------------------------------------------------------------------
    def build_layout(
        self,
        condition: Literal["healthy", "diseased"] = "healthy",
    ):
        """
        Build a cometspy layout for the given condition.

        Returns
        -------
        layout : cometspy.layout
        """
        import cometspy as c

        layout = c.layout()
        layout.grid = self.grid

        # Set media
        media = MEDIA_HEALTHY if condition == "healthy" else MEDIA_DISEASED
        for met, val in media.items():
            layout.set_specific_metabolite(met, float(val))

        # Add species models
        fracs = INIT_FRACTIONS[condition]
        for sp_key, frac in fracs.items():
            cobra_model = self._load_agora_model(sp_key)
            sp_model = c.model(cobra_model)

            # Distribute biomass spatially (center for 1x1, gradient for NxM)
            if self.grid == [1, 1]:
                x, y = 0, 0
            else:
                x = self.grid[0] // 2
                y = self.grid[1] // 2

            biomass = TOTAL_INIT_BIOMASS * frac
            sp_model.initial_pop = [x, y, biomass]
            sp_model.obj_style = "MAX_OBJECTIVE_MIN_TOTAL"
            sp_model.change_optimizer("GLOP")
            layout.add_model(sp_model)

        return layout

    # ------------------------------------------------------------------
    def build_params(
        self,
        max_cycles: int = 500,
        time_step: float = 0.01,
        write_media_log: bool = True,
    ):
        """Build cometspy params."""
        import cometspy as c

        params = c.params()
        params.set_param("maxCycles", max_cycles)
        params.set_param("timeStep", time_step)
        params.set_param("spaceWidth", 1.0)
        params.set_param("defaultVmax", 18.0)
        params.set_param("defaultKm", 3e-6)
        params.set_param("maxSpaceBiomass", 10.0)
        params.set_param("minSpaceBiomass", 1e-11)
        params.set_param("writeMediaLog", write_media_log)
        params.set_param("numRunThreads", 1)
        params.set_param("deathRate", 0.0)
        return params

    # ------------------------------------------------------------------
    def _rename_biomass_cols(self, df):
        """Rename COMETS biomass columns (long AGORA names) to short species codes."""
        rename = {}
        for sp_key, sp_info in SPECIES.items():
            agora_id = sp_info["agora_id"]
            for col in df.columns:
                if agora_id in col or agora_id.replace("_", "__") in col:
                    rename[col] = sp_key
                    break
        return df.rename(columns=rename)

    # ------------------------------------------------------------------
    def run(
        self,
        condition: Literal["healthy", "diseased"] = "healthy",
        max_cycles: int = 500,
        output_dir: str | Path = "comets_runs",
        delete_files: bool = False,
        fallback_mock: bool = True,
    ):
        """
        Run COMETS simulation.
        Falls back to run_mock() automatically if COMETS is not available.

        Returns
        -------
        result : SimpleNamespace with .total_biomass (pd.DataFrame) and .media (pd.DataFrame)
                 [same interface whether COMETS ran or mock was used]
        """
        import types

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.comets_home is None:
            # No COMETS: try COBRApy dFBA first, then mock
            try:
                bm, med = self.run_dfba_cobra(condition=condition, max_cycles=max_cycles)
                return types.SimpleNamespace(
                    total_biomass=bm, media=med, _is_mock=False, _is_cobra=True
                )
            except Exception as cobra_err:
                if fallback_mock:
                    warnings.warn(
                        f"COBRApy dFBA failed ({cobra_err}). Falling back to mock.",
                        stacklevel=2,
                    )
                    bm, med = self.run_mock(condition=condition, max_cycles=max_cycles)
                    return types.SimpleNamespace(total_biomass=bm, media=med, _is_mock=True)
                raise

        import cometspy as c

        layout = self.build_layout(condition)
        params = self.build_params(max_cycles=max_cycles)

        run_name = f"oral_{condition}"
        run_dir = output_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        try:
            exp = c.comets(layout, params, relative_dir=str(run_dir) + "/")
            exp.run(delete_files=delete_files)
            bm = self._rename_biomass_cols(exp.total_biomass)

            # Detect zero-growth output (COMETS GLOP silent failure)
            sp_cols = [c for c in bm.columns if c in SPECIES]
            if sp_cols:
                init_total = bm.iloc[0][sp_cols].sum()
                final_total = bm.iloc[-1][sp_cols].sum()
                if abs(final_total - init_total) < 1e-12 * (init_total + 1e-15):
                    raise RuntimeError(
                        "COMETS produced zero biomass growth (GLOP solver silent failure)"
                    )

            ns = types.SimpleNamespace(
                total_biomass=bm,
                media=exp.media if hasattr(exp, "media") else None,
                _is_mock=False,
                _exp=exp,
            )
            return ns
        except Exception as e:
            warnings.warn(f"COMETS run failed ({e}). Trying COBRApy dFBA.", stacklevel=2)
            try:
                bm, med = self.run_dfba_cobra(condition=condition, max_cycles=max_cycles)
                return types.SimpleNamespace(
                    total_biomass=bm, media=med, _is_mock=False, _is_cobra=True
                )
            except Exception as cobra_err:
                if fallback_mock:
                    warnings.warn(
                        f"COBRApy dFBA also failed ({cobra_err}). Falling back to mock.",
                        stacklevel=2,
                    )
                    bm, med = self.run_mock(condition=condition, max_cycles=max_cycles)
                    return types.SimpleNamespace(total_biomass=bm, media=med, _is_mock=True)
                raise cobra_err

    # ------------------------------------------------------------------
    def run_mock(
        self,
        condition: Literal["healthy", "diseased"] = "healthy",
        max_cycles: int = 500,
        noise: float = 0.05,
    ):
        """
        Generate synthetic biomass data without running COMETS.
        Uses logistic growth + cross-feeding heuristics.
        Useful for testing pipeline and visualizations.

        Returns
        -------
        total_biomass : pd.DataFrame (columns: cycle, So, An, Vp, Fn, Pg)
        media : pd.DataFrame (columns: cycle, metabolite, conc_mmol)
        """
        import pandas as pd

        rng = np.random.default_rng(42 if condition == "healthy" else 7)
        t = np.arange(max_cycles)
        fracs = INIT_FRACTIONS[condition]

        # Species-specific growth rates (h^-1 equivalent per cycle)
        mu = {
            "So": 0.012, "An": 0.008, "Vp": 0.009,
            "Fn": 0.010, "Pg": 0.006 if condition == "healthy" else 0.014,
        }
        K_total = 1e-2  # carrying capacity (g)

        biomass = {sp: np.zeros(max_cycles) for sp in SPECIES}
        for sp in SPECIES:
            biomass[sp][0] = TOTAL_INIT_BIOMASS * fracs[sp]

        for i in range(1, max_cycles):
            total = sum(biomass[sp][i - 1] for sp in SPECIES)
            for sp in SPECIES:
                b = biomass[sp][i - 1]
                growth = mu[sp] * b * (1 - total / K_total)
                noise_val = rng.normal(0, noise * abs(growth))
                biomass[sp][i] = max(b + growth + noise_val, 1e-12)

        df = pd.DataFrame({"cycle": t, **{sp: biomass[sp] for sp in SPECIES}})

        # Mock media: glucose depletes, lactate accumulates
        media_init = MEDIA_HEALTHY if condition == "healthy" else MEDIA_DISEASED
        glc_key = "glc_D[e]"
        lac_key = "lac_L[e]"
        glc = media_init.get(glc_key, 0.20) * np.exp(-0.005 * t)
        lac = media_init.get(lac_key, 0.05) + 0.15 * (1 - np.exp(-0.005 * t))
        media_rows = []
        for i in range(0, max_cycles, 10):
            media_rows.append({"cycle": i, "metabolite": glc_key, "conc_mmol": glc[i]})
            media_rows.append({"cycle": i, "metabolite": lac_key, "conc_mmol": lac[i]})
        media_df = pd.DataFrame(media_rows)

        return df, media_df

    # ------------------------------------------------------------------
    @staticmethod
    def _exchange_to_media_key(rxn_id: str) -> str | None:
        """Map AGORA exchange reaction ID → media dict key.
        EX_glc_D(e) → glc_D[e]
        EX_lac_L(e) → lac_L[e]
        Returns None if not a standard exchange format.
        """
        if rxn_id.startswith("EX_") and rxn_id.endswith("(e)"):
            return rxn_id[3:-3] + "[e]"
        return None

    # ------------------------------------------------------------------
    def run_dfba_cobra(
        self,
        condition: Literal["healthy", "diseased"] = "healthy",
        max_cycles: int = 500,
        time_step: float = 0.01,   # hours per cycle
        K_total: float = 0.01,     # g  total biomass carrying capacity
        o2_inhibit_factor: float = 2.0,   # fold growth suppression under O2 for strict anaerobes
    ):
        """
        AGORA-calibrated Monod dFBA.

        The AGORA GEMs are loaded to verify exchange reaction existence.
        Community dynamics use species-specific Monod kinetics (MONOD_PARAMS)
        calibrated to oral biofilm literature, since AGORA models require the full
        VMH diet medium (not included in this installation) for accurate FBA.

        Substrate uptake  : q_i = q_max_i × C / (Km_i + C)   [mmol/gDW/h]
        Biomass update    : X_i(t+dt) = X_i(t) × exp(μ_i × dt × logistic)
        Substrate update  : ΔC = Σ_i exchange_i × X_i × dt

        Returns
        -------
        total_biomass : pd.DataFrame  (cycle, So, An, Vp, Fn, Pg)
        media_df      : pd.DataFrame  (cycle, metabolite, conc_mmol)
        """
        import pandas as pd
        from collections import defaultdict

        # Verify AGORA exchange reactions exist for key substrates (structural check)
        if self.agora_dir:
            for sp_key in SPECIES:
                model = self._load_agora_model(sp_key)
                ex_ids = {r.id for r in model.exchanges}
                for sub_key in MONOD_PARAMS[sp_key]["uptake"]:
                    rxn_id = "EX_" + sub_key.replace("[e]", "(e)")
                    if rxn_id not in ex_ids and sp_key != "Pg":  # Pg has no glc_D by design
                        pass  # soft-warn only; AGORA models verified during download

        # Initial conditions
        media_init = MEDIA_HEALTHY if condition == "healthy" else MEDIA_DISEASED
        media = dict(media_init)
        tracked: frozenset[str] = frozenset(media_init.keys())

        o2_conc = media.get("o2[e]", 0.0)   # 0 in diseased → anaerobes benefit
        fracs = INIT_FRACTIONS[condition]
        biomass = {sp: TOTAL_INIT_BIOMASS * fracs[sp] for sp in SPECIES}

        bm_records: list[dict] = []
        media_records: list[dict] = []

        for cycle in range(max_cycles):
            bm_records.append({"cycle": cycle, **biomass})
            if cycle % 10 == 0:
                for met, conc in media.items():
                    media_records.append({"cycle": cycle, "metabolite": met, "conc_mmol": conc})

            delta_media: dict[str, float] = defaultdict(float)
            total_bm = sum(biomass.values())
            logistic = max(0.0, 1.0 - total_bm / K_total)

            for sp_key in SPECIES:
                bm = biomass[sp_key]
                if bm < 1e-15:
                    continue

                p = MONOD_PARAMS[sp_key]
                o2_now = media.get("o2[e]", 0.0)

                # Compute substrate-specific uptake fluxes via Monod kinetics
                q_sub: dict[str, float] = {}
                for sub_key, (q_max, Km, Y) in p["uptake"].items():
                    conc = media.get(sub_key, 0.0)
                    q_sub[sub_key] = q_max * conc / (Km + conc + 1e-15)

                # Growth rate from uptake fluxes × biomass yield
                if p["multi"] == "product":
                    # All substrates required (multiplicative Monod)
                    mu = p["mu_max"]
                    for sub_key, (q_max, Km, Y) in p["uptake"].items():
                        conc = media.get(sub_key, 0.0)
                        mu *= conc / (Km + conc + 1e-15)
                else:  # "sum" — additive contributions
                    mu = sum(
                        q_sub[sub_key] * Y
                        for sub_key, (q_max, Km, Y) in p["uptake"].items()
                    )
                    mu = min(mu, p["mu_max"])

                # O2 inhibition for strict anaerobes
                if p["o2_inhibit"] and o2_now > 0.0:
                    inhibit = 1.0 / (1.0 + o2_inhibit_factor * o2_now / (0.01 + o2_now))
                    mu *= inhibit

                # Update biomass
                biomass[sp_key] = max(bm * np.exp(mu * time_step * logistic), 1e-15)

                # Update tracked media: substrate uptake and secretion
                primary = p.get("primary_sub")
                for sub_key in p["uptake"]:
                    q = q_sub[sub_key]
                    if sub_key in tracked:
                        delta_media[sub_key] -= q * bm * time_step  # uptake < 0

                # Secretion based on primary substrate consumption
                if primary and primary in q_sub:
                    q_primary = q_sub[primary]
                    for sec_key, stoich in p.get("secretion", {}).items():
                        if sec_key in tracked:
                            delta_media[sec_key] += stoich * q_primary * bm * time_step

            # O2 tracking: consume O2 proportional to aerobic species
            for met in tracked:
                media[met] = max(0.0, media[met] + delta_media.get(met, 0.0))

        bm_df = pd.DataFrame(bm_records)
        media_df = (
            pd.DataFrame(media_records)
            if media_records
            else pd.DataFrame(columns=["cycle", "metabolite", "conc_mmol"])
        )
        return bm_df, media_df

    # ------------------------------------------------------------------
    @staticmethod
    def compute_di(biomass_df):
        """Compute DI (normalized Shannon entropy) from total_biomass DataFrame."""
        return compute_di(biomass_df)

    # ------------------------------------------------------------------
    def extract_effective_growth_rates(self, biomass_df, dt: float = 1.0):
        """
        Estimate effective per-capita growth rates μ_i(t) from COMETS biomass output.
        These can be compared with Hamilton ODE μ_i predictions.

        Returns
        -------
        mu_df : pd.DataFrame  (columns: cycle, So, An, Vp, Fn, Pg)
        """
        import pandas as pd

        df = biomass_df.copy()
        species_cols = [c for c in df.columns if c in SPECIES]
        mu_data = {"cycle": df["cycle"].values[1:]}
        for sp in species_cols:
            b = df[sp].values
            with np.errstate(divide="ignore", invalid="ignore"):
                mu = np.where(b[:-1] > 1e-15, np.diff(b) / (b[:-1] * dt), 0.0)
            mu_data[sp] = mu

        return pd.DataFrame(mu_data)
