"""
tmcmc_bridge.py
===============
Bridge between Hamilton ODE / TMCMC posterior and COMETS dFBA parameters.

Key idea
--------
The Hamilton ODE interaction matrix A encodes net growth effects:
    dφ_i/dt = φ_i (a_ii + Σ_j a_ij φ_j)

In COMETS, these translate to exchange reaction bounds (cross-feeding fluxes):
    a_ij > 0  ←→  species j secretes a metabolite that species i can uptake
    a_ij < 0  ←→  competition for a shared resource

This module provides:
1.  TMCMCCoMETSBridge  — maps TMCMC posterior θ → COMETS exchange bounds
2.  compute_bayesian_di — DI uncertainty bands from posterior ensemble
3.  compare_ode_comets  — overlay Hamilton ODE vs COMETS trajectories

Reference
---------
Nishioka et al. 2026, "GPU-accelerated Bayesian inference of multi-species
biofilm interaction parameters via TMCMC"
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Sequence

import numpy as np

SPECIES_ORDER = ["So", "An", "Vp", "Fn", "Pg"]   # must match Hamilton ODE species index
N_SPECIES = len(SPECIES_ORDER)


# ---------------------------------------------------------------------------
# Parameter mapping: Hamilton A-matrix → COMETS exchange bounds
# ---------------------------------------------------------------------------

# Cross-feeding metabolites linking pairs of species
# key = (donor, receiver), value = exchange reaction ID in AGORA model
CROSS_FEED_MAP = {
    # AGORA exchange reaction IDs (EX_<met>(e) → used as bounds in cometspy)
    ("So", "Vp"): "EX_lac_L(e)",    # Streptococcus lactate → Veillonella
    ("An", "Vp"): "EX_lac_L(e)",    # Actinomyces lactate  → Veillonella
    ("Fn", "Pg"): "EX_succ(e)",     # Fusobacterium succinate → Pg
    ("Vp", "Fn"): "EX_pro_L(e)",    # Veillonella propionate → Fn
    ("So", "Fn"): "EX_ac(e)",       # Streptococcus acetate  → Fn
}


def hamilton_a_to_comets_bounds(
    A_matrix: np.ndarray,
    base_vmax: float = 18.0,
    scale: float = 0.5,
) -> dict[tuple[str, str], float]:
    """
    Map Hamilton ODE A-matrix to COMETS exchange reaction bounds.

    Parameters
    ----------
    A_matrix : (N, N) array
        Hamilton gLV interaction matrix (row = receiver, col = donor).
    base_vmax : float
        Default Vmax for exchange reactions.
    scale : float
        Scaling factor: bound = base_vmax * (1 + scale * |a_ij|)

    Returns
    -------
    bounds : dict {(donor_sp, receiver_sp): exchange_bound}
    """
    if A_matrix.shape != (N_SPECIES, N_SPECIES):
        raise ValueError(f"Expected ({N_SPECIES},{N_SPECIES}) A_matrix, got {A_matrix.shape}")

    bounds = {}
    for (donor, receiver), ex_rxn in CROSS_FEED_MAP.items():
        i = SPECIES_ORDER.index(receiver)
        j = SPECIES_ORDER.index(donor)
        a_ij = A_matrix[i, j]
        if a_ij > 0:
            # positive interaction → enhance uptake bound
            bound = base_vmax * (1 + scale * a_ij)
        else:
            # negative interaction → restrict exchange
            bound = max(base_vmax * (1 + scale * a_ij), 0.0)
        bounds[(donor, receiver)] = (ex_rxn, bound)

    return bounds


# ---------------------------------------------------------------------------
# TMCMC posterior → COMETS ensemble
# ---------------------------------------------------------------------------

class TMCMCCoMETSBridge:
    """
    Run COMETS simulations parametrized by TMCMC posterior samples.

    Parameters
    ----------
    oral_model : OralBiofilmComets
        COMETS model instance.
    posterior_samples : (M, N_params) array
        TMCMC posterior θ samples. Each row is one sample.
    param_index : dict
        Maps parameter name to column index in posterior_samples.
        Minimum required: 'A' block (indices for the N×N interaction matrix).
    """

    def __init__(self, oral_model, posterior_samples: np.ndarray, param_index: dict):
        self.model = oral_model
        self.samples = posterior_samples
        self.idx = param_index

    # ------------------------------------------------------------------
    def _extract_A_matrix(self, theta: np.ndarray) -> np.ndarray:
        """Extract 5×5 gLV A-matrix from parameter vector θ."""
        A = np.zeros((N_SPECIES, N_SPECIES))
        for i, sp_i in enumerate(SPECIES_ORDER):
            for j, sp_j in enumerate(SPECIES_ORDER):
                key = f"a_{sp_i}_{sp_j}"
                if key in self.idx:
                    A[i, j] = theta[self.idx[key]]
        return A

    # ------------------------------------------------------------------
    def run_ensemble(
        self,
        condition: str = "healthy",
        n_samples: int = 30,
        max_cycles: int = 300,
        use_mock: bool = True,
    ) -> list[dict]:
        """
        Run COMETS (or mock) for n_samples TMCMC posterior draws.

        Parameters
        ----------
        condition : "healthy" | "diseased"
        n_samples : int
            Number of posterior samples to use.
        max_cycles : int
        use_mock : bool
            If True, use run_mock() (no COMETS Java required).

        Returns
        -------
        results : list of dict with keys: theta, total_biomass, di_df
        """
        from nife.comets.oral_biofilm import compute_di

        idx = np.random.choice(len(self.samples), size=min(n_samples, len(self.samples)), replace=False)
        results = []

        for k, i in enumerate(idx):
            theta = self.samples[i]
            A = self._extract_A_matrix(theta)
            bounds = hamilton_a_to_comets_bounds(A)

            if use_mock:
                # Perturb mock initial fractions by A-matrix diagonal (self-growth rates)
                from nife.comets.oral_biofilm import INIT_FRACTIONS, TOTAL_INIT_BIOMASS
                import copy
                fracs = copy.deepcopy(INIT_FRACTIONS[condition])
                # Modulate Pg fraction by a_Pg_Pg strength
                if "a_Pg_Pg" in self.idx:
                    pg_growth = theta[self.idx["a_Pg_Pg"]]
                    fracs["Pg"] = np.clip(fracs["Pg"] * (1 + 0.3 * pg_growth), 0.01, 0.80)
                    total = sum(fracs.values())
                    fracs = {k: v / total for k, v in fracs.items()}

                # Temporarily override model fracs
                import nife.comets.oral_biofilm as _ob
                orig = _ob.INIT_FRACTIONS[condition]
                _ob.INIT_FRACTIONS[condition] = fracs
                biomass_df, media_df = self.model.run_mock(condition=condition, max_cycles=max_cycles, noise=0.03)
                _ob.INIT_FRACTIONS[condition] = orig
            else:
                exp = self.model.run(condition=condition, max_cycles=max_cycles)
                biomass_df = exp.total_biomass
                media_df = exp.media if hasattr(exp, "media") else None

            di_df = compute_di(biomass_df)
            results.append({"theta": theta, "A": A, "bounds": bounds,
                             "total_biomass": biomass_df, "di": di_df})

        return results

    # ------------------------------------------------------------------
    @staticmethod
    def compute_di_bands(results: list[dict], q_low: float = 0.05, q_high: float = 0.95):
        """
        Compute DI credible interval bands from ensemble.

        Returns
        -------
        dict with keys: cycles, di_median, di_low, di_high
        """
        import pandas as pd

        di_arrays = []
        n_cycles = None
        for r in results:
            di_vals = r["di"]["DI"].values
            if n_cycles is None:
                n_cycles = len(di_vals)
            di_arrays.append(di_vals[:n_cycles])

        arr = np.stack(di_arrays, axis=0)   # (n_samples, n_cycles)
        cycles = results[0]["di"]["cycle"].values[:n_cycles]
        return {
            "cycles": cycles,
            "di_median": np.median(arr, axis=0),
            "di_low": np.quantile(arr, q_low, axis=0),
            "di_high": np.quantile(arr, q_high, axis=0),
        }


# ---------------------------------------------------------------------------
# Comparison: Hamilton ODE vs COMETS
# ---------------------------------------------------------------------------

def compare_ode_comets(ode_biomass: np.ndarray, comets_df, species_order=None):
    """
    Compute correlation between Hamilton ODE species fractions and
    COMETS total_biomass fractions.

    Parameters
    ----------
    ode_biomass : (T, N) array — ODE solution (time x species)
    comets_df : pd.DataFrame — COMETS total_biomass output
    species_order : list[str] — species column names in comets_df

    Returns
    -------
    dict with per-species Pearson r and RMSE
    """
    if species_order is None:
        species_order = SPECIES_ORDER

    comets_arr = comets_df[[s for s in species_order if s in comets_df.columns]].values
    # Normalize to fractions
    ode_frac = ode_biomass / ode_biomass.sum(axis=1, keepdims=True)
    comets_frac = comets_arr / comets_arr.sum(axis=1, keepdims=True)

    # Interpolate to same length
    T_ode = ode_frac.shape[0]
    T_com = comets_frac.shape[0]
    T_min = min(T_ode, T_com)
    t_ode = np.linspace(0, 1, T_ode)
    t_com = np.linspace(0, 1, T_com)
    t_common = np.linspace(0, 1, T_min)

    results = {}
    for k, sp in enumerate(species_order):
        ode_interp = np.interp(t_common, t_ode, ode_frac[:, k])
        com_interp = np.interp(t_common, t_com, comets_frac[:, k])
        r = float(np.corrcoef(ode_interp, com_interp)[0, 1])
        rmse = float(np.sqrt(np.mean((ode_interp - com_interp) ** 2)))
        results[sp] = {"r": r, "rmse": rmse}

    return results
