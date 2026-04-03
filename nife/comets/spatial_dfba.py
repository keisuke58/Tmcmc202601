#!/usr/bin/env python3
"""
spatial_dfba.py — 2D spatial Monod dFBA for implant biofilm (NIFE/SIIRI)
=========================================================================

Simulates a vertical cross-section of an implant biofilm:
  z = 0:     implant surface (Neumann, no-flux)
  z = NZ-1:  GCF/saliva reservoir (Dirichlet, fixed concentration)
  x:         lateral (periodic)

Nutrient solver: quasi-steady-state (QSS) — pre-factorized SuperLU.
  PDE:  D ∇²c = net_consumption  →  A_base * c = rhs / D_nut
  A_base is the same matrix for all nutrients (L with Dirichlet rows).
  SuperLU factorization is done once; forward/backward substitution per step.

Ground truth (Dieckow et al. 2024, npj Biofilms Microbiomes 10:155):
  - V(t):      Week1 ~ 2.5e5, Week2 ~ 8.0e5, Week3 ~ 1.8e6 µm³
  - f_live(t): 0.87 → 0.84 → 0.81
  - Composition: Streptococcus ~50%, Veillonella ~20%, Actinomyces ~12%

Species (7 genera, core implant biofilm, Dieckow 2024):
  Str = Streptococcus spp.       (dominant, glucose→lactate)
  Act = Actinomyces/Schaalia     (scaffolding, early colonizer)
  Vel = Veillonella spp.         (obligate lactate cross-feeder, anaerobe)
  Hae = Haemophilus parainfluenz (aerobic/facultative, NO3 reducer)
  Rot = Rothia spp.              (health-associated, aerobic)
  Fus = Fusobacterium spp.       (bridge species, anaerobe, amino-acid consumer)
  Por = Porphyromonas spp.       (late pathogen, deep anaerobe)

Nutrients: glc, o2, lac, aa, no3

CLI:
  python spatial_dfba.py              # run 3 weeks + plots
  python spatial_dfba.py --fit        # fit Monod params to Dieckow 2024
  python spatial_dfba.py --fast       # NZ=20×NX=10 quick check
  python spatial_dfba.py --plot-only  # reload saved history

Reference: Dukovski et al. 2021 (COMETS, Nat. Protocols); Stewart 2003 (biofilm D_eff);
           Dieckow et al. 2024 (npj Biofilms Microbiomes 10:155);
           Periasamy & Kolenbrander 2009; Marsh & Martin 1999.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import scipy.optimize as opt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=sp.SparseEfficiencyWarning)

# ── grid (overridden by --fast) ───────────────────────────────────────────────
DZ = DX = 10.0       # µm per voxel
NZ = 60              # depth voxels  →  600 µm
NX = 40              # lateral voxels →  400 µm
DT = 0.5             # h per growth step
HOURS_PER_WEEK = 168.0

# ── diffusion coefficients [µm²/h] ───────────────────────────────────────────
# D_eff in dense oral biofilm: Beyenal & Lewandowski 2002 (Biotechnol Bioeng 81:402)
#   D_eff/D_water ≈ 0.1–0.5 for glucose/lactate; O2 can be much lower in dense EPS.
# O2: 2e4 µm²/h = D_eff ≈ 0.26% D_water → creates steep gradient for anaerobic niche.
#   Physically justified: EPS matrix + dense cell packing in mature biofilm.
# glc/lac/aa: 2–5% D_water → moderate restriction.
D_NUT = {
    "glc":  1.5e5,   # µm²/h  (D_eff ≈ 7% D_water)
    "o2":   2.0e4,   # µm²/h  (D_eff ≈ 0.26% D_water; creates anaerobic niche at z=0)
    "lac":  1.2e5,   # µm²/h
    "aa":   1.0e5,   # µm²/h
    "no3":  1.2e5,   # µm²/h
}
NUTS = list(D_NUT.keys())

# ── boundary concentrations (GCF/saliva at z = NZ-1) ─────────────────────────
C_TOP_DEFAULT = {
    "glc":  0.10,    # mM
    "o2":   0.03,    # mM  (peri-implant sulcus, micro-aerobic)
    "lac":  0.05,    # mM
    "aa":   2.00,    # mM  (host-derived peptides in GCF)
    "no3":  0.05,    # mM
}

BM_MAX_DENSITY_DEFAULT = 2.0   # gDW/L  (Schlafer & Meyer 2017: 0.5–3 gDW/L in mature biofilm)
SPREAD_FRAC_DEFAULT    = 0.035

# ── species ───────────────────────────────────────────────────────────────────
SPECIES = ["Str", "Act", "Vel", "Hae", "Rot", "Fus", "Por"]
NAMES   = {
    "Str": "Streptococcus",  "Act": "Actinomyces/Schaalia",
    "Vel": "Veillonella",    "Hae": "Haemophilus",
    "Rot": "Rothia",         "Fus": "Fusobacterium",
    "Por": "Porphyromonas",
}
COLORS = {
    "Str": "#2196F3", "Act": "#4CAF50", "Vel": "#FF9800",
    "Hae": "#00BCD4", "Rot": "#8BC34A", "Fus": "#9C27B0", "Por": "#F44336",
}

# ── default Monod parameters ──────────────────────────────────────────────────
# mu_max [h⁻¹]; substrates: {nut: (q_max [mmol/gDW/h], Km [mM], Y [gDW/mmol])}
MONOD_DEFAULT: dict[str, dict[str, Any]] = {
    "Str": dict(mu_max=0.50,
                substrates={"glc": (8.0, 0.05, 0.06)},
                o2_inhibit=False, needs_o2=False, o2_inhib_factor=0.0,
                o2_aerobic_boost=True,
                secretion={"lac": 1.8}, primary="glc"),
    # Act: mu_max=0.26 → ~12% share (Dieckow 2024; Actinomyces is a slow grower)
    "Act": dict(mu_max=0.26,
                substrates={"aa":  (3.0, 0.40, 0.05),
                            "glc": (1.5, 0.20, 0.03)},
                o2_inhibit=False, needs_o2=False, o2_inhib_factor=0.0,
                o2_aerobic_boost=False,
                secretion={}, primary="aa"),
    # Vel: obligate anaerobe; Km_lac=0.15 → grows readily when lactate > 0.15 mM
    "Vel": dict(mu_max=0.30,
                substrates={"lac": (8.0, 0.15, 0.06)},
                o2_inhibit=True, needs_o2=False, o2_inhib_factor=6.0,
                o2_aerobic_boost=False,
                secretion={}, primary="lac"),
    "Hae": dict(mu_max=0.28,
                substrates={"glc": (3.0, 0.10, 0.04)},
                o2_inhibit=False, needs_o2=True, o2_inhib_factor=0.0,
                o2_aerobic_boost=False,
                secretion={}, primary="glc"),
    "Rot": dict(mu_max=0.18,
                substrates={"glc": (3.5, 0.07, 0.04),
                            "aa":  (1.5, 0.45, 0.04)},
                o2_inhibit=False, needs_o2=True, o2_inhib_factor=0.0,
                o2_aerobic_boost=False,
                secretion={}, primary="glc"),
    "Fus": dict(mu_max=0.22,
                substrates={"lac": (2.0, 0.25, 0.03),
                            "aa":  (5.0, 0.30, 0.05)},
                o2_inhibit=True, needs_o2=False, o2_inhib_factor=3.0,
                o2_aerobic_boost=False,
                secretion={}, primary="aa"),
    "Por": dict(mu_max=0.08,
                substrates={"aa": (5.0, 0.25, 0.06)},
                o2_inhibit=True, needs_o2=False, o2_inhib_factor=6.0,
                o2_aerobic_boost=False,
                secretion={}, primary="aa"),
}

# ── Dieckow 2024 ground truth ─────────────────────────────────────────────────
GT_WEEKS  = np.array([1.0, 2.0, 3.0])
GT_VOLUME = np.array([2.5e5, 8.0e5, 1.8e6])
GT_LIVE   = np.array([0.87,  0.84,  0.81])
GT_COMP   = {
    "Str": np.array([0.50, 0.48, 0.45]),
    "Act": np.array([0.10, 0.12, 0.13]),
    "Vel": np.array([0.18, 0.20, 0.22]),
    "Hae": np.array([0.08, 0.07, 0.06]),
    "Rot": np.array([0.06, 0.06, 0.05]),
    "Fus": np.array([0.03, 0.04, 0.05]),
    "Por": np.array([0.01, 0.01, 0.02]),
}
# Uncertainty from Dieckow Fig 3 (IQR over 12 patients)
GT_VOLUME_SD  = GT_VOLUME * 0.40       # ~40% CV
GT_LIVE_SD    = np.array([0.05, 0.05, 0.05])
GT_COMP_SD    = {sp: np.maximum(GT_COMP[sp] * 0.30, 0.02) for sp in SPECIES}


# ── solver infrastructure ─────────────────────────────────────────────────────

def build_laplacian_2d(nz: int, nx: int, dz: float, dx: float) -> sp.csr_matrix:
    """2D Laplacian with Neumann (z=0) + Dirichlet (z=nz-1) + periodic-x BCs."""
    n = nz * nx
    rows, cols, data = [], [], []

    for iz in range(nz):
        for ix in range(nx):
            i = iz * nx + ix
            if iz == nz - 1:
                # Dirichlet row: identity (RHS = c_top)
                rows.append(i); cols.append(i); data.append(1.0)
                continue

            # z-direction
            if iz == 0:
                # Neumann: ghost cell → forward difference only
                i_zp = (iz + 1) * nx + ix
                rows += [i, i, i]
                cols += [i, i, i_zp]
                data += [-1/dz**2, -1/dz**2, 2/dz**2]
            else:
                i_zm = (iz - 1) * nx + ix
                i_zp = (iz + 1) * nx + ix
                rows += [i, i, i]
                cols += [i, i_zm, i_zp]
                data += [-2/dz**2, 1/dz**2, 1/dz**2]

            # x-direction (periodic)
            ix_m = (ix - 1) % nx
            ix_p = (ix + 1) % nx
            rows += [i, i, i]
            cols += [iz * nx + ix_m, iz * nx + ix_p, i]
            data += [1/dx**2, 1/dx**2, -2/dx**2]

    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))


class NutrientSolver:
    """
    Pre-factorized QSS nutrient solver.

    The system matrix A_base = L with Dirichlet rows replaced by identity.
    A_base does NOT depend on D (D only scales the RHS).
    → Single SuperLU factorization reused for all nutrients and all time steps.

    System: A_base * c = rhs_nut
      interior: rhs_nut[i] = -uptake_rate[i] / D_nut   (L * c = -uptake/D)
      Dirichlet: rhs_nut[i] = c_top_nut
    """

    def __init__(self, nz: int, nx: int, dz: float, dx: float):
        self.nz = nz
        self.nx = nx
        L = build_laplacian_2d(nz, nx, dz, dx)
        # L already has Dirichlet rows as identity → directly factorize
        self._lu = spla.splu(L.tocsc())
        self._dirichlet_mask = np.zeros(nz * nx, dtype=bool)
        for ix in range(nx):
            self._dirichlet_mask[(nz - 1) * nx + ix] = True

    def solve(self, c_top: float, uptake_rate: np.ndarray, D: float) -> np.ndarray:
        """
        Solve QSS diffusion-reaction: D ∇²c = uptake_rate (positive = consumed).

        PDE:  D ∇²c = R  →  L * c = R / D  (L = discrete ∇², rhs > 0 for sinks)
        Result: concave-up profile, minimum at implant surface (z=0).

        uptake_rate > 0 → consumed (net sink)
        uptake_rate < 0 → produced (net source)
        """
        rhs = uptake_rate / D           # positive for sinks → ∇²c > 0 → concave-up ✓
        rhs[self._dirichlet_mask] = c_top
        c = self._lu.solve(rhs)
        return np.clip(c, 0.0, None).reshape(self.nz, self.nx)


# ── initial conditions ────────────────────────────────────────────────────────

def init_state(rng: np.random.Generator, solver: NutrientSolver,
               monod: dict, bm_max: float, c_top: dict,
               nz: int, nx: int) -> tuple[dict, dict]:
    """
    Week-1 equivalent established biofilm with O2/lactate zonation.
    Starts from a thin layer (~80 µm) matching Dieckow Week-1 properties.
    """
    INIT_THICKNESS = min(8, nz // 3)

    LAYER_BIAS = {
        "Str": [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8],
        "Act": [0.5, 0.7, 0.9, 1.1, 1.2, 1.3, 1.4, 1.4],
        "Vel": [2.5, 2.2, 1.8, 1.4, 1.0, 0.6, 0.3, 0.1],
        "Hae": [0.2, 0.4, 0.7, 1.0, 1.3, 1.5, 1.6, 1.5],
        "Rot": [0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.7, 1.8],
        "Fus": [3.0, 2.5, 1.8, 1.2, 0.8, 0.4, 0.2, 0.1],
        "Por": [4.0, 3.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.0],
    }

    biomass = {sp: np.zeros((nz, nx)) for sp in SPECIES}
    rho_base = bm_max * 0.60

    for iz in range(INIT_THICKNESS):
        rho_iz = rho_base * (1.0 - 0.025 * iz)
        bias_idx = min(iz, 7)
        raw = {sp: GT_COMP[sp][0] * LAYER_BIAS[sp][bias_idx] for sp in SPECIES}
        tot = sum(raw.values())
        fracs = {sp: raw[sp] / tot for sp in SPECIES}
        for sp in SPECIES:
            noise = 1.0 + 0.08 * rng.standard_normal(nx)
            biomass[sp][iz, :] = rho_iz * fracs[sp] * np.clip(noise, 0.8, 1.2)

    # 4 Picard iterations to get self-consistent initial nutrient fields
    nutrients = {nut: np.full((nz, nx), c_top[nut]) for nut in c_top}
    for _ in range(4):
        # uptake_fields > 0 = consumed (sink), < 0 = produced (source)
        uptake_fields = {k: np.zeros((nz, nx)) for k in c_top}
        for sp in SPECIES:
            p  = monod[sp]
            bm = biomass[sp]
            for nut, (q_max, Km, Y) in p["substrates"].items():
                c = nutrients[nut]
                uptake_fields[nut] += q_max * c / (Km + c + 1e-15) * bm   # +: consumed
            prim = p["primary"]
            for sec_nut, stoich in p.get("secretion", {}).items():
                c_p = nutrients[prim]
                q_p_max, Km_p, _ = p["substrates"][prim]
                uptake_fields[sec_nut] -= stoich * q_p_max * c_p / (Km_p + c_p + 1e-15) * bm  # -: produced
        for nut in NUTS:
            c_qss = solver.solve(c_top[nut], uptake_fields[nut].ravel(), D_NUT[nut])
            nutrients[nut] = 0.5 * nutrients[nut] + 0.5 * c_qss

    return nutrients, biomass


# ── main simulation loop ──────────────────────────────────────────────────────

def run(n_weeks: int = 3, seed: int = 42, verbose: bool = True,
        monod: dict | None = None, bm_max: float | None = None,
        spread_frac: float | None = None, c_top: dict | None = None,
        nz: int | None = None, nx: int | None = None,
        dt: float | None = None) -> list[dict]:
    """
    Run 2D spatial dFBA simulation.

    Parameters
    ----------
    monod        : Monod parameters dict (default: MONOD_DEFAULT)
    bm_max       : carrying capacity gDW/L (default: BM_MAX_DENSITY_DEFAULT)
    spread_frac  : shoving spread fraction (default: SPREAD_FRAC_DEFAULT)
    c_top        : boundary concentrations dict (default: C_TOP_DEFAULT)
    nz, nx       : grid size (default: NZ, NX)
    dt           : time step hours (default: DT)
    """
    monod = monod if monod is not None else MONOD_DEFAULT
    bm_max = bm_max if bm_max is not None else BM_MAX_DENSITY_DEFAULT
    sf = spread_frac if spread_frac is not None else SPREAD_FRAC_DEFAULT
    c_top = c_top if c_top is not None else C_TOP_DEFAULT
    _nz = nz if nz is not None else NZ
    _nx = nx if nx is not None else NX
    _dt = dt if dt is not None else DT

    rng    = np.random.default_rng(seed)
    solver = NutrientSolver(_nz, _nx, DZ, DX)
    nutrients, biomass = init_state(rng, solver, monod, bm_max, c_top, _nz, _nx)

    n_steps    = int(n_weeks * HOURS_PER_WEEK / _dt)
    save_every = max(1, int(24.0 / _dt))
    history: list[dict] = []

    t0 = time.time()
    for step in range(n_steps + 1):
        hour = step * _dt

        total_bm = sum(biomass[sp] for sp in SPECIES)
        logistic  = np.clip(1.0 - total_bm / bm_max, 0.0, 1.0)

        # ── 1. QSS nutrient convergence (4 Picard, α=0.7) ────────────────
        RELAX = 0.7
        for _iter in range(4):
            uptake_fields = {k: np.zeros((_nz, _nx)) for k in NUTS}
            for sp in SPECIES:
                p  = monod[sp]
                bm = biomass[sp]
                o2 = nutrients["o2"]

                # growth-rate estimate for O2 consumption scaling
                mu_est = np.zeros((_nz, _nx))
                for nut, (q_max, Km, Y) in p["substrates"].items():
                    c = nutrients[nut]
                    q = q_max * c / (Km + c + 1e-15)
                    mu_est += q * Y
                    uptake_fields[nut] += q * bm

                mu_est = np.minimum(mu_est, p["mu_max"])
                if p.get("o2_aerobic_boost"):
                    mu_est *= 0.45 + 0.55 * o2 / (0.05 + o2)
                if p["o2_inhibit"]:
                    mu_est *= 1.0 / (1.0 + p["o2_inhib_factor"] * o2 / (0.02 + o2))
                if p["needs_o2"]:
                    mu_est *= o2 / (0.05 + o2)

                # O2 consumption by aerotolerant species (stoich=3.0 mmol O2/gDW/h)
                # Ref: typical aerobic respiration O2 demand 2-5 mmol/gDW/h
                if not p["o2_inhibit"]:
                    uptake_fields["o2"] += (mu_est / max(p["mu_max"], 1e-9)) * bm * 3.0

                # lactate secretion by Str
                prim = p["primary"]
                for sec_nut, stoich in p.get("secretion", {}).items():
                    c_p = nutrients[prim]
                    q_p_max, Km_p, _ = p["substrates"][prim]
                    q_p = q_p_max * c_p / (Km_p + c_p + 1e-15)
                    uptake_fields[sec_nut] -= stoich * q_p * bm   # negative = produced

            for nut in NUTS:
                c_new = solver.solve(c_top[nut], uptake_fields[nut].ravel(), D_NUT[nut])
                nutrients[nut] = (1.0 - RELAX) * nutrients[nut] + RELAX * c_new

        # ── 2. Growth rates from converged nutrients ──────────────────────
        growth_rates: dict[str, np.ndarray] = {}
        for sp in SPECIES:
            p  = monod[sp]
            o2 = nutrients["o2"]

            mu = np.zeros((_nz, _nx))
            for nut, (q_max, Km, Y) in p["substrates"].items():
                c = nutrients[nut]
                q = q_max * c / (Km + c + 1e-15)
                mu += q * Y
            mu = np.minimum(mu, p["mu_max"])

            if p.get("o2_aerobic_boost"):
                c_lac = nutrients["lac"]
                pH = np.maximum(5.8, 7.4 - 0.8 * c_lac)
                acid_inhib = np.exp(-2.0 * np.maximum(0.0, 7.0 - pH))
                aerobic_f  = 0.45 + 0.55 * o2 / (0.05 + o2)
                mu *= aerobic_f * acid_inhib
            if p["o2_inhibit"]:
                mu *= 1.0 / (1.0 + p["o2_inhib_factor"] * o2 / (0.02 + o2))
            if p["needs_o2"]:
                mu *= o2 / (0.05 + o2)

            growth_rates[sp] = mu * logistic

        # ── 3. Biomass update ─────────────────────────────────────────────
        for sp in SPECIES:
            biomass[sp] = biomass[sp] * np.exp(growth_rates[sp] * _dt)

        total_new = sum(biomass[sp] for sp in SPECIES) + 1e-30
        overshoot = np.maximum(total_new / bm_max, 1.0)
        for sp in SPECIES:
            biomass[sp] /= overshoot

        # ── 4. Shoving / spreading ────────────────────────────────────────
        total_bm2 = sum(biomass[sp] for sp in SPECIES)
        saturated = total_bm2 > bm_max * 0.80

        for sp in SPECIES:
            bm    = biomass[sp]
            excess = bm * saturated * sf
            spread_up = np.zeros_like(excess)
            spread_up[1:, :] = excess[:-1, :] * 3.0
            spread_rx = np.roll(excess, -1, axis=1)
            spread_lx = np.roll(excess,  1, axis=1)
            biomass[sp] = bm - excess + (spread_up + spread_rx + spread_lx) / 5.0
            biomass[sp][0,  :] = np.maximum(biomass[sp][0, :], 0.0)
            biomass[sp][-1, :] *= 0.5   # sloughing at GCF interface

        # ── 5. Snapshot every 24 h ────────────────────────────────────────
        if step % save_every == 0:
            snap = _metrics(hour, biomass, nutrients, bm_max, _nz, _nx)
            history.append(snap)
            if verbose and step % (save_every * 7) == 0:
                wk = hour / HOURS_PER_WEEK
                comp = snap["composition"]
                print(f"  Week {wk:.1f}: V={snap['volume_scaled']:.2e} µm³  "
                      f"live={snap['live_frac']:.2f}  "
                      f"Str={comp['Str']:.0%}  Vel={comp['Vel']:.0%}  "
                      f"Fus={comp['Fus']:.0%}  Por={comp['Por']:.0%}")

    elapsed = time.time() - t0
    if verbose:
        print(f"Done: {n_weeks} weeks / {n_steps} steps in {elapsed:.1f}s "
              f"({_nz}×{_nx} grid, DT={_dt}h)")
    return history


def _metrics(hour, biomass, nutrients, bm_max, nz, nx) -> dict:
    total = sum(biomass[sp] for sp in SPECIES) + 1e-30
    occ   = total > bm_max * 1e-4

    volume     = float(occ.sum()) * DZ * DX
    volume_sc  = volume * (800.0 * 800.0) / (nx * DX * nz * DZ)
    live_mask  = (nutrients["glc"] > 5e-3) & occ
    live_frac  = float(live_mask.sum()) / max(float(occ.sum()), 1)
    area_frac  = float(occ.any(axis=0).sum()) / nx

    total_each = {sp: float(biomass[sp].sum()) for sp in SPECIES}
    total_all  = max(sum(total_each.values()), 1e-30)
    comp       = {sp: total_each[sp] / total_all for sp in SPECIES}

    return dict(
        hour=hour, week=hour / HOURS_PER_WEEK,
        biomass={sp: biomass[sp].copy() for sp in SPECIES},
        nutrients={k: v.copy() for k, v in nutrients.items()},
        volume=volume, volume_scaled=volume_sc,
        area_frac=area_frac, live_frac=live_frac,
        composition=comp,
    )


def _snap_at_week(history, target_week) -> dict:
    weeks = np.array([s["week"] for s in history])
    return history[int(np.argmin(np.abs(weeks - target_week)))]


# ── fitting ───────────────────────────────────────────────────────────────────

# Parameters to fit (log-scale for positive params):
#   log_mu_max[7]            log-space growth rates
#   log_bm_max               carrying capacity
#   log_spread_frac          shoving fraction
#   log_o2_inhib[Vel,Fus,Por] O2 inhibition factors
#   log_c_top_o2             GCF O2 boundary concentration

FIT_PARAM_NAMES = (
    ["log_mu_" + sp for sp in SPECIES]
    + ["log_bm_max", "log_spread_frac"]
    + ["log_o2_inhib_Vel", "log_o2_inhib_Fus", "log_o2_inhib_Por"]
    + ["log_c_top_o2"]
)
N_FIT_PARAMS = len(FIT_PARAM_NAMES)


def _params_to_config(x: np.ndarray, nz_fit: int, nx_fit: int) -> dict:
    """Unpack log-space parameter vector → dicts for run()."""
    mu = {sp: float(np.exp(x[i])) for i, sp in enumerate(SPECIES)}
    bm_max     = float(np.exp(x[7]))
    spread_frac = float(np.exp(x[8]))
    o2_inhib   = {"Vel": float(np.exp(x[9])),
                  "Fus": float(np.exp(x[10])),
                  "Por": float(np.exp(x[11]))}
    c_top_o2   = float(np.exp(x[12]))

    monod = {}
    for sp in SPECIES:
        m = {k: (list(v) if isinstance(v, tuple) else v)
             for k, v in MONOD_DEFAULT[sp].items()}
        m["mu_max"] = mu[sp]
        if sp in o2_inhib:
            m["o2_inhib_factor"] = o2_inhib[sp]
        monod[sp] = m

    c_top = dict(C_TOP_DEFAULT)
    c_top["o2"] = c_top_o2

    return dict(monod=monod, bm_max=bm_max, spread_frac=spread_frac,
                c_top=c_top, nz=nz_fit, nx=nx_fit)


def _default_x0() -> np.ndarray:
    x = np.zeros(N_FIT_PARAMS)
    for i, sp in enumerate(SPECIES):
        x[i] = np.log(MONOD_DEFAULT[sp]["mu_max"])
    x[7]  = np.log(BM_MAX_DENSITY_DEFAULT)
    x[8]  = np.log(SPREAD_FRAC_DEFAULT)
    x[9]  = np.log(MONOD_DEFAULT["Vel"]["o2_inhib_factor"])
    x[10] = np.log(MONOD_DEFAULT["Fus"]["o2_inhib_factor"])
    x[11] = np.log(MONOD_DEFAULT["Por"]["o2_inhib_factor"])
    x[12] = np.log(C_TOP_DEFAULT["o2"])
    return x


def rmse_loss(history, verbose_fit: bool = False) -> float:
    """
    RMSE loss vs Dieckow 2024 ground truth.

    Components:
      - Relative volume growth: log(V_wk/V_wk1) vs log(GT_wk/GT_wk1)
        Domain-size independent → valid for coarse fitting grid.
      - live_frac at weeks 1, 2, 3
      - composition at weeks 1, 2, 3 for all 7 species
    """
    if not history:
        return 1e6

    # reference at Week 1
    snap1 = _snap_at_week(history, 1.0)
    v_wk1 = max(snap1["volume_scaled"], 1.0)
    gt_v1 = GT_VOLUME[0]

    # log-ratios for GT volume growth
    gt_log_growth = np.log(GT_VOLUME / gt_v1)   # [0, 1.16, 1.97]

    loss_v = loss_l = loss_c = 0.0
    for wi, wk in enumerate(GT_WEEKS):
        snap = _snap_at_week(history, wk)
        v_sim = max(snap["volume_scaled"], 1.0)

        # relative volume growth (log-ratio)
        sim_log_growth = np.log(v_sim / v_wk1)
        loss_v += (sim_log_growth - gt_log_growth[wi])**2

        # live fraction
        loss_l += ((snap["live_frac"] - GT_LIVE[wi]) / GT_LIVE_SD[wi])**2

        # composition
        for sp in SPECIES:
            d = (snap["composition"][sp] - GT_COMP[sp][wi]) / GT_COMP_SD[sp][wi]
            loss_c += d**2

    loss = np.sqrt((loss_v / 3 + loss_l / 3 + loss_c / (3 * len(SPECIES))) / 3)
    if verbose_fit:
        print(f"    loss={loss:.4f}  "
              f"loss_v={loss_v/3:.3f}  loss_l={loss_l/3:.3f}  loss_c={loss_c/(3*len(SPECIES)):.3f}")
    return float(loss)


def fit(n_iter: int = 200, nz_fit: int = 20, nx_fit: int = 10,
        outfile: Path | None = None, verbose: bool = True) -> dict:
    """
    Fit Monod parameters to Dieckow 2024 ground truth using Nelder-Mead.

    Uses coarse grid (nz_fit × nx_fit) for speed, DT=1.0h.
    """
    call_count = [0]
    best = {"loss": np.inf, "x": None}

    def objective(x: np.ndarray) -> float:
        call_count[0] += 1
        try:
            cfg = _params_to_config(x, nz_fit, nx_fit)
            hist = run(n_weeks=3, seed=42, verbose=False, dt=1.0, **cfg)
            loss = rmse_loss(hist)
        except Exception as e:
            loss = 1e6
        if loss < best["loss"]:
            best["loss"] = loss
            best["x"] = x.copy()
            if verbose:
                print(f"  [{call_count[0]:4d}] new best loss={loss:.4f}")
        return loss

    x0 = _default_x0()
    if verbose:
        loss0 = objective(x0)
        print(f"Initial loss: {loss0:.4f}")
        print(f"Fitting {N_FIT_PARAMS} params on {nz_fit}×{nx_fit} grid ...")

    result = opt.minimize(
        objective, x0,
        method="Nelder-Mead",
        options=dict(maxiter=n_iter, xatol=1e-3, fatol=1e-4, disp=False),
    )

    x_best = best["x"] if best["x"] is not None else result.x
    params_out = {name: float(np.exp(x_best[i]))
                  for i, name in enumerate(FIT_PARAM_NAMES)}
    params_out["final_loss"] = float(best["loss"])

    if verbose:
        print(f"\nFit complete ({call_count[0]} evaluations), loss={best['loss']:.4f}")
        for name, val in params_out.items():
            if name != "final_loss":
                print(f"  {name:<28} = {val:.4f}")

    if outfile:
        outfile.parent.mkdir(parents=True, exist_ok=True)
        with open(outfile, "w") as f:
            json.dump(params_out, f, indent=2)
        print(f"Saved fit params: {outfile}")

    return params_out


def load_fit_params(param_file: Path) -> tuple[dict, dict, float, dict]:
    """Load fit result JSON and return (monod, bm_max, spread_frac, c_top)."""
    with open(param_file) as f:
        p = json.load(f)

    monod = {}
    for sp in SPECIES:
        m = {k: (list(v) if isinstance(v, tuple) else v)
             for k, v in MONOD_DEFAULT[sp].items()}
        m["mu_max"] = p[f"log_mu_{sp}"]   # already exp'd in params_out
        if f"log_o2_inhib_{sp}" in p:
            m["o2_inhib_factor"] = p[f"log_o2_inhib_{sp}"]
        monod[sp] = m

    c_top = dict(C_TOP_DEFAULT)
    c_top["o2"] = p["log_c_top_o2"]

    return monod, p["log_bm_max"], p["log_spread_frac"], c_top


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_all(history: list[dict], outdir: Path, title_suffix: str = "") -> None:
    weeks   = np.array([s["week"] for s in history])
    vol_sc  = np.array([s["volume_scaled"] for s in history])
    live_f  = np.array([s["live_frac"]     for s in history])
    comp_ts = {sp: np.array([s["composition"][sp] for s in history]) for sp in SPECIES}

    # ── A: scalars ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(
        f"2D Spatial Monod dFBA — Implant Biofilm (NIFE/SIIRI) {title_suffix}\n"
        "Dots = Dieckow et al. 2024 (12 patients, CLSM + 16S)",
        fontsize=9, fontweight="bold"
    )

    ax = axes[0]
    ax.plot(weeks, vol_sc, color="steelblue", lw=2, label="Simulation")
    ax.errorbar(GT_WEEKS, GT_VOLUME, yerr=GT_VOLUME_SD,
                fmt="o", color="firebrick", ms=9, capsize=5,
                label="Dieckow 2024 (median ± SD)")
    ax.set_xlabel("Time (weeks)"); ax.set_ylabel("Biofilm volume (µm³)")
    ax.set_title("Biofilm volume"); ax.set_yscale("log")
    ax.legend(fontsize=8); ax.spines[["top","right"]].set_visible(False)

    ax = axes[1]
    ax.plot(weeks, live_f, color="steelblue", lw=2)
    ax.errorbar(GT_WEEKS, GT_LIVE, yerr=GT_LIVE_SD,
                fmt="o", color="firebrick", ms=9, capsize=5)
    ax.set_xlabel("Time (weeks)"); ax.set_ylabel("Live cell fraction")
    ax.set_title("Live fraction (glc > 5 µM)")
    ax.set_ylim(0, 1); ax.spines[["top","right"]].set_visible(False)

    ax = axes[2]
    for sp in SPECIES:
        ax.plot(weeks, comp_ts[sp], color=COLORS[sp], lw=1.8, label=NAMES[sp])
        ax.errorbar(GT_WEEKS, GT_COMP[sp], yerr=GT_COMP_SD[sp],
                    fmt="o", color=COLORS[sp], ms=7, mec="k", mew=0.5,
                    capsize=3)
    ax.set_xlabel("Time (weeks)"); ax.set_ylabel("Relative abundance")
    ax.set_title("Composition  (lines=sim, dots=Dieckow 2024)")
    ax.set_ylim(0, 0.75); ax.legend(fontsize=7, ncol=2)
    ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    p = outdir / "nife_spatial_scalars.png"
    fig.savefig(p, dpi=180, bbox_inches="tight"); print(f"Saved: {p}")

    # ── B: spatial snapshots ─────────────────────────────────────────────
    _nz = list(history[0]["biomass"].values())[0].shape[0]
    _nx = list(history[0]["biomass"].values())[0].shape[1]
    snap_weeks = [1, 2, 3]
    snaps = [_snap_at_week(history, w) for w in snap_weeks]
    n_rows = len(SPECIES) + 2

    fig, axes = plt.subplots(n_rows, 3, figsize=(12, n_rows * 1.8 + 0.5))
    fig.suptitle("φ_i(z, x)  [z=0: implant surface, top: GCF]",
                 fontsize=10, fontweight="bold")

    for ci, (snap, tw) in enumerate(zip(snaps, snap_weeks)):
        tot = sum(snap["biomass"][sp] for sp in SPECIES) + 1e-30
        for ri, sp in enumerate(SPECIES):
            ax = axes[ri, ci]
            ax.imshow(snap["biomass"][sp] / tot, origin="lower",
                      vmin=0, vmax=0.8, cmap="Blues", aspect="auto",
                      extent=[0, _nx*DX, 0, _nz*DZ])
            if ri == 0: ax.set_title(f"Week {tw}", fontweight="bold")
            if ci == 0: ax.set_ylabel(NAMES[sp], fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])

        ax = axes[len(SPECIES), ci]
        ax.imshow(snap["nutrients"]["glc"], origin="lower",
                  vmin=0, vmax=C_TOP_DEFAULT["glc"], cmap="YlOrRd", aspect="auto",
                  extent=[0, _nx*DX, 0, _nz*DZ])
        if ci == 0: ax.set_ylabel("Glucose (mM)", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

        ax = axes[len(SPECIES)+1, ci]
        ax.imshow(snap["nutrients"]["o2"], origin="lower",
                  vmin=0, vmax=C_TOP_DEFAULT["o2"], cmap="Blues", aspect="auto",
                  extent=[0, _nx*DX, 0, _nz*DZ])
        if ci == 0: ax.set_ylabel("O₂ (mM)", fontsize=8)
        ax.set_xlabel("x (µm)"); ax.set_yticks([])

    plt.tight_layout()
    p = outdir / "nife_spatial_snapshots.png"
    fig.savefig(p, dpi=160, bbox_inches="tight"); print(f"Saved: {p}")

    # ── C: depth profiles at Week 3 ──────────────────────────────────────
    s3  = _snap_at_week(history, 3.0)
    tot = sum(s3["biomass"][sp] for sp in SPECIES) + 1e-30
    z   = np.arange(_nz) * DZ

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle("Depth profiles at Week 3  (z=0: implant, top: GCF)",
                 fontsize=10, fontweight="bold")

    ax = axes[0]
    for sp in SPECIES:
        frac = (s3["biomass"][sp] / tot).mean(axis=1)
        ax.plot(frac, z, color=COLORS[sp], lw=2, label=NAMES[sp])
    ax.set_xlabel("Relative abundance"); ax.set_ylabel("Depth z (µm)")
    ax.set_title("Species stratification"); ax.set_ylim(0, _nz*DZ)
    ax.legend(fontsize=8); ax.spines[["top","right"]].set_visible(False)

    ax = axes[1]
    for nut, col, lab in [
        ("glc", "tomato", "Glucose"), ("o2", "steelblue", "O₂"),
        ("lac", "darkorange", "Lactate"), ("aa", "mediumpurple", "Amino acids"),
    ]:
        ax.plot(s3["nutrients"][nut].mean(axis=1), z, color=col, lw=2, label=lab)
    ax.set_xlabel("Concentration (mM)"); ax.set_ylabel("Depth z (µm)")
    ax.set_title("Nutrient gradients"); ax.set_ylim(0, _nz*DZ)
    ax.legend(); ax.spines[["top","right"]].set_visible(False)

    ax = axes[2]
    aerobe_frac = sum((s3["biomass"][sp]/tot).mean(axis=1) for sp in ["Str","Act","Hae","Rot"])
    anaerobe_frac = sum((s3["biomass"][sp]/tot).mean(axis=1) for sp in ["Vel","Fus","Por"])
    o2_norm = s3["nutrients"]["o2"].mean(axis=1) / C_TOP_DEFAULT["o2"]
    ax.fill_betweenx(z, 0, aerobe_frac,   alpha=0.5, color="steelblue",
                     label="Aerotolerant (Str+Act+Hae+Rot)")
    ax.fill_betweenx(z, 0, anaerobe_frac, alpha=0.5, color="firebrick",
                     label="Strict anaerobe (Vel+Fus+Por)")
    ax.plot(o2_norm, z, "k--", lw=1.5, label="O₂/O₂_top")
    ax.set_xlabel("Fraction / norm. O₂"); ax.set_ylabel("Depth z (µm)")
    ax.set_title("Aerobe/anaerobe zonation"); ax.set_ylim(0, _nz*DZ)
    ax.legend(fontsize=8); ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    p = outdir / "nife_spatial_depth.png"
    fig.savefig(p, dpi=180, bbox_inches="tight"); print(f"Saved: {p}")


def print_comparison_table(history: list[dict]) -> None:
    print(f"\n{'':=<62}")
    print(" vs Dieckow 2024 ground truth")
    print(f"{'':=<62}")
    header = f"  {'Metric':<30}" + "".join(f"  {'Wk'+str(w):>8}" for w in [1,2,3])
    print(header)
    print("-" * 62)

    def row(label, sim_vals, gt_vals, sd_vals=None, fmt=".2e"):
        fmt_fn = lambda v: f"{v:{fmt}}"
        sim_str = "".join(f"  {fmt_fn(v):>8}" for v in sim_vals)
        gt_str  = "".join(f"  {fmt_fn(v):>8}" for v in gt_vals)
        err_str = ""
        if sd_vals is not None:
            rel = [(abs(s - g) / max(abs(g), 1e-9)) for s, g in zip(sim_vals, gt_vals)]
            err_str = "   ΔΡ " + " ".join(f"{r:.0%}" for r in rel)
        print(f"  {label:<30}{sim_str}   [sim]")
        print(f"  {'':<30}{gt_str}   [Dieckow 2024]{err_str}")

    sim_vol   = [_snap_at_week(history, w)["volume_scaled"] for w in [1,2,3]]
    sim_live  = [_snap_at_week(history, w)["live_frac"]     for w in [1,2,3]]
    v1_sim    = max(sim_vol[0], 1.0)
    v1_gt     = GT_VOLUME[0]
    sim_rel   = [np.log(max(v, 1.0)/v1_sim) for v in sim_vol]
    gt_rel    = [np.log(v/v1_gt) for v in GT_VOLUME]
    row("Volume (µm³)",          sim_vol,  GT_VOLUME, GT_VOLUME_SD, fmt=".2e")
    row("Vol growth log(V/V₁)", sim_rel,  gt_rel,  fmt=".2f")
    row("Live fraction",         sim_live, GT_LIVE,  GT_LIVE_SD, fmt=".2f")

    loss = rmse_loss(history, verbose_fit=True)
    print(f"\n  Total RMSE loss: {loss:.4f}")

    print(f"\n  Composition comparison (Week 3):")
    s3 = _snap_at_week(history, 3)
    for sp in SPECIES:
        sv = s3["composition"][sp]
        gv = GT_COMP[sp][2]
        ok = "✓" if abs(sv - gv) / max(gv, 0.01) < 0.30 else "✗"
        print(f"    {NAMES[sp]:<28} sim={sv:.1%}  Dieckow={gv:.1%}  {ok}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    pa = argparse.ArgumentParser(
        description="2D spatial Monod dFBA — implant biofilm (Dieckow 2024 fit)"
    )
    pa.add_argument("--weeks",     type=int,   default=3)
    pa.add_argument("--seed",      type=int,   default=42)
    pa.add_argument("--outdir",    type=Path,  default=Path("nife/comets/"))
    pa.add_argument("--plot-only", action="store_true",
                    help="Reload saved history and replot")
    pa.add_argument("--fast",      action="store_true",
                    help="Coarse grid (NZ=20×NX=10) for quick testing")
    pa.add_argument("--fit",       action="store_true",
                    help="Fit Monod params to Dieckow 2024 data")
    pa.add_argument("--fit-iter",  type=int,  default=300,
                    help="Nelder-Mead max iterations (default 300)")
    pa.add_argument("--fit-nz",    type=int,  default=20)
    pa.add_argument("--fit-nx",    type=int,  default=10)
    pa.add_argument("--params",    type=Path, default=None,
                    help="Load fit params JSON (skip fitting)")
    args = pa.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    hist_f   = args.outdir / "spatial_history.npy"
    param_f  = args.outdir / "spatial_fit_params.json"

    # ── fit mode ──────────────────────────────────────────────────────────
    if args.fit:
        fitted = fit(n_iter=args.fit_iter, nz_fit=args.fit_nz, nx_fit=args.fit_nx,
                     outfile=param_f)
        print("\nRe-running full grid with fitted params for final plots...")
        monod, bm_max, sf, c_top = load_fit_params(param_f)
        history = run(n_weeks=args.weeks, seed=args.seed,
                      monod=monod, bm_max=bm_max, spread_frac=sf, c_top=c_top)
        np.save(str(hist_f), history)
        plot_all(history, args.outdir, title_suffix="[fitted]")
        print_comparison_table(history)
        return

    # ── simulation mode ───────────────────────────────────────────────────
    nz = 20 if args.fast else NZ
    nx = 10 if args.fast else NX
    dt = 1.0 if args.fast else DT

    if args.plot_only and hist_f.exists():
        history = list(np.load(str(hist_f), allow_pickle=True))
        print(f"Loaded {len(history)} snapshots from {hist_f}")
    else:
        kwargs: dict = dict(nz=nz, nx=nx, dt=dt)
        if args.params and args.params.exists():
            monod, bm_max, sf, c_top = load_fit_params(args.params)
            kwargs.update(monod=monod, bm_max=bm_max, spread_frac=sf, c_top=c_top)
            title_suf = "[fitted params]"
        else:
            title_suf = ""

        print(f"Grid: {nz}×{nx} voxels ({nz*DZ:.0f}×{nx*DX:.0f} µm), "
              f"DT={dt}h, QSS SuperLU solver")
        history = run(n_weeks=args.weeks, seed=args.seed, **kwargs)
        np.save(str(hist_f), history)

    plot_all(history, args.outdir)
    print_comparison_table(history)


if __name__ == "__main__":
    main()
