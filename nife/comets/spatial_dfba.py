#!/usr/bin/env python3
"""
spatial_dfba.py — 2D spatial Monod dFBA for implant biofilm (NIFE/SIIRI)
=========================================================================

Simulates a vertical cross-section of an implant biofilm:
  z = 0:     implant surface (Neumann, no-flux)
  z = NZ-1:  GCF/saliva reservoir (Dirichlet, fixed concentration)
  x:         lateral (periodic)

Nutrient solver: quasi-steady-state (QSS) at each growth step.
  Rationale: O2/glucose equilibrate in t ~ L²/D ≈ (200µm)²/2e6 µm²/h ≈ 0.02h,
  much faster than growth (DT=0.25h). So nutrients track biomass instantly.
  This avoids the explicit Euler CFL instability (requires DT < DZ²/4D ~ 1e-5h).

  Nutrient ODE (QSS):  0 = D ∇²c - R(c, X)   →  sparse linear solve (scipy)

Ground truth (Dieckow et al. 2024, npj Biofilms Microbiomes 10:155):
  - V(t):      Week1 ~ 2.5e5, Week2 ~ 8e5, Week3 ~ 1.8e6 µm³
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

Nutrients (Dieckow DOMINO interactions + Joshi 2025 amino-acid metabolism):
  glc = glucose, o2 = oxygen, lac = lactate, aa = amino acids, no3 = nitrate

Reference: Dukovski et al. 2021 (COMETS, Nat. Protocols); Stewart 2003 (biofilm D_eff);
           Periasamy & Kolenbrander 2009; Marsh & Martin 1999 (Oral Microbiology).
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=sp.SparseEfficiencyWarning)

# ── grid ──────────────────────────────────────────────────────────────────────
DZ = DX = 10.0       # µm per voxel
NZ = 60              # depth (z) voxels  →  600 µm total (must fit 7× biofilm growth)
NX = 40              # lateral (x) voxels →  400 µm total
N  = NZ * NX         # total voxels
DT = 0.25            # h per growth step  (nutrients solved at QSS each step)
HOURS_PER_WEEK = 168.0

# ── diffusion coefficients [µm²/h] ───────────────────────────────────────────
# Ref: Stewart 2003 (Appl Env Microbiol 69:7188): D_eff ≈ 0.3–0.8 × D_water in biofilm
# D_O2_water = 7.56e6 µm²/h; D_glc_water = 2.16e6 µm²/h
# We use ~0.4× for early loose biofilm, decreasing implicitly via biomass density.
D = {
    "glc":  4.3e5,   # µm²/h  (D_eff ≈ 20% × D_glc_water; Stewart 2003: 0.2-0.4× in biofilm)
    "o2":   1.5e6,   # µm²/h  (D_eff ≈ 20% × D_O2_water)
    "lac":  3.6e5,   # µm²/h
    "aa":   2.5e5,   # µm²/h
    "no3":  3.5e5,   # µm²/h
}

# ── boundary concentrations (GCF/saliva at z = NZ-1) ─────────────────────────
# Ref: Aas et al. 2005 (oral environment); Soukos & Goodson 2011 (GCF composition)
C_TOP = {
    "glc":  0.10,    # mM  (GCF glucose — low subgingivally)
    "o2":   0.03,    # mM  (peri-implant sulcus: micro-aerobic, ~10% air-sat;
                     #       Ref: Mettraux et al. 1984 J Periodontol; deep sites < 0.02 mM)
    "lac":  0.05,    # mM  (trace background from host cell catabolism in GCF)
    "aa":   2.00,    # mM  (host-derived peptides/amino acids, abundant in GCF)
    "no3":  0.05,    # mM  (saliva nitrate; Haemophilus electron acceptor)
}
NUTS = list(C_TOP.keys())

# ── biomass carrying capacity ─────────────────────────────────────────────────
# Dense oral biofilm: ~0.5–1 gDW/L wet.
# Units: gDW/L — consistent with q [mmol/gDW/h] → uptake = q*bm in mM/h
# (1 mM = 1 mmol/L; D in µm²/h → RHS = uptake/D in mM/µm² for ∇²c)
# Ref: Schlafer & Meyer 2017 (J Oral Microbiol): dental biofilm ~0.5 gDW/L
BM_MAX_DENSITY = 0.5    # gDW/L

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

# ── Monod parameters ──────────────────────────────────────────────────────────
# mu_max [h⁻¹]; substrates: {nut: (q_max [mmol/gDW/h], Km [mM], Y [gDW/mmol])}
# o2_inhibit: strict anaerobe (O2 suppresses growth)
# needs_o2:   obligate aerobe (mu → 0 without O2)
# secretion:  {nut: stoich}  mmol secreted per mmol primary substrate consumed
# References:
#   Str: Kolenbrander 2000 (Annu Rev Microbiol); mu_max from Periasamy 2009
#   Vel: Periasamy & Kolenbrander 2009 — obligate lactate cross-feeder
#   Hae: Murphy & Sethi 1992; nitrate respiration Haemophilus parainfluenzae
#   Fus: Sakanaka 2022 (mSystems) — amino acid catabolism
#   Por: Lamont & Jenkinson 1998; Darveau 2010 — hemin + proteolysis
MONOD = {
    # Str: dominant early colonizer.
    # FACULTATIVE: aerobic respiration → faster growth; anaerobic fermentation → slower.
    # In O2-replete zones: mu ≈ mu_max; in anaerobic zones: mu ≈ mu_max * 0.45.
    # This creates spatial niche for strict anaerobes (Vel) at the implant surface.
    # Ref: Kolenbrander 2000; Streptococcus oralis aerobic/anaerobic growth ratio ~2:1
    "Str": dict(mu_max=0.50,
                substrates={"glc": (8.0, 0.05, 0.06)},
                o2_inhibit=False, needs_o2=False, o2_inhib_factor=0.0,
                o2_aerobic_boost=True,   # grows faster with O2 (facultative)
                secretion={"lac": 1.8}, primary="glc"),
    # Act: structural scaffolding, primary aa consumer + secondary glucose.
    # mu_max=0.40 → competes effectively at 10-15% share.
    # PRIMARY substrate = aa (host protein degradation, abundant throughout biofilm).
    # Ref: Kolenbrander 2000 (Annu Rev Microbiol); Palmer et al. 2001 (Actinomyces adhesins)
    "Act": dict(mu_max=0.40,
                substrates={"aa":  (5.0, 0.40, 0.06),
                            "glc": (2.0, 0.15, 0.03)},
                o2_inhibit=False, needs_o2=False, o2_inhib_factor=0.0,
                o2_aerobic_boost=False,   # not strongly aerobic-dependent (surface colonizer)
                secretion={}, primary="aa"),
    # Vel: obligate lactate cross-feeder; anaerobe confined to low-O2 base.
    # mu_max reduced (0.25) so Vel doesn't outcompete Str for the whole biofilm.
    # Km_lac=0.30 mM → Vel growth limited unless lactate > 0.3 mM (only in deep layer).
    # o2_inhib_factor=5 → strong suppression at GCF O2=0.03: inhib=1/(1+5×0.6)=0.25
    # Ref: Periasamy & Kolenbrander 2009 (J Bacteriol); Lemos et al. 2019 (Veillonella)
    "Vel": dict(mu_max=0.25,
                substrates={"lac": (8.0, 0.30, 0.05)},
                o2_inhibit=True, needs_o2=False, o2_inhib_factor=5.0,
                o2_aerobic_boost=False,
                secretion={}, primary="lac"),
    # Hae: facultative, requires O2 → ~6% at micro-aerobic sulcus
    "Hae": dict(mu_max=0.28,
                substrates={"glc": (3.0, 0.10, 0.04)},
                o2_inhibit=False, needs_o2=True, o2_inhib_factor=0.0,
                o2_aerobic_boost=False,
                secretion={}, primary="glc"),
    # Rot: aerobic health-associated → ~5-6% share
    "Rot": dict(mu_max=0.18,
                substrates={"glc": (3.5, 0.07, 0.04),
                            "aa":  (1.5, 0.45, 0.04)},
                o2_inhibit=False, needs_o2=True, o2_inhib_factor=0.0,
                o2_aerobic_boost=False,
                secretion={}, primary="glc"),
    # Fus: bridge species, amino-acid specialist; microaerophilic tolerance
    # mu_max=0.22 → ~3-5% share; main substrate = aa (less competition with Str/Vel)
    "Fus": dict(mu_max=0.22,
                substrates={"lac": (2.0, 0.25, 0.03),
                            "aa":  (5.0, 0.30, 0.05)},
                o2_inhibit=True, needs_o2=False, o2_inhib_factor=3.0,
                o2_aerobic_boost=False,
                secretion={}, primary="aa"),
    # Por: deep anaerobe, very slow grower → ~1-2% share
    "Por": dict(mu_max=0.08,
                substrates={"aa": (5.0, 0.25, 0.06)},
                o2_inhibit=True, needs_o2=False, o2_inhib_factor=6.0,
                o2_aerobic_boost=False,
                secretion={}, primary="aa"),
}

# ── Dieckow 2024 ground truth (Suppl Fig. 1 & 3) ─────────────────────────────
GT_WEEKS    = np.array([1.0, 2.0, 3.0])
GT_VOLUME   = np.array([2.5e5, 8.0e5, 1.8e6])   # µm³ per 800×800 µm CLSM image
GT_LIVE     = np.array([0.87,  0.84,  0.81 ])
GT_COMP     = {                                    # mean genus relative abundance
    "Str": np.array([0.50, 0.48, 0.45]),
    "Act": np.array([0.10, 0.12, 0.13]),
    "Vel": np.array([0.18, 0.20, 0.22]),
    "Hae": np.array([0.08, 0.07, 0.06]),
    "Rot": np.array([0.06, 0.06, 0.05]),
    "Fus": np.array([0.03, 0.04, 0.05]),
    "Por": np.array([0.01, 0.01, 0.02]),
}
# Scale from Dieckow imaging field (800×800 µm) to our grid (NX*DX × NZ*DZ µm)
GRID_SCALE = (NX * DX * NZ * DZ) / (800.0 * 800.0)


# ── sparse Laplacian with BCs ─────────────────────────────────────────────────

def build_laplacian_2d(nz: int, nx: int, dz: float, dx: float) -> sp.csr_matrix:
    """
    2D Laplacian on (nz, nx) grid.
    Index ordering: i = iz * nx + ix  (row-major)

    BCs:
      z=nz-1  (top)  : Dirichlet — excluded from solve, handled via RHS
      z=0     (bottom): Neumann  — ghost cell trick (row i reflects as i+nx)
      x: periodic
    """
    n = nz * nx
    rows, cols, data = [], [], []

    for iz in range(nz):
        for ix in range(nx):
            i = iz * nx + ix

            # z-direction (Neumann at z=0, Dirichlet handled at z=nz-1)
            if iz == 0:
                # dC/dz = 0 → C[iz=0] = C[iz=1]  →  ghost: iz=-1 = iz=1
                i_zp = (iz + 1) * nx + ix
                rows += [i, i, i]
                cols += [i, i, i_zp]
                data += [-1/dz**2, -1/dz**2, 2/dz**2]   # (C[1]-C[0])/dz² = 0 → LC = -C[0]+C[1]
            elif iz == nz - 1:
                # Dirichlet: this row will be replaced by identity
                rows.append(i); cols.append(i); data.append(1.0)
                continue
            else:
                i_zm = (iz - 1) * nx + ix
                i_zp = (iz + 1) * nx + ix
                rows += [i, i, i]
                cols += [i, i_zm, i_zp]
                data += [-2/dz**2, 1/dz**2, 1/dz**2]

            # x-direction (periodic)
            ix_m = (ix - 1) % nx
            ix_p = (ix + 1) % nx
            i_xm = iz * nx + ix_m
            i_xp = iz * nx + ix_p
            rows += [i, i]
            cols += [i_xm, i_xp]
            data += [1/dx**2, 1/dx**2]
            rows.append(i); cols.append(i); data.append(-2/dx**2)

    L = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    return L


def solve_nutrient_qss(L: sp.csr_matrix,
                       c_top: float, uptake_rate: np.ndarray,
                       nut_D: float) -> np.ndarray:
    """
    Solve 2D QSS nutrient diffusion-reaction equation.

    Physical PDE (steady state):
        D * ∇²c = net_consumption  (net_consumption > 0 = consumed, < 0 = produced)

    Since uptake_rate = secretion - consumption  (positive = net production):
        D * ∇²c = -uptake_rate
        L * c = -uptake_rate / D

    BCs:
        z = NZ-1 (top, GCF):    Dirichlet, c = c_top
        z = 0    (bottom, impl): Neumann,   dc/dz = 0  (no-flux)
        x:                       periodic
    """
    nx = NX

    # Correct sign: L*c = -uptake_rate / D
    rhs = -uptake_rate.copy() / nut_D   # (N,) negative where consumed → ∇²c > 0 → depleted at bottom ✓

    # Apply Dirichlet BC at top (z=NZ-1): replace rows with identity
    A = L.copy().tolil()
    for ix in range(nx):
        i = (NZ - 1) * nx + ix
        A[i, :] = 0.0
        A[i, i] = 1.0
        rhs[i] = c_top   # fixed concentration at GCF boundary

    A = A.tocsr()
    c_flat = spla.spsolve(A, rhs)
    c_flat = np.clip(c_flat, 0.0, None)
    return c_flat.reshape(NZ, NX)


# ── initial conditions ────────────────────────────────────────────────────────

def init_state(rng: np.random.Generator, L: "sp.csr_matrix"):
    """
    Initialize a Week-1 equivalent established biofilm with correct spatial zonation.

    Strategy: rather than simulating colonization from scratch (which requires
    resolving sub-micron scales), start from an established early biofilm
    matching Dieckow 2024 Week-1 average properties:
      - Total biomass: ~50% capacity at z=0..3 (thin layer ≈ 20-30 µm)
      - Zonation: aerobic-species-rich at z=3+ (near GCF), anaerobe-rich at z=0-1
      - Then compute QSS nutrients → correct O2/lactate gradients from step 0

    This is physically motivated: Dieckow samples Week-1 biofilms that have
    already undergone initial succession; we simulate Week 1 → 3 dynamics.
    """
    # ── Biomass: Week-1 established biofilm (~80 µm) ─────────────────────
    # Dieckow Week-1: V ≈ 2.5e5 µm³ in 800×800 µm field → ~8 µm mean height
    # (CLSM biovolume, patchy coverage). We use 8 voxels (80 µm) at 55-70%
    # capacity with zonation already present (Str/Act at top, Vel/Fus at base).
    # z=0: implant surface (anaerobic niche; Vel, Fus, Por enriched)
    # z=7: biofilm surface (aerobic niche; Str, Act, Hae, Rot enriched)
    INIT_THICKNESS = 8   # voxels = 80 µm thick Week-1 equivalent biofilm

    # Layer biases: innermost (z=0) = deep anaerobe; outermost = aerotolerant
    # Each row: [z=0, z=1, z=2, z=3, z=4, z=5, z=6, z=7] bias relative to GT_COMP
    LAYER_BIAS = {
        "Str": [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8],
        "Act": [0.5, 0.7, 0.9, 1.1, 1.2, 1.3, 1.4, 1.4],
        "Vel": [2.5, 2.2, 1.8, 1.4, 1.0, 0.6, 0.3, 0.1],
        "Hae": [0.2, 0.4, 0.7, 1.0, 1.3, 1.5, 1.6, 1.5],
        "Rot": [0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.7, 1.8],
        "Fus": [3.0, 2.5, 1.8, 1.2, 0.8, 0.4, 0.2, 0.1],
        "Por": [4.0, 3.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.0],
    }

    biomass = {sp: np.zeros((NZ, NX)) for sp in SPECIES}
    rho_base = BM_MAX_DENSITY * 0.60   # 60% capacity (Week-1 established biofilm)

    for iz in range(INIT_THICKNESS):
        # Density: 65% at base (most mature), 45% at top (growing front)
        rho_iz = rho_base * (1.0 - 0.025 * iz)

        # Layer-specific composition from biased GT Week-1 fractions
        raw = {sp: GT_COMP[sp][0] * LAYER_BIAS[sp][iz] for sp in SPECIES}
        total_raw = sum(raw.values())
        layer_fracs = {sp: raw[sp] / total_raw for sp in SPECIES}

        for sp in SPECIES:
            noise = 1.0 + 0.08 * rng.standard_normal(NX)
            biomass[sp][iz, :] = rho_iz * layer_fracs[sp] * np.clip(noise, 0.8, 1.2)

    # ── Nutrients: 2-step Picard to get self-consistent initial fields ────
    # Straight QSS from initial-guess uptake → nutrients are inaccurate.
    # Two Picard iterations converge to ~true fixed point before simulation starts.
    nutrients = {nut: np.full((NZ, NX), C_TOP[nut]) for nut in C_TOP}
    for _init_iter in range(4):
        uptake_fields = {k: np.zeros((NZ, NX)) for k in C_TOP}
        for sp in SPECIES:
            p  = MONOD[sp]
            bm = biomass[sp]
            for nut, (q_max, Km, Y) in p["substrates"].items():
                c = nutrients[nut]
                q = q_max * c / (Km + c + 1e-15)
                uptake_fields[nut] -= q * bm
            prim = p["primary"]
            for sec_nut, stoich in p.get("secretion", {}).items():
                c_p = nutrients[prim]
                q_p_max, Km_p, _ = p["substrates"][prim]
                q_p = q_p_max * c_p / (Km_p + c_p + 1e-15)
                uptake_fields[sec_nut] += stoich * q_p * bm
        for nut in C_TOP:
            c_qss = solve_nutrient_qss(L, C_TOP[nut], uptake_fields[nut].ravel(), D[nut])
            nutrients[nut] = 0.5 * nutrients[nut] + 0.5 * c_qss

    return nutrients, biomass


# ── main loop ─────────────────────────────────────────────────────────────────

def run(n_weeks: int = 3, seed: int = 42, verbose: bool = True):
    rng = np.random.default_rng(seed)
    L = build_laplacian_2d(NZ, NX, DZ, DX)
    nutrients, biomass = init_state(rng, L)

    n_steps    = int(n_weeks * HOURS_PER_WEEK / DT)
    save_every = max(1, int(24.0 / DT))   # save every 24 h
    history    = []

    t0 = time.time()
    for step in range(n_steps + 1):
        hour = step * DT

        total_bm = sum(biomass[sp] for sp in SPECIES)   # (NZ, NX) [gDW/L]
        logistic  = np.clip(1.0 - total_bm / BM_MAX_DENSITY, 0.0, 1.0)

        # ── 1. Converge QSS nutrients (6 Picard iterations, α=0.7) ───────
        # Period-2 QSS oscillation suppressed by Picard with over-relaxation α=0.7.
        # After k iterations: c_err ≈ C_top × (1-α)^k = C_top × 0.3^k
        # After 6 iterations: 0.3^6 = 7×10⁻⁴ → c_err << Km → correct live_frac.
        # Ref: Kelley 1995 (Iterative Methods for Nonlinear Equations), §2.2
        RELAX = 0.7
        for _qss_iter in range(6):
            uptake_fields = {k: np.zeros((NZ, NX)) for k in NUTS}
            for sp in SPECIES:
                p  = MONOD[sp]
                bm = biomass[sp]
                o2 = nutrients["o2"]

                mu_est = np.zeros((NZ, NX))
                for nut, (q_max, Km, Y) in p["substrates"].items():
                    c = nutrients[nut]
                    q = q_max * c / (Km + c + 1e-15)
                    mu_est += q * Y
                    uptake_fields[nut] -= q * bm
                mu_est = np.minimum(mu_est, p["mu_max"])

                if p.get("o2_aerobic_boost", False):
                    mu_est *= 0.45 + 0.55 * o2 / (0.05 + o2)
                if p["o2_inhibit"]:
                    mu_est *= 1.0 / (1.0 + p.get("o2_inhib_factor", 3.0) * o2 / (0.02 + o2))
                if p["needs_o2"]:
                    mu_est *= o2 / (0.05 + o2)
                if not p["o2_inhibit"]:
                    uptake_fields["o2"] -= (mu_est / max(p["mu_max"], 1e-9)) * bm * 0.5
                prim = p["primary"]
                for sec_nut, stoich in p.get("secretion", {}).items():
                    c_p = nutrients[prim]
                    q_p_max, Km_p, _ = p["substrates"][prim]
                    q_p = q_p_max * c_p / (Km_p + c_p + 1e-15)
                    uptake_fields[sec_nut] += stoich * q_p * bm

            for nut in NUTS:
                c_qss = solve_nutrient_qss(L, C_TOP[nut], uptake_fields[nut].ravel(), D[nut])
                nutrients[nut] = (1.0 - RELAX) * nutrients[nut] + RELAX * c_qss

        # ── 2. Compute growth rates from converged nutrients ──────────────
        growth_rates = {}
        for sp in SPECIES:
            p  = MONOD[sp]
            bm = biomass[sp]
            o2 = nutrients["o2"]

            mu = np.zeros((NZ, NX))
            for nut, (q_max, Km, Y) in p["substrates"].items():
                c = nutrients[nut]
                q = q_max * c / (Km + c + 1e-15)
                mu += q * Y
            mu = np.minimum(mu, p["mu_max"])

            if p.get("o2_aerobic_boost", False):
                aerobic_factor = 0.45 + 0.55 * o2 / (0.05 + o2)
                # pH self-inhibition: Str lactic acid secretion → pH drop → self-limit.
                # Creates anaerobic niche for Vel in lactate-rich base layer.
                # Ref: Marsh & Martin 1999 (Oral Microbiology); Sissons et al. 1988
                c_lac = nutrients["lac"]
                pH = np.maximum(5.8, 7.4 - 0.8 * c_lac)
                acid_inhib = np.exp(-2.0 * np.maximum(0.0, 7.0 - pH))
                mu *= aerobic_factor * acid_inhib

            if p["o2_inhibit"]:
                mu *= 1.0 / (1.0 + p.get("o2_inhib_factor", 3.0) * o2 / (0.02 + o2))
            if p["needs_o2"]:
                mu *= o2 / (0.05 + o2)

            growth_rates[sp] = mu * logistic

        # ── 3. Update biomass ─────────────────────────────────────────────
        for sp in SPECIES:
            biomass[sp] = np.maximum(biomass[sp] * np.exp(growth_rates[sp] * DT), 0.0)

        total_new = sum(biomass[sp] for sp in SPECIES) + 1e-30
        overshoot = np.maximum(total_new / BM_MAX_DENSITY, 1.0)
        for sp in SPECIES:
            biomass[sp] /= overshoot

        # ── 4. Biomass spreading (shoving) ────────────────────────────────
        # Saturated voxels (>80% capacity) push excess biomass upward (primary
        # growth direction, away from implant) and laterally.
        # SPREAD_FRAC=0.020 calibrated so biofilm grows ~7× in 2 weeks on 600µm domain.
        # Ref: Picioreanu et al. 1998 (Biotech Bioeng 56:652); Lardon 2011 (iDynoMiCS)
        total_bm2 = sum(biomass[sp] for sp in SPECIES)
        saturated = total_bm2 > BM_MAX_DENSITY * 0.80

        SPREAD_FRAC = 0.020
        for sp in SPECIES:
            bm    = biomass[sp]
            excess = bm * saturated * SPREAD_FRAC
            spread_up = np.zeros_like(excess)
            spread_up[1:, :] = excess[:-1, :] * 3.0   # upward (z+1), strongest
            spread_rx = np.roll(excess, -1, axis=1)    # lateral ×1
            spread_lx = np.roll(excess,  1, axis=1)
            biomass[sp] = bm - excess + (spread_up + spread_rx + spread_lx) / 5.0

        for sp in SPECIES:
            biomass[sp][0,  :] = np.maximum(biomass[sp][0, :], 0.0)
            biomass[sp][-1, :] *= 0.5   # sloughing at GCF interface

        # ── 4. Save snapshot ──────────────────────────────────────────────
        if step % save_every == 0:
            snap = _metrics(hour, biomass, nutrients)
            history.append(snap)
            if verbose and step % (save_every * 7) == 0:
                wk = hour / HOURS_PER_WEEK
                comp = snap["composition"]
                print(f"  Week {wk:.1f}: V_scaled={snap['volume_scaled']:.2e} µm³  "
                      f"live={snap['live_frac']:.2f}  "
                      f"Str={comp['Str']:.0%}  Vel={comp['Vel']:.0%}  "
                      f"Fus={comp['Fus']:.0%}  Por={comp['Por']:.0%}")

    elapsed = time.time() - t0
    if verbose:
        print(f"Done: {n_weeks} weeks × {n_steps} steps in {elapsed:.1f}s "
              f"({NZ}×{NX} grid, DT={DT}h, QSS solver)")
    return history


def _metrics(hour, biomass, nutrients):
    total = sum(biomass[sp] for sp in SPECIES)
    occ   = total > BM_MAX_DENSITY * 1e-4   # occupied voxel threshold

    volume     = float(occ.sum()) * DZ * DX   # µm³ (assuming 1 µm depth layer)
    area_frac  = float(occ.any(axis=0).sum()) / NX
    # Live = voxels where glucose > 5 µM (below Km/10 = nutrient-starved/dead).
    # Threshold: Km_glc(Str) = 0.05 mM; < 10% Km → growth rate < 17% of max → "dead".
    # Ref: Stewart 1994 (Biofilm Sci Tech); Wanner et al. 2006 (Math Biofilm Model)
    live_mask  = (nutrients["glc"] > 5e-3) & occ
    live_frac  = float(live_mask.sum()) / max(float(occ.sum()), 1)

    total_each = {sp: float(biomass[sp].sum()) for sp in SPECIES}
    total_all  = max(sum(total_each.values()), 1e-30)
    comp       = {sp: total_each[sp] / total_all for sp in SPECIES}

    return dict(
        hour=hour, week=hour / HOURS_PER_WEEK,
        biomass={sp: biomass[sp].copy() for sp in SPECIES},
        nutrients={k: v.copy() for k, v in nutrients.items()},
        volume=volume,
        volume_scaled=volume * (800.0 * 800.0) / (NX * DX * NZ * DZ),
        area_frac=area_frac, live_frac=live_frac,
        composition=comp,
    )


# ── plotting ──────────────────────────────────────────────────────────────────

def _snap_at_week(history, target_week):
    weeks = np.array([s["week"] for s in history])
    return history[int(np.argmin(np.abs(weeks - target_week)))]


def plot_all(history, outdir):
    weeks   = np.array([s["week"] for s in history])
    vol_sc  = np.array([s["volume_scaled"] for s in history])
    live_f  = np.array([s["live_frac"]     for s in history])
    comp_ts = {sp: np.array([s["composition"][sp] for s in history]) for sp in SPECIES}

    # ── A: scalar trajectories vs Dieckow 2024 ──────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(
        "2D Spatial Monod dFBA — Implant Biofilm (NIFE/SIIRI)\n"
        "Dots = Dieckow et al. 2024 ground truth (12 patients, CLSM + 16S full-length)",
        fontsize=10, fontweight="bold"
    )

    ax = axes[0]
    ax.plot(weeks, vol_sc, color="steelblue", lw=2, label="Simulation")
    ax.errorbar(GT_WEEKS, GT_VOLUME,
                yerr=[v * 0.45 for v in GT_VOLUME],
                fmt="o", color="firebrick", ms=9, capsize=5,
                label="Dieckow 2024 (median ± IQR)")
    ax.set_xlabel("Time (weeks)"); ax.set_ylabel("Biofilm volume (µm³)")
    ax.set_title("Biofilm volume"); ax.set_yscale("log")
    ax.legend(fontsize=8); ax.spines[["top","right"]].set_visible(False)

    ax = axes[1]
    ax.plot(weeks, live_f, color="steelblue", lw=2)
    ax.errorbar(GT_WEEKS, GT_LIVE, yerr=0.05,
                fmt="o", color="firebrick", ms=9, capsize=5)
    ax.set_xlabel("Time (weeks)"); ax.set_ylabel("Live cell fraction")
    ax.set_title("Live cell fraction (glucose > 1 µM)")
    ax.set_ylim(0, 1); ax.spines[["top","right"]].set_visible(False)

    ax = axes[2]
    for sp in SPECIES:
        ax.plot(weeks, comp_ts[sp], color=COLORS[sp], lw=1.8, label=NAMES[sp])
        ax.plot(GT_WEEKS, GT_COMP[sp], "o", color=COLORS[sp],
                ms=7, mec="k", mew=0.5)
    ax.set_xlabel("Time (weeks)"); ax.set_ylabel("Relative abundance")
    ax.set_title("Genus composition  (lines=sim, dots=Dieckow 2024)")
    ax.set_ylim(0, 0.8); ax.legend(fontsize=7, ncol=2)
    ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    p = outdir / "nife_spatial_scalars.png"
    fig.savefig(p, dpi=180, bbox_inches="tight"); print(f"Saved: {p}")

    # ── B: spatial snapshots (species fractions) at Week 1, 2, 3 ────────
    snap_weeks = [1, 2, 3]
    snaps = [_snap_at_week(history, w) for w in snap_weeks]
    n_rows = len(SPECIES) + 2   # species + glc + o2

    fig, axes = plt.subplots(n_rows, 3, figsize=(12, n_rows * 1.8 + 0.5))
    fig.suptitle("φ_i(z, x) spatial distribution  [z=0: implant surface, z=200µm: GCF]",
                 fontsize=10, fontweight="bold")

    for ci, (snap, tw) in enumerate(zip(snaps, snap_weeks)):
        tot = sum(snap["biomass"][sp] for sp in SPECIES) + 1e-30
        for ri, sp in enumerate(SPECIES):
            ax = axes[ri, ci]
            im = ax.imshow(snap["biomass"][sp] / tot, origin="lower",
                           vmin=0, vmax=0.8, cmap="Blues", aspect="auto",
                           extent=[0, NX*DX, 0, NZ*DZ])
            if ri == 0:
                ax.set_title(f"Week {tw}", fontweight="bold")
            if ci == 0:
                ax.set_ylabel(NAMES[sp], fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])

        ax = axes[len(SPECIES), ci]
        ax.imshow(snap["nutrients"]["glc"], origin="lower",
                  vmin=0, vmax=C_TOP["glc"], cmap="YlOrRd", aspect="auto",
                  extent=[0, NX*DX, 0, NZ*DZ])
        if ci == 0:
            ax.set_ylabel("Glucose (mM)", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

        ax = axes[len(SPECIES)+1, ci]
        ax.imshow(snap["nutrients"]["o2"], origin="lower",
                  vmin=0, vmax=C_TOP["o2"], cmap="Blues", aspect="auto",
                  extent=[0, NX*DX, 0, NZ*DZ])
        if ci == 0:
            ax.set_ylabel("O₂ (mM)", fontsize=8)
        ax.set_xlabel("x (µm)")
        ax.set_yticks([])

    plt.tight_layout()
    p = outdir / "nife_spatial_snapshots.png"
    fig.savefig(p, dpi=160, bbox_inches="tight"); print(f"Saved: {p}")

    # ── C: depth profiles at Week 3 ──────────────────────────────────────
    s3  = _snap_at_week(history, 3.0)
    tot = sum(s3["biomass"][sp] for sp in SPECIES) + 1e-30
    z   = np.arange(NZ) * DZ

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle("Depth profiles at Week 3  (z=0: implant surface, z=200µm: GCF)",
                 fontsize=10, fontweight="bold")

    ax = axes[0]
    for sp in SPECIES:
        frac = (s3["biomass"][sp] / tot).mean(axis=1)
        ax.plot(frac, z, color=COLORS[sp], lw=2, label=NAMES[sp])
    ax.set_xlabel("Relative abundance (lateral mean)"); ax.set_ylabel("Depth z (µm)")
    ax.set_title("Species stratification"); ax.set_ylim(0, NZ*DZ)
    ax.legend(fontsize=8); ax.spines[["top","right"]].set_visible(False)

    ax = axes[1]
    for nut, col, lab in [
        ("glc", "tomato", "Glucose"), ("o2", "steelblue", "O₂"),
        ("lac", "darkorange", "Lactate"), ("aa", "mediumpurple", "Amino acids"),
    ]:
        ax.plot(s3["nutrients"][nut].mean(axis=1), z, color=col, lw=2, label=lab)
    ax.set_xlabel("Concentration (mM)"); ax.set_ylabel("Depth z (µm)")
    ax.set_title("Nutrient gradients"); ax.set_ylim(0, NZ*DZ)
    ax.legend(); ax.spines[["top","right"]].set_visible(False)

    ax = axes[2]
    aerobe_frac = sum(
        (s3["biomass"][sp] / tot).mean(axis=1)
        for sp in ["Str", "Act", "Hae", "Rot"]
    )
    anaerobe_frac = sum(
        (s3["biomass"][sp] / tot).mean(axis=1)
        for sp in ["Vel", "Fus", "Por"]
    )
    o2_norm = s3["nutrients"]["o2"].mean(axis=1) / C_TOP["o2"]
    ax.fill_betweenx(z, 0, aerobe_frac,    alpha=0.5, color="steelblue",
                     label="Aerotolerant (Str+Act+Hae+Rot)")
    ax.fill_betweenx(z, 0, anaerobe_frac,  alpha=0.5, color="firebrick",
                     label="Strict anaerobe (Vel+Fus+Por)")
    ax.plot(o2_norm, z, "k--", lw=1.5, label="O₂ / O₂_top")
    ax.set_xlabel("Fraction / normalized O₂"); ax.set_ylabel("Depth z (µm)")
    ax.set_title("Aerobe/anaerobe zonation"); ax.set_ylim(0, NZ*DZ)
    ax.legend(fontsize=8); ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    p = outdir / "nife_spatial_depth.png"
    fig.savefig(p, dpi=180, bbox_inches="tight"); print(f"Saved: {p}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--weeks",  type=int,  default=3)
    pa.add_argument("--seed",   type=int,  default=42)
    pa.add_argument("--outdir", type=Path, default=Path("nife/comets/"))
    pa.add_argument("--plot-only", action="store_true")
    args = pa.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    hist_f = args.outdir / "spatial_history.npy"

    if args.plot_only and hist_f.exists():
        history = list(np.load(str(hist_f), allow_pickle=True))
    else:
        print(f"Grid: {NZ}×{NX} voxels ({NZ*DZ:.0f}×{NX*DX:.0f} µm), "
              f"DT={DT}h, QSS nutrient solver")
        history = run(n_weeks=args.weeks, seed=args.seed)
        np.save(str(hist_f), history)

    plot_all(history, args.outdir)

    # ── comparison table ─────────────────────────────────────────────────
    print(f"\n{'=== vs Dieckow 2024 ground truth ==='}")
    header = f"{'Metric':<28}" + "".join(f"  {'Wk'+str(w):>8}" for w in [1,2,3])
    print(header + "   source")
    print("-" * 62)
    weeks = np.array([s["week"] for s in history])

    def row(label, sim_vals, gt_vals, fmt=".2e"):
        fmt_fn = lambda v: f"{v:{fmt}}"
        sim_str = "".join(f"  {fmt_fn(v):>8}" for v in sim_vals)
        gt_str  = "".join(f"  {fmt_fn(v):>8}" for v in gt_vals)
        print(f"  {label:<26}{sim_str}   [sim]")
        print(f"  {'':<26}{gt_str}   [Dieckow 2024]")

    sim_vol  = [_snap_at_week(history, w)["volume_scaled"] for w in [1,2,3]]
    sim_live = [_snap_at_week(history, w)["live_frac"]     for w in [1,2,3]]
    row("Volume (µm³)", sim_vol,  GT_VOLUME, fmt=".2e")
    row("Live fraction", sim_live, GT_LIVE,  fmt=".2f")

    print(f"\n  Composition at Week 3:")
    s3 = _snap_at_week(history, 3)
    for sp in SPECIES:
        sv = s3["composition"][sp]
        gv = GT_COMP[sp][2]
        ok = "✓" if abs(sv - gv) < 0.10 else "✗"
        print(f"    {NAMES[sp]:<28} sim={sv:.1%}  Dieckow={gv:.1%}  {ok}")


if __name__ == "__main__":
    main()
