#!/usr/bin/env python3
"""
fem_spatial_extension.py

1D FEM Spatial Extension of the 5-Species Hamilton Biofilm Model.

Workflow
--------
1. Load θ_MAP estimated by estimate_reduced_nishioka.py  (JSON)
2. Set up 1D P1-FEM mesh on domain x ∈ [0, L]  (biofilm depth, dimensionless)
3. Initialise φᵢ(x, 0) with spatial heterogeneity
4. Time integration via Lie operator splitting:
     (a) Reaction  – advance Hamilton 0D model at every node independently
     (b) Diffusion – FEM backward-Euler solve for φᵢ spread across space
5. Visualise space-time evolution of all 5 species

PDE at each node x:
   ∂φᵢ/∂t = R_i^Hamilton(φ, ψ, γ ; θ)  +  Dᵢ ∂²φᵢ/∂x²

Internal variables ψᵢ, γ stay local (no diffusion).
After every diffusion step φ₀ is updated via the volume constraint:
   φ₀ = 1 − Σφᵢ

Usage
-----
  # With MAP estimate from TMCMC run
  python fem_spatial_extension.py \\
      --theta-json /path/to/theta_MAP.json \\
      --condition Dysbiotic_HOBIC

  # Standalone demo (built-in θ)
  python fem_spatial_extension.py --demo

  # Custom diffusion & mesh
  python fem_spatial_extension.py \\
      --theta-json theta_MAP.json \\
      --n-nodes 40 --n-macro 200 --D 0.001 0.001 0.0005 0.0005 0.0001
"""

from __future__ import annotations
import argparse
import sys
import json
import time
import warnings
import numpy as np
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import diags

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent  # Tmcmc202601/FEM/
_TMCMC_ROOT = _HERE.parent  # Tmcmc202601/
_MODEL_PATH = _TMCMC_ROOT / "tmcmc" / "program2602"
sys.path.insert(0, str(_MODEL_PATH))

try:
    from improved_5species_jit import (
        _compute_Q_vector_numpy,
        theta_to_matrices_numpy,
        HAS_NUMBA,
    )

    if HAS_NUMBA:
        from improved_5species_jit import _newton_step_jit  # Numba compiled
    HAVE_MODEL = True
except ImportError:
    HAVE_MODEL = False
    HAS_NUMBA = False
    warnings.warn(
        "Could not import improved_5species_jit from Tmcmc202601. "
        "Falling back to standalone solver.",
        stacklevel=2,
    )

# ── Species metadata ──────────────────────────────────────────────────────────
SPECIES_NAMES = ["S. oralis", "A. naeslundii", "Veillonella", "F. nucleatum", "P. gingivalis"]
SPECIES_SHORT = ["S.o", "A.n", "Vei", "F.n", "P.g"]
SPECIES_COLORS = ["royalblue", "forestgreen", "goldenrod", "mediumpurple", "crimson"]

THETA_NAMES = [
    "a11",
    "a12",
    "a22",
    "b1",
    "b2",
    "a33",
    "a34",
    "a44",
    "b3",
    "b4",
    "a13",
    "a14",
    "a23",
    "a24",
    "a55",
    "b5",
    "a15",
    "a25",
    "a35",
    "a45",
]

# ── Default θ (mild-weight best MAP, 2026-02-18) ─────────────────────────────
# Use as fallback when no JSON is provided (--demo mode).
_THETA_DEMO = np.array(
    [
        1.34,
        -0.18,
        1.79,
        1.17,
        2.58,  # M1: S.o, A.n
        3.51,
        2.73,
        0.71,
        2.10,
        0.37,  # M2: Vei, F.n
        2.05,
        -0.15,
        3.56,
        0.16,  # M3: cross (a23 clamped to 3.56)
        0.12,
        0.32,  # M4: P.g self
        1.49,
        2.10,
        2.41,
        2.50,  # M5: P.g cross (a35, a45 clamped)
    ],
    dtype=np.float64,
)


# ── Standalone Q-vector (in case import fails) ────────────────────────────────
def _Q_standalone(
    phi_new,
    phi0_new,
    psi_new,
    gamma_new,
    phi_old,
    phi0_old,
    psi_old,
    dt,
    Kp1,
    Eta,
    EtaPhi,
    c,
    alpha,
    K_hill,
    n_hill,
    A,
    b_diag,
    active_mask,
):
    """Pure-numpy residual (identical to _compute_Q_vector_numpy)."""
    eps = 1e-12
    Q = np.zeros(12)
    phidot = (phi_new - phi_old) / dt
    phi0dot = (phi0_new - phi0_old) / dt
    psidot = (psi_new - psi_old) / dt
    Ia = A @ (phi_new * psi_new)
    if K_hill > 1e-9 and active_mask[4] == 1:
        fn = max(phi_new[3] * psi_new[3], 0.0)
        num = fn**n_hill
        den = K_hill**n_hill + num
        Ia[4] *= (num / den) if den > eps else 0.0
    for i in range(5):
        if active_mask[i]:
            t1 = Kp1 * (2 - 4 * phi_new[i]) / ((phi_new[i] - 1) ** 3 * phi_new[i] ** 3)
            t2 = (1 / Eta[i]) * (
                gamma_new
                + (EtaPhi[i] + Eta[i] * psi_new[i] ** 2) * phidot[i]
                + Eta[i] * phi_new[i] * psi_new[i] * psidot[i]
            )
            t3 = (c / Eta[i]) * psi_new[i] * Ia[i]
            Q[i] = t1 + t2 - t3
        else:
            Q[i] = phi_new[i]
    Q[5] = gamma_new + Kp1 * (2 - 4 * phi0_new) / ((phi0_new - 1) ** 3 * phi0_new**3) + phi0dot
    for i in range(5):
        if active_mask[i]:
            t1 = (-2 * Kp1) / ((psi_new[i] - 1) ** 2 * psi_new[i] ** 3) - (2 * Kp1) / (
                (psi_new[i] - 1) ** 3 * psi_new[i] ** 2
            )
            t2 = (b_diag[i] * alpha / Eta[i]) * psi_new[i]
            t3 = phi_new[i] * psi_new[i] * phidot[i] + phi_new[i] ** 2 * psidot[i]
            t4 = (c / Eta[i]) * phi_new[i] * Ia[i]
            Q[6 + i] = t1 + t2 + t3 - t4
        else:
            Q[6 + i] = psi_new[i]
    Q[11] = np.sum(phi_new) + phi0_new - 1.0
    return Q


def _theta_to_matrices_standalone(theta):
    """Pure-numpy theta→(A, b_diag)."""
    A = np.zeros((5, 5))
    b = np.zeros(5)
    A[0, 0] = theta[0]
    A[0, 1] = A[1, 0] = theta[1]
    A[1, 1] = theta[2]
    b[0] = theta[3]
    b[1] = theta[4]
    A[2, 2] = theta[5]
    A[2, 3] = A[3, 2] = theta[6]
    A[3, 3] = theta[7]
    b[2] = theta[8]
    b[3] = theta[9]
    A[0, 2] = A[2, 0] = theta[10]
    A[0, 3] = A[3, 0] = theta[11]
    A[1, 2] = A[2, 1] = theta[12]
    A[1, 3] = A[3, 1] = theta[13]
    A[4, 4] = theta[14]
    b[4] = theta[15]
    A[0, 4] = A[4, 0] = theta[16]
    A[1, 4] = A[4, 1] = theta[17]
    A[2, 4] = A[4, 2] = theta[18]
    A[3, 4] = A[4, 3] = theta[19]
    return A, b


# ── Newton step (Python fallback) ─────────────────────────────────────────────
def _newton_step_python(
    g_prev,
    dt,
    A,
    b_diag,
    Kp1,
    Eta,
    EtaPhi,
    c,
    alpha,
    K_hill,
    n_hill,
    active_mask,
    eps_tol=1e-6,
    max_iter=30,
):
    """One backward-Euler Hamilton step via Newton-Raphson (Python/numpy)."""
    Q_fn = _compute_Q_vector_numpy if HAVE_MODEL else _Q_standalone
    eps = 1e-10
    g = g_prev.copy()
    for i in range(5):
        if active_mask[i]:
            g[i] = np.clip(g[i], eps, 1 - eps)
            g[6 + i] = np.clip(g[6 + i], eps, 1 - eps)
        else:
            g[i] = g[6 + i] = 0.0
    g[5] = np.clip(g[5], eps, 1 - eps)

    for _ in range(max_iter):
        Q = Q_fn(
            g[0:5].copy(),
            g[5],
            g[6:11].copy(),
            g[11],
            g_prev[0:5],
            g_prev[5],
            g_prev[6:11],
            dt,
            Kp1,
            Eta,
            EtaPhi,
            c,
            alpha,
            K_hill,
            n_hill,
            A,
            b_diag,
            active_mask,
        )
        res = np.linalg.norm(Q)
        if res < eps_tol:
            break
        # Finite-difference Jacobian (12 × 12 evaluations)
        h_fd = max(1e-7, res * 1e-4)
        J = np.zeros((12, 12))
        for j in range(12):
            gp = g.copy()
            gp[j] += h_fd
            Qp = Q_fn(
                gp[0:5].copy(),
                gp[5],
                gp[6:11].copy(),
                gp[11],
                g_prev[0:5],
                g_prev[5],
                g_prev[6:11],
                dt,
                Kp1,
                Eta,
                EtaPhi,
                c,
                alpha,
                K_hill,
                n_hill,
                A,
                b_diag,
                active_mask,
            )
            J[:, j] = (Qp - Q) / h_fd
        try:
            delta = np.linalg.solve(J, -Q)
        except np.linalg.LinAlgError:
            break
        g += delta
        for i in range(5):
            if active_mask[i]:
                g[i] = np.clip(g[i], eps, 1 - eps)
                g[6 + i] = np.clip(g[6 + i], eps, 1 - eps)
            else:
                g[i] = g[6 + i] = 0.0
        g[5] = np.clip(g[5], eps, 1 - eps)
    return g


def hamilton_node_step(g_prev, dt, A, b_diag, solver_params):
    """
    Advance one Hamilton backward-Euler time step at a single spatial node.
    Uses Numba JIT if available, otherwise Python fallback.
    """
    sp = solver_params
    if HAS_NUMBA and HAVE_MODEL:
        g_new = g_prev.copy()
        K_buf = np.zeros((12, 12))
        Q_buf = np.zeros(12)
        _newton_step_jit(
            g_prev,
            dt,
            sp["Kp1"],
            sp["Eta"],
            sp["EtaPhi"],
            sp["c"],
            sp["alpha"],
            sp["K_hill"],
            sp["n_hill"],
            A,
            b_diag,
            sp["eps_tol"],
            sp["max_newton_iter"],
            sp["active_mask"],
            g_new,
            K_buf,
            Q_buf,
        )
        return g_new
    else:
        return _newton_step_python(
            g_prev,
            dt,
            A,
            b_diag,
            sp["Kp1"],
            sp["Eta"],
            sp["EtaPhi"],
            sp["c"],
            sp["alpha"],
            sp["K_hill"],
            sp["n_hill"],
            sp["active_mask"],
            eps_tol=sp["eps_tol"],
            max_iter=sp["max_newton_iter"],
        )


# ── FEM assembly (1D P1 elements, uniform mesh) ───────────────────────────────
def assemble_1d_fem(n_nodes: int, L: float):
    """
    Assemble 1D P1 FEM global mass M and stiffness K on uniform mesh.

    Boundary conditions: zero-flux (Neumann) at both ends — natural BC,
    no extra modifications needed.

    Returns
    -------
    M : scipy sparse CSR  (n_nodes × n_nodes)  lumped mass
    K : scipy sparse CSR  (n_nodes × n_nodes)  stiffness
    """
    n_el = n_nodes - 1
    h = L / n_el

    # Consistent mass matrix
    main_M = np.full(n_nodes, 2 * h / 3)
    main_M[0] = h / 3
    main_M[-1] = h / 3
    off_M = np.full(n_nodes - 1, h / 6)
    M = diags([off_M, main_M, off_M], [-1, 0, 1], format="csr")

    # Stiffness matrix
    main_K = np.full(n_nodes, 2 / h)
    main_K[0] = 1 / h
    main_K[-1] = 1 / h
    off_K = np.full(n_nodes - 1, -1 / h)
    K = diags([off_K, main_K, off_K], [-1, 0, 1], format="csr")

    return M, K


# ── Spatial initial conditions ────────────────────────────────────────────────
def make_initial_state(
    n_nodes: int, L: float, active_mask: np.ndarray, phi_init_mode: str = "gradient"
) -> np.ndarray:
    """
    Build initial state G[n_nodes, 12]:
      columns 0-4  : φᵢ  (volume fractions, 5 species)
      column  5    : φ₀  (void fraction)
      columns 6-10 : ψᵢ  (internal variables)
      column  11   : γ   (potential)

    phi_init_mode
    -------------
    "gradient" : commensal species uniform, pathogens seeded at x=0
    "uniform"  : all species uniform (same as 0D model)
    "dysbiotic": pathogen seeds dominate near x=0
    """
    x = np.linspace(0, L, n_nodes)
    G = np.zeros((n_nodes, 12))
    eps = 1e-6

    if phi_init_mode == "uniform":
        phi_base = np.array([0.12, 0.12, 0.08, 0.05, 0.01])
    elif phi_init_mode == "dysbiotic":
        phi_base = np.array([0.10, 0.10, 0.10, 0.10, 0.00])
    else:  # "gradient" (default)
        phi_base = np.array([0.12, 0.12, 0.08, 0.05, 0.00])

    for k, xk in enumerate(x):
        phi = phi_base.copy()

        if phi_init_mode == "gradient":
            # F.nucleatum concentrated near implant surface (x=0)
            phi[3] = 0.05 + 0.10 * np.exp(-15 * xk / L)
            # P.gingivalis small seed at x=0 only
            phi[4] = 0.02 * np.exp(-30 * xk / L)
        elif phi_init_mode == "dysbiotic":
            # Commensal right side, pathogen left side
            phi[0] = 0.10 + 0.15 * (xk / L)
            phi[1] = 0.10 + 0.10 * (xk / L)
            phi[4] = 0.15 * np.exp(-10 * xk / L)

        # Apply active mask
        for i in range(5):
            if not active_mask[i]:
                phi[i] = 0.0
            else:
                phi[i] = max(phi[i], eps)

        # Volume constraint
        phi_sum = np.sum(phi)
        if phi_sum > 1.0 - eps:
            phi *= (1.0 - eps) / phi_sum
            phi_sum = np.sum(phi)
        phi0 = 1.0 - phi_sum

        G[k, 0:5] = phi
        G[k, 5] = phi0
        G[k, 6:11] = np.where(active_mask == 1, 0.999, 0.0)  # ψᵢ initialised high
        G[k, 11] = 0.0  # γ

    return G


# ── Main simulation class ─────────────────────────────────────────────────────
class FEMBiofilmSimulation:
    """
    1D FEM spatial extension of the Hamilton 5-species biofilm model.

    Parameters
    ----------
    theta       : array (20,) – estimated parameter vector
    n_nodes     : number of spatial nodes
    L           : domain length (dimensionless)
    D_eff       : diffusion coefficients per species (5,)
    dt_h        : Hamilton micro-step (dimensionless time)
    n_react_sub : Hamilton sub-steps per macro step
    n_macro     : total number of macro steps
    solver_params : dict with Hamilton solver settings
    save_every  : save snapshot every N macro steps
    phi_init_mode : initial condition type
    """

    def __init__(
        self,
        theta: np.ndarray,
        n_nodes: int = 30,
        L: float = 1.0,
        D_eff: np.ndarray | None = None,
        dt_h: float = 1e-5,
        n_react_sub: int = 50,
        n_macro: int = 100,
        solver_params: dict | None = None,
        save_every: int = 5,
        phi_init_mode: str = "gradient",
    ):
        self.theta = np.asarray(theta, dtype=np.float64)
        self.n_nodes = n_nodes
        self.L = L
        self.dt_h = dt_h
        self.n_react_sub = n_react_sub
        self.n_macro = n_macro
        self.save_every = save_every
        self.phi_init_mode = phi_init_mode

        if D_eff is None:
            # Default: commensal species diffuse faster, pathogen slower
            D_eff = np.array([0.001, 0.001, 0.0008, 0.0005, 0.0002])
        self.D_eff = np.asarray(D_eff, dtype=np.float64)

        # Solver settings
        if solver_params is None:
            solver_params = {}
        self.sp = {
            "Kp1": solver_params.get("Kp1", 1e-4),
            "Eta": solver_params.get("Eta", np.ones(5)),
            "EtaPhi": solver_params.get("EtaPhi", np.ones(5)),
            "c": solver_params.get("c", 100.0),
            "alpha": solver_params.get("alpha", 100.0),
            "K_hill": solver_params.get("K_hill", 0.05),
            "n_hill": solver_params.get("n_hill", 4.0),
            "active_mask": solver_params.get("active_mask", np.ones(5, dtype=np.int64)),
            "eps_tol": solver_params.get("eps_tol", 1e-6),
            "max_newton_iter": solver_params.get("max_newton_iter", 50),
        }

        # Build interaction matrices from θ
        fn = theta_to_matrices_numpy if HAVE_MODEL else _theta_to_matrices_standalone
        self.A, self.b_diag = fn(self.theta)

        # FEM matrices
        self.M, self.K_fem = assemble_1d_fem(n_nodes, L)

        # Pre-build diffusion system matrices: (M + dt_diff * Dᵢ * K) per species
        dt_diff = dt_h * n_react_sub
        self.lhs = [(self.M + dt_diff * self.D_eff[i] * self.K_fem).toarray() for i in range(5)]

        # Spatial coordinate
        self.x = np.linspace(0, L, n_nodes)

        # Initialise state
        self.G = make_initial_state(n_nodes, L, self.sp["active_mask"], phi_init_mode)

        # Storage
        self.snapshots = []  # list of (t, G_copy)
        self.t_current = 0.0

    def _reaction_step(self):
        """Advance every node n_react_sub Hamilton micro-steps (reaction only)."""
        for k in range(self.n_nodes):
            g = self.G[k].copy()
            for _ in range(self.n_react_sub):
                g = hamilton_node_step(g, self.dt_h, self.A, self.b_diag, self.sp)
            self.G[k] = g

    def _diffusion_step(self):
        """
        FEM backward-Euler diffusion step for each species.
        Solves: (M + Δt_diff · Dᵢ · K) φᵢ_new = M φᵢ_old
        Then renormalises φ₀.
        """
        eps = 1e-10
        for i in range(5):
            if not self.sp["active_mask"][i]:
                continue
            phi_old = self.G[:, i]
            rhs = self.M @ phi_old
            phi_new = np.linalg.solve(self.lhs[i], rhs)
            phi_new = np.clip(phi_new, 0.0, 1.0)
            self.G[:, i] = phi_new

        # Update void fraction to maintain volume constraint
        phi_sum = np.sum(self.G[:, 0:5], axis=1)
        self.G[:, 5] = np.clip(1.0 - phi_sum, 0.0, 1.0)

    def run(self, verbose: bool = True):
        """Execute the full space-time simulation."""
        t0 = time.time()
        dt_macro = self.dt_h * self.n_react_sub
        total_time = dt_macro * self.n_macro

        print("FEM Biofilm Simulation")
        print(f"  Nodes     : {self.n_nodes}  |  Domain L = {self.L}")
        print(f"  dt_h      : {self.dt_h:.1e}  |  n_react_sub = {self.n_react_sub}")
        print(f"  Macro steps: {self.n_macro}  |  dt_macro = {dt_macro:.1e}")
        print(f"  Total Hamilton time: {total_time:.4f}")
        print(f"  Diffusion D_eff: {self.D_eff}")
        print(f"  Using Numba: {HAS_NUMBA and HAVE_MODEL}")
        print(f"  Initial mode: {self.phi_init_mode}")
        print()

        # Save initial snapshot
        self.snapshots.append((0.0, self.G.copy()))

        for step in range(self.n_macro):
            self._reaction_step()
            self._diffusion_step()
            self.t_current += dt_macro

            if (step + 1) % self.save_every == 0:
                self.snapshots.append((self.t_current, self.G.copy()))

            if verbose and (step + 1) % max(1, self.n_macro // 10) == 0:
                phi_mean = np.mean(self.G[:, 0:5], axis=0)
                elapsed = time.time() - t0
                pct = (step + 1) / self.n_macro * 100
                print(
                    f"  [{pct:5.1f}%] t={self.t_current:.4f}  "
                    f"φ̄=[{', '.join(f'{p:.3f}' for p in phi_mean)}]  "
                    f"elapsed={elapsed:.1f}s"
                )

        elapsed = time.time() - t0
        print(f"\nSimulation complete in {elapsed:.1f}s  |  {len(self.snapshots)} snapshots saved.")

    def plot_results(self, out_dir: Path | None = None, condition_label: str = ""):
        """Produce 3 figures:
        Fig 1 – Space-time heatmap per species (φᵢ(x, t))
        Fig 2 – Time series at x=0, L/2, L
        Fig 3 – Spatial profile at t=0, T/4, T/2, T
        """
        if out_dir is None:
            out_dir = _HERE / "_fem_results"
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        times = [s[0] for s in self.snapshots]
        G_stack = np.stack([s[1] for s in self.snapshots], axis=0)  # (n_snap, n_nodes, 12)
        phi_t = G_stack[:, :, 0:5]  # (n_snap, n_nodes, 5)
        x = self.x
        t_arr = np.array(times)

        # ── Fig 1: Heatmaps ──────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 5, figsize=(18, 4.5), sharey=True)
        fig.suptitle(
            f"Space-Time Evolution of φᵢ(x, t){' — ' + condition_label if condition_label else ''}",
            fontsize=13,
            fontweight="bold",
        )

        for i, ax in enumerate(axes):
            z = phi_t[:, :, i].T  # (n_nodes, n_snap)
            im = ax.pcolormesh(t_arr, x, z, cmap="viridis", vmin=0.0, vmax=min(0.5, z.max() + 0.05))
            ax.set_title(SPECIES_NAMES[i], color=SPECIES_COLORS[i], fontweight="bold")
            ax.set_xlabel("Hamilton time t")
            if i == 0:
                ax.set_ylabel("Spatial position x")
            plt.colorbar(im, ax=ax, label="φᵢ")

        fig.tight_layout()
        p1 = out_dir / "fig1_spacetime_heatmaps.png"
        fig.savefig(p1, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {p1}")

        # ── Fig 2: Time series at 3 positions ────────────────────────────────
        nodes_of_interest = {
            "x=0 (surface)": 0,
            "x=L/2 (mid)": self.n_nodes // 2,
            "x=L (bulk)": self.n_nodes - 1,
        }
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
        fig.suptitle(
            f"Time Series at Key Spatial Locations{' — ' + condition_label if condition_label else ''}",
            fontsize=13,
            fontweight="bold",
        )
        for ax, (label, k) in zip(axes, nodes_of_interest.items()):
            for i in range(5):
                ax.plot(
                    t_arr, phi_t[:, k, i], color=SPECIES_COLORS[i], label=SPECIES_SHORT[i], lw=1.8
                )
            ax.set_title(label)
            ax.set_xlabel("Hamilton time t")
            ax.set_ylim(bottom=0)
            ax.legend(fontsize=8)
            if ax is axes[0]:
                ax.set_ylabel("φᵢ (volume fraction)")

        fig.tight_layout()
        p2 = out_dir / "fig2_time_series_at_nodes.png"
        fig.savefig(p2, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {p2}")

        # ── Fig 3: Spatial profiles at selected times ─────────────────────────
        n_snap = len(times)
        raw = [0, n_snap // 4, n_snap // 2, n_snap - 1]
        snap_indices = sorted(set(min(max(0, i), n_snap - 1) for i in raw))
        # Pad to 4 if fewer unique indices
        while len(snap_indices) < 4:
            snap_indices.append(snap_indices[-1])
        snap_labels = [f"t={times[s]:.4f}" for s in snap_indices]

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(
            f"Spatial Profiles at Key Times{' — ' + condition_label if condition_label else ''}",
            fontsize=13,
            fontweight="bold",
        )
        for ax, si, lbl in zip(axes.flat, snap_indices, snap_labels):
            for i in range(5):
                ax.plot(x, phi_t[si, :, i], color=SPECIES_COLORS[i], label=SPECIES_SHORT[i], lw=1.8)
            ax.set_title(lbl)
            ax.set_xlabel("Position x")
            ax.set_ylabel("φᵢ (volume fraction)")
            ax.set_ylim(bottom=0)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        p3 = out_dir / "fig3_spatial_profiles.png"
        fig.savefig(p3, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {p3}")

        # ── Fig 4: Stacked area – spatial composition at final time ───────────
        fig, ax = plt.subplots(figsize=(9, 4))
        final_phi = phi_t[-1, :, :]  # (n_nodes, 5)
        ax.stackplot(
            x,
            [final_phi[:, i] for i in range(5)],
            labels=SPECIES_NAMES,
            colors=SPECIES_COLORS,
            alpha=0.75,
        )
        ax.set_xlabel("Position x", fontsize=11)
        ax.set_ylabel("Volume fraction φᵢ", fontsize=11)
        ax.set_title(
            f"Final Spatial Composition (t={times[-1]:.4f})"
            + (f" — {condition_label}" if condition_label else ""),
            fontsize=12,
        )
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xlim(0, self.L)
        ax.set_ylim(0)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p4 = out_dir / "fig4_final_composition.png"
        fig.savefig(p4, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {p4}")

    def save_snapshots(self, out_dir: Path):
        """Save G snapshots as numpy arrays for post-processing."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        times = np.array([s[0] for s in self.snapshots])
        G_stack = np.stack([s[1] for s in self.snapshots], axis=0)
        np.save(out_dir / "snapshots_G.npy", G_stack)
        np.save(out_dir / "snapshots_t.npy", times)
        np.save(out_dir / "mesh_x.npy", self.x)
        np.save(out_dir / "theta_MAP.npy", self.theta)
        print(f"  Snapshots saved to {out_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def _default_theta_map_path() -> Path | None:
    """Try to find the most recent theta_MAP.json in data_5species/_runs."""
    base = _TMCMC_ROOT / "data_5species" / "_runs"
    if not base.exists():
        return None
    # sweep_pg baseline is the best available run (Feb 2026)
    candidate = base / "sweep_pg_20260217_081459" / "dh_baseline" / "theta_MAP.json"
    if candidate.exists():
        return candidate
    # Fall back: latest JSON by modification time
    maps = sorted(base.rglob("theta_MAP.json"), key=lambda p: p.stat().st_mtime)
    return maps[-1] if maps else None


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--theta-json",
        type=Path,
        default=None,
        help="Path to theta_MAP.json from estimate_reduced_nishioka.py",
    )
    p.add_argument(
        "--demo", action="store_true", help="Use built-in demo parameters (no JSON needed)"
    )
    p.add_argument(
        "--condition", default="", help="Condition label for plot titles (e.g. Dysbiotic_HOBIC)"
    )
    p.add_argument("--n-nodes", type=int, default=30)
    p.add_argument("--L", type=float, default=1.0)
    p.add_argument("--dt-h", type=float, default=1e-5, help="Hamilton micro time step")
    p.add_argument("--n-react-sub", type=int, default=50, help="Hamilton sub-steps per macro step")
    p.add_argument("--n-macro", type=int, default=100, help="Total macro (splitting) steps")
    p.add_argument(
        "--D",
        type=float,
        nargs=5,
        default=[0.001, 0.001, 0.0008, 0.0005, 0.0002],
        help="Diffusion coefficient per species (5 values)",
    )
    p.add_argument("--K-hill", type=float, default=0.05)
    p.add_argument("--n-hill", type=float, default=4.0)
    p.add_argument(
        "--init-mode",
        default="gradient",
        choices=["gradient", "uniform", "dysbiotic"],
        help="Spatial initial condition type",
    )
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load θ ───────────────────────────────────────────────────────────────
    if args.demo:
        theta = _THETA_DEMO.copy()
        print("Using built-in demo θ (mild-weight MAP, 2026-02-18).")
    else:
        json_path = args.theta_json or _default_theta_map_path()
        if json_path is None or not json_path.exists():
            print("No theta_MAP.json found. Pass --theta-json or use --demo.")
            sys.exit(1)
        with open(json_path) as f:
            data = json.load(f)
        # Support both "theta_full" and "theta_sub" keys
        theta = np.asarray(data.get("theta_full") or data.get("theta_sub"), dtype=np.float64)
        print(f"Loaded θ from: {json_path}")

    print(f"\nθ vector ({len(theta)} params):")
    for i, (n, v) in enumerate(zip(THETA_NAMES, theta)):
        print(f"  [{i:2d}] {n:5s} = {v:8.4f}")
    print()

    # ── Build solver params ───────────────────────────────────────────────────
    solver_params = {
        "Kp1": 1e-4,
        "Eta": np.ones(5),
        "EtaPhi": np.ones(5),
        "c": 100.0,
        "alpha": 100.0,
        "K_hill": args.K_hill,
        "n_hill": args.n_hill,
        "active_mask": np.ones(5, dtype=np.int64),
        "eps_tol": 1e-6,
        "max_newton_iter": 50,
    }

    # ── Run simulation ────────────────────────────────────────────────────────
    out_dir = args.out_dir or (_HERE / "_fem_results")
    sim = FEMBiofilmSimulation(
        theta=theta,
        n_nodes=args.n_nodes,
        L=args.L,
        D_eff=np.array(args.D),
        dt_h=args.dt_h,
        n_react_sub=args.n_react_sub,
        n_macro=args.n_macro,
        solver_params=solver_params,
        save_every=args.save_every,
        phi_init_mode=args.init_mode,
    )

    sim.run(verbose=True)

    # ── Save & visualise ──────────────────────────────────────────────────────
    print(f"\nSaving results to: {out_dir}")
    sim.save_snapshots(out_dir)
    sim.plot_results(out_dir, condition_label=args.condition)

    print("\nDone.")


if __name__ == "__main__":
    main()
