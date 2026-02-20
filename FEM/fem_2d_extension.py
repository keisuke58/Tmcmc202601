#!/usr/bin/env python3
"""
fem_2d_extension.py  –  2D reaction-diffusion FEM for the 5-species Hamilton biofilm model

Domain  : [0, Lx] × [0, Ly]   (x = depth perpendicular to substratum, y = lateral)
Grid    : Nx × Ny uniform nodes,  node index k = ix*Ny + iy  (row-major)

Method  : Lie operator splitting per macro step
  ① Reaction  – Numba-parallel 0D Hamilton Newton at every (ix,iy) node  (prange)
  ② Diffusion – 2D backward-Euler with precomputed SuperLU factorisation  (scipy.sparse)
                L_2D = kron(Lx, Iy) + kron(Ix, Ly)  (Neumann BCs on all four walls)

State
  G  : (Nx, Ny, 12)   Hamilton state  [φ₁..φ₅, φ₀, ψ₁..ψ₅, γ]

Outputs (saved to --out-dir)
  snapshots_phi.npy   (n_snap, 5, Nx, Ny)
  snapshots_t.npy     (n_snap,)
  mesh_x.npy          (Nx,)
  mesh_y.npy          (Ny,)
  theta_MAP.npy       (20,)

Usage
-----
  python fem_2d_extension.py \\
      --theta-json ../data_5species/_runs/.../theta_MAP.json \\
      --condition "dh_baseline" \\
      --nx 20 --ny 20 --n-macro 100 --n-react-sub 50 \\
      --out-dir _results_2d/dh_baseline
"""
import argparse, json, time, sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# ── module paths ─────────────────────────────────────────────────────────────
_HERE       = Path(__file__).resolve().parent
_TMCMC_ROOT = _HERE.parent                          # Tmcmc202601/
_MODEL_PATH = _TMCMC_ROOT / "tmcmc" / "program2602"
sys.path.insert(0, str(_MODEL_PATH))                # import improved_5species_jit directly

try:
    from improved_5species_jit import (
        _newton_step_jit,   # Numba JIT Newton step (full signature)
        HAS_NUMBA,
    )
    from numba import njit, prange
    _HAVE_MODEL = True
except ImportError:
    _HAVE_MODEL = False
    HAS_NUMBA   = False


# ── Numba parallel reaction kernel ───────────────────────────────────────────
# Defined only when Numba is available.  Called once per macro step.
# Each prange worker allocates its own Newton buffers (g_new_buf, K_buf, Q_buf)
# so there are no race conditions.
if HAS_NUMBA and _HAVE_MODEL:
    @njit(parallel=True, cache=False)
    def _reaction_step_2d(
        G_flat,          # (N, 12)  Hamilton state (row-major over grid)
        A,               # (5, 5)   interaction matrix from theta
        b_diag,          # (5,)     self-growth rates from theta
        n_sub,           # int      Hamilton sub-steps per macro step
        dt_h,            # float    Hamilton time step
        Kp1,             # float
        Eta_vec,         # (5,)
        Eta_phi_vec,     # (5,)
        c_val,           # float
        alpha_val,       # float
        K_hill,          # float
        n_hill,          # float
        eps_tol,         # float
        active_mask,     # (5,) int
    ):
        N = G_flat.shape[0]
        G_out = np.empty_like(G_flat)
        for k in prange(N):
            g = G_flat[k].copy()
            # Per-thread Newton buffers (no shared state)
            g_new_buf = np.zeros(12)
            K_buf     = np.zeros((12, 12))
            Q_buf     = np.zeros(12)
            for _ in range(n_sub):
                _newton_step_jit(
                    g, dt_h,
                    Kp1, Eta_vec, Eta_phi_vec,
                    c_val, alpha_val,
                    K_hill, n_hill,
                    A, b_diag,
                    eps_tol, 50, active_mask,
                    g_new_buf, K_buf, Q_buf,
                )
                g[:] = g_new_buf[:]
            G_out[k] = g
        return G_out


# ── theta → A, b_diag ────────────────────────────────────────────────────────
def _theta_to_matrices(theta: np.ndarray):
    """Convert 20-vector θ to interaction matrix A (5×5) and b_diag (5,)."""
    A      = np.zeros((5, 5), dtype=np.float64)
    b_diag = np.zeros(5,      dtype=np.float64)
    # M1: S.o, A.n
    A[0,0] = theta[0];  A[0,1] = theta[1];  A[1,0] = theta[1]
    A[1,1] = theta[2];  b_diag[0] = theta[3]; b_diag[1] = theta[4]
    # M2: Vei, F.n
    A[2,2] = theta[5];  A[2,3] = theta[6];  A[3,2] = theta[6]
    A[3,3] = theta[7];  b_diag[2] = theta[8]; b_diag[3] = theta[9]
    # M3: cross
    A[0,2] = theta[10]; A[2,0] = theta[10]
    A[0,3] = theta[11]; A[3,0] = theta[11]
    A[1,2] = theta[12]; A[2,1] = theta[12]
    A[1,3] = theta[13]; A[3,1] = theta[13]
    # M4: P.g self
    A[4,4] = theta[14]; b_diag[4] = theta[15]
    # M5: P.g cross
    A[0,4] = theta[16]; A[4,0] = theta[16]
    A[1,4] = theta[17]; A[4,1] = theta[17]
    A[2,4] = theta[18]; A[4,2] = theta[18]
    A[3,4] = theta[19]; A[4,3] = theta[19]
    return A, b_diag


# ── 2D Laplacian (finite-difference, Neumann BCs) ────────────────────────────
def _build_1d_laplacian_neumann(N: int, h: float) -> sp.csr_matrix:
    """1-D Neumann Laplacian / h²  (ghost-node: half-stencil at walls)."""
    h2    = h * h
    diags = np.full(N, -2.0 / h2)
    diags[0] = diags[-1] = -1.0 / h2   # Neumann → ghost = interior, stencil halves
    off   = np.ones(N - 1) / h2
    return sp.diags([off, diags, off], [-1, 0, 1], format="csr")


def build_2d_laplacian(Nx: int, Ny: int, dx: float, dy: float) -> sp.csr_matrix:
    """2-D Laplacian  L = kron(Lx, Iy) + kron(Ix, Ly).

    Node ordering  k = ix*Ny + iy  (row-major, x outer / depth, y inner / lateral).
    Shape: (Nx*Ny, Nx*Ny), Neumann on all four walls.
    """
    Lx = _build_1d_laplacian_neumann(Nx, dx)
    Ly = _build_1d_laplacian_neumann(Ny, dy)
    Ix = sp.eye(Nx, format="csr")
    Iy = sp.eye(Ny, format="csr")
    return sp.kron(Lx, Iy, format="csr") + sp.kron(Ix, Ly, format="csr")


# ── diffusion coefficients ────────────────────────────────────────────────────
_D_EFF = np.array([1e-3, 1e-3, 8e-4, 5e-4, 2e-4])   # S.o, A.n, Vei, F.n, P.g


# ── Simulation class ──────────────────────────────────────────────────────────
class FEM2DBiofilm:
    SPECIES = ["S.oralis", "A.naeslundii", "Veillonella", "F.nucleatum", "P.gingivalis"]

    # Physical constants (matching BiofilmNewtonSolver5S defaults)
    _KP1       = 1e-4
    _C_CONST   = 100.0
    _ALPHA     = 100.0
    _K_HILL    = 0.0
    _N_HILL    = 2.0
    _EPS_TOL   = 1e-6

    def __init__(
        self,
        theta: np.ndarray,
        Nx: int = 20,
        Ny: int = 20,
        Lx: float = 1.0,
        Ly: float = 1.0,
        n_macro: int = 100,
        n_react_sub: int = 50,
        dt_h: float = 1e-5,
        D_eff: np.ndarray = None,
        save_every: int = 5,
        condition: str = "",
    ):
        self.theta       = theta.astype(np.float64)
        self.Nx, self.Ny = Nx, Ny
        self.Lx, self.Ly = Lx, Ly
        self.n_macro     = n_macro
        self.n_react_sub = n_react_sub
        self.dt_h        = dt_h
        self.D_eff       = _D_EFF.copy() if D_eff is None else np.asarray(D_eff, dtype=np.float64)
        self.save_every  = save_every
        self.condition   = condition

        self.dx        = Lx / max(Nx - 1, 1)
        self.dy        = Ly / max(Ny - 1, 1)
        self.dt_macro  = dt_h * n_react_sub
        self.t_total   = self.dt_macro * n_macro

        self.x_mesh = np.linspace(0, Lx, Nx)
        self.y_mesh = np.linspace(0, Ly, Ny)

        # Precompute A, b_diag from theta
        self.A, self.b_diag = _theta_to_matrices(self.theta)

        # Solver constants as float64 arrays (required by Numba)
        self._Eta_vec    = np.ones(5, dtype=np.float64)
        self._Eta_phi    = np.ones(5, dtype=np.float64)
        self._active     = np.ones(5, dtype=np.int64)

        # ── precompute SuperLU factorisations for 5 diffusion systems ─────
        print("Assembling 2D Laplacian and factorising ... ", end="", flush=True)
        N_nodes = Nx * Ny
        L    = build_2d_laplacian(Nx, Ny, self.dx, self.dy)
        I_sp = sp.eye(N_nodes, format="csr")
        self._solvers = []
        for D_i in self.D_eff:
            A_sys = (I_sp - self.dt_macro * D_i * L).tocsc()
            self._solvers.append(spla.factorized(A_sys))
        print("done.")

        # ── Numba warm-up ─────────────────────────────────────────────────
        self.use_numba = False
        if HAS_NUMBA and _HAVE_MODEL:
            try:
                _g0 = np.zeros((1, 12), dtype=np.float64)
                _g0[0, 5] = 1.0
                _A0 = self.A.copy()
                _b0 = self.b_diag.copy()
                _ = _reaction_step_2d(
                    _g0, _A0, _b0, 1, 1e-5,
                    self._KP1, self._Eta_vec, self._Eta_phi,
                    self._C_CONST, self._ALPHA,
                    self._K_HILL, self._N_HILL,
                    self._EPS_TOL, self._active,
                )
                self.use_numba = True
            except Exception as exc:
                print(f"  [warn] Numba warm-up failed: {exc}")
        print(f"Using Numba parallel: {self.use_numba}")

    # ── initial state ─────────────────────────────────────────────────────────
    def _make_G0(self, mode: str = "gradient") -> np.ndarray:
        """Build initial Hamilton state G (Nx, Ny, 12)."""
        Nx, Ny = self.Nx, self.Ny
        G      = np.zeros((Nx, Ny, 12), dtype=np.float64)
        rng    = np.random.default_rng(42)

        if mode == "gradient":
            def noise(s):
                return s * rng.standard_normal((Nx, Ny))
            # Commensals uniform
            G[:, :, 0] = (0.13 + noise(0.01)).clip(0)
            G[:, :, 1] = (0.13 + noise(0.01)).clip(0)
            # Veillonella
            G[:, :, 2] = (0.08 + noise(0.005)).clip(0)
            # F.nucleatum: surface-enriched
            xp_fn = np.exp(-3.0 * self.x_mesh / self.Lx)
            G[:, :, 3] = (0.05 * xp_fn[:, None] + noise(0.005)).clip(0)
            # P.gingivalis: focal seed at x=0, y-centre
            yc, ys = 0.5 * self.Ly, 0.1 * self.Ly
            yp_pg  = np.exp(-0.5 * ((self.y_mesh - yc) / ys) ** 2)
            xp_pg  = np.exp(-5.0 * self.x_mesh / self.Lx)
            G[:, :, 4] = (0.01 * xp_pg[:, None] * yp_pg[None, :] + noise(0.002)).clip(1e-6)
        else:
            for i, phi0 in enumerate([0.13, 0.13, 0.08, 0.06, 0.02]):
                G[:, :, i] = (phi0 + 0.01 * rng.standard_normal((Nx, Ny))).clip(0)

        phi_sum         = G[:, :, :5].sum(axis=2)
        G[:, :, 5]      = (1.0 - phi_sum).clip(0)   # void φ₀
        G[:, :, 6:11]   = G[:, :, :5]               # ψᵢ = φᵢ initial
        G[:, :, 11]     = 1.0                        # γ
        return G

    # ── operator splitting ────────────────────────────────────────────────────
    def _react(self, G: np.ndarray) -> np.ndarray:
        """Reaction step: Numba-parallel 0D Hamilton at each node."""
        if not self.use_numba:
            raise RuntimeError("Numba unavailable – 2D simulation requires Numba.")
        G_flat = G.reshape(self.Nx * self.Ny, 12)
        G_flat = _reaction_step_2d(
            G_flat, self.A, self.b_diag, self.n_react_sub, self.dt_h,
            self._KP1, self._Eta_vec, self._Eta_phi,
            self._C_CONST, self._ALPHA,
            self._K_HILL, self._N_HILL,
            self._EPS_TOL, self._active,
        )
        return G_flat.reshape(self.Nx, self.Ny, 12)

    def _diffuse(self, G: np.ndarray) -> np.ndarray:
        """Diffusion step: backward-Euler per species using precomputed SuperLU."""
        G_new = G.copy()
        for i, solve_i in enumerate(self._solvers):
            phi_new = solve_i(G[:, :, i].ravel()).clip(0)
            G_new[:, :, i] = phi_new.reshape(self.Nx, self.Ny)
        phi_sum         = G_new[:, :, :5].sum(axis=2)
        G_new[:, :, 5]  = (1.0 - phi_sum).clip(0)   # update void
        return G_new

    # ── main loop ─────────────────────────────────────────────────────────────
    def run(self):
        G          = self._make_G0("gradient")
        # Store φ as (5, Nx, Ny) → np.array gives (n_snap, 5, Nx, Ny)
        snaps_phi  = [G[:, :, :5].transpose(2, 0, 1).copy()]
        snaps_t    = [0.0]

        print(f"\n{'='*62}")
        print(f"2D FEM Biofilm  |  condition = {self.condition!r}")
        print(f"  Grid   : {self.Nx}×{self.Ny}  ({self.Nx*self.Ny} nodes)")
        print(f"  Domain : Lx={self.Lx:.2f}  Ly={self.Ly:.2f}")
        print(f"  dt_h   : {self.dt_h:.1e}  |  n_sub={self.n_react_sub}")
        print(f"  dt_mac : {self.dt_macro:.1e}  |  n_mac={self.n_macro}")
        print(f"  t_tot  : {self.t_total:.4f}")
        print(f"  D_eff  : {self.D_eff}")
        print(f"{'='*62}\n")

        t0 = time.perf_counter()
        for step in range(1, self.n_macro + 1):
            t = step * self.dt_macro
            G = self._react(G)
            G = self._diffuse(G)
            if step % self.save_every == 0 or step == self.n_macro:
                snaps_phi.append(G[:, :, :5].transpose(2, 0, 1).copy())
                snaps_t.append(t)
                phi_mean = G[:, :, :5].mean(axis=(0, 1))
                bar = "[" + ", ".join(f"{v:.3f}" for v in phi_mean) + "]"
                print(f"  [{100*step/self.n_macro:5.1f}%] t={t:.4f}  φ̄={bar}  "
                      f"elapsed={time.perf_counter()-t0:.1f}s")

        elapsed = time.perf_counter() - t0
        print(f"\nDone in {elapsed:.1f}s  |  {len(snaps_phi)} snapshots")
        return np.array(snaps_phi), np.array(snaps_t)

    def save(self, out_dir: Path, snaps_phi: np.ndarray, snaps_t: np.ndarray):
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "snapshots_phi.npy", snaps_phi)
        np.save(out_dir / "snapshots_t.npy",   snaps_t)
        np.save(out_dir / "mesh_x.npy",        self.x_mesh)
        np.save(out_dir / "mesh_y.npy",        self.y_mesh)
        np.save(out_dir / "theta_MAP.npy",     self.theta)
        print(f"Saved to: {out_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────
_RUNS_ROOT     = _TMCMC_ROOT / "data_5species" / "_runs"
_DEFAULT_THETA = str(
    _RUNS_ROOT / "sweep_pg_20260217_081459" / "dh_baseline" / "theta_MAP.json"
)

_PARAM_KEYS = [
    "a11","a12","a22","b1","b2",
    "a33","a34","a44","b3","b4",
    "a13","a14","a23","a24",
    "a55","b5",
    "a15","a25","a35","a45",
]


def _load_theta(path: str) -> np.ndarray:
    with open(path) as f:
        d = json.load(f)
    # Support both {"theta_full": [...]} and {"a11": v, ...} formats
    if "theta_full" in d:
        vec = np.array(d["theta_full"])
    elif "theta_sub" in d:
        vec = np.array(d["theta_sub"])
    else:
        vec = np.array([d[k] for k in _PARAM_KEYS])
    print(f"Loaded θ from: {path}")
    for i, (k, v) in enumerate(zip(_PARAM_KEYS, vec)):
        print(f"  [{i:2d}] {k:5s} = {v:8.4f}")
    return vec


def main():
    ap = argparse.ArgumentParser(description="2D FEM biofilm simulation")
    ap.add_argument("--theta-json",  default=_DEFAULT_THETA)
    ap.add_argument("--condition",   default="unknown")
    ap.add_argument("--nx",          type=int,   default=20)
    ap.add_argument("--ny",          type=int,   default=20)
    ap.add_argument("--lx",          type=float, default=1.0)
    ap.add_argument("--ly",          type=float, default=1.0)
    ap.add_argument("--n-macro",     type=int,   default=100)
    ap.add_argument("--n-react-sub", type=int,   default=50)
    ap.add_argument("--dt-h",        type=float, default=1e-5)
    ap.add_argument("--save-every",  type=int,   default=5)
    ap.add_argument("--out-dir",     default="_results_2d/run")
    args = ap.parse_args()

    theta = _load_theta(args.theta_json)
    sim   = FEM2DBiofilm(
        theta       = theta,
        Nx          = args.nx,
        Ny          = args.ny,
        Lx          = args.lx,
        Ly          = args.ly,
        n_macro     = args.n_macro,
        n_react_sub = args.n_react_sub,
        dt_h        = args.dt_h,
        save_every  = args.save_every,
        condition   = args.condition,
    )
    snaps_phi, snaps_t = sim.run()
    sim.save(Path(args.out_dir), snaps_phi, snaps_t)


if __name__ == "__main__":
    main()
