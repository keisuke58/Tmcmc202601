#!/usr/bin/env python3
"""
fem_3d_extension.py  –  3D reaction-diffusion FEM for the 5-species Hamilton biofilm model

Domain  : [0, Lx] × [0, Ly] × [0, Lz]
            x = depth (perpendicular to substratum)
            y = lateral 1,  z = lateral 2
Grid    : Nx × Ny × Nz uniform nodes
Node idx: k = ix*(Ny*Nz) + iy*Nz + iz  (row-major, x outer / z inner)

Method  : Lie operator splitting per macro step
  ① Reaction  – Numba prange parallel 0D Hamilton Newton at every node
  ② Diffusion – 3D backward-Euler with SuperLU factorisation per species
                L_3D = kron(kron(Lx,Iy),Iz) + kron(kron(Ix,Ly),Iz) + kron(kron(Ix,Iy),Lz)
                (Neumann BCs on all six faces)

For grids ≤ 20³ (8 000 nodes) SuperLU works well.
For larger grids, switch to CG + ILU (see --solver flag).

Outputs (saved to --out-dir)
  snapshots_phi.npy   (n_snap, 5, Nx, Ny, Nz)
  snapshots_t.npy     (n_snap,)
  mesh_x.npy  mesh_y.npy  mesh_z.npy
  theta_MAP.npy       (20,)

Usage
-----
  python fem_3d_extension.py \\
      --theta-json ../data_5species/_runs/.../theta_MAP.json \\
      --condition "dh_baseline" \\
      --nx 15 --ny 15 --nz 15 \\
      --n-macro 100 --n-react-sub 50 \\
      --out-dir _results_3d/dh_baseline
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# ── module paths ──────────────────────────────────────────────────────────────
_HERE       = Path(__file__).resolve().parent
_TMCMC_ROOT = _HERE.parent
_MODEL_PATH = _TMCMC_ROOT / "tmcmc" / "program2602"
sys.path.insert(0, str(_MODEL_PATH))

try:
    from improved_5species_jit import _newton_step_jit, HAS_NUMBA
    from numba import njit, prange
    _HAVE_MODEL = True
except ImportError:
    _HAVE_MODEL = False
    HAS_NUMBA   = False


# ── Numba reaction kernel (identical to 2D, just more nodes) ─────────────────
if HAS_NUMBA and _HAVE_MODEL:
    @njit(parallel=True, cache=False)
    def _reaction_step_3d(G_flat, A, b_diag, n_sub, dt_h,
                          Kp1, Eta_vec, Eta_phi_vec, c_val, alpha_val,
                          K_hill, n_hill, eps_tol, active_mask):
        N     = G_flat.shape[0]
        G_out = np.empty_like(G_flat)
        for k in prange(N):
            g         = G_flat[k].copy()
            g_new_buf = np.zeros(12)
            K_buf     = np.zeros((12, 12))
            Q_buf     = np.zeros(12)
            for _ in range(n_sub):
                _newton_step_jit(
                    g, dt_h,
                    Kp1, Eta_vec, Eta_phi_vec, c_val, alpha_val,
                    K_hill, n_hill, A, b_diag,
                    eps_tol, 50, active_mask,
                    g_new_buf, K_buf, Q_buf,
                )
                g[:] = g_new_buf[:]
            G_out[k] = g
        return G_out


# ── theta → A, b_diag ────────────────────────────────────────────────────────
def _theta_to_matrices(theta: np.ndarray):
    A      = np.zeros((5, 5), dtype=np.float64)
    b_diag = np.zeros(5,      dtype=np.float64)
    A[0,0]=theta[0]; A[0,1]=theta[1]; A[1,0]=theta[1]; A[1,1]=theta[2]
    b_diag[0]=theta[3]; b_diag[1]=theta[4]
    A[2,2]=theta[5]; A[2,3]=theta[6]; A[3,2]=theta[6]; A[3,3]=theta[7]
    b_diag[2]=theta[8]; b_diag[3]=theta[9]
    A[0,2]=theta[10]; A[2,0]=theta[10]; A[0,3]=theta[11]; A[3,0]=theta[11]
    A[1,2]=theta[12]; A[2,1]=theta[12]; A[1,3]=theta[13]; A[3,1]=theta[13]
    A[4,4]=theta[14]; b_diag[4]=theta[15]
    A[0,4]=theta[16]; A[4,0]=theta[16]; A[1,4]=theta[17]; A[4,1]=theta[17]
    A[2,4]=theta[18]; A[4,2]=theta[18]; A[3,4]=theta[19]; A[4,3]=theta[19]
    return A, b_diag


# ── 3D Laplacian ──────────────────────────────────────────────────────────────
def _build_1d_lap_neu(N: int, h: float) -> sp.csr_matrix:
    h2    = h * h
    diags = np.full(N, -2.0 / h2)
    diags[0] = diags[-1] = -1.0 / h2
    off   = np.ones(N - 1) / h2
    return sp.diags([off, diags, off], [-1, 0, 1], format="csr")


def build_3d_laplacian(Nx, Ny, Nz, dx, dy, dz) -> sp.csr_matrix:
    """L = kron(kron(Lx,Iy),Iz) + kron(kron(Ix,Ly),Iz) + kron(kron(Ix,Iy),Lz).

    Node ordering: k = ix*(Ny*Nz) + iy*Nz + iz  (row-major, x outer / z inner).
    """
    Lx = _build_1d_lap_neu(Nx, dx)
    Ly = _build_1d_lap_neu(Ny, dy)
    Lz = _build_1d_lap_neu(Nz, dz)
    Ix = sp.eye(Nx, format="csr")
    Iy = sp.eye(Ny, format="csr")
    Iz = sp.eye(Nz, format="csr")
    term_x = sp.kron(sp.kron(Lx, Iy, format="csr"), Iz, format="csr")
    term_y = sp.kron(sp.kron(Ix, Ly, format="csr"), Iz, format="csr")
    term_z = sp.kron(sp.kron(Ix, Iy, format="csr"), Lz, format="csr")
    return term_x + term_y + term_z


_D_EFF = np.array([1e-3, 1e-3, 8e-4, 5e-4, 2e-4])


# ── Simulation class ──────────────────────────────────────────────────────────
class FEM3DBiofilm:
    SPECIES = ["S.oralis", "A.naeslundii", "Veillonella", "F.nucleatum", "P.gingivalis"]

    _KP1     = 1e-4
    _C_CONST = 100.0
    _ALPHA   = 100.0
    _K_HILL  = 0.0
    _N_HILL  = 2.0
    _EPS_TOL = 1e-6

    def __init__(
        self,
        theta: np.ndarray,
        Nx: int = 15, Ny: int = 15, Nz: int = 15,
        Lx: float = 1.0, Ly: float = 1.0, Lz: float = 1.0,
        n_macro: int = 100,
        n_react_sub: int = 50,
        dt_h: float = 1e-5,
        D_eff: np.ndarray = None,
        save_every: int = 5,
        condition: str = "",
        solver: str = "superlu",   # "superlu" or "cg"
    ):
        self.theta       = theta.astype(np.float64)
        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        self.Lx, self.Ly, self.Lz = Lx, Ly, Lz
        self.n_macro     = n_macro
        self.n_react_sub = n_react_sub
        self.dt_h        = dt_h
        self.D_eff       = _D_EFF.copy() if D_eff is None else np.asarray(D_eff, dtype=np.float64)
        self.save_every  = save_every
        self.condition   = condition
        self.solver_type = solver

        self.dx = Lx / max(Nx - 1, 1)
        self.dy = Ly / max(Ny - 1, 1)
        self.dz = Lz / max(Nz - 1, 1)
        self.dt_macro = dt_h * n_react_sub
        self.t_total  = self.dt_macro * n_macro

        self.x_mesh = np.linspace(0, Lx, Nx)
        self.y_mesh = np.linspace(0, Ly, Ny)
        self.z_mesh = np.linspace(0, Lz, Nz)

        self.A, self.b_diag = _theta_to_matrices(self.theta)
        self._Eta_vec  = np.ones(5, dtype=np.float64)
        self._Eta_phi  = np.ones(5, dtype=np.float64)
        self._active   = np.ones(5, dtype=np.int64)

        # ── assemble 3D Laplacian + factorize ────────────────────────────
        N_nodes = Nx * Ny * Nz
        print(f"Assembling 3D Laplacian ({Nx}×{Ny}×{Nz} = {N_nodes} nodes) "
              f"and factorising ... ", end="", flush=True)
        L    = build_3d_laplacian(Nx, Ny, Nz, self.dx, self.dy, self.dz)
        I_sp = sp.eye(N_nodes, format="csr")
        self._A_diff  = []
        self._solvers = []
        for D_i in self.D_eff:
            A_sys = (I_sp - self.dt_macro * D_i * L).tocsc()
            self._A_diff.append(A_sys)
            if solver == "superlu":
                self._solvers.append(spla.factorized(A_sys))
            else:
                # CG with ILU preconditioner (better for large grids)
                ilu  = spla.spilu(A_sys, fill_factor=5)
                prec = spla.LinearOperator(A_sys.shape,
                                           matvec=lambda v, M=ilu: M.solve(v))
                self._solvers.append((A_sys, prec))
        print("done.")

        # ── Numba warm-up ─────────────────────────────────────────────────
        self.use_numba = False
        if HAS_NUMBA and _HAVE_MODEL:
            try:
                _g0 = np.zeros((1, 12), dtype=np.float64); _g0[0, 5] = 1.0
                _   = _reaction_step_3d(
                    _g0, self.A, self.b_diag, 1, 1e-5,
                    self._KP1, self._Eta_vec, self._Eta_phi,
                    self._C_CONST, self._ALPHA, self._K_HILL, self._N_HILL,
                    self._EPS_TOL, self._active,
                )
                self.use_numba = True
            except Exception as exc:
                print(f"  [warn] Numba warm-up failed: {exc}")
        print(f"Using Numba parallel: {self.use_numba}")

    # ── initial conditions ────────────────────────────────────────────────────
    def _make_G0(self) -> np.ndarray:
        """G : (Nx, Ny, Nz, 12)"""
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        G   = np.zeros((Nx, Ny, Nz, 12), dtype=np.float64)
        rng = np.random.default_rng(42)

        def noise(s):
            return s * rng.standard_normal((Nx, Ny, Nz))

        # Commensals
        G[:, :, :, 0] = (0.13 + noise(0.01)).clip(0)
        G[:, :, :, 1] = (0.13 + noise(0.01)).clip(0)
        # Veillonella
        G[:, :, :, 2] = (0.08 + noise(0.005)).clip(0)
        # F.nucleatum: exponential decay from substratum (x=0)
        xp = np.exp(-3.0 * self.x_mesh / self.Lx)          # (Nx,)
        G[:, :, :, 3] = (0.05 * xp[:, None, None] + noise(0.005)).clip(0)
        # P.gingivalis: focal sphere at (x=0, y_c, z_c)
        yc, ys = 0.5 * self.Ly, 0.1 * self.Ly
        zc, zs = 0.5 * self.Lz, 0.1 * self.Lz
        xp2 = np.exp(-5.0 * self.x_mesh / self.Lx)
        yp  = np.exp(-0.5 * ((self.y_mesh - yc) / ys) ** 2)
        zp  = np.exp(-0.5 * ((self.z_mesh - zc) / zs) ** 2)
        G[:, :, :, 4] = (
            0.01 * xp2[:, None, None] * yp[None, :, None] * zp[None, None, :]
            + noise(0.002)
        ).clip(1e-6)

        phi_sum         = G[:, :, :, :5].sum(axis=3)
        G[:, :, :, 5]   = (1.0 - phi_sum).clip(0)
        G[:, :, :, 6:11]= G[:, :, :, :5]
        G[:, :, :, 11]  = 1.0
        return G

    # ── reaction ──────────────────────────────────────────────────────────────
    def _react(self, G: np.ndarray) -> np.ndarray:
        if not self.use_numba:
            raise RuntimeError("Numba unavailable – 3D simulation requires Numba.")
        N      = self.Nx * self.Ny * self.Nz
        G_flat = G.reshape(N, 12)
        G_flat = _reaction_step_3d(
            G_flat, self.A, self.b_diag, self.n_react_sub, self.dt_h,
            self._KP1, self._Eta_vec, self._Eta_phi,
            self._C_CONST, self._ALPHA, self._K_HILL, self._N_HILL,
            self._EPS_TOL, self._active,
        )
        return G_flat.reshape(self.Nx, self.Ny, self.Nz, 12)

    # ── diffusion ─────────────────────────────────────────────────────────────
    def _diffuse(self, G: np.ndarray) -> np.ndarray:
        G_new  = G.copy()
        N      = self.Nx * self.Ny * self.Nz
        for i, solver_i in enumerate(self._solvers):
            rhs = G[:, :, :, i].ravel()
            if self.solver_type == "superlu":
                phi_new = solver_i(rhs).clip(0)
            else:
                A_sys, prec = solver_i
                phi_new, info = spla.cg(A_sys, rhs, M=prec, rtol=1e-8)
                phi_new = phi_new.clip(0)
            G_new[:, :, :, i] = phi_new.reshape(self.Nx, self.Ny, self.Nz)
        phi_sum          = G_new[:, :, :, :5].sum(axis=3)
        G_new[:, :, :, 5]= (1.0 - phi_sum).clip(0)
        return G_new

    # ── main loop ─────────────────────────────────────────────────────────────
    def run(self):
        G         = self._make_G0()
        snaps_phi = [G[:, :, :, :5].transpose(3, 0, 1, 2).copy()]  # (5,Nx,Ny,Nz)
        snaps_t   = [0.0]

        print(f"\n{'='*65}")
        print(f"3D FEM Biofilm  |  condition = {self.condition!r}")
        print(f"  Grid   : {self.Nx}×{self.Ny}×{self.Nz}  ({self.Nx*self.Ny*self.Nz} nodes)")
        print(f"  Domain : Lx={self.Lx:.2f}  Ly={self.Ly:.2f}  Lz={self.Lz:.2f}")
        print(f"  dt_h   : {self.dt_h:.1e}  |  n_sub={self.n_react_sub}")
        print(f"  dt_mac : {self.dt_macro:.1e}  |  n_mac={self.n_macro}")
        print(f"  t_tot  : {self.t_total:.4f}  |  solver={self.solver_type}")
        print(f"  D_eff  : {self.D_eff}")
        print(f"{'='*65}\n")

        t0 = time.perf_counter()
        for step in range(1, self.n_macro + 1):
            t = step * self.dt_macro
            G = self._react(G)
            G = self._diffuse(G)
            if step % self.save_every == 0 or step == self.n_macro:
                snaps_phi.append(G[:, :, :, :5].transpose(3, 0, 1, 2).copy())
                snaps_t.append(t)
                pm  = G[:, :, :, :5].mean(axis=(0, 1, 2))
                bar = "[" + ", ".join(f"{v:.3f}" for v in pm) + "]"
                print(f"  [{100*step/self.n_macro:5.1f}%] t={t:.4f}  φ̄={bar}  "
                      f"elapsed={time.perf_counter()-t0:.1f}s")

        elapsed = time.perf_counter() - t0
        print(f"\nDone in {elapsed:.1f}s  |  {len(snaps_phi)} snapshots")
        return np.array(snaps_phi), np.array(snaps_t)

    def save(self, out_dir: Path, snaps_phi, snaps_t):
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "snapshots_phi.npy", snaps_phi)   # (n_snap,5,Nx,Ny,Nz)
        np.save(out_dir / "snapshots_t.npy",   snaps_t)
        np.save(out_dir / "mesh_x.npy",        self.x_mesh)
        np.save(out_dir / "mesh_y.npy",        self.y_mesh)
        np.save(out_dir / "mesh_z.npy",        self.z_mesh)
        np.save(out_dir / "theta_MAP.npy",     self.theta)
        print(f"Saved to: {out_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────
_RUNS_ROOT     = _TMCMC_ROOT / "data_5species" / "_runs"
_DEFAULT_THETA = str(
    _RUNS_ROOT / "sweep_pg_20260217_081459" / "dh_baseline" / "theta_MAP.json"
)
_PARAM_KEYS = [
    "a11","a12","a22","b1","b2","a33","a34","a44","b3","b4",
    "a13","a14","a23","a24","a55","b5","a15","a25","a35","a45",
]


def _load_theta(path: str) -> np.ndarray:
    with open(path) as f:
        d = json.load(f)
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
    ap = argparse.ArgumentParser(description="3D FEM biofilm simulation")
    ap.add_argument("--theta-json",  default=_DEFAULT_THETA)
    ap.add_argument("--condition",   default="unknown")
    ap.add_argument("--nx",          type=int,   default=15)
    ap.add_argument("--ny",          type=int,   default=15)
    ap.add_argument("--nz",          type=int,   default=15)
    ap.add_argument("--lx",          type=float, default=1.0)
    ap.add_argument("--ly",          type=float, default=1.0)
    ap.add_argument("--lz",          type=float, default=1.0)
    ap.add_argument("--n-macro",     type=int,   default=100)
    ap.add_argument("--n-react-sub", type=int,   default=50)
    ap.add_argument("--dt-h",        type=float, default=1e-5)
    ap.add_argument("--save-every",  type=int,   default=5)
    ap.add_argument("--out-dir",     default="_results_3d/run")
    ap.add_argument("--solver",      default="superlu", choices=["superlu", "cg"],
                    help="Linear solver for diffusion step (superlu or cg)")
    args = ap.parse_args()

    theta = _load_theta(args.theta_json)
    sim   = FEM3DBiofilm(
        theta       = theta,
        Nx=args.nx, Ny=args.ny, Nz=args.nz,
        Lx=args.lx, Ly=args.ly, Lz=args.lz,
        n_macro     = args.n_macro,
        n_react_sub = args.n_react_sub,
        dt_h        = args.dt_h,
        save_every  = args.save_every,
        condition   = args.condition,
        solver      = args.solver,
    )
    snaps_phi, snaps_t = sim.run()
    sim.save(Path(args.out_dir), snaps_phi, snaps_t)


if __name__ == "__main__":
    main()
