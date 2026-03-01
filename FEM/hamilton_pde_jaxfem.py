#!/usr/bin/env python3
"""
hamilton_pde_jaxfem.py
======================
5-species Hamilton biofilm PDE — JAX-FEM skeleton.

This module defines a jax_fem Problem subclass for the full variational
formulation of the Hamilton 5-species biofilm model as a coupled
reaction-diffusion PDE system.

Governing equations (strong form, per species i):
    dphi_i/dt = D_i lap(phi_i) + R_i(phi, psi, c)

where R_i is the Hamilton reaction term from the Cahn-Hilliard-type
free energy functional, and c(x,t) is the nutrient field satisfying:
    dc/dt = D_c lap(c) - sum_i g_i phi_i c / (k_M + c)

Weak form (for FEM assembly):
    int_Omega [ dphi_i/dt * v + D_i grad(phi_i) . grad(v) - R_i * v ] dx = 0

This skeleton implements:
  - HamiltonBiofilm5S(Problem): vec=5 species volume fractions
  - get_tensor_map(): species-specific diffusion
  - get_mass_map(): Hamilton reaction source terms
  - Backward-Euler time stepping framework (TODO)
  - Nutrient PDE coupling (TODO)

Long-term goal: replace Numba/scipy splitting solver with fully implicit
JAX-FEM solve using jax_fem's Newton solver and AD framework.

Run with klempt_fem conda env:
    ~/.pyenv/versions/miniconda3-latest/envs/klempt_fem/bin/python hamilton_pde_jaxfem.py

Usage
-----
    python hamilton_pde_jaxfem.py [--setup-only] [--nx 20] [--ny 20]
"""

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

# ── jax_fem imports ──────────────────────────────────────────────────────────
try:
    from jax_fem.problem import Problem
    from jax_fem.solver import solver as jaxfem_solver
    from jax_fem.generate_mesh import rectangle_mesh, Mesh, get_meshio_cell_type
    from jax_fem.utils import save_sol

    _HAS_JAXFEM = True
except ImportError:
    _HAS_JAXFEM = False
    print("[warn] jax_fem not available — skeleton mode only")

# ── Hamilton model parameters ────────────────────────────────────────────────

_PARAM_KEYS = [
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

SPECIES_NAMES = ["S.oralis", "A.naeslundii", "Veillonella", "F.nucleatum", "P.gingivalis"]

# Default diffusion coefficients (dimensional, matching fem_2d_extension.py)
D_EFF_DEFAULT = jnp.array([1e-3, 1e-3, 8e-4, 5e-4, 2e-4])

# Demo theta from TMCMC MAP (mild-weight experiment)
THETA_DEMO = jnp.array(
    [
        1.34,
        -0.18,
        1.79,
        1.17,
        2.58,
        3.51,
        2.73,
        0.71,
        2.1,
        0.37,
        2.05,
        -0.15,
        3.56,
        0.16,
        0.12,
        0.32,
        1.49,
        2.1,
        2.41,
        2.5,
    ]
)


# ── theta → matrices ────────────────────────────────────────────────────────


def theta_to_matrices(theta):
    """Convert 20-vector theta to interaction matrix A (5x5) and b_diag (5,)."""
    A = jnp.zeros((5, 5))
    b = jnp.zeros(5)
    A = A.at[0, 0].set(theta[0])
    A = A.at[0, 1].set(theta[1])
    A = A.at[1, 0].set(theta[1])
    A = A.at[1, 1].set(theta[2])
    b = b.at[0].set(theta[3])
    b = b.at[1].set(theta[4])
    A = A.at[2, 2].set(theta[5])
    A = A.at[2, 3].set(theta[6])
    A = A.at[3, 2].set(theta[6])
    A = A.at[3, 3].set(theta[7])
    b = b.at[2].set(theta[8])
    b = b.at[3].set(theta[9])
    A = A.at[0, 2].set(theta[10])
    A = A.at[2, 0].set(theta[10])
    A = A.at[0, 3].set(theta[11])
    A = A.at[3, 0].set(theta[11])
    A = A.at[1, 2].set(theta[12])
    A = A.at[2, 1].set(theta[12])
    A = A.at[1, 3].set(theta[13])
    A = A.at[3, 1].set(theta[13])
    A = A.at[4, 4].set(theta[14])
    b = b.at[4].set(theta[15])
    A = A.at[0, 4].set(theta[16])
    A = A.at[4, 0].set(theta[16])
    A = A.at[1, 4].set(theta[17])
    A = A.at[4, 1].set(theta[17])
    A = A.at[2, 4].set(theta[18])
    A = A.at[4, 2].set(theta[18])
    A = A.at[3, 4].set(theta[19])
    A = A.at[4, 3].set(theta[19])
    return A, b


# ── Hamilton reaction source term ────────────────────────────────────────────


def hamilton_reaction(phi, A, b_diag, K_hill=0.05, n_hill=4.0, c_nutrient=1.0, k_monod=1.0):
    """
    Compute Hamilton reaction rates R_i for 5 species.

    This is a simplified steady-state form of the Hamilton dynamics:
      R_i = b_i * phi_i * (1 - sum_j(A_ij * phi_j)) * monod(c)
            + Hill_gate(phi_Fn) for species i=Pg

    Parameters
    ----------
    phi : (5,)  species volume fractions
    A   : (5,5) interaction matrix
    b_diag : (5,) growth rate vector
    K_hill : float  Hill gate half-saturation for Fn->Pg
    n_hill : float  Hill gate exponent
    c_nutrient : float  local nutrient concentration
    k_monod : float  Monod half-saturation constant

    Returns
    -------
    R : (5,)  reaction source terms
    """
    eps = 1e-12

    # Monod factor: nutrient limitation
    monod = c_nutrient / (k_monod + c_nutrient)

    # Lotka-Volterra-type interaction
    Ia = A @ phi
    growth = b_diag * phi * (1.0 - Ia) * monod

    # Hill gate for P.gingivalis (species 4), gated by F.nucleatum (species 3)
    fn_activity = jnp.maximum(phi[3], 0.0)
    num = fn_activity**n_hill
    den = K_hill**n_hill + num
    hill_factor = jnp.where(den > eps, num / den, 0.0)

    # Apply Hill gate only to Pg growth
    growth = growth.at[4].set(growth[4] * hill_factor)

    # Logistic saturation: prevent phi > 1
    phi_total = jnp.sum(phi)
    saturation = jnp.clip(1.0 - phi_total, 0.0, 1.0)
    R = growth * saturation

    return R


# ── JAX-FEM Problem class ───────────────────────────────────────────────────

if _HAS_JAXFEM:

    class HamiltonBiofilm5S(Problem):
        """
        5-species Hamilton biofilm PDE formulated as a jax_fem Problem.

        State: u = [phi_1, ..., phi_5]  (vec=5)
        Diffusion: tensor_map returns D_i * grad(phi_i) per species
        Reaction: mass_map returns Hamilton reaction source R_i(phi, x)

        The resulting weak form (steady-state) is:
            int [ D_i grad(phi_i) . grad(v_i) + R_i * v_i ] dx = 0

        For time-dependent problems, use backward-Euler:
            int [ (phi^{n+1} - phi^n)/dt * v + D_i grad(phi^{n+1}) . grad(v)
                  - R(phi^{n+1}) * v ] dx = 0

        TODO: Implement time stepping loop with phi^n as internal_vars.
        TODO: Add nutrient PDE as second coupled Problem or additional DOFs.
        TODO: Add volume constraint (phi_0 = 1 - sum(phi_i)) as penalty or Lagrange mult.
        """

        def __init__(
            self, theta, D_eff=None, K_hill=0.05, n_hill=4.0, k_monod=1.0, *args, **kwargs
        ):
            super().__init__(*args, **kwargs)
            self.A_mat, self.b_diag = theta_to_matrices(theta)
            self.D_eff = D_EFF_DEFAULT if D_eff is None else jnp.asarray(D_eff)
            self.K_hill = K_hill
            self.n_hill = n_hill
            self.k_monod = k_monod

            # TODO: For time-dependent problems, store phi_old as internal_vars
            # self.phi_old = None
            # self.dt = None

        def get_tensor_map(self):
            """
            Diffusion flux: sigma_i = D_i * grad(phi_i)

            For vec=5, dim=2:
              u_grads : (5, 2) — gradient of each species
              returns : (5, 2) — diffusion flux for each species
            """
            D = self.D_eff

            def tensor_map(u_grads):
                # u_grads: (vec, dim) = (5, 2)
                # D[:, None] broadcasts to (5, 1) * (5, 2) = (5, 2)
                return D[:, None] * u_grads

            return tensor_map

        def get_mass_map(self):
            """
            Reaction source: R_i(phi, x)

            For vec=5, dim=2:
              u : (5,)  — species volume fractions at quadrature point
              x : (2,)  — physical coordinates of quadrature point
              returns : (5,)  — reaction contribution to weak form

            Sign convention: positive R_i means growth (source).
            In the weak form, this appears as -R_i * v_i,
            so we return -R to match jax_fem's convention
            (mass_map is added to the left-hand side residual).
            """
            A_mat = self.A_mat
            b_diag = self.b_diag
            K_hill = self.K_hill
            n_hill = self.n_hill
            k_monod = self.k_monod

            def mass_map(u, x):
                # u: (vec,) = (5,) — current phi values
                # x: (dim,) = (2,) — spatial coordinates
                phi = jnp.clip(u, 1e-10, 1.0 - 1e-10)

                # TODO: c_nutrient should come from coupled nutrient PDE
                # For now, use spatially uniform nutrient c=1
                c_nutrient = 1.0

                R = hamilton_reaction(
                    phi,
                    A_mat,
                    b_diag,
                    K_hill=K_hill,
                    n_hill=n_hill,
                    c_nutrient=c_nutrient,
                    k_monod=k_monod,
                )

                # TODO: For time-dependent, add mass term:
                # dt = self.dt
                # phi_old = <from internal_vars>
                # mass_term = (phi - phi_old) / dt
                # return mass_term - R

                # Steady-state: return -R (source on RHS)
                return -R

            return mass_map


# ── Bilinear interpolation helper (for backward-Euler phi_old lookup) ────────


def _make_bilinear_interp(Lx, Ly, Nx, Ny):
    """
    Return a JAX-compatible function that interpolates a 2D nodal grid at (x, y).

    The grid has (Nx+1) x (Ny+1) nodes on [0, Lx] x [0, Ly].
    Values: (Nx+1, Ny+1, vec) array stored in a mutable list [grid].

    Returns (interp_fn, grid_holder) where:
      grid_holder[0] = (Nx+1, Ny+1, vec) jnp array (mutable reference)
      interp_fn(x) = bilinearly interpolated value at x = (x_val, y_val)
    """
    dx = Lx / Nx
    dy = Ly / Ny

    # Mutable container: grid_holder[0] is updated each time step
    grid_holder = [None]

    def interp_fn(x_phys):
        """Bilinear interpolation at physical coordinate x_phys = (x, y)."""
        grid = grid_holder[0]  # (Nx+1, Ny+1, vec)
        xv = jnp.clip(x_phys[0], 0.0, Lx)
        yv = jnp.clip(x_phys[1], 0.0, Ly)

        # Continuous cell indices
        ix_f = xv / dx
        iy_f = yv / dy

        # Integer cell indices (clamped)
        ix = jnp.clip(jnp.floor(ix_f).astype(jnp.int32), 0, Nx - 1)
        iy = jnp.clip(jnp.floor(iy_f).astype(jnp.int32), 0, Ny - 1)

        # Fractional part
        fx = ix_f - ix.astype(jnp.float64)
        fy = iy_f - iy.astype(jnp.float64)

        # Bilinear interpolation
        v00 = grid[ix, iy]
        v10 = grid[ix + 1, iy]
        v01 = grid[ix, iy + 1]
        v11 = grid[ix + 1, iy + 1]

        val = (
            (1.0 - fx) * (1.0 - fy) * v00
            + fx * (1.0 - fy) * v10
            + (1.0 - fx) * fy * v01
            + fx * fy * v11
        )
        return val

    return interp_fn, grid_holder


# ── Time-dependent Problem class ────────────────────────────────────────────

if _HAS_JAXFEM:

    class HamiltonBiofilm5STimeDep(Problem):
        """
        Time-dependent 5-species Hamilton biofilm PDE (backward-Euler).

        Weak form at each time step:
            int [ (phi^{n+1} - phi^n)/dt * v
                  + D_i grad(phi^{n+1}) . grad(v)
                  - R(phi^{n+1}) * v ] dx = 0

        phi^n is stored on a structured grid and interpolated at quadrature
        points via bilinear interpolation (regular mesh only).
        """

        def __init__(
            self,
            theta,
            dt,
            Nx,
            Ny,
            Lx=1.0,
            Ly=1.0,
            D_eff=None,
            K_hill=0.05,
            n_hill=4.0,
            k_monod=1.0,
            *args,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            self.A_mat, self.b_diag = theta_to_matrices(theta)
            self.D_eff = D_EFF_DEFAULT if D_eff is None else jnp.asarray(D_eff)
            self.K_hill = K_hill
            self.n_hill = n_hill
            self.k_monod = k_monod
            self.dt = dt
            self.Nx = Nx
            self.Ny = Ny
            self.Lx = Lx
            self.Ly = Ly

            # Bilinear interpolation for phi_old
            self._interp_fn, self._grid_holder = _make_bilinear_interp(Lx, Ly, Nx, Ny)

        def set_phi_old(self, phi_old_nodal):
            """
            Update phi^n for the next backward-Euler step.

            Parameters
            ----------
            phi_old_nodal : (num_total_nodes, 5) nodal values of phi at t^n

            The nodal values are reshaped to (Nx+1, Ny+1, 5) grid for
            bilinear interpolation inside mass_map.
            """
            grid = jnp.asarray(phi_old_nodal).reshape(self.Nx + 1, self.Ny + 1, 5)
            self._grid_holder[0] = grid

        def get_tensor_map(self):
            D = self.D_eff

            def tensor_map(u_grads):
                return D[:, None] * u_grads

            return tensor_map

        def get_mass_map(self):
            A_mat = self.A_mat
            b_diag = self.b_diag
            K_hill = self.K_hill
            n_hill = self.n_hill
            k_monod = self.k_monod
            dt = self.dt
            interp_fn = self._interp_fn

            def mass_map(u, x):
                phi = jnp.clip(u, 1e-10, 1.0 - 1e-10)

                # Interpolate phi_old at this quadrature point
                phi_old = interp_fn(x)

                # Time derivative (backward-Euler)
                time_deriv = (phi - phi_old) / dt

                # Reaction
                c_nutrient = 1.0  # TODO: couple with nutrient PDE
                R = hamilton_reaction(
                    phi,
                    A_mat,
                    b_diag,
                    K_hill=K_hill,
                    n_hill=n_hill,
                    c_nutrient=c_nutrient,
                    k_monod=k_monod,
                )

                # Residual: time_deriv - R = 0  (on LHS)
                return time_deriv - R

            return mass_map


# ── Mesh and Problem setup ───────────────────────────────────────────────────


def setup_problem(theta, Nx=20, Ny=20, Lx=1.0, Ly=1.0, D_eff=None, K_hill=0.05, n_hill=4.0):
    """
    Build the steady-state jax_fem Problem for 5-species Hamilton PDE.

    Parameters
    ----------
    theta : (20,) TMCMC parameter vector
    Nx, Ny : int  mesh resolution
    Lx, Ly : float  domain size
    D_eff : (5,) diffusion coefficients (optional)
    K_hill, n_hill : Hill gate parameters

    Returns
    -------
    problem : HamiltonBiofilm5S
    """
    if not _HAS_JAXFEM:
        raise RuntimeError("jax_fem is required. Install in klempt_fem env.")

    ele_type = "QUAD4"
    cell_type = get_meshio_cell_type(ele_type)

    meshio_mesh = rectangle_mesh(Nx, Ny, Lx, Ly)
    points = meshio_mesh.points[:, :2]
    cells = meshio_mesh.cells_dict[cell_type]
    mesh = Mesh(points, cells, ele_type)

    problem = HamiltonBiofilm5S(
        theta=jnp.asarray(theta),
        D_eff=D_eff,
        K_hill=K_hill,
        n_hill=n_hill,
        mesh=mesh,
        vec=5,
        dim=2,
        ele_type=ele_type,
        dirichlet_bc_info=None,  # Pure Neumann for species
    )

    return problem


def setup_timedep_problem(
    theta, dt, Nx=20, Ny=20, Lx=1.0, Ly=1.0, D_eff=None, K_hill=0.05, n_hill=4.0, k_monod=1.0
):
    """
    Build the time-dependent (backward-Euler) jax_fem Problem.

    Parameters
    ----------
    theta : (20,) TMCMC parameter vector
    dt : float  time step size
    Nx, Ny : int  mesh resolution
    Lx, Ly : float  domain size

    Returns
    -------
    problem : HamiltonBiofilm5STimeDep
    """
    if not _HAS_JAXFEM:
        raise RuntimeError("jax_fem is required. Install in klempt_fem env.")

    ele_type = "QUAD4"
    cell_type = get_meshio_cell_type(ele_type)

    meshio_mesh = rectangle_mesh(Nx, Ny, Lx, Ly)
    points = meshio_mesh.points[:, :2]
    cells = meshio_mesh.cells_dict[cell_type]
    mesh = Mesh(points, cells, ele_type)

    problem = HamiltonBiofilm5STimeDep(
        theta=jnp.asarray(theta),
        dt=dt,
        Nx=Nx,
        Ny=Ny,
        Lx=Lx,
        Ly=Ly,
        D_eff=D_eff,
        K_hill=K_hill,
        n_hill=n_hill,
        k_monod=k_monod,
        mesh=mesh,
        vec=5,
        dim=2,
        ele_type=ele_type,
        dirichlet_bc_info=None,
    )

    return problem


# ── Initial condition ────────────────────────────────────────────────────────


def make_initial_condition(problem, mode="uniform"):
    """
    Create initial species distribution phi_0(x).

    Parameters
    ----------
    problem : HamiltonBiofilm5S
    mode : str  "uniform" or "gradient"

    Returns
    -------
    u0 : (num_total_nodes, 5)
    """
    n_nodes = problem.fes[0].num_total_nodes
    coords = problem.fes[0].points  # (n_nodes, 2)

    if mode == "uniform":
        # Uniform initial distribution
        phi0 = jnp.array([0.15, 0.15, 0.10, 0.08, 0.02])
        u0 = jnp.tile(phi0, (n_nodes, 1))
    elif mode == "gradient":
        # Depth-dependent: Pg enriched near x=0 (substrate)
        x = coords[:, 0]
        Lx = x.max()

        u0 = jnp.zeros((n_nodes, 5))
        u0 = u0.at[:, 0].set(0.15)  # S.oralis: uniform
        u0 = u0.at[:, 1].set(0.15)  # A.naeslundii: uniform
        u0 = u0.at[:, 2].set(0.10)  # Veillonella: uniform
        u0 = u0.at[:, 3].set(0.05 + 0.05 * jnp.exp(-3.0 * x / Lx))  # Fn: substrate-enriched
        u0 = u0.at[:, 4].set(0.01 * jnp.exp(-5.0 * x / Lx))  # Pg: focal near substrate
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Ensure sum < 1
    phi_sum = jnp.sum(u0, axis=1, keepdims=True)
    scale = jnp.where(phi_sum > 0.95, 0.95 / phi_sum, 1.0)
    u0 = u0 * scale

    return u0


# ── Time stepping (TODO: full implementation) ────────────────────────────────


def backward_euler_step(cfg, u_old, solver_options=None):
    """
    One backward-Euler step for the Hamilton PDE.

    Rebuilds the Problem each step to ensure JAX sees the updated phi_old
    (JAX JIT caches closed-over values as constants, so a mutable reference
    to phi_old would be ignored after the first compilation).

    Parameters
    ----------
    cfg : dict  problem configuration (theta, dt, Nx, Ny, Lx, Ly, etc.)
    u_old : (num_total_nodes, 5) solution at time t^n

    Returns
    -------
    u_new : (num_total_nodes, 5) solution at time t^{n+1}
    """
    # Rebuild problem with current phi_old baked into mass_map
    problem = setup_timedep_problem(**cfg)
    problem.set_phi_old(u_old)

    opts = dict(solver_options or {})
    opts["initial_guess"] = [u_old]
    sol_list = jaxfem_solver(problem, opts)
    u_new = sol_list[0]

    # Clip to physical range and enforce sum constraint
    u_new = jnp.clip(u_new, 0.0, 1.0)
    phi_sum = jnp.sum(u_new, axis=1, keepdims=True)
    scale = jnp.where(phi_sum > 0.999, 0.999 / phi_sum, 1.0)
    u_new = u_new * scale

    return u_new


def run_time_integration(cfg, u0, t_final, save_every=10, solver_options=None):
    """
    Run backward-Euler time integration using JAX-FEM Newton solver.

    Rebuilds the Problem each step to ensure correct phi_old propagation.

    Parameters
    ----------
    cfg : dict  problem configuration (passed to setup_timedep_problem)
    u0 : (num_total_nodes, 5) initial condition
    t_final : float  end time
    save_every : int  save snapshot every N steps
    solver_options : dict  options for jax_fem solver

    Returns
    -------
    dict with:
        phi_snaps : list of (num_total_nodes, 5) arrays
        t_snaps : (n_snap,) array
    """
    dt = cfg["dt"]
    n_steps = int(t_final / dt)

    u = jnp.asarray(u0)
    snaps = [np.asarray(u)]
    t_snaps = [0.0]

    n_nodes = u.shape[0]
    print(f"\n{'='*60}")
    print("Backward-Euler time integration (JAX-FEM)")
    print(f"  dt={dt:.1e}  n_steps={n_steps}  t_final={t_final}")
    print(f"  nodes={n_nodes}  vec=5")
    print(f"{'='*60}\n")

    for step in range(1, n_steps + 1):
        t = step * dt

        try:
            u = backward_euler_step(cfg, u, solver_options)
        except Exception as exc:
            print(f"  [FAIL] Step {step}, t={t:.5f}: {exc}")
            break

        if step % save_every == 0 or step == n_steps:
            phi_mean = jnp.mean(u, axis=0)
            bar = ", ".join(f"{float(v):.4f}" for v in phi_mean)
            print(f"  [{100*step/n_steps:5.1f}%] t={t:.5f}  phi_mean=[{bar}]")
            snaps.append(np.asarray(u))
            t_snaps.append(t)

    return {"phi_snaps": snaps, "t_snaps": np.array(t_snaps)}


# ── CLI & main ───────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(
        description="5-species Hamilton PDE in JAX-FEM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  --setup-only   : Verify problem structure (no solve)
  --time-dep     : Run backward-Euler time integration
  (default)      : Attempt steady-state solve

Next steps:
  1. Add nutrient PDE as coupled second Problem or additional DOFs
  2. Add volume constraint (phi_0 = 1 - sum phi_i)
  3. Validate against splitting solver (run_hamilton_2d_nutrient.py)
  4. Enable adjoint sensitivity via ad_wrapper for inverse problems
""",
    )
    ap.add_argument(
        "--setup-only", action="store_true", help="Just set up Problem and print info (no solve)"
    )
    ap.add_argument("--time-dep", action="store_true", help="Run backward-Euler time integration")
    ap.add_argument("--nx", type=int, default=10)
    ap.add_argument("--ny", type=int, default=10)
    ap.add_argument("--lx", type=float, default=1.0)
    ap.add_argument("--ly", type=float, default=1.0)
    ap.add_argument("--dt", type=float, default=1e-3, help="Time step for backward-Euler")
    ap.add_argument("--t-final", type=float, default=0.01, help="End time for time integration")
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--K-hill", type=float, default=0.05)
    ap.add_argument("--n-hill", type=float, default=4.0)
    args = ap.parse_args()

    if not _HAS_JAXFEM:
        print("jax_fem not available. Run with klempt_fem conda env:")
        print(
            "  ~/.pyenv/versions/miniconda3-latest/envs/klempt_fem/bin/python "
            "hamilton_pde_jaxfem.py"
        )
        sys.exit(1)

    print("=" * 60)
    print("Hamilton 5-Species Biofilm PDE — JAX-FEM")
    print("=" * 60)

    theta = THETA_DEMO

    # ── Setup-only or steady-state ─────────────────────────────────
    if not args.time_dep:
        problem = setup_problem(
            theta,
            Nx=args.nx,
            Ny=args.ny,
            Lx=args.lx,
            Ly=args.ly,
            K_hill=args.K_hill,
            n_hill=args.n_hill,
        )

        fe = problem.fes[0]
        print("\nMesh:")
        print(f"  Nodes: {fe.num_total_nodes}")
        print(f"  Cells: {problem.num_cells}")
        print(f"  Element: {problem.ele_type}")
        print(f"  vec={problem.vec}, dim={problem.dim}")

        A, b = theta_to_matrices(theta)
        print("\nModel parameters:")
        print("  A (interaction matrix):")
        for i, name in enumerate(SPECIES_NAMES):
            row = "  ".join(f"{float(A[i, j]):6.3f}" for j in range(5))
            print(f"    {name:15s}: [{row}]")
        print(f"  b (growth rates): {[f'{float(v):.3f}' for v in b]}")
        print(f"  D_eff: {[f'{float(v):.1e}' for v in D_EFF_DEFAULT]}")

        u0 = make_initial_condition(problem, mode="gradient")
        print(f"\nInitial condition (gradient mode): shape={u0.shape}")
        phi_mean = jnp.mean(u0, axis=0)
        for i, name in enumerate(SPECIES_NAMES):
            print(
                f"    {name:15s}: mean={float(phi_mean[i]):.4f}"
                f"  [{float(u0[:, i].min()):.4f}, {float(u0[:, i].max()):.4f}]"
            )

        R = hamilton_reaction(phi_mean, A, b, K_hill=args.K_hill, n_hill=args.n_hill)
        print("\nReaction rates at mean phi:")
        for i, name in enumerate(SPECIES_NAMES):
            print(f"    R_{name:15s} = {float(R[i]):+.6f}")

        if args.setup_only:
            print("\n[setup-only] Problem verified. Use --time-dep to run.")
            return

        print("\n[Attempting steady-state solve...]")
        try:
            sol_list = jaxfem_solver(problem, solver_options={})
            sol = sol_list[0]
            print(f"  Solution shape: {sol.shape}")
            phi_final = jnp.mean(sol, axis=0)
            for i, name in enumerate(SPECIES_NAMES):
                print(f"    phi_{name}: mean={float(phi_final[i]):.4f}")
        except Exception as exc:
            print(f"  Steady-state solve failed: {exc}")
            print("  Use --time-dep for backward-Euler time integration.")
        return

    # ── Time-dependent backward-Euler ──────────────────────────────
    print("\n[Time-dependent backward-Euler]")
    print(f"  dt={args.dt:.1e}  t_final={args.t_final}  grid={args.nx}x{args.ny}")

    # Build cfg dict for run_time_integration (rebuilds Problem each step)
    cfg = dict(
        theta=theta,
        dt=args.dt,
        Nx=args.nx,
        Ny=args.ny,
        Lx=args.lx,
        Ly=args.ly,
        K_hill=args.K_hill,
        n_hill=args.n_hill,
    )

    # Build problem once just for IC generation (mesh/nodes info)
    problem_for_ic = setup_timedep_problem(**cfg)
    u0 = make_initial_condition(problem_for_ic, mode="gradient")
    print(f"  u0 shape: {u0.shape}")

    result = run_time_integration(
        cfg,
        u0,
        t_final=args.t_final,
        save_every=args.save_every,
    )

    t_snaps = result["t_snaps"]
    phi_snaps = result["phi_snaps"]
    print(f"\nCompleted: {len(t_snaps)} snapshots, t=[{t_snaps[0]:.5f}, {t_snaps[-1]:.5f}]")

    # Save results
    out_dir = Path("_results_jaxfem_timedep")
    out_dir.mkdir(exist_ok=True)
    np.save(out_dir / "t_snaps.npy", t_snaps)
    np.save(out_dir / "phi_snaps.npy", np.array(phi_snaps))
    print(f"  Saved to: {out_dir}")


if __name__ == "__main__":
    main()
