#!/usr/bin/env python3
"""
solve_stress_2d.py — 2D plane-strain linear elasticity with spatially-varying
eigenstrain (growth-driven stress in biofilm).

Solves:  -div(σ) = 0  on Ω = [0,Lx]×[0,Ly]
         σ = C(x) : (ε(u) - ε_growth(x))
         BC:  u = 0 on tooth surface (y=0)
              traction-free on other boundaries

Uses scipy sparse QUAD4 FEM on a regular grid.

Pipeline:
    theta → 2D Hamilton ODE → phi(x,y), c(x,y)
    → DI(x,y) → E(x,y), ε_growth(x,y)
    → FEM solve → σ_vm(x,y), u(x,y)

Usage:
    python solve_stress_2d.py                          # demo with MAP theta
    python solve_stress_2d.py --condition dh_baseline   # from TMCMC result
    python solve_stress_2d.py --from-npy phi.npy c.npy  # from pre-computed fields
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from material_models import (
    compute_E_phi_pg,
    compute_E_virulence,
    compute_E_eps_synergy,
    compute_di,
)


# ============================================================================
# 2D QUAD4 FEM Core
# ============================================================================


def _gauss_2x2():
    """2×2 Gauss quadrature points and weights on [-1,1]²."""
    g = 1.0 / np.sqrt(3.0)
    pts = np.array([[-g, -g], [g, -g], [g, g], [-g, g]])
    wts = np.array([1.0, 1.0, 1.0, 1.0])
    return pts, wts


def _shape_grad(xi, eta):
    """Shape function gradients for QUAD4 in natural coords.

    Returns dN/dxi (4,) and dN/deta (4,).
    Node order: 0=(0,0), 1=(1,0), 2=(1,1), 3=(0,1) in physical space.
    """
    dN_dxi = np.array(
        [
            -(1 - eta) / 4,
            (1 - eta) / 4,
            (1 + eta) / 4,
            -(1 + eta) / 4,
        ]
    )
    dN_deta = np.array(
        [
            -(1 - xi) / 4,
            -(1 + xi) / 4,
            (1 + xi) / 4,
            (1 - xi) / 4,
        ]
    )
    return dN_dxi, dN_deta


def _B_matrix(dN_dx, dN_dy):
    """Strain-displacement matrix B (3×8) for 2D plane strain.

    ε = [ε_xx, ε_yy, γ_xy]^T = B · u_e
    u_e = [u0x, u0y, u1x, u1y, u2x, u2y, u3x, u3y]
    """
    B = np.zeros((3, 8))
    for i in range(4):
        B[0, 2 * i] = dN_dx[i]  # ε_xx = ∂u_x/∂x
        B[1, 2 * i + 1] = dN_dy[i]  # ε_yy = ∂u_y/∂y
        B[2, 2 * i] = dN_dy[i]  # γ_xy = ∂u_x/∂y + ∂u_y/∂x
        B[2, 2 * i + 1] = dN_dx[i]
    return B


def _C_plane_strain(E, nu):
    """Constitutive matrix for 2D plane strain (3×3)."""
    f = E / ((1 + nu) * (1 - 2 * nu))
    C = f * np.array(
        [
            [1 - nu, nu, 0],
            [nu, 1 - nu, 0],
            [0, 0, (1 - 2 * nu) / 2],
        ]
    )
    return C


def solve_2d_fem(E_field, nu, eps_growth_field, Nx, Ny, Lx=1.0, Ly=1.0, bc_type="bottom_fixed"):
    """Solve 2D plane-strain elasticity on regular QUAD4 grid.

    Parameters
    ----------
    E_field : (Nx, Ny) — Young's modulus at each node [Pa]
    nu : float — Poisson's ratio
    eps_growth_field : (Nx, Ny) — isotropic eigenstrain at each node
    Nx, Ny : int — number of nodes in x, y
    Lx, Ly : float — domain size
    bc_type : str — "bottom_fixed" or "left_fixed"

    Returns
    -------
    dict with keys:
        u : (Nx*Ny, 2) — displacement [m]
        sigma_xx, sigma_yy, sigma_xy : (n_elem,) — element-average stress [Pa]
        sigma_vm : (n_elem,) — von Mises stress [Pa]
        elem_centers : (n_elem, 2) — element center coordinates
    """
    n_nodes = Nx * Ny
    n_dof = 2 * n_nodes
    n_elem_x = Nx - 1
    n_elem_y = Ny - 1
    n_elem = n_elem_x * n_elem_y

    dx = Lx / n_elem_x
    dy = Ly / n_elem_y

    # Node coordinates
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    coords = np.column_stack([X.ravel(), Y.ravel()])  # (n_nodes, 2)

    # Element connectivity: node indices for each element
    # elem (i,j) has nodes: (i,j), (i+1,j), (i+1,j+1), (i,j+1)
    def node_idx(ix, iy):
        return ix * Ny + iy

    elems = np.zeros((n_elem, 4), dtype=int)
    for i in range(n_elem_x):
        for j in range(n_elem_y):
            e = i * n_elem_y + j
            elems[e] = [
                node_idx(i, j),
                node_idx(i + 1, j),
                node_idx(i + 1, j + 1),
                node_idx(i, j + 1),
            ]

    # Gauss quadrature
    gpts, gwts = _gauss_2x2()

    # Jacobian for regular rectangular elements
    J = np.array([[dx / 2, 0], [0, dy / 2]])
    detJ = dx * dy / 4
    Jinv = np.array([[2 / dx, 0], [0, 2 / dy]])

    # Assembly
    rows = []
    cols = []
    vals = []
    F = np.zeros(n_dof)

    for e in range(n_elem):
        enodes = elems[e]
        edof = np.zeros(8, dtype=int)
        for i in range(4):
            edof[2 * i] = 2 * enodes[i]
            edof[2 * i + 1] = 2 * enodes[i] + 1

        # Element-average material properties
        E_avg = np.mean(E_field.ravel()[enodes])
        eps_avg = np.mean(eps_growth_field.ravel()[enodes])

        C = _C_plane_strain(E_avg, nu)
        eps_growth_voigt = np.array([eps_avg, eps_avg, 0.0])

        Ke = np.zeros((8, 8))
        Fe = np.zeros(8)

        for gp in range(4):
            xi, eta = gpts[gp]
            w = gwts[gp]

            dN_dxi, dN_deta = _shape_grad(xi, eta)
            # dN/dx = Jinv @ [dN/dxi; dN/deta]
            dN_dx = Jinv[0, 0] * dN_dxi + Jinv[0, 1] * dN_deta
            dN_dy = Jinv[1, 0] * dN_dxi + Jinv[1, 1] * dN_deta

            B = _B_matrix(dN_dx, dN_dy)

            # Ke += B^T C B * detJ * w
            Ke += (B.T @ C @ B) * detJ * w

            # Fe += B^T C ε_growth * detJ * w  (eigenstrain load)
            Fe += (B.T @ C @ eps_growth_voigt) * detJ * w

        # Assemble global K (COO format)
        for i in range(8):
            for j in range(8):
                rows.append(edof[i])
                cols.append(edof[j])
                vals.append(Ke[i, j])

        # Assemble global F
        F[edof] += Fe

    K = sparse.coo_matrix((vals, (rows, cols)), shape=(n_dof, n_dof)).tocsr()

    # Boundary conditions
    fixed_dofs = set()
    if bc_type == "bottom_fixed":
        # Fix u_x and u_y on y=0 (tooth surface)
        for i in range(Nx):
            n = node_idx(i, 0)
            fixed_dofs.add(2 * n)
            fixed_dofs.add(2 * n + 1)
    elif bc_type == "left_fixed":
        # Fix u_x and u_y on x=0
        for j in range(Ny):
            n = node_idx(0, j)
            fixed_dofs.add(2 * n)
            fixed_dofs.add(2 * n + 1)

    fixed_dofs = sorted(fixed_dofs)
    free_dofs = sorted(set(range(n_dof)) - set(fixed_dofs))

    # Solve K_ff u_f = F_f
    K_ff = K[np.ix_(free_dofs, free_dofs)]
    F_f = F[free_dofs]

    u = np.zeros(n_dof)
    u[free_dofs] = spsolve(K_ff, F_f)

    # Post-process: element stress
    sigma_xx = np.zeros(n_elem)
    sigma_yy = np.zeros(n_elem)
    sigma_xy = np.zeros(n_elem)
    elem_centers = np.zeros((n_elem, 2))

    for e in range(n_elem):
        enodes = elems[e]
        edof = np.zeros(8, dtype=int)
        for i in range(4):
            edof[2 * i] = 2 * enodes[i]
            edof[2 * i + 1] = 2 * enodes[i] + 1

        E_avg = np.mean(E_field.ravel()[enodes])
        eps_avg = np.mean(eps_growth_field.ravel()[enodes])
        C = _C_plane_strain(E_avg, nu)

        u_e = u[edof]
        elem_centers[e] = np.mean(coords[enodes], axis=0)

        # Stress at element center (ξ=0, η=0)
        dN_dxi, dN_deta = _shape_grad(0, 0)
        dN_dx = Jinv[0, 0] * dN_dxi + Jinv[0, 1] * dN_deta
        dN_dy = Jinv[1, 0] * dN_dxi + Jinv[1, 1] * dN_deta
        B = _B_matrix(dN_dx, dN_dy)

        eps_total = B @ u_e
        eps_elastic = eps_total - np.array([eps_avg, eps_avg, 0.0])
        sigma = C @ eps_elastic

        sigma_xx[e] = sigma[0]
        sigma_yy[e] = sigma[1]
        sigma_xy[e] = sigma[2]

    # Von Mises stress (plane strain)
    sigma_vm = np.sqrt(sigma_xx**2 + sigma_yy**2 - sigma_xx * sigma_yy + 3 * sigma_xy**2)

    return {
        "u": u.reshape(-1, 2),
        "sigma_xx": sigma_xx,
        "sigma_yy": sigma_yy,
        "sigma_xy": sigma_xy,
        "sigma_vm": sigma_vm,
        "elem_centers": elem_centers,
        "coords": coords.reshape(Nx, Ny, 2),
        "u_grid": u.reshape(Nx, Ny, 2),
    }


# ============================================================================
# 2D Viscoelastic QUAD4 FEM (Generalized Maxwell / SLS)
# ============================================================================


def solve_2d_fem_viscoelastic(
    E_field,
    nu,
    eps_growth_field,
    Nx,
    Ny,
    Lx=1.0,
    Ly=1.0,
    bc_type="bottom_fixed",
    g1=0.5,
    tau1=10.0,
    t_total=100.0,
    dt=1.0,
    load_ramp_time=None,
):
    """Solve 2D plane-strain viscoelastic problem (1-term Prony / SLS).

    Generalized Maxwell model (1-term):
        G(t) = G_inf + G_1 * exp(-t/tau_1)
        G_inf = G_0 * (1 - g_1)
        G_1 = G_0 * g_1

    Time integration: exponential map (exact for piecewise-constant stress).
    State variable per Gauss point: eps_v (3,) viscous strain in Voigt notation.

    Parameters
    ----------
    E_field : (Nx, Ny) — instantaneous Young's modulus E_0 [Pa]
    nu : float — Poisson's ratio (assumed constant)
    eps_growth_field : (Nx, Ny) — isotropic eigenstrain
    g1 : float — Prony ratio G_1/G_0 (0 < g1 < 1)
    tau1 : float — relaxation time [s]
    t_total : float — total simulation time [s]
    dt : float — time step size [s]
    load_ramp_time : float — linear ramp duration (None = step load)

    Returns
    -------
    dict with time history of stress and displacement snapshots
    """
    n_nodes = Nx * Ny
    n_dof = 2 * n_nodes
    n_elem_x = Nx - 1
    n_elem_y = Ny - 1
    n_elem = n_elem_x * n_elem_y

    dx = Lx / n_elem_x
    dy = Ly / n_elem_y

    # Node coordinates
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    coords = np.column_stack([X.ravel(), Y.ravel()])

    def node_idx(ix, iy):
        return ix * Ny + iy

    elems = np.zeros((n_elem, 4), dtype=int)
    for i in range(n_elem_x):
        for j in range(n_elem_y):
            e = i * n_elem_y + j
            elems[e] = [
                node_idx(i, j),
                node_idx(i + 1, j),
                node_idx(i + 1, j + 1),
                node_idx(i, j + 1),
            ]

    # Gauss quadrature
    gpts, gwts = _gauss_2x2()

    # Jacobian for regular rectangular elements
    Jinv = np.array([[2 / dx, 0], [0, 2 / dy]])
    detJ = dx * dy / 4

    # Precompute B matrices at Gauss points
    B_gps = []
    for gp in range(4):
        xi, eta = gpts[gp]
        dN_dxi, dN_deta = _shape_grad(xi, eta)
        dN_dx = Jinv[0, 0] * dN_dxi + Jinv[0, 1] * dN_deta
        dN_dy = Jinv[1, 0] * dN_dxi + Jinv[1, 1] * dN_deta
        B_gps.append(_B_matrix(dN_dx, dN_dy))

    # BCs
    fixed_dofs = set()
    if bc_type == "bottom_fixed":
        for i in range(Nx):
            n = node_idx(i, 0)
            fixed_dofs.add(2 * n)
            fixed_dofs.add(2 * n + 1)
    elif bc_type == "left_fixed":
        for j in range(Ny):
            n = node_idx(0, j)
            fixed_dofs.add(2 * n)
            fixed_dofs.add(2 * n + 1)
    fixed_dofs = sorted(fixed_dofs)
    free_dofs = sorted(set(range(n_dof)) - set(fixed_dofs))

    # Precompute element-level data
    E_elem = np.zeros(n_elem)
    eps_g_elem = np.zeros(n_elem)
    C_elem = []
    for e in range(n_elem):
        enodes = elems[e]
        E_elem[e] = np.mean(E_field.ravel()[enodes])
        eps_g_elem[e] = np.mean(eps_growth_field.ravel()[enodes])
        C_elem.append(_C_plane_strain(E_elem[e], nu))

    # Deviatoric projection matrix: P_dev such that eps_dev = P_dev @ eps
    # In Voigt: eps_dev = eps - (1/3)*tr(eps)*[1,1,0]^T
    P_dev = np.array(
        [
            [2 / 3, -1 / 3, 0],
            [-1 / 3, 2 / 3, 0],
            [0, 0, 1],
        ]
    )

    # State variable: viscous deviatoric strain at each Gauss point
    eps_v = np.zeros((n_elem, 4, 3))  # (n_elem, 4 GPs, 3 Voigt)

    # Time stepping
    n_steps = max(1, int(np.ceil(t_total / dt)))
    dt_actual = t_total / n_steps

    # Exponential map factor
    exp_f = np.exp(-dt_actual / tau1)

    # Store snapshots
    snap_times = []
    snap_sigma_vm = []
    snap_u_max = []
    sigma_vm_all = None

    for step in range(n_steps + 1):
        t = step * dt_actual

        # Load factor (ramp or step)
        if load_ramp_time is not None and load_ramp_time > 0:
            lf = min(t / load_ramp_time, 1.0)
        else:
            lf = 1.0

        # Assembly with viscoelastic correction
        rows_k = []
        cols_k = []
        vals_k = []
        F_global = np.zeros(n_dof)

        for e in range(n_elem):
            enodes = elems[e]
            edof = np.zeros(8, dtype=int)
            for i in range(4):
                edof[2 * i] = 2 * enodes[i]
                edof[2 * i + 1] = 2 * enodes[i] + 1

            C = C_elem[e]
            eps_g_voigt = lf * np.array([eps_g_elem[e], eps_g_elem[e], 0.0])

            Ke = np.zeros((8, 8))
            Fe = np.zeros(8)

            for gp in range(4):
                B = B_gps[gp]
                w = gwts[gp]

                # Stiffness: use long-term modulus G_inf = G_0*(1-g1)
                # C_inf = C * (1-g1) for deviatoric, full C for volumetric
                # Simplified: K_eff assembled with E_inf = E_0*(1-g1) for shear
                # but this is approximate. Better: use full E and subtract viscous stress.
                Ke += (B.T @ C @ B) * detJ * w

                # Eigenstrain load + viscous stress correction
                # sigma = C @ (eps_total - eps_growth) - C_1 @ eps_v
                # F += B^T @ C @ eps_growth - B^T @ C_1 @ eps_v
                Fe += (B.T @ C @ eps_g_voigt) * detJ * w

                # Subtract viscous stress: C_dev * g1 * eps_v
                # C_dev = C @ P_dev (deviatoric part of C)
                C_dev_g1 = g1 * (C @ P_dev)
                Fe -= (B.T @ C_dev_g1 @ eps_v[e, gp]) * detJ * w

            for i in range(8):
                for j in range(8):
                    rows_k.append(edof[i])
                    cols_k.append(edof[j])
                    vals_k.append(Ke[i, j])
            F_global[edof] += Fe

        K = sparse.coo_matrix((vals_k, (rows_k, cols_k)), shape=(n_dof, n_dof)).tocsr()
        K_ff = K[np.ix_(free_dofs, free_dofs)]
        F_f = F_global[free_dofs]

        u = np.zeros(n_dof)
        u[free_dofs] = spsolve(K_ff, F_f)

        # Post-process: compute stress and update viscous strain
        sigma_xx = np.zeros(n_elem)
        sigma_yy = np.zeros(n_elem)
        sigma_xy = np.zeros(n_elem)

        for e in range(n_elem):
            enodes = elems[e]
            edof = np.zeros(8, dtype=int)
            for i in range(4):
                edof[2 * i] = 2 * enodes[i]
                edof[2 * i + 1] = 2 * enodes[i] + 1

            C = C_elem[e]
            eps_g_voigt = lf * np.array([eps_g_elem[e], eps_g_elem[e], 0.0])
            u_e = u[edof]

            # Stress at element center (average over GPs for simplicity)
            sig_sum = np.zeros(3)
            for gp in range(4):
                B = B_gps[gp]
                eps_total = B @ u_e
                eps_elastic = eps_total - eps_g_voigt

                # Deviatoric elastic strain
                eps_dev_elastic = P_dev @ eps_elastic

                # Viscous strain update: exponential map
                if step > 0:
                    eps_v[e, gp] = exp_f * eps_v[e, gp] + g1 * (1.0 - exp_f) * eps_dev_elastic

                # Stress: sigma = C @ eps_elastic - g1 * C @ P_dev @ eps_v
                sigma_gp = C @ eps_elastic - g1 * (C @ P_dev @ eps_v[e, gp])
                sig_sum += sigma_gp

            sig_avg = sig_sum / 4.0
            sigma_xx[e] = sig_avg[0]
            sigma_yy[e] = sig_avg[1]
            sigma_xy[e] = sig_avg[2]

        sigma_vm = np.sqrt(sigma_xx**2 + sigma_yy**2 - sigma_xx * sigma_yy + 3 * sigma_xy**2)
        sigma_vm_all = sigma_vm.copy()

        # Record snapshot
        u_mag = np.sqrt(u.reshape(-1, 2)[:, 0] ** 2 + u.reshape(-1, 2)[:, 1] ** 2)
        snap_times.append(t)
        snap_sigma_vm.append(sigma_vm.mean())
        snap_u_max.append(u_mag.max())

    # Final post-process using B at center for element stress
    elem_centers = np.zeros((n_elem, 2))
    for e in range(n_elem):
        elem_centers[e] = np.mean(coords[elems[e]], axis=0)

    return {
        "u": u.reshape(-1, 2),
        "sigma_xx": sigma_xx,
        "sigma_yy": sigma_yy,
        "sigma_xy": sigma_xy,
        "sigma_vm": sigma_vm_all,
        "elem_centers": elem_centers,
        "coords": coords.reshape(Nx, Ny, 2),
        "u_grid": u.reshape(Nx, Ny, 2),
        # Viscoelastic time history
        "snap_times": np.array(snap_times),
        "snap_sigma_vm_mean": np.array(snap_sigma_vm),
        "snap_u_max": np.array(snap_u_max),
        "g1": g1,
        "tau1": tau1,
        "t_total": t_total,
        "dt": dt_actual,
        "n_steps": n_steps,
    }


# ============================================================================
# 2D Viscoelastic QUAD4 FEM — Spatially-varying SLS (Simo 1987)
# ============================================================================


def solve_2d_fem_viscoelastic_sls(
    E_inf_field,
    E_1_field,
    tau_field,
    nu,
    eps_growth_field,
    Nx,
    Ny,
    t_array,
    Lx=1.0,
    Ly=1.0,
    bc_type="bottom_fixed",
):
    """Solve 2D plane-strain viscoelastic (SLS) with spatially-varying params.

    Uses Simo (1987) exponential integrator. Unconditionally stable.
    Internal variable h per element (Voigt: [h_xx, h_yy, h_xy]).

    Parameters
    ----------
    E_inf_field : (Nx, Ny) — equilibrium modulus [Pa]
    E_1_field   : (Nx, Ny) — Maxwell arm spring [Pa]
    tau_field   : (Nx, Ny) — relaxation time [s]
    nu : float — Poisson's ratio
    eps_growth_field : (Nx, Ny) — isotropic eigenstrain (step at t=0)
    t_array : (n_t,) — time points [s], t_array[0] = 0

    Returns
    -------
    dict with u_history, sigma_vm_history, snap data
    """
    n_nodes = Nx * Ny
    n_dof = 2 * n_nodes
    n_elem_x = Nx - 1
    n_elem_y = Ny - 1
    n_elem = n_elem_x * n_elem_y
    n_t = len(t_array)

    dx = Lx / n_elem_x
    dy = Ly / n_elem_y

    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    coords = np.column_stack([X.ravel(), Y.ravel()])

    def node_idx(ix, iy):
        return ix * Ny + iy

    elems = np.zeros((n_elem, 4), dtype=int)
    for i in range(n_elem_x):
        for j in range(n_elem_y):
            e = i * n_elem_y + j
            elems[e] = [
                node_idx(i, j),
                node_idx(i + 1, j),
                node_idx(i + 1, j + 1),
                node_idx(i, j + 1),
            ]

    gpts, gwts = _gauss_2x2()
    Jinv = np.array([[2 / dx, 0], [0, 2 / dy]])
    detJ = dx * dy / 4

    # Precompute B at each GP
    B_gps = []
    for gp in range(4):
        xi, eta = gpts[gp]
        dN_dxi, dN_deta = _shape_grad(xi, eta)
        dN_dx = Jinv[0, 0] * dN_dxi + Jinv[0, 1] * dN_deta
        dN_dy = Jinv[1, 0] * dN_dxi + Jinv[1, 1] * dN_deta
        B_gps.append(_B_matrix(dN_dx, dN_dy))

    # B at center for post-processing
    dN_dxi_c, dN_deta_c = _shape_grad(0, 0)
    dN_dx_c = Jinv[0, 0] * dN_dxi_c + Jinv[0, 1] * dN_deta_c
    dN_dy_c = Jinv[1, 0] * dN_dxi_c + Jinv[1, 1] * dN_deta_c
    B_center = _B_matrix(dN_dx_c, dN_dy_c)

    # BCs
    fixed_dofs = set()
    if bc_type == "bottom_fixed":
        for i in range(Nx):
            n = node_idx(i, 0)
            fixed_dofs.add(2 * n)
            fixed_dofs.add(2 * n + 1)
    elif bc_type == "left_fixed":
        for j in range(Ny):
            n = node_idx(0, j)
            fixed_dofs.add(2 * n)
            fixed_dofs.add(2 * n + 1)
    fixed_dofs = sorted(fixed_dofs)
    free_dofs = sorted(set(range(n_dof)) - set(fixed_dofs))

    # Element-average material properties + precompute C matrices
    E_inf_e = np.zeros(n_elem)
    E_1_e = np.zeros(n_elem)
    tau_e = np.zeros(n_elem)
    eps_g_e = np.zeros(n_elem)
    C_inf_list = []  # C(E_inf, nu) per element
    C_1_list = []  # C(E_1, nu) per element
    for e in range(n_elem):
        enodes = elems[e]
        E_inf_e[e] = np.mean(E_inf_field.ravel()[enodes])
        E_1_e[e] = np.mean(E_1_field.ravel()[enodes])
        tau_e[e] = np.mean(tau_field.ravel()[enodes])
        eps_g_e[e] = np.mean(eps_growth_field.ravel()[enodes])
        C_inf_list.append(_C_plane_strain(E_inf_e[e], nu))
        C_1_list.append(_C_plane_strain(E_1_e[e], nu))

    # Internal variable h per element (Voigt stress: [h_xx, h_yy, h_xy])
    h = np.zeros((n_elem, 3))
    eps_e_prev = np.zeros((n_elem, 3))

    u_history = np.zeros((n_t, n_nodes, 2))
    sigma_vm_history = np.zeros((n_t, n_elem))

    for ti in range(n_t):
        dt = t_array[ti] - t_array[ti - 1] if ti > 0 else 0.0

        if ti == 0:
            gamma_e = np.ones(n_elem)
            exp_dt_e = np.zeros(n_elem)
        elif dt > 1e-15:
            exp_dt_e = np.exp(-dt / tau_e)
            gamma_e = tau_e / dt * (1.0 - exp_dt_e)
        else:
            gamma_e = np.ones(n_elem)
            exp_dt_e = np.ones(n_elem)

        # h_star per element: exp(-dt/τ)*h - γ*C_1@ε_e_prev
        h_star = np.zeros((n_elem, 3))
        if ti > 0:
            for e in range(n_elem):
                h_star[e] = exp_dt_e[e] * h[e] - gamma_e[e] * (C_1_list[e] @ eps_e_prev[e])

        # Assemble: C_alg = C_inf + γ*C_1 = C(E_inf + γ*E_1, nu)
        rows_k, cols_k, vals_k = [], [], []
        F_global = np.zeros(n_dof)

        for e in range(n_elem):
            enodes = elems[e]
            edof = np.zeros(8, dtype=int)
            for i in range(4):
                edof[2 * i] = 2 * enodes[i]
                edof[2 * i + 1] = 2 * enodes[i] + 1

            C_alg = _C_plane_strain(E_inf_e[e] + E_1_e[e] * gamma_e[e], nu)
            eps_g_voigt = np.array([eps_g_e[e], eps_g_e[e], 0.0])

            Ke = np.zeros((8, 8))
            Fe = np.zeros(8)

            for gp in range(4):
                B = B_gps[gp]
                w = gwts[gp]
                Ke += (B.T @ C_alg @ B) * detJ * w
                Fe += (B.T @ (C_alg @ eps_g_voigt + h_star[e])) * detJ * w

            for i in range(8):
                for j in range(8):
                    rows_k.append(edof[i])
                    cols_k.append(edof[j])
                    vals_k.append(Ke[i, j])
            F_global[edof] += Fe

        K = sparse.coo_matrix((vals_k, (rows_k, cols_k)), shape=(n_dof, n_dof)).tocsr()
        K_ff = K[np.ix_(free_dofs, free_dofs)]
        F_f = F_global[free_dofs]

        u_vec = np.zeros(n_dof)
        u_vec[free_dofs] = spsolve(K_ff, F_f)

        # Post-process: stress and internal variable update
        sigma_xx = np.zeros(n_elem)
        sigma_yy = np.zeros(n_elem)
        sigma_xy = np.zeros(n_elem)

        for e in range(n_elem):
            enodes = elems[e]
            edof = np.zeros(8, dtype=int)
            for i in range(4):
                edof[2 * i] = 2 * enodes[i]
                edof[2 * i + 1] = 2 * enodes[i] + 1

            u_e = u_vec[edof]
            eps_total = B_center @ u_e
            eps_elastic = eps_total - np.array([eps_g_e[e], eps_g_e[e], 0.0])

            # Update h: h_{n+1} = exp*h_n + C_1*γ*(ε_e_{n+1} - ε_e_n)
            C_1 = C_1_list[e]
            if ti == 0:
                h[e] = C_1 @ eps_elastic
            else:
                h[e] = exp_dt_e[e] * h[e] + gamma_e[e] * (C_1 @ (eps_elastic - eps_e_prev[e]))

            eps_e_prev[e] = eps_elastic.copy()

            # σ = C_inf @ ε_e + h
            sigma = C_inf_list[e] @ eps_elastic + h[e]
            sigma_xx[e] = sigma[0]
            sigma_yy[e] = sigma[1]
            sigma_xy[e] = sigma[2]

        sigma_vm = np.sqrt(sigma_xx**2 + sigma_yy**2 - sigma_xx * sigma_yy + 3 * sigma_xy**2)

        u_history[ti] = u_vec.reshape(-1, 2)
        sigma_vm_history[ti] = sigma_vm

    # Element centers
    elem_centers = np.zeros((n_elem, 2))
    for e in range(n_elem):
        elem_centers[e] = np.mean(coords[elems[e]], axis=0)

    return {
        "u": u_vec.reshape(-1, 2),
        "u_history": u_history,
        "sigma_xx": sigma_xx,
        "sigma_yy": sigma_yy,
        "sigma_xy": sigma_xy,
        "sigma_vm": sigma_vm,
        "sigma_vm_history": sigma_vm_history,
        "elem_centers": elem_centers,
        "coords": coords.reshape(Nx, Ny, 2),
        "u_grid": u_vec.reshape(Nx, Ny, 2),
        "t_array": t_array,
    }


# ============================================================================
# Pipeline: Hamilton 2D ODE → DI → E(x,y) → FEM → stress
# ============================================================================


def run_2d_stress_pipeline(
    theta,
    Nx=20,
    Ny=20,
    Lx=1.0,
    Ly=1.0,
    n_macro=60,
    dt_h=1e-5,
    n_react_sub=20,
    save_every=60,
    K_hill=0.05,
    n_hill=4.0,
    nu=0.30,
    alpha_coeff=0.05,
    e_model="phi_pg",
    **kwargs,
):
    """Full pipeline: theta → 2D fields → FEM stress.

    Parameters
    ----------
    theta : (20,) array — Hamilton ODE parameters
    e_model : "phi_pg" | "virulence" | "di" — material model
    alpha_coeff : float — base eigenstrain coefficient

    Returns
    -------
    dict with phi_final, c_final, DI, E_field, eps_growth,
         and all FEM outputs (sigma_vm, u, etc.)
    """
    from JAXFEM.core_hamilton_2d_nutrient import run_simulation, Config2D

    cfg = Config2D(
        Nx=Nx,
        Ny=Ny,
        Lx=Lx,
        Ly=Ly,
        n_macro=n_macro,
        dt_h=dt_h,
        n_react_sub=n_react_sub,
        save_every=save_every,
        K_hill=K_hill,
        n_hill=n_hill,
    )

    t0 = time.perf_counter()
    result = run_simulation(theta, cfg)
    t_ode = time.perf_counter() - t0

    phi_snaps = np.array(result["phi_snaps"])  # (n_snap, 5, Nx, Ny)
    c_snaps = np.array(result["c_snaps"])  # (n_snap, Nx, Ny)

    phi_final = phi_snaps[-1]  # (5, Nx, Ny)
    c_final = c_snaps[-1]  # (Nx, Ny)

    # phi_final → (Nx, Ny, 5) for material_models
    phi_nxy5 = phi_final.transpose(1, 2, 0)

    # DI field
    DI = compute_di(phi_nxy5)  # (Nx, Ny)

    # E field
    if e_model == "phi_pg":
        E_field = compute_E_phi_pg(phi_nxy5)
    elif e_model == "virulence":
        E_field = compute_E_virulence(phi_nxy5)
    else:
        from material_models import compute_E_di

        E_field = compute_E_di(DI)

    # Eigenstrain: Monod-coupled growth
    phi_total = phi_nxy5.sum(axis=-1)  # (Nx, Ny)
    monod = c_final / (1.0 + c_final)
    eps_growth = alpha_coeff * phi_total * monod / 3.0  # isotropic → /3

    # FEM solve
    t1 = time.perf_counter()
    visco_kwargs = kwargs.get("visco", None)
    if visco_kwargs is not None:
        fem_result = solve_2d_fem_viscoelastic(
            E_field,
            nu,
            eps_growth,
            Nx,
            Ny,
            Lx,
            Ly,
            bc_type="bottom_fixed",
            **visco_kwargs,
        )
    else:
        fem_result = solve_2d_fem(
            E_field,
            nu,
            eps_growth,
            Nx,
            Ny,
            Lx,
            Ly,
            bc_type="bottom_fixed",
        )
    t_fem = time.perf_counter() - t1

    # Merge
    fem_result.update(
        {
            "phi_final": phi_final,
            "c_final": c_final,
            "DI": DI,
            "E_field": E_field,
            "eps_growth": eps_growth,
            "phi_total": phi_total,
            "e_model": e_model,
            "timing_ode_s": round(t_ode, 2),
            "timing_fem_s": round(t_fem, 4),
            "Nx": Nx,
            "Ny": Ny,
        }
    )
    return fem_result


def plot_2d_stress(result, outdir, condition="demo", show_title=True):
    """Generate 6-panel figure: fields + stress."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Nx, Ny = result["Nx"], result["Ny"]
    n_ex = Nx - 1
    n_ey = Ny - 1

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) DI field
    ax = axes[0, 0]
    im = ax.imshow(
        result["DI"].T, origin="lower", cmap="RdYlGn_r", aspect="equal", extent=[0, 1, 0, 1]
    )
    plt.colorbar(im, ax=ax, label="DI")
    ax.set_title("(a) Dysbiosis Index DI(x,y)")
    ax.set_xlabel("x (depth)")
    ax.set_ylabel("y (lateral)")

    # (b) E field
    ax = axes[0, 1]
    im = ax.imshow(
        result["E_field"].T, origin="lower", cmap="viridis", aspect="equal", extent=[0, 1, 0, 1]
    )
    plt.colorbar(im, ax=ax, label="E [Pa]")
    ax.set_title(f"(b) Young's modulus E(x,y) [{result['e_model']}]")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # (c) Eigenstrain
    ax = axes[0, 2]
    im = ax.imshow(
        result["eps_growth"].T, origin="lower", cmap="hot", aspect="equal", extent=[0, 1, 0, 1]
    )
    plt.colorbar(im, ax=ax, label="ε_growth")
    ax.set_title("(c) Growth eigenstrain ε_g(x,y)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # (d) σ_vm on element grid
    ax = axes[1, 0]
    svm = result["sigma_vm"].reshape(n_ex, n_ey)
    im = ax.imshow(svm.T, origin="lower", cmap="jet", aspect="equal", extent=[0, 1, 0, 1])
    plt.colorbar(im, ax=ax, label="σ_vm [Pa]")
    ax.set_title(f"(d) von Mises stress (max={svm.max():.2f} Pa)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # (e) Displacement magnitude
    ax = axes[1, 1]
    u_grid = result["u_grid"]  # (Nx, Ny, 2)
    u_mag = np.sqrt(u_grid[..., 0] ** 2 + u_grid[..., 1] ** 2)
    im = ax.imshow(u_mag.T, origin="lower", cmap="plasma", aspect="equal", extent=[0, 1, 0, 1])
    plt.colorbar(im, ax=ax, label="|u| [m]")
    ax.set_title(f"(e) Displacement magnitude (max={u_mag.max():.2e})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # (f) σ_xx profile along x at mid-y
    ax = axes[1, 2]
    mid_j = n_ey // 2
    sxx_mid = result["sigma_xx"].reshape(n_ex, n_ey)[:, mid_j]
    syy_mid = result["sigma_yy"].reshape(n_ex, n_ey)[:, mid_j]
    svm_mid = svm[:, mid_j]
    cx = result["elem_centers"][:, 0].reshape(n_ex, n_ey)[:, mid_j]
    ax.plot(cx, sxx_mid, "b-", lw=2, label="σ_xx")
    ax.plot(cx, syy_mid, "r--", lw=2, label="σ_yy")
    ax.plot(cx, svm_mid, "k:", lw=2, label="σ_vm")
    ax.set_xlabel("x (depth)")
    ax.set_ylabel("Stress [Pa]")
    ax.set_title("(f) Stress profile at y=0.5")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    if show_title:
        fig.suptitle(
            f"2D Biofilm Stress Analysis: {condition}\n"
            f"E model: {result['e_model']}, ν={0.3}, "
            f"ODE: {result['timing_ode_s']}s, FEM: {result['timing_fem_s']}s",
            fontsize=13,
            weight="bold",
        )

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    outpath = Path(outdir) / f"stress_2d_{condition}_{result['e_model']}.png"
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"  Figure: {outpath}")
    return str(outpath)


# ============================================================================
# CLI
# ============================================================================


def main():
    ap = argparse.ArgumentParser(description="2D FEM stress analysis")
    ap.add_argument("--condition", default="demo")
    ap.add_argument("--nx", type=int, default=20)
    ap.add_argument("--ny", type=int, default=20)
    ap.add_argument("--n-macro", type=int, default=60)
    ap.add_argument("--dt-h", type=float, default=1e-5)
    ap.add_argument("--n-react-sub", type=int, default=20)
    ap.add_argument("--k-hill", type=float, default=0.05)
    ap.add_argument("--n-hill", type=float, default=4.0)
    ap.add_argument("--nu", type=float, default=0.30)
    ap.add_argument("--alpha-coeff", type=float, default=0.05)
    ap.add_argument("--e-model", choices=["phi_pg", "virulence", "di"], default="phi_pg")
    ap.add_argument("--outdir", default=None)
    ap.add_argument(
        "--viscoelastic", action="store_true", help="Use viscoelastic (SLS) time-dependent solver"
    )
    ap.add_argument(
        "--prony-g1",
        type=float,
        default=0.5,
        help="Prony g1 = G_1/G_0 shear relaxation ratio (default 0.5)",
    )
    ap.add_argument(
        "--prony-tau1",
        type=float,
        default=10.0,
        help="Prony tau_1 relaxation time in seconds (default 10.0)",
    )
    ap.add_argument(
        "--t-total",
        type=float,
        default=100.0,
        help="Total viscoelastic simulation time (seconds, default 100)",
    )
    ap.add_argument(
        "--dt-visco", type=float, default=1.0, help="Viscoelastic time step (seconds, default 1.0)"
    )
    args = ap.parse_args()

    # Load theta
    _TMCMC = _HERE.parent
    _RUNS = _TMCMC / "data_5species" / "_runs"
    theta = None

    if args.condition != "demo":
        theta_path = _RUNS / args.condition / "theta_MAP.json"
        if theta_path.exists():
            with open(theta_path) as f:
                d = json.load(f)
            if "theta_full" in d:
                theta = np.array(d["theta_full"], dtype=np.float64)
            elif "theta_sub" in d:
                theta = np.array(d["theta_sub"], dtype=np.float64)

    if theta is None:
        # Demo theta (mild-weight MAP)
        theta = np.array(
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

    outdir = args.outdir or str(_HERE.parent / "_stress_2d_results")
    Path(outdir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("2D FEM Stress Analysis")
    print(f"  Condition: {args.condition}")
    print(f"  Grid: {args.nx}×{args.ny}")
    print(f"  E model: {args.e_model}")
    if args.viscoelastic:
        print(
            f"  VISCO: g1={args.prony_g1}, tau1={args.prony_tau1}s, "
            f"t_total={args.t_total}s, dt={args.dt_visco}s"
        )
    print(f"  Output: {outdir}")
    print("=" * 60)

    extra_kwargs = {}
    if args.viscoelastic:
        extra_kwargs["visco"] = {
            "g1": args.prony_g1,
            "tau1": args.prony_tau1,
            "t_total": args.t_total,
            "dt": args.dt_visco,
        }

    result = run_2d_stress_pipeline(
        theta,
        Nx=args.nx,
        Ny=args.ny,
        n_macro=args.n_macro,
        dt_h=args.dt_h,
        n_react_sub=args.n_react_sub,
        K_hill=args.k_hill,
        n_hill=args.n_hill,
        nu=args.nu,
        alpha_coeff=args.alpha_coeff,
        e_model=args.e_model,
        **extra_kwargs,
    )

    # Summary
    svm = result["sigma_vm"]
    print(f"\n  DI: mean={result['DI'].mean():.4f}, max={result['DI'].max():.4f}")
    print(f"  E:  mean={result['E_field'].mean():.1f} Pa, min={result['E_field'].min():.1f}")
    print(f"  ε_g: mean={result['eps_growth'].mean():.6f}, max={result['eps_growth'].max():.6f}")
    print(f"  σ_vm: mean={svm.mean():.3f} Pa, max={svm.max():.3f} Pa")
    print(f"  |u|_max: {np.max(np.sqrt(result['u'][:, 0]**2 + result['u'][:, 1]**2)):.6e}")
    print(f"  Time: ODE={result['timing_ode_s']}s, FEM={result['timing_fem_s']}s")

    # Save results
    summary = {
        "condition": args.condition,
        "e_model": args.e_model,
        "grid": f"{args.nx}x{args.ny}",
        "DI_mean": float(result["DI"].mean()),
        "DI_max": float(result["DI"].max()),
        "E_mean_pa": float(result["E_field"].mean()),
        "E_min_pa": float(result["E_field"].min()),
        "eps_growth_max": float(result["eps_growth"].max()),
        "sigma_vm_max_pa": float(svm.max()),
        "sigma_vm_mean_pa": float(svm.mean()),
        "u_max": float(np.max(np.sqrt(result["u"][:, 0] ** 2 + result["u"][:, 1] ** 2))),
        "timing_ode_s": result["timing_ode_s"],
        "timing_fem_s": result["timing_fem_s"],
    }
    with open(Path(outdir) / f"summary_{args.condition}_{args.e_model}.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Plot
    plot_2d_stress(result, outdir, args.condition)

    # Also run with virulence model for comparison if not already
    if args.e_model == "phi_pg":
        print("\n--- Also running virulence model for comparison ---")
        result_vir = run_2d_stress_pipeline(
            theta,
            Nx=args.nx,
            Ny=args.ny,
            n_macro=args.n_macro,
            dt_h=args.dt_h,
            n_react_sub=args.n_react_sub,
            K_hill=args.k_hill,
            n_hill=args.n_hill,
            nu=args.nu,
            alpha_coeff=args.alpha_coeff,
            e_model="virulence",
        )
        svm_v = result_vir["sigma_vm"]
        print(f"  σ_vm (virulence): mean={svm_v.mean():.3f}, max={svm_v.max():.3f} Pa")
        plot_2d_stress(result_vir, outdir, args.condition)

    print(f"\nDone. Results in: {outdir}")


if __name__ == "__main__":
    main()
