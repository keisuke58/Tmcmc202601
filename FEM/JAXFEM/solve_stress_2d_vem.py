#!/usr/bin/env python3
"""
solve_stress_2d_vem.py — VEM drop-in replacement for solve_2d_fem.
============================================================================

Same interface as solve_2d_fem() in solve_stress_2d.py:
  Input:  E_field (Nx,Ny), eps_growth_field (Nx,Ny), nu, Nx, Ny, Lx, Ly
  Output: dict with u, sigma_vm, sigma_xx, sigma_yy, sigma_xy, elem_centers, ...

Internally uses the VEM elasticity solver from VirtualElementMethods repo
on a Voronoi mesh generated from a regular seed grid (matching Nx×Ny).

Two mesh modes:
  - 'grid':    regular quadrilateral mesh (≈ equivalent to Q4 FEM)
  - 'voronoi': Voronoi tessellation from jittered seeds (arbitrary polygons)

Usage in staggered coupling:
    from solve_stress_2d_vem import solve_2d_vem
    result = solve_2d_vem(E_field, nu, eps_growth, Nx, Ny, Lx, Ly)
"""

import sys
from pathlib import Path

import numpy as np
from scipy.spatial import Voronoi
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

_HERE = Path(__file__).resolve().parent
_VEM_REPO = Path.home() / "IKM_Hiwi" / "VirtualElementMethods"
if str(_VEM_REPO) not in sys.path:
    sys.path.insert(0, str(_VEM_REPO))

from vem_elasticity import vem_elasticity


# ── Mesh generation ──────────────────────────────────────────────────────


def _generate_voronoi_mesh(Nx, Ny, Lx, Ly, jitter=0.25, seed=42):
    """Generate clipped Voronoi mesh on [0,Lx]×[0,Ly] from (Nx-1)×(Ny-1) seeds.

    Returns vertices, elements (list of int arrays), seed_to_elem mapping.
    """
    rng = np.random.default_rng(seed)
    n_ex, n_ey = Nx - 1, Ny - 1
    dx = Lx / n_ex
    dy = Ly / n_ey

    # Seed points at element centers with optional jitter
    seeds = []
    for i in range(n_ex):
        for j in range(n_ey):
            cx = (i + 0.5) * dx + jitter * dx * (rng.random() - 0.5)
            cy = (j + 0.5) * dy + jitter * dy * (rng.random() - 0.5)
            cx = np.clip(cx, 0.05 * dx, Lx - 0.05 * dx)
            cy = np.clip(cy, 0.05 * dy, Ly - 0.05 * dy)
            seeds.append([cx, cy])
    seeds = np.array(seeds)
    n_seeds = len(seeds)

    # Mirror seeds for bounded Voronoi
    mirrored = np.vstack(
        [
            seeds,
            np.column_stack([-seeds[:, 0], seeds[:, 1]]),  # left
            np.column_stack([2 * Lx - seeds[:, 0], seeds[:, 1]]),  # right
            np.column_stack([seeds[:, 0], -seeds[:, 1]]),  # bottom
            np.column_stack([seeds[:, 0], 2 * Ly - seeds[:, 1]]),  # top
        ]
    )

    vor = Voronoi(mirrored)

    # Clip each real seed's region to [0,Lx]×[0,Ly]
    all_verts = []
    elements = []
    vert_map = {}  # (rounded_x, rounded_y) → global vertex index

    def _get_vert_id(x, y):
        key = (round(x, 10), round(y, 10))
        if key not in vert_map:
            vert_map[key] = len(all_verts)
            all_verts.append([x, y])
        return vert_map[key]

    for s in range(n_seeds):
        region_idx = vor.point_region[s]
        region_verts = vor.regions[region_idx]
        if -1 in region_verts or len(region_verts) < 3:
            continue

        poly = vor.vertices[region_verts]
        clipped = _clip_polygon_to_box(poly, 0, Lx, 0, Ly)
        if len(clipped) < 3:
            continue

        # Order vertices CCW
        cx, cy = clipped.mean(axis=0)
        angles = np.arctan2(clipped[:, 1] - cy, clipped[:, 0] - cx)
        order = np.argsort(angles)
        clipped = clipped[order]

        el_vids = []
        for v in clipped:
            el_vids.append(_get_vert_id(v[0], v[1]))
        elements.append(np.array(el_vids, dtype=int))

    vertices = np.array(all_verts)
    return vertices, elements, seeds


def _clip_polygon_to_box(poly, xmin, xmax, ymin, ymax):
    """Sutherland-Hodgman clipping of polygon to axis-aligned box."""
    output = poly.tolist()
    for edge in [
        (lambda p: p[0] >= xmin, lambda p, q: _intersect_x(p, q, xmin)),
        (lambda p: p[0] <= xmax, lambda p, q: _intersect_x(p, q, xmax)),
        (lambda p: p[1] >= ymin, lambda p, q: _intersect_y(p, q, ymin)),
        (lambda p: p[1] <= ymax, lambda p, q: _intersect_y(p, q, ymax)),
    ]:
        inside, intersect = edge
        if len(output) == 0:
            break
        inp = output
        output = []
        for i in range(len(inp)):
            curr = inp[i]
            prev = inp[i - 1]
            if inside(curr):
                if not inside(prev):
                    output.append(intersect(prev, curr))
                output.append(curr)
            elif inside(prev):
                output.append(intersect(prev, curr))
    return np.array(output) if len(output) >= 3 else np.zeros((0, 2))


def _intersect_x(p, q, x):
    t = (x - p[0]) / (q[0] - p[0]) if abs(q[0] - p[0]) > 1e-15 else 0.5
    return [x, p[1] + t * (q[1] - p[1])]


def _intersect_y(p, q, y):
    t = (y - p[1]) / (q[1] - p[1]) if abs(q[1] - p[1]) > 1e-15 else 0.5
    return [p[0] + t * (q[0] - p[0]), y]


def _generate_quad_mesh(Nx, Ny, Lx, Ly):
    """Generate regular quadrilateral mesh (equivalent to Q4 FEM grid)."""
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    vertices = np.column_stack([X.ravel(), Y.ravel()])

    n_ex, n_ey = Nx - 1, Ny - 1
    elements = []
    for i in range(n_ex):
        for j in range(n_ey):
            n0 = i * Ny + j
            n1 = (i + 1) * Ny + j
            n2 = (i + 1) * Ny + (j + 1)
            n3 = i * Ny + (j + 1)
            elements.append(np.array([n0, n1, n2, n3], dtype=int))

    return vertices, elements


# ── Eigenstrain load vector for VEM ──────────────────────────────────────


def _compute_eigenstrain_load(
    vertices, elements, E_per_elem, nu, eps_per_elem, stress_type="plane_strain"
):
    """Compute global force vector from eigenstrain loading for VEM.

    For each element, F_e = K_e @ u_growth where u_growth produces
    the target eigenstrain. Since eigenstrain is isotropic (ε_xx = ε_yy = ε_g),
    the equivalent thermal displacement at each vertex is:
        u_growth_x = ε_g * (x - xc)
        u_growth_y = ε_g * (y - yc)
    """
    n_dofs = 2 * vertices.shape[0]
    F = np.zeros(n_dofs)

    for el_id, el in enumerate(elements):
        verts = vertices[el.astype(int)]
        n_v = len(el)
        cx, cy = verts.mean(axis=0)

        eps_g = eps_per_elem[el_id]
        if abs(eps_g) < 1e-15:
            continue

        E_el = E_per_elem[el_id]
        if stress_type == "plane_stress":
            factor = E_el / (1.0 - nu**2)
            C = factor * np.array(
                [
                    [1.0, nu, 0.0],
                    [nu, 1.0, 0.0],
                    [0.0, 0.0, (1.0 - nu) / 2.0],
                ]
            )
        else:
            factor = E_el / ((1.0 + nu) * (1.0 - 2.0 * nu))
            C = factor * np.array(
                [
                    [1.0 - nu, nu, 0.0],
                    [nu, 1.0 - nu, 0.0],
                    [0.0, 0.0, (1.0 - 2.0 * nu) / 2.0],
                ]
            )

        # σ_growth = C @ [ε_g, ε_g, 0]
        sigma_g = C @ np.array([eps_g, eps_g, 0.0])

        # Integrate traction from growth stress over element boundary
        # For polygon: ∮ N^T σ·n ds  ≈  Σ_edges (σ·n_edge * L_edge / 2) at each vertex
        for i in range(n_v):
            i_next = (i + 1) % n_v
            v0 = verts[i]
            v1 = verts[i_next]
            edge = v1 - v0
            L_edge = np.linalg.norm(edge)
            # Outward normal (CCW polygon → rotate edge 90° CW)
            n_edge = np.array([edge[1], -edge[0]])
            n_norm = np.linalg.norm(n_edge)
            if n_norm < 1e-15:
                continue
            n_edge = n_edge / n_norm

            # Traction: t = σ · n
            tx = sigma_g[0] * n_edge[0] + sigma_g[2] * n_edge[1]
            ty = sigma_g[2] * n_edge[0] + sigma_g[1] * n_edge[1]

            # Distribute to both vertices of edge (linear shape function)
            for vid, weight in [(el[i], 0.5), (el[i_next], 0.5)]:
                F[2 * vid] += tx * L_edge * weight
                F[2 * vid + 1] += ty * L_edge * weight

    return F


# ── Stress postprocessing for VEM ────────────────────────────────────────


def _postprocess_stress_vem(
    vertices, elements, u_flat, E_per_elem, nu, eps_per_elem, stress_type="plane_strain"
):
    """Compute element-average stress from VEM displacement solution.

    Uses the VEM gradient projector Π^∇: the B matrix rows 3-5 give
    the projected average strain directly from nodal DOFs.
    This is consistent with the energy used in assembly (no post-hoc
    boundary integral needed).

    ε̄ = [B[3,:], B[4,:], B[5,:]] @ u_local  (scaled by strain_basis)
    where strain_basis[α] = [1/h, 0, 0; 0, 1/h, 0; 0, 0, 2/h]
    so the raw projection coefficients c = projector[3:6, :] @ u_local
    and ε̄ = diag(1/h, 1/h, 2/h) @ c  (Voigt: [ε_xx, ε_yy, 2ε_xy])
    → actual [ε_xx, ε_yy, ε_xy] = [c[0]/h, c[1]/h, c[2]/h]
    """
    n_elem = len(elements)
    sigma_xx = np.zeros(n_elem)
    sigma_yy = np.zeros(n_elem)
    sigma_xy = np.zeros(n_elem)
    elem_centers = np.zeros((n_elem, 2))

    for el_id, el in enumerate(elements):
        el_int = el.astype(int)
        verts = vertices[el_int]
        n_v = len(el_int)
        n_el_dofs = 2 * n_v

        # Element geometry
        area_comp = verts[:, 0] * np.roll(verts[:, 1], -1) - np.roll(verts[:, 0], -1) * verts[:, 1]
        area = 0.5 * abs(np.sum(area_comp))
        centroid = np.sum((np.roll(verts, -1, axis=0) + verts) * area_comp[:, None], axis=0) / (
            6.0 * area
        )
        elem_centers[el_id] = centroid

        if area < 1e-20:
            continue

        h = max(np.linalg.norm(verts[i] - verts[j]) for i in range(n_v) for j in range(i + 1, n_v))
        xc, yc = centroid
        E_el = E_per_elem[el_id]

        # Constitutive matrix (same as assembly)
        if stress_type == "plane_stress":
            factor = E_el / (1.0 - nu**2)
            C = factor * np.array(
                [
                    [1.0, nu, 0.0],
                    [nu, 1.0, 0.0],
                    [0.0, 0.0, (1.0 - nu) / 2.0],
                ]
            )
        else:
            factor = E_el / ((1.0 + nu) * (1.0 - 2.0 * nu))
            C = factor * np.array(
                [
                    [1.0 - nu, nu, 0.0],
                    [nu, 1.0 - nu, 0.0],
                    [0.0, 0.0, (1.0 - 2.0 * nu) / 2.0],
                ]
            )

        # ── Rebuild VEM projector (same as in assembly) ──
        D = np.zeros((n_el_dofs, 6))
        for i in range(n_v):
            dx = (verts[i, 0] - xc) / h
            dy = (verts[i, 1] - yc) / h
            D[2 * i, :] = [1.0, 0.0, -dy, dx, 0.0, dy]
            D[2 * i + 1, :] = [0.0, 1.0, dx, 0.0, dy, dx]

        B = np.zeros((6, n_el_dofs))
        vertex_normals = np.zeros((n_v, 2))
        for i in range(n_v):
            prev_v = verts[(i - 1) % n_v]
            next_v = verts[(i + 1) % n_v]
            vertex_normals[i] = [next_v[1] - prev_v[1], prev_v[0] - next_v[0]]

        for i in range(n_v):
            B[0, 2 * i] = 1.0 / n_v
            B[1, 2 * i + 1] = 1.0 / n_v
        for i in range(n_v):
            B[2, 2 * i] = -vertex_normals[i, 1] / (4.0 * area)
            B[2, 2 * i + 1] = vertex_normals[i, 0] / (4.0 * area)

        strain_basis = np.array(
            [
                [1.0 / h, 0.0, 0.0],
                [0.0, 1.0 / h, 0.0],
                [0.0, 0.0, 2.0 / h],
            ]
        )
        for i in range(n_v):
            vn = vertex_normals[i]
            for alpha in range(3):
                sigma_b = C @ strain_basis[alpha]
                tx = sigma_b[0] * vn[0] + sigma_b[2] * vn[1]
                ty = sigma_b[2] * vn[0] + sigma_b[1] * vn[1]
                B[3 + alpha, 2 * i] += 0.5 * tx
                B[3 + alpha, 2 * i + 1] += 0.5 * ty

        G = B @ D
        projector = np.linalg.solve(G, B)  # (6, n_el_dofs)

        # Extract local displacement vector
        u_local = np.zeros(n_el_dofs)
        for i in range(n_v):
            u_local[2 * i] = u_flat[2 * el_int[i]]
            u_local[2 * i + 1] = u_flat[2 * el_int[i] + 1]

        # Project: polynomial coefficients c = projector @ u_local
        c = projector @ u_local  # (6,): [c_tx, c_ty, c_rot, c_exx, c_eyy, c_exy]

        # Strain from polynomial basis:
        #   p4 = ((x-xc)/h, 0)       → ε_xx = 1/h, ε_yy = 0, ε_xy = 0
        #   p5 = (0, (y-yc)/h)       → ε_xx = 0, ε_yy = 1/h, ε_xy = 0
        #   p6 = ((y-yc)/h, (x-xc)/h)→ ε_xx = 0, ε_yy = 0, ε_xy = 1/h
        eps_total = np.array([c[3] / h, c[4] / h, c[5] / h])

        # Elastic strain = total - eigenstrain
        eps_g = eps_per_elem[el_id]
        eps_el = eps_total - np.array([eps_g, eps_g, 0.0])

        sigma = C @ eps_el
        sigma_xx[el_id] = sigma[0]
        sigma_yy[el_id] = sigma[1]
        sigma_xy[el_id] = sigma[2]

    sigma_vm = np.sqrt(sigma_xx**2 + sigma_yy**2 - sigma_xx * sigma_yy + 3 * sigma_xy**2)

    return sigma_xx, sigma_yy, sigma_xy, sigma_vm, elem_centers


def _spr_stress_recovery(vertices, elements, elem_centers, sigma_xx, sigma_yy, sigma_xy):
    """Superconvergent Patch Recovery (SPR, Zienkiewicz-Zhu 1992).

    For each element, fit a local linear polynomial through the stress
    values of neighboring elements (patch = elements sharing a node),
    then evaluate at the element centroid.  This smooths noise from
    the constant-strain VEM projection and recovers superconvergent
    stress at element centers.

    Returns improved sigma_xx, sigma_yy, sigma_xy, sigma_vm.
    """
    n_nodes = vertices.shape[0]
    n_elem = len(elements)

    # Build node → element adjacency
    node_to_elems = [[] for _ in range(n_nodes)]
    for el_id, el in enumerate(elements):
        for vid in el.astype(int):
            node_to_elems[vid].append(el_id)

    # Build element → neighbor set (elements sharing at least one node)
    elem_neighbors = [set() for _ in range(n_elem)]
    for el_id, el in enumerate(elements):
        for vid in el.astype(int):
            for nb in node_to_elems[vid]:
                elem_neighbors[el_id].add(nb)

    sxx_spr = np.zeros(n_elem)
    syy_spr = np.zeros(n_elem)
    sxy_spr = np.zeros(n_elem)

    for el_id in range(n_elem):
        patch = sorted(elem_neighbors[el_id])
        n_patch = len(patch)
        cx, cy = elem_centers[el_id]

        if n_patch < 3:
            # Not enough neighbors for linear fit → keep raw
            sxx_spr[el_id] = sigma_xx[el_id]
            syy_spr[el_id] = sigma_yy[el_id]
            sxy_spr[el_id] = sigma_xy[el_id]
            continue

        # Fit linear polynomial σ(x,y) = a0 + a1*(x-cx) + a2*(y-cy)
        # through patch element centroids
        A = np.zeros((n_patch, 3))
        for i, nb in enumerate(patch):
            A[i, 0] = 1.0
            A[i, 1] = elem_centers[nb, 0] - cx
            A[i, 2] = elem_centers[nb, 1] - cy

        # Weighted least squares (closer elements have higher weight)
        # w = 1/d² with regularization
        dists = np.sqrt((A[:, 1]) ** 2 + (A[:, 2]) ** 2)
        h_avg = np.median(dists[dists > 1e-15]) if np.any(dists > 1e-15) else 1.0
        w = 1.0 / (dists**2 + (0.1 * h_avg) ** 2)
        W = np.diag(w)

        AtWA = A.T @ W @ A
        try:
            AtWA_inv = np.linalg.solve(AtWA, np.eye(3))
        except np.linalg.LinAlgError:
            sxx_spr[el_id] = sigma_xx[el_id]
            syy_spr[el_id] = sigma_yy[el_id]
            sxy_spr[el_id] = sigma_xy[el_id]
            continue

        # Evaluate at element centroid (dx=0, dy=0) → only a0 matters
        # coeffs = AtWA_inv @ A^T @ W @ σ_patch
        for comp, raw, spr_out in [
            (0, sigma_xx, sxx_spr),
            (1, sigma_yy, syy_spr),
            (2, sigma_xy, sxy_spr),
        ]:
            b = np.array([raw[nb] for nb in patch])
            coeffs = AtWA_inv @ (A.T @ (W @ b))
            spr_out[el_id] = coeffs[0]  # value at centroid

    svm_spr = np.sqrt(sxx_spr**2 + syy_spr**2 - sxx_spr * syy_spr + 3 * sxy_spr**2)

    return sxx_spr, syy_spr, sxy_spr, svm_spr


# ── Main solver (drop-in for solve_2d_fem) ───────────────────────────────


def solve_2d_vem(
    E_field,
    nu,
    eps_growth_field,
    Nx,
    Ny,
    Lx=1.0,
    Ly=1.0,
    bc_type="bottom_fixed",
    stress_type="plane_strain",
    mesh_type="voronoi",
    voronoi_jitter=0.25,
    voronoi_seed=42,
    stabilization_alpha=0.5,
    stress_recovery="raw",
):
    """Solve 2D elasticity using VEM on polygonal mesh.

    Drop-in replacement for solve_2d_fem(). Same input/output interface.

    Parameters
    ----------
    E_field : (Nx, Ny) — Young's modulus at each grid node [Pa]
    nu : float — Poisson's ratio
    eps_growth_field : (Nx, Ny) — isotropic eigenstrain at each grid node
    Nx, Ny : int — grid dimensions (for field interpolation)
    Lx, Ly : float — domain size [m]
    bc_type : str — "bottom_fixed" or "left_fixed"
    stress_type : str — "plane_strain" or "plane_stress"
    mesh_type : str — "voronoi" (arbitrary polygons) or "grid" (quads)
    voronoi_jitter : float — seed perturbation (0 = regular, 0.5 = max)
    voronoi_seed : int — RNG seed for reproducibility
    stabilization_alpha : float — VEM stabilization parameter

    Returns
    -------
    dict with same keys as solve_2d_fem:
        u, sigma_xx, sigma_yy, sigma_xy, sigma_vm, elem_centers,
        coords, u_grid, geom_nonlin_ratio
    Plus VEM-specific:
        vem_vertices, vem_elements (the polygonal mesh)
    """
    # Generate mesh
    if mesh_type == "voronoi":
        vertices, elements, seeds = _generate_voronoi_mesh(
            Nx, Ny, Lx, Ly, jitter=voronoi_jitter, seed=voronoi_seed
        )
    else:
        vertices, elements = _generate_quad_mesh(Nx, Ny, Lx, Ly)

    n_nodes = vertices.shape[0]
    n_dofs = 2 * n_nodes
    n_elem = len(elements)

    # Interpolate E and eps_growth from grid to VEM elements
    # (element centroid → bilinear interpolation from grid)
    E_per_elem = np.zeros(n_elem)
    eps_per_elem = np.zeros(n_elem)

    for el_id, el in enumerate(elements):
        verts = vertices[el.astype(int)]
        cx, cy = verts.mean(axis=0)

        # Grid coordinates for interpolation
        ix = cx / Lx * (Nx - 1)
        iy = cy / Ly * (Ny - 1)
        ix = np.clip(ix, 0, Nx - 2)
        iy = np.clip(iy, 0, Ny - 2)
        i0, j0 = int(ix), int(iy)
        fx, fy = ix - i0, iy - j0

        # Bilinear interpolation
        E_per_elem[el_id] = (
            (1 - fx) * (1 - fy) * E_field[i0, j0]
            + fx * (1 - fy) * E_field[min(i0 + 1, Nx - 1), j0]
            + (1 - fx) * fy * E_field[i0, min(j0 + 1, Ny - 1)]
            + fx * fy * E_field[min(i0 + 1, Nx - 1), min(j0 + 1, Ny - 1)]
        )
        eps_per_elem[el_id] = (
            (1 - fx) * (1 - fy) * eps_growth_field[i0, j0]
            + fx * (1 - fy) * eps_growth_field[min(i0 + 1, Nx - 1), j0]
            + (1 - fx) * fy * eps_growth_field[i0, min(j0 + 1, Ny - 1)]
            + fx * fy * eps_growth_field[min(i0 + 1, Nx - 1), min(j0 + 1, Ny - 1)]
        )

    # Boundary conditions: fix bottom (y=0) or left (x=0)
    tol = Ly / (Ny - 1) * 0.5 if bc_type == "bottom_fixed" else Lx / (Nx - 1) * 0.5
    bc_fixed_dofs = []
    bc_vals = []
    for i in range(n_nodes):
        if bc_type == "bottom_fixed" and vertices[i, 1] < tol:
            bc_fixed_dofs.extend([2 * i, 2 * i + 1])
            bc_vals.extend([0.0, 0.0])
        elif bc_type == "left_fixed" and vertices[i, 0] < tol:
            bc_fixed_dofs.extend([2 * i, 2 * i + 1])
            bc_vals.extend([0.0, 0.0])

    bc_fixed_dofs = np.array(bc_fixed_dofs, dtype=int)
    bc_vals = np.array(bc_vals)

    # Compute eigenstrain load vector
    F_growth = _compute_eigenstrain_load(
        vertices, elements, E_per_elem, nu, eps_per_elem, stress_type
    )

    # Solve VEM with eigenstrain as external load
    # We need to solve K u = F_growth with BCs
    # vem_elasticity only supports point loads, so we use it for K assembly
    # then apply our own load vector.
    u_flat = _solve_vem_with_body_load(
        vertices,
        elements,
        E_per_elem,
        nu,
        bc_fixed_dofs,
        bc_vals,
        F_growth,
        stabilization_alpha,
        stress_type,
    )

    # Stress postprocessing (projector-based)
    sigma_xx, sigma_yy, sigma_xy, sigma_vm, elem_centers = _postprocess_stress_vem(
        vertices, elements, u_flat, E_per_elem, nu, eps_per_elem, stress_type
    )

    # Optional SPR stress recovery (Zienkiewicz-Zhu 1992)
    if stress_recovery == "spr" and n_elem >= 3:
        sigma_xx, sigma_yy, sigma_xy, sigma_vm = _spr_stress_recovery(
            vertices, elements, elem_centers, sigma_xx, sigma_yy, sigma_xy
        )

    # Interpolate displacement back to grid for compatibility
    u_grid = np.zeros((Nx, Ny, 2))
    x_grid = np.linspace(0, Lx, Nx)
    y_grid = np.linspace(0, Ly, Ny)

    # For each grid point, find nearest VEM node
    for i in range(Nx):
        for j in range(Ny):
            gp = np.array([x_grid[i], y_grid[j]])
            dists = np.linalg.norm(vertices - gp, axis=1)
            nearest = np.argmin(dists)
            u_grid[i, j, 0] = u_flat[2 * nearest]
            u_grid[i, j, 1] = u_flat[2 * nearest + 1]

    # Reshape u to (n_grid_nodes, 2) for compatibility
    u_on_grid = u_grid.reshape(-1, 2)

    # Geometric nonlinearity diagnostic
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    if Nx > 2 and Ny > 2:
        du_dx = np.zeros((Nx - 2, Ny - 2, 2, 2))
        for d in range(2):
            du_dx[:, :, d, 0] = (u_grid[2:, 1:-1, d] - u_grid[:-2, 1:-1, d]) / (2 * dx)
            du_dx[:, :, d, 1] = (u_grid[1:-1, 2:, d] - u_grid[1:-1, :-2, d]) / (2 * dy)
        grad_u_norm = np.sqrt(np.sum(du_dx**2, axis=(2, 3)))
        geom_nonlin_ratio = float(grad_u_norm.max())
    else:
        geom_nonlin_ratio = 0.0

    # Grid coords for compatibility
    X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")
    coords = np.stack([X, Y], axis=-1)

    return {
        "u": u_on_grid,
        "sigma_xx": sigma_xx,
        "sigma_yy": sigma_yy,
        "sigma_xy": sigma_xy,
        "sigma_vm": sigma_vm,
        "elem_centers": elem_centers,
        "coords": coords,
        "u_grid": u_grid,
        "geom_nonlin_ratio": geom_nonlin_ratio,
        # VEM-specific
        "vem_vertices": vertices,
        "vem_elements": elements,
    }


def _solve_vem_with_body_load(
    vertices,
    elements,
    E_per_elem,
    nu,
    bc_fixed_dofs,
    bc_vals,
    F_load,
    stabilization_alpha,
    stress_type,
):
    """Assemble VEM stiffness + apply body load + solve.

    Reimplements vem_elasticity assembly to support arbitrary load vectors
    and plane strain/stress selection.
    """
    n_nodes = vertices.shape[0]
    n_dofs = 2 * n_nodes
    n_polys = 6

    row_idx = []
    col_idx = []
    val_data = []

    for el_id in range(len(elements)):
        vert_ids = elements[el_id].astype(int)
        verts = vertices[vert_ids]
        n_v = len(vert_ids)
        n_el_dofs = 2 * n_v

        E_el = E_per_elem[el_id]

        # Constitutive matrix
        if stress_type == "plane_stress":
            factor = E_el / (1.0 - nu**2)
            C = factor * np.array(
                [
                    [1.0, nu, 0.0],
                    [nu, 1.0, 0.0],
                    [0.0, 0.0, (1.0 - nu) / 2.0],
                ]
            )
        else:
            factor = E_el / ((1.0 + nu) * (1.0 - 2.0 * nu))
            C = factor * np.array(
                [
                    [1.0 - nu, nu, 0.0],
                    [nu, 1.0 - nu, 0.0],
                    [0.0, 0.0, (1.0 - 2.0 * nu) / 2.0],
                ]
            )

        # Geometry
        area_comp = verts[:, 0] * np.roll(verts[:, 1], -1) - np.roll(verts[:, 0], -1) * verts[:, 1]
        area = 0.5 * abs(np.sum(area_comp))
        centroid = np.sum((np.roll(verts, -1, axis=0) + verts) * area_comp[:, None], axis=0) / (
            6.0 * area
        )
        h = max(np.linalg.norm(verts[i] - verts[j]) for i in range(n_v) for j in range(i + 1, n_v))

        xc, yc = centroid

        # D matrix (n_el_dofs × 6)
        D = np.zeros((n_el_dofs, n_polys))
        for i in range(n_v):
            dx = (verts[i, 0] - xc) / h
            dy = (verts[i, 1] - yc) / h
            D[2 * i, :] = [1.0, 0.0, -dy, dx, 0.0, dy]
            D[2 * i + 1, :] = [0.0, 1.0, dx, 0.0, dy, dx]

        # B matrix (6 × n_el_dofs)
        B = np.zeros((n_polys, n_el_dofs))

        vertex_normals = np.zeros((n_v, 2))
        for i in range(n_v):
            prev_v = verts[(i - 1) % n_v]
            next_v = verts[(i + 1) % n_v]
            vertex_normals[i] = [next_v[1] - prev_v[1], prev_v[0] - next_v[0]]

        for i in range(n_v):
            B[0, 2 * i] = 1.0 / n_v
            B[1, 2 * i + 1] = 1.0 / n_v

        for i in range(n_v):
            B[2, 2 * i] = -vertex_normals[i, 1] / (4.0 * area)
            B[2, 2 * i + 1] = vertex_normals[i, 0] / (4.0 * area)

        strain_basis = np.array(
            [
                [1.0 / h, 0.0, 0.0],
                [0.0, 1.0 / h, 0.0],
                [0.0, 0.0, 2.0 / h],
            ]
        )

        for i in range(n_v):
            vn = vertex_normals[i]
            for alpha in range(3):
                sigma = C @ strain_basis[alpha]
                tx = sigma[0] * vn[0] + sigma[2] * vn[1]
                ty = sigma[2] * vn[0] + sigma[1] * vn[1]
                B[3 + alpha, 2 * i] += 0.5 * tx
                B[3 + alpha, 2 * i + 1] += 0.5 * ty

        # Projector
        G = B @ D
        projector = np.linalg.solve(G, B)

        G_tilde = G.copy()
        G_tilde[:3, :] = 0.0

        K_cons = projector.T @ G_tilde @ projector

        # Stabilization
        I_minus_PiD = np.eye(n_el_dofs) - D @ projector
        trace_k = np.trace(K_cons)
        stab_param = stabilization_alpha * trace_k / n_el_dofs if trace_k > 0 else E_el
        K_stab = stab_param * (I_minus_PiD.T @ I_minus_PiD)

        K_local = K_cons + K_stab

        # Assemble
        gdofs = np.zeros(n_el_dofs, dtype=int)
        for i in range(n_v):
            gdofs[2 * i] = 2 * vert_ids[i]
            gdofs[2 * i + 1] = 2 * vert_ids[i] + 1

        ii, jj = np.meshgrid(gdofs, gdofs, indexing="ij")
        row_idx.append(ii.ravel())
        col_idx.append(jj.ravel())
        val_data.append(K_local.ravel())

    row_idx = np.concatenate(row_idx)
    col_idx = np.concatenate(col_idx)
    val_data = np.concatenate(val_data)
    K_global = sp.csr_matrix((val_data, (row_idx, col_idx)), shape=(n_dofs, n_dofs))

    # Apply BCs and solve
    u = np.zeros(n_dofs)
    F = F_load.copy()

    if len(bc_fixed_dofs) > 0:
        bc_set = set(bc_fixed_dofs.tolist())
        internal = np.array([i for i in range(n_dofs) if i not in bc_set])

        u[bc_fixed_dofs] = bc_vals
        F -= K_global[:, bc_fixed_dofs].toarray() @ bc_vals

        K_ii = K_global[np.ix_(internal, internal)]
        u[internal] = spsolve(K_ii, F[internal])
    else:
        u = spsolve(K_global, F)

    return u


# ── CLI demo ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.collections import PatchCollection

    print("=== VEM solver demo ===")

    # Simple test: uniform E, uniform eigenstrain
    Nx, Ny = 15, 15
    Lx, Ly = 200e-6, 100e-6  # 200×100 μm biofilm

    E_field = 500.0 * np.ones((Nx, Ny))
    eps_growth = 0.02 * np.ones((Nx, Ny))
    nu = 0.30

    for mesh_type in ["voronoi", "grid"]:
        result = solve_2d_vem(
            E_field,
            nu,
            eps_growth,
            Nx,
            Ny,
            Lx,
            Ly,
            bc_type="bottom_fixed",
            stress_type="plane_stress",
            mesh_type=mesh_type,
        )

        u_mag = np.sqrt(result["u"][:, 0] ** 2 + result["u"][:, 1] ** 2)
        print(
            f"\n  [{mesh_type}] n_elem={len(result['vem_elements'])}, "
            f"u_max={u_mag.max():.3e} m, "
            f"σ_vm_max={result['sigma_vm'].max():.1f} Pa, "
            f"geom_nonlin={result['geom_nonlin_ratio']:.4f}"
        )

    # Heterogeneous E test (biofilm-like)
    E_field_hetero = np.zeros((Nx, Ny))
    for i in range(Nx):
        for j in range(Ny):
            # Gradient: stiffer at bottom (tooth), softer at top
            E_field_hetero[i, j] = 900.0 - 800.0 * (j / (Ny - 1))

    result_hetero = solve_2d_vem(
        E_field_hetero,
        nu,
        eps_growth,
        Nx,
        Ny,
        Lx,
        Ly,
        mesh_type="voronoi",
        stress_type="plane_stress",
    )

    # Plot VEM mesh + σ_vm
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, (res, title) in enumerate(
        [
            (result, "Uniform E=500 Pa"),
            (result_hetero, "Gradient E=100-900 Pa"),
        ]
    ):
        ax = axes[ax_idx]
        patches = []
        colors = []
        verts = res["vem_vertices"]
        for el_id, el in enumerate(res["vem_elements"]):
            poly = MplPolygon(verts[el.astype(int)], closed=True)
            patches.append(poly)
            colors.append(res["sigma_vm"][el_id])

        pc = PatchCollection(patches, cmap="hot")
        pc.set_array(np.array(colors))
        ax.add_collection(pc)
        ax.set_xlim(-Lx * 0.05, Lx * 1.05)
        ax.set_ylim(-Ly * 0.05, Ly * 1.05)
        ax.set_aspect("equal")
        ax.set_title(f"VEM σ_vm [Pa] — {title}")
        plt.colorbar(pc, ax=ax, label="σ_vm [Pa]")

    outpath = _HERE / "vem_solver_demo.png"
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    print(f"\n  Saved: {outpath}")

    # Comparison: FEM vs VEM
    from solve_stress_2d import solve_2d_fem

    print("\n=== FEM vs VEM comparison ===")
    for label, E_f in [("uniform", E_field), ("gradient", E_field_hetero)]:
        fem_res = solve_2d_fem(E_f, nu, eps_growth, Nx, Ny, Lx, Ly, stress_type="plane_stress")
        vem_res = solve_2d_vem(
            E_f, nu, eps_growth, Nx, Ny, Lx, Ly, stress_type="plane_stress", mesh_type="grid"
        )

        u_fem = np.sqrt(fem_res["u"][:, 0] ** 2 + fem_res["u"][:, 1] ** 2).max()
        u_vem = np.sqrt(vem_res["u"][:, 0] ** 2 + vem_res["u"][:, 1] ** 2).max()
        svm_fem = fem_res["sigma_vm"].max()
        svm_vem = vem_res["sigma_vm"].max()
        print(f"  [{label}] u_max: FEM={u_fem:.3e}, VEM={u_vem:.3e} " f"(ratio={u_vem/u_fem:.3f})")
        print(
            f"  [{label}] σ_vm_max: FEM={svm_fem:.1f}, VEM={svm_vem:.1f} "
            f"(ratio={svm_vem/svm_fem:.3f})"
        )
