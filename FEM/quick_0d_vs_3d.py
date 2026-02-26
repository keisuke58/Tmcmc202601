#!/usr/bin/env python3
"""quick_0d_vs_3d.py - 0D vs 3D reaction-diffusion comparison."""
import json
import sys
import time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "tmcmc" / "program2602"))
sys.path.insert(0, str(HERE))

from improved_5species_jit import _newton_step_jit_5s as _newton_step_jit, HAS_NUMBA
from numba import njit, prange
from material_models import compute_di

assert HAS_NUMBA

RUNS = ROOT / "data_5species" / "_runs"
CONDITIONS = {
    "commensal_static": RUNS / "commensal_static" / "theta_MAP.json",
    "commensal_hobic": RUNS / "commensal_hobic" / "theta_MAP.json",
    "dh_baseline": RUNS / "dh_baseline" / "theta_MAP.json",
    "dysbiotic_static": RUNS / "dysbiotic_static" / "theta_MAP.json",
}
_D_EFF = np.array([1e-3, 1e-3, 8e-4, 5e-4, 2e-4])


def load_theta(path):
    with open(path) as f:
        return np.array(json.load(f)["theta_full"], dtype=np.float64)


def _theta_to_matrices(theta):
    A = np.zeros((5, 5), dtype=np.float64)
    b = np.zeros(5, dtype=np.float64)
    A[0, 0] = theta[0]
    A[0, 1] = theta[1]
    A[1, 0] = theta[1]
    A[1, 1] = theta[2]
    b[0] = theta[3]
    b[1] = theta[4]
    A[2, 2] = theta[5]
    A[2, 3] = theta[6]
    A[3, 2] = theta[6]
    A[3, 3] = theta[7]
    b[2] = theta[8]
    b[3] = theta[9]
    A[0, 2] = theta[10]
    A[2, 0] = theta[10]
    A[0, 3] = theta[11]
    A[3, 0] = theta[11]
    A[1, 2] = theta[12]
    A[2, 1] = theta[12]
    A[1, 3] = theta[13]
    A[3, 1] = theta[13]
    A[4, 4] = theta[14]
    b[4] = theta[15]
    A[0, 4] = theta[16]
    A[4, 0] = theta[16]
    A[1, 4] = theta[17]
    A[4, 1] = theta[17]
    A[2, 4] = theta[18]
    A[4, 2] = theta[18]
    A[3, 4] = theta[19]
    A[4, 3] = theta[19]
    return A, b


@njit(parallel=True, cache=False)
def _react_3d(G_flat, A, b_diag, n_sub, dt_h, Kp1, Eta, Eta_phi, c_val, alpha, K_hill, n_hill, eps):
    N = G_flat.shape[0]
    G_out = np.empty_like(G_flat)
    for k in prange(N):
        g = G_flat[k].copy()
        for _ in range(n_sub):
            g = _newton_step_jit(
                g, dt_h, Kp1, Eta, Eta_phi, c_val, alpha, A, b_diag, eps, 50, K_hill, n_hill
            )
        G_out[k] = g
    return G_out


def _lap1d(N, h):
    h2 = h * h
    d = np.full(N, -2.0 / h2)
    d[0] = d[-1] = -1.0 / h2
    return sp.diags([np.ones(N - 1) / h2, d, np.ones(N - 1) / h2], [-1, 0, 1], format="csr")


def _build_solvers(Nx, dx, D_eff, dt_mac):
    L = _lap1d(Nx, dx)
    I = sp.eye(Nx, format="csr")
    L3 = (
        sp.kron(sp.kron(L, I, "csr"), I, "csr")
        + sp.kron(sp.kron(I, L, "csr"), I, "csr")
        + sp.kron(sp.kron(I, I, "csr"), L, "csr")
    )
    Isp = sp.eye(Nx**3, format="csr")
    return [spla.factorized((Isp - dt_mac * Di * L3).tocsc()) for Di in D_eff]


def run_0d(theta, n_steps, n_sub, dt_h):
    A, bd = _theta_to_matrices(theta)
    g = np.zeros(12, dtype=np.float64)
    g[0] = 0.13
    g[1] = 0.13
    g[2] = 0.08
    g[3] = 0.05
    g[4] = 0.01
    g[5] = 1.0 - g[:5].sum()
    g[6:11] = g[:5]
    g[11] = 1.0
    E = np.ones(5, dtype=np.float64)
    for _ in range(n_steps):
        for __ in range(n_sub):
            g = _newton_step_jit(g, dt_h, 1e-4, E, E, 100.0, 100.0, A, bd, 1e-6, 50, 0.0, 2.0)
    return g[:5]


def run_3d(theta, Nx, n_macro, n_sub, dt_h):
    A, bd = _theta_to_matrices(theta)
    dx = 1.0 / (Nx - 1)
    dt_mac = dt_h * n_sub
    print("    Assembling 3D Laplacian...", end=" ", flush=True)
    solvers = _build_solvers(Nx, dx, _D_EFF, dt_mac)
    print("done.")
    rng = np.random.default_rng(42)
    G = np.zeros((Nx, Nx, Nx, 12), dtype=np.float64)
    ns = lambda s: s * rng.standard_normal((Nx, Nx, Nx))
    x = np.linspace(0, 1, Nx)
    G[:, :, :, 0] = (0.13 + ns(0.01)).clip(0)
    G[:, :, :, 1] = (0.13 + ns(0.01)).clip(0)
    G[:, :, :, 2] = (0.08 + ns(0.005)).clip(0)
    G[:, :, :, 3] = (0.05 * np.exp(-3 * x)[:, None, None] + ns(0.005)).clip(0)
    xp2 = np.exp(-5 * x)
    yp = np.exp(-0.5 * ((x - 0.5) / 0.1) ** 2)
    G[:, :, :, 4] = (
        0.01 * xp2[:, None, None] * yp[None, :, None] * yp[None, None, :] + ns(0.002)
    ).clip(1e-6)
    G[:, :, :, 5] = (1 - G[:, :, :, :5].sum(3)).clip(0)
    G[:, :, :, 6:11] = G[:, :, :, :5]
    G[:, :, :, 11] = 1.0
    E = np.ones(5, dtype=np.float64)
    NN = Nx**3
    for step in range(1, n_macro + 1):
        Gf = _react_3d(
            G.reshape(NN, 12), A, bd, n_sub, dt_h, 1e-4, E, E, 100.0, 100.0, 0.0, 2.0, 1e-6
        )
        G = Gf.reshape(Nx, Nx, Nx, 12)
        for i, s in enumerate(solvers):
            G[:, :, :, i] = s(G[:, :, :, i].ravel()).clip(0).reshape(Nx, Nx, Nx)
        G[:, :, :, 5] = (1 - G[:, :, :, :5].sum(3)).clip(0)
        if step % 25 == 0:
            pm = G[:, :, :, :5].mean(axis=(0, 1, 2))
            print(f"    [{100*step/n_macro:5.1f}%] phi=[{','.join(f'{v:.3f}' for v in pm)}]")
    return G[:, :, :, :5].transpose(3, 0, 1, 2)


def main():
    NX, NM, NS, DH = 10, 100, 50, 1e-5
    print("=" * 72)
    print(f"  0D vs 3D | {NX}^3={NX**3} nodes | {NM}x{NS} steps")
    print("=" * 72)
    print("  Warming up Numba...", end=" ", flush=True)
    d = np.zeros((1, 12), dtype=np.float64)
    d[0, 5] = 1.0
    _react_3d(
        d,
        np.eye(5),
        np.ones(5),
        1,
        1e-5,
        1e-4,
        np.ones(5),
        np.ones(5),
        100.0,
        100.0,
        0.0,
        2.0,
        1e-6,
    )
    print("done.\n")
    results = {}
    for cond, tp in CONDITIONS.items():
        print(f"{'='*60}\n  {cond}\n{'='*60}")
        theta = load_theta(tp)
        t0 = time.perf_counter()
        p0d = run_0d(theta, NM, NS, DH)
        t0d = time.perf_counter() - t0
        di0d = float(compute_di(p0d))
        print(f"  0D ({t0d:.1f}s) phi=[{','.join(f'{v:.4f}' for v in p0d)}] DI={di0d:.4f}")
        t0 = time.perf_counter()
        p3d = run_3d(theta, NX, NM, NS, DH)
        t3d = time.perf_counter() - t0
        pm = p3d.mean(axis=(1, 2, 3))
        dif = compute_di(p3d.transpose(1, 2, 3, 0))
        dm, dn, dx, ds = float(dif.mean()), float(dif.min()), float(dif.max()), float(dif.std())
        dd = abs(dm - di0d)
        print(f"  3D ({t3d:.1f}s) phi=[{','.join(f'{v:.4f}' for v in pm)}] DI={dm:.4f}")
        print(
            f"  dDI={dd:.4f} ({dd/max(di0d,1e-8)*100:.1f}%) range=[{dn:.4f},{dx:.4f}] std={ds:.4f}"
        )
        print(f"  Pg: 0D={p0d[4]:.4f} 3D={pm[4]:.4f} [{p3d[4].min():.4f},{p3d[4].max():.4f}]\n")
        results[cond] = dict(
            di_0d=di0d,
            di_3d_mean=dm,
            di_3d_min=dn,
            di_3d_max=dx,
            di_3d_std=ds,
            delta_di=dd,
            phi_0d=p0d.tolist(),
            phi_3d_mean=pm.tolist(),
            pg_0d=float(p0d[4]),
            pg_3d_mean=float(pm[4]),
            pg_3d_max=float(p3d[4].max()),
            time_0d=t0d,
            time_3d=t3d,
        )
    print("=" * 72 + "\n  SUMMARY\n" + "=" * 72)
    fmt = f"  {'Cond':<22} {'DI0D':>7} {'DI3D':>7} {'dDI':>7} {'d%':>6} {'range':>10} {'Pg0D':>7} {'Pg3D':>7}"
    print(fmt)
    print("  " + "-" * 72)
    for c, r in results.items():
        p = r["delta_di"] / max(r["di_0d"], 1e-8) * 100
        print(
            f"  {c:<22} {r['di_0d']:7.4f} {r['di_3d_mean']:7.4f} {r['delta_di']:7.4f} "
            f"{p:5.1f}% {r['di_3d_max']-r['di_3d_min']:10.4f} {r['pg_0d']:7.4f} {r['pg_3d_mean']:7.4f}"
        )
    md = max(r["delta_di"] for r in results.values())
    mr = max(r["di_3d_max"] - r["di_3d_min"] for r in results.values())
    print(f"\n  Max dDI={md:.4f} | Max range={mr:.4f}")
    if md > 0.05 or mr > 0.1:
        print("  => Worth adding 3D to paper.")
    else:
        print("  => 0D/Hybrid sufficient. 3D = Future Work.")
    out = HERE / "_results_3d"
    out.mkdir(exist_ok=True)
    with open(out / "quick_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {out/'quick_comparison.json'}")


if __name__ == "__main__":
    main()
