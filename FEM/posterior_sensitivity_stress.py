#!/usr/bin/env python3
"""
posterior_sensitivity_stress.py
================================
For each condition with a completed posterior Abaqus ensemble, plot scatter of
each posterior parameter component theta_i vs von Mises stress (substrate and
surface) and compute Spearman rank correlation.

Reads from:
  _posterior_abaqus/{cond}/sample_{k:04d}/theta.npy
  _posterior_abaqus/{cond}/sample_{k:04d}/stress.json

Outputs (per condition + combined):
  _posterior_abaqus/{cond}/sensitivity_stress.png
  _posterior_abaqus/sensitivity_stress_combined.png
  _posterior_abaqus/{cond}/sensitivity_spearman.json

Usage:
  python posterior_sensitivity_stress.py
  python posterior_sensitivity_stress.py --conditions dh_baseline commensal_static
"""
import argparse
import json
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_OUT_BASE = _HERE / "_posterior_abaqus"

# Hamilton 5-species model parameter layout (20-dim theta)
# Species: 1=S.oralis, 2=A.naeslundii, 3=Veillonella, 4=F.nucleatum, 5=P.gingivalis
# aij: interaction coeff (col j inhibits/promotes row i), bi: net growth/death rate
_PARAM_NAMES_20 = [
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
_PARAM_TEX_20 = [
    r"$a_{11}$",
    r"$a_{12}$",
    r"$a_{22}$",
    r"$b_1$",
    r"$b_2$",
    r"$a_{33}$",
    r"$a_{34}$",
    r"$a_{44}$",
    r"$b_3$",
    r"$b_4$",
    r"$a_{13}$",
    r"$a_{14}$",
    r"$a_{23}$",
    r"$a_{24}$",
    r"$a_{55}$",
    r"$b_5$",
    r"$a_{15}$",
    r"$a_{25}$",
    r"$a_{35}$",
    r"$a_{45}$",
]
_PARAM_DESC_20 = [
    "S.or self-inh.",
    "S.or–A.na",
    "A.na self-inh.",
    "S.or growth",
    "A.na growth",
    "Vei self-inh.",
    "Vei–F.nu",
    "F.nu self-inh.",
    "Vei growth",
    "F.nu growth",
    "S.or–Vei",
    "S.or–F.nu",
    "A.na–Vei",
    "A.na–F.nu",
    "Pg self-inh.",
    "Pg growth",
    "S.or–Pg",
    "A.na–Pg",
    "Vei–Pg",
    "F.nu–Pg",
]


def _theta_labels(n_params: int, mode: str = "tex") -> list:
    """Return parameter labels (tex or short names)."""
    if n_params == 20:
        return _PARAM_TEX_20 if mode == "tex" else _PARAM_NAMES_20
    return [r"$\theta_{%d}$" % (i + 1) for i in range(n_params)]


def _theta_desc(n_params: int) -> list:
    if n_params == 20:
        return _PARAM_DESC_20
    return ["param %d" % (i + 1) for i in range(n_params)]


COND_LABEL = {
    "dh_baseline": "DH Baseline",
    "commensal_static": "Commensal Static",
    "commensal_hobic": "Commensal HOBIC",
    "dysbiotic_static": "Dysbiotic Static",
}
COND_COLOR = {
    "dh_baseline": "#d62728",
    "commensal_static": "#1f77b4",
    "commensal_hobic": "#2ca02c",
    "dysbiotic_static": "#ff7f0e",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_condition(cond_dir: Path) -> dict | None:
    """
    Load all completed samples for one condition.
    Returns dict with keys: theta (N,P), substrate (N,), surface (N,), n_valid.
    Returns None if no samples found.
    """
    thetas, subs, surfs = [], [], []
    for sample_dir in sorted(cond_dir.glob("sample_*")):
        theta_path = sample_dir / "theta.npy"
        stress_path = sample_dir / "stress.json"
        done_flag = sample_dir / "done.flag"
        if not (theta_path.exists() and stress_path.exists() and done_flag.exists()):
            continue
        theta = np.load(theta_path)
        with stress_path.open() as f:
            s = json.load(f)
        thetas.append(theta)
        subs.append(s["substrate_smises"])
        surfs.append(s["surface_smises"])
    if not thetas:
        return None
    return {
        "theta": np.array(thetas),  # (N, P)
        "substrate": np.array(subs),  # (N,)
        "surface": np.array(surfs),  # (N,)
        "n_valid": len(thetas),
    }


# ---------------------------------------------------------------------------
# Spearman correlation
# ---------------------------------------------------------------------------


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation coefficient."""
    n = len(x)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    d2 = ((rx - ry) ** 2).sum()
    return 1.0 - 6.0 * d2 / (n * (n * n - 1))


# ---------------------------------------------------------------------------
# Per-condition plot
# ---------------------------------------------------------------------------


def plot_condition(data: dict, cond: str, out_path: Path) -> dict:
    """
    Scatter plot grid: rows = theta_i (top 10 by |Spearman|), cols = substrate/surface.
    Returns Spearman coefficients dict.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    theta = data["theta"]  # (N, P)
    sub = data["substrate"]
    surf = data["surface"]
    n, p = theta.shape
    color = COND_COLOR.get(cond, "#888888")
    label = COND_LABEL.get(cond, cond)
    tlabels = _theta_labels(p, mode="tex")
    tnames = _theta_labels(p, mode="short")
    tdesc = _theta_desc(p)

    # compute Spearman for each parameter × depth
    rho_sub = np.array([spearman(theta[:, i], sub) for i in range(p)])
    rho_surf = np.array([spearman(theta[:, i], surf) for i in range(p)])
    rho_max = np.maximum(np.abs(rho_sub), np.abs(rho_surf))

    # pick top N_SHOW by max |rho|
    N_SHOW = min(p, 12)
    top_idx = np.argsort(rho_max)[::-1][:N_SHOW]

    ncols = 2
    nrows = N_SHOW
    fig, axes = plt.subplots(nrows, ncols, figsize=(6, 1.8 * nrows), squeeze=False)

    for row, pi in enumerate(top_idx):
        for col, (y_arr, y_label, rho) in enumerate(
            [
                (sub, "Substrate $\\sigma_{vM}$ [Pa]", rho_sub[pi]),
                (surf, "Surface $\\sigma_{vM}$ [Pa]", rho_surf[pi]),
            ]
        ):
            ax = axes[row, col]
            ax.scatter(theta[:, pi], y_arr, c=color, alpha=0.7, s=30, linewidths=0)
            c0, c1 = np.polyfit(theta[:, pi], y_arr, 1)
            xr = np.linspace(theta[:, pi].min(), theta[:, pi].max(), 50)
            ax.plot(xr, c0 * xr + c1, "k--", linewidth=0.8, alpha=0.6)
            ax.set_xlabel("%s  (%s)" % (tlabels[pi], tdesc[pi]), fontsize=7)
            ax.set_ylabel(y_label if col == 0 else "", fontsize=7)
            ax.set_title(r"$\rho_S=%.3f$" % rho, fontsize=8)
            ax.tick_params(labelsize=7)

    fig.suptitle("%s  –  parameter sensitivity  (N=%d)" % (label, n), fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  Saved →", out_path)

    result = {
        "n_valid": n,
        "param_names": tnames,
        "param_desc": tdesc,
        "spearman_substrate": {tnames[i]: float(rho_sub[i]) for i in range(p)},
        "spearman_surface": {tnames[i]: float(rho_surf[i]) for i in range(p)},
        "top_params_by_max_rho": [tnames[i] for i in top_idx],
    }
    return result


# ---------------------------------------------------------------------------
# Combined bar chart: top parameters across all conditions
# ---------------------------------------------------------------------------


def plot_combined(all_data: dict, out_path: Path) -> None:
    """
    For each condition overlay: bar chart of |Spearman| per parameter (substrate).
    Shows the top 10 parameters by mean |rho| across all conditions.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not all_data:
        return

    # Collect rho arrays: cond → (P,) for substrate and surface
    conds = list(all_data.keys())

    rho_sub_all = {}
    rho_surf_all = {}
    for cond, data in all_data.items():
        theta = data["theta"]
        sub = data["substrate"]
        surf = data["surface"]
        p = theta.shape[1]
        rho_sub_all[cond] = np.array([spearman(theta[:, i], sub) for i in range(p)])
        rho_surf_all[cond] = np.array([spearman(theta[:, i], surf) for i in range(p)])

    p_ref = next(iter(all_data.values()))["theta"].shape[1]
    tlabels = _theta_labels(p_ref, mode="tex")
    tdesc = _theta_desc(p_ref)

    # top 10 by mean |rho_sub| across conditions
    mean_rho = np.mean(np.abs(np.stack([rho_sub_all[c] for c in conds], axis=0)), axis=0)
    top10 = np.argsort(mean_rho)[::-1][:10]

    x = np.arange(10)
    w = 0.8 / len(conds)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    for ax, (rho_dict, depth_name) in zip(
        axes,
        [(rho_sub_all, "Substrate"), (rho_surf_all, "Surface")],
    ):
        for ci, cond in enumerate(conds):
            rho = rho_dict[cond][top10]
            offs = (ci - (len(conds) - 1) / 2.0) * w
            ax.bar(
                x + offs,
                rho,
                width=w * 0.9,
                color=COND_COLOR.get(cond, "#888888"),
                alpha=0.8,
                label=COND_LABEL.get(cond, cond),
            )
        ax.axhline(0, color="k", linewidth=0.6)
        ax.set_xticks(x)
        xlabels = ["%s\n(%s)" % (tlabels[i], tdesc[i]) for i in top10]
        ax.set_xticklabels(xlabels, fontsize=8, rotation=20, ha="right")
        ax.set_ylabel(r"Spearman $\rho_S$")
        ax.set_title("%s $\\sigma_{vM}$ sensitivity" % depth_name)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Parameter sensitivity: Spearman correlation with von Mises stress\n"
        "(top 10 parameters by mean |ρ| at substrate)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Combined plot →", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description="Posterior Abaqus sensitivity scatter")
    ap.add_argument(
        "--conditions",
        nargs="+",
        default=["dh_baseline", "commensal_static", "commensal_hobic", "dysbiotic_static"],
    )
    ap.add_argument("--out-dir", type=Path, default=_OUT_BASE)
    args = ap.parse_args()

    all_data = {}
    for cond in args.conditions:
        cond_dir = args.out_dir / cond
        if not cond_dir.exists():
            print("[skip] %s – directory not found" % cond)
            continue
        print("\n[%s]" % cond)
        data = load_condition(cond_dir)
        if data is None:
            print("  no completed samples yet")
            continue
        print("  loaded %d samples, theta shape %s" % (data["n_valid"], data["theta"].shape))
        all_data[cond] = data

        # per-condition scatter
        out_png = cond_dir / "sensitivity_stress.png"
        spearman_res = plot_condition(data, cond, out_png)

        out_json = cond_dir / "sensitivity_spearman.json"
        with out_json.open("w") as f:
            json.dump(spearman_res, f, indent=2)
        print("  Spearman JSON →", out_json)

        # print top 5
        rho_sub = spearman_res["spearman_substrate"]
        top5 = sorted(rho_sub.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
        print("  Top-5 substrate |ρ|:", [(k, "%.3f" % v) for k, v in top5])

    if len(all_data) >= 2:
        plot_combined(all_data, args.out_dir / "sensitivity_stress_combined.png")

    if not all_data:
        print("\nNo data found – run the ensemble first.")


if __name__ == "__main__":
    main()
