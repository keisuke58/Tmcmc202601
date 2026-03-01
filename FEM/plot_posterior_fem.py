#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

COLORS = [
    "#1f77b4",
    "#2ca02c",
    "#bcbd22",
    "#9467bd",
    "#d62728",
]

SPECIES_NAMES = [
    "S. oralis",
    "A. naeslundii",
    "V. dispar",
    "F. nucleatum",
    "P. gingivalis",
]


def plot_phibar(posterior_dir: Path, condition: str, out_dir: Path):
    t = np.load(posterior_dir / "t_snap.npy")
    p05 = np.load(posterior_dir / "phibar_p05.npy")
    p50 = np.load(posterior_dir / "phibar_p50.npy")
    p95 = np.load(posterior_dir / "phibar_p95.npy")

    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharex=True, sharey=True)
    for i in range(5):
        ax = axes[i]
        c = COLORS[i]
        ax.fill_between(t, p05[:, i], p95[:, i], color=c, alpha=0.25)
        ax.plot(t, p50[:, i], color=c, linewidth=2)
        ax.set_title(SPECIES_NAMES[i])
        ax.set_xlabel("Model time")
        if i == 0:
            ax.set_ylabel("Mean volume fraction")
    fig.suptitle(f"Posterior φ̄(t) 3D FEM ({condition})")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / f"phibar_posterior_{condition}.png", dpi=300)
    plt.close(fig)


def plot_depth(posterior_dir: Path, condition: str, out_dir: Path):
    d05_path = posterior_dir / "depth_pg_p05.npy"
    d50_path = posterior_dir / "depth_pg_p50.npy"
    d95_path = posterior_dir / "depth_pg_p95.npy"
    if not d05_path.exists() or not d50_path.exists() or not d95_path.exists():
        return
    t = np.load(posterior_dir / "t_snap.npy")
    d05 = np.load(d05_path)
    d50 = np.load(d50_path)
    d95 = np.load(d95_path)

    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.fill_between(t, d05, d95, color=COLORS[4], alpha=0.25)
    ax.plot(t, d50, color=COLORS[4], linewidth=2)
    ax.set_xlabel("Model time")
    ax.set_ylabel("Pg depth (mean x)")
    ax.set_title(f"Pg penetration depth ({condition})")
    fig.tight_layout()
    fig.savefig(out_dir / f"depth_pg_posterior_{condition}.png", dpi=300)
    plt.close(fig)


COND_LABELS = {
    "dh_baseline": "DH Baseline",
    "commensal_static": "Commensal Static",
    "Commensal_Static": "Commensal Static",
    "dh": "DH Baseline",
    "commensal": "Commensal Static",
}


def _label(cond: str) -> str:
    return COND_LABELS.get(cond, cond.replace("_", " ").title())


def plot_comparison_depth(
    dirs_conds: list[tuple[Path, str]],
    out_path: Path,
) -> None:
    """Overlay Pg penetration-depth posteriors for multiple conditions."""
    cond_colors = {
        "dh_baseline": "#d62728",
        "commensal_static": "#1f77b4",
        "Commensal_Static": "#1f77b4",
    }
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for posterior_dir, cond in dirs_conds:
        d05_path = posterior_dir / "depth_pg_p05.npy"
        d50_path = posterior_dir / "depth_pg_p50.npy"
        d95_path = posterior_dir / "depth_pg_p95.npy"
        if not (d05_path.exists() and d50_path.exists() and d95_path.exists()):
            print(f"[warn] depth data missing for {cond}, skipping")
            continue
        t = np.load(posterior_dir / "t_snap.npy")
        d05 = np.load(d05_path)
        d50 = np.load(d50_path)
        d95 = np.load(d95_path)
        c = cond_colors.get(cond, COLORS[0])
        ax.fill_between(t, d05, d95, color=c, alpha=0.20)
        ax.plot(t, d50, color=c, linewidth=2.0, label=_label(cond))
    ax.set_xlabel("Model time")
    ax.set_ylabel("P. gingivalis depth (mean x)")
    ax.set_title("Pg penetration depth – condition comparison")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved comparison plot → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")

    # ── single condition ──────────────────────────────────────────────────
    p1 = sub.add_parser("single", help="Plot one condition")
    p1.add_argument("posterior_dir", type=Path)
    p1.add_argument("--condition", required=True)
    p1.add_argument("--out-dir", type=Path, default=Path("_posterior_plots"))

    # ── comparison of multiple conditions ─────────────────────────────────
    p2 = sub.add_parser("compare", help="Overlay multiple conditions on Pg depth")
    p2.add_argument(
        "--dirs",
        nargs="+",
        type=Path,
        required=True,
        help="Posterior directories (one per condition)",
    )
    p2.add_argument(
        "--conds", nargs="+", required=True, help="Condition labels (same order as --dirs)"
    )
    p2.add_argument("--out", type=Path, default=Path("_posterior_plots/comparison_depth_pg.png"))

    # ── legacy positional (backward compat) ───────────────────────────────
    ap.add_argument("posterior_dir_legacy", nargs="?", type=Path)
    ap.add_argument("--condition", default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("_posterior_plots"))

    args = ap.parse_args()

    if args.cmd == "single":
        plot_phibar(args.posterior_dir, args.condition, args.out_dir / args.condition)
        plot_depth(args.posterior_dir, args.condition, args.out_dir / args.condition)
    elif args.cmd == "compare":
        if len(args.dirs) != len(args.conds):
            ap.error("--dirs and --conds must have the same length")
        plot_comparison_depth(list(zip(args.dirs, args.conds)), args.out)
    else:
        # legacy: positional posterior_dir + --condition
        if args.posterior_dir_legacy is None or args.condition is None:
            ap.print_help()
            return
        out_sub = args.out_dir / args.condition
        plot_phibar(args.posterior_dir_legacy, args.condition, out_sub)
        plot_depth(args.posterior_dir_legacy, args.condition, out_sub)


if __name__ == "__main__":
    main()
