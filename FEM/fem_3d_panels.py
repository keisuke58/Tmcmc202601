#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def make_pg_overview_panel(out_path: Path):
    fem_dir = Path(__file__).resolve().parent
    base = fem_dir / "_results_3d"
    cfgs = [
        ("Commensal_Static",   "Commensal Static"),
        ("Dysbiotic_Static",   "Dysbiotic Static"),
        ("Commensal_HOBIC",    "Commensal HOBIC"),
        ("Dysbiotic_HOBIC",    "Dysbiotic HOBIC"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    axes = axes.flatten()

    for ax, (name, title) in zip(axes, cfgs):
        img_path = base / name / "fig6_overview_pg_3d.png"
        if not img_path.exists():
            ax.text(0.5, 0.5, f"Missing:\n{name}", ha="center", va="center")
            ax.axis("off")
            continue
        img = plt.imread(img_path)
        ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    fig.suptitle("3D P. gingivalis overview – 4 conditions", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Make publication panels for 3D FEM.")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("_results_3d"),
        help="Base directory for output panels",
    )
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = args.out_dir
    panel_pg = out_dir / "panel_pg_overview_4conditions.png"
    make_pg_overview_panel(panel_pg)
    print(f"Saved panel: {panel_pg}")


if __name__ == "__main__":
    main()

