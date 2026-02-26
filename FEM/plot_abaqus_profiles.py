from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_summary(base_dir: Path) -> pd.DataFrame:
    path = base_dir / "abaqus_depth_summary.csv"
    if not path.exists():
        raise SystemExit(f"summary not found: {path}")
    df = pd.read_csv(path)
    return df


def load_ratio(base_dir: Path) -> pd.DataFrame:
    path = base_dir / "abaqus_depth_3d_over_2d.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def plot_smises_depth(df: pd.DataFrame, out_path: Path) -> None:
    sns.set_context("paper", font_scale=1.4)
    sns.set_style("whitegrid")
    plt.figure(figsize=(6, 4))

    markers = {"2D": "o", "3D": "s"}
    linestyles = {"2D": "-", "3D": "--"}
    palette = {"dh": "#1f77b4", "cs": "#d62728"}

    for cond in sorted(df["cond"].unique()):
        sub = df[df["cond"] == cond]
        for dim in sorted(sub["dim"].unique()):
            dd = sub[sub["dim"] == dim]
            if dd.empty:
                continue
            color = palette.get(cond, "k")
            label = f"{cond}-{dim}"
            plt.plot(
                dd["depth_frac"],
                dd["S_Mises_mean"],
                marker=markers.get(dim, "o"),
                linestyle=linestyles.get(dim, "-"),
                color=color,
                label=label,
            )

    plt.xlabel("Normalized depth (0: substrate, 1: surface)")
    plt.ylabel("von Mises stress [Pa]")
    plt.title("Depth profile of von Mises stress (Abaqus)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_s11_depth(df: pd.DataFrame, out_path: Path) -> None:
    sns.set_context("paper", font_scale=1.4)
    sns.set_style("whitegrid")
    plt.figure(figsize=(6, 4))

    markers = {"2D": "o", "3D": "s"}
    linestyles = {"2D": "-", "3D": "--"}
    palette = {"dh": "#1f77b4", "cs": "#d62728"}

    for cond in sorted(df["cond"].unique()):
        sub = df[df["cond"] == cond]
        for dim in sorted(sub["dim"].unique()):
            dd = sub[sub["dim"] == dim]
            if dd.empty:
                continue
            color = palette.get(cond, "k")
            label = f"{cond}-{dim}"
            plt.plot(
                dd["depth_frac"],
                dd["S11_mean"],
                marker=markers.get(dim, "o"),
                linestyle=linestyles.get(dim, "-"),
                color=color,
                label=label,
            )

    plt.xlabel("Normalized depth (0: substrate, 1: surface)")
    plt.ylabel(r"$\sigma_{11}$ [Pa]")
    plt.title(r"Depth profile of $\sigma_{11}$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_ratio_smises(df_ratio: pd.DataFrame, out_path: Path) -> None:
    if df_ratio.empty:
        return
    sns.set_context("paper", font_scale=1.4)
    sns.set_style("whitegrid")
    plt.figure(figsize=(6, 4))

    palette = {"dh": "#1f77b4", "cs": "#d62728"}

    for cond in sorted(df_ratio["cond"].unique()):
        sub = df_ratio[df_ratio["cond"] == cond]
        if sub.empty:
            continue
        color = palette.get(cond, "k")
        plt.plot(
            sub["depth_frac"],
            sub["S_Mises_ratio_3D_over_2D"],
            marker="o",
            linestyle="-",
            color=color,
            label=cond,
        )

    plt.axhline(1.0, color="k", linestyle="--", linewidth=1)
    plt.xlabel("Normalized depth (0: substrate, 1: surface)")
    plt.ylabel("3D / 2D ratio (von Mises)")
    plt.title("3D/2D ratio of von Mises stress")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main(argv: list[str]) -> int:
    base_dir = Path(".")
    summary = load_summary(base_dir)
    ratio = load_ratio(base_dir)

    plot_smises_depth(summary, base_dir / "abaqus_profile_Smises.png")
    plot_s11_depth(summary, base_dir / "abaqus_profile_S11.png")
    plot_ratio_smises(ratio, base_dir / "abaqus_profile_Smises_ratio.png")

    print("Wrote PNGs: ")
    print("  abaqus_profile_Smises.png")
    print("  abaqus_profile_S11.png")
    if not ratio.empty:
        print("  abaqus_profile_Smises_ratio.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
