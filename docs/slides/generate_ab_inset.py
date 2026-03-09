"""Generate compact A-matrix + b-vector inset figures for each condition."""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")
from pathlib import Path

BASE = Path("/home/nishioka/IKM_Hiwi/Tmcmc202601")

RUNS = {
    "cs": BASE / "data_5species/_runs/commensal_static",
    "ch": BASE / "data_5species/_runs/commensal_hobic",
    "ds": BASE / "data_5species/_runs/dysbiotic_static",
    "dh": BASE / "data_5species/_runs/dh_baseline",
}

COND_NAMES = {
    "cs": "Commensal_Static",
    "ch": "Commensal_HOBIC",
    "ds": "Dysbiotic_Static",
    "dh": "Dysbiotic_HOBIC",
}

SPECIES = ["So", "An", "Vd", "Fn", "Pg"]


def load_locks():
    with open(BASE / "data_5species/model_config/prior_bounds.json") as f:
        pb = json.load(f)
    locks = {}
    for short, full in COND_NAMES.items():
        locks[short] = set(pb["strategies"][full]["locks"])
    return locks


def theta_to_A_b(theta):
    A = np.zeros((5, 5))
    A[0, 0] = theta[0]
    A[1, 1] = theta[2]
    A[2, 2] = theta[5]
    A[3, 3] = theta[7]
    A[4, 4] = theta[14]
    A[0, 1] = A[1, 0] = theta[1]
    A[0, 2] = A[2, 0] = theta[10]
    A[0, 3] = A[3, 0] = theta[11]
    A[0, 4] = A[4, 0] = theta[16]
    A[1, 2] = A[2, 1] = theta[12]
    A[1, 3] = A[3, 1] = theta[13]
    A[1, 4] = A[4, 1] = theta[17]
    A[2, 3] = A[3, 2] = theta[6]
    A[2, 4] = A[4, 2] = theta[18]
    A[3, 4] = A[4, 3] = theta[19]
    b = np.array([theta[3], theta[4], theta[8], theta[9], theta[15]])
    return A, b


# Theta index -> (i,j) in A, or 'b_k'
THETA_TO_IJ = {
    0: (0, 0),
    1: (0, 1),
    2: (1, 1),
    5: (2, 2),
    6: (2, 3),
    7: (3, 3),
    10: (0, 2),
    11: (0, 3),
    12: (1, 2),
    13: (1, 3),
    14: (4, 4),
    16: (0, 4),
    17: (1, 4),
    18: (2, 4),
    19: (3, 4),
}
THETA_TO_B = {3: 0, 4: 1, 8: 2, 9: 3, 15: 4}


def make_inset(cond, theta, locked_set, outpath):
    # Use raw MAP values (no locking applied — old runs estimated all 20)
    A, b = theta_to_A_b(theta)

    vmax_a = max(abs(A).max(), 0.5)

    fig = plt.figure(figsize=(3.6, 2.6), dpi=200)

    # A matrix heatmap
    ax_a = fig.add_axes([0.10, 0.16, 0.52, 0.72])

    im = ax_a.imshow(A, cmap="RdBu_r", vmin=-vmax_a, vmax=vmax_a, aspect="equal")

    # Annotate all values
    for i in range(5):
        for j in range(5):
            val = A[i, j]
            if abs(val) >= 10:
                txt = f"{val:.1f}"
            else:
                txt = f"{val:.2f}"
            brightness = abs(val) / vmax_a if vmax_a > 0 else 0
            color = "white" if brightness > 0.55 else "black"
            fw = "bold" if brightness > 0.4 else "normal"
            ax_a.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                fontsize=5.5 if abs(val) < 10 else 5,
                color=color,
                fontweight=fw,
            )

    ax_a.set_xticks(range(5))
    ax_a.set_yticks(range(5))
    ax_a.set_xticklabels(SPECIES, fontsize=7, fontweight="bold")
    ax_a.set_yticklabels(SPECIES, fontsize=7, fontweight="bold")
    ax_a.set_title("A (MAP)", fontsize=9, fontweight="bold", pad=4)
    ax_a.tick_params(length=0)

    # Colorbar
    cax = fig.add_axes([0.10, 0.06, 0.52, 0.035])
    cb = plt.colorbar(im, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=5.5)

    # b vector bar chart
    ax_b = fig.add_axes([0.72, 0.16, 0.24, 0.72])
    colors_b = ["#1976D2", "#388E3C", "#F57C00", "#7B1FA2", "#D32F2F"]

    for i in range(5):
        ax_b.barh(i, b[i], color=colors_b[i], edgecolor="white", linewidth=0.5, height=0.6)
        ax_b.text(
            b[i] + 0.03 * max(b.max() * 1.1, 0.5),
            i,
            f"{b[i]:.2f}",
            va="center",
            fontsize=5.5,
            color="#333",
        )

    ax_b.set_yticks(range(5))
    ax_b.set_yticklabels(SPECIES, fontsize=7, fontweight="bold")
    ax_b.invert_yaxis()
    ax_b.set_title("b (MAP)", fontsize=9, fontweight="bold", pad=4)
    ax_b.tick_params(labelsize=5.5, length=2)
    ax_b.set_xlim(0, max(b.max() * 1.3, 0.5))

    # Grid lines for b
    ax_b.set_axisbelow(True)
    ax_b.xaxis.grid(True, alpha=0.3, linewidth=0.5)

    fig.savefig(outpath, bbox_inches="tight", facecolor="white", pad_inches=0.03)
    plt.close(fig)
    print(
        f"  {cond.upper()}: saved {outpath.name} | "
        f"free={20-len(locked_set)}, A[{A.min():.2f},{A.max():.2f}], b[{b.min():.2f},{b.max():.2f}]"
    )


def main():
    outdir = BASE / "docs/slides/public/img"
    locks = load_locks()

    for cond, run_dir in RUNS.items():
        with open(run_dir / "theta_MAP.json") as f:
            data = json.load(f)
        theta = np.array(data["theta_full"])
        make_inset(cond, theta, locks[cond], outdir / f"ab_map_{cond}.png")


if __name__ == "__main__":
    main()
