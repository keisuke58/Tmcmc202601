#!/usr/bin/env python3
"""
Plot Hamilton TMCMC posterior A-matrix (MAP + 90% CI) with KEGG/HMDB sign prior
for all 4 Heine conditions. Saves to docs/figures/dieckow/fig_heine_kegg_sign_comparison.*

Sign-agreement is checked vs the net_flow matrix built from Dieckow SF1 + eHOMD.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── paths ───────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent.parent  # IKM_Hiwi/
_RUNS = _HERE.parent / "_runs"
FIG_DIR = _ROOT / "docs" / "figures" / "dieckow"
NIFE_DIR = _ROOT / "nife"

SPECIES = ["So", "An", "Vd", "Fn", "Pg"]
N_SP = 5
SIGMA_KEGG = 0.15

CONDITIONS = [
    ("Commensal", "Static", "CS", "Commensal_Static"),
    ("Commensal", "HOBIC", "CH", "Commensal_HOBIC"),
    ("Dysbiotic", "Static", "DS", "Dysbiotic_Static"),
    ("Dysbiotic", "HOBIC", "DH", "Dysbiotic_HOBIC"),
]

# Run dirs (prefer kegg_fixed, fall back to most-recent matching)
RUN_DIRS = {
    "CS": "Commensal_Static_kegg_fixed",
    "CH": "Commensal_HOBIC_kegg_fixed",
    "DS": "Dysbiotic_Static_kegg_fixed",
    "DH": "Dysbiotic_HOBIC_kegg_fixed",
}

# ── theta → A matrix (Hamilton column-major upper-triangle) ─────────────────


def theta_to_A(theta: np.ndarray) -> np.ndarray:
    """Convert Hamilton 20-dim theta to 5×5 symmetric A matrix."""
    A = np.zeros((N_SP, N_SP))
    idx = 0
    for j in range(N_SP):
        for i in range(j + 1):
            A[i, j] = A[j, i] = theta[idx]
            idx += 1
    return A


# ── KEGG/eHOMD net-flow matrix ───────────────────────────────────────────────


def build_net_flow() -> np.ndarray:
    sf1 = None
    sf1_tail = (
        Path("Szafranski_Published_Work")
        / "Szafranski_Published_Work"
        / "public_data"
        / "Dieckow"
        / "Supplementary_File_1_microbe_metabolite_enzyme_interactions.tsv"
    )
    for p in [NIFE_DIR, _ROOT / "Tmcmc202601" / "nife"]:
        if (p / sf1_tail).exists():
            sf1 = p / sf1_tail
            break
    if sf1 is None:
        print("WARNING: SF1 TSV not found; net_flow = zeros", file=sys.stderr)
        return np.zeros((N_SP, N_SP))

    import pandas as pd

    df = pd.read_csv(sf1, sep="\t")
    genus_sp = {
        "Streptococcus": 0,
        "Schaalia": 0,
        "Actinomyces": 1,
        "Veillonella": 2,
        "Lancefieldella": 2,
        "Selenomonas": 2,
        "Fusobacterium": 3,
        "Leptotrichia": 3,
        "Porphyromonas": 4,
        "Prevotella": 4,
        "Tannerella": 4,
    }
    pos = np.zeros((N_SP, N_SP))
    neg = np.zeros((N_SP, N_SP))
    for met in df["OBJECT"].unique():
        mdf = df[df["OBJECT"] == met]

        def w(r):
            if str(r.get("KEGG", "")) not in ("n/a", "", "nan", "NaN"):
                return 2.0
            return 2.0 if "HMDB" in str(r.get("HMDB_ID", "")) else 1.0

        wt = float(mdf.apply(w, axis=1).max())
        prod, cons, inhib = set(), set(), set()
        for _, row in mdf.iterrows():
            g = genus_sp.get(str(row["TAXON"]).split()[0])
            if g is None:
                continue
            if row["RELATIONSHIP"] == "PRODUCES":
                prod.add(g)
            elif row["RELATIONSHIP"] == "USES":
                cons.add(g)
            elif row["RELATIONSHIP"] == "IS_INHIBITED_BY":
                inhib.add(g)
        for src in prod:
            for tgt in cons:
                if src != tgt:
                    pos[tgt, src] += wt
            for tgt in inhib:
                if src != tgt:
                    neg[tgt, src] += wt
    for i, j in [(0, 1), (1, 3), (2, 3), (3, 4)]:
        pos[j, i] += 1.0
        pos[i, j] += 1.0
    return pos - neg


# ── load posterior and compute A stats ──────────────────────────────────────


def load_posterior(label: str):
    d = _RUNS / RUN_DIRS[label]
    samples = np.load(d / "samples.npy")  # (1000, 20)
    logL = np.load(d / "logL.npy")
    map_theta = samples[np.argmax(logL)]
    A_matrices = np.array([theta_to_A(s) for s in samples])  # (1000, 5, 5)
    A_map = theta_to_A(map_theta)
    A_lo = np.percentile(A_matrices, 5, axis=0)
    A_hi = np.percentile(A_matrices, 95, axis=0)
    return A_map, A_lo, A_hi, A_matrices, float(logL.max())


def sign_agree(A, net_flow, tol=0.02):
    agree = total = 0
    for i in range(N_SP):
        for j in range(N_SP):
            if i == j:
                continue
            f = net_flow[i, j]
            if f == 0 or abs(A[i, j]) < tol:
                continue
            total += 1
            if np.sign(f) == np.sign(A[i, j]):
                agree += 1
    return agree, total


# ── main figure ─────────────────────────────────────────────────────────────


def main():
    net_flow = build_net_flow()
    nz = int((net_flow != 0).sum() - N_SP)
    print(f"net_flow non-zero off-diagonal pairs: {nz}")

    # Load gLV reference if available
    glv_json = NIFE_DIR / "results" / "heine2025" / "fit_glv_heine_kegg_prior.json"
    glv = json.load(open(glv_json)) if glv_json.exists() else {}

    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5))
    vmax = 0.5
    cmap = plt.cm.RdBu_r

    print_lines = []
    for col, (cond_type, flow_type, label, _) in enumerate(CONDITIONS):
        A_map, A_lo, A_hi, A_all, map_logL = load_posterior(label)
        agree, total = sign_agree(A_map, net_flow)
        pct = 100 * agree / total if total > 0 else 0
        print_lines.append(
            f"{label}: MAP logL={map_logL:.3f}  sign_agree={agree}/{total} ({pct:.0f}%)"
        )

        # Row 0: Hamilton MAP A matrix
        ax = axes[0, col]
        im = ax.imshow(A_map, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(N_SP))
        ax.set_xticklabels(SPECIES, fontsize=7)
        ax.set_yticks(range(N_SP))
        ax.set_yticklabels(SPECIES, fontsize=7)
        ax.set_title(f"{label} Hamilton\nlogL={map_logL:.2f}  SA={agree}/{total}", fontsize=7.5)
        for i in range(N_SP):
            for j in range(N_SP):
                v = A_map[i, j]
                lo, hi = A_lo[i, j], A_hi[i, j]
                # Check sign consistency with KEGG
                f = net_flow[i, j]
                if f != 0 and abs(v) > 0.02:
                    ok = np.sign(f) == np.sign(v)
                    edge = "lime" if ok else "red"
                    ax.add_patch(
                        plt.Rectangle(
                            (j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=edge, linewidth=1.5
                        )
                    )
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    color="k" if abs(v) < vmax * 0.7 else "w",
                )

        # Row 1: gLV MAP A matrix (reference)
        ax2 = axes[1, col]
        if label in glv.get("conditions", {}):
            A_glv = np.array(glv["conditions"][label]["A"])
            glv_rmse = glv["conditions"][label].get("rmse", float("nan"))
            glv_agree, glv_total = sign_agree(A_glv, net_flow)
        else:
            A_glv = np.zeros((N_SP, N_SP))
            glv_rmse, glv_agree, glv_total = float("nan"), 0, 0
        im2 = ax2.imshow(A_glv, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        ax2.set_xticks(range(N_SP))
        ax2.set_xticklabels(SPECIES, fontsize=7)
        ax2.set_yticks(range(N_SP))
        ax2.set_yticklabels(SPECIES, fontsize=7)
        sa_glv_str = f"SA={glv_agree}/{glv_total}" if glv_total > 0 else ""
        ax2.set_title(f"{label} gLV KEGG\nRMSE={glv_rmse:.4f}  {sa_glv_str}", fontsize=7.5)
        for i in range(N_SP):
            for j in range(N_SP):
                v = A_glv[i, j]
                f = net_flow[i, j]
                if f != 0 and abs(v) > 0.02:
                    ok = np.sign(f) == np.sign(v)
                    ax2.add_patch(
                        plt.Rectangle(
                            (j - 0.5, i - 0.5),
                            1,
                            1,
                            fill=False,
                            edgecolor="lime" if ok else "red",
                            linewidth=1.5,
                        )
                    )
                ax2.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    color="k" if abs(v) < vmax * 0.7 else "w",
                )

    axes[0, 0].set_ylabel("Hamilton TMCMC MAP", fontsize=8)
    axes[1, 0].set_ylabel("gLV KEGG-prior MAP", fontsize=8)

    plt.colorbar(im, ax=axes, orientation="vertical", fraction=0.02, pad=0.02, label="A[i,j]")
    legend_handles = [
        mpatches.Patch(edgecolor="lime", facecolor="none", label="KEGG sign ✓"),
        mpatches.Patch(edgecolor="red", facecolor="none", label="KEGG sign ✗"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.suptitle(
        "Heine 5-species A-matrix: Hamilton TMCMC (top) vs gLV KEGG-prior (bottom)\n"
        f"Sign prior: sigma={SIGMA_KEGG}, eHOMD supplement (lime=agree, red=disagree)",
        fontsize=9,
    )
    plt.tight_layout(rect=[0, 0.04, 0.97, 1])

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig_heine_kegg_sign_comparison.{ext}", dpi=300, bbox_inches="tight")
    plt.close()

    for line in print_lines:
        print(line)
    print(f"\nFigure saved to {FIG_DIR}/fig_heine_kegg_sign_comparison.*")


if __name__ == "__main__":
    main()
