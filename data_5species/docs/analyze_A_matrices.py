import json
import os
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path


BASE = "/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/_runs"

RUNS = {
    "Commensal_HOBIC": os.path.join(BASE, "Commensal_HOBIC_20260208_002100", "theta_MAP.json"),
    "Dysbiotic_HOBIC": os.path.join(BASE, "Dysbiotic_HOBIC_20260208_002100", "theta_MAP.json"),
    "Commensal_Static": os.path.join(BASE, "Commensal_Static_20260208_002100", "theta_MAP.json"),
    "Dysbiotic_Static": os.path.join(BASE, "Dysbiotic_Static_20260207_203752", "theta_MAP.json"),
}


def theta_to_A(theta):
    theta = np.asarray(theta).flatten()
    if theta.shape[0] != 20:
        raise ValueError(f"theta length must be 20, got {theta.shape[0]}")

    A = np.zeros((5, 5), dtype=float)

    A[0, 0] = theta[0]
    A[0, 1] = A[1, 0] = theta[1]
    A[1, 1] = theta[2]

    A[2, 2] = theta[5]
    A[2, 3] = A[3, 2] = theta[6]
    A[3, 3] = theta[7]

    A[0, 2] = A[2, 0] = theta[10]
    A[0, 3] = A[3, 0] = theta[11]
    A[1, 2] = A[2, 1] = theta[12]
    A[1, 3] = A[3, 1] = theta[13]

    A[4, 4] = theta[14]

    A[0, 4] = A[4, 0] = theta[16]
    A[1, 4] = A[4, 1] = theta[17]
    A[2, 4] = A[4, 2] = theta[18]
    A[3, 4] = A[4, 3] = theta[19]

    return A


def flatten(A):
    return A.reshape(-1)


def main():
    A_mats = {}

    for cond, path in RUNS.items():
        if not os.path.exists(path):
            print(f"[WARN] {cond}: file not found -> {path}")
            continue

        with open(path, "r") as f:
            theta_map = json.load(f)

        if isinstance(theta_map, list):
            theta_vec = theta_map
        elif isinstance(theta_map, dict):
            if "theta" in theta_map:
                theta_vec = theta_map["theta"]
            elif "theta_full" in theta_map:
                theta_vec = theta_map["theta_full"]
            else:
                raise ValueError(
                    f"Unknown theta_MAP.json format for {cond}: keys={list(theta_map.keys())}"
                )
        else:
            raise ValueError(f"Unknown theta_MAP.json format for {cond}: type={type(theta_map)}")

        A_mats[cond] = theta_to_A(theta_vec)

    if not A_mats:
        print("No A matrices loaded. Check RUNS paths.")
        return

    print("Loaded A matrices for conditions:")
    for cond in A_mats:
        print(f"  - {cond}")

    conds = list(A_mats.keys())

    print("\nPairwise comparisons of A matrices:")
    print("cond1 vs cond2 : corr(A1, A2),  " "rel_F_diff = ||A1-A2||_F / max(||A1||_F, ||A2||_F)")

    n_cond = len(conds)
    corr_mat = np.eye(n_cond)
    rel_mat = np.zeros((n_cond, n_cond))
    pair_labels = []
    corr_values = []
    rel_values = []

    for c1, c2 in combinations(conds, 2):
        A1 = A_mats[c1]
        A2 = A_mats[c2]

        v1 = flatten(A1)
        v2 = flatten(A2)

        corr = np.corrcoef(v1, v2)[0, 1]

        diff_norm = np.linalg.norm(A1 - A2, ord="fro")
        base_norm = max(np.linalg.norm(A1, ord="fro"), np.linalg.norm(A2, ord="fro"))
        rel_diff = diff_norm / base_norm if base_norm > 0 else np.nan

        print(f"{c1:17s} vs {c2:17s} :  corr = {corr:+.3f},  rel_F_diff = {rel_diff:.3f}")

        i = conds.index(c1)
        j = conds.index(c2)
        corr_mat[i, j] = corr
        corr_mat[j, i] = corr
        rel_mat[i, j] = rel_diff
        rel_mat[j, i] = rel_diff

        pair_labels.append(f"{c1} vs {c2}")
        corr_values.append(corr)
        rel_values.append(rel_diff)

    print("\nPer-condition A summary:")
    print("cond             : ||A||_F,  A[2,4]=A[4,2] (Vei-P.g)")

    for cond, A in A_mats.items():
        fnorm = np.linalg.norm(A, ord="fro")
        vei_pg = A[2, 4]
        print(f"{cond:17s} : {fnorm:.3f},  Vei-Pg={vei_pg:+.3f}")

    out_dir = Path(__file__).parent

    sns.set_context("paper", font_scale=1.4)
    sns.set_style("whitegrid")

    fig_corr, ax_corr = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        corr_mat,
        vmin=-1.0,
        vmax=1.0,
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        xticklabels=conds,
        yticklabels=conds,
        ax=ax_corr,
    )
    ax_corr.set_title("Correlation of A matrices")
    fig_corr.tight_layout()
    fig_corr.savefig(out_dir / "A_matrix_correlation_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig_corr)

    fig_rel, ax_rel = plt.subplots(figsize=(6, 5))
    vmax_rel = float(np.nanmax(rel_mat)) if np.isfinite(rel_mat).any() else 1.0
    sns.heatmap(
        rel_mat,
        vmin=0.0,
        vmax=vmax_rel,
        cmap="viridis",
        annot=True,
        fmt=".2f",
        xticklabels=conds,
        yticklabels=conds,
        ax=ax_rel,
    )
    ax_rel.set_title("Relative Frobenius difference of A matrices")
    fig_rel.tight_layout()
    fig_rel.savefig(out_dir / "A_matrix_relF_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig_rel)

    if pair_labels:
        idx = np.arange(len(pair_labels))
        fig_bar, (ax_b1, ax_b2) = plt.subplots(1, 2, figsize=(14, 4), sharey=True)

        ax_b1.barh(idx, corr_values, color="tab:blue")
        ax_b1.set_xlabel("Correlation")
        ax_b1.set_yticks(idx)
        ax_b1.set_yticklabels(pair_labels)
        ax_b1.set_title("Pairwise correlation of A")
        ax_b1.set_xlim(-1.0, 1.0)

        ax_b2.barh(idx, rel_values, color="tab:orange")
        ax_b2.set_xlabel("Relative Frobenius difference")
        ax_b2.set_title("Pairwise ΔF of A")
        ax_b2.set_xlim(0.0, max(rel_values) * 1.1 if rel_values else 1.0)

        fig_bar.tight_layout()
        fig_bar.savefig(
            out_dir / "A_matrix_pairwise_corr_relF_bar.png", dpi=300, bbox_inches="tight"
        )
        plt.close(fig_bar)


if __name__ == "__main__":
    main()
