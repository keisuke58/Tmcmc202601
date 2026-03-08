#!/usr/bin/env python3
"""
Heine 2025 独立バリデーション
============================
TMCMC calibration に使っていないデータ（pH, gingipain, 代謝ネットワーク, OD600）を
使ってモデルの validation を行う。

Usage:
    python validate_heine2025.py
"""

import sys, json, os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import curve_fit

# Paths
ROOT = Path(__file__).parent.parent  # data_5species/
EXPDATA = ROOT / "experiment_data"
RUNS = ROOT / "_runs"
OUTDIR = ROOT / "docs" / "paper_comprehensive_figs"
OUTDIR.mkdir(exist_ok=True)

# ============================================================
# Load experimental data
# ============================================================


def load_species_data():
    """Load species distribution (%) from CSV."""
    import csv

    rows = []
    with open(EXPDATA / "species_distribution_data.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_boxplot_data():
    """Load total volume boxplot data."""
    import csv

    rows = []
    with open(EXPDATA / "biofilm_boxplot_data.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_ph_data():
    """Load pH time series (HOBIC only)."""
    import csv

    times, ph_com, ph_dys = [], [], []
    with open(EXPDATA / "fig4A_pH_timeseries.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row["time_days"]))
            ph_com.append(float(row["pH_commensal"]))
            ph_dys.append(float(row["pH_dysbiotic"]))
    return np.array(times), np.array(ph_com), np.array(ph_dys)


def load_gingipain_data():
    """Load gingipain concentration."""
    import csv

    days, dys_mean, dys_err = [], [], []
    with open(EXPDATA / "fig4B_gingipain_concentration.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            days.append(int(row["day"]))
            dys_mean.append(float(row["dysbiotic_mean"]))
            dys_err.append(float(row["dysbiotic_error"]))
    return np.array(days), np.array(dys_mean), np.array(dys_err)


def load_od600_data():
    """Load OD600 growth curves."""
    import csv

    data = {"Commensal": {}, "Dysbiotic": {}}
    with open(EXPDATA / "fig1B_OD600_growth_curves.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row["model"]
            rep = int(row["replicate"])
            t = float(row["time_hours"])
            od = float(row["OD600"])
            if rep not in data[model]:
                data[model][rep] = {"t": [], "od": []}
            data[model][rep]["t"].append(t)
            data[model][rep]["od"].append(od)
    return data


def load_metabolic_network():
    """Load metabolic interaction network."""
    import csv

    edges = []
    with open(EXPDATA / "fig4C_metabolic_interactions.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            edges.append(row)
    return edges


def get_species_fractions(condition, cultivation, species_data):
    """Get species fractions at each timepoint for a condition."""
    days = [1, 3, 6, 10, 15, 21]
    # Color mapping
    if condition == "Commensal":
        color_map = {"Blue": 0, "Green": 1, "Yellow": 2, "Purple": 3, "Red": 4}
    else:
        color_map = {"Blue": 0, "Green": 1, "Orange": 2, "Purple": 3, "Red": 4}

    fracs = np.zeros((len(days), 5))
    for row in species_data:
        if row["condition"] == condition and row["cultivation"] == cultivation:
            day = int(row["day"])
            species = row["species"]
            median_pct = float(row["median"])
            if day in days and species in color_map:
                idx = days.index(day)
                fracs[idx, color_map[species]] = median_pct / 100.0
    return np.array(days), fracs


def load_map_theta(run_name):
    """Load MAP theta from a run."""
    path = RUNS / run_name / "theta_MAP.json"
    with open(path) as f:
        t = json.load(f)
    return np.array(t.get("theta_full", t.get("theta_sub")))


# ============================================================
# Panel 1: pH Independent Validation
# ============================================================


def validate_ph(ax1, ax2, species_data):
    """
    pH は TMCMC で一切使っていない。
    So → lactate → pH↓, Vd/Vp → lactate消費 → pH↑
    単純な線形モデルで pH(t) を species fractions から予測。
    """
    t_ph, ph_com, ph_dys = load_ph_data()

    # Get species fractions at measurement days (HOBIC only)
    days_ch, fracs_ch = get_species_fractions("Commensal", "HOBIC", species_data)
    days_dh, fracs_dh = get_species_fractions("Dysbiotic", "HOBIC", species_data)

    # pH at matching days
    ph_com_at_days = np.interp(days_ch, t_ph, ph_com)
    ph_dys_at_days = np.interp(days_dh, t_ph, ph_dys)

    # So fraction at matching days
    so_ch = fracs_ch[:, 0]  # S. oralis
    vd_ch = fracs_ch[:, 2]  # V. dispar
    so_dh = fracs_dh[:, 0]
    vd_dh = fracs_dh[:, 2]  # V. parvula

    # Combined data: pH = β0 + β1*So + β2*Vd
    all_so = np.concatenate([so_ch, so_dh])
    all_vd = np.concatenate([vd_ch, vd_dh])
    all_ph = np.concatenate([ph_com_at_days, ph_dys_at_days])

    # Fit linear model
    X = np.column_stack([np.ones(len(all_so)), all_so, all_vd])
    beta, _, _, _ = np.linalg.lstsq(X, all_ph, rcond=None)
    ph_pred = X @ beta

    r_pearson, p_val = pearsonr(all_ph, ph_pred)
    r_spearman, _ = spearmanr(all_ph, ph_pred)
    rmse = np.sqrt(np.mean((all_ph - ph_pred) ** 2))

    # Panel 1a: pH time series + model
    ax1.plot(t_ph, ph_com, "-", color="#2196F3", alpha=0.7, label="Commensal (measured)")
    ax1.plot(t_ph, ph_dys, "-", color="#F44336", alpha=0.7, label="Dysbiotic (measured)")

    # Predicted pH at species measurement days
    ph_pred_ch = beta[0] + beta[1] * so_ch + beta[2] * vd_ch
    ph_pred_dh = beta[0] + beta[1] * so_dh + beta[2] * vd_dh

    ax1.plot(
        days_ch, ph_pred_ch, "s", color="#2196F3", ms=8, zorder=5, label=f"Commensal (predicted)"
    )
    ax1.plot(
        days_dh, ph_pred_dh, "s", color="#F44336", ms=8, zorder=5, label=f"Dysbiotic (predicted)"
    )

    ax1.set_xlabel("Time (days)")
    ax1.set_ylabel("pH")
    ax1.set_title("(a) pH Independent Validation", fontweight="bold")
    ax1.legend(fontsize=7, loc="lower right")
    ax1.set_xlim(-0.5, 22)
    ax1.text(
        0.02,
        0.95,
        f"pH = {beta[0]:.2f} {beta[1]:+.2f}·φ_So {beta[2]:+.2f}·φ_Vd\n"
        f"R² = {r_pearson**2:.3f}, RMSE = {rmse:.3f}",
        transform=ax1.transAxes,
        fontsize=7,
        va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # Panel 1b: Scatter plot
    ax2.scatter(
        ph_com_at_days,
        ph_pred_ch,
        c="#2196F3",
        s=60,
        zorder=5,
        label="Commensal HOBIC",
        edgecolors="k",
        linewidth=0.5,
    )
    ax2.scatter(
        ph_dys_at_days,
        ph_pred_dh,
        c="#F44336",
        s=60,
        zorder=5,
        label="Dysbiotic HOBIC",
        edgecolors="k",
        linewidth=0.5,
    )
    lims = [min(all_ph.min(), ph_pred.min()) - 0.05, max(all_ph.max(), ph_pred.max()) + 0.05]
    ax2.plot(lims, lims, "k--", alpha=0.3, lw=1)
    ax2.set_xlabel("Measured pH")
    ax2.set_ylabel("Predicted pH")
    ax2.set_title("(b) pH: Measured vs Predicted", fontweight="bold")
    ax2.legend(fontsize=7)
    ax2.set_aspect("equal")
    ax2.set_xlim(lims)
    ax2.set_ylim(lims)
    ax2.text(
        0.02,
        0.95,
        f"r = {r_pearson:.3f} (p = {p_val:.1e})",
        transform=ax2.transAxes,
        fontsize=8,
        va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    return beta, r_pearson, rmse


# ============================================================
# Panel 2: Gingipain vs Pg
# ============================================================


def validate_gingipain(ax, species_data):
    """
    Gingipain (virulence factor) ∝ Pg abundance.
    """
    days_gin, gin_mean, gin_err = load_gingipain_data()
    days_dh, fracs_dh = get_species_fractions("Dysbiotic", "HOBIC", species_data)

    # Pg fraction at gingipain measurement days
    pg_at_gin = []
    for d in days_gin:
        if d in list(days_dh):
            idx = list(days_dh).index(d)
            pg_at_gin.append(fracs_dh[idx, 4])  # Pg = index 4
        else:
            pg_at_gin.append(0.0)
    pg_at_gin = np.array(pg_at_gin)

    # Twin axis: gingipain on left, Pg on right
    color_gin = "#9C27B0"
    color_pg = "#F44336"

    ax.errorbar(
        days_gin,
        gin_mean,
        yerr=gin_err,
        fmt="o-",
        color=color_gin,
        capsize=4,
        label="Gingipain (measured)",
        ms=6,
        zorder=5,
    )
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Gingipain concentration", color=color_gin)
    ax.tick_params(axis="y", labelcolor=color_gin)

    ax2 = ax.twinx()
    ax2.bar(
        days_dh,
        fracs_dh[:, 4] * 100,
        width=1.2,
        alpha=0.4,
        color=color_pg,
        label="P. gingivalis (%)",
        zorder=1,
    )
    ax2.set_ylabel("P. gingivalis (%)", color=color_pg)
    ax2.tick_params(axis="y", labelcolor=color_pg)

    # Correlation (skip day 0 where both are 0)
    mask = days_gin > 0
    if np.sum(mask) >= 3 and np.std(pg_at_gin[mask]) > 0:
        r, p = pearsonr(gin_mean[mask], pg_at_gin[mask])
        ax.set_title(f"(c) Gingipain vs Pg (DH)  r={r:.2f}", fontweight="bold")
    else:
        ax.set_title("(c) Gingipain vs Pg (DH)", fontweight="bold")

    # Legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left")


# ============================================================
# Panel 3: Metabolic Network → A matrix sign validation
# ============================================================


def validate_a_matrix_signs(ax, species_data):
    """
    Fig 4C metabolic network から予測される cross-feeding 方向と
    TMCMC MAP の A 行列符号を比較。
    """
    edges = load_metabolic_network()

    # Species list (model order)
    sp_names = ["S. oralis", "A. naeslundii", "V. dispar/parvula", "F. nucleatum", "P. gingivalis"]
    sp_short = ["So", "An", "Vd", "Fn", "Pg"]

    # Build producer/consumer map
    producers = {}  # metabolite -> [species]
    consumers = {}  # metabolite -> [species]
    for e in edges:
        src, tgt, itype = e["source"], e["target"], e["interaction_type"]
        if itype in ("produces", "produces_enzyme"):
            if src in sp_names:
                producers.setdefault(tgt, []).append(src)
        elif itype == "utilized_by":
            if tgt in sp_names:
                consumers.setdefault(src, []).append(tgt)

    # Predict positive A[consumer, producer] from shared metabolites
    predicted_positive = set()
    for metab in set(producers.keys()) & set(consumers.keys()):
        for prod in producers[metab]:
            for cons in consumers[metab]:
                if prod != cons:
                    i = sp_names.index(cons)
                    j = sp_names.index(prod)
                    predicted_positive.add((i, j))

    # A matrix index map: A[i,j] = effect of species j on species i
    # theta[0:5] = b_i (growth), theta[5:20] = a_ij (off-diagonal)
    # Off-diagonal indices (row-major, skip diagonal):
    # a_12, a_13, a_14, a_15,  (row 0: effect ON So)
    # a_21, a_23, a_24, a_25,  (row 1: effect ON An)
    # a_31, a_32, a_34, a_35,  (row 2: effect ON Vd)
    # a_41, a_42, a_43, a_45,  (row 3: effect ON Fn)
    # a_51, a_52, a_53, a_54   (row 4: effect ON Pg)

    def theta_to_A(theta):
        """Convert 20-param theta to 5x5 symmetric A matrix + b_diag.
        Follows improved_5species_jit.py theta_to_matrices().
        A is symmetric: A[i,j] = A[j,i].
        """
        A = np.zeros((5, 5))
        b = np.zeros(5)
        # M1 block (So=0, An=1)
        A[0, 0] = theta[0]
        A[0, 1] = theta[1]
        A[1, 0] = theta[1]
        A[1, 1] = theta[2]
        b[0] = theta[3]
        b[1] = theta[4]
        # M2 block (Vd=2, Fn=3)
        A[2, 2] = theta[5]
        A[2, 3] = theta[6]
        A[3, 2] = theta[6]
        A[3, 3] = theta[7]
        b[2] = theta[8]
        b[3] = theta[9]
        # Cross M1-M2
        A[0, 2] = theta[10]
        A[2, 0] = theta[10]  # So-Vd
        A[0, 3] = theta[11]
        A[3, 0] = theta[11]  # So-Fn
        A[1, 2] = theta[12]
        A[2, 1] = theta[12]  # An-Vd
        A[1, 3] = theta[13]
        A[3, 1] = theta[13]  # An-Fn
        # Pg (4)
        A[4, 4] = theta[14]
        b[4] = theta[15]
        A[0, 4] = theta[16]
        A[4, 0] = theta[16]  # So-Pg
        A[1, 4] = theta[17]
        A[4, 1] = theta[17]  # An-Pg
        A[2, 4] = theta[18]
        A[4, 2] = theta[18]  # Vd-Pg
        A[3, 4] = theta[19]
        A[4, 3] = theta[19]  # Fn-Pg
        return A, b

    # Load MAP for DH (most informative - all params active)
    theta_dh = load_map_theta("dh_baseline")
    A_dh, b_dh = theta_to_A(theta_dh)

    # Build comparison matrix
    match_matrix = np.full((5, 5), np.nan)
    for i in range(5):
        for j in range(5):
            if i == j:
                continue
            if (i, j) in predicted_positive:
                match_matrix[i, j] = 1 if A_dh[i, j] > 0 else -1
            else:
                match_matrix[i, j] = 0  # no prediction

    # Build display matrix: A + b on diagonal annotation
    disp = A_dh.copy()

    # Plot A matrix with metabolic prediction overlay
    im = ax.imshow(disp, cmap="RdBu_r", vmin=-1, vmax=5, aspect="equal")
    for i in range(5):
        for j in range(5):
            val = disp[i, j]
            label = f"{val:.1f}"
            if i == j:
                label += f"\nb={b_dh[i]:.1f}"
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=6,
                color="white" if abs(val) > 2.5 else "black",
            )
            if (i, j) in predicted_positive:
                # Green checkmark if positive, red X if negative
                symbol = "+" if val > 0 else "-"
                clr = "#4CAF50" if val > 0 else "#F44336"
                ax.text(
                    j + 0.38,
                    i - 0.38,
                    symbol,
                    ha="center",
                    va="center",
                    fontsize=12,
                    color=clr,
                    fontweight="bold",
                )

    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(sp_short, fontsize=8)
    ax.set_yticklabels(sp_short, fontsize=8)
    ax.set_xlabel("Species (j)")
    ax.set_ylabel("Species (i)")

    # Count matches (A is symmetric so count unique pairs)
    unique_pairs = set()
    for i, j in predicted_positive:
        unique_pairs.add((min(i, j), max(i, j)))
    n_predicted = len(unique_pairs)
    n_match = sum(1 for (i, j) in unique_pairs if A_dh[i, j] > 0)
    ax.set_title(
        f"(d) A matrix (DH) vs Metabolic Network\n" f"Sign match: {n_match}/{n_predicted} pairs",
        fontweight="bold",
        fontsize=9,
    )

    return predicted_positive, A_dh


# ============================================================
# Panel 4: OD600 early growth analysis
# ============================================================


def validate_od600(ax):
    """
    Planktonic growth (0-24h) から Commensal vs Dysbiotic の
    増殖速度差を定量化。
    """
    od_data = load_od600_data()

    colors = {"Commensal": "#2196F3", "Dysbiotic": "#F44336"}

    growth_rates = {}
    for model in ["Commensal", "Dysbiotic"]:
        all_t, all_od = [], []
        for rep in sorted(od_data[model].keys()):
            t = np.array(od_data[model][rep]["t"])
            od = np.array(od_data[model][rep]["od"])
            ax.plot(t, od, "-", color=colors[model], alpha=0.15, lw=0.8)
            all_t.extend(t)
            all_od.extend(od)

        # Mean curve
        t_arr = np.array(od_data[model][1]["t"])
        n_reps = len(od_data[model])
        od_matrix = np.zeros((n_reps, len(t_arr)))
        for i, rep in enumerate(sorted(od_data[model].keys())):
            od_r = np.array(od_data[model][rep]["od"])
            od_matrix[i, : len(od_r)] = od_r[: len(t_arr)]

        od_mean = np.mean(od_matrix, axis=0)
        od_std = np.std(od_matrix, axis=0)
        ax.plot(t_arr, od_mean, "-", color=colors[model], lw=2, label=f"{model}")
        ax.fill_between(t_arr, od_mean - od_std, od_mean + od_std, color=colors[model], alpha=0.15)

        # Fit logistic growth in log phase (6-12h)
        mask = (t_arr >= 5) & (t_arr <= 12) & (od_mean > 0.01)
        if np.sum(mask) >= 3:
            t_fit = t_arr[mask]
            od_fit = np.log(od_mean[mask] + 1e-6)
            slope = np.polyfit(t_fit, od_fit, 1)[0]
            growth_rates[model] = slope
            # Doubling time
            dt_hours = np.log(2) / slope if slope > 0 else np.inf
            ax.text(
                0.98,
                0.5 if model == "Commensal" else 0.35,
                f"{model}: µ={slope:.3f}/h, T₂={dt_hours:.1f}h",
                transform=ax.transAxes,
                fontsize=7,
                ha="right",
                color=colors[model],
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("OD600")
    ax.set_title("(e) Planktonic Growth (0–24h)", fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(-0.5, 25)

    return growth_rates


# ============================================================
# Panel 5: Cross-condition A matrix comparison
# ============================================================


def compare_a_matrices(ax):
    """
    4条件の A matrix を比較。条件間で保存される相互作用パターンを可視化。
    """
    conditions = {
        "CS": ("commensal_static", [9, 15, 6, 7, 11, 13, 14, 16, 17, 18, 19]),
        "CH": ("commensal_hobic", [6, 12, 13, 16, 17, 15, 18]),
        "DS": ("dysbiotic_static", [6, 12, 13, 16, 17]),
        "DH": ("dh_baseline", []),
    }
    sp_short = ["So", "An", "Vd", "Fn", "Pg"]

    # Parameter names matching theta[0:20] layout from model_constants.json
    param_labels = [
        "A(So,So)",
        "A(So,An)",
        "A(An,An)",
        "b_So",
        "b_An",
        "A(Vd,Vd)",
        "A(Vd,Fn)",
        "A(Fn,Fn)",
        "b_Vd",
        "b_Fn",
        "A(So,Vd)",
        "A(So,Fn)",
        "A(An,Vd)",
        "A(An,Fn)",
        "A(Pg,Pg)",
        "b_Pg",
        "A(So,Pg)",
        "A(An,Pg)",
        "A(Vd,Pg)",
        "A(Fn,Pg)",
    ]

    # Collect MAPs
    all_theta = {}
    for label, (run, locks) in conditions.items():
        theta = load_map_theta(run)
        # Zero out locked parameters
        for l in locks:
            theta[l] = 0.0
        all_theta[label] = theta

    # Plot heatmap of all 20 params across 4 conditions
    data = np.array([all_theta[c] for c in ["CS", "CH", "DS", "DH"]])

    im = ax.imshow(data, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=6)
    ax.set_yticks(range(4))
    ax.set_yticklabels(["CS", "CH", "DS", "DH"], fontsize=8)
    ax.set_xticks(range(20))
    ax.set_xticklabels(param_labels, fontsize=5, rotation=90)
    ax.set_title("(f) MAP Parameters Across 4 Conditions", fontweight="bold", fontsize=9)

    # Annotate values
    for i in range(4):
        for j in range(20):
            val = data[i, j]
            if abs(val) > 0.01:
                ax.text(
                    j,
                    i,
                    f"{val:.1f}",
                    ha="center",
                    va="center",
                    fontsize=4.5,
                    color="white" if abs(val) > 2 else "black",
                )

    plt.colorbar(im, ax=ax, shrink=0.6, label="Parameter value")


# ============================================================
# Main
# ============================================================


def main():
    print("Loading experimental data...")
    species_data = load_species_data()

    fig = plt.figure(figsize=(16, 10))

    # Layout: 2 rows, 3 columns
    ax1 = fig.add_subplot(2, 3, 1)  # pH time series
    ax2 = fig.add_subplot(2, 3, 2)  # pH scatter
    ax3 = fig.add_subplot(2, 3, 3)  # Gingipain vs Pg
    ax4 = fig.add_subplot(2, 3, 4)  # A matrix signs
    ax5 = fig.add_subplot(2, 3, 5)  # OD600
    ax6 = fig.add_subplot(2, 3, 6)  # Cross-condition comparison

    # Run validations
    print("1. pH independent validation...")
    beta, r_ph, rmse_ph = validate_ph(ax1, ax2, species_data)
    print(f"   pH model: β = {beta}")
    print(f"   R² = {r_ph**2:.4f}, RMSE = {rmse_ph:.4f}")

    print("2. Gingipain vs Pg biomarker...")
    validate_gingipain(ax3, species_data)

    print("3. Metabolic network → A matrix signs...")
    pred_pos, A_dh = validate_a_matrix_signs(ax4, species_data)
    n_match = sum(1 for (i, j) in pred_pos if A_dh[i, j] > 0)
    print(f"   Predicted positive: {len(pred_pos)}, matched: {n_match}")

    print("4. OD600 early growth analysis...")
    growth_rates = validate_od600(ax5)
    for model, rate in growth_rates.items():
        print(f"   {model}: µ = {rate:.4f}/h")

    print("5. Cross-condition A matrix comparison...")
    compare_a_matrices(ax6)

    fig.suptitle(
        "Heine 2025 Independent Validation\n" "(Data NOT used in TMCMC calibration)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    outpath = OUTDIR / "heine2025_independent_validation.png"
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    print(f"\nSaved: {outpath}")

    # Also save summary stats
    summary = {
        "pH_validation": {
            "model": f"pH = {beta[0]:.3f} + {beta[1]:.3f}*phi_So + {beta[2]:.3f}*phi_Vd",
            "R_squared": round(r_ph**2, 4),
            "RMSE": round(rmse_ph, 4),
            "interpretation": "pH predicted from species fractions alone (not used in TMCMC)",
        },
        "metabolic_sign_match": {
            "predicted_positive": len(pred_pos),
            "matched": n_match,
            "pairs": [
                f"{['So','An','Vd','Fn','Pg'][j]}→{['So','An','Vd','Fn','Pg'][i]}"
                for (i, j) in pred_pos
            ],
            "interpretation": "Cross-feeding from metabolic network vs TMCMC A matrix",
        },
        "growth_rates": {k: round(v, 4) for k, v in growth_rates.items()},
    }

    summary_path = OUTDIR / "heine2025_validation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
