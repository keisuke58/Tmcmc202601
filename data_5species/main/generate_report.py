#!/usr/bin/env python3
"""
Generate RESULTS_JP.md and RESULTS_EN.md for a given run directory.
Usage: python generate_report.py <run_dir>
"""

import argparse
import json
import numpy as np
from pathlib import Path
import sys
from datetime import datetime


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_npy(path):
    return np.load(path)


SPECIES_NAMES = ["S. oralis", "A. naeslundii", "V. dispar", "F. nucleatum", "P. gingivalis"]

PARAM_BLOCKS = [
    ("M1", ["a11", "a12", "a22", "b1", "b2"], "S. oralis / A. naeslundii interaction"),
    ("M2", ["a33", "a34", "a44", "b3", "b4"], "V. dispar / F. nucleatum interaction"),
    ("M3", ["a13", "a14", "a23", "a24"], "Cross-interaction (Early -> Late)"),
    ("M4", ["a55", "b5"], "P. gingivalis self"),
    ("M5", ["a15", "a25", "a35", "a45"], "P. gingivalis cross-interaction"),
]

# Flat list of param names in order
PARAM_NAMES = []
for block, params, _ in PARAM_BLOCKS:
    PARAM_NAMES.extend(params)


def get_param_desc(name, lang="EN"):
    # Simple mapping for description
    descriptions = {
        "a11": {"EN": "S. oralis self-interaction", "JP": "S. oralis 自己相互作用"},
        "a12": {"EN": "S. oralis - A. naeslundii", "JP": "S. oralis - A. naeslundii 間"},
        "a22": {"EN": "A. naeslundii self-interaction", "JP": "A. naeslundii 自己相互作用"},
        "b1": {"EN": "S. oralis decay", "JP": "S. oralis 減衰率"},
        "b2": {"EN": "A. naeslundii decay", "JP": "A. naeslundii 減衰率"},
        "a33": {"EN": "V. dispar self-interaction", "JP": "V. dispar 自己相互作用"},
        "a34": {"EN": "V. dispar - F. nucleatum", "JP": "V. dispar - F. nucleatum 間"},
        "a44": {"EN": "F. nucleatum self-interaction", "JP": "F. nucleatum 自己相互作用"},
        "b3": {"EN": "V. dispar decay", "JP": "V. dispar 減衰率"},
        "b4": {"EN": "F. nucleatum decay", "JP": "F. nucleatum 減衰率"},
        "a13": {"EN": "S. oralis - V. dispar", "JP": "S. oralis - V. dispar 間"},
        "a14": {"EN": "S. oralis - F. nucleatum", "JP": "S. oralis - F. nucleatum 間"},
        "a23": {"EN": "A. naeslundii - V. dispar", "JP": "A. naeslundii - V. dispar 間"},
        "a24": {"EN": "A. naeslundii - F. nucleatum", "JP": "A. naeslundii - F. nucleatum 間"},
        "a55": {"EN": "P. gingivalis self-interaction", "JP": "P. gingivalis 自己相互作用"},
        "b5": {"EN": "P. gingivalis decay", "JP": "P. gingivalis 減衰率"},
        "a15": {"EN": "S. oralis - P. gingivalis", "JP": "S. oralis - P. gingivalis 間"},
        "a25": {"EN": "A. naeslundii - P. gingivalis", "JP": "A. naeslundii - P. gingivalis 間"},
        "a35": {"EN": "V. dispar - P. gingivalis", "JP": "V. dispar - P. gingivalis 間"},
        "a45": {"EN": "F. nucleatum - P. gingivalis", "JP": "F. nucleatum - P. gingivalis 間"},
    }
    return descriptions.get(name, {}).get(lang, name)


def format_matrix(vec, lang="EN"):
    # Reconstruct 5x5 matrix A and vector b from 20-dim vector
    # Indices based on previous knowledge:
    # 0:a11, 1:a12, 2:a22, 3:b1, 4:b2
    # 5:a33, 6:a34, 7:a44, 8:b3, 9:b4
    # 10:a13, 11:a14, 12:a23, 13:a24
    # 14:a55, 15:b5
    # 16:a15, 17:a25, 18:a35, 19:a45

    # Mapping to matrix A (5x5)
    A = np.zeros((5, 5))
    b = np.zeros(5)

    # M1
    A[0, 0], A[0, 1], A[1, 1] = vec[0], vec[1], vec[2]
    A[1, 0] = vec[1]  # Symmetric
    b[0], b[1] = vec[3], vec[4]

    # M2
    A[2, 2], A[2, 3], A[3, 3] = vec[5], vec[6], vec[7]
    A[3, 2] = vec[6]
    b[2], b[3] = vec[8], vec[9]

    # M3
    A[0, 2], A[0, 3] = vec[10], vec[11]
    A[2, 0], A[3, 0] = vec[10], vec[11]
    A[1, 2], A[1, 3] = vec[12], vec[13]
    A[2, 1], A[3, 1] = vec[12], vec[13]

    # M4
    A[4, 4] = vec[14]
    b[4] = vec[15]

    # M5
    A[0, 4] = vec[16]
    A[4, 0] = vec[16]
    A[1, 4] = vec[17]
    A[4, 1] = vec[17]
    A[2, 4] = vec[18]
    A[4, 2] = vec[18]
    A[3, 4] = vec[19]
    A[4, 3] = vec[19]

    lines = []
    lines.append("```")
    header = "              " + "".join([f"{s[:7]:<10}" for s in SPECIES_NAMES])
    lines.append(header)
    for i in range(5):
        row_str = f"{SPECIES_NAMES[i][:10]:<12}[ "
        for j in range(5):
            row_str += f" {A[i,j]:.3f}    "
        row_str += "]"
        lines.append(row_str)
    lines.append("```")
    return "\n".join(lines), b


def generate_markdown(run_dir, lang="JP"):
    run_path = Path(run_dir)

    # Load data
    try:
        config = load_json(run_path / "config.json")
        results_summary = load_json(run_path / "results_summary.json")
        fit_metrics = load_json(run_path / "fit_metrics.json")
        # theta_MAP = load_json(run_path / "theta_MAP.json")["theta_full"] # results_summary has this too usually
        # theta_mean = load_json(run_path / "theta_mean.json")["theta_full"]
        logL = load_npy(run_path / "logL.npy")
        samples = load_npy(run_path / "samples.npy")
    except Exception as e:
        return f"Error loading files: {e}"

    map_vec = results_summary["MAP"]
    mean_vec = results_summary["mean"]

    # Calculate stats from samples for CI
    samples_std = np.std(samples, axis=0)
    samples_p05 = np.percentile(samples, 5, axis=0)
    samples_p95 = np.percentile(samples, 95, axis=0)

    # Meta info
    run_id = run_path.name
    date_str = datetime.now().strftime("%Y-%m-%d")
    condition = config["metadata"]["condition"]
    cultivation = config["metadata"]["cultivation"]

    # Translations
    text = {
        "JP": {
            "title": f"TMCMC パラメータ推定結果: {condition} {cultivation}",
            "run_id": "ラン ID",
            "date": "実行日",
            "model": "5菌種バイオフィルム Transient Storage Model (TSM)",
            "config": "1. 実行設定",
            "item": "設定項目",
            "value": "値",
            "condition": "条件 (Condition)",
            "cultivation_label": "培養方法 (Cultivation)",
            "dt": "時間刻み (dt)",
            "max_step": "最大タイムステップ",
            "sigma": "sigma_obs (観測ノイズ)",
            "particles": "粒子数 (N_particles)",
            "time": "計算時間",
            "converged": "収束状態",
            "init_cond": "初期条件",
            "species": "菌種",
            "vol_frac": "phi_init (初期体積分率)",
            "data": "2. 実験データ",
            "obs_table": "観測された生菌体積分率 (phibar = phi * psi)",
            "day": "日数",
            "total": "総体積",
            "fit": "3. フィッティング精度",
            "fit_global": "3.1 全体指標",
            "metric": "指標",
            "map": "MAP 推定値",
            "mean": "事後平均",
            "rmse_all": "RMSE (全体)",
            "mae_all": "MAE (全体)",
            "max_err": "最大絶対誤差",
            "fit_species": "3.2 菌種別 RMSE",
            "map_rmse": "MAP RMSE",
            "mean_rmse": "Mean RMSE",
            "params": "4. 推定パラメータ",
            "param_list": "4.1 パラメータ一覧",
            "block": "ブロック",
            "param": "パラメータ",
            "bio_mean": "生物学的意味",
            "std": "標準偏差",
            "ci": "90% 信頼区間",
            "matrix_a": "4.2 相互作用行列 A (MAP推定値)",
            "vec_b": "4.3 減衰ベクトル b (MAP推定値)",
            "decay": "b (減衰率)",
            "stats": "5. 対数尤度統計",
            "stat_item": "統計量",
            "max_logl": "最大 logL",
            "mean_logl": "平均 logL",
            "med_logl": "中央値 logL",
            "min_logl": "最小 logL",
            "samples": "サンプル数",
            "figs": "7. 図一覧 (Figures)",
            "fig_std": "標準図",
            "fig_extra": "追加分析図",
            "file": "ファイル名",
            "desc": "説明",
        },
        "EN": {
            "title": f"TMCMC Parameter Estimation Results: {condition} {cultivation}",
            "run_id": "Run ID",
            "date": "Date",
            "model": "5-species Biofilm Transient Storage Model (TSM)",
            "config": "1. Run Configuration",
            "item": "Item",
            "value": "Value",
            "condition": "Condition",
            "cultivation_label": "Cultivation",
            "dt": "Time Step (dt)",
            "max_step": "Max Timesteps",
            "sigma": "sigma_obs",
            "particles": "Particles (N_particles)",
            "time": "Elapsed Time",
            "converged": "Convergence",
            "init_cond": "Initial Conditions",
            "species": "Species",
            "vol_frac": "phi_init (Volume Fraction)",
            "data": "2. Experimental Data",
            "obs_table": "Observed Viable Volume Fraction (phibar = phi * psi)",
            "day": "Day",
            "total": "Total Volume",
            "fit": "3. Fitting Accuracy",
            "fit_global": "3.1 Global Metrics",
            "metric": "Metric",
            "map": "MAP Estimate",
            "mean": "Posterior Mean",
            "rmse_all": "RMSE (Total)",
            "mae_all": "MAE (Total)",
            "max_err": "Max Absolute Error",
            "fit_species": "3.2 Per-Species RMSE",
            "map_rmse": "MAP RMSE",
            "mean_rmse": "Mean RMSE",
            "params": "4. Estimated Parameters",
            "param_list": "4.1 Parameter List",
            "block": "Block",
            "param": "Parameter",
            "bio_mean": "Biological Meaning",
            "std": "Std Dev",
            "ci": "90% CI",
            "matrix_a": "4.2 Interaction Matrix A (MAP)",
            "vec_b": "4.3 Decay Vector b (MAP)",
            "decay": "b (Decay Rate)",
            "stats": "5. Log-Likelihood Statistics",
            "stat_item": "Statistic",
            "max_logl": "Max logL",
            "mean_logl": "Mean logL",
            "med_logl": "Median logL",
            "min_logl": "Min logL",
            "samples": "Num Samples",
            "figs": "7. Figures",
            "fig_std": "Standard Figures",
            "fig_extra": "Extra Analysis Figures",
            "file": "Filename",
            "desc": "Description",
        },
    }

    t = text[lang]

    md = []
    md.append(f"# {t['title']}\n")
    md.append(f"**{t['run_id']}:** `{run_id}`")
    md.append(f"**{t['date']}:** {date_str}")
    md.append(f"**{t['model']}:** {t['model']}\n")
    md.append("---\n")

    # 1. Config
    md.append(f"## {t['config']}\n")
    md.append(f"| {t['item']} | {t['value']} |")
    md.append("|---|---|")
    md.append(f"| {t['condition']} | {condition} |")
    md.append(f"| {t['cultivation_label']} | {cultivation} |")
    md.append(f"| {t['particles']} | {config['n_particles']} |")
    md.append(f"| {t['sigma']} | {config['sigma_obs']:.4f} |")
    elapsed_h = results_summary.get("elapsed_time", 0) / 3600.0
    md.append(f"| {t['time']} | {elapsed_h:.1f} hours |")

    conv_status = "Converged" if all(results_summary.get("converged", [False])) else "Not Converged"
    if lang == "JP":
        conv_status = "収束" if all(results_summary.get("converged", [False])) else "未収束"
    md.append(f"| {t['converged']} | {conv_status} |\n")

    # Init conditions
    md.append(f"### {t['init_cond']}\n")
    md.append(f"| {t['species']} | {t['vol_frac']} |")
    md.append("|---|:---:|")
    phi_init_exp = config["metadata"]["phi_init_exp"]
    for i, sp in enumerate(SPECIES_NAMES):
        md.append(f"| *{sp}* | {phi_init_exp[i]:.4f} |")
    md.append("\n---\n")

    # 2. Data
    md.append(f"## {t['data']}\n")
    md.append(f"{t['obs_table']}:\n")
    days = config["metadata"]["days"]
    # We need to load data.npy to display this table accurately or just use total volumes from config
    # Assuming standard structure, let's skip detailed data table for now to save complexity,
    # or reconstruct if needed. The original report had it.
    # Let's try to grab total volumes from config.
    total_vols = config["metadata"]["total_volumes"]

    header = (
        f"| {t['day']} | " + " | ".join([f"*{s}*" for s in SPECIES_NAMES]) + f" | {t['total']} |"
    )
    md.append(header)
    md.append("|:---:|" + ":---:|" * 5 + ":---:|")

    # Need actual species proportions data...
    # It's in data.npy. shape (n_days, 5) or similar.
    try:
        data_arr = np.load(run_path / "data.npy")
        # data_arr shape is (n_days, 5) usually representing phibar for each species
        for i, day in enumerate(days):
            row = f"| {day} | "
            sum_vol = 0
            for j in range(5):
                val = data_arr[i, j]
                row += f"{val:.4f} | "
                sum_vol += val  # Note: data is already fraction of total vol? No, it's phibar = phi * psi * total_vol?
                # Actually data.npy usually contains the target values for fitting.
                # If fitting to phibar, it is phibar.
            row += f"{total_vols[i]:.2f} |"
            md.append(row)
    except (IndexError, KeyError, TypeError) as e:
        md.append(f"(Data table could not be reconstructed: {e})")

    md.append("\n---\n")

    # 3. Fitting
    md.append(f"## {t['fit']}\n")
    md.append(f"### {t['fit_global']}\n")
    md.append(f"| {t['metric']} | {t['map']} | {t['mean']} |")
    md.append("|---|:---:|:---:|")
    md.append(
        f"| {t['rmse_all']} | **{fit_metrics['MAP']['rmse_total']:.4f}** | {fit_metrics['Mean']['rmse_total']:.4f} |"
    )
    md.append(
        f"| {t['mae_all']} | **{fit_metrics['MAP']['mae_total']:.4f}** | {fit_metrics['Mean']['mae_total']:.4f} |"
    )
    md.append(
        f"| {t['max_err']} | **{fit_metrics['MAP']['max_abs']:.4f}** | {fit_metrics['Mean']['max_abs']:.4f} |\n"
    )

    md.append(f"### {t['fit_species']}\n")
    md.append(f"| {t['species']} | {t['map_rmse']} | {t['mean_rmse']} |")
    md.append("|---|:---:|:---:|")
    for i, sp in enumerate(SPECIES_NAMES):
        md.append(
            f"| *{sp}* | {fit_metrics['MAP']['rmse_per_species'][i]:.4f} | {fit_metrics['Mean']['rmse_per_species'][i]:.4f} |"
        )
    md.append("\n---\n")

    # 4. Parameters
    md.append(f"## {t['params']}\n")
    md.append(f"### {t['param_list']}\n")
    md.append(
        f"| {t['block']} | {t['param']} | {t['bio_mean']} | MAP | Mean | {t['std']} | {t['ci']} |"
    )
    md.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    for i, name in enumerate(PARAM_NAMES):
        # Find block
        blk = "M?"
        for b_name, b_params, _ in PARAM_BLOCKS:
            if name in b_params:
                blk = b_name
                break

        desc = get_param_desc(name, lang)
        val_map = map_vec[i]
        val_mean = mean_vec[i]
        val_std = samples_std[i]
        val_p05 = samples_p05[i]
        val_p95 = samples_p95[i]

        md.append(
            f"| **{blk}** | {name} | {desc} | {val_map:.4f} | {val_mean:.4f} | {val_std:.4f} | [{val_p05:.2f}, {val_p95:.2f}] |"
        )

    md.append(f"\n### {t['matrix_a']}\n")
    mat_str, b_vec = format_matrix(map_vec, lang)
    md.append(mat_str)

    md.append(f"\n### {t['vec_b']}\n")
    md.append(f"| {t['species']} | {t['decay']} |")
    md.append("|---|:---:|")
    for i, sp in enumerate(SPECIES_NAMES):
        md.append(f"| *{sp}* | {b_vec[i]:.3f} |")

    md.append("\n---\n")

    # 5. LogL
    md.append(f"## {t['stats']}\n")
    md.append(f"| {t['stat_item']} | {t['value']} |")
    md.append("|---|:---:|")
    md.append(f"| {t['max_logl']} | {np.max(logL):.3f} |")
    md.append(f"| {t['mean_logl']} | {np.mean(logL):.3f} |")
    md.append(f"| {t['med_logl']} | {np.median(logL):.3f} |")
    md.append(f"| {t['min_logl']} | {np.min(logL):.3f} |")
    md.append(f"| {t['samples']} | {len(logL)} |")

    md.append("\n---\n")

    # 7. Figures
    md.append(f"## {t['figs']}\n")
    md.append(f"### {t['fig_std']}\n")
    md.append(f"| {t['file']} | {t['desc']} |")
    md.append("|---|---|")

    std_figs = [
        ("TSM_simulation_*_MAP_Fit_with_data.png", "MAP Fit vs Data"),
        ("TSM_simulation_*_Mean_Fit_with_data.png", "Mean Fit vs Data"),
        ("posterior_predictive_*_PosteriorBand.png", "90% Posterior Bands"),
        ("corner_plot_*_Corner.png", "Corner Plot"),
        ("parameter_distributions_*_Params.png", "Parameter Histograms"),
        ("residuals_*_Residuals.png", "Residuals"),
    ]
    if lang == "JP":
        std_figs = [
            ("TSM_simulation_*_MAP_Fit_with_data.png", "MAP推定値の時間発展 vs 観測データ"),
            ("TSM_simulation_*_Mean_Fit_with_data.png", "事後平均の時間発展 vs 観測データ"),
            ("posterior_predictive_*_PosteriorBand.png", "5-95% 事後予測バンド"),
            ("corner_plot_*_Corner.png", "20x20 コーナープロット"),
            ("parameter_distributions_*_Params.png", "パラメータ別ヒストグラム"),
            ("residuals_*_Residuals.png", "残差プロット"),
        ]

    for f, d in std_figs:
        md.append(f"| `{f}` | {d} |")

    md.append(f"\n### {t['fig_extra']}\n")
    md.append(f"| {t['file']} | {t['desc']} |")
    md.append("|---|---|")

    extra_figs = [
        ("Fig_A01_interaction_matrix_heatmap.png", "Interaction Matrix Heatmap"),
        ("Fig_A02_per_species_panel.png", "Per-species Panel"),
        ("Fig_A03_state_decomposition.png", "State Decomposition"),
        ("Fig_A04_species_composition.png", "Species Composition"),
        ("Fig_A05_parameter_violins.png", "Parameter Violins"),
        ("Fig_A06_correlation_matrix.png", "Correlation Matrix"),
        ("Fig_A07_loglikelihood_landscape.png", "LogL Landscape"),
        ("Fig_A08_posterior_predictive_check.png", "PPC"),
        ("Fig_A09_MAP_vs_Mean_comparison.png", "MAP vs Mean"),
    ]
    if lang == "JP":
        extra_figs = [
            ("Fig_A01_interaction_matrix_heatmap.png", "5x5 相互作用行列ヒートマップ"),
            ("Fig_A02_per_species_panel.png", "菌種別パネルプロット"),
            ("Fig_A03_state_decomposition.png", "状態変数分解 (phi, psi)"),
            ("Fig_A04_species_composition.png", "菌種組成推移"),
            ("Fig_A05_parameter_violins.png", "パラメータバイオリンプロット"),
            ("Fig_A06_correlation_matrix.png", "相関行列"),
            ("Fig_A07_loglikelihood_landscape.png", "対数尤度地形"),
            ("Fig_A08_posterior_predictive_check.png", "事後予測チェック"),
            ("Fig_A09_MAP_vs_Mean_comparison.png", "MAP vs Mean 比較"),
        ]

    for f, d in extra_figs:
        md.append(f"| `{f}` | {d} |")

    return "\n".join(md)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=str, help="Path to run directory")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: Directory {run_dir} not found.")
        sys.exit(1)

    # Generate JP
    print(f"Generating RESULTS_JP.md for {run_dir.name}...")
    md_jp = generate_markdown(run_dir, lang="JP")
    with open(run_dir / "RESULTS_JP.md", "w") as f:
        f.write(md_jp)

    # Generate EN
    print(f"Generating RESULTS_EN.md for {run_dir.name}...")
    md_en = generate_markdown(run_dir, lang="EN")
    with open(run_dir / "RESULTS_EN.md", "w") as f:
        f.write(md_en)

    print("Done.")
