#!/usr/bin/env bash
# create_github_issues.sh — P4〜P9 の GitHub Issues を一括作成
#
# 使い方:
#   gh auth login          # ← 先にブラウザ認証
#   bash create_github_issues.sh
#
# 注: gh CLI (GitHub CLI) が必要。インストール: https://cli.github.com
set -euo pipefail

REPO="keisuke58/Tmcmc202601"

echo "=== GitHub Issues 作成 (P4〜P9) ==="
echo "  リポジトリ: $REPO"
echo

# ── P4 ─────────────────────────────────────────────────────────────────────
echo "[P4] 3D 歯形状 + biofilm 固有ひずみ Abaqus モデル..."
gh issue create --repo "$REPO" \
  --title "[P4] Apply biofilm eigenstrain to 3D patient tooth geometry (OpenJaw STL)" \
  --label "enhancement,priority:high,FEM,3D-geometry" \
  --body "$(cat <<'BODY'
## 概要
現在の `biofilm_1d_bar_{condition}.inp` は T3D2 1D バー実証モデルにすぎない。
`external_tooth_models/` の Patient_1 STL と `biofilm_conformal_tet.py` を用い、
実歯形状上での成長固有ひずみ解析を完成させ、論文 Fig3 (3D 応力分布) の土台とする。

## タスク
- [ ] P1_Tooth_23.stl に biofilm_conformal_tet.py を適用 (90% mode, --neo-hookean)
- [ ] 4条件の INP 生成: `p23_{condition}_spatial_eigenstrain_nh.inp`
- [ ] 4条件の Abaqus GROWTH step 実行
- [ ] ODB から S_Mises, U, NT11 を CSV 抽出 (`odb_extract.py`)
- [ ] commensal vs dysbiotic の最大圧縮応力比 (≥10×) を確認
- [ ] `_abaqus_extracted/standard_3d/{condition}/` に結果保存
- [ ] `FEM/_results/` に T_field_map.png, S_mises_growth.png 生成

## Acceptance Criteria
- 4条件すべてで Abaqus が COMPLETED
- `dh_baseline` 基質応力が `commensal_static` の ≥10×
- T_field_map.png で歯周ポケット側に高 α 集中が確認できる

## 依存関係
- blocks: P6, P9 (Fig3)
- depends on: P8 (k_alpha 確定後が望ましい)
BODY
)"
echo "  → P4 作成完了"
echo

# ── P5 ─────────────────────────────────────────────────────────────────────
echo "[P5] α_Monod(x) の条件別空間プロファイル分化..."
gh issue create --repo "$REPO" \
  --title "[P5] Differentiate α_Monod(x) spatial profiles per condition via ODE coupling" \
  --label "enhancement,priority:medium,multiscale" \
  --body "$(cat <<'BODY'
## 概要
1D Hamilton PDE の拡散均質化により、4条件で `alpha_monod(x)` が同一になっている。
`multiscale_coupling_1d.py` を改修し、条件別の空間プロファイルを生成するか、
均質化限界を定量的に文書化する。

## タスク
- [ ] N=100 ノード × T=200 ステップの高解像度 1D PDE を 4条件で実行
- [ ] `phi_total(x, T_final)` の条件間差を定量化 (max/min 比)
- [ ] 差が < 5%: "均質化限界" として `methods_supplement_fem.md` に記載
- [ ] 差が > 5%: `macro_eigenstrain_{cond}_differentiated.csv` を 4条件分生成
- [ ] `generate_hybrid_macro_csv.py` に `--use-differentiated-alpha` オプションを追加
- [ ] `generate_pipeline_summary.py` の Row 2 を条件別カラーラインで更新
- [ ] `_multiscale_results/spatial_profile_comparison.png` 生成

## Acceptance Criteria
- 条件間差の結論 (< 5% or > 5%) が数値根拠付きで文書化される
- 条件差が有意な場合は differentiated CSV が 4条件分生成される
- `pipeline_summary.png` が更新版 CSV を反映して再生成できる

## 依存関係
- blocks: P9 (pipeline figure の完成度)
- depends on: なし (独立)
BODY
)"
echo "  → P5 作成完了"
echo

# ── P6 ─────────────────────────────────────────────────────────────────────
echo "[P6] 事後不確かさの FEM アンサンブル伝播..."
gh issue create --repo "$REPO" \
  --title "[P6] Propagate TMCMC posterior uncertainty into FEM stress ensemble" \
  --label "enhancement,priority:high,UQ,posterior,FEM" \
  --body "$(cat <<'BODY'
## 概要
`run_posterior_abaqus_ensemble.py` は骨格実装済みだが、実歯 3D モデル (P4) への
適用が未実施。4条件 × 20 サンプルのアンサンブル FEM を実行し、
S_Mises の 5%/50%/95% 信用バンドを生成する (論文 Fig4)。

## タスク
- [ ] `_posterior_abaqus/{cond}/meta.json` の現状確認 (1D バー vs 3D 歯)
- [ ] `run_posterior_abaqus_ensemble.py` の `_run_fem()` を P4 の 3D INP に切替
- [ ] 4条件 × N_SAMPLES=20 のアンサンブル実行 (done.flag で resume 対応)
- [ ] 各サンプルから `[substrate_stress, surface_stress]` を抽出し `stress_all.npy` 集約
- [ ] `plot_posterior_uncertainty.py` で 5 図を生成:
  - Fig1_stress_violin.png
  - Fig2_stress_ci_bars.png
  - Fig3_sensitivity_heatmap.png (Spearman ρ)
  - Fig4_top_params_scatter.png
  - Fig5_stress_summary_panel.png ← 論文 Fig4 候補
- [ ] Spearman ρ で `a35`, `a45` が `dh_baseline` 基質応力の上位に来ることを確認
- [ ] 結果を `docs/` に移動またはシンボリックリンク

## Acceptance Criteria
- 4条件 × 20 サンプル全ての `done.flag` 存在
- Fig2 で `dh_baseline` の p95 が `commensal_static` の p50 より高い
- Fig3 で `|a35 ρ| ≥ 0.3` or `|a45 ρ| ≥ 0.3` (dh_baseline 基質応力)
- Fig5 が単独で論文 Fig4 として成立する (タイトル・軸ラベル完備)

## 依存関係
- depends on: P4 (3D INP), P7 (バッチ自動化)
- blocks: P9 (Fig4)
BODY
)"
echo "  → P6 作成完了"
echo

# ── P7 ─────────────────────────────────────────────────────────────────────
echo "[P7] Abaqus バッチ自動化 + ODB 後処理..."
gh issue create --repo "$REPO" \
  --title "[P7] Abaqus batch automation and ODB post-processing pipeline" \
  --label "automation,priority:medium,infrastructure,ODB" \
  --body "$(cat <<'BODY'
## 概要
INP 生成 → Abaqus 実行 → ODB 抽出 → CSV 保存の完全自動化バッチパイプラインを整備。
P4/P6 の 4条件 × 複数サンプル実行を無人化し、resume 対応と実行ログ保存を確保する。

## タスク
- [ ] 以下の 4 ステップを行う 1 ファイルのバッチランナーを設計:
  1. `biofilm_conformal_tet.py` で INP 生成
  2. `abaqus job=... cpus=4 background` 実行
  3. `abaqus python odb_extract.py {ODB}` 実行
  4. CSV → `_abaqus_extracted/{condition}_{timestamp}/` 保存
- [ ] 完了判定: `{job}.sta` の最終行 "COMPLETED" 検出
- [ ] 失敗時: `{job}.log` の "error" 行を検出してアラート出力
- [ ] `run_all_conditions_3d.sh` の作成 (4条件順次実行)
- [ ] `odb_extract.py` を拡張: GROWTH step + LOAD step から S_Mises, U1/U2/U3, NT11 を抽出
- [ ] 実行時間計測と `timing_summary.csv` 生成
- [ ] CI 的チェック: 抽出 CSV の行数・カラム数・NaN 率をアサーション確認

## Acceptance Criteria
- `run_all_conditions_3d.sh` の単一実行で 4条件が無人完結
- 既完了条件は done.flag でスキップ (resume 対応)
- 各条件に `odb_nodes.csv`, `odb_elements.csv`, `odb_growth_step.csv` の 3 ファイル生成
- 失敗ジョブのエラーメッセージがターミナルに表示される

## 依存関係
- depends on: P4 (3D INP が存在することが前提)
- blocks: P6 (大規模アンサンブルの自動化)
BODY
)"
echo "  → P7 作成完了"
echo

# ── P8 ─────────────────────────────────────────────────────────────────────
echo "[P8] k_alpha 較正と固有ひずみ物理検証..."
gh issue create --repo "$REPO" \
  --title "[P8] Calibrate k_alpha against Klempt 2024 reference and validate eigenstrain magnitude" \
  --label "calibration,priority:medium,physics,validation" \
  --body "$(cat <<'BODY'
## 概要
`k_alpha = 0.05 [T*^-1]` は Klempt 2024 に文献値がなく物理検証未実施。
`compute_alpha_eigenstrain.py` を用いて 5 水準の k_alpha 感度解析を行い、
`eps_growth = alpha/3` の妥当性範囲を定量化する。

## タスク
- [ ] Klempt 2024 (Biomech Model Mechanobiol 23:2091-2113) から
  biofilm 成長ひずみの報告値を抽出 → `docs/klempt2024_eigenstrain_values.md`
- [ ] `k_alpha ∈ {0.01, 0.03, 0.05, 0.07, 0.10}` の 5 水準で 4 条件の `alpha_final` を計算
  → `_abaqus_input/k_alpha_sensitivity.csv`
- [ ] `eps_growth = alpha/3 ∈ O(0.01)〜O(0.30)` に収まる k_alpha を "合理範囲" として確定
- [ ] 1D バーモデルを 5 水準で実行し `sigma_0 = -E * eps_growth` を計算
- [ ] 生物学的根拠 (biofilm 厚さ成長速度 ~1 µm/h 等) と照合し最適 k_alpha を選定
- [ ] `compute_alpha_eigenstrain.py` の `--k-alpha` デフォルト値を更新
- [ ] `biofilm_conformal_tet.py` の `--k-alpha` デフォルト値も更新
- [ ] `docs/DI_Eeff_sensitivity.md` に k_alpha 感度解析結果を追記

## Acceptance Criteria
- `docs/klempt2024_eigenstrain_values.md` に参照値 (または不在の明示) が記録
- `k_alpha_sensitivity.csv` が 20 行 (5 水準 × 4 条件) で生成
- 推奨 k_alpha がコードのデフォルト値に反映され docstring に根拠記載
- 全水準で `eps_growth < 0.35` (非線形補正不要の限界値以下)

## 依存関係
- depends on: なし (1D バーモデルのみで完結)
- blocks: P4 の正確な alpha 入力値, P9 の論文記述精度
BODY
)"
echo "  → P8 作成完了"
echo

# ── P9 ─────────────────────────────────────────────────────────────────────
echo "[P9] 論文用メイン結果図 (Fig1〜Fig4) 生成..."
gh issue create --repo "$REPO" \
  --title "[P9] Generate paper-quality main result figures (Fig1-Fig4 for manuscript)" \
  --label "visualization,priority:high,paper,figures" \
  --body "$(cat <<'BODY'
## 概要
論文 Fig1〜Fig4 を 300 DPI で生成する。`generate_pipeline_summary.py` (Fig1)、
`plot_posterior_uncertainty.py` (Fig4) を起点とし、学術論文フォーマットで 4 図を完成させる。

## タスク

### Fig1: 全体フレームワーク図 (P4/P6 非依存, 先行着手可)
- [ ] `generate_pipeline_summary.py` の Row 0 (フロー図) を拡張
- [ ] OpenJaw Patient_1 の Mandible + Teeth STL を 3D scatter 可視化
- [ ] TMCMC → FEM → UQ の矢印・ボックス注釈追加
- [ ] 出力: `FEM/figures/Fig1_framework.png` (300 DPI, 180 mm 幅)

### Fig2: TMCMC パラメータ同定サマリ (P4/P6 非依存, 先行着手可)
- [ ] (a) `a35, a45, b5` の 4条件 posterior violin/KDE
  → `data_5species/_runs/{cond}/posterior_samples.npy` から読み込み
- [ ] (b) data vs model fit の縮小版 (4条件 × 5 菌種 時系列)
- [ ] 出力: `FEM/figures/Fig2_tmcmc_summary.png`

### Fig3: 3D 歯形状上の応力分布 (P4 完了後)
- [ ] T23 歯冠の S_Mises カラーマップ (4条件並列)
- [ ] 出力: `FEM/figures/Fig3_stress_distribution.png`

### Fig4: 不確実性伝播 (P6 完了後)
- [ ] (a) 代表部位の応力 p05/p50/p95 バンド (深さ方向)
- [ ] (b) `plot_posterior_uncertainty.py` の Fig5 を論文サイズに調整
- [ ] 出力: `FEM/figures/Fig4_uq_propagation.png`

### 共通
- [ ] 全図: DPI=300, フォントサイズ ≥10pt, colorblind-safe カラーマップ
- [ ] `FIGURES_MANIFEST.json` 更新
- [ ] `biofilm_3tooth_report.tex` の図挿入箇所を更新し pdflatex ビルド

## Acceptance Criteria
- 4 ファイル (`Fig1〜Fig4_*.png`) が 300 DPI で生成
- Fig3: `dh_baseline` と `commensal_static` の S_Mises に視覚的に明確な差異
- Fig4: dh_baseline のバンド幅 > commensal のバンド幅
- `biofilm_3tooth_report.pdf` が Fig1-4 を正しく取り込んだ状態でビルド成功

## 依存関係
- depends on: P4 (Fig3), P6 (Fig4)
- Fig1, Fig2 は P4/P6 非依存で先行着手可能
BODY
)"
echo "  → P9 作成完了"
echo

echo "========================================"
echo "全 Issues 作成完了 (P4〜P9)"
echo "確認: https://github.com/$REPO/issues"
echo "========================================"
echo
echo "推奨実行順序:"
echo "  1. P8 (k_alpha 較正, 独立・着手コスト最小)"
echo "  2. P5 (alpha_monod 分化, 独立)"
echo "  3. P4 (3D 歯形状 Abaqus, P8 後が望ましい)"
echo "  4. P7 (バッチ自動化, P4 と並行可)"
echo "  5. P6 (事後アンサンブル, P4+P7 後)"
echo "  6. P9 (論文図, P4+P6 後; Fig1/Fig2 は先行可)"
