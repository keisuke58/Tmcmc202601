# 次のアクションプラン

## 📋 現状確認

### ✅ 既に実装済み
- **p-box可視化**: `tmcmc/pbox.py` と `make_report.py` で実装済み
- **観測共分散**: `tmcmc/core/evaluator.py` の `log_likelihood_sparse` で `rho` パラメータによる相関対応済み
- **lock-paper-conditions**: `run_pipeline.py` と `case2_tmcmc_linearization.py` で実装済み
- **実験→レポートパイプライン**: `tmcmc/run_pipeline.py` で1コマンド実行可能

### 🔍 確認が必要な項目
- 論文条件固定runの確定とrun_idの資料への記載
- LaTeX資料（`docs/tmcmc_flow_report*.tex`）の最新実装への同期
- REPORT.mdでのp-box/共分散条件の明記

---

## 🎯 優先度別アクションプラン

### 【最優先】ゴールA: "再現条件"を固定し、比較の土台を作る

#### タスク1: 論文条件固定runの実行と確定
**目的**: 論文と同じ条件で再現可能なrunを1つ確定し、run_idを資料に固定

**手順**:
```bash
# M1モデルで論文条件固定runを実行
python tmcmc/run_pipeline.py \
  --mode paper \
  --lock-paper-conditions \
  --use-paper-analytical \
  --seed 123 \
  --run-id paper_M1_seed123_fixed \
  --models M1

# 実行結果を確認
# - βが1.0に到達しているか
# - REPORT.mdが生成されているか
# - 診断テーブルが存在するか
```

**成果物**:
- `tmcmc/_runs/paper_M1_seed123_fixed/` ディレクトリ
- 確定したrun_idを `docs/BEST_RUN.txt` または専用ファイルに記録

**DoD（完了条件）**:
- [ ] runが正常に完了（β=1.0到達）
- [ ] REPORT.mdが生成されている
- [ ] run_idが確定し、記録されている

---

#### タスク2: LaTeX資料の最新実装への同期
**目的**: `docs/tmcmc_flow_report*.tex` と `docs/tmcmc_flow_slides*.tex` を最新実装に合わせて更新

**確認・更新項目**:
1. **入口コマンドの記載**
   - 現在の実装に合わせた正確なコマンド例
   - `--lock-paper-conditions` フラグの説明
   - 確定したrun_idの参照方法

2. **固定条件の明記**
   - ノイズレベル（sigma_obs, cov_rel）
   - その他の論文条件パラメータ

3. **出力ファイルの参照**
   - `tmcmc/_runs/<run_id>/figures/*.png` の参照方法
   - 図の埋め込みではなく参照形式

**手順**:
```bash
# 1. 現在のLaTeXファイルを確認
cat docs/tmcmc_flow_report_en.tex | grep -A 5 "Reproducibility"

# 2. 確定したrun_idを確認
cat docs/BEST_RUN.txt

# 3. LaTeXファイルを更新（必要に応じて）
# - 入口コマンドの更新
# - run_idの記載
# - 条件の明記
```

**成果物**:
- 更新された `docs/tmcmc_flow_report*.tex`
- 更新された `docs/tmcmc_flow_slides*.tex`
- PDF再生成（`python docs/build_pdfs.py`）

**DoD**:
- [ ] 入口コマンドが正確に記載されている
- [ ] 確定したrun_idが資料に記載されている
- [ ] 固定条件（ノイズ等）が明記されている
- [ ] PDFが再生成され、内容が確認できる

---

### 【短期】ゴールB: "論文の要素"で比較できる出力を揃える

#### タスク3: REPORT.mdでのp-box/共分散条件の明記
**目的**: `REPORT.md` または `diagnostics_tables/` にp-box要約と共分散条件を明記

**確認項目**:
1. **p-box出力の確認**
   - `make_report.py` の `_plot_pbox_assets` 関数が正しく動作しているか
   - p-box図が `report_assets/` に生成されているか
   - REPORT.mdにp-box図が参照されているか

2. **共分散条件の記載**
   - 使用した `rho` 値の記録
   - 対角/相関の切り替え条件の明記
   - `likelihood_meta_*.json` に共分散情報が含まれているか

**手順**:
```bash
# 1. 既存のrunでp-boxが生成されているか確認
ls tmcmc/_runs/<run_id>/report_assets/*pbox*.png

# 2. REPORT.mdの内容を確認
cat tmcmc/_runs/<run_id>/REPORT.md | grep -i "p-box\|covariance\|rho"

# 3. likelihood_metaを確認
cat tmcmc/_runs/<run_id>/likelihood_meta_M1.json | jq '.covariance'
```

**成果物**:
- REPORT.mdにp-box要約セクションが追加されている
- 共分散条件が明記されている
- 代表runで「論文と同じ定義での比較図/表」が最低1つ作れる

**DoD**:
- [ ] REPORT.mdにp-box要約が記載されている
- [ ] 共分散条件（rho値等）が明記されている
- [ ] 代表runで比較図/表が確認できる

---

#### タスク4: 論文に沿った比較指標の確認
**目的**: βスケジュール、ESS、ROM誤差、MAP/mean fitがREPORT.mdで確認できる

**確認項目**:
- [ ] βスケジュールの可視化（`diagnostics_tables/*_beta_schedule.csv`）
- [ ] ESSの推移
- [ ] ROM誤差の推移
- [ ] MAP/mean fitの比較図

**手順**:
```bash
# 1. 診断テーブルの存在確認
ls tmcmc/_runs/<run_id>/diagnostics_tables/

# 2. REPORT.mdの内容確認
cat tmcmc/_runs/<run_id>/REPORT.md

# 3. 図の生成確認
ls tmcmc/_runs/<run_id>/figures/
```

**DoD**:
- [ ] すべての比較指標がREPORT.mdで確認できる
- [ ] 図表が適切に参照されている

---

### 【Phase 1】論文との差分を埋める（既に実装済みの確認）

#### タスク5: p-box実装の動作確認と改善
**目的**: 既存のp-box実装が正しく動作し、論文の定義に合致しているか確認

**確認項目**:
1. **p-boxの定義確認**
   - 論文でのp-box定義（credible interval vs min-max）
   - 現在の実装（`pbox.compute_pbox_bounds`）が適切か

2. **可視化の改善**
   - prior/posteriorの比較が明確か
   - 更新前後の比較図が生成されているか

**手順**:
```python
# pbox.pyの実装を確認
# - compute_pbox_bounds: min-max vs quantile
# - plot_pbox_comparison: prior/posterior/true/MAPの可視化
```

**DoD**:
- [ ] p-box定義が論文と一致している
- [ ] 更新前後の比較図が生成できる

---

#### タスク6: 共分散実装の動作確認
**目的**: 観測共分散（rho）の実装が正しく動作しているか確認

**確認項目**:
1. **尤度計算の確認**
   - `log_likelihood_sparse` の `rho` パラメータが正しく機能しているか
   - 対角/相関の切り替えが正しく動作しているか

2. **設定の記録**
   - `config.json` に `rho` 値が記録されているか
   - `likelihood_meta_*.json` に共分散情報が含まれているか

**手順**:
```bash
# 1. rho=0.5で実行して動作確認
python tmcmc/run_pipeline.py \
  --mode paper \
  --lock-paper-conditions \
  --seed 123 \
  --run-id verify_rho_05 \
  --models M1 \
  --rho 0.5

# 2. 結果を確認
cat tmcmc/_runs/verify_rho_05/config.json | jq '.rho'
cat tmcmc/_runs/verify_rho_05/likelihood_meta_M1.json | jq '.covariance'
```

**DoD**:
- [ ] rhoパラメータが正しく機能している
- [ ] 共分散情報が適切に記録されている

---

## 📅 推奨実行順序

### 今週中（最優先）
1. **タスク1**: 論文条件固定runの実行と確定
2. **タスク2**: LaTeX資料の最新実装への同期

### 来週中（短期）
3. **タスク3**: REPORT.mdでのp-box/共分散条件の明記
4. **タスク4**: 論文に沿った比較指標の確認

### 余裕があれば（Phase 1確認）
5. **タスク5**: p-box実装の動作確認と改善
6. **タスク6**: 共分散実装の動作確認

---

## 🔍 次のステップ（Phase 2以降）

### Phase 2: 2次TSMの実装（短期〜中期）
- `tmcmc/analytical_derivatives_jit.py`: ∂²G/∂θ²（2次微分）の実装
- `tmcmc/demo_analytical_tsm_with_linearization_jit.py`: 2次TSM-ROMへの拡張

### Phase 3: 線形化点更新の自動化強化（中期）
- 誤差指標ベースの更新条件
- 更新後の再評価コスト削減

---

## 📝 メモ

- 現在のbest-run: `m1_check_np100_ns15` (docs/BEST_RUN.txt)
- 論文条件固定runの実行例は `PLAN.md` の89-90行目を参照
- p-box実装は `tmcmc/pbox.py` と `tmcmc/make_report.py` の `_plot_pbox_assets` を参照
- 共分散実装は `tmcmc/core/evaluator.py` の `log_likelihood_sparse` を参照
