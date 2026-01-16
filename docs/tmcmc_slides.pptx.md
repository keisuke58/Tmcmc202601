# TMCMC×TSM-ROM（線形化管理 + 解析微分/JIT）

- 入口: `tmcmc/case2_tmcmc_linearization.py`
- 外枠: `tmcmc/run_pipeline.py` → `tmcmc/make_report.py`

---

## 何を説明する資料か

- 実行フロー（どの順で何が呼ばれるか）
- 主要モジュール境界（どこを見れば挙動が変わるか）
- 性能の支配要因（どこが重いか）
- 精度の支配要因（何が推定を決めるか）
- 再現性・監査ログ（何が残るか）

---

## 外枠（run_pipeline）

- `run_pipeline.py` が run_dir を作る
- subprocess で `case2_tmcmc_linearization.py` を実行
- その後 `make_report.py` で `REPORT.md` を生成
- ログは `subprocess.log / pipeline.log` に永続化

---

## 主要モジュール（実行に効くセット）

- 実験制御: `case2_tmcmc_linearization.py`
- 設定: `config.py`
- TSM（線形化点管理 + 解析微分/JIT）: `demo_analytical_tsm_with_linearization_jit.py`
- 物理モデル（Newton + 時間積分）: `improved1207_paper_jit.py`
- 解析微分（paper mode）: `paper_analytical_derivatives.py`
- 診断/レポート: `mcmc_diagnostics.py`, `make_report.py`

---

## TMCMC（β tempering）の要点

- prior（β=0）→ posterior（β=1）へ段階的遷移
- ESS 目標（target_ess_ratio）に基づき Δβ を調整（min/maxあり）
- 重み更新→ESS→リサンプル→mutation（MCMC）で混合を維持

**重要**: ログに「β reached 1.0」が出るか（posterior到達チェック）

---

## TSM-ROM（線形化管理）の要点

\[
x(\theta) \approx x(\theta_0) + \frac{\partial x}{\partial\theta}\Big|_{\theta_0}(\theta-\theta_0)
\]

- 探索初期（小β）: 線形化OFF（非線形TSM）
- 後半（大β）: 線形化ON（高速・MAP近傍で高精度）
- `update_linearization_point(θ0)` でキャッシュ無効化→再計算

---

## 物理モデル（Newton + 時間積分）

- `BiofilmNewtonSolver.run_deterministic()` が決定論的軌道の中核
- 残差 `Q`（compute_Q_vector）とヤコビアン `J`（compute_Jacobian_matrix）で Newton
- 抗生物質の時間依存: `alpha_schedule`（switch_time / switch_step / switch_frac）

---

## 解析微分（paper mode）

- `paper_analytical_derivatives.py` が ∂G/∂θ を paper-consistent に実装
- complex-step 参照で検証可能（verify関数）
- 前提: θ→(A,b) が複素dtypeを保持（complex-step整合）

---

## 再現性（監査ログ）

- `config.json`: seed / mode / TMCMC設定 / モデル設定
- `likelihood_meta_*.json`: 尤度定義（var_total 等の内訳）
- `diagnostics_tables/*.csv`: β, 受理率, ROM error, θ0履歴
- `REPORT.md`: PASS/WARN/FAIL（しきい値で整理）

---

## 性能ボトルネック（経験則）

計算時間 ≈（尤度評価回数）×（TSM1回のコスト）

- 最大: `BiofilmTSM_Analytical.solve_tsm()`
- 最大: `BiofilmNewtonSolver.run_deterministic()`（Q/J + Newton）
- 大: 感度 x^(1) 生成（線形化が効かない時期に重い）
- 中: TMCMC枠（β更新/リサンプル/mutation）
- 小: 可視化・I/O（条件次第で増える）

---

## 精度への寄与（重要度）

- 最大: 尤度定義（sigma_obs、Var(φψ)にCovを入れる/入れない）
- 大: TSM妥当性（ROM error、線形化点管理、解析微分）
- 大: 線形化ONタイミング／更新規則（早すぎると壊れる、遅いと遅い）
- 中: 数値安定化（dt, Newton許容, クリップ/ペナルティ）
- 中: TMCMC設定（粒子数、stage数、mutation steps）

---

## 図のネタ（そのまま発表に使える）

- βスケジュール（チェーン別）
- ROM error（更新イベント pre/post）
- θ0更新の ||Δθ0||（安定性）
- posterior（M1/M2/M3/M3_val）
- Cost–Accuracy tradeoff（FOM回数or wall-time vs MAP error）

---

## まとめ

- 外枠: `run_pipeline` → `case2` → `make_report`
- 中核コスト: `solve_tsm` と Newton 時間積分
- 監査の肝: **β=1到達** と **尤度定義の明文化（likelihood_meta）**

