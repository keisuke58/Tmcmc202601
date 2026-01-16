## 今後のプラン（最優先：論文再現・比較 / TMCMC × TSM-ROM / Biofilm）

このリポジトリは **TMCMC × TSM-ROM（線形化点更新あり）** によるベイズ推定パイプラインが中心で、実験→レポート生成までの導線がすでにあります（`tmcmc/run_pipeline.py`）。
ここでは **「論文（Bayesian updating…）に合わせた再現・比較」** を最優先に、差分（ギャップ）を潰す順で成果物ベースに整理します。

---

## 現状（入口と中核）

- **実験→レポート（1コマンド）**: `tmcmc/run_pipeline.py`
  - `tmcmc/case2_tmcmc_linearization.py` を実行し、`tmcmc/make_report.py` で `REPORT.md` を生成
- **ハイパラ探索（M1）**: `tmcmc/sweep_m1.sh`
  - グリッド実行 → `sweep_summary.csv` → best選定
- **best-run自動要約（docs）**: `docs/auto_pick_best_run.py`（`docs/auto_best_run/` を生成）
- **資料生成（LaTeX）**: `docs/build_pdfs.py` / `docs/tmcmc_flow_report*.tex` / `docs/tmcmc_flow_slides*.tex`
- **中核4ファイルの関係性整理**: `tmcmc/CORE_FILES_ANALYSIS.md`
- **論文PDF↔実装のギャップ整理**: `tmcmc/PDF_RELATIONSHIP.md`

---

## 直近ゴール（最短で「論文に近い比較」を作る）

### ゴールA: “再現条件”を固定し、比較の土台を作る（最優先）

- **成果物**
  - “論文条件固定”で生成した run を1つ確定し、`run_id` を資料に固定して参照できるようにする
  - `docs/tmcmc_flow_report(_en).pdf` / `docs/tmcmc_flow_slides(_en).pdf` の記述を最新実装に同期（入口コマンド・条件・出力ファイルを記載）
- **DoD（完了条件）**
  - “入口コマンド”と“再現run_id”と“固定した条件（ノイズ等）”が資料に記載されている
  - 図は `tmcmc/_runs/<run_id>/figures/*.png` を参照（埋め込み不要）

### ゴールB: “論文の要素”で比較できる出力を揃える（短期）

- **成果物**
  - 代表runについて、論文に沿った比較指標（例：\(\beta\) スケジュール、ESS、ROM誤差、MAP/mean fit）を `REPORT.md` で確認できる
  - best-run自動選定（`docs/auto_pick_best_run.py`）を「比較用の補助」として継続利用（主役は “条件固定run”）
- **DoD**
  - “条件固定run”で `python tmcmc/run_pipeline.py ...` が安定して PASS/WARN で終わる
  - 生成物が `tmcmc/_runs/<run_id>/REPORT.md` に残る

---

## 実装ロードマップ（優先度順）

### Phase 1（短期）: 論文との差分（p-box / 共分散）を埋める

- **やること**
  - **p-boxの表現/出力**: prior/posterior を「箱（区間）」として要約・可視化する出力を追加
    - 例：各パラメータの区間（credible intervalではなく “box” の定義に合わせる）と、更新前後の比較図
  - **観測共分散**: 対角だけでなく相関を入れられる尤度（設定で切替）を用意し、論文の仮定に合わせる
- **ねらい**
  - “論文にある概念（p-box / 共分散の扱い）で比較できる”状態にする
- **DoD**
  - `REPORT.md` または `diagnostics_tables/` に p-box要約と共分散条件が明記され、資料で引用できる
  - 代表runで「論文と同じ定義での比較図/表」が最低1つ作れる

### Phase 2（短期〜中期）: 精度の上限を上げる（2次TSM）

- **やること**
  - `tmcmc/analytical_derivatives_jit.py`: \(\partial^2 G/\partial\theta^2\)（2次微分）を実装
  - `tmcmc/demo_analytical_tsm_with_linearization_jit.py`: 2次TSM-ROMに拡張
- **ねらい**
  - posterior が線形化点から離れたときの ROM 誤差を抑える（論文比較時の破綻を避ける）
- **DoD**
  - FOM vs ROM の誤差曲線（θ距離に対する誤差）が改善し、`REPORT.md` に反映できる

### Phase 3（中期）: 線形化点更新の「自動化」を強くする（比較の安定化）

- **やること**
  - 線形化点更新条件を、固定間隔だけでなく **誤差指標ベース** にする（例: ROM誤差・尤度劣化・受理率低下）
  - 更新後の再評価コスト（logL再計算）を抑える工夫（キャッシュ/部分更新）
- **DoD**
  - “更新した方が良いケース”で ROM誤差が爆発せず、TMCMC が安定して \(\beta\to1\) に到達

### Phase 4（中期〜長期）: 高速化・スケール（必要になったら）

- **やること**
  - TMCMCの並列化（chains / particles）
  - GPU（CuPy/Numba CUDA）検討（まずは `analytical_derivatives_jit.py` のボトルネック計測から）

---

## 実験運用の型（おすすめ）

- **単発実験**:
  - `python tmcmc/run_pipeline.py --mode sanity`
  - `python tmcmc/run_pipeline.py --mode paper --seed 123 --run-id myrun`
- **論文比較（条件固定を優先）**:
  - `python tmcmc/run_pipeline.py --mode paper --lock-paper-conditions --use-paper-analytical --seed 123 --run-id paper_M1_seed123 --models M1`
  - Fig.8–15（Case II）狙い: `python tmcmc/run_pipeline.py --mode paper --models M1,M2,M3,M3_val --lock-paper-conditions --use-paper-analytical --seed 123 --run-id paper_caseII_seed123`
- **M1 sweep**:
  - `bash tmcmc/sweep_m1.sh`
- **best-run更新（docs）**:
  - `python docs/auto_pick_best_run.py`
- **PDF生成**:
  - `python3 docs/build_pdfs.py`

---

## 相談したい前提（ここが決まるとプランがさらに鋭くなる）

- **ターゲットモデル**: まずは M1 に集中するか、M2/M3（`tmcmc/config.py`）も同時に論文比較するか
- **“実データ”の投入予定**: あるなら、観測モデル（尤度）の設計を前倒しする

