# 論文パイプラインでゲート OFF を回し直す — GPU 側への手順書

作成 2026-10-02（クラウド側）。**これより前の GPU の run はすべて無効**（§1）。

---

## 0. 先に結論

- 9/30 までの GPU の run（50 粒子・1000 粒子 ×6・旧データ 25 本・5000 粒子 24 本）は**すべて無効**。
- 論文の MAP を作ったコード（2026-03-20, `ba1c285`）を **固有名のファイル**として復元した。
  これで**論文の 4 条件の RMSE を完全に再現**できる（CS 0.1190 / CH 0.1040 / DS 0.0327 / DH 0.0868）。
- 回すのは `paper_gateoff_job.sh`。**論文と同じ手順で、ゲートだけ外す**（K_hill=0、b=0 固定）。
- 結果は、解釈する前に必ず `tools/check_paper_runs.py` で収束を判定する。

---

## 1. 何が壊れていたか（2つの独立した不具合）

| | 内容 | 影響 |
|---|---|---|
| ① θ の並び | 2026-03-30 の `726afa7` で `data_5species/main/hamilton_ode_jax.py` だけが 15 次元（A のみ、θ[13]=a35, θ[14]=a45）になり、20 次元の θ を渡す `estimate_reduced_nishioka_jax.py` と食い違った | **θ[15..19]（b5, a15, a25, a35, a45）は前進モデルに読まれない**。a35/a45 の事後は事前分布のまま。b として 0 固定した θ[3,4,8,9] は実は a33/a34/a23/a24 |
| ② ゲート OFF | `data_5species/main/hamilton_ode_jax.py:130-134` は `K_hill=0` で Pg の相互作用行を丸ごと 0 倍する（`f25131e` は colab_package 側しか直していなかった） | 「ゲート OFF」ではなく「Pg に相互作用が無い」モデル |
| ③ 取り違え | 前進モデル・エンジン・ローダーが同名の別ファイルで、`sys.path` の順で暗黙に選ばれていた | run（main 側）と評価（`eval_gateoff_runs.py` は colab_package 側）が別モデル → 保存 logL と粒子が対応しない |

copaam 側の検証（4月版 ODE で再計算すると保存 logL と max|diff| 1e-12 で一致）は ① と ② で説明できる。
**エンジン（RW 分岐・init）にバグは無い。**

さらに、論文の2段階推定（多チャネル尤度・ψ 固定・ウォームスタート・事前分布の絞り込み）は
2026-03-22 の `f7d9ca3`（6 種ソルバー追加のコミット）で estimator から丸ごと消えていた。
同じコミットで前進モデルの出力も φ → φ̄=φ·ψ に変わっている。**論文の MAP は φ 版で作られている。**

---

## 2. 新しく入ったもの

| ファイル | 中身 |
|---|---|
| `data_5species/main/hamilton_ode_jax_paper.py` | ba1c285 の前進モデル（20 次元・φ を返す）＋ゲート OFF 修正だけ |
| `data_5species/main/tmcmc_engine_paper.py` | ba1c285 のエンジン（ウォームスタート・DE-MC 付き）＋ evidence 補正・事前分布の配線 |
| `data_5species/main/estimate_paper_jax.py` | ba1c285 の estimator ＋ `--estimate-b`（既定 OFF）・`--prior-scale`・`--box`・起動時の感度チェック・実効値の全記録 |
| `data_5species/model_config/prior_bounds_paper_20d.json` | ba1c285 時点の 20 次元の箱（今の `prior_bounds.json` は 15 次元用に並べ替え済みなので使わない） |
| `data_5species/main/paper_gateoff_job.sh` | PBS ジョブ（pilot / p1 / p2 / ult / ident） |
| `tools/test_paper_pipeline.py` | **estimator が実際に読み込むモジュール**に対する検証 8 項目 |
| `tools/check_paper_runs.py` | 収束の判定（解釈の前に必ず通す） |

既存の `estimate_reduced_nishioka_jax.py`・`hamilton_ode_jax.py`・`tmcmc_nuts_engine.py` には**触っていない**
（Siddiqui 6 種などほかの解析が使っている）。**論文の再推定には使わないこと。**
`dh_prior_check_job.sh` と `eval_gateoff_runs.py` も論文用には使わない。

### 起動時の感度チェック

estimator は JIT のあと、箱の中点から各自由次元を箱幅の 1/4 動かし、尤度が変わらない次元が
あれば **RuntimeError で止まる**。今回の事故（15 次元モデルに 20 次元 θ）を入れると
`a15, a25, a35, a45` を名指しして止まることをテスト [6] で確認済み。

---

## 3. 投入前に必ずやること

```bash
cd ~/Tmcmc202601 && git pull          # claude/gate-off-map-check
python3 tools/test_paper_pipeline.py  # 「すべて OK」を確認（CPU で数分、pandas/numba が要る）
```

8 項目のうち 1 つでも FAIL なら**投入しない**。特に [4]（論文の RMSE の再現）が通らないなら、
環境（JAX のバージョン・float64）が論文時点と違う。

---

## 4. 回すもの（2 本立て）

### A. 論文の再現（ゲートだけ外す）

論文と同じ 4 段の連鎖。前段の出力を `PREV` に渡す。**同じ seed の連鎖**で 3 seed（42 / 7 / 123）。

| 段 | 内容 | 粒子 | mutation | 事前分布 | walltime 目安 |
|---|---|---|---|---|---|
| pilot | ψ 固定・論文の箱 | 1000 | 40 | 論文の箱 | 2h |
| p1 | ψ 固定 | 2000 | 40 | pilot の MAP ± 4σ | 3h |
| p2 | ψ 自由・多チャネル（ch1 1.0 / ch3 2.0 / ch5 0.3 HOBIC のみ） | 2000 | 40 | p1 の MAP ± 3σ | 4h |
| ult | 同上 | 10000 | 50 | p2 の MAP ± 2σ | 10h |

```bash
cd ~/Tmcmc202601/data_5species/main
qsub -l walltime=02:00:00 -v STAGE=pilot,TAG=DH,SEED=42 paper_gateoff_job.sh
# 終わったら（出力は _runs/paper_gateoff/DH_pilot_seed42）
qsub -l walltime=03:00:00 -v STAGE=p1,TAG=DH,SEED=42,PREV=_runs/paper_gateoff/DH_pilot_seed42 paper_gateoff_job.sh
qsub -l walltime=04:00:00 -v STAGE=p2,TAG=DH,SEED=42,PREV=_runs/paper_gateoff/DH_p1_seed42 paper_gateoff_job.sh
qsub -l walltime=10:00:00 -v STAGE=ult,TAG=DH,SEED=42,PREV=_runs/paper_gateoff/DH_p2_seed42 paper_gateoff_job.sh
```

4 条件 × 3 seed × 4 段 = 48 本。**同時 10 本まで**なので、段ごとに 4 条件 × 3 seed = 12 本を
2 バッチに分ける。**各段が終わるたびに §5 の判定を通し、PASS した群だけ次の段に進める。**

### B. 識別性の検証（ψ 固定・箱 [−15, 20]）

```bash
for S in 42 7 123; do for P in 0 6; do
  qsub -l walltime=03:00:00 -v STAGE=ident,TAG=DH,SEED=$S,PRIOR_SCALE=$P paper_gateoff_job.sh
done; done
```

DH の 6 本から始める。PASS したら他の条件にも広げる。

**順番の提案:** B の DH 6 本 ＋ A の pilot 4 条件 × seed 42 の 4 本 = 10 本を最初のバッチにする。

---

## 5. 解釈の前に必ず通す判定

```bash
python3 ~/Tmcmc202601/tools/check_paper_runs.py _runs/paper_gateoff
```

同じ TAG・段・事前分布の seed 違いを 1 群として判定する:

1. 全 run が beta=1 に到達
2. 平均受理率 0.10〜0.60
3. seed 間の max logL の幅 ≤ 1 nat
4. seed 間の各自由次元の中央値の幅 ≤ プールした事後 sd の 0.5 倍
5. （ident のみ）事前分布なしの max logL ≥ 事前分布ありの max logL − 0.5

**FAIL を含む群は解釈しない。** 粒子数・mutation 数を増やして回し直す。
9/30 の 1000 粒子 run は 3・4・5 をすべて落としていた（事前分布なしの max logL が事前分布ありより
10 nats 悪い ＝ 事前分布なしの探索が主要なモードに届いていない）。

---

## 6. 報告してほしいもの

- `check_paper_runs.py` の出力をそのまま（全群）
- ult の 4 条件: ch1 の RMSE（`config.json` の `rmse`）を論文の値（ゲートあり）と並べて
- DH: Pg の Day21/Day15（実測 2.74、論文 MAP ゲートあり 2.78、論文 MAP ゲートなし 0.90）
- B: a35・a45 の 5/50/95% 点と、箱の端 5% にいる粒子の割合（判定 PASS の群だけ）

---

## 7. 原稿と実装の食い違い（今回見つかったもの・原稿修正の材料）

論文の数値はこのコードで再現できるが、原稿の記述とは次の点が違う。

| | 実装（論文の数値を出したコード） | 原稿 |
|---|---|---|
| 尤度の重み | `lambda_pg=5`, `lambda_late=3`（最後の 2 時点を 3 倍）、`lambda_rare=0.1`（平均 5% 未満の種を 1/10） | 記載なし |
| 生存率チャネルの重み | スクリプトは `--lambda-ch3 2.0` だが、コードが「2.0 なら未指定」とみなして **DH では 3.0、Commensal（CS・CH）では 1.5** に変える。DS だけが 2.0 | — |
| Phase 1 の「ψ 固定」 | ψ は全種に同じ値を掛けるので正規化で消え、ch1 の尤度には効かない。ODE の中の ψ も自由に動く | 「ψ を実測に固定し、ヤコビアンの次元を半分にする」 |
| 事前分布の絞り込み | pilot → p1 → p2 → ult で箱を前段の MAP ± 4σ → 3σ → 2σ に絞る | §6.8 の `r_i = σ_post/Δ_prior < 0.43` はこの絞った箱に対する比 |
| 観測量 | ch1 は正規化した φ（コメントには φ·ψ とあるが実装は φ） | — |
| CS / CH の箱 | 論文の本番（ultimate_10000p）では 20 次元すべてが自由。`prior_bounds.json` の locks は効いておらず、locks の次元は default_bounds [−1, 1] で推定されていた（`prior_bounds_paper_20d.json` もこれに合わせて locks を外した） | 「全条件で 15 成分を固定なしで推定」— 実装と一致 |
| **§6.8 の識別性の基準** | 論文の本番で **CS の a55・a35 の事後は U(−1, 1) と区別できない**（sd 0.572 / 0.571、一様分布の sd 0.577、KS p = 0.06 / 0.28）。それでも r = sd/幅 = 0.29 で、基準 **r < 0.43 を満たす**。一様分布の r は常に 1/√12 ≈ 0.289 なので、**事後が事前とまったく同じでも「識別できている」と判定される** | 「全パラメータで r_i < 0.43 なので過剰パラメータではない」— **この基準では非識別を検出できない** |
