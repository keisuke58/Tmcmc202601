# クラウド側 → GPU 側（2026-10-13a）: 2 通路の組成の尤度を用意した（**まだ投入しない**）

**今は投げない。** CPU のパイロット（`tools/two_channel_pilot.py`）の結果を見て、ユーザーが投入するか決める。本線（DS 予備 ult・DH の gate）と回収テストはそのまま。

## 何が変わるか（既定は今までと同じ）

`estimate_paper_jax.py --two-channel`（job では `TWO_CHANNEL=1`）:
- ch1: 生きた菌の組成（PMA、Fig. 3）を **φψ/Σφψ** と比べる（今までは φ/Σφ）
- ch2: 全菌の組成（Fig. S1）を **φ/Σφ** と比べる（重み `LAMBDA_CH2_2C`、既定 1.0）
- pilot・p1（`--fix-psi`）: ψ を全種同じ値ではなく、**データから出した種ごとの値**に固定（`species_viability_from_data`：生きた菌の割合 ÷ 全菌の割合 × 群集の生きた割合、全菌 < 2% の種は群集の値）
- p2・ult: ψ 自由、ch3（群集の生きた割合）は今までどおり

なぜ: 今の尤度は生きた菌のデータを φ と比べ「生存率は種によらない」と仮定している。データから出した種ごとの生存率（補足 `sec:S_psi`）では commensal の Fn・Pg で崩れ、モデルの ψ_i は 0/1 に張り付く。3 月の ch2 は ch1 と同じ φ と比べていたので「衝突」した（EML_PLAN §18〜19）。

## 確認済み（クラウド、CPU）

`python3 tools/test_two_channel.py` → ALL OK:
- 既定の経路: CH の MAP で logL = −8.6462（run の max logL と一致）
- 2 通路（Phase 2）の値が simulate_0d_full から手で組んだ値と一致（−22.3235）
- 種ごとの ψ（形 (n_obs, 5)）で値が変わる。一様な ψ を (n_obs, 5) で渡すと 1 次元と同じ

`tools/test_paper_pipeline.py` も既定の経路で通ることを確認してから push した。

## 投入するとき（ユーザーの指示が出たら）

RUNTAG に `_2c` を付けて今までの run と混ぜない。手順は今までと同じ（pilot → p1 → p2 → ult、各 3 seed、判定 (a)–(e)）。例:
```
qsub -l nodes=1:ppn=1:gpus=1:<node> -l walltime=03:00:00 \
     -v STAGE=pilot,TAG=CH,SEED=42,TWO_CHANNEL=1,RUNTAG=2c paper_gateoff_job.sh
```
- 箱は今の最終段の箱から始めてよいか、論文の箱からか → 投入の指示のときに決める。
- 最初は CH（いちばん速く通った条件）で 1 系列だけ回し、判定が通るか・ψ_i がデータに近づくかを見る。
