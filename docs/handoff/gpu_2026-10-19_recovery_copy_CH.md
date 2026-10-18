# GPU 側報告 2026-10-19（18b: CH の回収テストと polish.json をコピー、DS a45=0 seed123 の磨きを出し直し）

## 1. 回収テスト CH を `docs/revision/generated/recovery/` に（18b）
- `CH_syn_n1_r0/synthetic.json` と `CH_syn_n1_r0_seed{42,7,123}/`（samples・logL・theta_MAP・run_record・config・recovery_record の 6 ファイル）。
- **名前は 18b の形 `CH_syn_n1_r0_seed<N>` に付け替えた**。元の run 名は `_runs/recovery/CH_syn_n1_r0_2c_raw_seed<N>`（3614〜3616）。
- `synthetic.json` に実測の反復値が無いことを確認した: 中身は真値 θ・箱・σ（集計値）と、モデル出力＋ノイズの日ごと 5 点（composition・total・viability、noise_free 版も）だけ。
- 置いた場所で `tools/recovery_evaluate.py --case CH=docs/revision/generated/recovery/CH_syn_n1_r0/synthetic.json='docs/revision/generated/recovery/CH_syn_n1_r0_seed*'` が通ることを確認（出力は commit していない）:
  被覆 0.53（15 成分）、(a) 実データで同定 6 → 区間に真値 2・合成でも同定 6（最大誤差 0.344×箱）、(b) n.i. 9 → 合成でも 9、分類一致 15/15。
  max logL −18.98 / −19.03 / −19.00（seed 123・42・7）、log L(θ_true) −20.95。

## 2. polish.json を final_runs に（18b ついで）
- `CH_p1_2c_raw_seed42`（10/17 06:41）・`CH_p2_2c_raw_a33w_seed7`（10/17 04:50）・`DS_ult_2c_raw_seed7`・`DS_ult_2c_raw_seed123`。
- DS ult seed42 は磨き 3662（stuttgart01、16h）がまだ走っている。終わり次第コピーする → そのあとクラウドで DS.json を差し替え。

## 3. DS a45=0 seed123 の磨き（18a §3）
- 3651 は 4090 で 8h の walltime 切れ（Exit_status −11、10/18 21:52）。失敗時の再投入 3665（vancouver02・14h）は、vancouver02 が回収テストと他ユーザーの放置 jupyter で埋まっていて動けなかった。
- 3665 を qdel し、**3666 として stuttgart02（3090）・walltime 30h** で同じ設定（`RUN_GLOB=DS_pilot_2c_raw_a45zero_m120_seed123 TOP_K=3 MAXITER=3000 WRITE_RUN=1`）で出し直した（10/19 投入、走行中）。

## 4. 残り
- CS（3659〜3661）・DS（3653〜3655）・DH（3656〜3658）の回収テストは終わり次第、同じ形でコピー。
- GPU 11 本（回収 9、磨き 3662・3666）。
