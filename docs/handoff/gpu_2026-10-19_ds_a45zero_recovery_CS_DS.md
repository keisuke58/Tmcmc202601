# GPU 側報告 2026-10-19（夜）: DS a45=0 を確定、回収テスト CS・DS をコピー

## 1. DS a45 = 0 の確定（18a §3）
- 磨いた max logL（top-k 3、polish.json）: seed 42 / 7 / 123 = −57.89 / −57.78 / −57.78（幅 0.11）。seed123 は 3666 の出し直し。
- `check_paper_runs.py --glob 'DS_pilot_2c_raw_a45zero_m120_seed*'` → **判定 0〜4 全 PASS**（移動 13.9〜14.0 回/次元）。
- `final_runs/DS_pilot_2c_raw_a45zero_m120_seed{42,7,123}/` に 5 ファイル＋polish.json。
- ln Ẑ は磨きで変わらないので **DS の ln B = +7.46 で確定**（`gpu_2026-10-17_bayes_raw.md` を更新）。

## 2. 回収テスト CS・DS（18b・19a）
- `recovery/{CS,DS}_syn_n1_r0/synthetic.json` と `_seed{42,7,123}/` に 5 ファイル＋recovery_record.json（npy 12 個が git に入ることを確認）。
- 真値: CS = `CS_ult_2c_raw_seed123`、DS = `DS_ult_2c_raw_seed42`（どちらも粒子の最良値、生データ・2 通路）。
- synthetic.json のキーは CH と同じ（合成の composition / viability / total と noise_free、反復ごとの実測値なし）。

## 3. 走行中・残り
- 回収テスト DH: seed 42・123 は終了、**seed 7（3657、vancouver02）が走行中**。終わったら 3 seed まとめてコピー。
- 1 日外し CH（seed42）: 判定 3・4 は 1 seed なので出ない（19b の想定どおり）。
  - p1・p2: 判定 0〜2b PASS（p1 は 8.3 回/次元、p2 は 10.9 回/次元）。
  - pilot: 判定 2b 未達（2.5〜2.7 回/次元）。本線の CH pilot も 2.7〜2.8 回/次元で同じ（1000×40 の設定どおり）。
  - **pilot・p1 の 4 run は final_runs にコピー済み**（bfb6b29、追記 10/19 20:15）。
  - **p2 の磨き 3680・3681 が走行中**（WRITE_RUN=1、8h）。終わったら p2 の 2 run を polish.json 付きで final_runs に。
- 自動通知の「照合 NG」（RMSE が estimator と不一致）は、Day 10/15 を外した run を全日のデータで照合しているためと思われる（未確認）。
