# 論文に使う run の粒子 — 生データ・2 通路（2026-10-16〜）

このフォルダには **Heine の生データ（qPCR・LIVE/DEAD）で、2 通路の尤度で回した run だけ** を置く
（名前に `2c_raw` が入る）。各 run は `samples.npy  logL.npy  theta_MAP.json  run_record.json  config.json`。

- 群の名前: `figures/raw/spec_raw.json`（`tools/rebuild_eml_raw.sh` が読む）。
- 判定と最終 MAP の選び方: `docs/revision/generated/final_theta_MAP/README.md`、handoff `2026-10-16g`・`16j`。
- DH は `DH_p2_2c_raw_seed*` が最終（2026-10-16l）。`DH_ult_2c_raw_seed*` は判定 4 FAIL で、限界の節の数字の元として参考に置く（DH p2 には polish.json も入れた。λψ は run_record の `lambda_ch_effective` で 3.0）。
- DS は GPU で走行中。終わったらここに足す。

**論文の図から読み取ったデータ（digitised）で回した以前の run は `../final_runs_digitised/` に分けた**（2026-10-16、
混ざらないように）。そちらは `superseded` の MAP・旧版の図（`figures/provisional_2026-10-11/`・`figures/generated/`）の元で、
新しい図・表・数字には使わない。道具の既定の glob はすべて生データの群を指す（古い群を見るときは
`--root docs/revision/generated/final_runs_digitised` と群の名前を明示する）。
