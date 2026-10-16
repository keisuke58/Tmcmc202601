# クラウド側 → GPU 側（2026-10-16i）: 旧い合成データの再利用を止めるガード（16h への対応）

- `tools/recovery_run.py` が、合成データの `source_glob` を「今回の元の run 群」と比べ、違えば止まる（exit 1、メッセージで退避を促す）。
  今回の元は `--expect-glob` → 環境変数 `SRC_GLOB` → `recovery_common.GROUPS` の既定、の順。確認しないときは `--allow-other-source`。
- `recovery_common.GROUPS` の既定を CS `CS_ult_2c_raw_seed*`・CH `CH_ult_2c_raw_a33w_seed*` に更新（DS・DH は ult が通ったら SRC_GLOB で渡す、今までどおり）。
- **`recovery_job.sh` は変えていない**（走っている 3614〜3616 が読んでいる bash を書き換えると、pull したときに途中から壊れるため）。
  Python は起動時に読み込み済みなので、3614〜3616 にこの変更は効かない（影響もない）。
- 確認: 旧い CH（`CH_ult_mut150_wide2_noph_sd4_seed*`）の合成データで止まる、`SRC_GLOB` を合わせると通る、をクラウドで確認した。
- CS・DS・DH の回収テストを出すときも、16h のとおり 10/13 の `*_syn_*` を退避してから作る（ガードが止めるので、止まったら退避すればよい）。
