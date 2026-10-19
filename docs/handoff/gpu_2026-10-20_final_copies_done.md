# GPU 側 → クラウド側: 19c の最後のコピー完了（2026-10-20）

1. **CH の 1 日外し p2 を final_runs に**（6 ファイル、polish.json 込み）
   - `CH_p2_2c_raw_a33w_ho10_seed42`: 判定 0〜2b PASS（移動 10.9 回/次元）、磨き −21.497 → −20.487（575 反復・収束）
   - `CH_p2_2c_raw_a33w_ho15_seed42`: 判定 0〜2b PASS（移動 11.1 回/次元）、磨き −21.300 → −20.319（484 反復・収束）
   - 判定 3・4 の FAIL は 1 seed のため（19c の想定どおり）。check の「polish が未収束（0/1 seed）」は seed 数が 2 未満のときの文言で、polish.json は converged=True。
   - → クラウドで `holdout_eval.py`（Day 10・15）をお願いします。
2. **DH の回収テスト**は recovery/ に済み（seed 42・7・123、npy 込み、git 管理下を確認）。
3. 予備の磨き 3682・3683（`_bk`、同じ設定の重複）は本体 3680・3681 が収束済みのため qdel。走行中のジョブは 0 本。
4. `_runs/paper_gateoff/` と `_runs/recovery/` は消さずに残してある。
