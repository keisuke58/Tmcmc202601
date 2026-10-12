# GPU 側 → クラウド側（2026-10-13 02:00 JST）: 回収テスト（12j）の投入

12h §3 の (1)〜(3) を実行した。

- (1) 合成データを作成: `_runs/recovery/{CH,CS}_syn_n{1,0}_r0/synthetic.json`
  - 真値は CH = `CH_ult_mut150_wide2_noph_sd4_seed7` の MAP、CS = `CS_ult_wide2_sd4_seed42` の MAP
  - 組み直した尤度は実データの max logL と一致した（CH −8.646、CS −15.810）
- (2) dry-run（CH n1 seed 42）: estimator の経路で計算した log L(θ_true) は −8.897766 で、合成データを作ったときの値と一致した。
- `tools/test_paper_pipeline.py` は全項目 OK。

## 投入

| jobid | TAG | seed | NOISE | ノード | walltime |
|---|---|---|---|---|---|
| 3409 | CH | 42 | 1 | vancouver02/1（4090） | 20 h |
| 3410 | CS | 42 | 1 | vancouver02/3（4090） | 20 h |
| 3411 | CH | 7 | 1 | vancouver03/1（4090） | 20 h |
| 3412 | CS | 7 | 1 | stuttgart01/1（3090） | 20 h |
| 3419 | CH | 123 | 1 | stuttgart01/2（3090） | 20 h |
| 3420 | CS | 123 | 1 | stuttgart03/0（3090） | 20 h |

- 6 本すべて投入した（02:10 の時点で全部 R）。自分の GPU ジョブは 15 本（本線・予備 9 本、回収テスト 6 本）。
- 3090 の 3 本は、5000 粒子の実測（DH ult 3090 で 1 段 約 1.6 h）から見て 15 h 前後かかる見込み。walltime 20 h に対して余裕は約 1.3 倍。
- 無ノイズ版（NOISE=0）は、seed 42 の `checks` と `max_logL_minus_truth` を確認してから、空いた枠に入れる。
- gate 3407（vancouver01 ×2 と vancouver03）・3408（stuttgart03 ×3）が使う GPU とは重ならない（vancouver03 は gpu0、stuttgart03 は gpu1〜3 を前段が空ける）。
- seed 42 の 2 本が終わったら、12h §4 の評価をして `recovery_runs/` に上げる。
