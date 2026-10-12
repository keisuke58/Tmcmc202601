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

- 残り 4 本（CH・CS の seed 7・123）と無ノイズ版はまだ投げていない。
  - 自分の GPU ジョブはこれで 13 本。DH 仮 ult の 3388・3390（10-13 の 02:07 ごろに walltime）が終わると枠が 2 つ空く。
- seed 42 の 2 本が終わったら、12h §4 の評価をして `recovery_runs/` に上げる。
