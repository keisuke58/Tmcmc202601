# 論文に使う run の粒子（2026-10-11j）

元は GPU 機の `data_5species/main/_runs/paper_gateoff/<run 名>/`。各 run から次の 5 つだけコピーした:
`samples.npy  logL.npy  theta_MAP.json  run_record.json  config.json`

図・表はこのフォルダを root にして作り直せる（例: `python3 tools/make_paper_figures.py docs/revision/generated/final_runs --p2-glob-CS ... --out-dir ...`、`tools/make_supplementary_tables.py docs/revision/generated/final_runs`）。

## 群ごとの判定と用途

判定は 2026-10-11 にこのフォルダに対して `tools/check_paper_runs.py` を流した結果（その日の規則、判定 4 は 10-10b の新規則）。

| 群 | 条件 | 段 | 用途 | 判定（2026-10-11） | 仮 |
|---|---|---|---|---|---|
| `CS_ult_wide2_sd4_seed*` | CS | ult | 図・表（最終） | 全 PASS | |
| `CH_ult_mut150_wide2_noph_sd4_seed*` | CH | ult | 図・表（最終） | 全 PASS | |
| `DS_p2_wide80_p18k_wide2_seed*` | DS | p2 | 図・表 | 全 PASS | **仮**（ult `DS_ult_wide80_p18k_wide2_sd4` が走行中、3371-3373） |
| `DH_p2_nonarrow_w3_noph_p14k_seed*` | DH | p2 | 図・表 | **判定 4 FAIL** | **仮**（ult は未投入。判定を通る群が出たら置き換え） |
| `CS_p1_mut120_seed*` | CS | p1 | Phase 比較の図 | 全 PASS | |
| `CH_p1_mut150_seed*` | CH | p1 | Phase 比較の図 | 全 PASS | |
| `DS_p1_wide80_p3k_seed*` | DS | p1 | Phase 比較の図 | 全 PASS | |
| `DH_p1_mut80_seed*` | DH | p1 | Phase 比較の図 | 全 PASS | |
| `CS_pilot_mut80_seed*` | CS | pilot | 補足 S3（full） | 2b FAIL | |
| `CH_pilot_mut80_seed*` | CH | pilot | 補足 S3（full） | 2b FAIL | |
| `DS_pilot_wide80_seed*` | DS | pilot | 補足 S3（full） | 4 FAIL | |
| `DH_pilot_mut80_seed*` | DH | pilot | 補足 S3（full） | 全 PASS | |
| `CS_pilot_a45zero_m120_seed*` | CS | pilot | 補足 S3（a45=0） | 全 PASS | |
| `CH_pilot_a45zero_m120_seed7` | CH | pilot | 補足 S3（a45=0） | 3・4 FAIL（**seed7 しか無い**、seed42/123 は GPU 機にも無い） | |
| `DS_pilot_a45zero_seed*` | DS | pilot | 補足 S3（a45=0） | 全 PASS | |
| `DH_pilot_a45zero_seed*` | DH | pilot | 補足 S3（a45=0） | 全 PASS | |

S3 の群は `make_supplementary_tables.py` の `DEFAULT_SPEC`（pilot / a45zero）に合わせた。

## run ごと

粒子数 × 次元は `samples.npy` の形。完了日は GPU 機の `run_record.json` の更新日。

| run | 条件 | 段 | 粒子 × 次元 | max logL | ln Z | 完了日 |
|---|---|---|---|---|---|---|
| `CH_p1_mut150_seed123` | CH | p1 | 2000 × 20 | -3.15 | -6.72 | 2026-10-08 |
| `CH_p1_mut150_seed42` | CH | p1 | 2000 × 20 | -3.23 | -6.91 | 2026-10-08 |
| `CH_p1_mut150_seed7` | CH | p1 | 2000 × 20 | -3.17 | -7.08 | 2026-10-08 |
| `CH_pilot_a45zero_m120_seed7` | CH | pilot | 1000 × 20 | -3.25 | -9.66 | 2026-10-08 |
| `CH_pilot_mut80_seed123` | CH | pilot | 1000 × 20 | -3.26 | -9.66 | 2026-10-06 |
| `CH_pilot_mut80_seed42` | CH | pilot | 1000 × 20 | -3.29 | -9.71 | 2026-10-06 |
| `CH_pilot_mut80_seed7` | CH | pilot | 1000 × 20 | -3.19 | -9.47 | 2026-10-06 |
| `CH_ult_mut150_wide2_noph_sd4_seed123` | CH | ult | 5000 × 20 | -8.68 | -18.34 | 2026-10-10 |
| `CH_ult_mut150_wide2_noph_sd4_seed42` | CH | ult | 5000 × 20 | -8.76 | -18.61 | 2026-10-10 |
| `CH_ult_mut150_wide2_noph_sd4_seed7` | CH | ult | 5000 × 20 | -8.65 | -18.20 | 2026-10-10 |
| `CS_p1_mut120_seed123` | CS | p1 | 2000 × 20 | -5.67 | -12.90 | 2026-10-07 |
| `CS_p1_mut120_seed42` | CS | p1 | 2000 × 20 | -5.76 | -12.64 | 2026-10-07 |
| `CS_p1_mut120_seed7` | CS | p1 | 2000 × 20 | -5.93 | -12.37 | 2026-10-07 |
| `CS_pilot_a45zero_m120_seed123` | CS | pilot | 1000 × 20 | -5.87 | -15.37 | 2026-10-09 |
| `CS_pilot_a45zero_m120_seed42` | CS | pilot | 1000 × 20 | -5.76 | -15.30 | 2026-10-09 |
| `CS_pilot_a45zero_m120_seed7` | CS | pilot | 1000 × 20 | -5.62 | -15.10 | 2026-10-09 |
| `CS_pilot_mut80_seed123` | CS | pilot | 1000 × 20 | -5.86 | -15.21 | 2026-10-06 |
| `CS_pilot_mut80_seed42` | CS | pilot | 1000 × 20 | -5.97 | -15.21 | 2026-10-06 |
| `CS_pilot_mut80_seed7` | CS | pilot | 1000 × 20 | -5.97 | -15.06 | 2026-10-06 |
| `CS_ult_wide2_sd4_seed123` | CS | ult | 5000 × 20 | -16.05 | -25.65 | 2026-10-09 |
| `CS_ult_wide2_sd4_seed42` | CS | ult | 5000 × 20 | -15.81 | -25.93 | 2026-10-11 |
| `CS_ult_wide2_sd4_seed7` | CS | ult | 5000 × 20 | -15.97 | -25.52 | 2026-10-11 |
| `DH_p1_mut80_seed123` | DH | p1 | 2000 × 20 | -3.21 | -17.74 | 2026-10-07 |
| `DH_p1_mut80_seed42` | DH | p1 | 2000 × 20 | -3.43 | -17.36 | 2026-10-07 |
| `DH_p1_mut80_seed7` | DH | p1 | 2000 × 20 | -3.52 | -18.06 | 2026-10-07 |
| `DH_p2_nonarrow_w3_noph_p14k_seed123` | DH | p2 | 14000 × 20 | -14.65 | -40.32 | 2026-10-11 |
| `DH_p2_nonarrow_w3_noph_p14k_seed42` | DH | p2 | 14000 × 20 | -14.81 | -40.42 | 2026-10-11 |
| `DH_p2_nonarrow_w3_noph_p14k_seed7` | DH | p2 | 14000 × 20 | -14.67 | -41.53 | 2026-10-11 |
| `DH_pilot_a45zero_seed123` | DH | pilot | 1000 × 20 | -4.15 | -23.62 | 2026-10-08 |
| `DH_pilot_a45zero_seed42` | DH | pilot | 1000 × 20 | -3.90 | -23.32 | 2026-10-08 |
| `DH_pilot_a45zero_seed7` | DH | pilot | 1000 × 20 | -3.39 | -22.95 | 2026-10-08 |
| `DH_pilot_mut80_seed123` | DH | pilot | 1000 × 20 | -4.32 | -20.89 | 2026-10-06 |
| `DH_pilot_mut80_seed42` | DH | pilot | 1000 × 20 | -4.44 | -21.64 | 2026-10-05 |
| `DH_pilot_mut80_seed7` | DH | pilot | 1000 × 20 | -3.92 | -21.54 | 2026-10-06 |
| `DS_p1_wide80_p3k_seed123` | DS | p1 | 3000 × 20 | -0.85 | -16.60 | 2026-10-08 |
| `DS_p1_wide80_p3k_seed42` | DS | p1 | 3000 × 20 | -0.64 | -16.68 | 2026-10-08 |
| `DS_p1_wide80_p3k_seed7` | DS | p1 | 3000 × 20 | -0.58 | -16.70 | 2026-10-08 |
| `DS_p2_wide80_p18k_wide2_seed123` | DS | p2 | 18000 × 20 | -6.45 | -30.96 | 2026-10-11 |
| `DS_p2_wide80_p18k_wide2_seed42` | DS | p2 | 18000 × 20 | -6.99 | -31.59 | 2026-10-11 |
| `DS_p2_wide80_p18k_wide2_seed7` | DS | p2 | 18000 × 20 | -6.73 | -31.63 | 2026-10-11 |
| `DS_pilot_a45zero_seed123` | DS | pilot | 1000 × 20 | -1.51 | -18.99 | 2026-10-08 |
| `DS_pilot_a45zero_seed42` | DS | pilot | 1000 × 20 | -1.51 | -19.22 | 2026-10-08 |
| `DS_pilot_a45zero_seed7` | DS | pilot | 1000 × 20 | -1.50 | -18.21 | 2026-10-08 |
| `DS_pilot_wide80_seed123` | DS | pilot | 1000 × 20 | -0.91 | -20.89 | 2026-10-08 |
| `DS_pilot_wide80_seed42` | DS | pilot | 1000 × 20 | -1.63 | -22.30 | 2026-10-08 |
| `DS_pilot_wide80_seed7` | DS | pilot | 1000 × 20 | -1.49 | -21.25 | 2026-10-08 |
