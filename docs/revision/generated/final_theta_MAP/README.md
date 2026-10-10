# 最終 MAP（論文・修論・FEM 共通）

TMCMC 論文パイプライン（`data_5species/main/paper_gateoff_job.sh`、ゲートなし・pH は尤度に入れない λ_ch5=0）の
**最終段 (iv)（ult）**の MAP。各条件で判定（0〜4・2b・片側の端）を通った 3 seed のうち、**max logL が最大の seed** を採る。
FEM の要素へは `write_eco_cfg.py --theta-json <ここの json>` で渡す。

| 条件 | ファイル | 元の run | seed | max logL | RMSE（組成） | 状態 |
|---|---|---|---|---|---|---|
| CH | `CH.json` | `CH_ult_mut150_wide2_noph_sd4_seed7` | 7 | −8.65 | **0.0756**（3 seed で 0.0756〜0.0779） | **確定**（3 seed 全判定 PASS、2026-10-10）。json 配置済み（2026-10-10） |
| CS | `CS.json` | `CS_ult_wide2_sd4_seed*` | — | — | — | seed42・7 を回し直し中（10-11 朝〜昼） |
| DS | `DS.json` | `DS_ult_wide80_p18k_wide2_sd4_seed*` | — | — | — | p2（18000 粒子）走行中 |
| DH | `DH.json` | `DH_ult_nonarrow_w3_noph_p14k_sd4_seed*` | — | — | — | p2（a44 下限 −10）走行中 |

## CH の詳細

- 箱（OVERRIDE）: `1:-5:2.5;11:-5:1;16:-5:1;0:-5:1.5`（a12・a14・a15・a11 の下限を −5 に広げた）。ult は p2 の事後の ±4σ（この箱で切る）
- 粒子 5000、N_MUT 150、7 段、受理率 0.13、移動 9.0 回/次元
- 3 seed: max logL −8.65 / −8.68 / −8.76（幅 0.11）、RMSE 0.0756 / 0.0759 / 0.0779
- 同定される成分: a11・a12・a22・a33・a13・a23。符号だけ決まる: a14・a15（負）。ほかは箱全体に広がる（補足 S4）
- Pg の Day21/Day15 比: 0.95〜0.98（実測 1.00）

RMSE は組成（5 種の割合、Day 3・6・10・15・21）の、MAP での二乗平均平方根誤差。
