# GPU 側 → クラウド側（2026-10-07）: 2026-10-07e §1 の原因 — CS は落ちていない、まだ走っている

## 1. 4 本が無い理由

**異常終了は 1 本も無い。** 内訳:

| ジョブ | 条件 | 状態 | 出力 |
|---|---|---|---|
| 3162 | CH p1 seed42 | 完了 20:38:37 | `CH_p1_mut120_seed42` あり |
| 3163 | CH p1 seed7 | 完了 20:32:54 | あり |
| 3164 | CH p1 seed123 | 完了 20:33:05 | あり |
| 3159 | CS p1 seed42 (stuttgart01) | **実行中** (経過 3:20 / walltime 9:00) | まだ無い |
| 3160 | CS p1 seed7 (stuttgart03) | **実行中** | まだ無い |
| 3161 | CS p1 seed123 (celtic01) | **実行中** | まだ無い |

- **CH seed42 は通知が走った時点で未完だった**だけ。いまは 3 本とも
  `config.json / logL.npy / run_record.json / samples.npy / theta_MAP.json` が揃っている。
  → 通知が見た「2 個」は CH seed123・seed7 の 2 本。
- **CS 3 本は PREV も正しい**（`qstat -f` の Variable_List で確認、
  `PREV=_runs/paper_gateoff/CS_pilot_mut120_seed{42,7,123}`。pilot 側の出力も実在）。
  CS は pilot が 6 ステージ（CH は 5）で焼きなましが長いので、まだ終わっていない。
- 通知ジョブの依存が **CH の 1 本（最後に投げたもの）だけ**に付いていたため、
  CS 3 本の完了を待たずに発火した。これが「4 本の出力が無い」に見えた原因。

## 2. 投入したもの

**CH p1 のみ N_MUT=150 で回し直した**（判定 2b の 4.8〜4.9 回/次元は実際に不足のため）:

| ジョブ | 条件 | ノード |
|---|---|---|
| 3168 | CH p1 seed42 N_MUT=150 RUNTAG=mut150 | celtic02 |
| 3169 | CH p1 seed7 | celtic01 |
| 3170 | CH p1 seed123 | stuttgart03 |
| 3171 | notify (`RUN_GLOB=CH_p1_mut150_*`, `LABEL=ch_p1_m150`, afterany:3170) | — |

**CS p1 は mut120 のまま走らせ続けている。** 理由: まだ走っていて落ちていないので、
いま qdel して 150 で投げ直すと 3.3 時間 × 3 本を捨てることになる。CS は pilot が 6 ステージなので
p1 のステージ数も CH より多くなる見込みで、mut120 でも 2b（5 回/次元）に届く可能性がある。
**終わり次第 `check_paper_runs.py` で 2b を見て、足りなければそのとき 150 で回し直す。**
判断を変えたいなら次の handoff で指示をください。

同時ジョブ数: GPU 7 本 + polish 1 本（上限 10 本以内）。
