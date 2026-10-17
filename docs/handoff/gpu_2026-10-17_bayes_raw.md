# GPU 側 → クラウド: ベイズ因子（16k、生データ・2 通路）の CS・CH（2026-10-17）

16k の 24 本のうち CS・CH の 12 本が終わった。DS・DH は走行中（下）で、そろったらこのファイルに追記する。

## 置き場所
- `docs/revision/generated/final_runs/{CS,CH}_pilot_2c_raw_m120_seed{42,7,123}/`（全モデル）
- `docs/revision/generated/final_runs/{CS,CH}_pilot_2c_raw_a45zero_m120_seed{42,7,123}/`（a45 = 0）
- 各 5 ファイル（samples・logL・theta_MAP・run_record・config、15d と同じ形）。

## 判定（`tools/check_paper_runs.py`）
- 4 群とも判定 0〜4 全 PASS（判定 3 は粒子の最良値、polish なし）。
- 移動は 8.7〜10.2 回/次元（判定 (c) の 5 を超える。n_mut 40 のときは 2.7〜3.0 で落ちていた）。
- a45 = 0 の 6 本は run_record の `args.override_bounds` が `19:0:0`、`n_mutation_steps` 120 を確認。
- 全 run β = 1、8 段、`log_evidence` は数値。

## ln Ẑ と ln B（ln B = ln Ẑ_full − ln Ẑ_{a45=0}、3 seed の平均）

| 条件 | ln Ẑ_full（seed 42 / 7 / 123） | 平均（幅） | ln Ẑ_{a45=0}（seed 42 / 7 / 123） | 平均（幅） | ln B |
|---|---|---|---|---|---|
| CS | −16.64 / −16.68 / −16.42 | −16.58（0.26） | −16.55 / −16.49 / −16.50 | −16.51（0.06） | **−0.06** |
| CH | −13.17 / −13.13 / −13.23 | −13.18（0.11） | −13.31 / −13.16 / −13.15 | −13.21（0.17） | **+0.03** |

- どちらも |ln B| < 0.1 で seed の幅（0.06〜0.26）より小さい → **a45 の有無はデータで区別できない**（どちらの向きにも支持なし）。
- 事後の a45（全モデル）: CS は −0.01〜+0.03（ほぼ箱の中央、一様に近い）、CH は +0.44〜+0.53（幅 [−0.88, +1.85]）。

## DS・DH（走行中）
- 9 本を投入: DS 6 本（3633〜3638、stuttgart01〜03）、DH seed42 の 2 本と seed7 全モデル（3639〜3641、celtic01・02）。walltime 12h。
- 残り 3 本（DH seed7 a45=0、DH seed123 の 2 本）は DS ult が入ってから投入する。
