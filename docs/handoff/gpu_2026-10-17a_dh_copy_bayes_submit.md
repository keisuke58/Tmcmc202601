# GPU 側報告 2026-10-17a: DH の段 (i)・(ii) を final_runs に（17a）、ベイズ因子（16k）の CS・CH 12 本を投入

## 1. 17a: DH pilot・p1 を final_runs に（済、3478222）
- `DH_pilot_2c_raw_seed{7,42,123}`: config・logL・run_record・samples・theta_MAP の 5 ファイル。polish.json は元から無い。
- `DH_p1_2c_raw_seed{7,42,123}`: 上の 5 ファイル＋polish.json（10/16 の top-k 3 で磨いた最新の値）。
  `polish_prev_20261016.json`（その前の磨き）は入れていない。
- DS の段 (i)〜(iii) は、ult が通ってからまとめて置く。

## 2. 16k: ベイズ因子の CS・CH 12 本を投入（ユーザー了承済み）
GPU の枠が 5 本空いていて、DS の残りの段（磨き 3604〜3606 → gate 3607 → ult）はすべて vancouver02・03 指定の依存でつながっている。
そのため 10/16 の例外（DS の分を足しても 15 本に収まる）を満たすので、DS の ult を待たずに入れた。

設定は 16k のとおり: pilot・1000 粒子・`N_MUT=120`・`DATA_SOURCE=raw TWO_CHANNEL=1`。
a45 = 0 のほうは `OVERRIDE=19:0:0`。3621 で変数が正しく渡っていることを `qstat -f` で確認した。

| jobid | ノード | 条件 | seed | RUNTAG | 状態 |
|---|---|---|---|---|---|
| 3620 | stuttgart02 | CS | 42 | 2c_raw_m120 | R |
| 3621 | stuttgart02 | CS | 42 | 2c_raw_a45zero_m120 | R |
| 3622 | stuttgart02 | CS | 7 | 2c_raw_m120 | R |
| 3623 | stuttgart01 | CS | 7 | 2c_raw_a45zero_m120 | R |
| 3624 | stuttgart01 | CS | 123 | 2c_raw_m120 | R |
| 3625 | celtic01 | CS | 123 | 2c_raw_a45zero_m120 | H（afterany 3620） |
| 3626 | celtic01 | CH | 42 | 2c_raw_m120 | H（afterany 3621） |
| 3627 | celtic01 | CH | 42 | 2c_raw_a45zero_m120 | H（afterany 3622） |
| 3628 | celtic02 | CH | 7 | 2c_raw_m120 | H（afterany 3623） |
| 3629 | celtic02 | CH | 7 | 2c_raw_a45zero_m120 | H（afterany 3624） |
| 3630 | celtic01 | CH | 123 | 2c_raw_m120 | H（afterany 3625） |
| 3631 | celtic02 | CH | 123 | 2c_raw_a45zero_m120 | H（afterany 3626） |

- 待機中の 7 本は 1 本ずつ前のジョブに依存させているので、同時に走る GPU ジョブは 15 本を超えない（いまちょうど 15 本）。
- walltime は stuttgart が 10h、celtic（2080Ti）は 16h。
- 全部終わるのは 10/18 の昼〜夕方の見込み（10/23 の目安には間に合う）。
- 1 本目が終わったら、`log_evidence` が数値か・`override_bounds` に `19:0:0` があるか・移動が 5 回/次元を超えるかを確認する。
  12 本がそろったら判定と ln Ẑ・ln B を `gpu_<日付>_bayes_raw.md` にまとめ、24 本分は DS・DH のあとに final_runs へ置く。
- DS・DH の 12 本は、DS が ult に入って枠が空いてから入れる。DH の SRC は `DH_p2_2c_raw` を前提にしている（16l）。

## 3. 走っているジョブ（変化なし）
- DS p2（3601〜3603）は約 11h 経過。後段（3604〜3611）は依存で待機している。
- CH の磨き（3592・3612）、CH の回収テスト（3614〜3616）、CS ult の磨き（3617・3619）は走行中。
- 3596（CH の磨き）は `qstat` に出てこないので、終わっているはず（結果はまだ確認していない）。
