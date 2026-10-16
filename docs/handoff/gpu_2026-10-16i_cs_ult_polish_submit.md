# GPU 側報告 2026-10-16i（16j: CS ult の磨きを DH ult の後ろに投入、16i は確認のみ）

## 16j: CS ult の磨き（3 本）
DH ult（3573〜3575）が空ける vancouver の枠に依存でつないだ。磨きは 1 本 2〜7h なので、DS の ult が vancouver を使う 10/18 より前に終わる見込み。

| jobid | 対象 | ノード | 依存 |
|---|---|---|---|
| 3617 | CS_ult_2c_raw_seed42 | vancouver01 | afterany:3573:3574:3575 |
| 3618 | CS_ult_2c_raw_seed7 | vancouver01 | 同上 |
| 3619 | CS_ult_2c_raw_seed123 | vancouver02 | 同上 |

- 設定は CH と同じ（`TOP_K=3 MAXITER=3000 WRITE_RUN=1`、walltime 16h）。CS ult には polish.json が無かったので、上書きされる旧値は無い。
- 結果はジョブが `gpu_*_polish_CS_ult_2c_raw_seed*_k3.{md,json}` を push する。旧が無いので新だけ（16j §2）。

## 16i: ガードは確認のみ
e4487f0 の差分を読んだ。走っている回収テスト 3614〜3616 には影響しない。CS・DS・DH の回収テストのときは 10/13 の `*_syn_*` を退避してから作る。

## 本数
実行中 GPU 14 本（DH ult 3、DS p2 3、CH の磨き 5、CH 回収 3）。依存待ち: 3576（通知）、3604〜3607（DS）、3608〜3611（CPU、14c の DS）、3617〜3619（CS ult の磨き）。
DH ult が抜けて CS の磨き 3 本が入るので、同時に走るのは 15 本以内。
