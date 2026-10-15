# GPU 側報告 2026-10-15（CS の gate 3546 も walltime 切れ → polish を seed ごとに GPU で）

## 1. CS の gate 3546（8h）も polish 中に kill — CS の ult は未投入
- `logs/pbs/gate_ult_3546.copaam.log`: 07:08 起動、`job killed: walltime 28842 exceeded limit 28800`
- 3527（4h）に続いて 2 回目

## 2. 原因: gate の polish は GPU で走っていない
- `tools/gate_submit_ult.sh:77` の `export JAX_PLATFORMS=cpu` が polish にも効く
  （`POLISH_DEVICE=gpu` を渡しても JAX が CPU しか見ない）。`CUDA_VISIBLE_DEVICES` も PBS_GPUFILE から設定していない
- 証拠: `gate_ult_3544.copaam.log:34` に `JAX devices: [CpuDevice(id=0)]`。
  実行中の DH の gate 3525（vancouver01-gpu/0）は `gpualloc` で HELD-IDLE（GPU を確保したまま使っていない）
- DH の gate 3525（8h）と DS の gate 3529（24h）も同じ理由で切れるおそれがある。gate_submit_ult.sh は未修正

## 3. 対応（ユーザー判断）
- polish を seed ごとに `tools/polish_max_logL_job.sh`（GPU、`TOP_K=10 WRITE_RUN=1`、16h）で投入:

| jobid | seed | ノード |
|---|---|---|
| 3553.copaam | 42 | vancouver01 |
| 3554.copaam | 7 | vancouver02 |
| 3555.copaam | 123 | stuttgart03 |

- それに依存させて CS の gate を `POLISH=0` で投入: **3557.copaam**（stuttgart01、2h、afterany:3553:3554:3555）。
  中身は 3546 と同じ（OVERRIDE は p2 の run_record.json と一致を確認、ult は stuttgart03 ×3・18h、`TWO_CHANNEL=1 DATA_SOURCE=raw`）
- 3556 は OVERRIDE の写し間違いで走る前に qdel 済み（無視してよい）

## 4. CH の ult 投入報告が push されていなかった
- gate 3544 は vancouver03 で走り、`git: command not found`（gate_submit_ult.sh:197）で報告をコミットできなかった。
  報告 `gpu_2026-10-15_gate_CH_ult.md` をこのコミットで手で push した（ult 3547-3549 は正常に実行中）
