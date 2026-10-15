# GPU 側報告 2026-10-16（16a への対応: DS の gate を hold、DS p1 を maxiter 3000 で磨き直し中）

## 1. DS の gate 3529 → `qhold` した
- `Hold_Types = su`（依存待ちの s に、手で掛けた u を足した）。p2（3519〜3521）が終わっても動かない
- 出し直すか `qrls` するかは、DS p1 の磨き直しが終わってから決める。
  maxiter 3000 で 3 seed を 1 本の gate でまとめて磨くと walltime 24h に収まらないおそれがある
  （300 反復でも 1 seed 4.3〜4.5h、下記）。p2 も seed ごとに polish ジョブで磨き、gate は `POLISH=0` で出し直すのが確実だと考えている

## 2. DS p1 の磨き直し（`MAXITER=3000`、`TOP_K=10`、`WRITE_RUN=1`）
| jobid | run | ノード | walltime |
|---|---|---|---|
| 3582 | DS_p1_2c_raw_seed42 | stuttgart03 | 48h |
| 3583 | DS_p1_2c_raw_seed7 | stuttgart03 | 48h |
| 3584 | DS_p1_2c_raw_seed123 | stuttgart03 | 48h |

- 3 本とも `JAX devices: [CudaDevice(id=0)]`、`gpualloc` で BUSY を確認
- 終わると、ジョブが `gpu_<日付>_polish_DS_p1_2c_raw_seed*_m3000.{md,json}` を自動で push する。3 本そろったら判定し直して報告する

## 3. maxiter 3000 の所要時間（分かっている範囲）
- 300 反復では、DS p1 1 seed が 3090（stuttgart03）で **4.3〜4.5h**（2026-10-14、3505〜3507）
- 3000 反復は今回が初めてなので、実測は終わってから報告する。上限まで回ると 40h を超える見込み
- CPU の実測はない（DS の polish は CPU では終わらない）

## 4. ジョブの本数
実行中の GPU ジョブは 12 本（DS p2 3519〜3521、DH ult 3573〜3575、CS ult 3578〜3580、DS p1 polish 3）。上限 15 本の範囲内。
CH・CS・DH の磨き直し（16a §4）は、枠が空いてから投入する。
