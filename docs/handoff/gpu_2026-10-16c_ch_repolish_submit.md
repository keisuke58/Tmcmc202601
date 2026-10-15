# GPU 側報告 2026-10-16（16c への対応: CH の磨き直し 9 本を投入）

## 1. CH の磨き直し（`TOP_K=3 MAXITER=3000`、ftol 既定 1e-7、`WRITE_RUN=1`、stuttgart01、各 16h）
seed ごとに ult → p2 → p1 を `afterany` でつないだ（同時に走るのは 3 本）。

| seed | ult_2c_raw_a33w | p2_2c_raw_a33w | p1_2c_raw |
|---|---|---|---|
| 42 | 3588（R） | 3589（H） | 3590（H） |
| 7 | 3591（R） | 3592（H） | 3593（H） |
| 123 | 3594（R） | 3595（H） | 3596（H） |

- 走り始めた 3 本は `JAX devices: [CudaDevice(id=0)]` を確認
- 9 本そろったら `check_paper_runs.py` で 3 群を判定し直し、磨いた max logL・converged・幅・所要時間を 1 枚にまとめる。
  polish.json は `final_runs/` の同じ run にもコピーして push する

## 2. 回収テスト（16c §2）はまだ投入していない
- 実行中の GPU ジョブが 15 本（DS p2 3、DH ult 3、CS ult 3、DS p1 の磨き直し 3、CH の磨き直し 3）で上限に達したため
- 16c の順番どおり、CH の磨き直しのあと、枠が空き次第投入する
