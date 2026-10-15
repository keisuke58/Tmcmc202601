# GPU 側報告 2026-10-16（16b への対応: DS p1 の磨き直しを top-k 3・ftol 1e-7 で出し直した）

## 1. 3582〜3584 を qdel した
- 3 本とも走り始めて 10 分ほどで、終わったものはない（比べられる polish.json は無い）
- 3584 は最初の qdel で `Server could not connect to MOM` が出たので、もう一度 qdel して消えた。
  stuttgart03 に古い polish のプロセスが残っていないことは `ps` で確認した

## 2. 出し直した（`TOP_K=3 MAXITER=3000`、ftol は既定の 1e-7、`WRITE_RUN=1`、328c1d6 を pull してから）
| jobid | run | ノード | walltime |
|---|---|---|---|
| 3585 | DS_p1_2c_raw_seed42 | stuttgart03 | 48h |
| 3586 | DS_p1_2c_raw_seed7 | stuttgart03 | 48h |
| 3587 | DS_p1_2c_raw_seed123 | stuttgart03 | 48h |

- 3 本とも `JAX devices: [CudaDevice(id=0)]`、`gpualloc` で BUSY を確認
- 結果はジョブが `gpu_<日付>_polish_DS_p1_2c_raw_seed*_k3.{md,json}` として自動で push する
- 各 run の polish.json（10/14、300 反復・top-k 10）はこの 3 本で上書きされる。旧値は
  `gpu_2026-10-14_polish_DS_p1_2c_raw_seed*.json` に残っているので、比べるときはそちらを使う
- 3 本そろったら、所要時間・反復数・converged・幅を報告し、16a §2 の分け方で判定し直す

## 3. このあと
- DS p2（3519〜3521）は 10/16 10:00 JST ごろに終わる見込み（walltime 40h）。終わったら seed ごとに同じ設定で磨き、
  DS の gate は `POLISH=0` で出し直す（16b §2）。3529 は hold のまま
- CH・CS・DH の磨き直し（16a §4）は、枠が空いてから
