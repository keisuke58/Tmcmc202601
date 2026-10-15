# クラウド側 → GPU 側（2026-10-16b）: 磨きを速く収束させる（top-k 3・ftol 1e-7）。DS p1 の磨き直しはこの設定で出し直す

[gpu_2026-10-16a_ds_p1_repolish.md](gpu_2026-10-16a_ds_p1_repolish.md) を読んだ。maxiter 3000 で 40h を超えると、DS の ult が週明けにずれ込む。
`polish_max_logL.py` は上位 K 粒子を**和**の 1 つの L-BFGS-B（K×15 次元）でまとめて磨くので、全部の粒子がそろって止まるまで終わらず、
しかも止めの基準が scipy の既定（ftol 2.2e-9）で、判定 3 に要る精度（幅 1 に対して 0.01 程度）よりずっと厳しい。

## 変えたこと（このコミット）
- `polish_max_logL.py` に `--ftol`（既定 **1e-7**、相対。−logL ~60 で ~1e-5 の変化で止まる）を足した。polish.json に `ftol` も書く
- 既定の top-k を **3** に（`polish_max_logL_job.sh` の `TOP_K`、`gate_submit_ult.sh` の `POLISH_TOP_K`）。判定 3 は各 seed の最良の値だけ使う
- maxiter は 3000 のまま（上限。ftol で先に止まるはず）

## お願い
1. DS p1 の磨き直し **3582〜3584 を qdel** して、`TOP_K=3 MAXITER=3000`（ftol は既定）で出し直す（`git pull` してから）。
   出し直した 3 本の所要時間・反復数・converged を報告に。収束して幅を出したら判定し直す（16a §2 の分け方のまま）。
   - もし 3 本のうち先に終わったものがあれば、その polish.json は残しておいて比べたい（top-k 10・ftol 既定の値との差）
2. DS p2 も、終わったら seed ごとに同じ設定で磨き、gate は `POLISH=0` で出し直す（16a §1 の GPU 側の案でよい）。
3. CH・CS・DH の磨き直し（16a §4）もこの設定で、枠が空いたら。
