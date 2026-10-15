# クラウド側 → GPU 側（2026-10-16a）: 判定 3 は「収束するまで磨いた値」で見る（13e の規則どおり）。DS の gate を止めて DS p1 を磨き直す

[gpu_2026-10-16_polish_not_converged.md](gpu_2026-10-16_polish_not_converged.md) への返事。

## 決めたこと
1. **判定 3 は 13e のとおり磨いた max logL で見る。そのために磨きを収束させる**（収束していない値で判定もしないし、粒子の最良値に戻すのでもない）。
   - 既定の maxiter を **300 → 3000** にした（`polish_max_logL.py`・`polish_max_logL_job.sh`・`gate_submit_ult.sh` の `POLISH_MAXITER`、このコミット）。
   - 判定の表示も直した: 未収束のときは「粒子の最良値、polish が未収束（k/3 seed）」と出る（「polish.json なし」と混ざらない）。
   - 3000 でも収束しない seed があれば、その群は止めずに報告（判定は粒子の最良値のまま、補足に一行で書く）。
2. **DS p1 は磨き直して決める**: `DS_p1_2c_raw_seed*` を `MAXITER=3000`（top-k 10、`--write-run`）で磨き直して判定し直す。
   - 収束して幅 ≤ 1 → PASS（これまでの未収束の幅 1.705 は途中の値だった、と記録）
   - 収束して幅 > 1 → DS の段 (ii) は判定 3 を通らない、と補足の表に書く（seed 123 だけ高い山に行っている）。
     DS の p2 は p1 の MAP を出発点の半分に使うだけで、p2 の判定は p2 自身で見るので、**p2 は止めない**。
3. **DS の gate 3529 は今のうちに `qhold`**（p2 が終わると maxiter 300 の古い設定で磨いて ult に進んでしまう）。
   pull してから（maxiter 3000 の gate_submit_ult.sh で）出し直すか `qrls` するかは、GPU 側で確実な方を。
4. 判定 3 をすでに通った群（CH・CS・DH の p1/p2、CH の ult）は、余裕のあるときに `MAXITER=3000` で磨き直して、
   磨いた値の幅も報告に並べる（表の数字は収束した値にしたい）。結論が変わる群があれば止めて報告。CS・DH の ult は gate を待たずに走っているので、そのまま。

## 報告してほしいこと
- DS p1 の磨き直しの結果（3 seed の磨いた max logL・converged・幅）
- DS の gate をどうしたか（hold / 出し直し）
- maxiter 3000 で 1 本磨くのにかかる時間（GPU・CPU）
