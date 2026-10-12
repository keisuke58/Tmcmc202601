# クラウド側 → GPU 側（2026-10-13e）: 判定 3（(d)）を「磨いた max logL」で見る（ユーザー決定、2 通路の結果を見る前）

## 決定
- **判定 3 = seed 間の max logL の幅 ≤ 1 を、各 seed の上位粒子から尤度を局所最大化した値（磨いた値）で判定する。** 粒子の最良値は粒子数と偶然で揺れる（DS ult p5k は粒子の最良値で幅が大きかったが、磨くと 0.13）。「seed が同じ山に届いたか」を直接見る。
- **判定 4（中央値の幅 ≤ 0.5 SD）は変えない。** R̂ を判定の表に並べて報告する（`tools/rhat_seeds.py`）。
- 2 通路の run の結果を誰も見ていない時点で決めた。4 条件すべてに同じく当てる。今の尤度の結果には遡って当てない。

## 変えたもの（このコミット）
- `tools/polish_max_logL.py`: 2 通路（pilot・p1 の種ごとの ψ 固定、p2・ult の多チャネル）の run を組み直せるようにした（保存 logL の再現を確かめてから磨く、今までどおり）。`--write-run` で各 run に `polish.json` を書く。
- `tools/check_paper_runs.py`: 群の全 seed に `polish.json` がそろって収束していれば、判定 3 を磨いた値で判定（表示「3 maxlogL 幅<=1（磨いた値）」、粒子の最良値の幅も並べて出す）。そろっていなければ今までどおり粒子の最良値で、その旨を表示。
- `tools/gate_submit_ult.sh`: 判定の前に磨く（`POLISH=1` 既定、`POLISH_TOP_K` 10、`POLISH_MAXITER` 300）。**gate は `gpus=1` で投げる**（磨くのは ODE を通す勾配なので GPU が要る）。失敗しても判定は続ける（粒子の最良値で）。
- 本文の (d) の定義を「上位粒子からの局所最適化で磨いた max logL」に変えた。

## GPU 側でやること
- 13b の 2 通路の gate（p2 → ult）は、上の新しい gate を `-l nodes=1:ppn=1:gpus=1:<node>` で投げる（既に投げた gate があれば、qdel して投げ直し。前段のジョブは触らない）。
- p1 → p2 の手動の判定でも、先に `POLISH_DEVICE=gpu python3 tools/polish_max_logL.py <root> --glob '<群>' --top-k 10 --write-run`（`tools/polish_max_logL_job.sh` で GPU に投げてもよい）を回してから `check_paper_runs.py`。
- 報告の表に、磨く前・磨いた後の max logL の幅と、R̂（同定された成分の最大）を並べる。
