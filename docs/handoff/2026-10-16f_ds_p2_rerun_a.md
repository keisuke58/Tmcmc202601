# クラウド側 → GPU 側（2026-10-16f）: DS p2 の出し直しは **(a)**（ユーザー決定）

- **同じ設定のまま**（18000 粒子 × 150、13o の DS の OVERRIDE、`DATA_SOURCE=raw TWO_CHANNEL=1 RUNTAG=2c_raw`、PREV=`DS_p1_2c_raw_seed<S>`）、
  vancouver（4090）に 3 seed、**walltime 72h 程度**で出し直す。粒子数・箱は変えない（(b) はしない）。
- 後ろに 磨き（16b の設定、seed ごと）→ gate（`POLISH=0`）→ ult を依存でつなぎ直す。ult の walltime は DS p2 の実測（段の時間 × 段数）から見積もり直す。
- 16e §2 の 1（3519〜3521 の終わり方・到達した段・1 段の時間）を報告に。72h でも足りなさそうなら、投入前に知らせて。
- 4090 を DS に回すために CH の磨き直し（16c）の残りを hold してよい。
