# GPU 側報告 2026-10-16（CH cold p2 の本文の数字の照合・16g の旧 polish 値の退避）

## 1. dec2e44 の一文（CH cold p2）の照合
samples.npy から直接（5% / 50% / 95%）:

| run | max logL | a33 中央値 [5%, 95%] | a45 中央値 [5%, 95%] |
|---|---|---|---|
| CH_p2_2c_raw_seed42（warm） | −27.25 | −0.663 [−0.974, +0.082] | +0.438 [−0.844, +1.837] |
| CH_p2_2c_raw_seed7（warm） | −26.71 | −0.667 [−0.972, +0.109] | +0.424 [−0.851, +1.824] |
| CH_p2_2c_raw_seed123（warm） | −26.95 | −0.668 [−0.971, +0.093] | +0.543 [−0.834, +1.875] |
| CH_p2_2c_raw_cold_seed42 | −26.74 | −0.647 [−0.970, +0.070] | **+0.566** [−0.811, +1.860] |

- max logL が warm の幅に入る、a33 が 0.1 以内、は正しい。
- **a45 の中央値は warm の 3 seed の範囲（+0.424〜+0.543）の外**で、seed 7 との差は **0.14**。
  「to within 0.1」は「いちばん近い seed との差」（0.023）なら成り立つが、「どの seed とも」なら成り立たない。
  例: 「a33 は 0.03 以内、a45 は 0.15 以内」か「the cold-run medians lie within 0.03 of the warm-seed range」。
- 比べている warm は旧い箱（a33 の下限 −1、粒子の 2 割が下の端）の 3 seed。cold も同じ箱なので warm start の確認としては成り立つが、最終の a33w の箱とは別。
- check で cold の判定 3・4 が FAIL なのは seed が 1 本だけのため（中身の問題ではない）。

## 2. 16g: 旧 polish 値の退避
- `polish_max_logL.py --write-run` は旧 polish.json を残さず上書きする。
- 生データ・2 通路の run の polish.json 23 個を、同じディレクトリの `polish_prev_20261016.json` に退避した（_runs 内、git 管理外）。
  CH の磨き直し（3589〜3596）と DS p2 の磨き（3604〜3606）の報告では「旧・新・大きい方」を並べる。
- DS p1 の polish.json は既に 10/16 の値（seed 123 は −61.25）。判定に使う −59.07 は push 済みの `gpu_*_polish_*.json` にだけ残っている。
