# クラウド側 → GPU 側（2026-10-14a）: 尤度の重みの感度（raw・2 通路）の残り 8 本を GPU で

ユーザー決定（10/14）。補足 `sec:S_lambda` の表（`tools/weight_sens_table.py`）に使う。
**本線（p2 → gate → ult）を優先**。GPU が 1 枚空いたとき（例: CH・CS の p2 が終わったあと）に、本線のジョブと重ならないように回す。
13g（重みなしの p2 を TMCMC で 3 seed）は引き続き保留。こちらは決定論的な最適化だけ（TMCMC ではない）。

## 何をするか
`tools/two_channel_pilot_v2.py`（−logL を L-BFGS-B で最小化、段 (iii) の MAP から開始）を、条件 4 × 重み 2 通りで回す。
A（今の重み）の 4 本はクラウドの CPU で済み（このコミットで `docs/revision/generated/weight_sensitivity/raw/{CS,CH,DS,DH}_A.json` を push）。
CPU では 1 本 1〜2 時間かかり、バックグラウンドの時間の上限で止まったので、残りを GPU に頼む。

- B: 重みなし（`--lambda-pg 1 --lambda-late 1 --lambda-rare 1`、λψ は条件ごとの既定のまま）
- C: λψ を全条件 2.0（`--lambda-ch3 2.0`、ほかは A）

箱は**本番の raw の p2 と同じ OVERRIDE**（13o・13p）。A と同じ `--restarts 1 --maxiter 150`、`--data-source raw`。

```bash
cd ~/Tmcmc202601 && git pull
O=docs/revision/generated/weight_sensitivity/raw
declare -A OV=(
 [CH]="1:-5:2.5;11:-5:1;16:-5:1;0:-5:1.5;6:-5:1;7:-1:5;13:-5:1;14:-5:1.5;17:-5:1;18:-5:1"
 [CS]="5:-15:20;10:-1:20;0:-10:2;6:-5:1;11:-5:1;12:-5:3;13:-5:1;16:-5:1;17:-5:1;18:-5:1"
 [DS]="5:-15:20;6:-15:20;19:-15:20;14:-5:4;12:-5:3;16:-5:2.5;7:-5:2.5;10:-5:1.5"
 [DH]="12:-15:20;5:-15:20;6:-15:20;18:-25:20;13:-15:20;11:-8:3;19:-5:20;2:-10:5;7:-10:5;10:-8:3;0:-10:5;1:-10:3;17:-5:2.5"
)
run() { t=$1; v=$2; shift 2
  [ -s $O/${t}_$v.json ] && return
  python3 tools/two_channel_pilot_v2.py --tag $t --data-source raw --override "${OV[$t]}" \
      --restarts 1 --maxiter 150 "$@" --out $O/${t}_$v.json > $O/${t}_$v.log 2>&1
}
for t in CS CH DS DH; do
  run $t B --lambda-pg 1 --lambda-late 1 --lambda-rare 1
  run $t C --lambda-ch3 2.0
done
python3 tools/weight_sens_table.py $O --out docs/revision/EML_submission/figures/raw/supp/supp_S_lambda2c.tex
```

- PBS で投げる場合は `gpus=1`、walltime 4h で足りるはず（1 本数分〜十数分の見込み）。8 本は 1 ジョブで順に回してよい。
- 各ログの先頭に `DATA_SOURCE` が raw、`σ_viab` がバイオフィルム単位（13m: CS 0.113・CH 0.095・DS 0.059・DH 0.075）になっていることを確認。
- 終わったら `weight_sensitivity/raw/*.json`・`*.log`・`supp_S_lambda2c.tex` を push し、
  `gpu_2026-10-1X_weights_raw.md` に `weight_sens_table.py` の出力（12 / 12）と、A・B・C で a45 の符号が変わる条件があるかを 3 行で。
  json・log は RMSE・θ・ψ の中央値だけで、実験の反復の値は入っていない（push してよい）。
