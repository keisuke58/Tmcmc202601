# GPU 側報告 2026-10-16（CS ult 投入・判定 3 が磨いた値で見られていない）

## 1. CS の ult を投入した（15e）
- gate **3577**（stuttgart01、2h、`POLISH=0`、`EDGE_ALLOW=a33`）。それ以外は 3557 と同じ設定
  （OVERRIDE は p2 の run_record.json と同じ、`TWO_CHANNEL=1 DATA_SOURCE=raw`、N_MUT 150、5000 粒子）
- 判定: 全群 PASS、`EDGE_ALLOW=a33 の印は数えない` が出て、端の印（←）が 3 → **0**
- ult: **3578**（seed 42、vancouver03）、**3579**（seed 7、vancouver03）、**3580**（seed 123、vancouver02）、18h、通知 3581
  （空いていた 4090 に投入。前回の 3557 は stuttgart03 ×3 の予定だった）。gate の自動報告は 998c6d6
- 終わる見込み: 10/16 12:00〜13:00 JST（同じノードでの CH の ult 3547〜3549 は 7.8〜8.7h かかった）。全 PASS なら 15d と同じ形で final_runs へ

## 2. polish が収束していない → 判定 3 は粒子の最良値で見ている（13e と違う）
`check_paper_runs.py` が磨いた値を使うのは、全 seed の polish.json が `converged=True` のときだけ（:290）。
2c_raw の polish.json は 21 本のうち 20 本が **maxiter 300 に達して converged=False** なので、判定 3 は
どの群でも粒子の最良値で判定されている（gate の報告の「polish.json なし」は実際とずれた表示で、正しくは「収束していない」）。

| 群 | 粒子の最良値 3 seed の幅 | 磨いた値 3 seed の幅 | gain | 磨いた値で判定すると |
|---|---|---|---|---|
| CH_p1_2c_raw | 0.115 | 0.043 | 0.09〜0.23 | PASS |
| CH_p2_2c_raw_a33w | 0.228 | 0.072 | 1.13〜1.43 | PASS |
| CS_p1_2c_raw | 0.235 | 0.003 | 0.14〜0.37 | PASS |
| CS_p2_2c_raw | 0.038 | 0.206 | 0.29〜0.47 | PASS |
| DH_p1_2c_raw | 0.641 | 0.352 | 0.71〜1.21 | PASS |
| DH_p2_2c_raw | 0.361 | 0.087 | 0.80〜1.07 | PASS |
| **DS_p1_2c_raw** | 0.789 | **1.705** | 1.92〜4.38 | **FAIL（幅 > 1）** |

- CH・CS・DH は、磨いた値で見ても判定 3 の結論は変わらない
- **DS p1 は、磨いた値で見ると判定 3 が FAIL**（seed123 だけ −59.07、他の 2 本は −60.7〜−60.8）。
  ただし収束前の値なので、maxiter を増やせば幅が縮むのか広がるのかは未確認
- gain が 1 を超える群がある（CH p2・DH・DS）。粒子の最良値は局所最大より 1〜4 低い

## 3. 相談（クラウド側で決めてほしい）
- 判定 3 は「収束していなくても磨いた値を使う」のか、「maxiter を増やして収束させてから判定する」のか、
  「今のまま（収束しなければ粒子の最良値）」で良いのか
- DS p1 の判定 3 をどう扱うか（DS の p2 を投入する前に決めたい）
- コード（check_paper_runs.py・gate_submit_ult.sh・polish_max_logL.py）はまだ変えていない
