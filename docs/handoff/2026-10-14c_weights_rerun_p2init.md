# クラウド側 → GPU 側（2026-10-15、14c）: 重みの感度を「生データの p2 の MAP から」回し直す（12 本、CPU ジョブ）

14a の 12 本（A・B・C × 4 条件）はそろった（ありがとう）。ただ `two_channel_pilot_v2.py` の初期値が
**`final_theta_MAP/<tag>.json`（読み取りデータ・旧尤度の時代の MAP）**だったため、L-BFGS-B（150 反復）が古い MAP の近くの局所解で止まっている。
その証拠に、どの重みでも DH の Pg D21/D15 が約 1.9（実測 14.9、生データの p2 の事後は 9.6〜18）。
補足 `sec:S_lambda` は「局所解がどちらに動くか」としては書けるが、数字は生データの p2 の MAP から始めたものに差し替えたい。

## やること
`tools/two_channel_pilot_v2.py` に `--init-glob`（run の群で max logL の seed の `theta_MAP.json` から始める）を足し、
`tools/weights_raw_job.sh` に `VAR=A`・`INIT_GLOB`・`OUTDIR` を足した（このコミット）。CH の OVERRIDE は 14b と同じ `5:-15:20` を先頭に足した。

```bash
cd ~/Tmcmc202601 && git pull
R=data_5species/main/_runs/paper_gateoff
O=docs/revision/generated/weight_sensitivity/raw_p2init
mkdir -p $O
declare -A G=([CS]="$R/CS_p2_2c_raw_seed*" [CH]="$R/CH_p2_2c_raw_a33w_seed*" [DH]="$R/DH_p2_2c_raw_seed*" [DS]="$R/DS_p2_2c_raw_seed*")
for T in CS CH DH; do for V in A B C; do      # DS は p2 が 3 本そろってから同じように
  qsub -l nodes=1:ppn=1:<ノード> -l walltime=08:00:00 -v TAG=$T,VAR=$V,INIT_GLOB="${G[$T]}",OUTDIR=$O tools/weights_raw_job.sh
done; done
# 最後に表（依存: 上の全部）
qsub -W depend=afterany:<上の job id を : でつなぐ> -v VAR=TABLE,OUTDIR=$O tools/weights_raw_job.sh
```
- 表は `VAR=TABLE` が `$O` から作るが、出力先の tex は今の `supp_S_lambda2c.tex` を**上書きする**（それでよい。前の 12 本の json は `weight_sensitivity/raw/` に残る）。
  DS が後になる場合は、DS の 3 本が終わってからもう一度 `VAR=TABLE` を回す。
- 各ログの先頭に `初期値: <run>` の行が出ること、MAP の行（`MAP (now, under 2c logL)`）の −logL が p2 の max logL（の符号を変えたもの）に近いことを確認。
- GPU は使わない（CPU 1 コア）。本線（CS・DH の gate、DS の p2、CH の ult）を優先し、ノードの空きで。
- 終わったら `gpu_2026-10-1X_weights_p2init.md` に表（12 / 12）と、A・B・C で a45 の符号が変わる条件・DH の Pg 比を 3 行で。
