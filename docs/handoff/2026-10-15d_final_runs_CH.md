# クラウド側 → GPU 側（2026-10-15、14d）: 確定した CH の生データの run を final_runs に入れて push（ユーザー依頼: CH の図）

CH の ult（`CH_ult_2c_raw_a33w_seed*`）が全 PASS で確定した（[gpu_2026-10-15_CH_ult.md](gpu_2026-10-15_CH_ult.md)）。
クラウド側で **CH の図・表・数値を先に作る**ので、生データの CH の run を `docs/revision/generated/final_runs/` にコピーして push してほしい。
やり方は 11j と同じ（各 run から 5 つのファイルだけ）。

## コピーする群（各 3 seed）
| 群 | 段 | 用途 |
|---|---|---|
| `CH_pilot_2c_raw_seed*` | pilot | 段の比較の図 |
| `CH_p1_2c_raw_seed*` | p1 | 段の比較の図・Fig. 2 の p1 |
| `CH_p2_2c_raw_a33w_seed*` | p2 | 補足（箱・R̂） |
| `CH_ult_2c_raw_a33w_seed*` | ult | **本文の図・表（最終）** |
| `CH_p2_2c_raw_cold_seed42` | p2（cold） | 補足の一文（段分けの偏りの確認） |

```bash
cd ~/Tmcmc202601 && git pull
R=data_5species/main/_runs/paper_gateoff; F=docs/revision/generated/final_runs
for d in $R/CH_pilot_2c_raw_seed* $R/CH_p1_2c_raw_seed* $R/CH_p2_2c_raw_a33w_seed* $R/CH_ult_2c_raw_a33w_seed* $R/CH_p2_2c_raw_cold_seed42; do
  n=$(basename $d); mkdir -p $F/$n
  cp $d/{samples.npy,logL.npy,theta_MAP.json,run_record.json,config.json} $F/$n/
done
git add $F/CH_*2c_raw* && git commit -m "final_runs: CH の生データ・2 通路の run（pilot・p1・p2 a33w・ult a33w・cold）" && git push
```
- **実験データそのもの（反復の値・画像ごとの値・raw_dataset のファイル）は入れない**。上の 5 つのファイルだけ
  （run_record・config には Day 1 の組成の中央値と種ごとの SD しか入っていないことをクラウド側で確認済み）。
- 古い箱の `CH_p2_2c_raw_seed*`（a33 が端に張り付いた方）は入れなくてよい。
- 終わったら commit のハッシュだけ一行で。CS・DS・DH も ult が通ったら同じように頼む。
