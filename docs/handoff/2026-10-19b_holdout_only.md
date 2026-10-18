# 2026-10-19b（クラウド側）: 13f-5（1 日を外した予測の確認、CH）だけやる。13g・13f-4 は見送り（ユーザー決定）

## 決定（ユーザー、10/19）
- **やる**: 13f-5 = leave-one-day-out、CH、Day 10 と Day 15 の 2 系列。
- **やらない**: 13g（重みなしの p2 の TMCMC）と 13f-4（箱 1.5 倍の p2）。補足 S9 の決定論的な感度解析と表 S1 の箱の広げ方で足りる。査読で求められたら検討する（記録へ）。

## 投入（回収テスト CS・DS・DH が終わって GPU が空いてから、10/20 ごろ）
- 条件 CH、生データ・2 通路、箱は本線の CH と同じ（p2 の OVERRIDE は a33w: `5:-15:20;1:-5:2.5;11:-5:1;16:-5:1;0:-5:1.5;6:-5:1;7:-1:5;13:-5:1;14:-5:1.5;17:-5:1;18:-5:1`）。
- 段は pilot → p1 → p2（ult までは不要。比べる相手は `CH_p2_2c_raw_a33w_seed*`）。seed は **42 だけ**（13f §5 のとおり、予測の確認なので 1 系列でよい）。
- `HOLDOUT_DAY=10` と `HOLDOUT_DAY=15` の 2 系列。RUNTAG は `2c_raw_a33w_ho10`・`2c_raw_a33w_ho15`（job が `_ho<日>` を付けるなら既定でよい）。
- 粒子数・N_MUT は本線の CH と同じ（pilot 1000×40、p1 2000×150、p2 2000×150）。1 系列 約 14h（4090）。
- 各段の判定は通常どおり `check_paper_runs.py`（1 seed なので判定 3・4 は出ない。0〜2b を見る）。
- 終わったら 6 run（2 系列 × 3 段）を `docs/revision/generated/final_runs/` に 5 ファイルでコピー（p2 は polish.json も）。

## クラウド側でやること（届いたら）
```
python3 tools/holdout_eval.py docs/revision/generated/final_runs \
  --holdout 'CH_p2_2c_raw_a33w_ho10_seed*' --full 'CH_p2_2c_raw_a33w_seed*' --day 10 --out docs/revision/generated/holdout/CH_day10.json
（Day 15 も同様）
```
補足に 1 段落（S8 の確認の節のあと）と本文に 1 文（pH の確認の隣）。
