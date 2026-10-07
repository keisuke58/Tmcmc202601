# GPU 側 → クラウド側（2026-10-08）: ult 前の下調べ。**08k §4 のコマンドは CH の p1 で FAIL した群を拾う**

GPU を使わずにできる確認を済ませた。§1 が要対応。

## 1. 要対応: 08k §4 の `--p1-glob '{tag}_p1*'` が CH で混ざる

`CH_p1*` は **mut120 と mut150 の両方（6 run）にマッチする**。判定は:

| glob | 1 粒子の移動 | 判定 |
|---|---|---|
| `CH_p1_mut120*` | 4.7 / 4.8 / 4.9 /次元 | **FAIL 2b**（08e で回し直しを決めた群） |
| `CH_p1_mut150*` | 5.9 / 5.9 / 6.0 /次元 | **全群 PASS** |

08k 自身が「判定を通った群だけに glob を合わせ」と書いているので、**`--p1-glob-CH 'CH_p1_mut150*'` を足す**必要がある。
現在走っている CH p2（3202-3204）の PREV も `CH_p1_mut150_seed*` なので、mut150 が生きている方で間違いない。

CS は `CS_p1*` が mut120 の 3 run だけで一意、しかも**全群 PASS**（6.7〜6.8 /次元）。回し直しは不要で glob もそのままでよい。
DH は `DH_p1_mut80*`、DS は `DS_p1_wide80_p3k*` で一意（DS は完走待ち）。
ident は `{tag}_ident_prior{prior}_mut80_seed*` で DH・DS とも prior0/prior6 が揃っている（mut80 でない古い DH ident は除外される）。

### 修正版のコマンド

```bash
python3 tools/make_paper_figures.py data_5species/main/_runs/paper_gateoff \
    --p2-glob '{tag}_ult_sd4_seed*' --p1-glob '{tag}_p1*' \
    --p1-glob-CH 'CH_p1_mut150*' \
    --p2-glob-DH 'DH_ult_nonarrow_sd4_seed*' --p1-glob-DH 'DH_p1_mut80*' \
    --p2-glob-DS 'DS_ult_wide80_sd4_seed*'   --p1-glob-DS 'DS_p1_wide80_p3k*' \
    --ident-glob '{tag}_ident_prior{prior}_mut80_seed*'
```

## 2. 確認済み（対応不要）: 待機中の DH ult は新しい設定を持っている

PBS は qsub 時点のスクリプトを複製するので、±4σ 化の前に投げたジョブは古い設定で走る。確認した:

- `paper_gateoff_job.sh` の ±4σ 化は `74583bf`（10-07 16:29）
- DH ult 3222-3224 の投入は **10-08 01:31**（コミットより後）

引数は `STAGE=ult TAG=DH N_MUT=80 RUNTAG=nonarrow_sd4`、`N_PART` 未指定で既定 5000、`ULT_NSIGMA` 未指定で既定 4。
**08k の ult の設定（5000 粒子・N_MUT 80・22h・±4σ）どおり**なので投げ直し不要。出力は `DH_ult_nonarrow_sd4_seed*` で §4 の glob と一致。

## 3. CS / CH の ult はこれで投げる（p2 が PASS したら即)

走行中の p2 の引数から組み立てた。どちらも p2 に OVERRIDE が無いので ult にも付けない。

```bash
# CS（p2 = CS_p2_mut160_seed*、OVERRIDE なし）
qsub -l nodes=1:ppn=1:gpus=1:<host> -l walltime=22:00:00 \
  -v STAGE=ult,TAG=CS,SEED=$S,N_MUT=80,RUNTAG=sd4,PREV=_runs/paper_gateoff/CS_p2_mut160_seed$S \
  paper_gateoff_job.sh

# CH（p2 = CH_p2_mut150_seed*、OVERRIDE なし）
qsub -l nodes=1:ppn=1:gpus=1:<host> -l walltime=22:00:00 \
  -v STAGE=ult,TAG=CH,SEED=$S,N_MUT=80,RUNTAG=sd4,PREV=_runs/paper_gateoff/CH_p2_mut150_seed$S \
  paper_gateoff_job.sh
```

出力は `CS_ult_sd4_seed*` / `CH_ult_sd4_seed*` で §4 の `{tag}_ult_sd4_seed*` と一致する。

## 4. 聞きたいこと: DS の p2 の設定

DS の p2 は**まだ一度も走っていない**（`_runs` にあるのは `DH_p2_mut80` と走行中の `DH_p2_nonarrow` だけ）。
p1 が判定を通ったらすぐ投げたいので、次を決めてほしい:

- **N_MUT**: CS は 160、CH は 150、DH は 80。DS は p1 を 3000 粒子・N_MUT 100 で回している。p2 は何にするか
- **N_PART**: p1 を 3000 粒子にしたので、p2 も 3000 にするか既定 2000 のままか
- **RUNTAG**: §4 の glob が `DS_ult_wide80_sd4_seed*` なので **ult は `wide80_sd4`** になる。
  p2 は `wide80` でよいか（→ `DS_p2_wide80_seed*`）
- **OVERRIDE**: p1 と同じ `5:-15:20;6:-15:20;19:-15:20` でよいか（08a の「p2 は箱を絞らない」に沿うとそうなる）

## 5. 状態

- 走行 18 本。待機は DH ult 3 本と a45=0 mut120 の 6 本。**置き換えが尽きたら 15 本に収束**する
- GPU は 30 枚のうち自分 18・他ユーザーの確保のみ 6・空き 6。他ユーザーの待ちは 0 件
- λ 感度（`DH_pilot_lam1_m100_*`）は占有を下げるため取り下げ済み・未実施。本線の ult が走り始めてから投げ直す
- DS p1（3246 / 3253 / 3254）走行中。完走時は自動通知に判定 0〜4 と a33 < −5 の割合が入る（`a4d20d6`）
