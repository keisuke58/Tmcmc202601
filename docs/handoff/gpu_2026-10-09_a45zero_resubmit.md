# GPU 側報告 2026-10-09 04:0x（09a §0 の確認: **取り違えでした。CS 3 本を投げ直した**）

## 0. 3256-3261 は `a45zero_m120` だった — 私の qdel が誤り

09a の指摘どおりです。**私が `qstat -f` から `STAGE` / `TAG` / `SEED` だけを正規表現で抜き出し、
`N_MUT` / `RUNTAG` / `OVERRIDE` を見ていなかった**のが原因です（まさに 09a 末尾の推測どおり）。
「対応する pilot run は既に存在する」と書いたのは `*_pilot_a45zero_seed*`（mut80）を見ての誤りでした。

`_runs/paper_gateoff/` の実体を確認した結果:

| 群 | seed 42 | seed 7 | seed 123 |
|---|---|---|---|
| `CH_pilot_a45zero_m120` | **無し**（3259 を qdel） | **あり**（3260 が 10-08 11:18 完走） | **無し**（3261 を qdel） |
| `CS_pilot_a45zero_m120` | **無し**（3256） | **無し**（3257） | **無し**（3258） |
| `*_pilot_a45zero`（mut80） | あり | あり | あり | ← 2b FAIL、ln B には使えない |

欠落は **5 本**（CS 42/7/123、CH 42/123）。3260 だけは H から実行まで進んでいたので残っています。

## 1. CS 3 本は投げ直し済み（返事は待たない、の指示どおり）

```
qsub -l nodes=1:ppn=1:gpus=1:stuttgart01 -l walltime=09:00:00 \
     -v STAGE=pilot,TAG=CS,SEED=$S,N_MUT=120,RUNTAG=a45zero_m120,OVERRIDE='19:0:0' \
     paper_gateoff_job.sh
```

| jobid | seed | ノード | 状態 | 見込み |
|---|---|---|---|---|
| 3317 | 42 | stuttgart01（3090） | R | 約 3.5h（3260 の実測 3:35） |
| 3318 | 7 | stuttgart01 | R | 同 |
| 3319 | 123 | stuttgart01 | R | 同 |

投入後に `qstat -f` の `Variable_List` を**全文**出して 6 変数すべてを目視確認しました（同じ取り違えを防ぐため）。
`Override bound [19] -> [0, 0]` はログがバッファされるので完走後に確認して報告します。

## 2. CH 42/123 は枠が空いてから（ult 優先の指示どおり）

これで走行 **15 本（上限ちょうど）**です。CH p2（3307-3309）が 04:00〜05:00 に終わる見込みで、
09a §1 のとおり PASS なら **CH ult（`mut150_wide_noph_sd4`・N_MUT 150・OVERRIDE は p2 と同じ・
PREV=`CH_p2_mut150_wide_noph_seed$S`）に 3 枠を使う**ので、CH の a45zero 2 本はその次（DH p2 が
05:30〜07:30 に終わる見込み）に回します。

## 3. 09a の他の項目

- §1 CH ult: 了解。p2 が PASS なら返事を待たずに投入し、片側の端に新しく積む成分が出たら 08m の規則で p2 から回し直します。
- §2 08s の表: 了解。CH が終わったら CH の行だけで報告します。
- §3 旧 job 5 本: 了解、投稿後に `_archive/` へ。

## 4. ついでに `.gitignore`（ユーザー指示）

`data_5species/main/_runs/` を丸ごと ignore しようとしたら、**この下は 207 ファイルが既に tracked**
（`baseline_original_bounds` / `phibar_fix_*` / `stein2013_*` / `siddiqui_6sp_*` / `sm_*`）でした。
丸ごとだと今後それらに足したファイルが無視されるので、**commit しない大物だけ名前で外す**形にしました:

```
data_5species/main/_runs/paper_gateoff/
data_5species/main/_runs/*_20260930/
data_5species/main/_runs/gateoff_summary_20260930.*
```

`git ls-files ... | git check-ignore --stdin` が 0 件（tracked を一つも隠していない）を確認済み。
`git status` の untracked は 63 → 22 件になりました。残り 22 件は未コミットの図スクリプト 12 本
（`plot_paper_*.py`, `gen_heatmaps*.py` ほか）と `figS1_*` 6 件などで、08c で図を
`tools/make_paper_figures.py` に一本化した後の現役/旧版の区別がつかないため**触っていません**。
