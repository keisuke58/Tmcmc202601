# GPU 側報告 2026-10-09 03:4x（ログ整理・hold の残骸を qdel・走行 12 本の状況。指示を仰ぎたい点が §4）

## 1. ログを `logs/pbs/` に整理した（削除していない・移動のみ）

ユーザー指示「log とかきれいにして」。**1 件も消していない。**

| 移動元 | 移動先 | 件数 |
|---|---|---|
| `data_5species/main/*.log` | `data_5species/main/logs/pbs/{paper_gateoff,dh_prior_check,other}/` | 193（840K） |
| リポジトリ直下 `*.log`（notify_cloud, polish_maxlogL） | `logs/pbs/` | 36（148K） |

- `main/` 直下は 341 → 148 ファイル。`*.log` と `logs/` は両方 `.gitignore` 済みなので git は汚れない。
- `nife/results/` 配下のログ（54 件ほど）は結果ディレクトリと対応しているので触っていない。
- 既存の `main/logs/`（heine_kegg 系 7.9M）もそのまま。

今後も溜まらないよう、**現役の 4 本の job スクリプトだけ** `#PBS -o` の宛先を `logs/pbs/` に変えた
（`paper_gateoff_job.sh` / `dh_prior_check_job.sh` / `jax_gpu_job_template.sh` / `gateoff_logL_recheck_job.sh`）。
`-o` の相対パスは qsub した cwd 基準なので、`cd` の直後に `mkdir -p logs/pbs` も足した
（`logs/` は gitignore なので clone 直後に無く、無いと PBS が log を spool に落として失われる）。
**走行中の 3298-3312 の出力先は投入時に確定しているので影響なし**（従来の `main/` 直下に出る）。

## 2. hold のまま残っていた 6 本を qdel した（ユーザー承認済み）

10-08 06:18 投入で `Hold_Types=u` のまま残っていたもの。対応する pilot run は既に存在するので、
仮に走っても `既に存在する: ...` で即終了する分。

- 3256 / 3257 / 3258 = `STAGE=pilot TAG=CS` seed 42 / 7 / 123
- 3259 / 3261 = `STAGE=pilot TAG=CH` seed 42 / 123
- 3262 = `notify_cloud`（上記に `beforeany` 依存、永久に発火しない孤児）

キューは **走行 12 本 + 通知待ち 4 本（3313-3316、依存待ちの H で正常）** になった。同時本数の上限（15）に余裕 3。

## 3. 走行中 12 本（全部 `gpualloc` で BUSY 確認済み。03:32 時点）

全部 `STAGE=p2`・5000 粒子・pH なし（08r の指示どおり）。走行中は PBS が stdout をバッファするので
ログはまだ出ない（仕様）。ETA は同設定の過去 run の実測から。

| 群 | jobid | ノード | 経過 | walltime | 過去の実測 | ETA |
|---|---|---|---|---|---|---|
| DS `wide80_p5k_wide2` | 3298 / 3299 / 3300 | vancouver03 ×2, stuttgart02 | 5:16 | 22h | p3k が 11 stage / 9.0〜9.3h（3 粒子千） | **13〜15h（5 粒子千なので ×1.7）→ 10-09 昼頃** |
| DH `nonarrow_a45_noph` | 3304 / 3305 / 3306 | vancouver01 ×2, vancouver02 | 4:13 | 22h | `nonarrow_a11` が 11 stage / 5.8〜7.7h | **05:30〜07:30** |
| CH `mut150_wide_noph` | 3307 / 3308 / 3309 | stuttgart02 ×3 | 4:13 | 9h | `mut150_wide` が 6 stage / 4.3〜5.7h | **04:00〜05:00（最も早い）** |
| CS `mut160_wide2` | 3310 / 3311 / 3312 | vancouver02, stuttgart03 ×2 | 3:47 | 12h | `mut160_wide` が 7 stage / 7.4〜7.9h | **07:30〜08:00** |

walltime はどれも ETA の 1.4 倍以上あり、08s で起きた walltime 不足の事故は起きない。

## 4. 指示を仰ぎたい点

1. **CH p2（3307-3309）が 1 時間前後で終わる。** 判定（0〜4・2b）が PASS なら 08k / 08r の規則どおり
   **返事を待たずに CH ult（`N_MUT=150`、箱は平均 ± 4σ）へ進めてよいか**。08r は「ult は N_MUT 150」まで
   書いてあるが、pH なしの p2 から ult に渡すときの RUNTAG の付け方（`mut150_wide_noph_sd4` でよいか）が未定。
2. **08s の比較表（pH あり / なし）**は pH なし側が走行中なので、CH が終わった時点で CH の行から埋める予定。
   それでよいか、4 条件そろうまで待つか。
3. **`det_map_*.sh` / `final_check_job.sh` / `tmcmc_job.sh` / `tmcmc_joint_job.sh` の 5 本が
   `cd /home/nishioka/IKM_Hiwi/Tmcmc202601/...`（2026-03-05 で止まった古いコピー、git 管理外）を向いている。**
   今回は現役スクリプトと混ぜないため**変更していない**。直す（`$HOME/Tmcmc202601` に向ける）か、
   使っていないなら `_archive/` に退避するか、指示をもらいたい。投稿（〜10-16）後でもよいと思っている。

## 5. 判定の現状（`tools/check_paper_runs.py`、全 44 群）

走行中 4 群に関係する既存 run だけ:

| 群 | 判定 | 備考 |
|---|---|---|
| `CH_p2_mut150_wide` | PASS 6 / FAIL 0 | pH あり。今の `_noph` の比較相手 |
| `CS_p2_mut160_wide` | PASS 6 / FAIL 0 | 同上 |
| `DS_p1_wide80_p3k` | PASS 6 / FAIL 0 | 今の DS p2 の PREV |
| `DS_p2_wide80_p3k` | PASS 4 / **FAIL 2** | 中央値幅、maxlogL 幅 0.72 → 5 粒子千版が再試行 |
| `DH_p2_nonarrow` / `_a11` | PASS 5 / **FAIL 1** | 中央値幅 → `a45_noph` が再試行 |
| `DH_ident_prior0` / `prior6` | PASS 3 / **FAIL 3** | 加えて判定 5（prior なしは探索不足）も FAIL |

FAIL を含む群は 44 群中 24。群ごとの一覧が必要なら出す。
