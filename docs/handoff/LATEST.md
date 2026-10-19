# クラウド側 → GPU 側の最新の連絡

**GPU 側は `git pull` したら、まずこのファイルを読むこと。** 過去の連絡は同じフォルダに日付順で残す。

## いまの状態（2026-10-19、ここだけ読めば足りる）
- **データ・尤度**: Heine の生データ（`DATA_SOURCE=raw`）・2 通路（`TWO_CHANNEL=1`）、RUNTAG `2c_raw`。読み取りデータの run は `docs/revision/generated/final_runs_digitised/`（新しい数字には使わない）。
- **最終の run**（`docs/revision/generated/final_runs/`）: CH = ult a33w、CS = ult、**DS = ult**（10/18 追加）、**DH = p2**（ult は判定 4 FAIL、参考として置く、16l）。
- **走行中**（10/19 01:40 時点、GPU 13 本）:
  - 回収テスト CS・DS・DH 9 本（3653〜3661）。CS が 10/19 朝に最初に終わる見込み
  - DS ult の磨き seed42（3662）
  - DS a45=0 seed123 の磨き（3666。3651 は 8h 枠切れで出し直し）
  - CH の 1 日外し予測 Day 10・15（3667〜3674、19b）
- **最近済んだこと**: CH の回収テストと npy・polish.json 4 つ（18b・19a）、14c

## 今やること（上から優先）

| 日付 | ファイル | 要点 | 状態 |
|---|---|---|---|
| 2026-10-19 | [2026-10-19b_holdout_only.md](2026-10-19b_holdout_only.md) | **13f-5 だけやる（ユーザー決定）: CH、Day 10・15 を外した pilot→p1→p2、seed 42、本線と同じ箱、回収テストのあと（10/20 ごろ）。13g・13f-4 は見送り（査読対応の候補）** | 投入済み（3667〜3674、[gpu_2026-10-19_holdout_submit.md](gpu_2026-10-19_holdout_submit.md)） |
| 2026-10-19 | [2026-10-19a_recovery_npy.md](2026-10-19a_recovery_npy.md) | **回収テストの samples.npy・logL.npy が .gitignore で落ちていた → 例外を足した。GPU 側は pull 後に CH の npy を git add して push（CS・DS・DH も同様）** | CH 済（10/19）。CS・DS・DH はコピー時に git status で確認 |
| 2026-10-18 | [2026-10-18b_recovery_copy.md](2026-10-18b_recovery_copy.md) | **回収テストの出力（CH 今すぐ、CS・DS・DH は終わり次第）を `docs/revision/generated/recovery/` に 5 ファイル＋synthetic.json でコピー。CH p1 seed42・p2 seed7 の polish.json、DS ult の polish.json も final_runs に** | 一部済（CH と polish.json 4 つ、10/19）。CS・DS・DH 回収と DS ult seed42 の polish.json は終わり次第 |
| 2026-10-18 | [2026-10-18a_ds_final.md](2026-10-18a_ds_final.md) | **DS は ult で確定（DS.json は seed42 の粒子の最良値）。GPU 側へ: (1) DS の pilot・p1・p2 を final_runs に、(2) DS ult を top-k 3 で磨く、(3) DS a45=0 の磨きのあと final_runs と ln B を確定** | §1 済（67a0adf）。§2 は seed7・123 済、seed42 は 3662 走行中。§3 は seed42・7 済（−57.89 / −57.78）、seed123 は 3666 で出し直し中（3651 は 8h 枠切れ） |
| 2026-10-16 | [2026-10-16l_dh_p2_final.md](2026-10-16l_dh_p2_final.md) | **DH は p2 を最終に（選択肢 1）。確認: p2 は ult と同じ 2 通路の尤度で、狭めたのは ult の方。ただし p2 の D21/15 も seed で 9.6/13.5/18.0（実測 14.9）→ データで決まらない量と書く。GPU は DH p2・ult を final_runs にコピー。読み取りデータの run を final_runs_digitised/ に分離。pH 着** | 済（GPU 側: DH p2・ult を final_runs に、147f603）。選択肢 2 は DS の ult のあと検討 |
| 2026-10-16 | [2026-10-16k_bayes_raw.md](2026-10-16k_bayes_raw.md) | **ベイズ因子を生データ・2 通路で出し直す（ユーザー決定、案 A）**: 4 条件 × {全モデル, a45=0} × 3 seed、pilot・1000 粒子・n_mut 120（今の n_mut 40 は判定 (c) に落ちる）。**CS・CH の 12 本を先に、DS・DH はそのあと**。本線のあと、10/23 までが目安 | CS・CH 済（ln B −0.06 / +0.03）。DS・DH 済（+7.46 暫定 / +2.07、d412704）。残りは DS a45=0 seed123 の磨き（3666）→ DS の確定（18a §3） |

## 守る決まり（いつも有効）

| 日付 | ファイル | 要点 | 状態 |
|---|---|---|---|
| 2026-10-16 | [2026-10-16i_recovery_guard.md](2026-10-16i_recovery_guard.md) | 16h への対応: `recovery_run.py` が合成データの source_glob を SRC_GLOB（空なら GROUPS の既定、CS・CH は 2c_raw に更新）と比べ、違えば止まる。`recovery_job.sh` は走っているジョブのため変えていない | **有効**: 次の回収テストから |
| 2026-10-16 | [2026-10-16g_ds_p1_criterion3.md](2026-10-16g_ds_p1_criterion3.md) | **基準 3 は seed ごとの「見つかったいちばん高い max logL」で判定（ユーザー一任 → 案 1）。DS p1 は幅 1.70 で FAIL と正直に書く。DS p2 以降は止めない。今後の磨きは旧値・新値・大きい方を並べて報告** | **有効** |
| 2026-10-16 | [2026-10-16b_polish_faster.md](2026-10-16b_polish_faster.md) | **磨きを速く: top-k 3・`--ftol 1e-7`（このコミット）。DS p1 の磨き直し 3582〜3584 を qdel してこの設定で出し直す。DS p2 も seed ごとに磨いて gate は POLISH=0** | **有効**: すぐ |
| 2026-10-16 | [2026-10-16a_polish_converge.md](2026-10-16a_polish_converge.md) | **判定 3 は収束するまで磨いた値で（polish の maxiter 300 → 3000、このコミット）。DS の gate 3529 を hold、DS p1 を 3000 で磨き直して判定し直す。p2 は止めない** | **有効**: すぐ |
| 2026-10-13 | [2026-10-13e_criterion_d.md](2026-10-13e_criterion_d.md) | **判定 3（(d)）を磨いた max logL で判定（ユーザー決定、2 通路の結果を見る前）。判定 4 は 0.5 SD のまま、R̂ を並べる。polish に 2 通路と `--write-run`、check が polish.json を読む、gate は判定前に磨く（gpus=1 で投げる）** | **有効（規則）**: 判定 3 は polish 後の max logL、gate は gpus=1 |

## 記録（終わった・置き換わった・取り消した指示）

10/15 以前の指示と報告は [ARCHIVE_2026-10.md](ARCHIVE_2026-10.md) に移した（10/19）。

| 日付 | ファイル | 要点 | 状態 |
|---|---|---|---|
| 2026-10-15 | [2026-10-14c_weights_rerun_p2init.md](2026-10-14c_weights_rerun_p2init.md) | **重みの感度 12 本を生データの p2 の MAP から回し直す（`--init-glob`、出力 `weight_sensitivity/raw_p2init/`、CPU ジョブ）。14a は古い MAP から始めていた（DH の Pg 比 1.9）。DS は p2 のあと** | 済（12 / 12、DS は出し直した p2 から、d99cfd9。S_lambda は 230a0ff で反映）|
| 2026-10-13 | [2026-10-13g_weights.md](2026-10-13g_weights.md) | **重みの感度: CH・DH の 2 通路 p2 を重みなし（LAMBDA_PG=LAMBDA_LATE=LAMBDA_RARE=1、job に LAMBDA_RARE 追加）で 3 seed。13b → 13f のあと** | 見送り（19b、ユーザー決定。S9 で足りる） |
| 2026-10-13 | [2026-10-13f_box_holdout.md](2026-10-13f_box_holdout.md) | **質の確認 2 つ（2 通路の本線のあと）: 4. 箱を 1.5 倍に広げた p2（CH・CS、3 seed、`tools/widen_box.py`）、5. 1 日を外した予測の確認（CH、Day 10・15、`HOLDOUT_DAY`、`tools/holdout_eval.py`）** | 4 は見送り。5 は 19b で投入済み（3667〜3674） |
| 2026-10-17 | [gpu_2026-10-17_ds_edge_allow.md](gpu_2026-10-17_ds_edge_allow.md) | **GPU 側: DS p2 は全 PASS だが端の印 a44・a13・a25（3 seed とも）→ ユーザー決定で片側の限界として書き、gate を `EDGE_ALLOW=a44;a13;a25` で 3632 に仕掛け直し（3607 は起動前に qdel）** | 済（3632 が全 PASS → DS ult 投入 → 全 PASS で final_runs に） |
| 2026-10-16 | [2026-10-16j_cs_ult_polish.md](2026-10-16j_cs_ult_polish.md) | final_theta_MAP は「磨いたあとの最大の点」（CH.json = seed7 の磨いた θ −25.87、CS.json = seed123 の粒子の最良値 −32.86）。**CS ult の 3 seed を top-k 3 で磨く**（急がない） | 済（CS ult 3 seed の polish.json は final_runs に） |
| 2026-10-16 | [2026-10-16f_ds_p2_rerun_a.md](2026-10-16f_ds_p2_rerun_a.md) | **DS p2 の出し直しは (a)（ユーザー決定）: 同じ設定のまま 4090 に 3 seed、walltime 72h。磨き → gate → ult を依存でつなぎ直す** | 済（DS p2 全 PASS → ult へ） |
| 2026-10-16 | [2026-10-16e_ds_p2_lost.md](2026-10-16e_ds_p2_lost.md) | **最優先: DS p2（3519〜3521）が出力なしで終わった（walltime 40h 切れの可能性）。ログで原因と到達段を確認 → walltime を見積もり直して出し直し → 磨き → gate → ult。CH の磨き直しは続行、CS ult は全 PASS → final_runs へ** | 済（16f で出し直し） |
| 2026-10-16 | [2026-10-16c_ch_repolish_recovery.md](2026-10-16c_ch_repolish_recovery.md) | **空いている GPU で (1) CH の p1・p2・ult を 16b の設定で磨き直し → 判定し直して final_runs に polish.json、(2) そのあと CH の合成データ回収テスト（2 通路・生データ、3 seed、クラウドで dry-run 済み）。本線優先** | 済（CH の磨き直しは final_runs に、回収テスト CH は recovery/ に、19a） |

## GPU 側からの報告

| 日付 | ファイル | 要点 |
|---|---|---|
| 2026-10-19 | [gpu_2026-10-19_CH_p2_ho10.md](gpu_2026-10-19_CH_p2_ho10.md) | **自動通知**: run 終了。FAIL を含む群は回し直すまで解釈しない |
| 2026-10-19 | [gpu_2026-10-19_holdout_submit.md](gpu_2026-10-19_holdout_submit.md) | CH の 1 日外し予測 Day 10・15 を投入（3667〜3674）。submit_13f.sh を生データ・a33w の箱に合わせた |
| 2026-10-19 | [gpu_2026-10-19_recovery_copy_CH.md](gpu_2026-10-19_recovery_copy_CH.md) | 回収テスト CH と polish.json 4 つをコピー（18b）。3651 は 8h 枠切れ → 3666 で出し直し |
| 2026-10-18 | [gpu_2026-10-18_ds_ult_final_recovery_submit.md](gpu_2026-10-18_ds_ult_final_recovery_submit.md) | DS ult を final_runs に、回収テスト CS・DS・DH 9 本を投入、18a への対応 |
| 2026-10-18 | 磨きの報告: [DS ult seed7](gpu_2026-10-18_polish_DS_ult_seed7_k3.md)・[seed123](gpu_2026-10-18_polish_DS_ult_seed123_k3.md)、[DS a45=0 seed42](gpu_2026-10-18_polish_DS_pilot_a45zero_seed42_k3.md)・[seed7](gpu_2026-10-18_polish_DS_pilot_a45zero_seed7_k3.md) | 自動の報告（max logL の磨き直し） |
| 2026-10-18 | [gpu_2026-10-18_DS_ult.md](gpu_2026-10-18_DS_ult.md) | **自動通知**: run 終了。全群 PASS |
| 2026-10-17 | [2026-10-17b_status_remaining.md](2026-10-17b_status_remaining.md) | **いまの状態と投稿までの残り: CS・CH・DH は固まった。残りは DS・ベイズ因子・回収テスト・CS の磨き（GPU 待ち）と、語数（4000 超の見込み、削りの案 −340 語は採否待ち）・共著者の確認** |
| 2026-10-17 | [2026-10-17a_dh_tbd_fill.md](2026-10-17a_dh_tbd_fill.md) | **DH の \TBD を段 (iii) の数字で埋めた（本文・補足）。GPU 側へ: DH の pilot・p1（2c_raw）を final_runs にコピーして（表 S1・S2・コストの DH 行に要る）** |
| 2026-10-17 | [gpu_2026-10-17_DH_ult_decision.md](gpu_2026-10-17_DH_ult_decision.md) | **DH は p2（全 PASS）を最終の run に（ユーザー決定、いったん）。ult は判定 4 FAIL（seed で山が分かれる、D21/15 が 11.5〜18.9）で限界として書く。回し直しなし** |
| 2026-10-16 | [gpu_2026-10-16_DH_ult.md](gpu_2026-10-16_DH_ult.md) | **自動通知**: run 終了。FAIL を含む群は回し直すまで解釈しない |
| 2026-10-16 | [gpu_2026-10-16_CS_ult.md](gpu_2026-10-16_CS_ult.md) | **自動通知**: run 終了。全群 PASS |

## 連絡の決まり

- クラウド側は、GPU 側に伝えることを**毎回このフォルダにファイルで書いて push** する
  （チャット経由のコピペはしない）。このファイル（LATEST.md）の表のいちばん上が最新
- GPU 側の報告は、チャットに貼るか、`docs/handoff/` に `gpu_YYYY-MM-DD.md` として push する
