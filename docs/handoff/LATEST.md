# クラウド側 → GPU 側の最新の連絡

**GPU 側は `git pull` したら、まずこのファイルを読むこと。** 過去の連絡は同じフォルダに日付順で残す。

## いまの状態（2026-10-19、ここだけ読めば足りる）
- **データ・尤度**: Heine の生データ（`DATA_SOURCE=raw`）・2 通路（`TWO_CHANNEL=1`）、RUNTAG `2c_raw`。読み取りデータの run は `docs/revision/generated/final_runs_digitised/`（新しい数字には使わない）。
- **最終の run**（`docs/revision/generated/final_runs/`）: CH = ult a33w、CS = ult、**DS = ult**（10/18 追加）、**DH = p2**（ult は判定 4 FAIL、参考として置く、16l）。
- **走行中**（10/19 01:40 時点、GPU 13 本）: 回収テスト CS・DS・DH 9 本（3653〜3661、CS が 10/19 朝に最初に終わる見込み）。DS ult の磨き seed42（3662）。DS a45=0 seed123 の磨き（3666、3651 は 8h 枠切れで出し直し）。CH の 1 日外し予測 Day 10・15（3667〜3674、19b）。**済**: CH の回収テストと npy・polish.json 4 つ（18b・19a）、14c。

## 有効な指示（上から優先）

| 日付 | ファイル | 要点 | 状態 |
|---|---|---|---|
| 2026-10-19 | [2026-10-19b_holdout_only.md](2026-10-19b_holdout_only.md) | **13f-5 だけやる（ユーザー決定）: CH、Day 10・15 を外した pilot→p1→p2、seed 42、本線と同じ箱、回収テストのあと（10/20 ごろ）。13g・13f-4 は見送り（査読対応の候補）** | 投入済み（3667〜3674、[gpu_2026-10-19_holdout_submit.md](gpu_2026-10-19_holdout_submit.md)） |
| 2026-10-19 | [2026-10-19a_recovery_npy.md](2026-10-19a_recovery_npy.md) | **回収テストの samples.npy・logL.npy が .gitignore で落ちていた → 例外を足した。GPU 側は pull 後に CH の npy を git add して push（CS・DS・DH も同様）** | CH 済（10/19）。CS・DS・DH はコピー時に git status で確認 |
| 2026-10-18 | [2026-10-18b_recovery_copy.md](2026-10-18b_recovery_copy.md) | **回収テストの出力（CH 今すぐ、CS・DS・DH は終わり次第）を `docs/revision/generated/recovery/` に 5 ファイル＋synthetic.json でコピー。CH p1 seed42・p2 seed7 の polish.json、DS ult の polish.json も final_runs に** | 一部済（CH と polish.json 4 つ、10/19）。CS・DS・DH 回収と DS ult seed42 の polish.json は終わり次第 |
| 2026-10-18 | [2026-10-18a_ds_final.md](2026-10-18a_ds_final.md) | **DS は ult で確定（DS.json は seed42 の粒子の最良値）。GPU 側へ: (1) DS の pilot・p1・p2 を final_runs に、(2) DS ult を top-k 3 で磨く、(3) DS a45=0 の磨きのあと final_runs と ln B を確定** | §1 済（67a0adf）。§2 は seed7・123 済、seed42 は 3662 走行中。§3 は seed42・7 済（−57.89 / −57.78）、seed123 は 3666 で出し直し中（3651 は 8h 枠切れ） |
| 2026-10-16 | [2026-10-16l_dh_p2_final.md](2026-10-16l_dh_p2_final.md) | **DH は p2 を最終に（選択肢 1）。確認: p2 は ult と同じ 2 通路の尤度で、狭めたのは ult の方。ただし p2 の D21/15 も seed で 9.6/13.5/18.0（実測 14.9）→ データで決まらない量と書く。GPU は DH p2・ult を final_runs にコピー。読み取りデータの run を final_runs_digitised/ に分離。pH 着** | 済（GPU 側: DH p2・ult を final_runs に、147f603）。選択肢 2 は DS の ult のあと検討 |
| 2026-10-16 | [2026-10-16k_bayes_raw.md](2026-10-16k_bayes_raw.md) | **ベイズ因子を生データ・2 通路で出し直す（ユーザー決定、案 A）**: 4 条件 × {全モデル, a45=0} × 3 seed、pilot・1000 粒子・n_mut 120（今の n_mut 40 は判定 (c) に落ちる）。**CS・CH の 12 本を先に、DS・DH はそのあと**。本線のあと、10/23 までが目安 | CS・CH 済（ln B −0.06 / +0.03）。DS・DH 済（+7.46 暫定 / +2.07、d412704）。残りは DS a45=0 seed123 の磨き（3666）→ DS の確定（18a §3） |
| 2026-10-16 | [2026-10-16i_recovery_guard.md](2026-10-16i_recovery_guard.md) | 16h への対応: `recovery_run.py` が合成データの source_glob を SRC_GLOB（空なら GROUPS の既定、CS・CH は 2c_raw に更新）と比べ、違えば止まる。`recovery_job.sh` は走っているジョブのため変えていない | **有効**: 次の回収テストから |
| 2026-10-16 | [2026-10-16g_ds_p1_criterion3.md](2026-10-16g_ds_p1_criterion3.md) | **基準 3 は seed ごとの「見つかったいちばん高い max logL」で判定（ユーザー一任 → 案 1）。DS p1 は幅 1.70 で FAIL と正直に書く。DS p2 以降は止めない。今後の磨きは旧値・新値・大きい方を並べて報告** | **有効** |
| 2026-10-16 | [2026-10-16b_polish_faster.md](2026-10-16b_polish_faster.md) | **磨きを速く: top-k 3・`--ftol 1e-7`（このコミット）。DS p1 の磨き直し 3582〜3584 を qdel してこの設定で出し直す。DS p2 も seed ごとに磨いて gate は POLISH=0** | **有効**: すぐ |
| 2026-10-16 | [2026-10-16a_polish_converge.md](2026-10-16a_polish_converge.md) | **判定 3 は収束するまで磨いた値で（polish の maxiter 300 → 3000、このコミット）。DS の gate 3529 を hold、DS p1 を 3000 で磨き直して判定し直す。p2 は止めない** | **有効**: すぐ |
| 2026-10-15 | [2026-10-14c_weights_rerun_p2init.md](2026-10-14c_weights_rerun_p2init.md) | **重みの感度 12 本を生データの p2 の MAP から回し直す（`--init-glob`、出力 `weight_sensitivity/raw_p2init/`、CPU ジョブ）。14a は古い MAP から始めていた（DH の Pg 比 1.9）。DS は p2 のあと** | 済（12 / 12、DS は出し直した p2 から、d99cfd9。S_lambda は 230a0ff で反映）|
| 2026-10-13 | [2026-10-13g_weights.md](2026-10-13g_weights.md) | **重みの感度: CH・DH の 2 通路 p2 を重みなし（LAMBDA_PG=LAMBDA_LATE=LAMBDA_RARE=1、job に LAMBDA_RARE 追加）で 3 seed。13b → 13f のあと** | 見送り（19b、ユーザー決定。S9 で足りる） |
| 2026-10-13 | [2026-10-13f_box_holdout.md](2026-10-13f_box_holdout.md) | **質の確認 2 つ（2 通路の本線のあと）: 4. 箱を 1.5 倍に広げた p2（CH・CS、3 seed、`tools/widen_box.py`）、5. 1 日を外した予測の確認（CH、Day 10・15、`HOLDOUT_DAY`、`tools/holdout_eval.py`）** | 4 は見送り。5 は 19b で投入済み（3667〜3674） |
| 2026-10-13 | [2026-10-13e_criterion_d.md](2026-10-13e_criterion_d.md) | **判定 3（(d)）を磨いた max logL で判定（ユーザー決定、2 通路の結果を見る前）。判定 4 は 0.5 SD のまま、R̂ を並べる。polish に 2 通路と `--write-run`、check が polish.json を読む、gate は判定前に磨く（gpus=1 で投げる）** | **有効（規則）**: 判定 3 は polish 後の max logL、gate は gpus=1 |

## 記録（終わった・置き換わった・取り消した指示）

10/12 以前の指示と報告は [ARCHIVE_2026-10.md](ARCHIVE_2026-10.md) に移した（10/19）。

| 日付 | ファイル | 要点 | 状態 |
|---|---|---|---|
| 2026-10-17 | [gpu_2026-10-17_ds_edge_allow.md](gpu_2026-10-17_ds_edge_allow.md) | **GPU 側: DS p2 は全 PASS だが端の印 a44・a13・a25（3 seed とも）→ ユーザー決定で片側の限界として書き、gate を `EDGE_ALLOW=a44;a13;a25` で 3632 に仕掛け直し（3607 は起動前に qdel）** | 済（3632 が全 PASS → DS ult 投入 → 全 PASS で final_runs に） |
| 2026-10-16 | [2026-10-16j_cs_ult_polish.md](2026-10-16j_cs_ult_polish.md) | final_theta_MAP は「磨いたあとの最大の点」（CH.json = seed7 の磨いた θ −25.87、CS.json = seed123 の粒子の最良値 −32.86）。**CS ult の 3 seed を top-k 3 で磨く**（急がない） | 済（CS ult 3 seed の polish.json は final_runs に） |
| 2026-10-16 | [2026-10-16f_ds_p2_rerun_a.md](2026-10-16f_ds_p2_rerun_a.md) | **DS p2 の出し直しは (a)（ユーザー決定）: 同じ設定のまま 4090 に 3 seed、walltime 72h。磨き → gate → ult を依存でつなぎ直す** | 済（DS p2 全 PASS → ult へ） |
| 2026-10-16 | [2026-10-16e_ds_p2_lost.md](2026-10-16e_ds_p2_lost.md) | **最優先: DS p2（3519〜3521）が出力なしで終わった（walltime 40h 切れの可能性）。ログで原因と到達段を確認 → walltime を見積もり直して出し直し → 磨き → gate → ult。CH の磨き直しは続行、CS ult は全 PASS → final_runs へ** | 済（16f で出し直し） |
| 2026-10-16 | [2026-10-16c_ch_repolish_recovery.md](2026-10-16c_ch_repolish_recovery.md) | **空いている GPU で (1) CH の p1・p2・ult を 16b の設定で磨き直し → 判定し直して final_runs に polish.json、(2) そのあと CH の合成データ回収テスト（2 通路・生データ、3 seed、クラウドで dry-run 済み）。本線優先** | 済（CH の磨き直しは final_runs に、回収テスト CH は recovery/ に、19a） |
| 2026-10-15 | [2026-10-15e_cs_ult_a33_allow.md](2026-10-15e_cs_ult_a33_allow.md) | **CS の ult を今の箱のまま投入（ユーザー決定）。a33 は尤度が平らで片側の上限として書く。gate に `EDGE_ALLOW=a33`（このコミットで追加）。全 PASS なら final_runs へ** | 済（CS ult 全 PASS、final_runs に） |
| 2026-10-15 | [2026-10-15d_final_runs_CH.md](2026-10-15d_final_runs_CH.md) | **確定した CH の生データの run（pilot・p1・p2 a33w・ult a33w・cold）を final_runs に 5 ファイルずつコピーして push。クラウドで CH の図を先に作る** | 済 |
| 2026-10-14 | [2026-10-14b_ch_a33_widen.md](2026-10-14b_ch_a33_widen.md) | **CH の a33（θ[5]）の下限を広げて p2 から回し直す（ユーザー決定）。gate 3523・CH の ult を止める、cold p2 3522 は続ける。OVERRIDE に `5:-15:20`、RUNTAG `2c_raw_a33w`** | 済（CH ult a33w 全 PASS） |
| 2026-10-14 | [2026-10-14a_weights_raw.md](2026-10-14a_weights_raw.md) | **重みの感度（raw・2 通路、決定論的な最適化）の残り 8 本（B 重みなし・C λψ=2、4 条件）を GPU で。A はクラウドで済み。本線と重ならない空きの GPU 1 枚で、終わったら表と json を push** | 済（12 本、CPU ジョブ。初期値が古い MAP だったので 14c で回し直し） |
| 2026-10-13 | [2026-10-13p_cs_p2.md](2026-10-13p_cs_p2.md) | **CS の p2: λψ は既定（1.5）のまま、張り付いた側を広げた箱で p1 の判定のあと投入（ユーザー決定）。CS は組成と生きた割合が両立しない（限界として書く）** | 済（CS p2 全 PASS） |
| 2026-10-13 | [2026-10-13o_p2_boxes.md](2026-10-13o_p2_boxes.md) | **raw の p2 の箱: CH・DS・DH は端に張り付いた側だけ広げた OVERRIDE で、p1 の判定のあと投入。CS は λψ の確認待ちで保留（13p）。cold p2 も CH の箱で** | 済（p2 は 4 条件とも投入済み） |
| 2026-10-13 | [2026-10-13n_p2_hold.md](2026-10-13n_p2_hold.md) | **raw の p2 は箱をクラウドが 13o で渡すまで投入しない（13b の箱だと raw では端に張り付く）。13m の作り直し → p1 続行 → p2 保留。cold p2 も 13o 待ち** | 済（13o・13p で全条件の p2 の箱が出た） |
| 2026-10-13 | [2026-10-13m_live_sigma.md](2026-10-13m_live_sigma.md) | **生きた割合の σ をバイオフィルム単位（5 枚 = 1 個）に。p2 の前に build_raw_dataset.py を作り直す。pilot・p1 は影響なし。cold p2 も σ_viab を確認** | 済（p2 はこの σ で回した） |
| 2026-10-13 | [2026-10-13l_p2_cold.md](2026-10-13l_p2_cold.md) | **CH の p2 を warm start なし（`P2_COLD=1`、RUNTAG `2c_raw_cold`、seed 42）で 1 本。段階分けの偏りの確認（ユーザー承認）。warm の p2 と compare_run_groups で比べる** | 済（CH_p2_2c_raw_cold、判定 3・4 FAIL） |
| 2026-10-13 | [2026-10-13k_tools_raw.md](2026-10-13k_tools_raw.md) | **13j §5 の道具（fit_metrics_2c・psi_table_runs）はクラウドで対応済み。0a9cef7 を pull したら build_raw_dataset.py をジョブの起動と重ならないときに 1 回** | 済 |
| 2026-10-13 | [2026-10-13j_raw_go.md](2026-10-13j_raw_go.md) | **raw で投入（ユーザー決定、最優先）: digitised 2c は **CH・CS・DS・DH すべて止める**（追記 2）。**読み取りデータは digitised_dataset/ に隔離、既定 raw（追記 3）**。13i の置き場所のあと、`DATA_SOURCE=raw TWO_CHANNEL=1`、RUNTAG `2c_raw` で 4 条件を pilot から（設定は 13b と同じ）。gate に DATA_SOURCE を追加済み（pull してから gate）。GPU 側の道具に data_dir_for** | 済（raw の本線に移行） |
| 2026-10-13 | [2026-10-13i_raw_data_fetch.md](2026-10-13i_raw_data_fetch.md) | **生データ 2 つをサーバーに置く: b2751e9 から `git show` で取り出し → convert_raw_heine.py → build_raw_dataset.py。git add しない。raw での投入はまだ（→ 13j で投入に変更）** | 済（GPU に配置済み） |
| 2026-10-13 | [2026-10-13h_raw_data.md](2026-10-13h_raw_data.md) | **Heine の生データ（qPCR・LIVE/DEAD）が入った。`--data-source raw` / `DATA_SOURCE=raw` を追加（既定は今まで）。最終の run は生データで。今の 2c は止めない。切り替えの時期はユーザー判断。GPU 側の道具にも data_dir_for を** | 済（13i・13j に引き継ぎ） |
| 2026-10-13 | [2026-10-13d_pilot_v2_fix.md](2026-10-13d_pilot_v2_fix.md) | **パイロット v2 の CS の λ_ch3 が 2.0 → estimator と同じ 1.5 に訂正。v2 の CS を回し始めていたら pull して回し直し。CH・DS・DH と 13b の GPU ジョブは影響なし** | 済 |
| 2026-10-13 | [2026-10-13c_reply_fit_improve.md](2026-10-13c_reply_fit_improve.md) | **相談への返事: ψ^data の項は尤度に入れない（同じデータの二重計上、確認用に使う）。パイロットは推定と同じ尤度（重み・σ）と実際の箱で。日ごとの σ は今回やらない。2 通路の GPU はそのまま** | 済 |
| 2026-10-13 | [gpu_2026-10-13_fit_improve.md](gpu_2026-10-13_fit_improve.md) | **GPU 側 → クラウド: 「2 通路よりよく当てる」の検討。パイロットに `--psi-data W` を追加し CPU で収束まで掃引中。ψ^data を ch4 に入れるべきか相談** | 済（返事は 13c） |
| 2026-10-13 | [2026-10-13b_two_channel_go.md](2026-10-13b_two_channel_go.md) | **古い尤度のジョブを全部止める（DS 予備 ult・DH 本線と予備・回収テスト）。2 通路（TWO_CHANNEL=1、RUNTAG 2c）で 4 条件を pilot からやり直す。粒子数・箱は今通った設定から。gate にも TWO_CHANNEL=1** | 置き換え（13j: digitised の 2 通路は全停止、raw でやり直し） |
| 2026-10-13 | [2026-10-13a_two_channel.md](2026-10-13a_two_channel.md) | **2 通路の組成の尤度（生きた菌 ↔ φψ、全菌 ↔ φ）を `--two-channel` / `TWO_CHANNEL=1` で用意した。既定は今までと同じ。まだ投入しない（パイロットの結果を見てユーザーが決める）** | 置き換え（13b） |

## GPU 側からの報告

| 日付 | ファイル | 要点 |
|---|---|---|
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
| 2026-10-15 | [gpu_2026-10-15_CH_ult.md](gpu_2026-10-15_CH_ult.md) | **自動通知**: run 終了。全群 PASS |
| 2026-10-15 | [gpu_2026-10-15_cs_gate_polish_gpu.md](gpu_2026-10-15_cs_gate_polish_gpu.md) | **CS の gate 3546 も 8h で kill。gate の polish は `JAX_PLATFORMS=cpu` で CPU で走っていた。polish を seed ごとに GPU（3553-3555）→ CS の gate 3557（POLISH=0）。gate_submit_ult.sh を修正（29e21f3）、DH も polish 3558-3560 → gate 3561 に切り替え** |
| 2026-10-15 | [gpu_2026-10-15_gate_CH_ult.md](gpu_2026-10-15_gate_CH_ult.md) | **CH a33w の p2 が全群 PASS → ult 3547-3549 を投入（gate の git push が vancouver03 で失敗したので手で push）** |
| 2026-10-15 | [gpu_2026-10-15_2c_raw_p2_DH.md](gpu_2026-10-15_2c_raw_p2_DH.md) | **自動通知**: run 終了。全群 PASS |
| 2026-10-15 | [gpu_2026-10-15_2c_raw_a33w_p2_CH.md](gpu_2026-10-15_2c_raw_a33w_p2_CH.md) | **自動通知**: run 終了。全群 PASS |
| 2026-10-15 | [gpu_2026-10-15_cs_a33_flat.md](gpu_2026-10-15_cs_a33_flat.md) | **CS の a33 は −15 でも下端（尤度が平ら）→ 広げず片側の上限で書く提案。CS の gate 3527 が polish 中に walltime 4h で kill、CS の gate を 8h で投げ直し（3546）、CH a33w の gate 3544 も 8h に延長** |
| 2026-10-15 | [gpu_2026-10-15_2c_raw_p2_CH_cold.md](gpu_2026-10-15_2c_raw_p2_CH_cold.md) | **自動通知**: run 終了。FAIL を含む群は回し直すまで解釈しない |
| 2026-10-15 | [gpu_2026-10-15_2c_raw_p2_CS.md](gpu_2026-10-15_2c_raw_p2_CS.md) | **自動通知**: run 終了。全群 PASS |
| 2026-10-15 | [gpu_2026-10-15_2c_raw_p2_CH.md](gpu_2026-10-15_2c_raw_p2_CH.md) | **自動通知**: run 終了。全群 PASS |
| 2026-10-13 | [gpu_2026-10-13_2c_raw_pilot_DS.md](gpu_2026-10-13_2c_raw_pilot_DS.md) | **自動通知**: run 終了。FAIL を含む群は回し直すまで解釈しない |
| 2026-10-13 | [gpu_2026-10-13_2c_raw_pilot_DH.md](gpu_2026-10-13_2c_raw_pilot_DH.md) | **自動通知**: run 終了。FAIL を含む群は回し直すまで解釈しない |
| 2026-10-13 | [gpu_2026-10-13_2c_raw_pilot_CH.md](gpu_2026-10-13_2c_raw_pilot_CH.md) | **自動通知**: run 終了。FAIL を含む群は回し直すまで解釈しない |
| 2026-10-13 | [gpu_2026-10-13_2c_raw_pilot_CS.md](gpu_2026-10-13_2c_raw_pilot_CS.md) | **自動通知**: run 終了。FAIL を含む群は回し直すまで解釈しない |
| 2026-10-13 | [gpu_2026-10-13_13g_prep.md](gpu_2026-10-13_13g_prep.md) | **13g 準備済み・未投入**（`submit_13g.sh`、RMSE の内訳は新しい `fit_metrics_2c.py`）。ジョブは変化なし（p2 12・polish 3 が R）。リポジトリ整理 |
| 2026-10-13 | [gpu_2026-10-13_status_2c_p2.md](gpu_2026-10-13_status_2c_p2.md) | **中間報告**: 2c の p2 12 本＋gate 4 本を投入（DS は判定 4 FAIL のままユーザー判断で、a44 下限を −5 に）。p1 の polish 3 本実行中。ψ の表を通知に追加。13f は準備済み・未投入 |
| 2026-10-13 | [gpu_2026-10-13_DH_ult_prov.md](gpu_2026-10-13_DH_ult_prov.md) | **自動通知**: run 終了。FAIL を含む群は回し直すまで解釈しない |
| 2026-10-13 | [gpu_2026-10-13_DS_ult_p18k.md](gpu_2026-10-13_DS_ult_p18k.md) | **自動通知**: run 終了。FAIL を含む群は回し直すまで解釈しない |

## 連絡の決まり

- クラウド側は、GPU 側に伝えることを**毎回このフォルダにファイルで書いて push** する
  （チャット経由のコピペはしない）。このファイル（LATEST.md）の表のいちばん上が最新
- GPU 側の報告は、チャットに貼るか、`docs/handoff/` に `gpu_YYYY-MM-DD.md` として push する
