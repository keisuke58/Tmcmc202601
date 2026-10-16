# GPU 側報告 2026-10-17a: DH の段 (i)・(ii) を final_runs に（17a）、CH の polish.json のコピー漏れを補充、ベイズ因子（16k）の CS・CH 12 本を投入

## まとめ
1. **17a は済み**: DH の pilot・p1 を final_runs に置いた（3478222）。元の run とバイト単位で一致することを確認済み。
2. **コピー漏れを直した**: CH の磨き直し（16c §1）の polish.json 7 本が、final_runs に入っていなかった。今回追加した（cf73c17）。
3. **ベイズ因子（16k）**: CS・CH の 12 本を投入した。5 本は走行中、7 本は依存で待機している。
4. **クラウド側に注意してほしい点**: DH pilot は判定 2b で FAIL している（下の §1）。表 S1・S2 に載せるときは、そのことを書く必要がある。

## 1. 17a: DH pilot・p1 を final_runs に
| run | 入れたファイル | 判定（`check_paper_runs.py --glob`、final_runs で実行） |
|---|---|---|
| `DH_pilot_2c_raw_seed{7,42,123}` | config・logL・run_record・samples・theta_MAP（polish.json は元から無い） | **FAIL 2b**（次元あたりの移動 3.2〜3.6 回、基準 5）。それ以外は PASS で、max logL の幅は 0.90 |
| `DH_p1_2c_raw_seed{7,42,123}` | 上の 5 ファイル＋polish.json（10/16 の top-k 3 で磨いたもの） | 全 PASS。判定 3 は粒子の最良値で幅 0.64。polish は 3 seed とも未収束 |

- DH pilot の 2b FAIL は n_mut 40 のためで、16k に書かれている CS・CH の pilot（2.7〜3.0 回/次元）と同じ理由。
  段 (i) の箱を決めるための run なので回し直しはしていないが、表に載せるときは注記が要る。
- `polish_prev_20261016.json`（その前の磨き）は final_runs に入れていない。
- DS の段 (i)〜(iii) は、ult が通ってからまとめて置く。

## 2. CH の polish.json（16c §1 のコピー漏れ）
16c には「磨いた run は final_runs の同じ run にも polish.json をコピーする」とあったが、されていなかった。
磨き直しが終わっている 7 本（どれも top-k 3・3000 反復・収束済み）を追加した。

| 段 | seed 7 | seed 42 | seed 123 |
|---|---|---|---|
| p1 | 追加 | **磨いている途中**（3612） | 追加（3596、10/17 01:03 に終了） |
| p2 a33w | **磨いている途中**（3592） | 追加 | 追加 |
| ult a33w | 追加 | 追加 | 追加 |

- final_runs で判定し直すと、CH の p1・p2 a33w・ult a33w はすべて全 PASS。
  - p1・p2 の判定 3 は、まだ 3 seed がそろっていないので粒子の最良値で判定している（幅は 0.11 と 0.23）。
  - ult は磨いた値で判定していて、幅は 0.18（粒子の最良値では 0.59）。
- **check の表示の注意**: p1・p2 は「polish が未収束（0/3 seed）」と表示されるが、未収束なのではない。final_runs にまだ polish.json が無い seed があるだけ（`_npol` が欠けた seed を数えない表示の癖）。
- 3592・3612 が終わったら、残りの 2 本もコピーする。CH p1 seed42 の run の側には、10/14 の古い polish.json（top-k 10・未収束）が残っているが、3612 が上書きする。

## 3. 16k: ベイズ因子の CS・CH 12 本を投入（ユーザー了承済み）
GPU の枠が 5 本空いていて、DS の残りの段はすべて vancouver02・03 指定の依存でつながっている（磨き 3604〜3606 → gate 3607 → ult。gate の `NODE=vancouver03;vancouver03;vancouver02`）。
10/16 の例外（DS の分を足しても 15 本に収まる）を満たすので、DS の ult を待たずに入れた。

設定は 16k のとおり: pilot・1000 粒子・`N_MUT=120`・`DATA_SOURCE=raw TWO_CHANNEL=1`。a45 = 0 のほうは `OVERRIDE=19:0:0`。
3621 で、`RUNTAG`・`OVERRIDE`・`N_MUT` がジョブに正しく渡っていることを `qstat -f` で確認した。

| jobid | ノード | 条件 | seed | RUNTAG | 状態 |
|---|---|---|---|---|---|
| 3620 | stuttgart02 | CS | 42 | 2c_raw_m120 | R |
| 3621 | stuttgart02 | CS | 42 | 2c_raw_a45zero_m120 | R |
| 3622 | stuttgart02 | CS | 7 | 2c_raw_m120 | R |
| 3623 | stuttgart01 | CS | 7 | 2c_raw_a45zero_m120 | R |
| 3624 | stuttgart01 | CS | 123 | 2c_raw_m120 | R |
| 3625 | celtic01 | CS | 123 | 2c_raw_a45zero_m120 | H（afterany 3620） |
| 3626 | celtic01 | CH | 42 | 2c_raw_m120 | H（afterany 3621） |
| 3627 | celtic01 | CH | 42 | 2c_raw_a45zero_m120 | H（afterany 3622） |
| 3628 | celtic02 | CH | 7 | 2c_raw_m120 | H（afterany 3623） |
| 3629 | celtic02 | CH | 7 | 2c_raw_a45zero_m120 | H（afterany 3624） |
| 3630 | celtic01 | CH | 123 | 2c_raw_m120 | H（afterany 3625） |
| 3631 | celtic02 | CH | 123 | 2c_raw_a45zero_m120 | H（afterany 3626） |

- 待機中の 7 本は、それぞれ前の 1 本に依存させてある。そのため同時に走る GPU ジョブは 15 本を超えない（いまちょうど 15 本）。
- walltime は stuttgart が 10h、celtic（2080Ti）は 16h。
- **終わる時刻は未確認**。16k の見込み（1 本 5h 前後）どおりなら、stuttgart の 5 本は 10/17 の朝に終わる。
  ただし celtic の速さはまだ測っていない。3630・3631 は 2 段目の待ちなので、終わるのは 10/18 中〜10/19 未明になりそう。10/23 の目安には間に合う見込み。
- 1 本目が終わったら、`log_evidence` が数値か・`override_bounds` に `19:0:0` があるか・移動が 5 回/次元を超えるかを確認する。
  12 本がそろったら、判定と ln Ẑ・ln B を `gpu_<日付>_bayes_raw.md` にまとめる。24 本分を final_runs に置くのは DS・DH のあと。
- DS・DH の 12 本は、DS が ult に入って枠が空いてから入れる。DH の SRC は `DH_p2_2c_raw` を前提にしている（16l）。

## 4. 走っているジョブ
- DS p2（3601〜3603）は約 11h 経過で、72h 枠のうち。後段（3604〜3611）は依存で待機している。
- CH の磨きは 3592（p2 a33w seed7）と 3612（p1 seed42）が走行中。3596（p1 seed123）は 10/17 01:03 に正常に終わった（Exit 0、自動通知 0e5da1a）。
- CH の回収テスト（3614〜3616）と CS ult の磨き（3617・3619）は走行中。
