# GPU 側 → クラウド側（2026-10-13 01:50 JST）: DH 仮 ult は投げ直さない／gate の ult walltime 確認

## 1. DH 仮 ult（`nonarrow_w3_noph_p14k_sd4_prov`）: 走っている分は走らせて、切れても投げ直さない（ユーザー判断）

12a §3 の「walltime で切れたら 18h で投げ直し」で投げ直した 3388-3390（stuttgart、walltime 18h）の状況:

| job | seed | ノード | 状態 |
|---|---|---|---|
| 3389 | 123 | stuttgart02 | 完走（10-13 01:21、11 stages、61797 s = 17.2 h、max logL −14.7） |
| 3388 | 42 | stuttgart01 | 走行中、01:37 の時点で経過 17:30 / 18:00 |
| 3390 | 7 | stuttgart03 | 走行中、01:37 の時点で経過 17:30 / 18:00 |

- 3388 と 3390 は、seed123 の実測（17.2 h）から見て walltime で切れる可能性がある。切れても **24h での投げ直しはしない**。
  - 投げ直すと終わるのは 10-14 の深夜〜10-15。本線（w4 p20k → ult）とほぼ同じ時刻なので、仮で先に図を作る意味がない。
  - 12f で投稿は急がないことになっている。12a・11l の「仮」は、本線が判定を通るまでのつなぎだった。
  - 仮は判定 4 が FAIL の p2 から始めた版なので、本線の代わりにもならない。
- 完走した場合は 11l の追記どおり、判定 → `final_runs/` に `_prov` で上げる。切れた場合は 12a §3 のとおり、ディレクトリに `_killed` を付けて残す（通知 3391 は afterany なので、どちらの場合も自動で報告が出る）。

### 追記（02:10）: 3388・3390 は walltime の前に完走した

- 3388（seed 42）は 01:50 に、3390（seed 7）は 01:49 に完走した。どちらも 63494 s = 17.6 h、11 stages で、walltime まで残り約 20 分だった。
- これで 3 seed そろった。判定は **PASS 0・1・2・2b・3、FAIL 4**
  - 判定 3: maxlogL 幅 0.14（−14.71 / −14.79 / −14.86）
  - 判定 4: 中央値幅/sd の上位は a24 0.63、a13 0.58、a34 0.55
- 片側の端の印は、seed 42 が a44・a22（箱 −10・−5）、seed 7 が a35・a15。
- 11l の手順（`final_runs/` に `_prov` で上げる、仮の図、DH.json）は、まだやっていない。

## 2. gate が投げる ult の walltime: 足りる見込み

実測から求めた 1 粒子 × 1 mutation × 1 stage あたりの時間:

| run | GPU | 粒子 × mut × stage | 時間 | 係数 |
|---|---|---|---|---|
| DH p2 w3 p20k（3 seed） | 4090 | 20000 × 120 × 12 | 18.7〜20.3 h | 2.3〜2.5 µs |
| DS ult p18k（3 seed） | 4090 | 18000 × 150 × 10 | 16.5〜17.7 h | 2.2〜2.4 µs |
| DS ult 5000（3 seed） | 4090 | 5000 × 150 × 10 | 9.3〜10.6 h | 4.5〜5.1 µs（粒子が少ないと GPU が埋まらない） |
| DH ult prov 5000 | 3090 | 5000 × 150 × 11 | 17.2 h | 7.5 µs |

| gate | 投げる ult | 見積もり（4090, 12 stage） | walltime | 判定 |
|---|---|---|---|---|
| 3399 | DH `nonarrow_w4_noph_p20k_sd4`（N_PART 20000, N_MUT 150） | 約 25 h（3090 なら約 38 h） | 48 h | 足りる |
| 3406 | DH `nonarrow_w4_noph_p25k_sd4`（N_PART 25000, N_MUT 150） | 約 31 h（3090 なら約 47 h） | 72 h | 足りる |

stage 数が 14 に増えても、4090 であれば余裕がある。どちらの gate も 12d の修正後なので、`N_PART` は ult に渡る（`tools/gate_submit_ult.sh` 133 行目）。

## 3. 要対応: どちらの gate も投入先が `NODE=vancouver01` で、空きが 2 枚しかない

`gpualloc` の結果（01:45）: vancouver01 の gpu1（ishihara）と gpu3（ikeda）は、qjupyterlab が確保したまま使っていない（HELD-IDLE）。そのため、3395・3396 が終わっても取れる GPU は **2 枚**だけ。

- 3399 が ult を 3 本投げると、3 本目は Q で止まり、最初の 2 本のどちらかが終わる（約 25 h 後）まで始まらない。
- 3406（p25k）の ult 3 本も同じ vancouver01 に並ぶので、p20k の ult が終わるまで待つ。
- 01:45 の時点で FREE だったのは、vancouver02 の gpu1・gpu3、vancouver03 の gpu1、stuttgart01 の gpu2・gpu3、stuttgart02 の gpu0。

**対応済み（ユーザー承認、02:00）**: `tools/gate_submit_ult.sh` の `NODE` で、seed ごとにノードを `;` 区切りで指定できるようにした（88c188e）。gate は起動時に git の最新版で実行し直すので、この変更は届く。そのうえで gate を差し替えた。前段のジョブは触っていない。

| 旧 gate | 新 gate | 前段 | NODE（seed 42 / 7 / 123） | walltime |
|---|---|---|---|---|
| 3399（qdel 済み） | **3407** | 3395-3397（DH p2 w4 p20k） | vancouver01 / vancouver01 / vancouver03（どれも前段が空ける 4090） | 48 h |
| 3406（qdel 済み） | **3408** | 3403-3405（DH p2 w4 p25k） | stuttgart03 ×3（前段が空ける 3090、見積もり 約 47 h） | 72 h |

ほかの変数（TAG・RUNTAG・N_MUT 150・N_PART・OVERRIDE）は旧 gate と同じ。`qstat -f` で確認した。
