# GPU 側報告 2026-10-18（DS ult を final_runs に、CS・DS・DH の回収テストを投入）

## 1. DS ult を final_runs に（ユーザー指示）
- `DS_ult_2c_raw_seed{42,7,123}` の 5 ファイル（config・logL・run_record・samples・theta_MAP）を
  `docs/revision/generated/final_runs/` にコピーした。判定は全 PASS（[gpu_2026-10-18_DS_ult.md](gpu_2026-10-18_DS_ult.md)）、max logL −96.80 / −96.84 / −96.94（幅 0.14）。
- polish はしていないので polish.json は無い。`final_theta_MAP/DS.json` はまだ作っていない（16j の定義は「磨いたあとの最大の点」）。

## 2. 回収テストの投入（16c §2 の形、ユーザー指示）
CH（3614〜3616）は 10/17 に終了済み。残りの CS・DS・DH を 3 seed ずつ。`RUNTAG=2c_raw`、`SRC_GLOB` は明示。

| jobid | 内容 | ノード・walltime |
|---|---|---|
| 3653・3654・3655 | DS seed 42・7・123（`SRC_GLOB='DS_ult_2c_raw_seed*'`） | vancouver01×2・vancouver02、36h |
| 3656・3657・3658 | DH seed 42・7・123（`SRC_GLOB='DH_p2_2c_raw_seed*'`、DH は p2 が最終） | vancouver03・vancouver02・vancouver03、48h（3657・3658 は磨き 3651・3652 の後で Q） |
| 3659・3660・3661 | CS seed 42・7・123（`SRC_GLOB='CS_ult_2c_raw_seed*'`） | stuttgart01・02・03、20h |

walltime の根拠: 元 run の実測（CS ult 7〜8h、DS ult 17〜19h、DH p2 21〜23h）× 1.5 ＋余裕。CH は ult 7.8h → 回収 10.5h（3090）だった。

## 3. 合成データ
- 10/13 の旧い CS の合成データ（`CS_ult_wide2_sd4_seed42` から作ったもの）を `CS_syn_n{1,0}_r0__old_ult_wide2_sd4/` に退避（消していない）。
- 投入前に CPU で作り直した（`--noise 1 --rng-seed 0`）:

| 条件 | 真値（max logL の seed） | 実データでの log L（組み直し） | 合成での log L | パーセンタイル | σ_viab | sha256 |
|---|---|---|---|---|---|---|
| CS | CS_ult_2c_raw_seed123 | −32.857（一致） | −21.174 | 70 | 0.113 | 30ea288c2f96 |
| DS | DS_ult_2c_raw_seed42 | −96.796（一致） | −49.817 | 60 | 0.059 | 23947cef913d |
| DH | DH_p2_2c_raw_seed42 | −51.413（一致） | −18.747 | **97** | 0.075 | c252c60ba159 |

- DH の実現値は 97 パーセンタイル（ノイズが当たりやすい側に出た）。乱数の種は手順どおり 0 のまま。解釈するときは注意。

## 4. 本数
GPU 11 本（磨き 3651・3652、回収 9 本）。15 本の内。
