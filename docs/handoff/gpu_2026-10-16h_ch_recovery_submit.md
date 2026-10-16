# GPU 側報告 2026-10-16h（CH の合成データ回収テストを投入、旧い合成データの罠）

## 投入（16c §2、ユーザー承認で前倒し）
DS の残りの段はすべて vancouver 指定の依存でつないであるので、stuttgart03 の空きで先に回した。CH ult の磨き直しは 3 本とも済み。

| jobid | 内容 | ノード・walltime |
|---|---|---|
| 3614 | CH 回収テスト seed 42（`TAG=CH SRC_GLOB='CH_ult_2c_raw_a33w_seed*' RUNTAG=2c_raw`） | stuttgart03・20h |
| 3615 | 同 seed 7 | stuttgart03・20h |
| 3616 | 同 seed 123 | stuttgart03・20h |

出力は `_runs/recovery/CH_syn_n1_r0_2c_raw_seed*`。終わったら 16c §2 のとおり `checks`・`beta_final`・`max_logL_minus_truth` を確かめて push する。

## 罠: 旧い合成データが再利用されるところだった
- `recovery_job.sh` は `_runs/recovery/<TAG>_syn_n<NOISE>_r<RNG>/synthetic.json` が**あればそのまま使う**（SRC_GLOB は作るときにしか効かない）。
- GPU には 10/13 の `CH_syn_n1_r0/synthetic.json` が残っていて、中身は旧い run（`CH_ult_mut150_wide2_noph_sd4_seed7`、max logL −8.65、σ_viab 0.128、読み取りデータ・旧尤度）から作ったものだった。
  16c のコマンドのまま投げると、これで推定していた。
- 旧いものは `_runs/recovery/CH_syn_n1_r0__old_ult_mut150_wide2_noph_sd4/` に退避（消していない）。
  新しい合成データを投入前に CPU で作り、クラウドの dry-run と一致することを確かめた:
  真値 `CH_ult_2c_raw_a33w_seed123` の MAP、実データで −26.523、合成で **−20.946**（72 パーセンタイル）、σ_viab 0.095、sha256 `b066fd70733f`。
- **CS にも 10/13 の `CS_syn_n1_r0`・`CS_syn_n0_r0` が残っている**（CH の `n0` も）。CS・DS・DH の回収テストを出すときは、同じように退避してから作る。
  スクリプトで防ぐなら、`recovery_job.sh` が既存の synthetic.json の `source_glob` と SRC_GLOB を比べて、違えば止まるようにするのがよい（未実装）。

## 本数
GPU 14 本（DH ult 3、DS p2 3、CH の磨き 5、CH 回収 3）。DS の ult（3 本）が動く 10/18 頃には CH の磨きが終わっている見込みで、15 本に収まる。
