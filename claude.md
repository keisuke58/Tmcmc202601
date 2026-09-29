## 基本情報

- 名前: ニシオカケイスケ / 西岡佳祐 / Keisuke Nishioka
- GitHub: keisuke58

## 解答スタイル

- 結論ファースト
- 簡潔でわかりやすく

## プロジェクト概要

- 口腔バイオフィルムの力学・ダイナミクスを扱う TMCMC + FEM パイプライン
- TMCMC でパラメータ推定 → DI / FEM 入力を生成 → Abaqus で応力・変形解析

## 主なディレクトリ

- `FEM/` : 有限要素解析用スクリプトとジョブ管理
- `FEM/_job_archive/` : Abaqus ジョブ一式（等方性デモ, DI 信頼区間, 異方性など）
- `FEM/docs/` : FEM パイプラインの数式付き PDF（fem_pipeline.pdf など）
- `docs/` : TMCMC 側のドキュメント（fem_pipeline.pdf のコピーも配置）

## 典型的なワークフロー

1. TMCMC スクリプトでパラメータ推定を実行
2. 推定結果（MAP / 信頼区間）から DI・FEM 入力ファイルを生成
3. FEM/ 以下のスクリプトで Abaqus ジョブを投入・結果を整理

## メモ

- このファイルは AI アシスタント（Claude, 他）がリポジトリ構造を把握するための基本情報メモです
- 追加で共有したいルールや注意点があれば、このファイルに追記してください

## Git / ドキュメント運用ルール

- 小さめの変更単位でこまめに commit する
- コード変更とドキュメント変更は可能なら commit を分ける
- 仕様やパイプラインを変えたら FEM_README.md / LaTeX docs も更新する
- ログや巨大出力ファイルは commit しない

## 実行・リソースに関するルール

- 重い TMCMC 推定はローカルで実行しない（remote サーバーで実行する）
- ローカルでは主に可視化・軽い検証だけ行う
- 生データは上書き禁止。派生データは別ディレクトリに保存する

## GPU クラスタ運用（copaam / stuttgart 等）— クラウド Claude 連携用

このリポジトリの `~/Tmcmc202601` は **copaam を含む全ノードで NFS 共有のホームディレクトリ**にある。
クラウド上の Claude セッションが `ssh copaam` で入った場合、`cd ~/Tmcmc202601` は
stuttgart01 などの GPU ノードと同じファイルを直接編集できる（rsync 不要）。

- **実体パスは `~/Tmcmc202601` であって `~/IKM_Hiwi/Tmcmc202601` ではない。**
  後者は 2026-03-05 で止まっている古いコピーで git 管理下にもない。混同注意。
- **このクラスタには PBS/Torque ジョブスケジューラがある（`qsub`/`qstat`/`pbsnodes`、server: copaam）。**
  GPU ジョブは `ssh` して直接バックグラウンド実行せず、必ず `qsub` を使うこと。
  他ユーザーのジョブと GPU が衝突しないよう、`nvidia-smi` の手動確認だけに頼らず
  `-l nodes=1:ppn=N:gpus=1:<hostname>` で GPU を要求すると Torque が空き GPU を自動割当する
  （`qstat -f <jobid>` の `exec_gpus` で確認できる）。
- **ノード構成**: stuttgart01-03 (RTX3090 x4/node), vancouver01-02 (RTX4090 x4/node),
  celtic01-04 (RTX2080Ti x4/node, celtic04 は down のことがある)。
- **既知の罠: PBS バッチジョブは `LD_LIBRARY_PATH=/usr/local/cuda/lib64`（システム CUDA）を
  自動設定しており、pip の `nvidia-cusparse-cu12` 等と衝突して JAX が GPU を見つけられなくなる**
  （`RuntimeError: Unable to load cuSPARSE`）。対話 ssh セッションではこの変数が未設定なので
  再現せず気づきにくい。ジョブスクリプト内で `unset LD_LIBRARY_PATH` してから python を呼ぶこと。
  診断は `data_5species/main/check_cuda_jax.py`。
- Python は miniforge/conda 管理。JAX 系は `klempt_fem2` env
  (`/home/nishioka/miniforge3/envs/klempt_fem2/bin/python3`)。
- PBS ジョブスクリプトのテンプレート: `data_5species/main/tmcmc_job.sh`（numba/CPU 版の例）、
  `data_5species/main/dh_prior_check_job.sh`（JAX/GPU 版・上記の罠への対処込みの例、2026-09-30 作成）。
  新しい GPU ジョブはこれをコピーして書き換えるのが早い。
- copaam の `~/.local/bin`（PATH 済み）に頻出操作のヘルパーを置いてある（2026-09-30 作成）:
  - `gpufree [host...]` — GPU 空き確認（省略時は stuttgart01-03 + vancouver01-02）
  - `pbsme` — 自分の PBS ジョブ一覧（`qstat -u -n1`）
  - `pbslog <jobid> [行数]` — jobid から Job_Name/出力先を自動解決してログを tail
    （PBS は実行中ジョブの stdout をバッファするため、完了/クラッシュ前はログが無いのが正常）

## 対話スタイル詳細

- 日本語で回答する（論文テキストだけ英語にする場合は指示する）
- 絵文字は基本的に不要
- 重要な部分は箇条書きや短い見出しで整理する
