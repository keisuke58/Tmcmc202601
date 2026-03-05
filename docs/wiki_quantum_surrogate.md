# 量子サロゲート概要

- 目的: TMCMC の高価な ODE 解析を量子回路推論で高速化するサロゲート（QNN）を提供
- アーキテクチャ: Ry+Rz 回転＋リング型 CNOT のハードウェア効率化アンサッツ
- 観測量: 全量子ビットの Z 期待値の平均（回帰タスク向け）
- 実装: tmcmc/program2602/quantum_surrogate.py（QuantumBiofilmSurrogate）

## 関連スクリプト

- ベンチマーク: tmcmc/program2602/benchmark_quantum_speedup.py
  古典ソルバ vs 量子サロゲートの 1 サンプル時間を比較して速度向上係数を出力

- 訓練デモ: tmcmc/program2602/train_quantum_surrogate.py
  合成データで QNN を訓練し、重みを保存（quantum_weights.json）

- TMCMC 実験: tmcmc/program2602/run_quantum_tmcmc_experiment.py
  量子サロゲートを TMCMC の計算に組み込み、全粒子で速度評価

- 解析補助: tmcmc/program2602/quantum_kernel_pca.py
  量子カーネルの PCA 可視化など（参考ツール）

## 使い方

### ベンチマーク

```bash
python tmcmc/program2602/benchmark_quantum_speedup.py
```

### 訓練（重み保存）

```bash
python tmcmc/program2602/train_quantum_surrogate.py
# 実行後に quantum_weights.json を生成
```

### TMCMC 連携実験

```bash
python tmcmc/program2602/run_quantum_tmcmc_experiment.py
```

量子重みが存在しない場合はランダム初期値で推論し、精度低下の警告を出します。
精度検証時は事前に train スクリプトで重みを作成し、tmcmc/program2602/ 配下に配置してください。

## 管理方針

- 実験成果物（quantum_weights.json、量子関連画像、data_5species/_runs/ 配下）は .gitignore で除外
- コード（.py）は Git 管理に追加済み

## 参考

- 速度向上目安（ローカル環境の一例）
  古典: 約 1.2s/サンプル、量子サロゲート: 約 0.002s/サンプル → 500–700×
