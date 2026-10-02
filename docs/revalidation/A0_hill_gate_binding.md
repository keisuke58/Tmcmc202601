# A0 手順1 — Hill ゲートは事後領域で binding しているか

調査日: 2026-09-25 / スクリプト: `tools/hill_gate_binding_check.py`（N=300/ラン）

ゲート定義: `I_Pg ← h(φ̄_Fn)·I_Pg`, `h(x)=xⁿ/(Kⁿ+xⁿ)`, K=0.05, n=4（`improved_5species_jit.py`）。
各事後サンプルを「ゲートあり(K=0.05)」と「ゲートなし(K=0)」で再計算して比較した。

> **対象範囲の注意**: 提出版 `docs/nishioka_paper_publish.tex`（15 パラメータ、Phase 2 = `jax_ode_nuts_*_20260320_*`）
> の事後サンプルと GPU TMCMC (N_p=10,000) のコードはこのリポジトリに存在しない。
> 以下は repo 内に残る 20 パラメータ版 TMCMC の事後（`nishioka_latex20260218.tex` 系）での結果。

## 1. binding の有無

| ラン | h（観測時刻 day1→21, 事後中央値） | ゲート除去時の max\|ΔPg\|（中央値 / 95%） | 判定 |
|---|---|---|---|
| DH 1k30 (N=1000×30) | 0.139, 0.004, 0.002, 0.002, 0.002, 0.002 | 0.443 / 0.505 | **強く binding（閉）** |
| DH sweep baseline | 0.025, 0.001, 0, 0, 0, 0 | 0.471 / 0.534 | **強く binding（閉）** |
| DH sweep K.05n4 (mild bounds) | 0.019, 0, 0, 0, 0, 0 | 0.010 / 0.389 | 閉だが大半のサンプルで効果小 |
| DS posterior | 0.014 → 0.184（後期に開く） | 0.017 / 0.105 | 部分的（最終時刻で 58% が h<0.5） |
| CH posterior | 0.013 → 0.577（後期に開く） | 0.015 / 0.111 | 部分的（最終時刻で 49% が h<0.5） |
| CS posterior | ≤0.03 全期間 | 0 / 0 | 無関係（Pg 相互作用がロック） |

**DH ではゲートは「閾値スイッチ」ではなく「Pg 抑制器」として働いている。**
wide bounds の MAP は a₃₅≈17–27 と大きく、ゲートを外すと φ̄_Pg が 0.48–0.53 に暴走する。
ゲートが day 3 以降ほぼ閉じたまま（h≈0.002–0.03）で、これを抑えている。
その結果、DH MAP（1k30）は day 21 の surge を再現していない:

| day | 1 | 3 | 6 | 10 | 15 | 21 |
|---|---|---|---|---|---|---|
| model φ̄_Fn | 0.035 | 0.034 | 0.021 | 0.018 | 0.018 | 0.018 |
| data Fn | 0.002 | 0.002 | 0.004 | 0.050 | 0.120 | 0.180 |
| model φ̄_Pg | 0.033 | 0.159 | 0.016 | 0.015 | 0.014 | 0.014 |
| data Pg | 0.002 | 0.002 | 0.004 | 0.003 | 0.012 | 0.108 |

## 2. 付随して見つかったバグ（後処理でゲートが抜けている）

`data_5species/main/estimate_reduced_nishioka.py:3850` の後処理用ソルバーに `K_hill`/`n_hill` が渡されていない。
そのため既定値 `K_hill=0.0` で計算され、推定（ゲートあり）と `fit_metrics.json`・自動生成図（ゲートなし）でモデルが食い違う。

6 ラン全てで `fit_metrics.json` の MAP RMSE がゲートなし再計算値と 5 桁一致:

| ラン | fit_metrics.json | 再計算 ゲートあり | 再計算 ゲートなし |
|---|---|---|---|
| baseline_original_bounds (a₃₅=17.3) | 0.2230 | **0.1353** | 0.2230 |
| _sweeps/K0.05_n4.0 (a₃₅=3.56) | 0.1557 | **0.1604** | 0.1557 |
| DH 1k30 | 0.2218 | 0.1266 | 0.2218 |
| DH sweep baseline | 0.2401 | 0.1351 | 0.2401 |
| DS posterior | 0.1115 | 0.0913 | 0.1115 |
| CH posterior | 0.0899 | 0.0896 | 0.0899 |

**影響**: `nishioka_latex20260218.tex` の「mild bounds で RMSE 30% 改善（0.223→0.156）」（l.67, 345, 1804）は
この 2 ランの `fit_metrics.json` そのもの。推定に使ったモデル（ゲートあり）で評価すると 0.135→0.160 で、
**mild bounds のほうが 19% 悪い**（結論が逆転）。

## 3. 提出版論文への含意（未検証・要確認）

- 提出版 `nishioka_paper_publish.tex` は Hill ゲートに一切言及していないが、図生成コード
  （`docs/generate_fig2_phi_version.py`, `data_5species/docs/fig_ph_independent_validation.py`）は K=0.05, n=4 を固定で使う。
- 一方 `estimate_reduced_nishioka_jax.py` の既定は **n_hill=2.0** で、`run_jax_ode_nuts.sh` は上書きしていない。
  → 推定 n=2 / 図 n=4 の食い違いの可能性。実際の Phase 1/2 ランの config.json で要確認。
- Phase 2 のラン ID が図スクリプト間で 2 系統ある（`..._015113` 系: `generate_fig_so_sensitivity_gpu.py`、`..._043505` 系: `generate_fig2_phi_version.py`）。
- `hamilton_ode_jax.py` は 726afa7 で 15 パラメータ版になり `simulate_0d_full` が消えたが、図スクリプトは 20 次元 θ と `simulate_0d_full` を前提としており HEAD では動かない。

## 4. 次の手順（A0 手順2 以降の候補）

1. ローカル（`~/IKM_Hiwi/Tmcmc202601/data_5species/main/_runs/jax_ode_nuts_*`）で、提出版 Phase 1/2 の
   `config.json` の `K_hill`/`n_hill` を確認し、本スクリプトを 15 パラメータ JAX ソルバーで回す。
2. `estimate_reduced_nishioka.py:3850` に `K_hill=args.K_hill, n_hill=args.n_hill` を追加し、既存ランの
   `fit_metrics.json` と図を再生成（再推定は不要、後処理のみ）。
3. DH でゲートを抑制器として使う解（a₃₅ 大 + h≈0）を許すかどうかを、B1（事前の見直し）と合わせて判断。
