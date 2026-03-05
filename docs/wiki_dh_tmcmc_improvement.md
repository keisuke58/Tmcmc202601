# DH (Dysbiotic HOBIC) TMCMC 精度改善 — Wiki 概要

> **詳細:** [DH_TMCMC_IMPROVEMENT_PLAN.md](DH_TMCMC_IMPROVEMENT_PLAN.md)

---

## 概要

Dysbiotic HOBIC (DH) 条件は 20 free params（他条件 9–15）で探索空間が広く、現状 MAP RMSE 0.075（他条件 0.054–0.063）。本ページは DH の TMCMC 精度改善のための施策・実行手順をまとめる。

## 推奨実行順序

| 順序 | 施策 | 期待効果 | 所要時間 |
|------|------|----------|----------|
| 1 | 500p, 30 stages, use_exp_init | MAP RMSE 0.075 → 0.06–0.07 | 1–2 時間 |
| 2 | GNN prior | 収束安定化、ESS 向上 | 数時間 |
| 3 | posterior_frac=0.7 で DeepONet 再学習 | overlap 17/20 → 18–19/20 | 1日 |
| 4 | NUTS（DeepONet 経由） | acceptance 向上 | 数時間 |

## 最小構成の実行コマンド

```bash
cd Tmcmc202601/data_5species/main
python estimate_reduced_nishioka.py \
  --condition Dysbiotic --cultivation HOBIC \
  --n-particles 500 --n-stages 30 \
  --use-exp-init \
  --checkpoint-every 5
```

## 関連

- [DH_TMCMC_IMPROVEMENT_PLAN.md](DH_TMCMC_IMPROVEMENT_PLAN.md) — 詳細プラン
- [TMCMC Guide](https://github.com/keisuke58/Tmcmc202601/wiki/TMCMC-Guide) — 基本ガイド
