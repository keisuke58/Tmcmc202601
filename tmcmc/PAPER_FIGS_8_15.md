## Paper Fig. 8–15 再現メモ（case II / hierarchical updating）

対象論文: `tmcmc/Bayesian updating of bacterial microfilms under hybrid uncertainties with a novel surrogate model - Kopie.pdf`

このリポジトリでは `tmcmc/case2_tmcmc_linearization.py` が **Case II（階層推定: M1→M2→M3→M3_val）** の入口です。

---

## 最短コマンド（Fig.8–15の生成を狙う）

Fig.8–15の条件（Table 3）に寄せるため、paper条件固定で実行します。

```bash
python tmcmc/run_pipeline.py \
  --mode paper \
  --models M1,M2,M3,M3_val \
  --seed 123 \
  --run-id paper_caseII_seed123 \
  --lock-paper-conditions \
  --use-paper-analytical
```

生成先:
- `tmcmc/_runs/paper_caseII_seed123/REPORT.md`
- `tmcmc/_runs/paper_caseII_seed123/figures/*.png`

---

## Fig.8–15 と生成物の対応

PDFから抽出したキャプション（要旨）:
- **Fig.8**: θ(1)（5パラメータ）の posterior samples（M1, 2-species）
- **Fig.9**: M1 のモデル出力（posterior samples の “shaded” + data scatter）
- **Fig.10**: θ(2)（5パラメータ）の posterior samples（M2, 2-species）
- **Fig.11**: M2 のモデル出力（posterior samples の “shaded” + data scatter）
- **Fig.12**: θ(3)（4パラメータ）の posterior samples（M3, 4-species）
- **Fig.13**: M3 のモデル出力（posterior samples の “shaded” + data scatter）
- **Fig.14**: 同定したパラメータ平均 vs 真値（error bar = posterior std）
- **Fig.15**: M3_val（time-dependent antibiotics）のモデル出力（posterior samples の “shaded” + data scatter）

コード上の対応（このリポジトリでの出力名）:

- **Fig.8** → `figures/posterior_M1.png`
- **Fig.9** → `figures/PaperFig09_posterior_predictive_M1.png`

- **Fig.10** → `figures/posterior_M2.png`
- **Fig.11** → `figures/PaperFig11_posterior_predictive_M2.png`

- **Fig.12** → `figures/posterior_M3_TMCMC.png`
- **Fig.13** → `figures/PaperFig13_posterior_predictive_M3.png`

- **Fig.14** → `figures/PaperFig14_parameter_mean_vs_true.png`

- **Fig.15** → `figures/PaperFig15_posterior_predictive_M3_val.png`

注:
- Paperの “shaded” は本実装では **posterior draw の 5–95% 区間（band）+ median 線** で表現しています。
- 速度のため posterior draw は一部のみサンプリング（デフォルト最大120本）しています。

