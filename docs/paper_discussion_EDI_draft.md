# E(DI) 文献裏付け — Discussion セクション案

> **Issue:** [#81](https://github.com/keisuke58/Tmcmc202601/issues/81)
> **参照:** [Literature Reinforcement §1](https://github.com/keisuke58/Tmcmc202601/wiki/Literature-Reinforcement#1-edi-モデルの文献的根拠強化)

---

## 挿入候補: Discussion 前半（E(DI) の正当化）

### パラグラフ 1: 因果チェーンの明示

```
Our constitutive law E(DI) maps Shannon diversity to elastic modulus. While no study has
directly measured Shannon diversity and elastic modulus simultaneously on the same
biofilm sample, the literature supports a three-step causal chain: (i) community
composition determines EPS composition (Flemming & Wingender 2010), (ii) EPS
composition determines biofilm mechanics (Billings et al. 2015; Peterson et al. 2015),
and (iii) sucrose-driven ecological shifts (diversity loss) correlate with 10–80×
stiffness reduction (Pattem et al. 2018). Our E(DI) mapping is thus a well-motivated
constitutive hypothesis that formalizes this chain.
```

### パラグラフ 2: E のレンジの文献的根拠

```
The elastic modulus range E ∈ [E_min, E_max] = [10, 1000] Pa is consistent with
reported biofilm stiffness: Billings et al. (2015) cite 0.1–100,000 Pa across
biofilm types; Pattem et al. (2018) report 14 kPa vs 0.55 kPa (≈25×) for
high- vs low-diversity oral biofilms; our fitted range 30–900 Pa falls within
these bounds. The variational structure of the underlying Hamilton-ODE model
ensures thermodynamic consistency automatically (Junker & Balzani 2021),
eliminating the need for separate entropy inequality verification.
```

### パラグラフ 3: Low diversity = soft の直接証拠

```
Koo et al. (2013) showed that S. mutans-dominated biofilms exhibit porous,
structurally weaker EPS compared to diverse communities. Houry et al. (2012)
demonstrated that second-species infiltration dramatically alters matrix
mechanics. These findings support our hypothesis that diversity loss
(dysbiosis) reduces effective stiffness, as captured by E(DI).
```

---

## 引用リスト（BibTeX 追加推奨）

```bibtex
@article{Flemming2010EPS,
  author = {Flemming, Hans-Curt and Wingender, Jost},
  title = {The biofilm matrix},
  journal = {Nature Reviews Microbiology},
  volume = {8}, number = {9}, pages = {623--633}, year = {2010},
  doi = {10.1038/nrmicro2415}
}

@article{Billings2015BiofilmMechanics,
  author = {Billings, Nicole and Birjiniuk, Avraham and Sampson, Timothy D. and others},
  title = {Material properties of biofilms — a review of methods for understanding
           permeability and mechanics},
  journal = {Reports on Progress in Physics},
  volume = {78}, number = {3}, pages = {036601}, year = {2015},
  doi = {10.1088/0034-4885/78/3/036601}
}

@article{Peterson2015Viscoelasticity,
  author = {Peterson, Brandon W. and others},
  title = {Species identity determines biofilm viscoelasticity},
  journal = {FEMS Microbiology Reviews},
  volume = {39}, number = {2}, pages = {234--245}, year = {2015}
}

@article{Koo2013SucroseBiofilm,
  author = {Koo, Hyun and Falsetta, Megan L. and Klein, Marlise I.},
  title = {The exopolysaccharide matrix: a virulence determinant of cariogenic biofilm},
  journal = {Journal of Dental Research},
  volume = {92}, number = {12}, pages = {1065--1073}, year = {2013},
  doi = {10.1177/0022034513504218}
}

@article{Pattem2018Stiffness,
  author = {Pattem, J. and others},
  title = {AFM measurements of biofilm stiffness},
  journal = {[要確認]},
  year = {2018}
}

@article{Houry2012SecondSpecies,
  author = {Houry, A. and others},
  title = {Bacterial swimmers that infiltrate and take over the biofilm matrix},
  journal = {Proceedings of the National Academy of Sciences},
  volume = {109}, number = {32}, pages = {13088--13093}, year = {2012},
  doi = {10.1073/pnas.1200791109}
}
```

---

## チェックリスト

- [ ] Pattem 2018 の正確な出典を確認（Literature Reinforcement に記載あり）
- [ ] 上記 3 パラグラフを paper LaTeX の Discussion に挿入
- [ ] BibTeX を .bib に追加
- [ ] 図表参照（Fig. X: E(DI) curve, Billings range）を追加
