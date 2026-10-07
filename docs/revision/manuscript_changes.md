# 原稿の書き直し — 差し替え用 LaTeX（v1, 2026-10-07）

> **2026-10-07 追記: 0916 版のソースに適用済み** → `docs/revision/manuscript/nishioka_biofilm_tmcmc.tex`
> （起点の無変更版は commit 015d627。差分は `git diff 015d627 -- docs/revision/manuscript/`）。
> pdflatex でエラー・未定義参照なしでコンパイル済み（17 ページ、Abstract 203 語）。
>
> 適用時に**このメモに無かった修正**を 2 つ足した（実装を読んで確認）:
> 1. **「ψ を実測に固定」は実装と違う。** fix-ψ の段では ψ が全種に同じ値で掛かるので正規化で消え、
>    尤度は組成だけ。ODE の中の ψ も動く（「ヤコビアンを 2n+2 → n+2 に半減」は成り立たない）。
>    → Phase 1 = 組成のみ、Phase 2 = 組成＋生存率（＋HOBIC は pH）と書き直した。Abstract・Intro・§5・
>    Algorithm・表の見出し・Discussion・Conclusions すべて。
> 2. **pH は Phase 2 の尤度に入っている**（`paper_gateoff_job.sh` で HOBIC は `--lambda-ch5 0.3`、
>    観測モデルは `pH = 7.5 − 0.74·So − 0.48·Vei`（生きている割合）、σ=0.15）。
>    → §6.7 の pH 検証（回帰式・図）は本文から外してコメントに残し、見出しを Post hoc consistency checks に。
>    尤度の節に pH チャネルを明記。**pH 関係式の出典は要確認**（`\TBD{source of the pH relation}`）。
>
> 投稿前に決めること: Keywords が 7 個（BMB は 4–6）、タイトル（D1）、Funding の文言。

**対象**: 共著者に回した 2026-09-16 版（Drive `Nishioka_biofilm_tmcmc_final_0916.pdf`）。
その LaTeX ソースはこの repo にも Drive にも無い（`docs/nishioka_paper_publish.tex` は 06-04 版で、著者・所属・
Introduction の文言が 0916 版と違う）。**最新のソースに下の段落を貼り替える形で使う。**

方針:
- モデルは**最初からゲートなし**として書く（ゲートには言及しない）。式は §2 のまま（Hill は元から本文に無い）
- 数値は ult の結果待ち。`\TBD{...}` で囲み、PDF で目立つようにする（プリアンブルに下の定義を足す）
- 番号は共著者レビューの仕分け（`1030_Masterarbeit/notes/paper_review_20260925_coauthors.md`）に合わせた

```latex
% preamble
\newcommand{\TBD}[1]{\textcolor{red}{\textbf{[#1]}}}
```

---

## 0. タイトル・著者（D1–D3）

- タイトル案（D1、共著者で決める）: `GPU-accelerated Bayesian inference of effective interaction parameters in multispecies oral biofilms via TMCMC`
- 著者順（D2、Meisam 氏と合意済み）: Nishioka¹ʼ⁴, Klempt¹, Geisler¹, Mukherjee²ʼ³, Heine²ʼ³, Doll-Nikutta²ʼ³, Stiesch²ʼ³, Szafrański²ʼ³, Soleimani¹, Junker¹
- 所属（D3）: Nishioka に 2（MHH）と 3（NIFE）も付ける → `Nishioka^{1,2,3,4}`
- 責任著者: Keisuke Nishioka, `keisuke.nishioka@stud.uni-hannover.de`（sn-jnl では `\email{}` と `\author*`）

## 1. Abstract（C1・C2・C3、数値）

```latex
We present a GPU-accelerated Bayesian framework for inferring the interaction parameters of a five-species oral biofilm model derived from the extended Hamilton principle.
The symmetric $5\times5$ interaction matrix is parametrised by its 15 independent entries.
Inference proceeds in stages: the viability fraction $\psi_i$ is first fixed to experimentally measured membrane-intact ratios, and is then released while the viability data enter the likelihood as an additional channel.
The forward model is compiled with JAX and evaluated for all particles in parallel on a GPU, giving a \TBD{$\sim$200}$\times$ speed-up over the CPU baseline.
No interaction parameter is fixed to zero a priori.
Applied to in-vitro time-course data of Heine et al.\ under four combinations of community state (commensal vs.\ dysbiotic) and cultivation (static vs.\ dynamic, the latter in a HOBIC flow reactor), the inferred effective parameters reproduce the observed trajectories, including the late \textit{P.~gingivalis} increase under dysbiotic dynamic cultivation (RMSE \TBD{0.05--0.12}).
The Fn--Pg coefficient is positive only in that condition (90\% interval \TBD{[+0.8, +5.2]}) and indistinguishable from zero in both commensal conditions.
Posterior predictive simulations without \textit{F.~nucleatum} suppress the late Pg increase with probability \TBD{0.87--0.90}, yielding a directly testable prediction.
```

## 2. Introduction

- **株**: `Pg ATCC 20709` → `Pg DSM 20709`（Heine et al. の表記）
- 3 つめの戦略の文（「iteratively narrowed priors … convergence in 5–7 stages」）を差し替え:

```latex
Third, we infer the posterior in stages with progressively narrowed prior boxes and verify every stage with independent seeds and explicit convergence criteria (Section~\ref{sec:convergence}).
```

## 3. §2.5 Biological interaction network（C14、A4）

冒頭段落を差し替え:

```latex
Figure~\ref{fig:network} summarises previously reported or biologically supported interactions among the five taxa, compiled from Heine et al.~\cite{Heine2025PeriImplant} and the literature; it is used only to interpret the inferred parameters, not as a constraint.
The listed pairs rest on different kinds of evidence (metabolic cross-feeding, co-aggregation, pH modification, peptide utilisation), which Table~\ref{tab:network} states for each pair.
Because $\mathbf{A}$ is symmetric by construction, an inferred coefficient $a_{ij}$ describes an undirected effective interaction between species $i$ and $j$; the arrows in Figure~\ref{fig:network} represent metabolite flows reported in the literature, not directions recovered by the model.
All 15 entries of $\mathbf{A}$ are estimated, so that no reported interaction is imposed and no unreported one is excluded.
```

- 図 1 の凡例: `Solid blue: experimentally confirmed pathways` → `Solid: previously reported or biologically supported interactions considered in the interpretation of the model`
- 本文・表・図の `Vei→Pg`・`Fn→Pg`・`So→Vei` などは**すべて `Vei–Pg`・`Fn–Pg`・`So–Vei`**（en dash）に。方向は代謝物の流れとして書くときだけ使う

## 4. §3.2 Likelihood（B5、N3）— 全面差し替え

```latex
\subsection{Likelihood}
\label{sec:likelihood}

The observations are the species fractions at Days~3, 6, 10, 15 and 21, normalised to unit sum at each time point; Day~1 serves as the initial condition and is excluded from the likelihood.
The model predicts volume fractions $\phi_i(t)$ together with a void fraction that is not observed; predictions are therefore renormalised to species fractions,
$\hat{y}_i(t_k;\btheta) = \phi_i(t_k)\big/\sum_{j}\phi_j(t_k)$, before comparison.
Model time is mapped linearly to experimental days.
The initial condition is the normalised Day-1 observation, with a lower bound of $10^{-3}$ on each fraction.

We use a weighted Gaussian log-likelihood
\begin{equation}
\ln p(\yobs\mid\btheta)
= -\frac12\sum_{k=1}^{N_t}\sum_{i=1}^{5} w_{ik}\,
\frac{\bigl(y_{\mathrm{obs},i}(t_k)-\hat{y}_i(t_k;\btheta)\bigr)^2}{\sigma_i^2} + \text{const},
\label{eq:likelihood}
\end{equation}
where $\sigma_i$ is a species-specific observation noise obtained by error propagation from the replicate spread ($N=8$ biological replicates per condition)~\cite{Heine2025PeriImplant}.
The weights $w_{ik}$ are listed in Table~\ref{tab:weights}: Pg is up-weighted ($\lambda_{\mathrm{Pg}}=5$) because it is a low-abundance species whose late increase is the main dynamic feature of the dysbiotic data; the last two time points are up-weighted ($\lambda_{\mathrm{late}}=3$); and species whose mean fraction is below 5\% are down-weighted ($\lambda_{\mathrm{rare}}=0.1$).
In the stages with free $\psi$, the viability data enter as a second Gaussian channel with weight $\lambda_{\psi}$ (Table~\ref{tab:weights}).
Treating the compositional data with independent Gaussian errors ignores the unit-sum constraint; we assess its effect in Section~\TBD{B4 の感度解析 — Meisam 氏と相談}.
```

```latex
\begin{table}[htbp]\centering\small
\begin{tabular}{@{}lll@{}}\toprule
Weight & Value & Applies to\\\midrule
$\lambda_{\mathrm{Pg}}$ & 5 & Pg, all time points\\
$\lambda_{\mathrm{late}}$ & 3 & Days 15 and 21, all species\\
$\lambda_{\mathrm{rare}}$ & 0.1 & species with mean fraction $<5\%$\\
$\lambda_{\psi}$ & 3.0 (DH), 2.0 (DS), 1.5 (CS, CH) & viability channel, free-$\psi$ stages\\
$\lambda_{\mathrm{pH}}$ & 0.3 & pH channel, HOBIC conditions, free-$\psi$ stages\\\bottomrule
\end{tabular}
\caption{Likelihood weights.}
\label{tab:weights}
\end{table}
```

> 要確認: pH チャネル（λ=0.3、HOBIC のみ）を本文に書くなら §6.7 の「pH は較正に使っていない」と矛盾する。
> **p2・ult で pH チャネルを入れるなら、pH の検証（§6.7）は削るか「較正に使った」と書き換える。** → 下の §9

## 5. §3.3 Prior（A6、B1、DS の箱）— 全面差し替え

```latex
\subsection{Prior}
\label{sec:prior}

The prior is a product of independent uniform distributions on condition-specific boxes $[l_k,u_k]$ (Supplementary Table~S\TBD{n}).
All 15 entries of $\mathbf{A}$ are estimated in every condition; no entry is fixed a priori.
The decay vector $\mathbf{b}$ does not enter the model without antibiotic treatment ($\alpha^*=0$) and is not estimated.
For Dysbiotic Static, the boxes of $a_{33}$, $a_{34}$ and $a_{45}$ were widened to $[-15,20]$ because posterior mass accumulated at the bounds of the initial box.

Inference proceeds in four stages that share the same data:
(i)~fixed $\psi$ on the initial box;
(ii)~fixed $\psi$, box narrowed to the stage-(i) MAP $\pm4$ posterior SDs;
(iii)~free $\psi$ with the viability channel, box narrowed to $\pm3$ SDs;
(iv)~as (iii) with $\pm2$ SDs and \TBD{$N_p$} particles.
Narrowing is clipped to the initial box and initialised from the previous stage's particles.
Because the data are reused across stages, we report alongside the final posterior the stage-(i) posterior, which involves no data-dependent narrowing, and an inference on a much wider box ($[-15,20]$ for all entries) with and without an additional $\mathcal{N}(0,6^2)$ prior (Section~\ref{sec:identifiability}).
We monitor the fraction of posterior mass within 5\% of each bound at every stage.
```

## 6. §4 TMCMC（A5、A1、B3）

- §4.2 の `All parameters are clipped to their prior bounds after each proposal.` →
  `Proposals outside the prior box are rejected without evaluating the likelihood.`
- 変異（mutation）の記述に足す:

```latex
Random-walk proposals alternate with differential-evolution proposals~\cite{terBraak2006DEMC}, $\btheta^*=\btheta_j+\gamma_{\mathrm{DE}}(\btheta_a-\btheta_b)+\boldsymbol{\epsilon}$ with $\gamma_{\mathrm{DE}}=2.38/\sqrt{2d}$, which adapt to strongly correlated posteriors.
Each stage applies $K$ mutation steps to every particle (Table~\ref{tab:runs}).
```
  （`terBraak2006DEMC` を bib に追加: ter Braak, C. J. F. (2006) Stat. Comput. 16:239–249, doi:10.1007/s11222-006-8769-1）

- evidence の段落: `enabling Bayesian model comparison … cross-condition model comparison.` を削り、次に差し替え（A1）:

```latex
The estimate satisfies $\ln\hat{Z}\le\max_j\ln p(\yobs\mid\btheta_j)$, which we check for every run.
Because the likelihoods of different conditions involve different data, $\ln\hat{Z}$ is not compared across conditions.
```

- **新設 §4.4 Convergence assessment（B3）**:

```latex
\subsection{Convergence assessment}
\label{sec:convergence}

Every stage is run with three independent seeds and accepted only if all of the following hold:
(i)~the stored log-likelihood of every particle agrees with a recomputation from the stored parameters;
(ii)~the tempering reaches $\beta=1$;
(iii)~each particle is moved on average at least twice per stage and at least five times per free parameter over the run;
(iv)~the maximum log-likelihood differs by at most 1 across seeds; and
(v)~the posterior median of every parameter differs across seeds by at most half a posterior standard deviation.
Criterion~(iii) uses accepted moves rather than the acceptance rate: with optimally scaled random-walk proposals, $5d$ accepted moves displace each coordinate by about $2.38\sqrt5\approx5$ posterior standard deviations, independently of~$d$.
Effective sample sizes after resampling are not reported, as they reflect the resampling rather than mixing.
```

## 7. §5 GPU / Algorithm

- `Fix-ψ and iteratively narrowed priors` 段落の 2 文目（`[q_{0.01}-1.5σ, …]`）を削り、§3.3 を参照
- Algorithm 1: 行 `clip to prior bounds` → `reject proposals outside the box`、
  Phase 1a/1b/2 を stage (i)–(iv) に合わせる（`MAP ± 4σ / 3σ / 2σ`）、`Draw from wide prior` → `Draw from the stage prior`
- Table 2（計算時間）: \TBD{ult の実測で更新}。GPU は RTX 3090（stuttgart）／2080 Ti（celtic）に変わる → 表の機種を実際に合わせる

## 8. §6 Results の冒頭（run の設定）

```latex
Each stage is run with three seeds; Table~\ref{tab:runs} lists particles, mutation steps and number of tempering stages per condition.
\TBD{表: 条件 × 段の粒子数・K・ステージ数・判定}
```

## 9. §6.7 Independent validation → Post hoc consistency checks（C6、C7、C8）

- 見出し: `Independent validation` → `Post hoc consistency checks`
- 冒頭文: `… four measurements … that were not used in calibration.` →
  `We compare the inferred trajectories with measurements of Heine et al.\ that are related to, but not identical with, the calibration data. These comparisons are consistency checks rather than independent validation: the pH regression is fitted to the same species-fraction data, and the gingipain comparison is a temporal correlation with the predicted Pg abundance.`
- **pH**: p2・ult で pH チャネル（λ=0.3）を尤度に入れるので、「pH は較正に使っていない」は成り立たない → **この小節の pH 部分は削除**（または「pH チャネルとの整合」に書き換える）。要判断
- `Metabolic sign consistency`（C8）: `So→Vei (… ) indicates niche competition dominates` →
  `the So–Vei coefficient is negative (\TBD{値}); this is a property of the effective model and should not be read as evidence against the well-established lactate cross-feeding between streptococci and \textit{Veillonella}.`

## 10. §6.8 Effective dimensionality → Identifiability（B2、§6.8 の基準）— 段落を全面差し替え

```latex
\paragraph{Identifiability.}
The ratio $r_i=\sigma_{\mathrm{post},i}/\Delta_{\mathrm{prior}}$ cannot detect non-identifiability: a posterior identical to a uniform prior already gives $r=1/\sqrt{12}\approx0.29$.
We therefore assess identifiability by (i)~agreement across independent seeds (Section~\ref{sec:convergence}), (ii)~the dependence of each marginal on the prior, comparing inference on the wide box $[-15,20]$ with and without an $\mathcal{N}(0,6^2)$ prior, and (iii)~the fraction of posterior mass at the box bounds.
\TBD{結果: どの成分がデータで決まり、どれが事前分布しだいか（ident DH・DS の表）}
Parameters whose marginals change with the prior are reported as weakly identified, and no biological interpretation is based on them.
```

- 「Even Pg-related parameters in commensal conditions satisfy $r_i\le0.10$ …」「not over-parametrised」は削除
- Conclusions の 5 項目め（Model parsimony）も同じく差し替え

## 11. §7 Discussion

**Biological interpretation**（A4、C1）:
- `the transition to dysbiosis activates cooperative Vei→Pg and Fn→Pg pathways` →
  `the effective Vei--Pg and Fn--Pg coefficients become positive under dysbiotic conditions`
- `the posterior … recover the biologically expected sparsity … as an emergent property` →
  `Pg-related coefficients in both commensal conditions are not distinguishable from zero`
- Table 5 の within/cross 相関（C9）と転移表（C10）は ult で作り直し、`\TBD{}`

**A falsifiable prediction** — 段落を全面差し替え:

```latex
\paragraph{A testable prediction.}
Beyond reproducing the observed community states, the inferred model yields a specific prediction.
In posterior predictive simulations of the dysbiotic dynamic condition, the late Pg increase (Day-21/Day-15 ratio $\ge1.5$) occurs in \TBD{99--100}\% of posterior samples.
Removing \textit{F.~nucleatum} from the consortium (its volume fraction held at zero) reduces this to \TBD{10--13}\%, and setting only the Fn--Pg coefficient $a_{45}$ to zero reduces it to \TBD{2--6}\%.
The late increase thus arises in the model from a delayed rise of \textit{F.~nucleatum} transmitted to Pg through $a_{45}$, consistent with the bridging role of \textit{F.~nucleatum}~\cite{Kolenbrander2010OralMultispecies}.
The model therefore predicts, with posterior probability \TBD{0.87--0.90}, that a leave-\textit{F.~nucleatum}-out co-culture under otherwise identical dynamic cultivation suppresses the late Pg increase.
Because commensal and dysbiotic consortia differ in \textit{Veillonella} species and Pg strain as well as in community state, the prediction concerns this consortium and should not be generalised to other strain combinations.
```

**Model evidence** 段落（A1）: `higher for commensal conditions … Occam factor` は条件間比較なので**削除**。

**Computational considerations**（C11）: `Gradient-based alternatives (NUTS, HMC) fail on GPU due to …` →
`For this solver and workload, gradient-based samplers (HMC, NUTS) were less efficient on the GPU, because trajectories of variable length limit batched evaluation.`

**交絡（C15）** — Biological interpretation の末尾に 1 文:

```latex
Differences between commensal and dysbiotic matrices combine the effect of community state with those of the \textit{Veillonella} species (\textit{V.~dispar} vs.\ \textit{V.~parvula}) and the Pg strain (DSM\,20709 vs.\ W83), and cannot be attributed to the community state alone.
```

**Modelling note**（新規、§2 の末尾か Methods に）:

```latex
Removing a species is represented by switching it off (its volume fraction is held at zero), not by a zero initial value: in this formulation a species with a vanishing initial fraction can regrow from numerical floor values.
```

## 12. Data and Code Availability（D5）

```latex
Code, run configurations, convergence records and posterior samples are archived at Zenodo (\TBD{DOI}) and developed at \url{https://github.com/keisuke58/Tmcmc202601}.
The experimental data are those of Heine et al.~\cite{Heine2025PeriImplant}.
```

---

## ult を待つもの（`\TBD` の一覧）

| 箇所 | 中身 | 出所 |
|---|---|---|
| Abstract | 速度比、RMSE 範囲、a45 区間、ノックアウト確率 | ult、`knockout_fn` |
| §3.3 | 最終段の粒子数 | 返答その20 の決め方 |
| §5 Table 2 | 計算時間（機種も） | ult の実測 |
| §6 | 段ごとの run 表 | `run_record.json` |
| §6.1 Table 3 | RMSE・MAE・max lnL・ln Z（自動生成） | `config.json` |
| §6.7 | So–Vei の値 | ult MAP |
| §6.8 | 識別性の表 | ident DH・DS |
| §7 | Table 5・転移表、ノックアウトの数字 | ult |
| D5 | Zenodo DOI | 投稿前に発行 |
