# EML 本文の削りの案（2026-10-17）

上限は本文 4000 語未満。今の語数は、私の数え方で本文＋図の説明 4537 語（10/16 は 4267）、メモの数え方の目安で約 4100 語。
DS の文が入るとさらに増えるので、300〜400 語削っておきたい。

各案: 場所・理由・減る語数、そのあとに今の文と案の文。どれも中身は補足にあるか、補足に移す。

**合計: 約 −342 語**（14 か所）。I4・I5 の中身は補足 S5（run と判定）に移す。

## I1（2 推定・尤度）: −28 語
読み取り値とのずれの詳細は補足 S10 にある

**今**

> ; values read off the published box plots deviate from these by up to 0.42 in a species fraction in the dysbiotic conditions, while the live fraction and the biofilm volume are read off to within 0.015 and a few percent (Supplementary Section~\ref{S-sec:S_raw}).

**案**

> (values read off the published box plots deviate by up to 0.42; Supplementary Section~\ref{S-sec:S_raw}).

## I2（2 推定・尤度）: −44 語
Day 1 起点の 3 文を 1 文に

**今**

> Model time is the time since inoculation, $t=s\,d$ with $s=0.2375/21$; the day-1 composition, the earliest measurement, serves as the initial state at $t=0$.
In the flow reactor the inoculated medium is replaced by sterile 1:2-diluted medium after 24\,h~\citep{heine2025influence}, so the growth phase there can also be taken to start at day~1.
Starting the model at day~1 ($t=s\,(d-1)$) changes the composition errors of the maximum a posteriori estimate (MAP) by at most 0.009 in CS and CH and 0.014 in DH (where the viable-composition error decreases) without re-inference

**案**

> Model time is $t=s\,d$ ($s=0.2375/21$) with the day-1 composition as the initial state; starting at day~1 instead ($t=s\,(d-1)$), as the medium change after 24\,h in the flow reactor~\citep{heine2025influence} would suggest, changes the composition errors of the maximum a posteriori estimate (MAP) by at most 0.014

## I3（2 推定・尤度）: −10 語
生きた割合の画像枚数などは補足 S10 にある

**今**

> (image based, not resolved by species; mean over 15--45 confocal images, five per biofilm, per time point, provided by the authors)

**案**

> (image based, not resolved by species; Supplementary Section~\ref{S-sec:S_raw})

## I4（2 推定・段）: −26 語
warm start の確認を短く（数字は補足 S5 へ）

**今**

> To check that the warm start does not bias stage (iii), one CH stage-(iii) run was started with all particles drawn from the prior, on the same box as the three warm-started seeds: its maximum log-likelihood ($-26.74$) lies within their range ($-26.71$ to $-27.25$), and the medians and 90\% bounds of $a_{33}$ and $a_{45}$ lie within 0.03 of the range spanned by the three warm-started seeds.

**案**

> A CH stage-(iii) run started entirely from the prior reaches the same maximum log-likelihood and the same posterior of $a_{33}$ and $a_{45}$ as the warm-started seeds (Supplementary Section~\ref{S-sec:S_runs}), so the warm start does not bias stage (iii).

## I5（2 推定・判定）: −33 語
判定 (c) の理論的な根拠は補足 S3 へ移す

**今**

> Criterion~(c) counts accepted moves rather than the acceptance rate: with optimally scaled random-walk proposals~\citep{RobertsRosenthal2001Optimal}, $5d$ accepted moves displace each coordinate by about five posterior standard deviations, independently of the dimension~$d$.

**案**

> （削除）

## I6（2 推定・判定）: −26 語
段 (i) の位置づけを短く

**今**

> Stage (i) is exploratory; it sets the stage-(ii) box and gives the evidence estimates of TMCMC, which are used only to compare models on the same data and box and are reported with the caveat that not all stage-(i) groups meet the criteria (Section~\ref{sec:results}).

**案**

> Stage (i) is exploratory: it sets the stage-(ii) box and gives the evidence estimates used for model comparison (Section~\ref{sec:results}).

## R1（4.1 当てはまり）: −0 語
「何度もやり直した」を短く（カンマ抜けも直す）

**今**

> live fraction 0.11--0.12) 0.090--0.093 in CS

**案**

> live fraction 0.11--0.12), 0.090--0.093 in CS

## R2（4.1 当てはまり）: −25 語
同上

**今**

> Accepting a stage often required more than one attempt: posterior mass accumulated at a bound of the initial box in every condition and was released by widening that bound (Supplementary Table~\ref{S-tab:S_boxes}), and the dysbiotic conditions needed 18000 (DS) and 20000 (DH) particles in stage (iii) (Supplementary Table~\ref{S-tab:S_runs}).
Without the acceptance criteria, several of these runs would have been read as converged.

**案**

> Accepting a stage often required more than one attempt (widened bounds in every condition, up to 20000 particles in DH; Supplementary Tables~\ref{S-tab:S_boxes} and \ref{S-tab:S_runs}); without the criteria, several of these runs would have been read as converged.

## R3（4.4 確認・pH）: −20 語
回帰の説明を短く

**今**

> A linear regression of the measured pH on the measured species fractions (intercept and four fractions, Vei as reference; $R^2=0.88$ over the continuous pH records of CH and DH, with the fractions interpolated between sampling days) was fitted once on the measured data and then applied, unchanged, to the posterior predictive composition, so the check tests the predicted composition, not a model of pH.

**案**

> A linear regression of the measured pH on the measured species fractions ($R^2=0.88$ over the continuous records of CH and DH) was fitted once and applied, unchanged, to the posterior predictive composition, so the check tests the predicted composition, not a model of pH.

## R4（4.4 確認・ノックアウト）: −28 語
a45=0 の説明を 1 文に

**今**

> Setting only $a_{45}=0$ removes it in most samples (median 0.8--0.9; 18--25\% of the samples keep a ratio of 1.5 or more), consistently across seeds; the other interactions of Fn partly offset this coefficient, so that removing Fn altogether has a smaller effect. The late increase of Pg in the model therefore rests mainly on the Fn--Pg coefficient, and the predicted reduction can be tested directly by a co-culture without Fn.

**案**

> Setting only $a_{45}=0$ removes it in most samples (median 0.8--0.9 in all seeds), so the increase rests mainly on the Fn--Pg coefficient, partly offset by the other interactions of Fn; the predicted reduction can be tested directly by a co-culture without Fn.

## L1（4.5 限界・CS）: −30 語
初期状態・σ の試しは補足 S13 と重複

**今**

> Neither an initial state taken from the day-1 data (total-count composition and species viabilities instead of the viable composition and $\psi_i=0.999$) nor a day-specific $\sigma_\ell$ (which only moves CS along the same curve, to errors of 0.18 for the live fraction and 0.12 for the viable composition) changes this.

**案**

> Neither an initial state from the day-1 data nor a day-specific $\sigma_\ell$ changes this (Supplementary Section~\ref{S-sec:S_tradeoff}).

## L2（4.5 限界・CS）: −22 語
重みの文と死滅項の文をまとめる

**今**

> We use the same weights as in the other conditions and report this tension rather than tune $\lambda_\psi$ for CS; a viability equation with a death term is left to future work.

**案**

> We report this tension rather than tune $\lambda_\psi$ for CS.

## L3（4.5 限界・ψ）: −17 語
S12 の要約を短く

**今**

> With the viable and the total-count compositions as separate channels, the species viabilities $\psi_i$ are constrained species by species; in CH the model follows the viability implied by the data for the dominant species So, while in CS it keeps the main species almost fully viable, the tension described above (Supplementary Section~\ref{S-sec:S_psi}).

**案**

> The two composition channels constrain the viabilities $\psi_i$ species by species; in CH the model follows the data for the dominant species So, in CS it keeps the main species almost fully viable (Supplementary Section~\ref{S-sec:S_psi}).

## L4（4.5 限界・連続体）: −33 語
連続体モデルへのつなぎ 3 文を 2 文に

**今**

> The calibrated dynamics are, however, the input that continuum growth models of biofilms require~\citep{Klempt2024ContinuumGrowth}: the total volume fraction of the point model can drive the growth variable of a finite-element model, so that differences in growth-induced stress between conditions follow from the calibrated interactions rather than from assumed species-specific growth weights.
Because the likelihood uses species fractions only, the absolute amount of biomass is a prediction of the model rather than a calibrated quantity, and its posterior spread has to be propagated into such a coupling.

**案**

> The calibrated dynamics are, however, the input that continuum growth models of biofilms require~\citep{Klempt2024ContinuumGrowth}, so that differences in growth-induced stress follow from calibrated interactions rather than assumed growth weights; since the likelihood uses fractions only, the absolute biomass is a model prediction whose posterior spread has to be propagated into such a coupling.
