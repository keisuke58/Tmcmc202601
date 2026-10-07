# Response to co-author reviews — DRAFT v0

*GPU-accelerated Bayesian inference of multi-species biofilm interaction parameters via TMCMC*
(Nishioka et al., circulated 2026-09-17)

作成 2026-10-07（クラウド側）。下書き。**送らない。**

## この下書きの使い方（内部メモ）

- 宛先: Szymon P. Szafrański・Rumjhum Mukherjee（2026-09-25 のレビュー）。統計の項目（A1, B1–B4）は
  Szymon 氏の依頼どおり Meisam Soleimani 氏と先に相談してから確定する
- 指摘の番号は `1030_Masterarbeit/notes/paper_review_20260925_coauthors.md`（LUH repo、branch
  `claude/paper-review-triage`）の仕分け（A0–A8, B1–B5, C1–C15, D1–D5, N1–N9）に合わせた
- 数値の出所は `docs/paper_gateoff_pipeline.md` §8 と `docs/handoff/gpu_2026-10-0*.md`
- **`[[FINAL: ult]]` の数字は pilot / p1 段階のもの。** ult（10000 粒子・多チャネル）が出たら置き換える。
  それまで本文に転記しない
- 状態: ✅ 回答できる / 🟡 方針は決まった・数値待ち / ⬜ 未着手

---

Dear Szymon, dear Rumjhum,

Thank you very much for the careful and constructive reviews. Your comments, together with a
systematic check of the code behind every number in the manuscript, led us to re-run the entire
inference. The most important outcome is summarised first; point-by-point replies follow.

## 0. Summary of the main change

While tracing the code behind the terminal *P. gingivalis* (Pg) surge, we found that the
production model contained a multiplicative Hill-type gate on the Pg interaction row,
h(Fn) = Fn^n / (K^n + Fn^n), that was **not described anywhere in the manuscript**. Because
h → 0 when Fn is absent, the gate guaranteed our "falsifiable prediction" (omitting *F. nucleatum*
suppresses the late Pg bloom) independently of the inferred coefficient a₄₅. It also made the
effective dynamics asymmetric although A is symmetric by construction.

We therefore **removed the gate and repeated the full two-phase inference**, using the same data,
likelihood and stage sequence as the original runs, with three independent seeds per condition
and explicit convergence criteria (Section B3 below). Results so far `[[FINAL: ult]]`:

1. **The gate does not improve the fit.** With and without the gate, RMSE and the Pg
   Day 21/Day 15 ratio are the same within seed-to-seed variation (DH: RMSE 0.058–0.065 without
   vs 0.057–0.062 with the gate; Pg D21/D15 2.43–2.87 vs 2.31–2.66; observed 2.74).
2. **The model without the gate reproduces the terminal Pg surge** (DH, p1 stage:
   D21/D15 = 2.63–3.02, observed 2.74; RMSE 0.054–0.056).
3. **a₄₅ (Fn–Pg) is positive only under dysbiotic-HOBIC conditions**: 90% interval
   [+0.8, +5.2] (median +3.0) in DH, versus intervals straddling zero in both commensal
   conditions (CS [−0.9, +0.9], CH [−0.9, +1.8]).
4. **The prediction now follows from the inferred parameters.** In posterior samples of the
   gate-free DH model, the Pg surge (D21/D15 ≥ 1.5) occurs in 99–100% of samples; it occurs in
   only 10–13% when Fn is removed from the consortium, and in 2–6% when a₄₅ alone is set to zero.
   The posterior probability that omitting Fn suppresses the bloom is therefore ≈ 0.87–0.90,
   an inferred and testable statement rather than a consequence of an assumed gate.

We believe this makes the paper's central biological message stronger and, importantly, honest.

---

## A. Errors that we have confirmed and corrected

### A0 — Undocumented Hill gate ✅
See Section 0. The gate is removed from the model and from all re-run results. The original
gated results are retained only as a control in the Supplement, with the gate fully specified
(functional form, K = 0.05, n = 4) so that the comparison is transparent.

> 内部メモ: 対照として ln Z（ゲート ON / OFF）を Supplement に載せるかは Meisam 氏と相談。
> pilot 段階では max logL が seed 間の幅の中で同じ。ln Z は事前分布の箱が同じ run どうしでのみ比べる。

### A1 — ln Ẑ exceeds max ln L in Table 3 ✅（原因）／🟡（数値）
You are right that ln Ẑ ≤ max ln L must hold for the same likelihood. Two causes:
(i) the evidence accumulator in the TMCMC engine omitted the log-sum-exp offset of the
incremental weights, and (ii) in Phase 2 the reported max ln L and ln Ẑ were computed from
different likelihood definitions. Both are fixed; the inequality is now checked automatically

> 内部メモ: (i) はエンジンで修正済み・テストで確認済み。**(ii) は未確認**（Table 3 の脚注からの推測）。
> 原論文の run ログが無いので確かめられない。確かめられなければ (ii) は削って「(i) を修正し、全 run で不等式を確認した」だけにする。

for every run (unit test and per-run check). The corrected Table 3 will be regenerated directly
from the saved run outputs `[[FINAL: ult]]`.

### A2 — RMSE < MAE in Table 3 (DS, Phase 1) ✅
This was a transcription error. Table 3 will be generated programmatically from the saved run
records, so values can no longer be copied by hand.

### A3 — Explanation of R² reversed ✅
Corrected: for fixed SS_res, a larger SS_tot *increases* R². We now state the actual reason
for low or negative R² of *S. oralis* (small temporal variance relative to the residual).

### A4 — Directionality from a symmetric matrix ✅
A is symmetric by construction, so directional labels are not supported. All edges are now
written as undirected (Fn–Pg, Vei–Pg, …) in text, Table 1 and Fig. 1. Directional biological
knowledge is discussed separately as external evidence (see C14).

### A5 — Clipping in the MH step ✅（指摘は取り下げ・本文を修正）
The production sampler rejects proposals outside the prior bounds (they are not evaluated);
it does not clip them, so detailed balance holds. The manuscript text ("clipped to prior bounds")
was wrong and is corrected to "proposals outside the prior bounds are rejected".

### A6 — a₄₅ outside the stated prior range ✅
The prior bounds stated in §6.8 did not match those used in the runs. In the re-run, the
actual bounds per condition and parameter are reported in a Supplementary table generated from
the run records, and every posterior lies within its stated bounds (DH a₄₅: [+0.8, +5.2]).

### A7 — Flat likelihood in a₄₅ versus a narrow interval ✅
In the original (gated) MAP, a₄₅ was indeed almost inert (Δχ ≤ 0.004), i.e. the reported
interval reflected the prior narrowing rather than the data. In the gate-free re-estimation
the posterior moves to a region where a₄₅ carries the surge: setting a₄₅ = 0 removes the surge
in 94–98% of posterior samples (Section 0, item 4).

### A8 — The five b parameters do not enter the likelihood ✅
Confirmed (α* = 0, trajectories are bit-identical for any b). The manuscript already states
that B is excluded; the implementation now matches this (15-dimensional θ, b fixed).

---

## B. Statistical points (to be finalised with Meisam)

### B1 — Triple use of the data through prior narrowing 🟡
We now report, for each condition, (a) a single-stage inference on the original prior box
without any data-dependent narrowing (pilot) alongside (b) the final staged result, and
(c) an inference on a much wider box ([−15, 20]) with and without an additional N(0, 6²) prior.
The sign pattern of a₄₅ and the Pg surge are the same in (a) and (b) `[[FINAL: ult]]`. We also
monitor whether narrowing pushes posterior mass to the box edges and report it.

> 内部メモ: p1 で a34 / a44 / a22 が箱の端に 11〜21%。p2 で増えたら narrowing が縛っている → 本文に書く。

### B2 — "Sparsity emerges" / identifiability 🟡
We agree that σ_post/Δ_prior is not evidence of identifiability. In fact the criterion
r_i < 0.43 in §6.8 is satisfied by *any* uniform posterior (r = 1/√12 ≈ 0.29), so it cannot
detect non-identifiability; in the original CS run, a₅₅ and a₃₅ were statistically
indistinguishable from their uniform priors yet passed it. We remove this criterion and instead
report: reproducibility across independent seeds; prior sensitivity (wide box, with/without
prior); and which parameters remain prior-dominated. The claim of "emerging sparsity" is
withdrawn; we describe commensal Pg-related terms as "not distinguishable from zero".

### B3 — Convergence diagnostics for Phase 2 🟡
Every stage is now run with three independent seeds and accepted only if all of the following
hold: saved log-likelihoods match a recomputation for every particle; β reaches 1; each
particle moves at least twice per stage and at least five times per free dimension over the
run; the spread of max ln L across seeds is ≤ 1; and posterior medians agree across seeds within
0.5 posterior SD. We agree that ESS after resampling is not evidence and no longer report it.

> 内部メモ: 収束不足の原因だった 2 つの実装上の不具合（低 β で mutation 回数が 1/3 に絞られる、受理率を名目値で割る）は
> Methods に「事前に修正した」と 1 文で書くかを Meisam 氏と相談。

### B4 — Independent Gaussian errors on compositional data ⬜
We will justify the choice and add a sensitivity analysis (Meisam).

### B5 — Model-to-data mapping 🟡
Methods will state explicitly: predicted volume fractions are renormalised to species
fractions at each observation time; the void fraction is not observed; model time is mapped
linearly to experimental days; the initial condition is the normalised Day-1 observation
(lower bound 0.001), and Day 1 is excluded from the likelihood.

### Likelihood weights (N3, not raised in the reviews but found during the check) ✅
The likelihood up-weights Pg (λ_Pg = 5) and the last two time points (λ_late = 3) and
down-weights rare species (mean fraction < 5%, λ = 0.1); in Phase 2 the viability channel
weight is condition dependent (DH 3.0, DS 2.0, commensal 1.5). None of this was in the
manuscript. All weights are now reported in a Methods table.

---

## C. Wording and interpretation

We accept the general principle (Szymon): inferred **effective interaction parameters** are
distinguished throughout from experimentally demonstrated biological interactions.

| # | Change | 状態 |
|---|---|---|
| C1, C2 | Abstract: "the posterior discovers the biological network" / "activating cross-feeding terms" → effective parameters that reproduce the observed trajectories | ✅ |
| C3 | "static vs. dynamic (the latter in a HOBIC reactor)" (Rumjhum) | ✅ |
| C4 | Fit quality stated per species and condition; negative R² for *S. oralis* reported; Pg D21/D15 added as the surge metric (RMSE alone does not discriminate the surge because Pg is a small fraction) | 🟡 |
| C5 | UMAP kept as visualisation only, not as validation | ✅ |
| C6, C7 | pH and gingipain: "post hoc consistency check", not "independent validation" | ✅ |
| C8 | So–Vei sign reported as a model property; no "niche competition" interpretation | ✅ |
| C9, C10, C12 | Table 5, transfer table, Fig. 7 rewritten from the new numbers, including unfavourable off-diagonal cases | 🟡 `[[FINAL: ult]]` |
| C11 | NUTS/HMC: limited to "less efficient for this solver and workload" | ✅ |
| C13 | Stage counts, timings, speed-up figures and a/A notation made consistent | 🟡 |
| C14 | §2.5 rewritten: reported or hypothesised interactions, evidence type per edge, undirected | 🟡（Szymon 氏と相談） |
| C15 | Confounding (Veillonella species, Pg strain, cultivation) stated wherever condition differences are discussed | ✅ |

### Modelling note to add in Methods（新規）
Removing a species is represented by switching it off in the model (its volume fraction is held
at zero), not by a zero initial value: in the extended-Hamilton formulation a species with a
vanishing initial fraction regrows from numerical floor values. The knockout predictions in
Section 0 use the former.

---

## D. Administrative

| # | Item | 状態 |
|---|---|---|
| D1 | Title: "… of effective interaction parameters in multispecies biofilms via TMCMC" | ⬜（共著者で決める） |
| D2 | Author order as agreed with Meisam | ✅ |
| D3 | MHH and NIFE affiliations added for KN | ✅ |
| D4 | Szymon's Introduction adopted (with DOIs) | 🟡 |
| D5 | Code, run records and posterior samples archived on Zenodo with a DOI | 🟡（ult 後） |
| — | Pg strain: "ATCC 20709" → ATCC 33277 / DSM 20709 (to be confirmed against Heine et al.) | ⬜ |

---

## Discussion restructuring (Szymon's proposal) 🟡
We will follow the proposed structure: (1) which effective interaction terms change most
consistently between conditions; (2) which known oral-biofilm mechanisms could explain the
pattern, presented as biologically plausible explanations; (3) the Fn-knockout prediction as
the main experimentally testable outcome, now with its posterior probability. Szymon, we would
gladly take up your offer to contribute to this section — could we agree briefly on the
biological message first?

With best regards,
Keisuke

---

## 内部メモ: 送る前に埋めるもの

1. ult（4 条件）の RMSE・Pg 比・a45 区間・ノックアウト確率で `[[FINAL: ult]]` を置き換える
2. DS の a33 が箱の下限に張り付く件（ident DS の結果）→ A6 / B1 に 1 文
3. 識別性検証（ident DH）の判定 4・5 がどうなったか → B2 の書き方
4. Meisam 氏と A1・B1〜B4 の文言を確認
5. Szymon 氏に Discussion のメッセージを先に短く送る（D4・C14 と合わせて）
