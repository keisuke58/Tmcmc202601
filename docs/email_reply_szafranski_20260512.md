# Reply Draft — Szafranski Progress Meeting Follow-up
**To**: Szafranski.Szymon@mh-hannover.de  
**Date**: 2026-05-12

---

Dear Szymon,

Thank you for the detailed summary. It is very helpful to have everything written down so clearly. Below I address each point from your action item list.

---

## 1. What the model aims to achieve

The model provides a quantitative, mechanistically grounded description of how five key oral biofilm species — *S. oralis* (So), *A. naeslundii* (An), *Veillonella* spp. (Vei), *F. nucleatum* (Fn), and *P. gingivalis* (Pg) — interact over time under different conditions (commensal vs. dysbiotic, static vs. HOBIC flow). The ultimate goal is to link community composition to the mechanical stiffness of the biofilm using a multi-scale pipeline (community dynamics → diversity index → material model → finite-element simulation).

## 2. What has already been achieved

- **Bayesian parameter estimation (TMCMC)** for all four experimental conditions from Heine et al. 2025. Posterior distributions over 20 interaction parameters are available, with MAP-RMSE 0.054–0.075 across conditions.
- **Species-resolved fit**: the model reproduces the key biological observation that *P. gingivalis* surges only in the Dysbiotic HOBIC condition and is suppressed in all others.
- **Multi-scale mechanical pipeline**: estimated community composition feeds into a diversity index (DI), which drives an elastic material model (E ≈ 31–908 Pa), which is finally used in a 3D finite-element simulation of a T23 tooth–biofilm assembly. The resulting maximum displacement varies 29×  between commensal and dysbiotic conditions.
- **Model selection**: five alternative material models (DI, Composite, EPS-synergy, φ_Pg, Virulence) were compared via pseudo-Bayes factors. The DI model is decisively preferred.

## 3. Which results were expected

- The surge of *P. gingivalis* in Dysbiotic HOBIC conditions was the target outcome and was successfully reproduced by strong estimated Vei→Pg and Fn→Pg interaction coefficients (a₃₅ and a₄₅).
- Commensal conditions show a more balanced community composition, consistent with the experimental data.
- Dysbiotic communities produce lower mechanical stiffness (higher DI = more disorder), consistent with the biological hypothesis.

## 4. Which results were unexpected

- **Condition-specific interaction matrices**: the four estimated A matrices differ substantially (Frobenius relative difference 0.42–0.71 between pairs; correlation even negative across health vs. dysbiotic groups). This suggests that the effective interaction strengths are not universal but condition-dependent, which was not assumed a priori.
- **Multi-attractor behaviour**: the DH baseline posterior is bimodal in DI (values cluster near 0.16 and 0.87). A basin sensitivity analysis showed that 49 of 51 commensal_static perturbed samples jump to the DI ≈ 0.85 attractor — indicating the system is close to a tipping point, which was not anticipated.
- **Vei–Pg cooperation is not uniquely high in Dysbiotic HOBIC**: the estimated A[Vei, Pg] is actually lower in DH than in CS or DS. The surge is driven by the combination of many blocks, not a single elevated entry.

## 5. How the model relates to experimental and clinical data

The model is calibrated to the six-timepoint compositional fractions from Heine et al. 2025 (four conditions, five species). As an independent check, we validated against unused Heine 2025 data: the model correctly predicts pH trends (R² = 0.71, N = 12), Gingipain–Pg correlation (r = 0.90), and growth rate parity (~0.41–0.42/h).

## 6. How database-derived interactions are transformed into model parameters

This is the point that I need to document more clearly. Here is the current logic:

The **interaction network topology** (which entries of A are non-zero) comes from the literature: specifically, documented metabolic cross-feeding and co-aggregation relationships from the Dieckow npjB&M paper and KEGG pathway data. Each species-pair was assigned a binary status (interaction present / absent) based on whether a direct metabolic or physical coupling has been experimentally reported.

**Directionality**: In the current Hamilton-ODE formulation, the interaction matrix A is **symmetric** by construction (A_ij = A_ji), which follows from the thermodynamic consistency condition of the extended Hamilton principle. This means the model does not distinguish A→B from B→A — the coefficient represents the strength of mutual coupling, not direction. For example, the So–Vei lactate exchange is captured by a single symmetric entry, even though biologically it is So producing lactate and Vei consuming it.

**Sign convention**: For the non-zero pairs, A_ij > 0 means mutual benefit (growth-promoting), A_ij < 0 means mutual inhibition. When conflicting evidence exists (both positive and negative), the sign is currently resolved by expert judgment based on which mechanism dominates under the modeled conditions. This needs to be documented explicitly and is an acknowledged limitation.

**Zero-locked pairs** (set to 0 regardless of the data): An–Vei, An–Fn, Vei–Fn, So–Pg, An–Pg. These are pairs for which no direct metabolic pathway has been reported.

The **magnitude** of non-zero entries is not taken from databases — it is inferred purely from the TMCMC fit to the Heine 2025 time-series data.

I agree that this step should be described more clearly in the paper. I will prepare an explicit table showing: source → binary interaction decision → sign assignment → zero-locked or free, for all 15 pairs, as an intermediate data sheet.

## 7. The sucrose–lactate scenario

Your description maps directly onto a known limitation of the current model. The Hamilton ODE uses a **time-invariant** A matrix, which cannot capture the regime shift where lactate accumulation and pH drop change the nature of the So–Vei interaction from beneficial to inhibitory. I have a design draft for a pH/metabolite-dependent A(t) extension (sigmoid-gated regime shift, or A(t) = A_comm + (A_dys − A_comm) × σ(pH − pH_crit)). Whether the Heine 2025 data provides sufficient information to calibrate this extension remains to be checked. This would also address the "condition-dependent interaction" point from your summary.

## 8. Compositional data tables

I will prepare:
- **Simulated profiles**: for each of the four conditions, one row = one TMCMC posterior sample, columns = φ_So, φ_An, φ_Vei, φ_Fn, φ_Pg at each timepoint, plus condition and cultivation metadata.
- **Experimental profiles**: from Heine 2025, same structure.
- **Sample factors**: condition (commensal/dysbiotic), cultivation (static/HOBIC), timepoint (days 1–21), replicate ID.
- Format: CSV with a metadata sheet, ready for R/vegan or Python/scikit-bio multivariate analysis (Bray-Curtis + PCoA or PERMANOVA).

I will send these as a follow-up attachment once prepared.

---

## Summary of outstanding items on my side

| # | Status | Item |
|---|--------|------|
| 1 | In progress | Model description section added to PDF |
| 2 | Will prepare | Interaction decision table (DB → topology → sign → zero-lock) |
| 3 | Documented above | Directionality: A is symmetric; no directed A→B distinction in current model |
| 4 | Will prepare | Intermediate data sheets (input params → ODE → simulated φ → DI → E → u) |
| 5 | Will prepare | Compositional data tables (experimental + simulated, long format) |
| 6 | Will prepare | Sample-factor + feature table, multivariate-analysis ready |
| 7 | Waiting | Your WhatsApp sketches on lactate–sucrose scenarios |
| 8 | Waiting | SimCom results from Radek/Room Drum |

Looking forward to the sketches and the SimCom data — especially the lactate–sucrose scenarios will be useful context for the A(t) extension.

Best regards,  
Keisuke
