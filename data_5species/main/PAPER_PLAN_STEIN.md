# Paper Plan: Hamilton ODE vs gLV for Gut Microbiome

**Target**: PLoS Computational Biology (same journal as Stein 2013)
**Format**: Short paper / Research Article

## Title (draft)

"Variational Hamilton ODE outperforms generalized Lotka-Volterra
for gut microbiome dynamics with 42% fewer parameters"

## Key Selling Points

1. **Same data, fewer params, better fit**: 77 vs 132 params, Spearman 65% vs 62%
2. **Symmetric A → identifiability**: variational structure as inductive bias
3. **Compositional by design**: no absolute count normalization needed
4. **Cross-domain transferability**: oral (Siddiqui) + gut (Stein) with same framework
5. **GPU-accelerated TMCMC**: 5000 particles, full Bayesian posterior

## Structure

### 1. Introduction (~1 page)
- gLV is the standard model for microbiome time series
- Limitations: N² params, no compositional constraint, no variational structure
- Hamilton ODE: N(N+1)/2 params, compositional, energy-conserving
- Objective: direct comparison on Stein 2013 benchmark

### 2. Methods (~2 pages)

#### 2.1 Hamilton ODE for N species
- State: φ_i (volume), ψ_i (fitness), phibar = φ×ψ (observable)
- Symmetric A matrix: variational justification
- N-species generalized solver (hamilton_ode_jax_nsp.py)

#### 2.2 TMCMC with GPU vmap
- Prior from gLV: symmetrize A^gLV → informative prior
- max_delta_beta tempering
- RW mutation ×20-30 steps, vmap parallelization

#### 2.3 Initial condition strategy
- Day 2 IC (skip clindamycin acute phase)
- Day 7/9 IC for oscillatory populations

### 3. Results (~2 pages)

#### 3.1 Fit quality (Table: 9 mice × RMSE/Cosine/Spearman)
- pop1: RMSE 0.065-0.071
- pop3: RMSE 0.056-0.104
- pop2: RMSE ~0.15 (oscillatory, limitations)

#### 3.2 Hamilton vs gLV comparison (Table + Figure)
- 77 vs 132 params
- Per-timepoint Spearman: 65% vs 62%
- A matrix side-by-side (Fig)

#### 3.3 Interaction matrix analysis
- Key interactions: Coprobacillus self-inhibition, C.diff colonization dynamics
- Symmetric vs non-symmetric: what's gained, what's lost

#### 3.4 Cross-domain validation
- Brief mention: oral (Siddiqui) RMSE < 0.10 with same framework
- Same solver, same TMCMC, different biology → generalizability

### 4. Discussion (~1.5 pages)
- Why symmetric A works: variational principle as regularization
- Limitations: time-invariant A, pop2 oscillations, phibar observable
- Implications for microbiome modeling: fewer params = more interpretable
- Future: time-varying A(t), antibiotic perturbation terms, spatial extension

### 5. Conclusions (~0.5 page)

## Figures (5-6)
1. **Schematic**: Hamilton ODE vs gLV (structure diagram)
2. **Best fit**: pop3_rep2 species panel (RMSE=0.056)
3. **A matrix comparison**: Hamilton vs gLV side-by-side
4. **Results summary**: all 9 mice RMSE bar chart
5. **Growth rates**: bar chart for best mouse
6. **Cross-domain**: oral + gut RMSE comparison

## Data Availability
- Stein 2013: publicly available (PLoS Comput Biol S1)
- Our code: github.com/keisuke58/Tmcmc202601
- TMCMC samples: available on request

## Timeline
- Week 1: Draft methods + results (figures done)
- Week 2: Introduction + discussion
- Week 3: Internal review
- Week 4: Submit
