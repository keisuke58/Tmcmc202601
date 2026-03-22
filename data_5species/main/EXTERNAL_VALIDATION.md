# External Validation: Hamilton ODE on Independent Datasets

## Established Method

The 6-species Hamilton ODE + GPU TMCMC pipeline has been validated on **Siddiqui et al. 2021** (PMC8828709) — the most challenging publicly available oral biofilm succession dataset (6 species, 21 days, 6 timepoints, planktonic + adherent).

### Achieved Results
| Dataset | RMSE | Cosine | Params | K_hill |
|---------|------|--------|--------|--------|
| Planktonic | **0.086** | 0.962 | 28 | 0.115 (free) |
| Adherent | **0.097** | 0.928 | 27 | 0.15 (fixed) |

Both **RMSE < 0.10** — demonstrating model transferability from Heine 2025 (5 species, hydroxyapatite) to an independent dataset with different species count, substrate, and culture conditions.

### Why Siddiqui Was the Hardest
- **6 species** (vs 5 in Heine) → 28 parameters (21 A_ij + 6 μ_i + K_hill)
- **No condition manipulation** — pure succession, single culture
- **Non-monotonic adherent dynamics** — Day 7 So recovery
- **Pg dominance** (60% planktonic Day 21) — requires Hill gate tuning
- **n=1 biological replicate** — no error bars for Bayesian guidance
- **Digitized data** (±3% precision from stacked bar charts)

### Key Technical Innovations
1. **phibar = φ × ψ** as correct observable (not φ alone)
2. **Heine informative prior** for shared 20 params (reduces search space)
3. **K_hill as free parameter** — sensitivity analysis showed RMSE 0.16→0.08
4. **max_delta_beta = 0.1** — forces ≥10 TMCMC stages for proper tempering
5. **Day 0.25 as IC, fit Day 1+** — avoids psi transient artifact
6. **vmap GPU parallelization** — 5000 particles on RTX 3090/4090

---

## Candidate Datasets for Further Validation

### Tier 1: Directly Applicable (oral, multispecies, time series)

#### Siddiqui 2021 — DONE ✅
- PMC8828709, Dental Materials 38:384-396
- 6 species (So, An, Aa, Vp, Fn, Pg), polished cpTi + ZrO₂
- 6 timepoints (6h, 1, 3, 7, 14, 21d), qPCR
- **Status: Completed, RMSE < 0.10**

#### Ren et al. 2007 (PMC2045240)
- "Use of qPCR and Culture Methods To Characterize Ecological Flux in Bacterial Biofilms"
- 4 species, 6 timepoints (4, 8, 12, 16, 20, 24h)
- qPCR + CFU dual quantification
- **Difficulty: Easy** (4 species, 15 params, short time scale)
- **Data: Supplementary tables with species counts**
- Adaptation: Reduce to 4-species solver, adjust time scale

#### Ciric et al. 2017 (PMC5352027) — Hannover HOBIC Model
- "An oral multispecies biofilm model for high content screening"
- 4 species (So, An, Vd, Pg), 24h + 48h
- qRT-PCR, genome-weight normalized
- **Difficulty: Easy** (4 species, 2 timepoints, but our Heine data already uses this system)
- **Data: S2/S3 Tables with primer info and genome weights**
- Note: Same HOBIC system as Heine 2025 → direct comparison possible

#### Fenn et al. 2024 (PMC12051517)
- "PMA-qPCR to quantify viable cells in multispecies oral biofilm"
- 5 species (Ao, Fn, So, Sm, Vd), 64h harvest only
- **Difficulty: Not applicable** (single timepoint, no succession dynamics)

### Tier 2: Adaptable (oral, fewer species or timepoints)

#### Quorum Sensing 3-Species (PMC10626739)
- "Bacterial quorum sensing orchestrates longitudinal interactions"
- 3 species, time series
- **Difficulty: Very easy** (3 species, 9 params)
- Need to check species overlap with our model

#### RNA-based qPCR Dual-Species (PMC6754382)
- 2 species (S. mutans + C. albicans), multiple timepoints
- **Difficulty: Trivial** (2 species, 5 params)
- Cross-kingdom (bacteria + fungus) — different biology

### Tier 3: Requires Significant Adaptation

#### In Vivo Plaque Studies
- Human plaque samples with 16S rRNA sequencing
- 100+ species → need dimensionality reduction
- Cross-sectional (not longitudinal within same patient)
- **Not directly applicable to ODE calibration**

#### Mark Welch CLASI-FISH
- Spatial imaging (confocal), not time series
- **Not applicable to ODE TMCMC**
- Applicable to VEM spatial pipeline instead

---

## Recommended Next Steps

1. **Ren 2007 (4 species, 24h)** — Quick win, validates short-timescale dynamics
2. **Ciric 2017 (HOBIC)** — Direct comparison with Heine system
3. **Joint calibration** — Fit Heine + Siddiqui simultaneously with shared A_ij
4. **Publish** — External validation section for paper

## Pipeline Summary

```
Input: JSON/CSV with {species, timepoints, fractions}
  ↓
hamilton_ode_jax_6sp.py (N-species solver, phibar output)
  ↓
estimate_siddiqui_6sp_jax.py (TMCMC + GPU vmap)
  ↓
generate_siddiqui_6sp_report.py (figures + LaTeX)
  ↓
Output: PDF report + posterior trajectories
```

Total time: ~2-4h per dataset (compilation + TMCMC + report generation)
