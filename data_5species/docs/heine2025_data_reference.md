# Heine et al. 2025 — Data Reference for TMCMC Modeling

> **Citation**: Heine N, Bittroff K, Szafrański SP, et al.
> "Influence of species composition and cultivation condition on peri-implant biofilm dysbiosis in vitro."
> *Front. Oral Health* 6:1649419 (2025). DOI: [10.3389/froh.2025.1649419](https://doi.org/10.3389/froh.2025.1649419)
>
> **Supplementary**: Data Sheet 1 (10 pages) — `data_5species/docs/Data Sheet 1.pdf`

---

## 1. Experimental Setup

| Item | Value |
|------|-------|
| Species (Commensal) | S. oralis, A. naeslundii, **V. dispar**, F. nucleatum, P. gingivalis DSM 20709 |
| Species (Dysbiotic) | S. oralis, A. naeslundii, **V. parvula**, F. nucleatum, P. gingivalis **W83** |
| Cultivation | Static (6-well plates, polystyrene) / HOBIC (flow chamber, Ti grade 4) |
| Flow rate (HOBIC) | 100 µl/min |
| Temperature | 37°C, anaerobic |
| Medium | BHI + VitK/Hem (full first 24h → 1:2 diluted from day 1) |
| Timepoints | Day 1, 3, 6, 10, 15, 21 |
| Inoculum | OD600 = 0.05 per strain |
| Ti disc | 12 mm diameter, 1.5 mm height, Ra = 0.31 µm |
| Replicates | N=5 (OD, pH), N≥15 (CLSM), N=9 (qRT-PCR) |

---

## 2. Data Already Extracted (CSV files in `experiment_data/`)

### 2a. Calibration Target Data (TMCMC input)

| File | Content | Source |
|------|---------|--------|
| `biofilm_boxplot_data.csv` | Total biofilm volume (µm³/µm²): median, Q1, Q3, whiskers | Fig 2A |
| `species_distribution_data.csv` | Species % (median, IQR): 4 conditions × 6 days × 5 species | Fig 3A |
| `fig3_species_distribution_summary.csv` | Per-replicate stats: mean, std, min, max, Q1, Q3 | Fig 3A |
| `fig3_species_distribution_replicates.csv` | Raw replicate-level species % (N=9 per condition) | Fig 3A |

**Species color mapping:**

| Color | Commensal model | Dysbiotic model | Model index |
|-------|----------------|-----------------|-------------|
| Blue | S. oralis | S. oralis | 0 |
| Green | A. naeslundii | A. naeslundii | 1 |
| Yellow | V. dispar | — | 2 |
| Orange | — | V. parvula | 2 |
| Purple | F. nucleatum | F. nucleatum | 3 |
| Red | P. gingivalis 20709 | P. gingivalis W83 | 4 |

### 2b. Metabolism Data

| File | Content | Source |
|------|---------|--------|
| `fig4A_pH_timeseries.csv` | pH (0.5-day resolution, 0–21 days), Commensal vs Dysbiotic HOBIC | Fig 4A |
| `fig4B_gingipain_concentration.csv` | Gingipain protein (ng/ml?), day 0–21, Dysbiotic only | Fig 4B |
| `fig4C_metabolic_interactions.csv` | Metabolite production/consumption edges | Fig 4C |
| `fig4C_culture_medium.csv` | Culture medium composition | Fig 4C |
| `fig1B_OD600_growth_curves.csv` | OD600 growth curves (0–24h), 5 replicates per model | Fig 1B |

---

## 3. Data from Supplementary (Data Sheet 1) — Not Yet in CSV

### 3a. Table S4: Genome Size / Weight (qRT-PCR → Cell Number Conversion)

| Species | Genome size (bp) | Genome weight (ng) | Notes |
|---------|-----------------|---------------------|-------|
| S. oralis | 1.96 × 10⁶ | 2.15 × 10⁻⁵ | |
| A. naeslundii | 3.04 × 10⁶ | 3.33 × 10⁻⁵ | |
| V. dispar | 2.12 × 10⁶ | 2.32 × 10⁻⁶ | Commensal model |
| V. parvula | 2.16 × 10⁶ | 2.37 × 10⁻⁶ | Dysbiotic model |
| F. nucleatum | 2.17 × 10⁶ | 2.38 × 10⁻⁶ | |
| P. gingivalis | 2.34 × 10⁶ | 2.57 × 10⁻⁶ | Both strains |

**Usage**: qRT-PCR gives DNA mass (ng) → divide by genome weight → cell number.
Cell number ratio ≠ volume fraction ratio (different cell sizes).
Our model uses volume fractions directly from % data, so this is for **cross-validation** or
converting to CFU-based comparisons with other literature.

### 3b. Table S1–S3: qRT-PCR Primers and Conditions

| Species | Target gene | Annealing T (°C) |
|---------|------------|-------------------|
| S. oralis | gtfR | 58 |
| A. naeslundii | gyrA | 58 |
| V. dispar/parvula | 16S rRNA | 58 |
| F. nucleatum | 16S rRNA | 60 |
| P. gingivalis | 16S rRNA | 56 |

**Usage**: Methods section reference. The different target genes (housekeeping vs 16S)
may introduce quantification bias — see Discussion in paper.

### 3c. Table S5: FISH Probes

| Species | Probe | Label |
|---------|-------|-------|
| S. oralis (So405) | ACA gCC TTT AAC TTC agA CTT ATC TAA | Alexa Fluor 405 |
| A. naeslundii (An488) | Cgg TTA TCC AgA AgA Agg gg | Alexa Fluor 488 |
| V. dispar/parvula (Vd568) | AAT CCC CTC CTT CAg TgA | Alexa Fluor 568 |
| P. gingivalis (Pg647) | CAA TAC TCg TAT CgC CCg TTA TTC | Alexa Fluor 647 |
| F. nucleatum (FUS664) | CTT gTA gTT CCg CYT ACC TC | AF405 + AF647 (dual) |

**Usage**: Methods section reference. Dual labeling of Fn allows separation from So (both blue).

### 3d. Tables S6–S9: Statistical Tests (p-values, in image form)

| Table | Test | What's compared | Key findings |
|-------|------|-----------------|--------------|
| S6 | Kruskal-Wallis + Dunn's | Biofilm volume over time | DS/DH volume increase significant from day 6 |
| S7 | 2-way ANOVA + Tukey's | Viability (membrane integrity) | All conditions except CS show significant decrease |
| S8 | 2-way ANOVA + Dunnett's | Viable species % vs day 1 | CH: So↓ from day 15; DH: Vp↓ from day 10, An/Fn/Pg↑ |
| S9 | 2-way ANOVA + Dunnett's | Total species % vs day 1 | Similar to S8 but weaker effects |

**Usage**: Model validation — our TMCMC predictions should reproduce the same
significant temporal changes. Relevant for posterior predictive checks (Fig 2a-d in paper).

### 3e. Figure S1: Total (Non-Viable) Species Distribution

Box plots of **total** bacterial species distribution (qRT-PCR without PMA pre-treatment).
Changes less pronounced than viable distribution (Fig 3A).
**Usage**: Comparison with viable data to assess how much dead DNA affects our model fit.

### 3f. Figure S2: pH-Dependent Growth & Veillonella-Conditioned Medium

**(A) pH-dependent growth (24h, BHI+VitK/Hem):**
- All 6 strains grown at different pH values (pH 5.5–7.5 estimated)
- P. gingivalis W83 grows significantly more than P. gingivalis 20709 at neutral/alkaline pH
- V. parvula grows more than V. dispar across all pH
- So, An: relatively pH-insensitive

**(B) Veillonella-conditioned medium:**
- Fn and Pg grown in V. dispar-conditioned, V. parvula-conditioned, and fresh medium
- **No significant difference** between conditioned media for either Fn or Pg
- → Veillonella's effect on pathogen growth is NOT via secreted metabolites

**Usage for prior bounds**: pH-growth data supports:
- b_Pg (growth rate, idx 15) should be **higher in Dysbiotic** (W83 > 20709 at all pH)
- b_Vd/Vp (growth rate, idx 3) should be **higher for V. parvula** in Dysbiotic
- Cross-feeding via Veillonella metabolites is **not the mechanism** → interaction via
  coaggregation/spatial structure is more likely (supports Hamilton model's φ·ψ coupling)

---

## 4. Key Quantitative Facts from Main Paper Text

### 4a. Species Distribution Trends (Section 3.2, viable cells)

**Commensal HOBIC:**
- So: 70% (day 1) → 35% (day 21) — significant decrease
- Vd: stable ~30% from day 6 onward
- An: 0% → ~30% (gradual increase)
- Fn, Pg: undetectable throughout

**Commensal Static:**
- Vd dominant initially, replaced by So by day 6
- So: ~18% (day 1) → 70% (day 3–6) → 35% (day 21)
- Vd: 83% (day 1) → 20% (day 3) → 30% (stable)
- An: gradual increase to ~30%
- Fn, Pg: undetectable

**Dysbiotic Static:**
- Vp: dominant 50–60% throughout
- Pg: ~25% (stable)
- Fn: ~20% (stable)
- So, An: very low (<5%)

**Dysbiotic HOBIC ("The Surge"):**
- Vp: 95% (day 1) → 20% (day 21) — dramatic decline
- An: increases from day 6, reaches 15–30%
- Fn: increases from day 10, reaches 15–30%
- Pg: increases from day 21 to 15–30%
- So: minimal (~1–4%)

### 4b. pH Development (HOBIC, Fig 4A)

| Period | Commensal | Dysbiotic |
|--------|-----------|-----------|
| Day 0 | 7.5 | 7.5 |
| Day 0.5 (lag phase) | 6.2 | 6.2 |
| Day 1 (post medium change) | 6.3 | 6.3 |
| Day 1–6 (equilibrium) | ~6.3 (stable) | 6.4–6.5 (rising) |
| Day 10–21 | ~6.4 (stable) | 6.7–6.9 (rising) |

**Interpretation**: Higher pH in Dysbiotic is due to less So (less lactate production)
and more Veillonella (lactate → acetate/propionate, less acidic).

### 4c. Gingipain Concentration (Dysbiotic HOBIC only, Fig 4B)

| Day | Mean (arb. units) | Significant vs day 0? |
|-----|-------------------|-----------------------|
| 0 | 0 | — |
| 3 | 0.3 ± 0.2 | No |
| 6 | 0.8 ± 0.3 | No |
| 10 | 1.2 ± 0.5 | No |
| 15 | 3.2 ± 1.8 | Yes (p ≤ 0.05) |
| 21 | 5.5 ± 3.2 | Yes (p ≤ 0.05) |

Commensal: 0 at all timepoints.

---

## 5. Hamilton Model Parameters — What's Fixed vs Estimated

### 5a. Original Klempt 2024 Definitions

| Symbol | Klempt Definition | Type in PDE | Nishioka 0D Usage |
|--------|-------------------|-------------|-------------------|
| **c** | Nutrient concentration c = ĉ/ĉ_max ∈ (0,1] | Spatial state variable | Fixed constant = 25.0 |
| **α** | Local expansion parameter (Fg = αI) | Spatial state variable | Fixed constant = 0.0 |

In Klempt's continuum PDE model, `c(x,t)` is a spatially-varying **nutrient field** that
evolves with a reaction-diffusion PDE. `α(x,t)` is the local isotropic growth tensor.

In our 0D Hamilton ODE reduction:
- `c_const = 25.0` → scaling factor for interaction term (A @ (φ·ψ))
- `alpha_const = 0.0` → decay modulation in ψ equation (disabled)

### 5b. Why Not Estimate c and α

| Parameter | Problem if estimated |
|-----------|---------------------|
| c | **Confounded with A matrix**: c·A[i,j] = (2c)·(A[i,j]/2), non-identifiable |
| α | **Confounded with b_i** (diagonal of A): both scale decay of species i |

**Conclusion**: c and α are absorbed into the 20 estimated parameters (a_ij, b_i).
Keeping them fixed is correct for the 0D formulation. Nutrient effects are
implicitly captured by the condition-specific A matrix estimates.

---

## 6. Data Files Inventory

```
experiment_data/
├── biofilm_boxplot_data.csv          ← TMCMC input: total volume
├── species_distribution_data.csv     ← TMCMC input: species %
├── fig3_species_distribution_summary.csv  ← per-replicate stats
├── fig3_species_distribution_replicates.csv ← raw replicates (N=9)
├── fig1B_OD600_growth_curves.csv     ← planktonic growth 0-24h
├── fig4A_pH_timeseries.csv           ← pH time series (HOBIC)
├── fig4B_gingipain_concentration.csv ← gingipain (Dysbiotic)
├── fig4C_metabolic_interactions.csv  ← metabolite network
├── fig4C_culture_medium.csv          ← medium composition
├── expected_species_volumes.csv      ← computed absolute volumes
├── boxplot_*.csv                     ← per-condition volume boxplots
├── barplot_*.csv                     ← per-condition volume barplots
├── *_all.csv                         ← aggregated per-condition data
└── summary_for_claude.md             ← overview (this doc extends it)
```

---

## 7. Potential Additional Uses of Supplementary Data

| Data | Potential Use | Priority |
|------|---------------|----------|
| Table S4 (genome weights) | Cross-validate volume-based model with CFU-based literature | Low |
| Figure S2A (pH-growth) | Informative prior on b_i (growth rates) per condition | **Medium** |
| Figure S2B (conditioned medium) | Supports: Vd/Vp effect is NOT via metabolites → coaggregation | Narrative |
| Tables S6-S9 (p-values) | Posterior predictive check: model should match significance patterns | **Medium** |
| Figure S1 (total vs viable) | Dead DNA correction factor for model | Low |
| Table S1-S3 (PCR primers) | Methods citation only | Low |
| Table S5 (FISH probes) | Methods citation only | Low |
