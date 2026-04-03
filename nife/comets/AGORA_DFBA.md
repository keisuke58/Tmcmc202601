# AGORA-Based dFBA for Oral Biofilm Community Dynamics

## Overview

`oral_biofilm.py` implements a 5-species dynamic Flux Balance Analysis (dFBA)
for oral biofilm communities targeting the NIFE/SIIRI peri-implantitis project.

**Species** (Hamilton ODE compatible):

| Code | Name | Role |
|------|------|------|
| So | *Streptococcus oralis* Uo5 | Early colonizer, glucose fermenter |
| An | *Actinomyces naeslundii* str. Howell 279 | Early colonizer |
| Vp | *Veillonella parvula* Te3 DSM 2008 | Obligate lactate cross-feeder |
| Fn | *Fusobacterium nucleatum* ATCC 25586 | Bridge species |
| Pg | *Porphyromonas gingivalis* W83 | Late colonizer, hemin-dependent pathogen |

## Quick Start

```python
from nife.comets.oral_biofilm import OralBiofilmComets, compute_di

m = OralBiofilmComets()                      # auto-detects agora_gems/
result = m.run(condition="healthy", max_cycles=500)

print(result._is_cobra)                      # True  (COBRApy dFBA)
print(result.total_biomass.head())           # cycle, So, An, Vp, Fn, Pg

di = compute_di(result.total_biomass)
print(di.tail())                             # cycle, DI
```

## Simulation Pipeline

```
run()
 ├─ COMETS Java (cometspy)  →  zero-growth detected (GLOP solver limitation)
 ├─ run_dfba_cobra()        →  AGORA-calibrated Monod dFBA  ← primary path
 └─ run_mock()              →  synthetic logistic growth  (ultimate fallback)
```

Result namespace fields:

| Field | Type | Description |
|-------|------|-------------|
| `total_biomass` | `pd.DataFrame` | columns: cycle, So, An, Vp, Fn, Pg |
| `media` | `pd.DataFrame` | columns: cycle, metabolite, conc_mmol |
| `_is_mock` | `bool` | True if mock simulation was used |
| `_is_cobra` | `bool` | True if COBRApy dFBA was used |

## AGORA GEMs

Files in `agora_gems/` (AGORA 1.03, downloaded from GitHub):

```
Streptococcus_oralis_Uo5.xml                        (2.16 MB)
Actinomyces_naeslundii_str_Howell_279.xml            (2.35 MB)
Veillonella_parvula_Te3_DSM_2008.xml                 (2.07 MB)
Fusobacterium_nucleatum_subsp_nucleatum_ATCC_25586.xml (2.09 MB)
Porphyromonas_gingivalis_W83.xml                     (1.93 MB)
```

Source: `https://github.com/VirtualMetabolicHuman/AGORA/tree/master/CurrentVersion/AGORA_1_03/AGORA_1_03_With_Mucins`

The GEMs are loaded via COBRApy (`cobra.io.read_sbml_model`) for structural
verification of exchange reaction existence. Strain-level IDs are the actual
available strains in AGORA 1.03 (not always matching the Hamilton ODE species).

## dFBA Method: AGORA-Calibrated Monod

### Why not pure FBA at each timestep?

AGORA models require the VMH diet medium (hundreds of tracked metabolites at
specific flux bounds) to return non-zero growth rates. Without the VMH diet files,
every FBA in a closed medium configuration returns μ = 0. The AGORA GEMs are
used here for **structural validation** (exchange reaction existence, cross-feeding
stoichiometry); kinetic parameters come from oral bacteria literature.

### Monod dynamics

At each timestep `dt`:

```
q_i(s) = q_max_i × C_s / (Km_i + C_s)          # substrate uptake [mmol/gDW/h]
μ_i    = min(Σ_s q_i(s) × Y_i,  μ_max_i)        # growth rate [h⁻¹]  (sum Monod)
       = μ_max_i × Π_s C_s/(Km_i+C_s)            # (product Monod for Pg)

X_i(t+dt) = X_i(t) × exp(μ_i × dt × logistic)   # biomass update
ΔC_s      = Σ_i (-q_i × X_i + secretion_i) × dt  # media update
```

Logistic damping: `logistic = max(0, 1 - X_total / K_total)` prevents unbounded growth.

### Species-specific Monod parameters

From Marsh & Martin (1999), Jenkinson (1997), Periasamy et al. (2009):

| Species | μ_max [h⁻¹] | Primary substrate | Km [mM] | q_max [mmol/gDW/h] | Y [gDW/mmol] |
|---------|------------|-------------------|---------|---------------------|--------------|
| So | 0.50 | glucose | 0.05 | 8.0 | 0.10 |
| An | 0.35 | glucose | 0.08 | 6.0 | 0.08 |
| Vp | 0.40 | lactate | 0.15 | 10.0 | 0.07 |
| Fn | 0.32 | glucose + lactate | 0.12 / 0.18 | 4.0 / 5.0 | 0.09 / 0.08 |
| Pg | 0.20 | succ × hemin (product) | 0.08 / 0.005 | 3.0 / 0.5 | 0.10 / 0.12 |

**Cross-feeding stoichiometry** (from AGORA exchange reactions):

- So → Vp: 1.8 mmol lactate / mmol glucose consumed
- An → Vp: 1.2 mmol lactate / mmol glucose consumed

**O2 inhibition** (strict anaerobes Vp, Fn, Pg):

```
μ_effective = μ × 1 / (1 + 2.0 × [O2] / (0.01 + [O2]))
```

This reduces Vp/Fn/Pg growth ~3-fold in healthy (aerobic) conditions.

## Media Conditions

### Healthy (aerobic, glucose-rich)

```python
MEDIA_HEALTHY = {
    "glc_D[e]": 0.20,   # mM  (dietary glucose)
    "o2[e]":    0.50,   # mM  (aerobic surface)
    "lac_L[e]": 0.05,   # mM  (trace lactate)
    "nh4[e]":  10.0,
    "pi[e]":   10.0,
    "h2o[e]":  1000.0,
    "ca2[e]":   2.0,
    "mg2[e]":   1.0,
}
```

### Diseased (anaerobic, hemin-rich, peri-implantitis)

```python
MEDIA_DISEASED = {
    "glc_D[e]":  0.05,  # mM  (glucose depleted)
    "lac_L[e]":  0.20,  # mM  (accumulated lactate)
    "succ[e]":   0.10,  # mM  (succinate, Pg substrate)
    "pheme[e]":  0.50,  # mM  (protoheme/hemin from bleeding)
    "nh4[e]":   10.0,
    "pi[e]":    10.0,
    "h2o[e]":  1000.0,
    "ca2[e]":    2.0,
    "mg2[e]":    1.0,
}
```

## Expected Simulation Behavior (500 cycles, dt=0.01 h)

| Condition | Species trend | DI trajectory | Key driver |
|-----------|--------------|---------------|------------|
| Healthy | So ↑↑ (7×), Pg flat | 0.898 → 0.52 | Glucose → So dominance |
| Diseased | Fn ↑↑ (3×), Pg ↑ (1.5×), Vp ↑ | 0.886 → 0.91 | Lactate + hemin |

- **Healthy**: So grows fastest on abundant glucose; Vp O2-inhibited; Pg stagnates (no hemin)
- **Diseased**: Fn uses lactate + trace glucose; Vp freed from O2 inhibition; Pg needs hemin (product Monod)

## DI Computation

```python
from nife.comets.oral_biofilm import compute_di

di_df = compute_di(result.total_biomass)
# Returns pd.DataFrame with columns: cycle, DI
# DI = normalized Shannon entropy = H(φ) / log(N_species)
# DI ∈ [0, 1]:  0 = one species dominates,  1 = all equal
```

## API Reference

```python
OralBiofilmComets(
    comets_home=None,    # path to COMETS; auto-detected if None
    agora_dir=None,      # path to agora_gems/; auto-detected if None
    grid=(1, 1),         # spatial grid ((1,1) = well-mixed)
)

.run(condition, max_cycles, output_dir, delete_files, fallback_mock)
.run_dfba_cobra(condition, max_cycles, time_step, K_total, o2_inhibit_factor)
.run_mock(condition, max_cycles, noise)
.build_layout(condition)       # cometspy layout (for COMETS)
.build_params(max_cycles, ...) # cometspy params (for COMETS)
.extract_effective_growth_rates(biomass_df, dt)
.compute_di(biomass_df)        # static method
```

## Integration with Hamilton ODE / TMCMC

`tmcmc_bridge.py` provides cross-feeding flux maps in AGORA exchange format:

```python
CROSS_FEED_MAP = {
    ("So", "Vp"): "EX_lac_L(e)",
    ("An", "Vp"): "EX_lac_L(e)",
    ("Fn", "Pg"): "EX_succ(e)",
    ("Vp", "Fn"): "EX_pro_L(e)",
    ("So", "Fn"): "EX_ac(e)",
}
```

These cross-feeding fluxes correspond directly to the off-diagonal elements of the
Hamilton ODE A matrix calibrated in the TMCMC paper.

## Future Improvements

1. **VMH diet medium**: Obtain the full VMH diet files (Western/European diet) and
   switch `run_dfba_cobra()` to true FBA at each timestep. This would make the AGORA
   exchange stoichiometry fully active.

2. **Spatial dFBA**: Use `grid=(10, 10)` with COMETS (requires or-tools license) to
   model diffusion gradients (glucose depletes near the base, O2 depletes near the top).

3. **TMCMC calibration**: Replace literature Monod parameters with posterior MAP
   estimates from the TMCMC inference (θ → A matrix → growth rate corrections).
