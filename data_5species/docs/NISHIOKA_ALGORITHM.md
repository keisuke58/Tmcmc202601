# Nishioka Algorithm: Biologically-Constrained Parameter Reduction

## Overview

The Nishioka Algorithm is a parameter reduction technique for the 5-species biofilm model. Instead of estimating all 20 interaction parameters freely, it constrains certain parameters to zero based on biological knowledge from experimental interaction networks (Figure 4C).

## Biological Basis

The algorithm is derived from observed species interactions in biofilm communities:

### Species in the Model

| ID | Species | Abbreviation | Role |
|----|---------|--------------|------|
| 0 | *Streptococcus oralis* | S.o | Early colonizer |
| 1 | *Actinomyces naeslundii* | A.n | Early colonizer |
| 2 | *Veillonella* spp. | Vei | Metabolic bridge |
| 3 | *Fusobacterium nucleatum* | F.n | Bridge organism |
| 4 | *Porphyromonas gingivalis* | P.g | Late colonizer (pathogen) |

### Interaction Network (Figure 4C)

```
     S. oralis (0)
      /   |   \
     /    |    \
   A.n   Vei   F.n
   (1)   (2)   (3)
          \     /
           \   /
           P.g (4)
```

**Active Interactions:**
- S. oralis ↔ A. naeslundii: Co-aggregation
- S. oralis ↔ Veillonella: Lactate production/consumption
- S. oralis ↔ F. nucleatum: Formate/Acetate symbiosis
- Veillonella ↔ P. gingivalis: pH rise support
- F. nucleatum ↔ P. gingivalis: Co-aggregation, peptide provision

**Absent Interactions (Locked to 0):**
- A. naeslundii ↔ Veillonella: No direct metabolic link
- A. naeslundii ↔ F. nucleatum: No direct interaction
- A. naeslundii ↔ P. gingivalis: No direct interaction
- Veillonella ↔ F. nucleatum: No direct interaction
- S. oralis ↔ P. gingivalis: No direct interaction

## Mathematical Formulation

### Symmetric Matrix Assumption

**Critical assumption**: The interaction matrix **A** is symmetric:

```
A[i,j] = A[j,i]  for all i, j ∈ {0, 1, 2, 3, 4}
```

This means each off-diagonal interaction is represented by a single parameter. For example:
- **Lactate Handover** (S.o ↔ Vei): `A[0,2] = A[2,0] = theta[10]` (stored as `a13`)
- **pH Trigger** (Vei ↔ P.g): `A[2,4] = A[4,2] = theta[18]` (stored as `a35`)

### Parameter Vector Definition

The full 20-parameter vector θ is organized into 5 blocks:

```
θ = [Block M1] ⊕ [Block M2] ⊕ [Block M3] ⊕ [Block M4] ⊕ [Block M5]

Block M1 (Species 1-2):     a11, a12, a22, b1, b2     (indices 0-4)
Block M2 (Species 3-4):     a33, a34, a44, b3, b4     (indices 5-9)
Block M3 (Cross 1-2 vs 3-4): a13, a14, a23, a24       (indices 10-13)
Block M4 (Species 5):       a55, b5                   (indices 14-15)
Block M5 (Cross with 5):    a15, a25, a35, a45        (indices 16-19)
```

### Complete Parameter Mapping Table

| Index | Name | Matrix Element | Species Pair | Biological Role | Status |
|-------|------|----------------|--------------|-----------------|--------|
| 0 | a₁₁ | A[0,0] | S.o self | Self-regulation | Free |
| 1 | a₁₂ | A[0,1]=A[1,0] | S.o ↔ A.n | Co-aggregation | Free |
| 2 | a₂₂ | A[1,1] | A.n self | Self-regulation | Free |
| 3 | b₁ | b[0] | S.o | Decay rate | Free |
| 4 | b₂ | b[1] | A.n | Decay rate | Free |
| 5 | a₃₃ | A[2,2] | Vei self | Self-regulation | Free |
| **6** | **a₃₄** | **A[2,3]=A[3,2]** | **Vei ↔ F.n** | *No interaction* | **🔒 LOCKED** |
| 7 | a₄₄ | A[3,3] | F.n self | Self-regulation | Free |
| 8 | b₃ | b[2] | Vei | Decay rate | Free |
| 9 | b₄ | b[3] | F.n | Decay rate | Free |
| 10 | a₁₃ | A[0,2]=A[2,0] | S.o ↔ Vei | **Lactate handover** | Free |
| 11 | a₁₄ | A[0,3]=A[3,0] | S.o ↔ F.n | Formate symbiosis | Free |
| **12** | **a₂₃** | **A[1,2]=A[2,1]** | **A.n ↔ Vei** | *No interaction* | **🔒 LOCKED** |
| **13** | **a₂₄** | **A[1,3]=A[3,1]** | **A.n ↔ F.n** | *No interaction* | **🔒 LOCKED** |
| 14 | a₅₅ | A[4,4] | P.g self | Self-regulation | Free |
| 15 | b₅ | b[4] | P.g | Decay rate | Free |
| **16** | **a₁₅** | **A[0,4]=A[4,0]** | **S.o ↔ P.g** | *No interaction* | **🔒 LOCKED** |
| **17** | **a₂₅** | **A[1,4]=A[4,1]** | **A.n ↔ P.g** | *No interaction* | **🔒 LOCKED** |
| 18 | a₃₅ | A[2,4]=A[4,2] | Vei ↔ P.g | **pH trigger** | Free* |
| 19 | a₄₅ | A[3,4]=A[4,3] | F.n ↔ P.g | Co-aggregation | Free |

*Index 18 bounds vary by condition (see Prior Bounds section).

### Locked Parameter Indices

```python
LOCKED_INDICES = [6, 12, 13, 16, 17]
```

| Index | Param | Species Pair | Matrix | Biological Reason |
|-------|-------|--------------|--------|-------------------|
| 6 | a₃₄ | Vei (2) ↔ F.n (3) | A[2,3]=A[3,2] | No direct metabolic pathway |
| 12 | a₂₃ | A.n (1) ↔ Vei (2) | A[1,2]=A[2,1] | No direct metabolic link |
| 13 | a₂₄ | A.n (1) ↔ F.n (3) | A[1,3]=A[3,1] | No direct interaction |
| 16 | a₁₅ | S.o (0) ↔ P.g (4) | A[0,4]=A[4,0] | No direct interaction |
| 17 | a₂₅ | A.n (1) ↔ P.g (4) | A[1,4]=A[4,1] | No direct interaction |

### Prior Bounds

**Base bounds** (Commensal/Dysbiotic Static):

| Parameter Type | Bounds |
|----------------|--------|
| Locked interactions (k ∈ {6,12,13,16,17}) | [0.0, 0.0] |
| Vei → P.g (idx 18) | [0.0, 1.0] (positive only) |
| All other parameters | [-1.0, 1.0] |

**Dysbiotic HOBIC ("Surge" condition):**

| Parameter | Bounds | Reason |
|-----------|--------|--------|
| θ₁₀ (a₁₃, Lactate) | [-3.0, 0.0] | S1 helps S3 grow |
| θ₁₈ (a₃₅, pH) | **[-3.0, -1.0]** | Strong S3→S5 cooperation |
| θ₁₉ (a₄₅) | [-2.0, 0.0] | S4→S5 cooperation |

### Effective Parameter Space

```
n_free = 20 - |L| = 20 - 5 = 15 parameters
```

## Experiment Conditions

| Condition | Cultivation | Locked | Estimated | Key Constraint |
|-----------|-------------|--------|-----------|----------------|
| 1. Commensal | Static | 9 | 11 | Zero pathogen interactions |
| 2. Dysbiotic | Static | 5 | 15 | Standard Nishioka locks |
| 3. Commensal | HOBIC | 8 | 12 | Zero pathogen interactions |
| 4. **Dysbiotic** | **HOBIC** | **0** | **20** | **Unlock all (Surge)** |

## Implementation

### Key Code (`core/nishioka_model.py`)

```python
LOCKED_INDICES = [6, 12, 13, 16, 17]

def get_nishioka_bounds():
    bounds = [(-1.0, 1.0)] * 20

    # Lock absent interactions
    for idx in LOCKED_INDICES:
        bounds[idx] = (0.0, 0.0)

    # Positive constraint for Vei→P.g (base case)
    bounds[18] = (0.0, 1.0)

    return bounds, LOCKED_INDICES
```

### Usage in Estimation

```python
from core.nishioka_model import get_nishioka_bounds

# Get constrained bounds
nishioka_bounds, LOCKED_INDICES = get_nishioka_bounds()

# Lock parameters in theta_base
for idx in LOCKED_INDICES:
    theta_base[idx] = 0.0

# Update active indices (only estimate non-locked)
active_indices = [i for i in range(20) if i not in LOCKED_INDICES]
# Result: 15 free parameters instead of 20
```

## Comparison: Standard vs Nishioka

| Aspect | Standard | Nishioka |
|--------|----------|----------|
| Free parameters | 20 | 15 |
| Biological constraints | None | Fig 4C network |
| Prior knowledge | Minimal | Species interactions |
| Computational cost | Higher | Lower (fewer params) |
| Identifiability | May have issues | Improved |
| Interpretation | All params estimated | Biologically meaningful |

## Advantages

1. **Reduced Parameter Space**: 15 vs 20 parameters improves sampling efficiency
2. **Biological Validity**: Estimates respect known interaction networks
3. **Better Identifiability**: Fewer parameters to estimate from limited data
4. **Interpretability**: Non-zero parameters correspond to real interactions
5. **Regularization**: Implicit regularization by fixing parameters

## Limitations

1. **Model Dependence**: Requires accurate prior knowledge of interactions
2. **Rigidity**: Cannot discover unexpected interactions
3. **Network Uncertainty**: If Figure 4C is incomplete, model may be biased

## Running the Nishioka Estimation

```bash
# On marinos server (high memory)
nohup python main/estimate_reduced_nishioka.py \
    --condition Commensal --cultivation Static \
    --n-particles 2000 --n-stages 30 --n-chains 2 \
    --use-exp-init --start-from-day 3 --normalize-data \
    --output-dir _runs/nishioka_v1 \
    > nishioka.log 2>&1 &
```

## Expected Output

```
_runs/nishioka_v1_YYYYMMDD_HHMMSS/
├── config.json              # Includes locked_indices
├── posterior_samples.csv    # 15 active parameters
├── theta_MAP.json
├── theta_MEAN.json
├── fit_metrics.json
└── figures/
    ├── TSM_simulation_MAP_fit.png
    ├── posterior_corner.png
    └── ...
```

## References

- Heine et al. (2025): Original paper describing species interaction network
- Figure 4C: Experimentally determined species interaction diagram
- TSM Model: Taylor Series Method for biofilm dynamics (Geisler et al.)
- TMCMC: Ching & Chen (2007) Transitional MCMC algorithm

---

*Last updated: 2026-02-05*
