# Parameter Reference

## The 20-Parameter Hamilton ODE System

The 5-species biofilm dynamics are governed by:

```
dφᵢ/dt = φᵢ · (rᵢ − dᵢ·φᵢ + Σⱼ aᵢⱼ · H(φⱼ; K, n))
```

where `H(φ; K, n) = φⁿ / (Kⁿ + φⁿ)` is the Hill gate (K=0.05, n=4 fixed).

Species index: 1=So, 2=An, 3=Vd, 4=Fn, 5=Pg

---

## Full Parameter Table

| θ index | Symbol | Meaning | Prior bounds | Units |
|---------|--------|---------|:------------:|-------|
| θ[0] | r₁ | *S. oralis* growth rate | [0, 5] | 1/h |
| θ[1] | r₂ | *A. naeslundii* growth rate | [0, 5] | 1/h |
| θ[2] | r₃ | *V. dispar* growth rate | [0, 5] | 1/h |
| θ[3] | r₄ | *F. nucleatum* growth rate | [0, 5] | 1/h |
| θ[4] | r₅ | *P. gingivalis* growth rate | [0, 5] | 1/h |
| θ[5] | d₁ | *S. oralis* self-inhibition | [0, 10] | — |
| θ[6] | d₂ | *A. naeslundii* self-inhibition | [0, 10] | — |
| θ[7] | d₃ | *V. dispar* self-inhibition | [0, 10] | — |
| θ[8] | d₄ | *F. nucleatum* self-inhibition | [0, 10] | — |
| θ[9] | d₅ | *P. gingivalis* self-inhibition | [0, 10] | — |
| θ[10] | a₁₂ | An → So cross-feeding | [0, 5] | — |
| θ[11] | a₂₁ | So → An cross-feeding | [0, 5] | — |
| θ[12] | **a₂₃** | So → An cross-feeding (bridge) | [0, 5] | — |
| θ[13] | a₃₁ | Vd → So | [0, 5] | — |
| θ[14] | a₃₂ | Vd → An | [0, 5] | — |
| θ[15] | a₄₁ | Fn → So | [0, 5] | — |
| θ[16] | a₄₂ | Fn → An | [0, 5] | — |
| θ[17] | a₄₃ | Fn → Vd | [0, 5] | — |
| θ[18] | **a₃₅** | *V. dispar* → *P. gingivalis* facilitation | [0, 5] | — |
| θ[19] | **a₄₅** | *F. nucleatum* → *P. gingivalis* facilitation | [0, 5] | — |

**Bold** = ecologically significant bridge-mediated parameters.

---

## Nishioka Interaction Mask

Not all 5×5 = 25 interactions are free. The **Nishioka mask** fixes 10 pairs (zero by assumption) and estimates 15 active parameters (including the 5 diagonal self-inhibition terms).

```python
from data_5species.core.nishioka_model import get_nishioka_mask
mask = get_nishioka_mask()  # shape (5, 5), True = estimated
```

Active edges (off-diagonal): `a₁₂, a₂₁, a₂₃, a₃₁, a₃₂, a₄₁, a₄₂, a₄₃, a₃₅, a₄₅`

Locked to zero: `a₁₃, a₁₄, a₁₅, a₂₄, a₂₅, a₃₄, a₅₁, a₅₂, a₅₃, a₅₄`

---

## Hill Gate Parameters (Fixed)

| Parameter | Value | Meaning |
|-----------|:-----:|---------|
| K_hill | 0.05 | half-saturation constant |
| n_hill | 4 | Hill cooperativity exponent |

These are fixed (not estimated) to reduce identifiability issues. Set in `tmcmc/program2602/improved_5species_jit.py`.

---

## Prior Bounds File

Location: `data_5species/model_config/prior_bounds.json`

```json
{
  "r": [0, 5],
  "d": [0, 10],
  "a_bridge": [0, 5],
  "a35": [0, 5],
  "a45": [0, 5]
}
```

Backups:
- `prior_bounds_original.json` — original wide bounds (a35: [0,30], a45: [0,20])
- `prior_bounds_narrow.json` — tight bounds for constrained runs

---

## Best MAP Estimates (Dysbiotic HOBIC, 2026-02-08)

| Symbol | θ_MAP | θ_MEAN | Interpretation |
|--------|:-----:|:------:|----------------|
| a₃₅ | 3.56 | 3.21 | Vd→Pg facilitation: **active** |
| a₄₅ | 2.41 | 2.18 | Fn→Pg facilitation: **active** |
| r₅ | 1.12 | 1.04 | Pg growth rate |
| d₅ | 4.30 | 3.95 | Pg self-inhibition |

Compared to commensal conditions where a₃₅ ≈ 0.3 and a₄₅ ≈ 0.1, the ~10× elevation confirms bridge-mediated dysbiosis.
