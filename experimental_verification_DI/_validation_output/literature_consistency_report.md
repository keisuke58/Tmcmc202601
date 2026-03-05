# Literature Consistency Report

## Model Parameters
- E_max = 909.0 Pa, E_min = 32.0 Pa
- di_scale = 1.0, exponent = 2.0

## Comparison with Literature

| Source | DI (approx) | E_lit [kPa] | E_model [kPa] | Residual [%] |
|--------|-------------|-------------|---------------|---------------|
| Pattem 2018 (low-suc) | 0.15 | 14.35 | 0.66 | -95 |
| Pattem 2018 (high-suc) | 0.85 | 0.55 | 0.05 | -91 |
| Pattem 2021 (LC) | 0.20 | 10.40 | 0.59 | -94 |
| Pattem 2021 (HC) | 0.75 | 2.80 | 0.08 | -97 |
| Gloag 2019 (dual) | 0.50 | 0.48 | 0.24 | -49 |

## Summary
- Mean |residual| = 86%
- Literature (Pattem, Gloag) reports different biofilm systems and protocols; absolute E values differ. Our model captures the qualitative trend (diverse→stiffer, dysbiotic→softer). Direct validation requires same-sample 16S+AFM data.

## Next Step
Direct experimental validation: 16S rRNA + AFM on same sample. See EXPERIMENTAL_PROTOCOL.md.
