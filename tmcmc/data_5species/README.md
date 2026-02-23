# 5-Species Model Synthetic Data

## Overview
This dataset was synthetically generated to compensate for the missing 5th species data in the original paper.
It uses an extended version of the continuum biofilm model (`improved_5species_jit.py`) with hypothetical but plausible parameters for the 5th species.

## Files
- `t_arr.npy`: Time steps array (Shape: (1001,))
- `g_arr.npy`: State vector array (Shape: (1001, 12))
  - Indices 0-4: $\phi_1$ to $\phi_5$ (Volume fractions)
  - Index 5: $\phi_0$ (Solvent)
  - Indices 6-10: $\psi_1$ to $\psi_5$ (Nutrients)
  - Index 11: $\gamma$ (Growth potential)
- `theta_true.npy`: True parameters used for generation (Shape: (20,))
- `5species_dynamics.png`: Visualization of the dynamics

## Parameters Used (True Theta)
The parameter vector $\theta$ consists of 20 elements:

| Index | Name | Value | Block | Description |
|---|---|---|---|---|
| 0 | a11 | 0.80 | M1 | S1 Self-interaction |
| 1 | a12 | 2.00 | M1 | S1-S2 Interaction |
| 2 | a22 | 1.00 | M1 | S2 Self-interaction |
| 3 | b1 | 0.10 | M1 | S1 Decay |
| 4 | b2 | 0.20 | M1 | S2 Decay |
| 5 | a33 | 1.50 | M2 | S3 Self-interaction |
| 6 | a34 | 1.00 | M2 | S3-S4 Interaction |
| 7 | a44 | 2.00 | M2 | S4 Self-interaction |
| 8 | b3 | 0.30 | M2 | S3 Decay |
| 9 | b4 | 0.40 | M2 | S4 Decay |
| 10 | a13 | 2.00 | M3 | S1-S3 Interaction |
| 11 | a14 | 1.00 | M3 | S1-S4 Interaction |
| 12 | a23 | 2.00 | M3 | S2-S3 Interaction |
| 13 | a24 | 1.00 | M3 | S2-S4 Interaction |
| 14 | a55 | 1.20 | M4 | S5 Self-interaction (**Hypothetical**) |
| 15 | b5 | 0.25 | M4 | S5 Decay (**Hypothetical**) |
| 16 | a15 | 1.00 | M5 | S1-S5 Interaction (**Hypothetical**) |
| 17 | a25 | 1.00 | M5 | S2-S5 Interaction (**Hypothetical**) |
| 18 | a35 | 1.00 | M5 | S3-S5 Interaction (**Hypothetical**) |
| 19 | a45 | 1.00 | M5 | S4-S5 Interaction (**Hypothetical**) |

## Simulation Details
- **Time steps**: 1001 (dt=1e-5)
- **Initial Condition**:
  - $\phi_i = 0.2$ for all active species
  - $\psi_i = 0.999$
  - $\gamma = 0.0$
- **Solver**: Newton-Raphson with Numba JIT acceleration (`improved_5species_jit.py`)
