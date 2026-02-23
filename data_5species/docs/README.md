# 5-Species Biofilm TMCMC Parameter Estimation

## Overview

This project implements Transitional Markov Chain Monte Carlo (TMCMC) for parameter estimation in a 5-species biofilm model. The model estimates 20 parameters describing species interactions and decay rates from experimental data.

## Project Structure

```
data_5species/
├── main/                          # Main scripts
│   ├── estimate_commensal_static.py   # TMCMC estimation
│   ├── generate_all_figures.py        # Post-processing visualization
│   └── compare_runs.py                # Compare two runs
├── core/                          # Core TMCMC implementation
├── visualization/                 # Plotting utilities
├── experiment_data/               # Experimental data files
├── _runs/                         # Output directories
├── docs/                          # Documentation
└── *.sh                           # Run scripts
```

## Experimental Data

### Conditions Available
| Condition | Cultivation | Data File |
|-----------|-------------|-----------|
| Commensal | Static | `boxplot_Commensal_Static.csv` |
| Commensal | HOBIC | `boxplot_Commensal_HOBIC.csv` |
| Dysbiotic | Static | `boxplot_Dysbiotic_Static.csv` |
| Dysbiotic | HOBIC | `boxplot_Dysbiotic_HOBIC.csv` |

### Species Mapping
| Index | Species | Color Code |
|-------|---------|------------|
| 0 | S. oralis | Blue |
| 1 | A. naeslundii | Green |
| 2 | V. dispar | Yellow |
| 3 | F. nucleatum | Purple |
| 4 | P. gingivalis | Red |

## Model Parameters (20 total)

### Parameter Structure
```
θ = [a11, a12, a22, b1, b2,     # M1: Species 0-1 block (5 params)
     a33, a34, a44, b3, b4,     # M2: Species 2-3 block (5 params)
     a13, a14, a23, a24,        # M3: Cross-interactions (4 params)
     a55, b5,                   # M4: Species 4 self (2 params)
     a15, a25, a35, a45]        # M5: Species 4 cross (4 params)
```

### Parameter Types
- **a_ij**: Interaction coefficients (symmetric: a_ij = a_ji)
- **b_i**: Decay/death rate for species i

## Quick Start

### 1. Run Basic Estimation
```bash
python main/estimate_commensal_static.py \
    --condition Commensal --cultivation Static \
    --n-particles 500 --n-stages 30 --n-chains 2 \
    --use-exp-init --start-from-day 3 --normalize-data
```

### 2. Run with More Particles (Improved)
```bash
nohup bash run_improved_estimation.sh > improved.log 2>&1 &
```

### 3. Run with Tight Decay Priors
```bash
nohup bash run_tight_decay_estimation.sh > tight_decay.log 2>&1 &
```

### 4. Run HOBIC Condition
```bash
nohup bash run_hobic_estimation.sh > hobic.log 2>&1 &
```

### 5. Generate Figures from Existing Run
```bash
python main/generate_all_figures.py --run-dir _runs/Commensal_Static_20260204_062733
```

### 6. Compare Two Runs
```bash
python main/compare_runs.py --run1 _runs/run1 --run2 _runs/run2
```

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--condition` | Commensal | Commensal or Dysbiotic |
| `--cultivation` | Static | Static or HOBIC |
| `--n-particles` | 500 | Number of TMCMC particles |
| `--n-stages` | 30 | Maximum TMCMC stages |
| `--n-chains` | 2 | Number of parallel chains |
| `--use-exp-init` | False | Use experimental data as initial conditions |
| `--start-from-day` | 1 | Start fitting from this day |
| `--normalize-data` | False | Normalize to species fractions |
| `--prior-decay-max` | None | Limit decay parameters b_i to [0, max] |
| `--n-jobs` | None | Parallel workers (None = auto) |
| `--lambda-pg` | 1.0 | Weight multiplier for P. gingivalis (species 5) |
| `--lambda-late` | 1.0 | Weight multiplier for the last n timepoints |
| `--n-late` | 2 | Number of late timepoints to upweight |
| `--seed` | None | Random seed for reproducibility |
| `--compute-evidence` | False | Compute log-model-evidence from TMCMC |
| `--output-dir` | None | Custom output directory |

## Weighted Likelihood (P. gingivalis Emphasis)

For the Dysbiotic HOBIC condition, P. gingivalis growth at late timepoints is critical.
A **weighted power-likelihood** amplifies the contribution of specific species and timepoints:

```bash
python main/estimate_reduced_nishioka.py \
    --condition Dysbiotic --cultivation HOBIC \
    --lambda-pg 5.0 --lambda-late 3.0 --n-late 2 \
    --n-particles 500 --n-stages 30 --n-chains 2 --seed 42
```

The weight matrix `W[t,s] = w_species[s] * w_time[t]` is constructed as:
- `w_species = [1, 1, 1, 1, lambda_pg]`
- `w_time = [1, ..., 1, lambda_late, ..., lambda_late]` (last `n_late` timepoints)

With `lambda_pg=5, lambda_late=3, n_late=2`: P.g at day 21 gets weight **15x** vs 1x for early commensals.

### Hill Function Gating

The Hill function gates P. gingivalis interactions based on F. nucleatum biomass:
- `K_hill`: half-saturation constant (0.0 = disabled, recommended: 0.02-0.10)
- `n_hill`: Hill coefficient (fixed at 2)

Set via `model_config/` or directly in the estimation script.

### Sweep Script

Run multiple configurations in parallel:
```bash
nohup bash run_pg_weighted_sweep.sh > sweep.log 2>&1 &
bash check_sweep.sh   # monitor progress
```

## Output Files

Each run creates a directory with:

| File | Description |
|------|-------------|
| `config.json` | Run configuration |
| `samples.npy` | Posterior samples (n_samples × 20) |
| `logL.npy` | Log-likelihood values |
| `theta_MAP.json` | Maximum a posteriori estimate |
| `theta_mean.json` | Posterior mean estimate |
| `fit_metrics.json` | RMSE, MAE per species |
| `figures/` | Generated plots |

## Generated Figures

1. `TSM_simulation_*_MAP_Fit_with_data.png` - MAP estimate fit
2. `TSM_simulation_*_Mean_Fit_with_data.png` - Mean estimate fit
3. `residuals_*.png` - Per-species residuals
4. `parameter_distributions_*.png` - Parameter histograms
5. `corner_plot_*.png` - Corner plot (20×20)
6. `posterior_predictive_*.png` - 5-95% confidence bands
7. `posterior_predictive_spaghetti_*.png` - Sample trajectories

## Key Findings

### Data Patterns (Commensal Static)
- **S. oralis**: 71% → 74% → 66% → 37% → 35% (plateau then decline)
- **A. naeslundii**: 8% → 5% → 9% → 24% → 35% (dip then growth)
- **V. dispar**: 20% → 20% → 24% → 38% → 30% (growth then decline)

### Model Limitations
- Model predicts monotonic decrease for S. oralis
- Data shows initial stability/increase before decline
- Tighter decay priors may help: `--prior-decay-max 1.0`

## References

- TMCMC: Ching & Chen (2007)
- Biofilm model: [Project-specific reference]
