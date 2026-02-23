# TMCMC Guide

## Running an Estimation

### Basic Run

```bash
cd Tmcmc202601
python data_5species/main/estimate_reduced_nishioka.py \
    --condition "Dysbiotic_HOBIC" \
    --n-particles 150 \
    --n-stages 8
```

### Full Options

```bash
python data_5species/main/estimate_reduced_nishioka.py \
    --condition "Dysbiotic_HOBIC"   # CS / CH / DS / DH
    --n-particles 1000              # posterior resolution (150 = fast, 1000 = paper quality)
    --n-stages 8                    # TMCMC tempering stages (5–15 typical)
    --lambda-pg 2.0                 # likelihood weight for P. gingivalis
    --lambda-late 1.5               # extra weight on late time points
    --seed 42                       # random seed
```

### Production Run (All 4 Conditions)

```bash
python run_bridge_sweep.py --n-particles 1000 --n-stages 10
```

Output goes to `data_5species/_runs/<Condition>_<timestamp>/`.

---

## Output Directory Layout

```
_runs/Dysbiotic_HOBIC_20260208_002100/
├── theta_MAP.json           # MAP estimate (20 params)
├── theta_MEAN.json          # Posterior mean
├── posterior_samples.npy    # (N_particles × 20) array
├── figures/
│   ├── TSM_simulation_*_MAP_Fit_with_data.png
│   ├── posterior_predictive_*_PosteriorBand.png
│   ├── pub_interaction_heatmap_*.png
│   └── Fig01_TMCMC_beta_schedule_*.png
└── config.json              # Run configuration
```

---

## Algorithm: TMCMC

TMCMC (Ching & Chen 2007) bridges the prior to the posterior via a sequence of intermediate distributions:

```
p_j(θ) ∝ p(θ) · L(θ|data)^{β_j}     β_0=0, β_J=1
```

Each stage:
1. Compute importance weights `w_i = L(θᵢ)^{Δβ}`
2. Resample particles proportional to `wᵢ`
3. MCMC step (Metropolis) to diversify

The annealing schedule β is chosen adaptively so that the **Coefficient of Variation** of weights ≈ a target (typically 1.0).

### Key Diagnostics

| Metric | Healthy range | Warning |
|--------|:-------------:|---------|
| ESS (Effective Sample Size) | > 0.5 × N | < 0.2 × N → particle collapse |
| β progression | smooth 0→1 | many stages = narrow likelihood |
| Acceptance rate | 0.2–0.5 | < 0.1 → proposal too wide |
| R-hat | < 1.1 | > 1.2 → poor mixing |

---

## Likelihood Weights

The likelihood is a weighted Gaussian:

```
log L(θ) = −½ Σᵢ λᵢ · ||y_obs,i − y_model,i||² / σᵢ²
```

| Species | Default λ | Notes |
|---------|:---------:|-------|
| So, An, Vd, Fn | 1.0 | standard |
| Pg | 2.0 (`--lambda-pg`) | upweighted — harder to fit |

`--lambda-late` applies additional weight to the 24h and 48h time points.

Weights are set in `data_5species/core/evaluator.py` → `build_likelihood_weights()`.

---

## Tuning Tips

| Goal | Adjustment |
|------|-----------|
| Faster run | Reduce `--n-particles` to 150, `--n-stages` to 6 |
| Better Pg fit | Increase `--lambda-pg` (2.0–4.0) |
| Tighter posterior | Narrow prior bounds in `model_config/prior_bounds.json` |
| Poor convergence | Increase `--n-particles` or check initial proposal scale |

### Prior Bounds

Edit `data_5species/model_config/prior_bounds.json`. Key entries:

```json
{
  "a35": [0, 5],   // Vd → Pg facilitation (was [0,30])
  "a45": [0, 5],   // Fn → Pg facilitation (was [0,20])
  ...
}
```

Backups: `prior_bounds_original.json`, `prior_bounds_narrow.json`.

---

## Interpreting Results

### theta_MAP.json

```json
{
  "theta": [r1, r2, ..., r20],   // 20 parameters
  "log_likelihood": -45.3,
  "rmse_per_species": [0.04, 0.07, 0.10, 0.08, 0.06]
}
```

### Key Parameters to Check

| θ index | Symbol | Expected (Dysbiotic HOBIC) |
|---------|--------|--------------------------|
| θ[18] | a₃₅ (Vd→Pg) | 2–5 (large = bridge active) |
| θ[19] | a₄₅ (Fn→Pg) | 1–4 |
| θ[12] | a₂₃ (So→An) | small positive |

If a₃₅ or a₄₅ hit the upper bound → widen prior or increase `--lambda-pg`.
