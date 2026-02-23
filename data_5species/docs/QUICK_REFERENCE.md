# Quick Reference Guide

## Commands

### Run Estimation
```bash
# Basic run
python main/estimate_commensal_static.py --condition Commensal --cultivation Static

# Full options
python main/estimate_commensal_static.py \
    --condition Commensal \
    --cultivation Static \
    --n-particles 1000 \
    --n-stages 50 \
    --n-chains 2 \
    --use-exp-init \
    --start-from-day 3 \
    --normalize-data \
    --prior-decay-max 1.0 \
    --n-jobs 10 \
    --seed 42 \
    --output-dir _runs/my_run
```

### Weighted Likelihood (P.g. emphasis)
```bash
python main/estimate_reduced_nishioka.py \
    --condition Dysbiotic \
    --cultivation HOBIC \
    --lambda-pg 5.0 \
    --lambda-late 3.0 \
    --n-late 2 \
    --n-particles 500 \
    --n-stages 30 \
    --n-chains 2 \
    --seed 42 \
    --compute-evidence \
    --output-dir _runs/my_weighted_run
```

### Background Execution
```bash
nohup bash run_improved_estimation.sh > improved.log 2>&1 &
nohup bash run_pg_weighted_sweep.sh > sweep.log 2>&1 &
```

### Monitor Jobs
```bash
# Check if running
ps aux | grep estimate_reduced | grep -v grep

# View log
tail -f improved.log

# Check sweep progress
bash check_sweep.sh

# Check progress
grep "Stage" improved.log | tail -5
```

### Post-Processing
```bash
# Generate all figures
python main/generate_all_figures.py --run-dir _runs/YOUR_RUN

# Compare runs
python main/compare_runs.py --run1 _runs/baseline --run2 _runs/improved
```

---

## Parameter Indices

| Index | Name | Description |
|-------|------|-------------|
| 0 | a11 | S. oralis self-interaction |
| 1 | a12 | S. oralis ↔ A. naeslundii |
| 2 | a22 | A. naeslundii self-interaction |
| 3 | b1 | S. oralis decay |
| 4 | b2 | A. naeslundii decay |
| 5 | a33 | V. dispar self-interaction |
| 6 | a34 | V. dispar ↔ F. nucleatum |
| 7 | a44 | F. nucleatum self-interaction |
| 8 | b3 | V. dispar decay |
| 9 | b4 | F. nucleatum decay |
| 10 | a13 | S. oralis ↔ V. dispar |
| 11 | a14 | S. oralis ↔ F. nucleatum |
| 12 | a23 | A. naeslundii ↔ V. dispar |
| 13 | a24 | A. naeslundii ↔ F. nucleatum |
| 14 | a55 | P. gingivalis self-interaction |
| 15 | b5 | P. gingivalis decay |
| 16 | a15 | S. oralis ↔ P. gingivalis |
| 17 | a25 | A. naeslundii ↔ P. gingivalis |
| 18 | a35 | V. dispar ↔ P. gingivalis |
| 19 | a45 | F. nucleatum ↔ P. gingivalis |

---

## Output Files

| File | Description |
|------|-------------|
| `config.json` | Run settings |
| `samples.npy` | Posterior samples (n×20) |
| `logL.npy` | Log-likelihood values |
| `theta_MAP.json` | MAP estimate |
| `theta_mean.json` | Mean estimate |
| `fit_metrics.json` | RMSE, MAE |
| `figures/*.png` | Plots |

---

## Troubleshooting

### Job not progressing
```bash
# Check CPU usage
top -u $USER

# Check memory
free -h

# Kill stuck job
pkill -f estimate_commensal
```

### Weighted Likelihood Parameters
| Argument | Default | Description |
|----------|---------|-------------|
| `--lambda-pg` | 1.0 | Weight for P. gingivalis (species 5) |
| `--lambda-late` | 1.0 | Weight for last n timepoints |
| `--n-late` | 2 | Number of late timepoints to upweight |

Weight matrix: `W[t,s] = w_species[s] * w_time[t]`
- lambda_pg=5, lambda_late=3 → P.g at day 21 gets 15x weight

---

### Missing figures
```bash
# Regenerate all figures
python main/generate_all_figures.py --run-dir _runs/YOUR_RUN --force
```

### Import errors
```bash
# Ensure correct directory
cd /home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species

# Check Python path
python -c "import sys; print(sys.path)"
```
