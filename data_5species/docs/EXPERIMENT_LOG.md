# Experiment Log

## Run History

### Run 1: Baseline (Commensal Static)
- **Date**: 2026-02-04
- **Directory**: `_runs/Commensal_Static_20260204_062733`
- **Settings**: 500 particles, 30 stages, 2 chains
- **Runtime**: 3.4 hours
- **Status**: Completed, both chains converged

**Results**:
| Metric | MAP | Mean |
|--------|-----|------|
| Total RMSE | 0.0614 | 0.1051 |
| Total MAE | 0.0431 | 0.0705 |

**Issues**:
- Only 1 figure generated (plot errors)
- Poor fit for Species 1 (S. oralis)

**Fix Applied**: Created `generate_all_figures.py` to regenerate all plots

---

### Run 2: Improved (1000 particles)
- **Date**: 2026-02-04
- **Directory**: `_runs/improved_v1_20260204_191344`
- **Settings**: 1000 particles, 50 stages, 2 chains
- **Runtime**: In progress
- **Status**: Running (Chain 1, Stage 1)

**Purpose**: Test if more particles improve fit

---

### Run 3: Tight Decay (Queued)
- **Script**: `run_tight_decay_estimation.sh`
- **Settings**: 1000 particles, 50 stages, `--prior-decay-max 1.0`
- **Status**: Queued

**Purpose**: Limit decay parameters to [0, 1] to reduce over-prediction of decline

---

### Run 4: HOBIC
- **Date**: 2026-02-05
- **Server**: frontale03
- **Directory**: `_runs/Commensal_HOBIC_20260205_003113`
- **Settings**: 1000 particles, 50 stages, 2 chains
- **Status**: Running

**Purpose**: Test if model fits better for HOBIC cultivation condition

---

### Run 5: Nishioka Reduced (Parameter Reduction)
- **Date**: 2026-02-05
- **Server**: marinos01
- **Script**: `main/estimate_reduced_nishioka.py`
- **Settings**: 2000 particles, 30 stages, 24 jobs
- **Status**: Running

**Key Feature**: Nishioka Algorithm - locks 5 parameters to 0 based on biological knowledge
- **Locked indices**: [6, 12, 13, 16, 17]
- **Free parameters**: 15 (vs 20 in standard)
- **Biological basis**: Figure 4C interaction network

**Purpose**: Test if biologically-constrained parameter space improves estimation

See: `docs/NISHIOKA_ALGORITHM.md` for full documentation

---

## Data Analysis Summary

### Commensal Static - Species Patterns

| Day | S. oralis | A. naeslundii | V. dispar | Change S. oralis |
|-----|-----------|---------------|-----------|------------------|
| 3 | 70.7% | 8.1% | 20.2% | - |
| 6 | 74.3% | 5.0% | 19.8% | +3.6pp |
| 10 | 66.0% | 9.4% | 23.6% | -8.2pp |
| 15 | 36.5% | 24.0% | 38.5% | -29.5pp |
| 21 | 34.7% | 34.7% | 29.7% | -1.9pp |

**Key Finding**: S. oralis shows plateau (Day 3-6) then decline, but model predicts monotonic decrease.

### Commensal HOBIC - Species Patterns

| Day | S. oralis | A. naeslundii | V. dispar |
|-----|-----------|---------------|-----------|
| 3 | 95.0% | 0.5% | 10.0% |
| 6 | 75.0% | 0.5% | 25.0% |
| 10 | 55.0% | 12.0% | 30.0% |
| 15 | 35.0% | 28.0% | 32.0% |
| 21 | 45.0% | 25.0% | 25.0% |

**Key Finding**: HOBIC shows recovery at Day 21 (35% → 45%)

---

## Scripts Created

| Date | Script | Purpose |
|------|--------|---------|
| 2026-02-04 | `generate_all_figures.py` | Post-process existing runs |
| 2026-02-04 | `compare_runs.py` | Compare two runs |
| 2026-02-04 | `run_improved_estimation.sh` | 1000 particles run |
| 2026-02-04 | `run_tight_decay_estimation.sh` | Tight decay bounds |
| 2026-02-04 | `run_hobic_estimation.sh` | HOBIC condition |
| 2026-02-05 | `compare_nishioka_standard.py` | Compare Nishioka vs Standard results |
| 2026-02-05 | `check_jobs.sh` | Monitor jobs across servers |

---

## Code Changes

### 2026-02-04: estimate_commensal_static.py
- Added `--prior-decay-max` argument
- Added error handling for plot generation
- Added TMCMC diagnostics export
- Added Mean fit plot
- Added fit_metrics.json output

---

## Active Jobs (2026-02-05)

| Server | Job | Output Dir | Log |
|--------|-----|------------|-----|
| frontale01 | Improved (1000p) | `improved_v1_20260205_002904` | `frontale_improved.log` |
| frontale02 | Tight Decay | `tight_decay_v1_20260205_003111` | `frontale02_tight_decay.log` |
| frontale03 | HOBIC | `Commensal_HOBIC_20260205_003113` | `frontale03_hobic.log` |
| marinos01 | Nishioka Reduced | TBD | TBD |

### Monitor Commands
```bash
# Check all frontale servers
bash check_jobs.sh

# Check individual servers
ssh frontale01 "ps aux | grep estimate | grep -v grep | wc -l"
ssh marinos01 "ps aux | grep estimate | grep -v grep | wc -l"
```

---

## Next Steps

1. [x] Start improved estimation (frontale01)
2. [x] Start tight decay estimation (frontale02)
3. [x] Start HOBIC estimation (frontale03)
4. [x] Start Nishioka reduced estimation (marinos01)
5. [x] Document Nishioka Algorithm (`docs/NISHIOKA_ALGORITHM.md`)
6. [x] Create comparison script (`compare_nishioka_standard.py`)
7. [ ] Wait for all jobs to complete (~6-8 hours each)
8. [ ] Compare results:
   - `python compare_runs.py <run1> <run2>` (Standard comparisons)
   - `python compare_nishioka_standard.py <nishioka_run> <standard_run>` (Nishioka vs Standard)
9. [ ] Analyze which approach gives best fit
