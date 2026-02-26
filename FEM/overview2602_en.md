# Method Overview — Big Picture

## 1. What Are We Doing? (One Sentence)

**"Quantifying inter-species interactions in oral biofilm via Bayesian inference"**

- **Input:** Experimental time-series data
- **Process:** Parameter estimation (TMCMC)
- **Output:** Bacterial interaction network (strengths + uncertainty)

---

## 2. Overall Flow

| **Input** | **Model** | **Inference** | **Output** |
|-----------|-----------|---------------|------------|
| Experimental data | ODE model | TMCMC | Parameters |
| 5 species volume fractions | Growth & interaction | Prior → posterior | Interaction matrix **A[i,j]** |
| 6 time points | dφ/dt = f(θ) | Progressive sampling | (20 parameters) |
| 📊 [t×5 array] | 🧬 ODE | 🎲 TMCMC | 📈 MAP + 95% CI |

---

## 3. Input Data Structure

**Experimental conditions (4 types):**

- Commensal × Static (healthy × static)
- Commensal × HOBIC (healthy × high oxygen)
- Dysbiotic × Static (dysbiotic × static)
- **Dysbiotic × HOBIC** ← main focus

**Data format:**

- Time [days]: 1, 3, 6, 10, 15, 21
- Species: So (S. oralis), An (A. naeslundii), Vd (V. dispar), Fn (F. nucleatum), Pg (P. gingivalis)
- Shape: **[6 time points × 5 species]**; values in [0, 1] (volume fraction)
- Constraint: **Σ φᵢ = 1** (volume conservation)

---

## 4. Biological Model (ODE Intuition)

- **So → An**: Inter-species “help/competition” encoded in matrix **A**
- **Vd → Fn → Pg**: Fn as gatekeeper (Hill function)
- **Interaction matrix A [5×5]** = target of estimation
- Zeros fixed by biological knowledge (locked).
- Plus: viability ψᵢ and decay rates bᵢ → **20 parameters total**

---

## 5. TMCMC Algorithm (Intuition)

**Idea:** Gradually move from prior to posterior.

- **β = 0**: Prior — spread-out particles
- **β = 0.5**: Intermediate — pulled toward data
- **β = 1**: Posterior — concentrated distribution

**Stages:** 1 → 2 → … → N (e.g. 8 stages)

**Per stage:**

1. Set β (ESS ≈ 50% via bisection)
2. Weighted resampling
3. MCMC mutation (150 particles in parallel)
4. Update linearization point every 3 stages (ROM accuracy)

---

## 6. Computational Setup (Speed)

- **Heavy part:** One θ → ODE integration → likelihood
- **Speedups:**
  - **Numba JIT** (2–3×)
  - **TSM-ROM** (reduced model, ~1/10 time; check every 3 stages)
  - **Parallel evaluation** (ProcessPoolExecutor, 4–8× for 150 particles)

---

## 7. Output Overview

- `config.json` — run settings
- `posterior_samples.npy` — posterior samples
- `metrics.json` — RMSE, logL, convergence
- `parameter_summary.csv` — MAP, mean, 95% CI
- `diagnostics_tables/` — β schedule, acceptance rate, ESS, Rhat
- `figures/` — posterior plots, fit vs data, convergence

---

## 8. Why This Method? (Positioning)

- **Optimization only:** Fast but no uncertainty (no CI).
- **Plain MCMC:** Can fail on multimodal/high-dim.
- **Neural ODE etc.:** Need lots of data; less interpretable.
- **TMCMC:** Strong for **limited data, multimodality, high dimension** — full posterior + MAP + CI.

---

## 9. One-Page Summary

| **Experiment** | **Model** | **Inference** | **Insight** |
|----------------|-----------|---------------|-------------|
| [Species abundances] | [ODE] | [TMCMC] | [Interaction strengths] |
| [6 time points] | [20 params] | [8 stages] | [MAP + 95% CI] |
| [5 species] | [Aij matrix] | [150 particles] | [RMSE, diagnostics] |
| | [Hill gate] | [Parallel] | |

**Takeaway:**
- **What:** Bayesian estimation of 5-species interaction matrix (20 parameters).
- **How:** ODE model × TMCMC (temperature schedule) × parallel + ROM speedup.
- **Why it works:** Method combination suited to scarce data, nonlinearity, and high dimension.
