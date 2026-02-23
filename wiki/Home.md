# Tmcmc202601 Wiki

Welcome to the **Tmcmc202601** project wiki — a 5-species oral biofilm Bayesian inference + FEM stress analysis pipeline.

---

## Pages

| Page | Description |
|------|-------------|
| [Installation](Installation) | Environment setup (TMCMC + JAX-FEM conda) |
| [TMCMC Guide](TMCMC-Guide) | Running estimations, key options, interpreting output |
| [FEM Pipeline](FEM-Pipeline) | From TMCMC posterior to Abaqus stress analysis |
| [Multiscale Coupling](Multiscale-Coupling) | Micro→Macro: ODE ecology → spatial eigenstrain for Abaqus |
| [Parameter Reference](Parameter-Reference) | All 20 θ parameters with biological meaning |
| [Results Gallery](Results-Gallery) | Key figures from best runs (chronological) |
| [Results by Category](Results-by-Category) | Same figures, organized by analysis type |
| [TMCMC Per-Condition](TMCMC-Per-Condition) | Detailed posteriors, fits & diagnostics for all 4 conditions |
| [FEM Spatial Analysis](FEM-Spatial-Analysis) | 1D dynamics, 3D FEM fields, DI depth profiles |
| [Uncertainty & Mechanics](Uncertainty-and-Mechanics) | Stress UQ, anisotropy, CZM, benchmarks, nutrient PDE |
| [Troubleshooting](Troubleshooting) | Common errors and fixes |

---

## Quick Overview

```
In vitro data (4 conditions × 5 species × 5 time points)
        │
        ▼
  TMCMC Bayesian Inference
  Hamilton ODE · 20 parameters
  → θ_MAP · 1000 posterior samples
        │
        ├──────────────────────────────┐
        ▼                              ▼
  3D FEM Stress Analysis       Multiscale Micro→Macro
  DI(x) → E(x) → Abaqus       0D ODE → DI_0D (18× diff)
  → S_Mises · U_max            1D PDE → α_Monod(x)
                               → ε_growth(x) → Abaqus INP
```

### Five Species

| Abbr. | Species | Role |
|-------|---------|------|
| So | *Streptococcus oralis* | Early coloniser |
| An | *Actinomyces naeslundii* | Early coloniser |
| Vd | *Veillonella dispar* | Bridge organism |
| Fn | *Fusobacterium nucleatum* | Bridge organism |
| Pg | *Porphyromonas gingivalis* | Keystone pathogen |

### Four Conditions

| Condition | Short name |
|-----------|------------|
| Commensal Static | `CS` |
| Commensal HOBIC | `CH` |
| Dysbiotic Static | `DS` |
| Dysbiotic HOBIC | `DH` ← target |

---

## Best Run Summary (2026-02-08, 1000 particles)

| Condition | Total MAP RMSE |
|-----------|:--------------:|
| Commensal Static | 0.0547 |
| Commensal HOBIC | 0.0632 |
| Dysbiotic Static | 0.0538 |
| Dysbiotic HOBIC | 0.0746 |

→ See [Results Gallery](Results-Gallery) for figures.
