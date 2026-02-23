# Tmcmc202601 Wiki

Welcome to the **Tmcmc202601** project wiki — a 5-species oral biofilm Bayesian inference + FEM stress analysis pipeline.

---

## Pages

| Page | Description |
|------|-------------|
| [Installation](Installation) | Environment setup (TMCMC + JAX-FEM conda) |
| [TMCMC Guide](TMCMC-Guide) | Running estimations, key options, interpreting output |
| [FEM Pipeline](FEM-Pipeline) | From TMCMC posterior to Abaqus stress analysis |
| [Parameter Reference](Parameter-Reference) | All 20 θ parameters with biological meaning |
| [Results Gallery](Results-Gallery) | Key figures from best runs |
| [Troubleshooting](Troubleshooting) | Common errors and fixes |

---

## Quick Overview

```
In vitro data (4 conditions × 5 species)
        │
        ▼
  TMCMC Bayesian Inference
  Hamilton ODE · 20 parameters
  → θ_MAP · posterior samples
        │
        ▼
  3D FEM Stress Analysis
  DI(x) → E(x) → Abaqus
  → S_Mises · U_max
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
