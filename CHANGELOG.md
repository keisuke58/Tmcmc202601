# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.3.0] - 2026-02-24

### Added
- Multiscale coupling pipeline (0D ODE → 1D/2D PDE → 3D FEM)
- Hybrid DI × spatial α approach for condition-specific eigenstrain
- 3 E-model comparison (DI, φ_Pg, Virulence) with quantitative analysis
- Abaqus INP generator with thermal eigenstrain analogy (12 files: 3 models × 4 conditions)
- Species competition 6-panel analysis figure
- Pipeline summary 9-panel figure
- JAX adjoint inverse problem PoC (Lotka-Volterra + Hill gate)
- Klempt 2024 quantitative benchmark
- Posterior uncertainty propagation (TMCMC → DI → σ 90% CI)
- Paper figures generator (Fig. 8–15)
- a₃₅ sensitivity sweep with 51-point evaluation
- θ variant → FEM stress comparison (mild-weight vs dh-old)
- 2D reaction-diffusion extension (`multiscale_coupling_2d.py`)

### Changed
- README: complete rewrite with academic title, Japanese summary, methodology, fixed images
- README: added Limitations, Future Work, Data Preprocessing sections

### Fixed
- Mermaid rendering on GitHub (replaced `$...$` LaTeX with Unicode in node labels)
- 4 broken image paths in README

## [0.2.0] - 2026-02-18

### Added
- Mild-weight prior bounds: a₃₅ [0, 5], a₃₅ [0, 5]
- Likelihood weighting: λ_Pg = 2.0, λ_late = 1.5
- Controlled baseline comparison experiment

### Changed
- Default prior bounds for bridge organism interactions
- Pg RMSE improved from 0.435 to 0.103 (76% reduction)
- Total RMSE improved from 0.223 to 0.156 (30% reduction)

## [0.1.0] - 2026-02-08

### Added
- 5-species Hamilton ODE model with Hill gate (K=0.05, n=4)
- TMCMC Bayesian inference engine (sequential tempering)
- 4-condition estimation: Commensal/Dysbiotic × Static/HOBIC
- Production run: 1000 particles, ~90 h
- JAX-FEM Klempt 2024 nutrient diffusion demo
- 3D tooth FEM pipeline (Open-Full-Jaw Patient 1)
- DI → E(x) stiffness mapping
- Biofilm/substrate analysis modes
- CI workflow (py_compile + import test)

[Unreleased]: https://github.com/keisuke58/Tmcmc202601/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/keisuke58/Tmcmc202601/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/keisuke58/Tmcmc202601/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/keisuke58/Tmcmc202601/releases/tag/v0.1.0
