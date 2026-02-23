# Contributing to Tmcmc202601

Thank you for your interest in contributing!

## How to Contribute

### Bug Reports & Feature Requests

Please open an [Issue](https://github.com/keisuke58/Tmcmc202601/issues) with:
- A clear description of the problem or feature
- Steps to reproduce (for bugs)
- Relevant run configuration (n-particles, n-stages, condition name)

### Pull Requests

1. Fork the repository and create a feature branch from `master`
2. Make your changes in `data_5species/core/` or `FEM/`
3. Verify that the CI smoke test passes:
   ```bash
   cd data_5species
   python -m py_compile core/nishioka_model.py core/tmcmc.py core/evaluator.py core/mcmc.py
   ```
4. Open a Pull Request with a brief description of what changed and why

### Code Style

- Python 3.11 compatible
- NumPy-style docstrings for public functions
- Keep ODE parameters and prior bounds in `model_config/prior_bounds.json` (do not hard-code)

### Data & Results

- Do **not** commit large binary output files (`.csv` posterior samples, `.vtu` meshes)
- Add entries to `.gitignore` for new output directories
- Run outputs go into `data_5species/_runs/<timestamp>/`

## Contact

For questions about the science, open a [Discussion](https://github.com/keisuke58/Tmcmc202601/discussions).
