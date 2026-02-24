# Contributing to Tmcmc202601

Thank you for considering contributing to this project.

## Getting Started

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes with a descriptive message
4. Push to your fork: `git push origin feature/your-feature`
5. Open a Pull Request against `master`

## Development Setup

### TMCMC (Parameter Estimation)

```bash
# System Python with numpy, scipy, matplotlib
pip install numpy scipy matplotlib
```

### JAX-FEM (Reaction-Diffusion / Multiscale)

```bash
conda create -n klempt_fem python=3.11
conda activate klempt_fem
pip install jax[cpu]==0.9.0.1 jax-fem==0.0.11 basix matplotlib
```

### Abaqus (3D FEM)

Abaqus 2023 is required for the INP-based stress analysis pipeline. Access is typically provided through institutional HPC.

## Code Style

- Python: PEP 8 with 100-char line width
- Docstrings: Google style (brief; implementation comments in Japanese OK)
- Variable naming: follow existing conventions (e.g., `phi_i`, `a_ij`, `theta_MAP`)

## Testing

```bash
# Smoke test (CI)
python -m py_compile data_5species/core/tmcmc.py
python -m py_compile data_5species/core/evaluator.py
python -c "from data_5species.core.nishioka_model import INTERACTION_GRAPH_JSON; print('OK')"
```

## Commit Messages

- Use imperative mood: "Add feature" not "Added feature"
- Keep first line under 72 characters
- Reference issues where applicable: `Fix #42`

## Pull Requests

- Keep PRs focused on a single change
- Include a brief description of what and why
- Ensure CI passes before requesting review

## Reporting Issues

Use the [issue tracker](https://github.com/keisuke58/Tmcmc202601/issues) with the provided templates.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
