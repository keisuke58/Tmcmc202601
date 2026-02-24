# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.3.x   | Yes       |
| < 0.3   | No        |

## Reporting a Vulnerability

This is a research software project. If you discover a security vulnerability, please report it by opening a [GitHub Issue](https://github.com/keisuke58/Tmcmc202601/issues) with the label `security`.

For sensitive issues, please contact the author directly:
- Keisuke Nishioka (via GitHub: [@keisuke58](https://github.com/keisuke58))

## Scope

This project processes scientific data (CFU measurements, FEM mesh files) and does not handle user authentication, financial data, or PII. Security considerations are primarily related to:

- Integrity of computational results
- Safe handling of file paths in INP generators
- Dependency supply chain (numpy, scipy, jax, jax-fem)
