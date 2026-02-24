PYTHON_TMCMC = python
PYTHON_JAX   = $(HOME)/.pyenv/versions/miniconda3-latest/envs/klempt_fem/bin/python

# ── TMCMC ──────────────────────────────────────────────────
.PHONY: tmcmc tmcmc-quick

tmcmc:  ## Run TMCMC estimation (production: 1000 particles)
	$(PYTHON_TMCMC) data_5species/main/estimate_reduced_nishioka.py \
		--n-particles 1000 --n-stages 8 \
		--lambda-pg 2.0 --lambda-late 1.5

tmcmc-quick:  ## Run TMCMC estimation (quick: 150 particles)
	$(PYTHON_TMCMC) data_5species/main/estimate_reduced_nishioka.py \
		--n-particles 150 --n-stages 8 \
		--lambda-pg 2.0 --lambda-late 1.5

# ── Multiscale ─────────────────────────────────────────────
.PHONY: multiscale hybrid eigenstrain

multiscale:  ## Run 0D+1D multiscale pipeline
	$(PYTHON_JAX) FEM/multiscale_coupling_1d.py

hybrid:  ## Generate hybrid DI × spatial α CSV
	$(PYTHON_JAX) FEM/generate_hybrid_macro_csv.py

eigenstrain:  ## Generate Abaqus INP with thermal eigenstrain
	$(PYTHON_JAX) FEM/generate_abaqus_eigenstrain.py

# ── JAX-FEM ────────────────────────────────────────────────
.PHONY: klempt-demo

klempt-demo:  ## Run Klempt 2024 reaction-diffusion demo
	$(PYTHON_JAX) FEM/jax_fem_reaction_diffusion_demo.py

# ── Analysis ───────────────────────────────────────────────
.PHONY: competition summary uncertainty

competition:  ## Generate species competition analysis (6-panel)
	$(PYTHON_JAX) FEM/plot_species_competition.py

summary:  ## Generate pipeline summary figure (9-panel)
	$(PYTHON_JAX) FEM/generate_pipeline_summary.py

uncertainty:  ## Run posterior → stress uncertainty propagation
	$(PYTHON_JAX) FEM/posterior_uncertainty_propagation.py

# ── CI / Test ──────────────────────────────────────────────
.PHONY: check lint

check:  ## Smoke test: py_compile + import test
	$(PYTHON_TMCMC) -m py_compile data_5species/core/tmcmc.py
	$(PYTHON_TMCMC) -m py_compile data_5species/core/evaluator.py
	$(PYTHON_TMCMC) -c "from data_5species.core.nishioka_model import INTERACTION_GRAPH_JSON; print('OK')"

lint:  ## Run basic syntax checks on all Python files
	$(PYTHON_TMCMC) -m py_compile data_5species/core/tmcmc.py
	$(PYTHON_TMCMC) -m py_compile data_5species/core/evaluator.py
	$(PYTHON_TMCMC) -m py_compile data_5species/core/nishioka_model.py

# ── Full Pipeline ──────────────────────────────────────────
.PHONY: all-figures

all-figures: multiscale hybrid eigenstrain competition summary  ## Generate all figures
	@echo "All figures generated."

# ── Help ───────────────────────────────────────────────────
.PHONY: help
help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
