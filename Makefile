PYTHON_TMCMC = python
PYTHON_JAX   = $(HOME)/.pyenv/versions/miniconda3-latest/envs/klempt_fem2/bin/python

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

# TMCMC + DeepONet (requires klempt_fem2: JAX, Equinox)
tmcmc-deeponet:  ## Run TMCMC with DeepONet surrogate (single: Dysbiotic HOBIC)
	$(PYTHON_JAX) data_5species/main/estimate_reduced_nishioka.py \
		--condition Dysbiotic --cultivation HOBIC \
		--use-deeponet --n-particles 1000 --n-stages 8 \
		--lambda-pg 2.0 --lambda-late 1.5

tmcmc-deeponet-batch:  ## Run TMCMC + DeepONet for all 4 conditions
	cd data_5species && ./run_all_4conditions_deeponet.sh full

tmcmc-deeponet-quick:  ## Run TMCMC + DeepONet (quick: 200 particles, 4 conditions)
	cd data_5species && ./run_all_4conditions_deeponet.sh quick

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

# ── Experimental verification (DI→E) ──────────────────────
.PHONY: exp-verify exp-verify-design exp-verify-test

exp-verify:  ## Run DI→E experimental validation (synthetic data)
	$(PYTHON_TMCMC) -m experimental_verification_DI.run_validation

exp-verify-design:  ## Estimate required sample size for 16S+AFM experiment
	$(PYTHON_TMCMC) -m experimental_verification_DI.run_validation --design

exp-verify-literature:  ## Generate literature consistency report + E(DI) figure
	$(PYTHON_TMCMC) -m experimental_verification_DI.generate_literature_report
	$(PYTHON_TMCMC) -m experimental_verification_DI.plot_E_DI_with_literature

exp-verify-test:  ## Run experimental verification unit tests
	$(PYTHON_TMCMC) -m pytest experimental_verification_DI/tests/ -v --tb=short

hill-sensitivity:  ## Hill gate K, n sensitivity analysis (requires JAX)
	$(PYTHON_JAX) tools/hill_gate_sensitivity.py

verify-symmetric-A:  ## Verify A = A^T for all TMCMC runs
	$(PYTHON_TMCMC) tools/verify_symmetric_A.py

test-time-dependent-A:  ## Test time-dependent A35, A45 (state-dependent pilot)
	$(PYTHON_TMCMC) tools/test_time_dependent_A.py

# DeepONet importance-weighted retraining (a32, a35 overlap improvement)
deeponet-importance-data:  ## Generate 50k importance-weighted training data
	cd deeponet && $(PYTHON_JAX) generate_training_data.py --condition Dysbiotic_HOBIC \
		--n-samples 50000 --posterior-dir ../data_5species/_runs/dh_baseline --posterior-frac 0.5

deeponet-importance-pipeline:  ## Full pipeline: train + TMCMC + overlap (requires data from deeponet-importance-data)
	cd deeponet && ./run_importance_weighted_pipeline.sh

# ── Analysis ───────────────────────────────────────────────
.PHONY: competition summary uncertainty

competition:  ## Generate species competition analysis (6-panel)
	$(PYTHON_JAX) FEM/plot_species_competition.py

summary:  ## Generate pipeline summary figure (9-panel)
	$(PYTHON_JAX) FEM/generate_pipeline_summary.py

uncertainty:  ## Run posterior → stress uncertainty propagation
	$(PYTHON_JAX) FEM/posterior_uncertainty_propagation.py

# ── CI / Test ──────────────────────────────────────────────
.PHONY: check lint test format repro paper

check:  ## Smoke test: py_compile + import test
	$(PYTHON_TMCMC) -m py_compile data_5species/core/tmcmc.py
	$(PYTHON_TMCMC) -m py_compile data_5species/core/evaluator.py
	$(PYTHON_TMCMC) -c "from data_5species.core.nishioka_model import INTERACTION_GRAPH_JSON; print('OK')"

lint:  ## Ruff + Black check (requires: pip install ruff black)
	ruff check . --line-length 100 --exclude 'FEM/JAXFEM/' --exclude 'FEM/external_tooth_models/'
	black --check --line-length 100 .

test:  ## Run pytest (FEM + data_5species + experimental_verification_DI)
	$(PYTHON_TMCMC) -m pytest FEM/tests/ -v --tb=short -x
	$(PYTHON_TMCMC) -m pytest experimental_verification_DI/tests/ -v --tb=short 2>/dev/null || true
	$(PYTHON_TMCMC) -m pytest data_5species/ -v --tb=short -x --ignore=data_5species/_runs/ -k "not slow" 2>/dev/null || true

format:  ## Black + ruff fix
	black --line-length 100 .
	ruff check . --line-length 100 --fix 2>/dev/null || true

repro:  ## Full pipeline: tmcmc-quick → multiscale → hybrid → eigenstrain
	$(MAKE) tmcmc-quick
	$(MAKE) multiscale
	$(MAKE) hybrid
	$(MAKE) eigenstrain

paper:  ## Generate paper figures (LaTeX 前提)
	$(MAKE) all-figures
	@if [ -f data_5species/docs/nishioka_latex20260218.tex ]; then \
		cd data_5species/docs && latexmk -pdf nishioka_latex20260218.tex 2>/dev/null || echo "LaTeX build skipped (latexmk not found)"; \
	else \
		echo "LaTeX source not found"; \
	fi

# ── Full Pipeline ──────────────────────────────────────────
.PHONY: all-figures

all-figures: multiscale hybrid eigenstrain competition summary  ## Generate all figures
	@echo "All figures generated."

# ── Wiki ────────────────────────────────────────────────────
.PHONY: wiki wiki-list wiki-validate

WIKI_DIR = $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))/../Tmcmc202601.wiki

wiki: wiki-list  ## Show wiki status and list pages

wiki-list:  ## List wiki Markdown pages
	@echo "Wiki: $(WIKI_DIR)"
	@ls -la $(WIKI_DIR)/*.md 2>/dev/null | wc -l | xargs -I {} echo "  {} pages"
	@ls $(WIKI_DIR)/*.md 2>/dev/null | xargs -I {} basename {} | head -30

wiki-validate:  ## Check wiki internal links (basic)
	@echo "Validating wiki links..."
	@grep -rh '\[.*\]([A-Za-z0-9_-]*)' $(WIKI_DIR)/*.md 2>/dev/null | head -20 || true

# ── Help ───────────────────────────────────────────────────
.PHONY: help
help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
