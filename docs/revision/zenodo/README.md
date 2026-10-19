# Identifiability of species interactions in a living material — code, run records and posterior samples

Archive accompanying the article

> K. Nishioka, F. Klempt, H. Geisler, R. Mukherjee, N. Heine, K. Doll-Nikutta, M. Stiesch, S. P. Szafrański, M. Soleimani, P. Junker:
> *Identifiability of species interactions in a living material: Bayesian calibration of an extended Hamilton principle multiphase model.*
> Extreme Mechanics Letters (submitted 2026; the article DOI will be added in a later version of this record).

Archive DOI: **10.5281/zenodo.23320423** (reserved; registered when the record is published) · Version 1.0.0 · License: MIT (see `LICENSE`)

---

## 1. What the archive contains

| Component | Path (inside the archive) | Content |
|---|---|---|
| Inference code | `data_5species/main/estimate_paper_jax.py` | Entry point: TMCMC calibration of the 15 interaction coefficients (20-dim θ, 5 b-entries fixed at 0) for one condition and one stage |
| | `data_5species/main/hamilton_ode_jax_paper.py` | Forward model (extended-Hamilton-principle multiphase model, implicit solver in JAX, `vmap` over particles) |
| | `data_5species/main/tmcmc_engine_paper.py` | TMCMC engine (adaptive tempering, random-walk + DE-MC mutation, evidence) |
| | `data_5species/main/estimate_reduced_nishioka.py`, `data_5species/{core,utils,debug,visualization}/`, `tmcmc/program2602/` | Data loaders and helper modules imported by the entry point (import closure traced at build time) |
| | `data_5species/model_config/prior_bounds_paper_20d.json` | Prior box of the paper (before the per-condition overrides recorded in each run) |
| Job script | `data_5species/main/paper_gateoff_job.sh` | PBS job that defines the four stages (exact CLI arguments per stage) |
| | `tools/gate_submit_ult.sh` | Checks a finished stage (iii) group with `check_paper_runs.py` and submits stage (iv) only if it passes |
| Experimental data | — | **Not included** (Heine et al. 2025); see §5 for how to obtain them |
| Runs | `docs/revision/generated/final_runs/<COND>_<stage>_<tag>_seed{7,42,123}/` | Every run shown in the paper: `samples.npy` (final particles, N_p × 20), `logL.npy` (log-likelihood per particle), `theta_MAP.json`, `run_record.json` (effective arguments, prior box, initial state, tempering history, acceptance, moves per particle, ln Z, log-likelihood recheck), `config.json` (superset of run_record) |
| Final MAPs | `docs/revision/generated/final_theta_MAP/` | `CS.json`, `CH.json`, `DS.json` (stage (iv)), `DH.json` (stage (iii), see below), best seed after polishing where available; `phi_init.json` (initial state actually used); README with provenance |
| Figure/table scripts | `tools/*.py`, `docs/generate_fig2_phi_version.py`, `docs/generate_fig_phase_scatter.py` | See §3 |
| Manuscript outputs | `docs/revision/EML_submission/figures/raw/` | Figures (PDF), intermediate JSON and `supp/*.tex` tables exactly as used in the submitted manuscript, plus `spec_raw.json` (which run group is which stage; key `final` where the final stage is not stage (iv)) |
| Recovery test | `docs/revision/generated/recovery/<COND>_syn_n1_r0/synthetic.json`, `<COND>_syn_n1_r0_seed*` | Synthetic data generated from a posterior point of each condition (true θ included) and the three recovery runs (Supplementary Section S7) |
| Held-out day | `final_runs/CH_{pilot,p1,p2}_2c_raw_a33w_ho{10,15}_seed42/` | CH with day 10 or 15 removed from the likelihood (Supplementary Section S17); the evaluation needs the measured data of the removed day (§5) |
| Metadata | `README.md`, `CITATION.cff`, `.zenodo.json`, `MANIFEST.txt`, `SHA256SUMS` | `SHA256SUMS` covers every file; check with `sha256sum -c SHA256SUMS` |

The archive is built from an explicit list (`MANIFEST.txt`) by `docs/revision/zenodo/build_archive.sh`; nothing else from the development repository is included.
All paths keep the repository layout, so every command below runs from the unpacked archive root.

### Conditions and stages

Conditions (Heine et al. 2025): **CS** commensal/static, **CH** commensal/HOBIC (flow reactor), **DS** dysbiotic/static, **DH** dysbiotic/HOBIC.

| Paper | Run name | Likelihood | Prior box | Particles (typ.) |
|---|---|---|---|---|
| stage (i) | `pilot` | species composition (viability ψ fixed to the measured membrane integrity) | paper box (+ per-condition overrides) | 1000 |
| stage (ii) | `p1` | as (i) | stage-(i) posterior mean ± 4 SD | 2000–3000 |
| stage (iii) | `p2` | composition + viability (multichannel; pH weight λ_ch5 = 0) | back to the full box, warm start from (ii) | 2000–18000 |
| stage (iv) | `ult` | as (iii) | stage-(iii) posterior mean ± 4 SD | 5000 |

**Final stage**: stage (iv) for CS, CH and DS; **stage (iii) for DH** (stage (iv) of DH fails criterion (e), see the article; its runs are archived as `DH_ult_*`).
All reported runs use the measured replicate values (`*_2c_raw*`, two-channel likelihood: viable composition, total-count composition and live fraction).

`*_a45zero_*` runs are stage (i) with a₄₅ = 0, used only for the Bayes factor (Supplementary S3).
The exact arguments of every run are in its `run_record.json` (`argv`, `args`, `prior_bounds_final`, `phi_init`); the per-condition box overrides are in `args.override_bounds`.
Note: the stage-(iii) runs of CS and CH that served as warm start for their stage-(iv) runs are not part of the paper's figures and are not archived; their path is recorded in `args.init_from_dir`.

### Acceptance criteria (main text, Sect. 2; Supplementary S2/S4)

Every stage is run with three seeds (7, 42, 123) and accepted only if (checked by `tools/check_paper_runs.py`):

- (a) the stored log-likelihood of every particle agrees with a recomputation (`run_record.json: logL_consistent`);
- (b) tempering reaches β = 1;
- (c) each particle moves ≥ 2 times per tempering step and ≥ 5 times per free parameter over the run;
- (d) max log-likelihood differs by ≤ 1 across seeds;
- (e) every posterior median differs across seeds by ≤ 0.5 pooled posterior SD. Entries whose pooled posterior SD is ≥ 0.8 × the SD of a uniform distribution on the box are reported as *not identified* (n.i., or s.o. = sign only if the 90 % interval excludes 0) and excluded from (e); multimodal entries are checked per mode.

Box rule: if > 20 % of the posterior mass of an entry lies in the outer 5 % of the box on one side, the bound is widened and the stage repeated.
Stage-(i) runs used only for Bayes factors need not pass (c) and (e) (see `final_runs/README.md` for the verdict of each group).

## 2. Software and hardware

- Python ≥ 3.10; JAX in **double precision** (`jax.config.update("jax_enable_x64", True)` is set by the code; do not disable it).
- Package versions verified for this archive (CPU, Python 3.11.15, 2026-10-12): see `docs/revision/zenodo/requirements.txt`
  (jax/jaxlib 0.10.2, numpy 2.4.6, scipy 1.17.1, pandas 3.0.6, matplotlib 3.11.2; optional numba, umap-learn).
  `pip install -r docs/revision/zenodo/requirements.txt`
- GPU environment that produced the runs (conda env `klempt_fem2` on the IKM cluster, checked 2026-10-20): Python 3.10.19, **jax / jaxlib 0.6.2** (`jax-cuda12-plugin`, `jax-cuda12-pjrt` 0.6.2), numpy 2.2.6, scipy 1.15.3;
  CUDA from pip wheels (`nvidia-cuda-runtime-cu12` 12.9.79, `nvidia-cudnn-cu12` 9.19.0.56). GPUs: NVIDIA GeForce RTX 3090 (driver 550.100) and RTX 4090 (driver 550.90.07).
- Hardware: every inference run used one NVIDIA GPU (RTX 3090 or RTX 4090; paper timing: RTX 4090; cluster nodes with CUDA 12, `jax[cuda12]`). Wall-clock time per seed is in `run_record.json: total_time_s` and Supplementary Table S2/S3 (minutes for stage (i) to ~9 h for stage (iv)). The figure and table scripts run on a CPU (`--device cpu` is the default for all tools); the posterior-predictive figures take minutes to tens of minutes on a CPU.
- Figures use LaTeX text rendering when a LaTeX installation is available and fall back to matplotlib's renderer otherwise.
- Each `run_record.json` records `git_head` of the development repository at run time. The archived inference code is the state at archive build; changes after the earliest run only add options with backward-compatible defaults (e.g. `--ic-day`, default 1).

## 3. Reproducing the figures and tables

Run from the archive root. Almost all scripts read the archived runs only; those that compare with the data
(fit metrics, Fig. 2 data points, pH check, held-out day) need the experimental data in place (§5).
One script regenerates every figure, table and number of the article and the supplement:

```bash
R=docs/revision/generated/final_runs  F=docs/revision/EML_submission/figures/raw  bash tools/rebuild_eml_raw.sh
#   SPEC defaults to $F/spec_raw.json; SKIP="ph knockout" skips steps (space separated)
```

It runs, in order: the acceptance criteria (a)-(e) (`check_paper_runs.py`), per-run metrics (`eval_paper_runs.py`, `fit_metrics_2c.py`),
Fig. 2 (`make_paper_figures.py` -> `paper_fig2_v2.pdf`), the knock-out (`knockout_fn.py`), the pH check (`ph_check.py`),
Figs. 3-5 (`make_eml_figs56.py` -> `fig3_ident_matrix.pdf`, `fig4_posteriors_v2.pdf`, `fig6_checks_v2.pdf`), the supplementary tables
(`make_supplementary_tables.py`, `make_cost_table.py`, `ident_threshold_table.py`, `rhat_seeds.py`, `fisher_ident.py`) and the supplementary
figures (`plot_seed_maps.py`, `growth_stress_check.py`).

Supplementary checks:

```bash
# Recovery test (Supplementary Section S7, Table S8)
python3 tools/recovery_evaluate.py \
  --case CS=docs/revision/generated/recovery/CS_syn_n1_r0/synthetic.json='docs/revision/generated/recovery/CS_syn_n1_r0_seed*' \
  --case CH=docs/revision/generated/recovery/CH_syn_n1_r0/synthetic.json='docs/revision/generated/recovery/CH_syn_n1_r0_seed*' \
  --case DS=docs/revision/generated/recovery/DS_syn_n1_r0/synthetic.json='docs/revision/generated/recovery/DS_syn_n1_r0_seed*' \
  --case DH=docs/revision/generated/recovery/DH_syn_n1_r0/synthetic.json='docs/revision/generated/recovery/DH_syn_n1_r0_seed*' \
  --out $F/recovery
# Held-out day in CH (Supplementary Section S17, Table S15; needs the measured data, §5)
python3 tools/holdout_eval.py $R --holdout 'CH_p2_2c_raw_a33w_ho10_seed*' --full 'CH_p2_2c_raw_a33w_seed*' --day 10
python3 tools/holdout_eval.py $R --holdout 'CH_p2_2c_raw_a33w_ho15_seed*' --full 'CH_p2_2c_raw_a33w_seed*' --day 15
```

Fig. 1 is a schematic (TikZ source: `docs/revision/EML_submission/figures/fig1_pipeline_v2.tex`).
Outputs written by the commands above can be compared with the archived versions in `$F` (the commands overwrite them; copy `$F` first if you want a diff).

### Re-running an inference stage

The estimator is normally driven by `data_5species/main/paper_gateoff_job.sh` (PBS; the script `cd`s into `$HOME/Tmcmc202601/data_5species/main` and uses a fixed Python path — adapt these two lines). The equivalent direct call, e.g. stage (iv) of CH seed 7, is the `argv` stored in its run record:

```bash
cd data_5species/main
python3 -c "import json;print(' '.join(json.load(open('../../docs/revision/generated/final_runs/CH_ult_2c_raw_a33w_seed7/run_record.json'))['argv']))"
# python3 estimate_paper_jax.py --condition Commensal --cultivation HOBIC --K-hill 0.0 --n-hill 4.0 --use-exp-init \
#   --use-de-mc --mutation rw --seed 7 --device gpu ... --init-from-dir <stage (iii) run> --posterior-prior-nsigma 4 \
#   --override-bounds ... --output-dir _runs/paper_gateoff/<name>
```

Stage chain per condition and seed (job script; `PREV` = output of the previous stage, `OVERRIDE` = `args.override_bounds` of the run):

```bash
qsub -v STAGE=pilot,TAG=DH,SEED=42                     paper_gateoff_job.sh
qsub -v STAGE=p1,TAG=DH,SEED=42,PREV=<pilot dir>       paper_gateoff_job.sh
qsub -v STAGE=p2,TAG=DH,SEED=42,PREV=<p1 dir>          paper_gateoff_job.sh
qsub -v STAGE=ult,TAG=DH,SEED=42,PREV=<p2 dir>         paper_gateoff_job.sh
```

TMCMC is stochastic and GPU reductions are not bit-reproducible across hardware/driver versions; reruns with the same seed reproduce the posteriors within the seed-to-seed spread reported in Supplementary Table S2, not bit for bit.

## 4. Loading the samples

```python
import json, numpy as np
d = "docs/revision/generated/final_runs/CH_ult_2c_raw_a33w_seed7"
S = np.load(f"{d}/samples.npy")        # (N_p, 20) final particles (equally weighted after resampling)
L = np.load(f"{d}/logL.npy")           # (N_p,) log-likelihood
rec = json.load(open(f"{d}/run_record.json"))
names = rec["theta_names"]             # a11 a12 a22 b1 b2 a33 a34 a44 b3 b4 a13 a14 a23 a24 a55 b5 a15 a25 a35 a45
free = rec["free_dims"]                # the 15 interaction entries (b* fixed at 0)
```

## 5. Experimental data

The calibration data (species fractions at days 1, 3, 6, 10, 15, 21, total biofilm volume, membrane integrity / viability, and pH) are from

> N. Heine, K. Bittroff, S. P. Szafrański, M. Duitscher, W. Behrens, C. Vollmer, C. Mikolai, N. Kommerein, N. Debener, K. Frings, A. Heisterkamp, T. Scheper, M. L. Torres-Mapa, J. Bahnemann, M. Stiesch, K. Doll-Nikutta.
> Influence of species composition and cultivation condition on peri-implant biofilm dysbiosis in vitro. *Frontiers in Oral Health* 6 (2025) 1649419. <https://doi.org/10.3389/froh.2025.1649419>

**The experimental data are not redistributed in this archive.** The calibration uses the measured replicate values behind Figs. 3A and S1 of Heine et al. (2025) (viable-cell and total-count species fractions, three to five biological replicates per time point) and the live fraction from LIVE/DEAD staining (images per biofilm), which the authors provided for this work. They are available from the authors on request (Hannover Medical School, Department of Prosthetic Dentistry and Biomedical Materials Science / NIFE). Users of these data must cite that article.

To re-run the inference with the measured data (`--data-source raw`, the runs `*_2c_raw*` reported in the paper), convert the files provided by the authors with `tools/convert_raw_heine.py` and `tools/build_raw_dataset.py`; they write the tables that the estimator reads into `data_5species/raw_dataset/experiment_data/` (`estimate_paper_jax.DATA_DIRS`). The older runs on values read off the published box plots (`--data-source digitised`) read `data_5species/experiment_data/` instead (file names as expected by the code: `boxplot_{Commensal,Dysbiotic}_{Static,HOBIC}.csv`, `fig2_membrane_distribution.csv`, `fig3_species_distribution_{replicates,summary}.csv`, `fig4A_pH_timeseries.csv`, `figS1_total_species_distribution_{replicates,summary}.csv`).
The posterior samples, run records and figures in this archive can be inspected without the data; the run records contain only derived quantities used by the likelihood (initial composition `phi_init`, per-species noise levels, and in stages (i)--(ii) the fixed viabilities `psi_fixed` implied by the data), not the replicate values.

## 6. License and citation

Code and run outputs: MIT License (`LICENSE`, © 2026 Keisuke Nishioka). Please cite the article and this archive (`CITATION.cff`); for the experimental data cite Heine et al. (2025).
Funded by the European Union (ERC, Gen-TSM, project number 101124463). Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council Executive Agency. Neither the European Union nor the granting authority can be held responsible for them.

## 7. Publishing checklist (for the authors)

1. On Zenodo: *New upload* → *Reserve DOI*; put the DOI into `README.md`, `CITATION.cff` (`doi:`), the manuscript (Data and code availability), and `.zenodo.json` if uploading via API.
2. Fill in the GPU software versions (§2). The article DOI is not in `.zenodo.json` yet (Zenodo rejects an invalid DOI); add it as `isSupplementTo` in a new version after acceptance.
3. `bash docs/revision/zenodo/build_archive.sh` → upload `docs/revision/zenodo/dist/tmcmc_eml_archive_v1.0.0.zip`; copy title, authors, description, keywords, license and funding from `.zenodo.json`.

---

<!-- maintainer notes (remove before building the final archive) -->
## Maintainer checklist: Zenodo draft (status 2026-10-19)

Draft record saved; DOI reserved **10.5281/zenodo.23320423** (in manuscript, cover letter, CITATION.cff, .zenodo.json).
To fix in the web form before publishing (user, later):

1. **Delete `EML_manuscript_nishioka.pdf`** from Files (uploaded by mistake; the record holds only the archive zip).
2. Upload `dist/tmcmc_eml_archive_v1.0.0.zip` (`bash build_archive.sh`; all four conditions are final since 2026-10-19).
3. Title: "Code, run records and posterior samples for: Identifiability of species interactions in a living material: Bayesian calibration of an extended Hamilton principle multiphase model".
4. Resource type: Software (Dataset also acceptable).
5. License: **MIT** (form default is CC BY 4.0).
6. Creators: all 10 authors in manuscript order (Family name / Given names; Nishioka = family name). ORCID for K. Nishioka as `0009-0001-9360-1336` only (current entry gives "No valid scheme recognized").
7. Description, keywords, version 1.0.0, funding (ERC 101124463; DFG TRR 298 / 426335750): copy from `.zenodo.json`.
8. Publication date: the day of publishing. Leave the Thesis/Conference/Journal sections empty.
9. Experimental data: **not included** (decided 2026-10-19, EML_PLAN §24.11; available from Heine et al. on request).
10. Then Publish (irreversible).
