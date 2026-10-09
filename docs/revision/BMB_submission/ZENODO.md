> 2026-10-09: 投稿先が Extreme Mechanics Letters に変わったので、題名と版名を EML 用にした。

# Zenodo（投稿前に新しい版の DOI を作る）

- 既存: `10.5281/zenodo.18790007`（v2.0.0-paper、3 月の版）。**投稿用は新しい版（v3.0.0-eml）として追加**し、原稿の `\TBD{DOI}` にその DOI を入れる。
- GitHub release 連携だと `experiment_data/` が含まれる（履歴に残っている）ので、**手動アップロード**にする。

## 入れるもの（ult が全部そろってから、GPU 側で tar にする）
- `data_5species/main/_runs/paper_gateoff/` のうち原稿に使った run（pilot・p1・p2・ult・a45zero・ident・lam1）: `config.json`・`run_record.json`・`samples.npy`・`logL.npy`・MAP。約 20〜30 MB
- `tools/`（check_paper_runs・make_paper_figures・make_supplementary_tables・bayes_factor_a45・a33_mode_weight・compare_ph_runs）
- `data_5species/main/paper_gateoff_job.sh`・`estimate_paper_jax.py`・`hamilton_ode_jax_paper.py`・`core/`
- `docs/revision/generated/`（paper_numbers.json・tables.tex・supp_*.tex）と `docs/handoff/`（判定の記録）
- README（下）・LICENSE（MIT）・CITATION.cff

## 入れないもの
- `data_5species/experiment_data/`（Heine のデータ。論文の引用のみ）
- FEM・deeponet・gnn・nife（別プロジェクト）

## Zenodo の説明文（下書き）
Title: Code, run configurations, convergence records and posterior samples for "[EML の最終題名]" (Extreme Mechanics Letters, VSI: AI & CM)
Authors: 原稿と同じ 10 名・同じ順（ORCID: Nishioka 0009-0001-9360-1336）
Description: GPU-accelerated TMCMC inference of the 5×5 interaction matrix of a Hamilton-principle biofilm model from the in-vitro data of Heine et al. (2025). Contains the JAX forward model and estimator, PBS job scripts, per-run configurations, convergence checks (seed agreement, moves per parameter, posterior mass at the box bounds), posterior samples of every stage used in the manuscript, and the scripts that produce all figures and tables. The experimental data are not redistributed; see Heine et al. (2025), Front. Oral Health.
License: MIT. Related identifier: the GitHub repository (isSupplementTo の原稿 DOI は受理後に追加)。

## 直したこと
- CITATION.cff の Klempt の名前が Henrike / C. になっていた → Felix / F.（原稿と一致）。
