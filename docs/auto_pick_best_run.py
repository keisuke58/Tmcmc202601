#!/usr/bin/env python3
"""
auto_pick_best_run.py

Scan tmcmc/_runs and pick the "best" run by simple, auditable criteria:
- beta_final must reach ~1.0 (posterior reached)
- low RMSE_total (MAP fit) is better
- low ROM error is better (penalize large final ROM error)

Then generate docs under docs/auto_best_run/:
- best_run.json / best_run_summary.md
- auto_best_run_report.tex (article)
- auto_best_run_slides.tex (beamer)

This is intentionally lightweight and dependency-free (stdlib only).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def _last_finite(values: List[Optional[float]]) -> Optional[float]:
    for v in reversed(values):
        if v is not None and math.isfinite(v):
            return float(v)
    return None


def _mean_finite(values: List[Optional[float]]) -> Optional[float]:
    xs = [v for v in values if v is not None and math.isfinite(v)]
    if not xs:
        return None
    return float(sum(xs) / len(xs))


@dataclass
class ModelMetrics:
    model: str
    rmse_total_map: Optional[float]
    mae_total_map: Optional[float]
    beta_final_min_across_chains: Optional[float]
    rom_error_final_max_across_chains: Optional[float]
    accept_rate_mean_min_across_chains: Optional[float]


@dataclass
class RunCandidate:
    run_dir: str
    run_id: str
    score: float
    reasons: List[str]
    models: List[ModelMetrics]
    sigma_obs: Optional[float] = None
    cov_rel: Optional[float] = None
    n_particles: Optional[int] = None
    n_stages: Optional[int] = None
    n_mutation_steps: Optional[int] = None
    mode: Optional[str] = None
    seed: Optional[int] = None


def _collect_model_metrics(run_dir: Path, model: str) -> ModelMetrics:
    fm = _read_json(run_dir / f"fit_metrics_{model}.json") or {}
    fit_map = (fm.get("fit_MAP") or {}) if isinstance(fm.get("fit_MAP"), dict) else {}
    rmse_total = _safe_float(fit_map.get("rmse_total"))
    mae_total = _safe_float(fit_map.get("mae_total"))

    # beta_final: conservative = min over chains of last beta
    beta_rows = _read_csv_rows(run_dir / "diagnostics_tables" / f"{model}_beta_schedule.csv")
    betas_by_chain: Dict[str, List[float]] = {}
    for r in beta_rows:
        chain = (r.get("chain") or "1").strip()
        b = _safe_float(r.get("beta"))
        if b is None:
            continue
        betas_by_chain.setdefault(chain, []).append(float(b))
    beta_final_min = None
    if betas_by_chain:
        finals = []
        for _, bs in betas_by_chain.items():
            finals.append(bs[-1] if bs else float("nan"))
        finals = [v for v in finals if math.isfinite(v)]
        beta_final_min = min(finals) if finals else None

    # ROM error: conservative = max over chains of "final rom_error"
    rom_rows = _read_csv_rows(run_dir / "diagnostics_tables" / f"{model}_rom_error.csv")
    rom_by_chain: Dict[str, List[Optional[float]]] = {}
    for r in rom_rows:
        chain = (r.get("chain") or "1").strip()
        v = _safe_float(r.get("rom_error"))
        rom_by_chain.setdefault(chain, []).append(v)
    rom_final_max = None
    if rom_by_chain:
        finals = []
        for _, xs in rom_by_chain.items():
            finals.append(_last_finite(xs))
        finals = [v for v in finals if v is not None and math.isfinite(v)]
        rom_final_max = max(finals) if finals else None

    # acceptance mean: conservative = min over chains of mean accept_rate
    acc_rows = _read_csv_rows(run_dir / "diagnostics_tables" / f"{model}_acceptance_rate.csv")
    acc_by_chain: Dict[str, List[Optional[float]]] = {}
    for r in acc_rows:
        chain = (r.get("chain") or "1").strip()
        v = _safe_float(r.get("accept_rate"))
        acc_by_chain.setdefault(chain, []).append(v)
    acc_mean_min = None
    if acc_by_chain:
        means = []
        for _, xs in acc_by_chain.items():
            means.append(_mean_finite(xs))
        means = [v for v in means if v is not None and math.isfinite(v)]
        acc_mean_min = min(means) if means else None

    return ModelMetrics(
        model=model,
        rmse_total_map=rmse_total,
        mae_total_map=mae_total,
        beta_final_min_across_chains=beta_final_min,
        rom_error_final_max_across_chains=rom_final_max,
        accept_rate_mean_min_across_chains=acc_mean_min,
    )


def _extract_run_config_summary(run_dir: Path) -> Dict[str, Any]:
    cfg = _read_json(run_dir / "config.json") or {}
    metrics = _read_json(run_dir / "metrics.json") or {}

    exp = cfg.get("experiment") if isinstance(cfg.get("experiment"), dict) else {}
    tmcmc = cfg.get("tmcmc") if isinstance(cfg.get("tmcmc"), dict) else {}
    seeds = cfg.get("seeds") if isinstance(cfg.get("seeds"), dict) else {}

    out: Dict[str, Any] = {}
    out["mode"] = cfg.get("mode") or metrics.get("mode")
    out["seed"] = seeds.get("base_seed") if isinstance(seeds, dict) else None
    out["sigma_obs"] = exp.get("sigma_obs") if isinstance(exp, dict) else None
    out["cov_rel"] = exp.get("cov_rel") if isinstance(exp, dict) else None
    out["n_particles"] = tmcmc.get("n_particles") if isinstance(tmcmc, dict) else None
    out["n_stages"] = tmcmc.get("n_stages") if isinstance(tmcmc, dict) else None
    out["n_mutation_steps"] = tmcmc.get("n_mutation_steps") if isinstance(tmcmc, dict) else None
    return out


def _score_run(
    run_dir: Path,
    models: List[str],
    prefer_model: str,
    require_beta_one: bool,
) -> Tuple[float, List[str], List[ModelMetrics]]:
    reasons: List[str] = []
    model_metrics: List[ModelMetrics] = []

    for m in models:
        mm = _collect_model_metrics(run_dir, m)
        model_metrics.append(mm)

    # If prefer_model is present, score primarily on that; otherwise aggregate all.
    def pick_metrics(name: str) -> Optional[ModelMetrics]:
        for mm in model_metrics:
            if mm.model == name:
                return mm
        return None

    primary = pick_metrics(prefer_model) or (model_metrics[0] if model_metrics else None)
    if primary is None:
        return float("inf"), ["no fit_metrics found"], []

    # Base score: RMSE_total (MAP)
    score = 0.0
    rmse = primary.rmse_total_map
    if rmse is None:
        reasons.append("missing rmse_total (fit_MAP)")
        score += 100.0
    else:
        score += float(rmse)

    # Hard gate / penalty: beta_final ~= 1.0
    beta = primary.beta_final_min_across_chains
    if beta is None:
        reasons.append("missing beta_final")
        score += 200.0
    else:
        if require_beta_one and beta < 0.999:
            reasons.append(f"beta_final {beta:.6g} < 0.999 (posterior not reached)")
            score += 1000.0
        else:
            # small nudge: prefer exactly 1.0
            score += max(0.0, 1.0 - float(beta)) * 10.0

    # ROM error penalty: prefer small; penalize large
    rom = primary.rom_error_final_max_across_chains
    if rom is None:
        reasons.append("missing rom_error_final")
        score += 10.0
    else:
        # gentle penalty up to 0.3; steep beyond
        if rom <= 0.3:
            score += float(rom) * 0.5
        else:
            score += 0.15 + (float(rom) - 0.3) * 5.0
            reasons.append(f"rom_error_final {rom:.3g} (penalized)")

    # Acceptance-rate sanity (not dominating): penalize extremely low
    acc = primary.accept_rate_mean_min_across_chains
    if acc is not None and acc < 0.05:
        score += 50.0
        reasons.append(f"low acceptance mean {acc:.3g}")

    # Tie-breaker: if multiple models exist, add tiny aggregate term (encourage overall fit)
    if len(model_metrics) > 1:
        rmses = [mm.rmse_total_map for mm in model_metrics if mm.rmse_total_map is not None]
        if rmses:
            score += float(sum(rmses) / len(rmses)) * 0.05

    return score, reasons, model_metrics


def _escape_latex(s: str) -> str:
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("#", "\\#")
    )


def _render_tex_report(best: RunCandidate, fig_paths: List[str]) -> str:
    # NOTE: We intentionally do NOT embed images here.
    # On minimal Rocky/RHEL TeX installs, image embedding via (pdf|xe)latex can be
    # extremely slow or hang depending on the graphics toolchain. Instead, we
    # produce a PDF that contains figure *paths* (and the Markdown summary lists
    # them too). Users can open the PNGs directly from the run directory.
    fig_lines = []
    for rel in fig_paths[:20]:
        fig_lines.append(f"\\item \\texttt{{{_escape_latex(rel)}}}")
    figs_tex = (
        "\\begin{itemize}\\itemsep0.2em\\relax\n" + "\n".join(fig_lines) + "\n\\end{itemize}\n"
    )

    model_rows = []
    for mm in best.models:
        model_rows.append(
            " & ".join(
                [
                    _escape_latex(mm.model),
                    f"{mm.rmse_total_map:.4g}" if mm.rmse_total_map is not None else "missing",
                    (
                        f"{mm.beta_final_min_across_chains:.4g}"
                        if mm.beta_final_min_across_chains is not None
                        else "missing"
                    ),
                    (
                        f"{mm.rom_error_final_max_across_chains:.4g}"
                        if mm.rom_error_final_max_across_chains is not None
                        else "missing"
                    ),
                    (
                        f"{mm.accept_rate_mean_min_across_chains:.4g}"
                        if mm.accept_rate_mean_min_across_chains is not None
                        else "missing"
                    ),
                ]
            )
            + " \\\\"
        )

    return f"""\\documentclass[11pt,a4paper]{{article}}
\\usepackage[margin=25mm]{{geometry}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\title{{Auto-picked best TMCMC run summary}}
\\author{{auto\\_pick\\_best\\_run.py}}
\\date{{\\today}}
\\begin{{document}}
\\maketitle

\\section*{{Best run}}
\\begin{{itemize}}
  \\item run\\_id: \\texttt{{{_escape_latex(best.run_id)}}}
  \\item run\\_dir: \\texttt{{{_escape_latex(best.run_dir)}}}
  \\item score: \\texttt{{{best.score:.6g}}}
  \\item mode: \\texttt{{{_escape_latex(str(best.mode)) if best.mode is not None else "missing"}}}
  \\item seed: \\texttt{{{best.seed if best.seed is not None else "missing"}}}
  \\item sigma\\_obs: \\texttt{{{best.sigma_obs if best.sigma_obs is not None else "missing"}}}
  \\item cov\\_rel: \\texttt{{{best.cov_rel if best.cov_rel is not None else "missing"}}}
  \\item n\\_particles / n\\_stages / n\\_mutation\\_steps:
        \\texttt{{{best.n_particles if best.n_particles is not None else "missing"}}} /
        \\texttt{{{best.n_stages if best.n_stages is not None else "missing"}}} /
        \\texttt{{{best.n_mutation_steps if best.n_mutation_steps is not None else "missing"}}}
\\end{{itemize}}

\\section*{{Per-model key metrics}}
\\begin{{tabular}}{{lrrrr}}
\\toprule
Model & RMSE\\_MAP & $\\beta_\\mathrm{{final}}$ & ROM\\_err\\_final & Acc\\_mean \\\\
\\midrule
{chr(10).join(model_rows)}
\\bottomrule
\\end{{tabular}}

\\section*{{Figures (from run/figures)}}
{figs_tex if fig_paths else "No figures found."}

\\end{{document}}
"""


def _render_tex_slides(best: RunCandidate, fig_paths: List[str]) -> str:
    # Keep slides build robust: list figure paths instead of embedding images.
    items = []
    for rel in fig_paths[:12]:
        items.append("\\item \\texttt{" + _escape_latex(rel) + "}")
    frames = (
        [
            "\\begin{frame}{Figures (paths)}\n"
            "\\begin{itemize}\\itemsep0.2em\\relax\n" + "\n".join(items) + "\n\\end{itemize}\n"
            "\\end{frame}\n"
        ]
        if fig_paths
        else []
    )

    return f"""\\documentclass[aspectratio=169]{{beamer}}
\\usetheme{{Madrid}}
\\setbeamertemplate{{navigation symbols}}{{}}
\\usepackage{{graphicx}}
\\title{{Auto-picked best TMCMC run}}
\\subtitle{{run\\_id: {_escape_latex(best.run_id)}}}
\\author{{auto\\_pick\\_best\\_run.py}}
\\date{{\\today}}
\\begin{{document}}
\\begin{{frame}}
\\titlepage
\\end{{frame}}

\\begin{{frame}}{{Run summary}}
\\begin{{itemize}}
  \\item run\\_dir: \\texttt{{{_escape_latex(best.run_dir)}}}
  \\item score: \\texttt{{{best.score:.6g}}}
  \\item mode: \\texttt{{{_escape_latex(str(best.mode)) if best.mode is not None else "missing"}}}
  \\item sigma\\_obs: \\texttt{{{best.sigma_obs if best.sigma_obs is not None else "missing"}}}
  \\item cov\\_rel: \\texttt{{{best.cov_rel if best.cov_rel is not None else "missing"}}}
\\end{{itemize}}
\\end{{frame}}

{chr(10).join(frames) if frames else "% No figures found"}

\\end{{document}}
"""


def _collect_figures(run_dir: Path, models: List[str]) -> List[str]:
    figs_dir = run_dir / "figures"
    if not figs_dir.exists():
        return []

    manifest = _read_json(figs_dir / "FIGURES_MANIFEST.json")
    fig_list: List[str] = []
    if isinstance(manifest, dict) and isinstance(manifest.get("figures"), list):
        for fn in manifest["figures"]:
            if isinstance(fn, str) and fn.lower().endswith(".png"):
                fig_list.append(str((figs_dir / fn).resolve()))

    # Fallback: prefer common key figures by name (stable ordering)
    def add_if_exists(name: str):
        p = figs_dir / name
        if p.exists():
            fig_list.append(str(p.resolve()))

    for m in models:
        add_if_exists(f"posterior_{m}.png")
        add_if_exists(f"TSM_simulation_{m}_MAP_fit_with_data.png")
        add_if_exists(f"TSM_simulation_{m}_MEAN_fit_with_data.png")
        add_if_exists(f"TSM_simulation_{m}_with_data.png")

    # De-dup while preserving order
    seen = set()
    out: List[str] = []
    for p in fig_list:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def _relpath_for_docs(abs_path: str, docs_dir: Path) -> str:
    # Make paths relative to docs/ so LaTeX builds from docs/ without copying images.
    p = Path(abs_path)
    try:
        rel = p.relative_to(docs_dir)
        return str(rel.as_posix())
    except Exception:
        # docs_dir is typically <repo>/docs; run_dir is <repo>/tmcmc/_runs/...
        # so we want something like ../tmcmc/_runs/<run>/figures/...
        return str(Path("..") / p.relative_to(docs_dir.parent))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=str, default=str(Path("tmcmc") / "_runs"))
    ap.add_argument("--prefer-model", type=str, default="M1")
    ap.add_argument("--require-beta-one", action="store_true", default=True)
    ap.add_argument("--no-require-beta-one", dest="require_beta_one", action="store_false")
    ap.add_argument("--out-dir", type=str, default=str(Path("docs") / "auto_best_run"))
    ap.add_argument(
        "--build-pdf", action="store_true", default=True, help="Build PDFs via docs/build_pdfs.py"
    )
    ap.add_argument(
        "--no-build-pdf", dest="build_pdf", action="store_false", help="Skip PDF build step"
    )
    args = ap.parse_args()

    runs_root = Path(args.runs_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    docs_dir = out_dir.parent  # <repo>/docs

    # Discover candidates by config.json (run root marker)
    config_paths = sorted(runs_root.glob("**/config.json"))
    candidates: List[RunCandidate] = []
    for cfg_path in config_paths:
        run_dir = cfg_path.parent
        run_id = run_dir.relative_to(runs_root).as_posix()

        # Which models have fit_metrics?
        fit_files = sorted(run_dir.glob("fit_metrics_*.json"))
        models = [f.stem.replace("fit_metrics_", "") for f in fit_files]
        if not models:
            continue

        score, reasons, per_model = _score_run(
            run_dir=run_dir,
            models=models,
            prefer_model=str(args.prefer_model),
            require_beta_one=bool(args.require_beta_one),
        )
        cfg_sum = _extract_run_config_summary(run_dir)
        candidates.append(
            RunCandidate(
                run_dir=str(run_dir),
                run_id=run_id,
                score=float(score),
                reasons=reasons,
                models=per_model,
                mode=cfg_sum.get("mode"),
                seed=_safe_float(cfg_sum.get("seed")) if cfg_sum.get("seed") is not None else None,
                sigma_obs=_safe_float(cfg_sum.get("sigma_obs")),
                cov_rel=_safe_float(cfg_sum.get("cov_rel")),
                n_particles=(
                    int(cfg_sum["n_particles"])
                    if _safe_float(cfg_sum.get("n_particles")) is not None
                    else None
                ),
                n_stages=(
                    int(cfg_sum["n_stages"])
                    if _safe_float(cfg_sum.get("n_stages")) is not None
                    else None
                ),
                n_mutation_steps=(
                    int(cfg_sum["n_mutation_steps"])
                    if _safe_float(cfg_sum.get("n_mutation_steps")) is not None
                    else None
                ),
            )
        )

    if not candidates:
        raise SystemExit(f"No candidates found under {runs_root}")

    # Prefer candidates that include prefer_model
    def has_prefer(rc: RunCandidate) -> bool:
        return any(mm.model == args.prefer_model for mm in rc.models)

    preferred = [c for c in candidates if has_prefer(c)]
    pool = preferred if preferred else candidates
    pool.sort(key=lambda c: (c.score, -len(c.models), c.run_id))
    best = pool[0]

    out_dir.mkdir(parents=True, exist_ok=True)

    # Figures to embed
    model_names = [mm.model for mm in best.models]
    fig_abs = _collect_figures(Path(best.run_dir), model_names)
    fig_rel = [_relpath_for_docs(p, docs_dir=Path("docs").resolve()) for p in fig_abs]

    # Save artifacts
    (out_dir / "best_run.json").write_text(json.dumps(asdict(best), indent=2), encoding="utf-8")

    # Markdown summary
    lines = []
    lines.append("# Best run (auto-picked)")
    lines.append("")
    lines.append(f"- **run_id**: `{best.run_id}`")
    lines.append(f"- **run_dir**: `{best.run_dir}`")
    lines.append(f"- **score**: `{best.score:.6g}`")
    if best.reasons:
        lines.append("- **notes**:")
        for r in best.reasons:
            lines.append(f"  - {r}")
    lines.append("")
    lines.append("## Per-model key metrics")
    lines.append("")
    lines.append(
        "| Model | RMSE_MAP | beta_final(min over chains) | rom_error_final(max over chains) | acc_mean(min over chains) |"
    )
    lines.append("|---|---:|---:|---:|---:|")
    for mm in best.models:
        lines.append(
            "| "
            + " | ".join(
                [
                    mm.model,
                    f"{mm.rmse_total_map:.4g}" if mm.rmse_total_map is not None else "missing",
                    (
                        f"{mm.beta_final_min_across_chains:.4g}"
                        if mm.beta_final_min_across_chains is not None
                        else "missing"
                    ),
                    (
                        f"{mm.rom_error_final_max_across_chains:.4g}"
                        if mm.rom_error_final_max_across_chains is not None
                        else "missing"
                    ),
                    (
                        f"{mm.accept_rate_mean_min_across_chains:.4g}"
                        if mm.accept_rate_mean_min_across_chains is not None
                        else "missing"
                    ),
                ]
            )
            + " |"
        )
    lines.append("")
    if fig_rel:
        lines.append("## Figures")
        lines.append("")
        for p in fig_rel:
            lines.append(f"- `{p}`")
        lines.append("")
    (out_dir / "best_run_summary.md").write_text("\n".join(lines), encoding="utf-8")

    # TeX report/slides (paths are relative to docs/)
    (out_dir / "auto_best_run_report.tex").write_text(
        _render_tex_report(best, fig_rel), encoding="utf-8"
    )
    (out_dir / "auto_best_run_slides.tex").write_text(
        _render_tex_slides(best, fig_rel), encoding="utf-8"
    )

    # Also write a tiny pointer file at docs/ level for convenience
    pointer = Path("docs") / "BEST_RUN.txt"
    pointer.write_text(f"{best.run_id}\n", encoding="utf-8")

    # Also write a LaTeX-friendly macro file so docs/*.tex can reference the best run id
    # without hard-coding it. Use \detokenize{} to keep underscores safe.
    best_run_tex = Path("docs") / "best_run_id.tex"
    best_run_tex.write_text(
        "% Auto-generated by docs/auto_pick_best_run.py\n"
        f"\\def\\BestRunId{{\\detokenize{{{best.run_id}}}}}\n",
        encoding="utf-8",
    )

    if bool(args.build_pdf):
        build_script = Path("docs") / "build_pdfs.py"
        if build_script.exists():
            subprocess.run([sys.executable, str(build_script)], check=False)

    print(f"[auto_pick_best_run] Best run_id: {best.run_id}")
    print(f"[auto_pick_best_run] Output dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
