#!/usr/bin/env python3
"""Show summary table (RMSE, convergence, elapsed) for recent TMCMC runs."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_run(run_dir: Path) -> dict:
    """Load key results from a run directory."""
    info = {"dir": run_dir.name}

    # Config
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        cfg = json.load(open(cfg_path))
        info["condition"] = cfg.get("condition", "?")
        info["cultivation"] = cfg.get("cultivation", "?")
        info["n_particles"] = cfg.get("n_particles", "?")
        info["n_chains"] = cfg.get("n_chains", "?")

    # Fit metrics (RMSE)
    fm_path = run_dir / "fit_metrics.json"
    if fm_path.exists():
        fm = json.load(open(fm_path))
        if "MAP" in fm:
            info["rmse_map"] = fm["MAP"]["rmse_total"]
            info["rmse_map_species"] = fm["MAP"]["rmse_per_species"]
        if "Mean" in fm:
            info["rmse_mean"] = fm["Mean"]["rmse_total"]
        info["species_names"] = fm.get("species_names", [])

    # MCMC diagnostics
    diag_path = run_dir / "mcmc_diagnostics.json"
    if diag_path.exists():
        diag = json.load(open(diag_path))
        info["rhat_max"] = diag.get("rhat_max")
        # ess_min might be NaN
        ess_min = diag.get("ess_min")
        if ess_min is not None and not (isinstance(ess_min, float) and ess_min != ess_min):
            info["ess_min"] = ess_min
        info["converged"] = diag.get("converged", False)

    # Results summary (elapsed time, stages)
    rs_path = run_dir / "results_summary.json"
    if rs_path.exists():
        rs = json.load(open(rs_path))
        info["elapsed_s"] = rs.get("elapsed_seconds") or rs.get("elapsed_time")
        info["n_stages"] = rs.get("n_stages_completed")

    # Samples shape
    samples_path = run_dir / "samples.npy"
    if samples_path.exists():
        s = np.load(samples_path)
        info["n_samples"] = s.shape[0]
        info["n_params"] = s.shape[1]

    return info


def print_summary(runs: list, verbose: bool = False):
    """Print summary table."""
    header = f"{'Cond':<5} {'Cult':<8} {'Np':>4} {'Ch':>2} {'Stg':>3} {'Time':>7} {'RMSE(MAP)':>10} {'RMSE(Mean)':>11} {'R-hat':>6} {'ESS':>6} {'Conv':>4}"
    print(header)
    print("-" * len(header))

    for r in runs:
        cond = str(r.get("condition", "?"))[:4]
        cult = str(r.get("cultivation", "?"))[:7]
        np_ = str(r.get("n_particles", "-"))
        nc = str(r.get("n_chains", "-"))
        ns = str(r.get("n_stages") or "-")
        elapsed = r.get("elapsed_s")
        time_str = f"{elapsed/3600:.1f}h" if elapsed else "-"
        rmse_map = f"{r['rmse_map']:.4f}" if "rmse_map" in r else "-"
        rmse_mean = f"{r['rmse_mean']:.4f}" if "rmse_mean" in r else "-"
        rhat = f"{r['rhat_max']:.3f}" if r.get("rhat_max") else "-"
        ess = f"{r['ess_min']:.0f}" if r.get("ess_min") else "-"
        conv = "Y" if r.get("converged") else "N"
        print(
            f"{cond:<5} {cult:<8} {np_:>4} {nc:>2} {ns:>3} {time_str:>7} {rmse_map:>10} {rmse_mean:>11} {rhat:>6} {ess:>6} {conv:>4}"
        )

    if verbose:
        print()
        for r in runs:
            if "rmse_map_species" in r:
                names = r.get("species_names", [f"sp{i}" for i in range(5)])
                cond = f"{r.get('condition','?')}_{r.get('cultivation','?')}"
                vals = "  ".join(
                    f"{n[:6]:>6}={v:.3f}" for n, v in zip(names, r["rmse_map_species"])
                )
                print(f"  {cond}: {vals}")


def main():
    parser = argparse.ArgumentParser(description="Show TMCMC run summary")
    parser.add_argument(
        "patterns",
        nargs="*",
        default=["*500p_expIC_repSigma*"],
        help="Glob patterns for run dirs (default: *500p_expIC_repSigma*)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Show per-species RMSE")
    parser.add_argument("--runs-dir", default=None, help="Runs directory")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir) if args.runs_dir else Path(__file__).parent / "_runs"
    if not runs_dir.exists():
        print(f"Runs dir not found: {runs_dir}", file=sys.stderr)
        sys.exit(1)

    all_runs = []
    for pat in args.patterns:
        for d in sorted(runs_dir.glob(pat)):
            if d.is_dir():
                all_runs.append(load_run(d))

    if not all_runs:
        print("No matching runs found.", file=sys.stderr)
        sys.exit(1)

    print_summary(all_runs, verbose=args.verbose)


if __name__ == "__main__":
    main()
