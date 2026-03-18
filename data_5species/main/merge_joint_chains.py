#!/usr/bin/env python3
"""
Merge results from parallel joint TMCMC chain runs.

Usage:
    python merge_joint_chains.py _runs/joint_4cond_200p_20260318_XXXXXX
"""

import sys
import json
import numpy as np
from pathlib import Path

# Add paths
DATA_5SPECIES_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(DATA_5SPECIES_ROOT))
sys.path.insert(0, str(DATA_5SPECIES_ROOT.parent / "tmcmc" / "program2602"))

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "estimate_reduced_nishioka",
    str(Path(__file__).parent / "estimate_reduced_nishioka.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
compute_mcmc_diagnostics = _mod.compute_mcmc_diagnostics
compute_credible_intervals = _mod.compute_credible_intervals
compute_model_evidence = _mod.compute_model_evidence
compute_MAP_with_uncertainty = _mod.compute_MAP_with_uncertainty

from core.joint_evaluator import ALL_AIJ_INDICES, PARAM_NAMES


def main():
    if len(sys.argv) < 2:
        print("Usage: python merge_joint_chains.py <output_base_dir>")
        sys.exit(1)

    base_dir = Path(sys.argv[1])

    # Find chain directories
    chain_dirs = sorted(base_dir.glob("chain_*"))
    chain_dirs = [d for d in chain_dirs if d.is_dir()]

    if not chain_dirs:
        print(f"No chain directories found in {base_dir}")
        sys.exit(1)

    print(f"Found {len(chain_dirs)} chains")

    # Load samples and logL from each chain
    chains = []
    logL_chains = []
    for cd in chain_dirs:
        samples_file = cd / "samples.npy"
        logL_file = cd / "logL.npy"
        if samples_file.exists() and logL_file.exists():
            s = np.load(samples_file)
            l = np.load(logL_file)
            chains.append(s)
            logL_chains.append(l)
            print(f"  {cd.name}: {s.shape[0]} samples, logL range [{l.min():.2f}, {l.max():.2f}]")
        else:
            print(f"  {cd.name}: MISSING samples/logL — skipping")

    if len(chains) < 2:
        print("Need at least 2 chains for diagnostics")
        if len(chains) == 1:
            print("Copying single chain results as-is")
            import shutil

            for f in (chain_dirs[0]).glob("*"):
                if f.is_file():
                    shutil.copy2(f, base_dir / f.name)
            return

    # Merge
    all_samples = np.concatenate(chains, axis=0)
    all_logL = np.concatenate(logL_chains, axis=0)

    print(f"\nMerged: {all_samples.shape[0]} total samples, {all_samples.shape[1]} params")

    # Save merged
    np.save(base_dir / "samples.npy", all_samples)
    np.save(base_dir / "logL.npy", all_logL)

    # MAP
    results = compute_MAP_with_uncertainty(all_samples, all_logL)
    MAP_estimate = results["MAP"]
    n_joint = len(ALL_AIJ_INDICES)
    joint_param_names = [PARAM_NAMES[i] for i in ALL_AIJ_INDICES]

    # Diagnostics
    mcmc_diag = compute_mcmc_diagnostics(chains, logL_chains, joint_param_names)
    ci = compute_credible_intervals(all_samples, 0.95)

    # Evidence
    evidence = compute_model_evidence(
        chains=chains,
        logL=logL_chains,
        prior_bounds=[(0, 1)] * n_joint,  # placeholder, not critical
        active_indices=list(range(n_joint)),
    )

    # Save
    with open(base_dir / "mcmc_diagnostics.json", "w") as f:
        json.dump(
            {
                "rhat": mcmc_diag["rhat"].tolist(),
                "ess": mcmc_diag["ess"].tolist(),
                "rhat_max": mcmc_diag["rhat_max"],
                "ess_min": mcmc_diag["ess_min"],
                "converged": mcmc_diag["converged"],
                "n_chains": len(chains),
                "n_samples_per_chain": [len(c) for c in chains],
                "per_parameter": mcmc_diag["per_parameter"],
            },
            f,
            indent=2,
        )

    with open(base_dir / "credible_intervals.json", "w") as f:
        json.dump(
            {
                "hdi_lower": ci["hdi_lower"].tolist(),
                "hdi_upper": ci["hdi_upper"].tolist(),
                "et_lower": ci["et_lower"].tolist(),
                "et_upper": ci["et_upper"].tolist(),
                "median": ci["median"].tolist(),
                "param_names": joint_param_names,
            },
            f,
            indent=2,
        )

    theta_joint_MAP = MAP_estimate[:n_joint] if len(MAP_estimate) >= n_joint else MAP_estimate
    with open(base_dir / "theta_MAP.json", "w") as f:
        json.dump(
            {
                "theta_joint": theta_joint_MAP.tolist(),
                "joint_param_names": joint_param_names,
            },
            f,
            indent=2,
        )

    with open(base_dir / "evidence.json", "w") as f:
        json.dump(evidence, f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("MERGED JOINT TMCMC RESULTS")
    print("=" * 70)
    print(f"Chains: {len(chains)}, Total samples: {all_samples.shape[0]}")
    print(f"R-hat max: {mcmc_diag['rhat_max']:.4f} (target < 1.1)")
    print(f"ESS min: {mcmc_diag['ess_min']:.1f} (target > 100)")
    print(f"Converged: {mcmc_diag['converged']}")
    print(f"Evidence: log(p(D)) = {evidence['log_evidence']:.2f}")
    print("\nMAP estimates (15 A_ij):")
    for i, aij_idx in enumerate(ALL_AIJ_INDICES):
        ci_lo = ci["hdi_lower"][i]
        ci_hi = ci["hdi_upper"][i]
        print(
            f"  {PARAM_NAMES[aij_idx]:5s} = {theta_joint_MAP[i]:+.4f} [{ci_lo:+.4f}, {ci_hi:+.4f}]"
        )
    print(f"\nMax logL = {all_logL.max():.2f}")
    print(f"Results saved to: {base_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
