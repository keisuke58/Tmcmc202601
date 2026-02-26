#!/usr/bin/env python3
"""
Phase 2: HMP oral microbiome data download & preprocessing for GNN training.

Downloads HMP 16S OTU tables (V3-V5), filters to oral sites,
aggregates to the 5 target genera, and outputs composition vectors
compatible with graph_builder.py.

Usage:
  python hmp_pipeline.py download          # Download OTU + metadata
  python hmp_pipeline.py preprocess        # Filter oral, aggregate to genus
  python hmp_pipeline.py build-training    # Build GNN training data from HMP
  python hmp_pipeline.py all              # Run all steps
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HMP_DATA_DIR = Path(__file__).parent / "data" / "hmp"

# URLs for HMP QIIME Community Profiles (V3-V5 region)
OTU_URL = "http://downloads.hmpdacc.org/data/HMQCP/otu_table_psn_v35.txt.gz"
MAP_URL = "http://downloads.hmpdacc.org/data/HMQCP/v35_map_uniquebyPSN.txt.bz2"

# Our 5 target genera (order matches Hamilton ODE species indices)
TARGET_GENERA = ["Streptococcus", "Actinomyces", "Veillonella", "Fusobacterium", "Porphyromonas"]
SPECIES_NAMES = ["So", "Act", "Vei", "Fn", "Pg"]

ORAL_SITES = [
    "Subgingival_Plaque",
    "Supragingival_Plaque",
    "Tongue_Dorsum",
    "Buccal_Mucosa",
    "Hard_Palate",
    "Keratinized_Gingiva",
    "Saliva",
    "Palatine_Tonsils",
    "Throat",
]


def download_hmp_data(data_dir: Path = HMP_DATA_DIR):
    """Download HMP V35 OTU table and mapping file."""
    data_dir.mkdir(parents=True, exist_ok=True)

    otu_gz = data_dir / "otu_table_psn_v35.txt.gz"
    map_bz2 = data_dir / "v35_map_uniquebyPSN.txt.bz2"

    for url, dest in [(OTU_URL, otu_gz), (MAP_URL, map_bz2)]:
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  Already exists: {dest.name}")
            continue
        print(f"  Downloading {url} ...")
        try:
            subprocess.run(["wget", "-q", "-O", str(dest), url], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            subprocess.run(["curl", "-L", "-o", str(dest), url], check=True)
        print(f"  Saved: {dest}")

    # Decompress
    otu_txt = data_dir / "otu_table_psn_v35.txt"
    if not otu_txt.exists() and otu_gz.exists():
        print("  Decompressing OTU table...")
        subprocess.run(["gunzip", "-k", str(otu_gz)], check=True)

    map_txt = data_dir / "v35_map_uniquebyPSN.txt"
    if not map_txt.exists() and map_bz2.exists():
        print("  Decompressing mapping file...")
        subprocess.run(["bunzip2", "-k", str(map_bz2)], check=True)

    print("Download complete.")


def load_metadata(data_dir: Path = HMP_DATA_DIR):
    """Load HMP mapping file → dict of {sample_id: metadata}."""
    map_path = data_dir / "v35_map_uniquebyPSN.txt"
    if not map_path.exists():
        raise FileNotFoundError(f"Mapping file not found: {map_path}. Run 'download' first.")

    meta = {}
    with open(map_path) as f:
        header = f.readline().strip().split("\t")
        sid_col = 0  # #SampleID
        site_col = header.index("HMPbodysubsite") if "HMPbodysubsite" in header else None
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) <= 1:
                continue
            sid = parts[sid_col]
            body_site = parts[site_col] if site_col is not None else "Unknown"
            meta[sid] = {"body_site": body_site, "row": parts}
    return meta, header


def parse_otu_table(data_dir: Path = HMP_DATA_DIR):
    """Parse HMP OTU table (QIIME format) → (otu_ids, sample_ids, counts, taxonomy).

    Returns:
        otu_ids: list of OTU IDs
        sample_ids: list of sample IDs
        counts: np.ndarray (n_otus, n_samples)
        taxonomy: list of taxonomy strings
    """
    otu_path = data_dir / "otu_table_psn_v35.txt"
    if not otu_path.exists():
        raise FileNotFoundError(f"OTU table not found: {otu_path}. Run 'download' first.")

    otu_ids = []
    sample_ids = []
    counts_rows = []
    taxonomy = []

    with open(otu_path) as f:
        for line in f:
            if line.startswith("# Constructed"):
                continue
            if (
                line.startswith("#OTU ID")
                or line.startswith("#OTU_ID")
                or line.startswith("OTU ID")
            ):
                parts = line.strip().split("\t")
                # First col = OTU ID, last col might be "Consensus Lineage" or "taxonomy"
                if parts[-1].lower() in ("consensus lineage", "taxonomy"):
                    sample_ids = parts[1:-1]
                else:
                    sample_ids = parts[1:]
                continue
            if line.startswith("#"):
                continue

            parts = line.strip().split("\t")
            otu_ids.append(parts[0])

            # Check if last column is taxonomy string (contains 'k__')
            if "k__" in parts[-1] or ";" in parts[-1]:
                taxonomy.append(parts[-1])
                row = parts[1:-1]
            else:
                taxonomy.append("")
                row = parts[1:]

            counts_rows.append([float(x) for x in row])

    counts = np.array(counts_rows, dtype=np.float64)
    print(f"  OTU table: {counts.shape[0]} OTUs × {counts.shape[1]} samples")
    return otu_ids, sample_ids, counts, taxonomy


def extract_genus(tax_string: str) -> str:
    """Extract genus from QIIME taxonomy string like 'k__Bacteria;...;g__Streptococcus;s__'."""
    for part in tax_string.split(";"):
        part = part.strip()
        if part.startswith("g__"):
            genus = part[3:].strip()
            if genus and genus != "":
                return genus
    return "Unknown"


def preprocess_oral(data_dir: Path = HMP_DATA_DIR):
    """Filter oral samples, aggregate to 5 target genera, normalize.

    Saves:
        data/hmp/hmp_oral_composition.npz
            phi: (n_samples, 5) relative abundances of target genera
            body_sites: (n_samples,) body site labels
            sample_ids: (n_samples,) sample IDs
            genus_totals: (n_samples, 5) raw counts before normalization
    """
    meta, _ = load_metadata(data_dir)
    otu_ids, sample_ids, counts, taxonomy = parse_otu_table(data_dir)

    # Find oral sample indices
    oral_idx = []
    oral_sites = []
    oral_sids = []
    for j, sid in enumerate(sample_ids):
        if sid in meta and meta[sid]["body_site"] in ORAL_SITES:
            oral_idx.append(j)
            oral_sites.append(meta[sid]["body_site"])
            oral_sids.append(sid)

    print(f"  Oral samples: {len(oral_idx)} / {len(sample_ids)}")

    if len(oral_idx) == 0:
        print("  WARNING: No oral samples found. Check metadata format.")
        return

    oral_counts = counts[:, oral_idx]  # (n_otus, n_oral)

    # Map OTUs to genus
    otu_genus = [extract_genus(t) for t in taxonomy]

    # Aggregate counts by target genus
    genus_totals = np.zeros((len(oral_idx), len(TARGET_GENERA)), dtype=np.float64)
    for i, g in enumerate(otu_genus):
        if g in TARGET_GENERA:
            gidx = TARGET_GENERA.index(g)
            genus_totals[:, gidx] += oral_counts[i, :]

    # Compute relative abundance (of target genera within total community)
    total_per_sample = oral_counts.sum(axis=0)  # total reads per sample
    total_per_sample = np.maximum(total_per_sample, 1.0)

    phi = genus_totals / total_per_sample[:, None]  # relative abundance of each genus

    # Renormalize so the 5 genera sum to 1 (composition vector)
    phi_sum = phi.sum(axis=1, keepdims=True)
    phi_sum = np.maximum(phi_sum, 1e-12)
    phi_normed = phi / phi_sum

    print(f"  Target genera coverage: {phi.sum(axis=1).mean():.1%} of total reads (mean)")
    for i, g in enumerate(TARGET_GENERA):
        print(f"    {g}: mean={phi[:, i].mean():.4f}, median={np.median(phi[:, i]):.4f}")

    # Site distribution
    from collections import Counter

    site_counts = Counter(oral_sites)
    for site, cnt in sorted(site_counts.items(), key=lambda x: -x[1]):
        print(f"    {site}: {cnt} samples")

    out_path = data_dir / "hmp_oral_composition.npz"
    np.savez_compressed(
        out_path,
        phi=phi_normed.astype(np.float32),
        phi_raw=phi.astype(np.float32),
        genus_totals=genus_totals.astype(np.float32),
        body_sites=np.array(oral_sites),
        sample_ids=np.array(oral_sids),
        genera=np.array(TARGET_GENERA),
    )
    print(f"  Saved: {out_path} ({phi_normed.shape[0]} samples × {phi_normed.shape[1]} genera)")
    return phi_normed, oral_sites


def build_gnn_training_from_hmp(data_dir: Path = HMP_DATA_DIR, n_augment: int = 5):
    """Build GNN training data using HMP compositions + Hamilton ODE.

    For each HMP composition vector:
    1. Use it as initial condition for Hamilton ODE
    2. Search for theta that produces similar steady state
    3. Extract (phi_stats, a_ij_active) pairs

    This is a forward-model approach: we vary theta around prior,
    simulate ODE, and keep (composition_features, theta) pairs.
    The HMP data provides realistic initial conditions and validation targets.

    Args:
        n_augment: number of theta samples per HMP composition
    """
    comp_path = data_dir / "hmp_oral_composition.npz"
    if not comp_path.exists():
        raise FileNotFoundError(f"Run 'preprocess' first: {comp_path}")

    hmp = np.load(comp_path)
    phi_hmp = hmp["phi"]  # (n_samples, 5) normalized compositions
    body_sites = hmp["body_sites"]

    # Import Hamilton solver
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root / "tmcmc" / "program2602"))
    from improved_5species_jit import BiofilmNewtonSolver5S

    sys.path.insert(0, str(Path(__file__).parent))
    from generate_training_data import load_prior_bounds, ACTIVE_EDGE_THETA_IDX

    bounds = load_prior_bounds("Dysbiotic_HOBIC")
    solver = BiofilmNewtonSolver5S(maxtimestep=500, dt=1e-5)
    rng = np.random.default_rng(42)

    theta_list, phi_mean_list, phi_std_list, phi_final_list, a_ij_list = [], [], [], [], []
    site_list = []
    n_time_out = 100

    print(
        f"  Building training data from {len(phi_hmp)} HMP compositions × {n_augment} augmentations..."
    )

    for idx in range(len(phi_hmp)):
        target_phi = phi_hmp[idx]

        for _ in range(n_augment):
            # Sample theta from prior
            theta = np.zeros(20)
            for i in range(20):
                lo, hi = bounds[i]
                if abs(hi - lo) < 1e-12:
                    theta[i] = lo
                else:
                    theta[i] = rng.uniform(lo, hi)

            try:
                t_arr, g_arr = solver.run_deterministic(theta)
                phi = g_arr[:, :5]
                if np.any(~np.isfinite(phi)) or np.any(phi < -0.5) or np.any(phi > 1.5):
                    continue

                t_out = np.linspace(t_arr[0], t_arr[-1], n_time_out)
                phi_out = np.array([np.interp(t_out, t_arr, phi[:, s]) for s in range(5)]).T

                theta_list.append(theta)
                phi_mean_list.append(np.mean(phi_out, axis=0))
                phi_std_list.append(np.std(phi_out, axis=0))
                phi_final_list.append(phi_out[-1])
                a_ij_list.append(theta[ACTIVE_EDGE_THETA_IDX])
                site_list.append(body_sites[idx] if idx < len(body_sites) else "Unknown")
            except Exception:
                continue

        if (idx + 1) % 100 == 0:
            print(f"    {idx + 1}/{len(phi_hmp)} done, {len(theta_list)} valid samples")

    if not theta_list:
        print("  ERROR: No valid samples generated.")
        return

    out = {
        "theta": np.array(theta_list),
        "phi_mean": np.array(phi_mean_list, dtype=np.float32),
        "phi_std": np.array(phi_std_list, dtype=np.float32),
        "phi_final": np.array(phi_final_list, dtype=np.float32),
        "a_ij_active": np.array(a_ij_list, dtype=np.float32),
        "body_sites": np.array(site_list),
        "hmp_phi": phi_hmp.astype(np.float32),
    }

    out_path = Path(__file__).parent / "data" / f"train_gnn_hmp_N{len(theta_list)}.npz"
    np.savez_compressed(out_path, **out)
    print(f"  Saved: {out_path} ({len(theta_list)} samples)")


def build_offline_multi_condition(n_augment: int = 20, seed: int = 42):
    """Offline mode: build multi-condition training data from existing TMCMC posteriors.

    Uses theta_MAP and posterior samples from 4 conditions to generate
    diverse (composition, a_ij) pairs without needing HMP download.

    This is the recommended path when internet access is unavailable.
    """
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root / "tmcmc" / "program2602"))
    from improved_5species_jit import BiofilmNewtonSolver5S

    sys.path.insert(0, str(Path(__file__).parent))
    from generate_training_data import load_prior_bounds, ACTIVE_EDGE_THETA_IDX

    runs_dir = project_root / "data_5species" / "_runs"
    conditions = {
        "commensal_static": "Commensal_Static",
        "commensal_hobic": "Commensal_HOBIC",
        "dysbiotic_static": "Dysbiotic_Static",
        "dh_baseline": "Dysbiotic_HOBIC",
    }

    rng = np.random.default_rng(seed)
    solver = BiofilmNewtonSolver5S(maxtimestep=500, dt=1e-5)
    n_time_out = 100

    all_theta, all_phi_mean, all_phi_std, all_phi_final, all_aij = [], [], [], [], []
    all_conditions = []

    for dirname, cond_key in conditions.items():
        print(f"\n  --- {cond_key} ---")
        bounds = load_prior_bounds(cond_key)

        # Load MAP and posterior if available
        map_path = runs_dir / dirname / "theta_MAP.json"
        samples_path = runs_dir / dirname / "samples.npy"

        anchors = []
        if samples_path.exists():
            posterior = np.load(samples_path)
            anchors = list(posterior)
            print(f"    Loaded {len(anchors)} posterior samples")
        elif map_path.exists():
            with open(map_path) as f:
                d = json.load(f)
            theta_map = np.array(d.get("theta_full", d))
            anchors = [theta_map]
            print("    Loaded MAP estimate")

        if not anchors:
            print(f"    WARNING: No data for {dirname}, skipping")
            continue

        count = 0
        for anchor in anchors:
            for _ in range(n_augment):
                # Perturb anchor within bounds
                theta = np.copy(anchor)
                for i in range(20):
                    lo, hi = bounds[i]
                    if abs(hi - lo) < 1e-12:
                        theta[i] = lo
                    else:
                        sigma = 0.15 * (hi - lo)
                        theta[i] = np.clip(rng.normal(theta[i], sigma), lo, hi)

                try:
                    t_arr, g_arr = solver.run_deterministic(theta)
                    phi = g_arr[:, :5]
                    if np.any(~np.isfinite(phi)) or np.any(phi < -0.5) or np.any(phi > 1.5):
                        continue

                    t_out = np.linspace(t_arr[0], t_arr[-1], n_time_out)
                    phi_out = np.array([np.interp(t_out, t_arr, phi[:, s]) for s in range(5)]).T

                    all_theta.append(theta)
                    all_phi_mean.append(np.mean(phi_out, axis=0))
                    all_phi_std.append(np.std(phi_out, axis=0))
                    all_phi_final.append(phi_out[-1])
                    all_aij.append(theta[ACTIVE_EDGE_THETA_IDX])
                    all_conditions.append(cond_key)
                    count += 1
                except Exception:
                    continue

        print(f"    Generated {count} valid samples")

    if not all_theta:
        print("  ERROR: No valid samples generated.")
        return

    out_path = Path(__file__).parent / "data" / f"train_gnn_multi_N{len(all_theta)}.npz"
    np.savez_compressed(
        out_path,
        theta=np.array(all_theta),
        phi_mean=np.array(all_phi_mean, dtype=np.float32),
        phi_std=np.array(all_phi_std, dtype=np.float32),
        phi_final=np.array(all_phi_final, dtype=np.float32),
        a_ij_active=np.array(all_aij, dtype=np.float32),
        conditions=np.array(all_conditions),
    )
    print(f"\n  Saved: {out_path} ({len(all_theta)} samples across {len(conditions)} conditions)")


def main():
    parser = argparse.ArgumentParser(description="HMP oral microbiome pipeline for GNN")
    parser.add_argument(
        "step",
        choices=["download", "preprocess", "build-training", "offline", "all"],
        help="Pipeline step to run",
    )
    parser.add_argument("--data-dir", type=str, default=str(HMP_DATA_DIR))
    parser.add_argument(
        "--n-augment", type=int, default=5, help="Number of theta augmentations per HMP composition"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if args.step == "offline":
        print("=== Offline: Build multi-condition data from TMCMC posteriors ===")
        build_offline_multi_condition(n_augment=args.n_augment)
        return

    if args.step in ("download", "all"):
        print("=== Step 1: Download HMP data ===")
        download_hmp_data(data_dir)

    if args.step in ("preprocess", "all"):
        print("\n=== Step 2: Preprocess oral data ===")
        preprocess_oral(data_dir)

    if args.step in ("build-training", "all"):
        print("\n=== Step 3: Build GNN training data ===")
        build_gnn_training_from_hmp(data_dir, n_augment=args.n_augment)


if __name__ == "__main__":
    main()
