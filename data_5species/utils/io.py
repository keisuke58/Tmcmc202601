"""
File I/O utilities for tmcmc package.

Extracted from case2_tmcmc_linearization.py for better modularity.
"""

from __future__ import annotations

import json
import zlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def code_crc32(path: Path) -> str:
    """
    Generate a stable fingerprint of a file (hex crc32).

    Parameters
    ----------
    path : Path
        Path to the file

    Returns
    -------
    str
        Hex-encoded CRC32 checksum (8 characters)
    """
    try:
        b = path.read_bytes()
        return f"{(zlib.crc32(b) & 0xFFFFFFFF):08x}"
    except Exception:
        return "unknown"


def save_npy(path: Path, arr: np.ndarray) -> None:
    """
    Save numpy array with automatic parent directory creation.

    Parameters
    ----------
    path : Path
        Output file path
    arr : np.ndarray
        Array to save
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(arr))


def to_jsonable(obj: Any) -> Any:
    """
    Best-effort conversion of numpy-heavy objects into JSON-serializable types.

    Parameters
    ----------
    obj : Any
        Object to convert

    Returns
    -------
    Any
        JSON-serializable representation
    """
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    # numpy scalar
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    return str(obj)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    """
    Save JSON with numpy-safe conversion and automatic directory creation.

    Parameters
    ----------
    path : Path
        Output file path
    payload : Dict[str, Any]
        Data to save as JSON
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2, ensure_ascii=False)


def write_csv(path: Path, header: List[str], rows: List[List[Any]]) -> None:
    """
    Write a small CSV file with automatic directory creation.

    Parameters
    ----------
    path : Path
        Output file path
    header : List[str]
        Column headers
    rows : List[List[Any]]
        Data rows
    """
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def save_likelihood_meta(
    run_dir: Path,
    *,
    run_id: str,
    model: str,
    sigma_obs: float,
    cov_rel: float,
    n_data: int,
    active_species: List[int],
    active_indices: List[int],
    rho: float = 0.0,
    script_path: Path | None = None,
) -> None:
    """
    Persist a minimal, machine-readable description of the likelihood definition
    used for this run so results can be audited/recomputed later.

    Parameters
    ----------
    run_dir : Path
        Run directory where metadata will be saved
    run_id : str
        Run identifier
    model : str
        Model identifier (e.g., "M1", "M2", "M3")
    sigma_obs : float
        Observation noise standard deviation
    cov_rel : float
        Relative covariance parameter
    n_data : int
        Number of data points
    active_species : List[int]
        Active species indices
    active_indices : List[int]
        Active parameter indices
    rho : float, optional
        Correlation parameter (default: 0.0)
    script_path : Path, optional
        Path to the script file (for CRC32 calculation).
        If None, uses __file__ from caller's frame.
    """
    if script_path is None:
        import inspect

        frame = inspect.currentframe()
        if frame is not None and frame.f_back is not None:
            script_path = Path(frame.f_back.f_globals.get("__file__", ""))
        else:
            script_path = Path("unknown")

    meta = {
        "run_id": run_id,
        "model": model,
        "observable": "phibar = phi * psi",
        "likelihood": {
            "family": "Gaussian",
            "var_total": "sig + sigma_obs^2 (clipped at 1e-20)",
            "logL": "sum_{i,j} [-0.5*log(2*pi*var_total_ij) - 0.5*(data_ij-mu_ij)^2/var_total_ij]",
        },
        "sigma_obs": float(sigma_obs),
        "cov_rel": float(cov_rel),
        "rho": float(rho),
        "n_data": int(n_data),
        "active_species": list(map(int, active_species)),
        "active_indices": list(map(int, active_indices)),
        "script": {
            "path": str(script_path.resolve()) if script_path.exists() else str(script_path),
            "crc32": code_crc32(script_path) if script_path.exists() else "unknown",
        },
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_json(run_dir / f"likelihood_meta_{model}.json", meta)
