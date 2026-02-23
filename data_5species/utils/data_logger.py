"""
Data logger for saving TMCMC execution logs and data for analysis.

This module provides functionality to save:
- Detailed execution logs (TMCMC progress, ESS, acceptance rates, etc.)
- Posterior samples and predictive samples
- History data (beta schedules, parameter evolution, etc.)
- Metadata (experiment config, model configs, etc.)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class DataLogger:
    """Logger for saving TMCMC execution data and logs."""
    
    def __init__(self, output_dir: str | Path):
        """
        Initialize data logger.
        
        Parameters
        ----------
        output_dir : str | Path
            Root output directory for the run
        """
        self.output_dir = Path(output_dir)
        self.logs_dir = self.output_dir / "logs"
        self.data_dir = self.output_dir / "data"
        self.metadata_dir = self.output_dir / "metadata"
        
        # Create directories
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Log file handles
        self.tmcmc_log_file = None
        self.linearization_log_file = None
        self.rom_error_log_file = None
        self.parameter_evolution_log_file = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close log files."""
        self.close_logs()
    
    def close_logs(self):
        """Close all log file handles."""
        for log_file in [self.tmcmc_log_file, self.linearization_log_file, 
                         self.rom_error_log_file, self.parameter_evolution_log_file]:
            if log_file is not None:
                try:
                    log_file.close()
                except Exception:
                    pass
    
    def save_tmcmc_log_entry(
        self,
        model: str,
        stage: int,
        beta: float,
        beta_next: float,
        ess: float,
        ess_target: float,
        acc_rate: float,
        logL_min: float,
        logL_max: float,
        n_particles: int,
    ):
        """
        Save a TMCMC stage log entry.
        
        Parameters
        ----------
        model : str
            Model name (M1, M2, M3)
        stage : int
            Stage number
        beta : float
            Current beta value
        beta_next : float
            Next beta value
        ess : float
            Effective Sample Size
        ess_target : float
            Target ESS
        acc_rate : float
            Acceptance rate
        logL_min : float
            Minimum log-likelihood
        logL_max : float
            Maximum log-likelihood
        n_particles : int
            Number of particles
        """
        if self.tmcmc_log_file is None:
            self.tmcmc_log_file = open(self.logs_dir / "tmcmc_detailed.log", "a", encoding="utf-8")
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "stage": int(stage),
            "beta": float(beta),
            "beta_next": float(beta_next),
            "ess": float(ess),
            "ess_target": float(ess_target),
            "ess_ratio": float(ess / ess_target) if ess_target > 0 else 0.0,
            "acc_rate": float(acc_rate),
            "logL_min": float(logL_min),
            "logL_max": float(logL_max),
            "n_particles": int(n_particles),
        }
        
        self.tmcmc_log_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self.tmcmc_log_file.flush()
    
    def save_linearization_update(
        self,
        model: str,
        update_num: int,
        theta0_before: np.ndarray,
        theta0_after: np.ndarray,
        rom_error_before: float,
        rom_error_after: float,
        active_indices: Optional[List[int]] = None,
    ):
        """
        Save a linearization update log entry.
        
        Parameters
        ----------
        model : str
            Model name
        update_num : int
            Update number
        theta0_before : np.ndarray
            Linearization point before update
        theta0_after : np.ndarray
            Linearization point after update
        rom_error_before : float
            ROM error before update
        rom_error_after : float
            ROM error after update
        active_indices : Optional[List[int]]
            Active parameter indices (for subset logging)
        """
        if self.linearization_log_file is None:
            self.linearization_log_file = open(
                self.logs_dir / "linearization_updates.log", "a", encoding="utf-8"
            )
        
        if active_indices is not None:
            theta0_before_subset = theta0_before[active_indices].tolist()
            theta0_after_subset = theta0_after[active_indices].tolist()
            step_norm = float(np.linalg.norm(theta0_after[active_indices] - theta0_before[active_indices]))
        else:
            theta0_before_subset = theta0_before.tolist()
            theta0_after_subset = theta0_after.tolist()
            step_norm = float(np.linalg.norm(theta0_after - theta0_before))
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "update_num": int(update_num),
            "theta0_before": theta0_before_subset,
            "theta0_after": theta0_after_subset,
            "step_norm": step_norm,
            "rom_error_before": float(rom_error_before),
            "rom_error_after": float(rom_error_after),
            "rom_error_improvement": float(rom_error_before - rom_error_after),
        }
        
        self.linearization_log_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self.linearization_log_file.flush()
    
    def save_rom_error_entry(
        self,
        model: str,
        update_num: int,
        rom_error: float,
        threshold: float = 0.01,
    ):
        """
        Save a ROM error log entry.
        
        Parameters
        ----------
        model : str
            Model name
        update_num : int
            Update number
        rom_error : float
            ROM error value
        threshold : float
            ROM error threshold
        """
        if self.rom_error_log_file is None:
            self.rom_error_log_file = open(
                self.logs_dir / "rom_error_history.log", "a", encoding="utf-8"
            )
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "update_num": int(update_num),
            "rom_error": float(rom_error),
            "threshold": float(threshold),
            "below_threshold": rom_error < threshold,
        }
        
        self.rom_error_log_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self.rom_error_log_file.flush()
    
    def save_parameter_evolution(
        self,
        model: str,
        stage: int,
        theta_MAP: np.ndarray,
        theta_mean: np.ndarray,
        theta_std: np.ndarray,
        active_indices: Optional[List[int]] = None,
    ):
        """
        Save parameter evolution log entry.
        
        Parameters
        ----------
        model : str
            Model name
        stage : int
            Stage number
        theta_MAP : np.ndarray
            MAP estimate
        theta_mean : np.ndarray
            Mean estimate
        theta_std : np.ndarray
            Standard deviation
        active_indices : Optional[List[int]]
            Active parameter indices
        """
        if self.parameter_evolution_log_file is None:
            self.parameter_evolution_log_file = open(
                self.logs_dir / "parameter_evolution.log", "a", encoding="utf-8"
            )
        
        if active_indices is not None:
            theta_MAP_subset = theta_MAP[active_indices].tolist()
            theta_mean_subset = theta_mean[active_indices].tolist()
            theta_std_subset = theta_std[active_indices].tolist()
        else:
            theta_MAP_subset = theta_MAP.tolist()
            theta_mean_subset = theta_mean.tolist()
            theta_std_subset = theta_std.tolist()
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "stage": int(stage),
            "theta_MAP": theta_MAP_subset,
            "theta_mean": theta_mean_subset,
            "theta_std": theta_std_subset,
        }
        
        self.parameter_evolution_log_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self.parameter_evolution_log_file.flush()
    
    def save_posterior_samples(
        self,
        model: str,
        samples: np.ndarray,
        param_names: List[str],
    ):
        """
        Save posterior samples to NPZ file.
        
        Parameters
        ----------
        model : str
            Model name
        samples : np.ndarray
            Posterior samples (n_samples, n_params)
        param_names : List[str]
            Parameter names
        """
        filename = self.data_dir / f"posterior_samples_{model}.npz"
        np.savez_compressed(
            filename,
            samples=samples,
            param_names=np.array(param_names, dtype=object),
        )
        logger.info("Saved posterior samples: %s (%d samples, %d params)", 
                   filename.name, samples.shape[0], samples.shape[1])
    
    def save_phibar_samples(
        self,
        model: str,
        phibar_samples: np.ndarray,
        t_arr: np.ndarray,
        active_species: List[int],
    ):
        """
        Save posterior predictive phibar samples.
        
        Parameters
        ----------
        model : str
            Model name
        phibar_samples : np.ndarray
            phibar samples (n_draws, n_time, n_species)
        t_arr : np.ndarray
            Time array
        active_species : List[int]
            Active species indices
        """
        filename = self.data_dir / f"phibar_samples_{model}.npz"
        np.savez_compressed(
            filename,
            phibar_samples=phibar_samples,
            t_arr=t_arr,
            active_species=np.array(active_species),
        )
        logger.info("Saved phibar samples: %s (%d draws, %d time points, %d species)",
                   filename.name, phibar_samples.shape[0], phibar_samples.shape[1], 
                   phibar_samples.shape[2])
    
    def save_beta_schedules(
        self,
        beta_schedules: Dict[str, List[List[float]]],
    ):
        """
        Save beta schedules for all models.
        
        Parameters
        ----------
        beta_schedules : Dict[str, List[List[float]]]
            Beta schedules per model (model_name -> list of chains)
        """
        filename = self.data_dir / "beta_schedules.npz"
        
        # Convert to arrays
        beta_dict = {}
        for model, chains in beta_schedules.items():
            # Pad chains to same length
            max_len = max(len(chain) for chain in chains) if chains else 0
            if max_len > 0:
                padded_chains = []
                for chain in chains:
                    padded = chain + [chain[-1]] * (max_len - len(chain))
                    padded_chains.append(padded)
                beta_dict[f"{model}_beta_schedules"] = np.array(padded_chains)
        
        np.savez_compressed(filename, **beta_dict)
        logger.info("Saved beta schedules: %s", filename.name)
    
    def save_history_data(
        self,
        model: str,
        theta_MAP_history: List[np.ndarray],
        theta_mean_history: List[np.ndarray],
        ess_history: List[float],
        acc_rate_history: List[float],
        rom_error_history: Optional[List[float]] = None,
    ):
        """
        Save history data (MAP, Mean, ESS, acceptance rate, ROM error).
        
        Parameters
        ----------
        model : str
            Model name
        theta_MAP_history : List[np.ndarray]
            MAP estimate history
        theta_mean_history : List[np.ndarray]
            Mean estimate history
        ess_history : List[float]
            ESS history
        acc_rate_history : List[float]
            Acceptance rate history
        rom_error_history : Optional[List[float]]
            ROM error history
        """
        filename = self.data_dir / f"history_{model}.npz"
        
        save_dict = {
            "theta_MAP_history": np.array(theta_MAP_history),
            "theta_mean_history": np.array(theta_mean_history),
            "ess_history": np.array(ess_history),
            "acc_rate_history": np.array(acc_rate_history),
        }
        
        if rom_error_history is not None:
            save_dict["rom_error_history"] = np.array(rom_error_history)
        
        np.savez_compressed(filename, **save_dict)
        logger.info("Saved history data: %s", filename.name)
    
    def save_metadata(
        self,
        experiment_config: Dict[str, Any],
        model_configs: Dict[str, Any],
        data_generation: Dict[str, Any],
        convergence_summary: Dict[str, Any],
        timing_breakdown: Dict[str, Any],
    ):
        """
        Save metadata files.
        
        Parameters
        ----------
        experiment_config : Dict[str, Any]
            Experiment configuration
        model_configs : Dict[str, Any]
            Model configurations
        data_generation : Dict[str, Any]
            Data generation settings
        convergence_summary : Dict[str, Any]
            Convergence summary
        timing_breakdown : Dict[str, Any]
            Timing breakdown
        """
        def to_jsonable(obj: Any) -> Any:
            """Convert numpy types to JSON-serializable types."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: to_jsonable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [to_jsonable(item) for item in obj]
            else:
                return obj
        
        # Save experiment config
        with open(self.metadata_dir / "experiment_config.json", "w", encoding="utf-8") as f:
            json.dump(to_jsonable(experiment_config), f, indent=2, ensure_ascii=False)
        
        # Save model configs
        with open(self.metadata_dir / "model_configs.json", "w", encoding="utf-8") as f:
            json.dump(to_jsonable(model_configs), f, indent=2, ensure_ascii=False)
        
        # Save data generation
        with open(self.metadata_dir / "data_generation.json", "w", encoding="utf-8") as f:
            json.dump(to_jsonable(data_generation), f, indent=2, ensure_ascii=False)
        
        # Save convergence summary
        with open(self.metadata_dir / "convergence_summary.json", "w", encoding="utf-8") as f:
            json.dump(to_jsonable(convergence_summary), f, indent=2, ensure_ascii=False)
        
        # Save timing breakdown
        with open(self.metadata_dir / "timing_breakdown.json", "w", encoding="utf-8") as f:
            json.dump(to_jsonable(timing_breakdown), f, indent=2, ensure_ascii=False)
        
        logger.info("Saved metadata files to %s", self.metadata_dir)
