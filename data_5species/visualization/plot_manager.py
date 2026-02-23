"""
Plot manager for tmcmc package.

Manages plot generation and file tracking for TMCMC visualization.
Extracted from case2_tmcmc_linearization.py for better modularity.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from .helpers import compute_phibar

logger = logging.getLogger(__name__)


class PlotManager:
    """Manages plot generation and file tracking."""
    
    # Socransky Complexes Colors
    # S1: Blue (Health/Early), S2: Green (Early), S3: Purple (Bridge/Early)
    # S4: Orange (Orange Complex), S5: Red (Red Complex)
    COLORS = [
        '#1f77b4', # S1
        '#2ca02c', # S2
        '#9467bd', # S3
        '#ff7f0e', # S4
        '#d62728'  # S5
    ]

    def __init__(self, output_dir: str):
        """
        Initialize plot manager.
        
        Parameters
        ----------
        output_dir : str
            Output directory for figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.generated_figs: List[Path] = []
        self.figure_counter = 0  # Figure index for paper-style naming
    
    def save_figure(self, filename: str, dpi: int = 150, use_paper_naming: bool = False):
        """
        Save figure with optional paper-style naming (Fig##_filename).
        
        Parameters
        ----------
        filename : str
            Base filename (e.g., "TMCMC_beta_schedule_M1.png")
        dpi : int
            Resolution
        use_paper_naming : bool
            If True, prepend "Fig##_" to filename (e.g., "Fig07_TMCMC_beta_schedule_M1.png")
        """
        if use_paper_naming:
            self.figure_counter += 1
            fig_num = f"{self.figure_counter:02d}"
            filename = f"Fig{fig_num}_{filename}"
        
        path = self.output_dir / filename
        plt.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close()
        self.generated_figs.append(path)
        logger.info("Saved figure: %s", path.name)
    
    def plot_TSM_simulation(
        self,
        t_arr: np.ndarray,
        x0: np.ndarray,
        active_species: List[int],
        name: str,
        data: Optional[np.ndarray] = None,
        idx_sparse: Optional[np.ndarray] = None,
        phibar: Optional[np.ndarray] = None,
        t_days: Optional[np.ndarray] = None,
    ):
        """Plot TSM simulation results.

        Parameters
        ----------
        phibar : Optional[np.ndarray]
            Pre-computed phibar values. If None, will be computed from x0.
            This ensures consistency when data was generated from a specific phibar.
        t_days : Optional[np.ndarray]
            Experimental timepoints in days. If provided, x-axis shows Days instead of normalized time.
        """
        # CRITICAL: Use provided phibar if available, otherwise compute from x0
        # This ensures plot uses the exact same phibar as data generation
        if phibar is None:
            phibar = compute_phibar(x0, active_species)
        else:
            # Verify phibar shape matches x0
            expected_shape = (x0.shape[0], len(active_species))
            if phibar.shape != expected_shape:
                logger.warning(
                    f"phibar shape mismatch: expected {expected_shape}, got {phibar.shape}. "
                    f"Recomputing phibar from x0."
                )
                phibar = compute_phibar(x0, active_species)

        # Use days if provided, otherwise normalize time
        if t_days is not None and idx_sparse is not None:
            # Scale t_arr to days (model time to experimental days)
            t_min = t_arr.min()
            t_max = t_arr.max()
            day_min = t_days.min()
            day_max = t_days.max()
            if t_max > t_min:
                t_plot = day_min + (t_arr - t_min) / (t_max - t_min) * (day_max - day_min)
            else:
                t_plot = t_arr
            t_obs_plot = t_days
            xlabel = "Days"
            xlim = (day_min - 1, day_max + 1)
        else:
            # Normalize time to [0.0, 1.0]
            t_min = t_arr.min()
            t_max = t_arr.max()
            if t_max > t_min:
                t_plot = (t_arr - t_min) / (t_max - t_min)
            else:
                t_plot = t_arr
            t_obs_plot = t_plot[idx_sparse] if idx_sparse is not None else None
            xlabel = "Normalized Time [0.0, 1.0]"
            xlim = (-0.05, 1.05)

        plt.figure(figsize=(10, 6))
        for i, sp in enumerate(active_species):
            color = self.COLORS[sp] if sp < len(self.COLORS) else f"C{sp}"
            plt.plot(t_plot, phibar[:, i], label=f"φ̄{sp+1} (model)", linewidth=2, color=color)

        if data is not None and t_obs_plot is not None:
            # CRITICAL: Verify that data points match model at observation indices
            # This helps debug vertical mismatch issues
            for i, sp in enumerate(active_species):
                # Plot data points
                color = self.COLORS[sp] if sp < len(self.COLORS) else f"C{sp}"
                plt.scatter(
                    t_obs_plot, data[:, i], s=40, edgecolor="k",
                    label=f"Data φ̄{sp+1}", alpha=0.8, zorder=10,
                    color=color
                )

        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel("φ̄ = φ * ψ", fontsize=12)
        plt.title(f"TSM Simulation (φ̄) - {name}", fontsize=14)
        plt.xlim(xlim)
        if t_days is not None:
            plt.xticks(t_days, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()

        suffix = "_with_data" if data is not None else ""
        self.save_figure(f"TSM_simulation_{name}{suffix}.png")

    def plot_posterior_predictive_band(
        self,
        t_arr: np.ndarray,
        phibar_samples: np.ndarray,
        active_species: List[int],
        name: str,
        data: Optional[np.ndarray] = None,
        idx_sparse: Optional[np.ndarray] = None,
        *,
        filename: Optional[str] = None,
        use_paper_naming: bool = False,
        t_days: Optional[np.ndarray] = None,
    ) -> None:
        """
        Plot posterior predictive band for φ̄ = φ * ψ.

        Parameters
        ----------
        phibar_samples : ndarray, shape (n_draws, n_time, n_species)
            φ̄ trajectories for multiple posterior draws.
        t_days : Optional[np.ndarray]
            Experimental timepoints in days. If provided, x-axis shows Days.
        """
        if phibar_samples.ndim != 3:
            raise ValueError(f"phibar_samples must be 3D, got shape {phibar_samples.shape}")

        q05 = np.nanpercentile(phibar_samples, 5, axis=0)
        q50 = np.nanpercentile(phibar_samples, 50, axis=0)
        q95 = np.nanpercentile(phibar_samples, 95, axis=0)

        # Use days if provided, otherwise normalize time
        if t_days is not None and idx_sparse is not None:
            t_min = t_arr.min()
            t_max = t_arr.max()
            day_min = t_days.min()
            day_max = t_days.max()
            if t_max > t_min:
                t_plot = day_min + (t_arr - t_min) / (t_max - t_min) * (day_max - day_min)
            else:
                t_plot = t_arr
            t_obs_plot = t_days
            xlabel = "Days"
            xlim = (day_min - 1, day_max + 1)
        else:
            t_min = t_arr.min()
            t_max = t_arr.max()
            if t_max > t_min:
                t_plot = (t_arr - t_min) / (t_max - t_min)
            else:
                t_plot = t_arr
            t_obs_plot = t_plot[idx_sparse] if idx_sparse is not None else None
            xlabel = "Normalized Time [0.0, 1.0]"
            xlim = (0.0, 1.0)

        plt.figure(figsize=(10, 6))
        for i, sp in enumerate(active_species):
            color = self.COLORS[sp] if sp < len(self.COLORS) else f"C{sp}"
            plt.fill_between(t_plot, q05[:, i], q95[:, i], alpha=0.25, label=f"φ̄{sp+1} 5–95%", color=color)
            plt.plot(t_plot, q50[:, i], linewidth=2, label=f"φ̄{sp+1} median", color=color)

        if data is not None and t_obs_plot is not None:
            for i, sp in enumerate(active_species):
                color = self.COLORS[sp] if sp < len(self.COLORS) else f"C{sp}"
                plt.scatter(
                    t_obs_plot, data[:, i], s=40, edgecolor="k",
                    label=f"Data φ̄{sp+1}", alpha=0.85, zorder=10,
                    color=color
                )

        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel("φ̄ = φ * ψ", fontsize=12)
        plt.title(f"Posterior Predictive Band (φ̄) - {name}", fontsize=14)
        plt.xlim(xlim)
        if t_days is not None:
            plt.xticks(t_days, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=9, ncol=2)
        plt.tight_layout()
        out_name = filename or f"posterior_predictive_{name}.png"
        self.save_figure(out_name, use_paper_naming=use_paper_naming)
    
    def plot_posterior_predictive_spaghetti(
        self,
        t_arr: np.ndarray,
        phibar_samples: np.ndarray,
        active_species: List[int],
        name: str,
        data: Optional[np.ndarray] = None,
        idx_sparse: Optional[np.ndarray] = None,
        num_trajectories: int = 50,
        filename: Optional[str] = None,
        use_paper_naming: bool = False,
        t_days: Optional[np.ndarray] = None,
    ) -> None:
        """
        Plot spaghetti plot of posterior predictive trajectories.

        Parameters
        ----------
        t_days : Optional[np.ndarray]
            Experimental timepoints in days. If provided, x-axis shows Days.
        """
        # Select random subset of trajectories
        n_samples = phibar_samples.shape[0]
        if n_samples > num_trajectories:
            indices = np.random.choice(n_samples, num_trajectories, replace=False)
            phibar_subset = phibar_samples[indices]
        else:
            phibar_subset = phibar_samples

        # Use days if provided, otherwise normalize time
        if t_days is not None and idx_sparse is not None:
            t_min = t_arr.min()
            t_max = t_arr.max()
            day_min = t_days.min()
            day_max = t_days.max()
            if t_max > t_min:
                t_plot = day_min + (t_arr - t_min) / (t_max - t_min) * (day_max - day_min)
            else:
                t_plot = t_arr
            t_obs_plot = t_days
            xlabel = "Days"
            xlim = (day_min - 1, day_max + 1)
        else:
            t_min = t_arr.min()
            t_max = t_arr.max()
            if t_max > t_min:
                t_plot = (t_arr - t_min) / (t_max - t_min)
            else:
                t_plot = t_arr
            t_obs_plot = t_plot[idx_sparse] if idx_sparse is not None else None
            xlabel = "Normalized Time [0.0, 1.0]"
            xlim = (0.0, 1.0)

        plt.figure(figsize=(10, 6))

        for i, sp in enumerate(active_species):
            color = self.COLORS[sp] if sp < len(self.COLORS) else f"C{sp}"
            # Plot trajectories
            for j in range(phibar_subset.shape[0]):
                plt.plot(t_plot, phibar_subset[j, :, i],
                        color=color, alpha=0.1, linewidth=1)

            # Add dummy line for legend
            plt.plot([], [], color=color, label=f"φ̄{sp+1}", linewidth=2)

        if data is not None and t_obs_plot is not None:
            for i, sp in enumerate(active_species):
                color = self.COLORS[sp] if sp < len(self.COLORS) else f"C{sp}"
                plt.scatter(
                    t_obs_plot, data[:, i], s=40, edgecolor="k",
                    facecolor=color, label=f"Data φ̄{sp+1}", alpha=0.9, zorder=10,
                )

        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel("φ̄ = φ * ψ", fontsize=12)
        plt.title(f"Posterior Predictive (Spaghetti) - {name}", fontsize=14)
        plt.xlim(xlim)
        if t_days is not None:
            plt.xticks(t_days, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10, loc='upper right')
        plt.tight_layout()

        out_name = filename or f"posterior_predictive_spaghetti_{name}.png"
        self.save_figure(out_name, use_paper_naming=use_paper_naming)

    def plot_corner(
        self,
        samples: np.ndarray,
        param_names: List[str],
        name: str,
        truths: Optional[List[float]] = None,
        filename: Optional[str] = None,
        use_paper_naming: bool = False,
    ) -> None:
        """
        Generate a corner plot (pair plot) for parameter posteriors.
        Uses matplotlib to avoid dependency on 'corner' package.
        """
        n_params = samples.shape[1]
        n_samples = samples.shape[0]
        
        # Limit number of parameters if too many (e.g. > 10) to avoid clutter
        # For 5-species we have 20 params, which is a lot for a single plot.
        # Maybe split or just plot all small.
        
        fig, axes = plt.subplots(n_params, n_params, figsize=(n_params * 2, n_params * 2))
        
        # Adjust layout
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        
        for i in range(n_params):
            for j in range(n_params):
                ax = axes[i, j]
                
                if i == j:
                    # Diagonal: Histogram
                    ax.hist(samples[:, i], bins=30, density=True, color='k', alpha=0.6, histtype='stepfilled')
                    if truths is not None:
                        ax.axvline(truths[i], color='r', linestyle='--')
                    ax.set_title(param_names[i], fontsize=10)
                elif i > j:
                    # Lower triangle: Scatter/Contour
                    # Use scatter for simplicity and speed with many points
                    # Downsample if needed
                    if n_samples > 1000:
                        indices = np.random.choice(n_samples, 1000, replace=False)
                        x = samples[indices, j]
                        y = samples[indices, i]
                        alpha = 0.3
                    else:
                        x = samples[:, j]
                        y = samples[:, i]
                        alpha = 0.5
                        
                    ax.scatter(x, y, s=5, color='k', alpha=alpha)
                    if truths is not None:
                        ax.axvline(truths[j], color='r', linestyle='--')
                        ax.axhline(truths[i], color='r', linestyle='--')
                        ax.plot(truths[j], truths[i], 'rs')
                else:
                    # Upper triangle: Hide
                    ax.axis('off')
                
                # Labels
                if i == n_params - 1:
                    ax.set_xlabel(param_names[j], fontsize=8, rotation=45)
                else:
                    ax.set_xticklabels([])
                    
                if j == 0 and i > 0:
                    ax.set_ylabel(param_names[i], fontsize=8, rotation=45)
                else:
                    ax.set_yticklabels([])
        
        out_name = filename or f"corner_plot_{name}.png"
        self.save_figure(out_name, use_paper_naming=use_paper_naming)

    def plot_trace(
        self,
        samples: np.ndarray,
        logL: np.ndarray,
        param_names: List[str],
        name: str,
        filename: Optional[str] = None,
        use_paper_naming: bool = False,
    ) -> None:
        """
        Plot parameter traces and log-likelihood evolution.
        Since TMCMC is stage-wise, 'samples' are usually from the final stage.
        If provided samples are from all stages (concatenated), we might see jumps.
        Assuming 'samples' here is the final posterior.
        
        To see evolution, we'd need samples per stage. 
        For now, let's just plot the density of the final samples (violin/box)
        normalized by prior range if possible, or just raw values.
        """
        n_params = samples.shape[1]
        
        # Create a figure with subplots for each parameter + logL
        n_rows = (n_params + 1 + 3) // 4  # 4 cols
        n_cols = 4
        
        plt.figure(figsize=(16, 4 * n_rows))
        
        # Plot parameters
        for i in range(n_params):
            plt.subplot(n_rows, n_cols, i + 1)
            # Histogram/KDE
            plt.hist(samples[:, i], bins=30, density=True, alpha=0.7, color='C0')
            plt.title(param_names[i])
            plt.grid(True, alpha=0.3)
            
            # Add mean/median lines
            mean_val = np.mean(samples[:, i])
            median_val = np.median(samples[:, i])
            plt.axvline(mean_val, color='r', linestyle='-', label=f'Mean: {mean_val:.2f}')
            plt.axvline(median_val, color='g', linestyle='--', label=f'Med: {median_val:.2f}')
            plt.legend(fontsize=8)

        # Plot Log-Likelihood
        plt.subplot(n_rows, n_cols, n_params + 1)
        plt.hist(logL, bins=30, density=True, alpha=0.7, color='purple')
        plt.title("Log-Likelihood")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        out_name = filename or f"parameter_distributions_{name}.png"
        self.save_figure(out_name, use_paper_naming=use_paper_naming)

    def plot_residuals(
        self,
        t_arr: np.ndarray,
        phibar_map: np.ndarray,
        data: np.ndarray,
        idx_sparse: np.ndarray,
        active_species: List[int],
        name: str,
        filename: Optional[str] = None,
        use_paper_naming: bool = False,
        t_days: Optional[np.ndarray] = None,
    ) -> None:
        """
        Plot residuals (Data - Model) over time.

        Parameters
        ----------
        t_days : Optional[np.ndarray]
            Experimental days array for x-axis labeling. If provided, x-axis shows Days.
        """
        if data is None or idx_sparse is None:
            return

        # Extract model values at observation points
        # idx_sparse contains indices in t_arr corresponding to data points
        # phibar_map shape: (n_time, n_species)
        model_vals = phibar_map[idx_sparse]

        # Calculate residuals
        residuals = data - model_vals

        # Determine x-axis values and label
        if t_days is not None:
            t_obs_plot = t_days
            xlabel = "Days"
            xlim = (t_days.min() - 1, t_days.max() + 1)
        else:
            # Normalize time for plotting
            t_obs_plot = (t_arr[idx_sparse] - t_arr.min()) / (t_arr.max() - t_arr.min())
            xlabel = "Normalized Time"
            xlim = (-0.05, 1.05)

        plt.figure(figsize=(10, 6))

        for i, sp in enumerate(active_species):
            color = self.COLORS[sp] if sp < len(self.COLORS) else f"C{sp}"
            plt.plot(t_obs_plot, residuals[:, i], 'o--',
                    color=color, label=f"S{sp+1} Residuals", markersize=8)

        plt.axhline(0, color='k', linestyle='-', linewidth=1)
        plt.xlabel(xlabel, fontsize=14)
        plt.xlim(xlim)
        if t_days is not None:
            plt.xticks(t_days, fontsize=14, fontweight='bold')
        plt.ylabel("Residual (Data - Model)", fontsize=12)
        plt.title(f"Residuals - {name}", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        out_name = filename or f"residuals_{name}.png"
        self.save_figure(out_name, use_paper_naming=use_paper_naming)

    def plot_posterior_predictive_spaghetti(
        self,
        t_arr: np.ndarray,
        phibar_samples: np.ndarray,
        active_species: List[int],
        name: str,
        data: Optional[np.ndarray] = None,
        idx_sparse: Optional[np.ndarray] = None,
        *,
        filename: Optional[str] = None,
        use_paper_naming: bool = False,
        max_draws: int = 120,
        alpha_draws: float = 0.1,
        num_trajectories: int = 50,
        t_days: Optional[np.ndarray] = None,
    ) -> None:
        """
        Plot posterior predictive spaghetti plot (individual trajectories).

        Shows individual posterior sample trajectories as thin lines, along with
        median and 5-95% bands for reference.

        Parameters
        ----------
        t_arr : np.ndarray
            Time array
        phibar_samples : ndarray, shape (n_draws, n_time, n_species)
            φ̄ trajectories for multiple posterior draws
        active_species : List[int]
            Active species indices
        name : str
            Model name
        data : Optional[np.ndarray]
            Observation data
        idx_sparse : Optional[np.ndarray]
            Observation time indices
        filename : Optional[str]
            Output filename
        use_paper_naming : bool
            Use paper-style naming
        max_draws : int
            Maximum number of draws to plot (for performance)
        alpha_draws : float
            Transparency for individual trajectories
        t_days : Optional[np.ndarray]
            Experimental days array for x-axis labeling. If provided, x-axis shows Days.
        """
        if phibar_samples.ndim != 3:
            raise ValueError(f"phibar_samples must be 3D, got shape {phibar_samples.shape}")

        # Use num_trajectories if provided (legacy support)
        target_draws = num_trajectories if num_trajectories is not None else max_draws
        n_draws = min(target_draws, phibar_samples.shape[0])

        # Compute percentiles for reference bands
        q05 = np.nanpercentile(phibar_samples, 5, axis=0)
        q50 = np.nanpercentile(phibar_samples, 50, axis=0)
        q95 = np.nanpercentile(phibar_samples, 95, axis=0)

        # Determine x-axis mapping
        t_min = t_arr.min()
        t_max = t_arr.max()
        if t_days is not None and idx_sparse is not None:
            # Map model time to experimental days
            day_min = t_days.min()
            day_max = t_days.max()
            if t_max > t_min:
                t_plot = day_min + (t_arr - t_min) / (t_max - t_min) * (day_max - day_min)
            else:
                t_plot = t_arr
            t_obs_plot = t_days
            xlabel = "Days"
            xlim = (day_min - 1, day_max + 1)
        else:
            # Normalize time to [0.0, 1.0]
            if t_max > t_min:
                t_plot = (t_arr - t_min) / (t_max - t_min)
            else:
                t_plot = t_arr
            t_obs_plot = t_plot[idx_sparse] if idx_sparse is not None else None
            xlabel = "Normalized Time [0.0, 1.0]"
            xlim = (-0.05, 1.05)

        plt.figure(figsize=(10, 6))

        # Plot individual trajectories (spaghetti)
        for i, sp in enumerate(active_species):
            # Sample a subset of draws for visualization
            if phibar_samples.shape[0] > n_draws:
                rng = np.random.default_rng(42)  # Fixed seed for reproducibility
                draw_indices = rng.choice(phibar_samples.shape[0], size=n_draws, replace=False)
            else:
                draw_indices = np.arange(phibar_samples.shape[0])

            # Plot individual trajectories
            for d_idx in draw_indices:
                plt.plot(
                    t_plot, phibar_samples[d_idx, :, i],
                    color="steelblue", alpha=alpha_draws, linewidth=0.5, zorder=1
                )

            # Plot reference bands and median
            plt.fill_between(
                t_plot, q05[:, i], q95[:, i],
                alpha=0.2, color="steelblue", label=f"φ̄{sp+1} 5–95%" if i == 0 else ""
            )
            plt.plot(
                t_plot, q50[:, i],
                color="darkblue", linewidth=2.5, label=f"φ̄{sp+1} median" if i == 0 else "",
                zorder=5
            )

        # Plot observation data
        if data is not None and idx_sparse is not None and t_obs_plot is not None:
            for i, sp in enumerate(active_species):
                plt.scatter(
                    t_obs_plot, data[:, i], s=50, edgecolor="k", facecolor="red",
                    label=f"Data φ̄{sp+1}", alpha=0.9, zorder=10, linewidth=1.5
                )

        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel("φ̄ = φ * ψ", fontsize=12)
        plt.title(f"Posterior Predictive Spaghetti Plot (φ̄) - {name}", fontsize=14)
        plt.xlim(xlim)
        if t_days is not None:
            plt.xticks(t_days, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=9, ncol=2, loc="best")
        plt.tight_layout()

        out_name = filename or f"posterior_predictive_spaghetti_{name}.png"
        # Replace "posterior_predictive" with "spaghetti" in filename if paper naming
        if use_paper_naming and filename:
            # Extract figure number from filename (e.g., "PaperFig09_posterior_predictive_M1.png")
            base_name = filename.replace("posterior_predictive", "spaghetti")
            out_name = base_name

        self.save_figure(out_name, dpi=150)
    
    def plot_posterior(
        self,
        samples: np.ndarray,
        theta_true: np.ndarray,
        param_names: List[str],
        name_tag: str,
        MAP: np.ndarray,
        mean: np.ndarray,
        use_paper_naming: bool = False,
    ):
        """
        Plot posterior distributions for each parameter.
        
        Parameters
        ----------
        samples : np.ndarray
            Posterior samples (n_samples, n_params)
        theta_true : np.ndarray
            True parameter values
        param_names : List[str]
            Parameter names
        name_tag : str
            Model name tag (e.g., "M1")
        MAP : np.ndarray
            MAP estimates
        mean : np.ndarray
            Posterior mean estimates
        use_paper_naming : bool
            If True, use paper-style naming (PaperFig08_posterior_M1.png, etc.)
        """
        n_params = samples.shape[1]
        cols = min(3, n_params)
        rows = (n_params + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i in range(rows * cols):
            r, c = divmod(i, cols)
            ax = axes[r, c]
            
            if i >= n_params:
                ax.axis("off")
                continue
            
            # Paper style: steelblue histogram
            ax.hist(samples[:, i], bins=40, alpha=0.7, density=True, color="steelblue")
            ax.axvline(theta_true[i], color="red", linestyle="--", linewidth=2, label="True")
            ax.axvline(MAP[i], color="green", linestyle="-", linewidth=2, label="MAP")
            ax.axvline(mean[i], color="orange", linestyle=":", linewidth=2, label="Mean")
            ax.set_xlabel(param_names[i], fontsize=12)
            ax.set_ylabel("Density", fontsize=12)
            ax.grid(True, alpha=0.3)
            if i == 0:  # Only show legend on first subplot
                ax.legend(fontsize=10)
        
        fig.suptitle(f"Posterior Distributions ({name_tag})", fontsize=14)
        fig.tight_layout()
        
        # Paper-style naming: PaperFig08_posterior_M1.png, PaperFig10_posterior_M2.png, PaperFig12_posterior_M3.png
        if use_paper_naming:
            fig_num_map = {"M1": "08", "M2": "10", "M3": "12"}
            fig_num = fig_num_map.get(name_tag, "")
            if fig_num:
                self.save_figure(f"PaperFig{fig_num}_posterior_{name_tag}.png", dpi=150)
            else:
                self.save_figure(f"posterior_{name_tag}.png", dpi=150)
        else:
            self.save_figure(f"posterior_{name_tag}.png", dpi=150)
    
    def plot_pairplot_posterior(
        self,
        samples: np.ndarray,
        theta_true: np.ndarray,
        theta_MAP: np.ndarray,
        theta_mean: np.ndarray,
        param_names: List[str],
        name_tag: str,
    ):
        """
        Plot pairplot of posterior samples with true, MAP, and mean reference lines.
        
        Parameters
        ----------
        samples : np.ndarray
            Posterior samples with shape (n_samples, n_params)
        theta_true : np.ndarray
            True parameter values
        theta_MAP : np.ndarray
            MAP estimates
        theta_mean : np.ndarray
            Posterior mean estimates
        param_names : List[str]
            Parameter names
        name_tag : str
            Model name tag (e.g., "M1")
        """
        import pandas as pd
        
        n = len(param_names)
        if samples.shape[1] != n:
            raise ValueError(f"Number of parameters mismatch: samples has {samples.shape[1]}, param_names has {n}")
        
        df = pd.DataFrame(samples, columns=param_names)
        
        fig, axes = plt.subplots(n, n, figsize=(9, 9))
        
        for i in range(n):
            for j in range(n):
                ax = axes[i, j]
                
                if i == j:
                    # diagonal: histogram
                    ax.hist(df[param_names[i]].values, bins=40, alpha=0.7, color='steelblue')
                    # Add vertical lines for true, MAP, and mean
                    ax.axvline(theta_true[i], color='red', linestyle='--', linewidth=2, 
                              label='True' if i == 0 else '')
                    ax.axvline(theta_MAP[i], color='green', linestyle='--', linewidth=2, 
                              label='MAP' if i == 0 else '')
                    ax.axvline(theta_mean[i], color='orange', linestyle='--', linewidth=2, 
                              label='Mean' if i == 0 else '')
                elif i > j:
                    # lower triangle: scatter
                    ax.scatter(df[param_names[j]].values, df[param_names[i]].values, 
                              s=2, alpha=0.3, color='steelblue')
                    # Add vertical and horizontal lines for true, MAP, and mean
                    ax.axvline(theta_true[j], color='red', linestyle='--', linewidth=1.5, alpha=0.7)
                    ax.axhline(theta_true[i], color='red', linestyle='--', linewidth=1.5, alpha=0.7)
                    ax.axvline(theta_MAP[j], color='green', linestyle='--', linewidth=1.5, alpha=0.7)
                    ax.axhline(theta_MAP[i], color='green', linestyle='--', linewidth=1.5, alpha=0.7)
                    ax.axvline(theta_mean[j], color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
                    ax.axhline(theta_mean[i], color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
                else:
                    # upper triangle: empty (like Fig. 3)
                    ax.axis("off")
                
                # labels only on outer axes
                if i == n - 1 and j < i:
                    ax.set_xlabel(param_names[j], fontsize=9)
                else:
                    ax.set_xticklabels([])
                
                if j == 0 and i > 0:
                    ax.set_ylabel(param_names[i], fontsize=9)
                else:
                    ax.set_yticklabels([])
        
        # Add legend to first diagonal plot
        axes[0, 0].legend(loc='upper right', fontsize=8)
        
        plt.tight_layout()
        self.save_figure(f"pairplot_posterior_{name_tag}.png")
    
    def plot_parameter_comparison(
        self,
        theta_true: np.ndarray,
        theta_map: np.ndarray,
        theta_mean: np.ndarray,
        param_names: List[str],
    ):
        """Plot parameter comparison (True vs MAP vs Mean)."""
        idx = np.arange(len(param_names))
        width = 0.25
        
        plt.figure(figsize=(14, 6))
        plt.bar(idx - width, theta_true, width, label="True", alpha=0.8)
        plt.bar(idx, theta_map, width, label="MAP", alpha=0.8)
        plt.bar(idx + width, theta_mean, width, label="Mean", alpha=0.8)
        
        plt.xticks(idx, param_names, rotation=45, ha="right")
        plt.ylabel("Parameter Value", fontsize=12)
        plt.title("All Parameters: True vs MAP vs Mean", fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        
        self.save_figure("posterior_all_parameters.png")

    def plot_paper_fig14_mean_vs_true_with_std(
        self,
        theta_true: np.ndarray,
        posterior_mean: np.ndarray,
        posterior_std: np.ndarray,
        param_names: List[str],
    ) -> None:
        """
        Paper Fig.14 style:
        - Compare identified posterior mean vs true values
        - Error bars = posterior standard deviation
        """
        idx = np.arange(len(param_names))
        width = 0.38

        plt.figure(figsize=(16, 6))
        plt.bar(idx - width / 2, theta_true, width, label="True", alpha=0.85, color="gray")
        plt.bar(
            idx + width / 2,
            posterior_mean,
            width,
            yerr=posterior_std,
            capsize=3,
            label="Posterior mean ± std",
            alpha=0.85,
            color="steelblue",
        )

        plt.xticks(idx, param_names, rotation=45, ha="right")
        plt.ylabel("Parameter value", fontsize=12)
        plt.title("Paper Fig.14: Identified parameter means vs true values (± posterior std)", fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()

        self.save_figure("PaperFig14_parameter_mean_vs_true.png")
    
    def plot_linearization_improvement(
        self,
        MAP_phase1: np.ndarray,
        MAP_phase2: np.ndarray,
        theta_true_subset: np.ndarray,
        param_names: List[str],
    ):
        """Plot the improvement from linearization update."""
        n_params = len(param_names)
        idx = np.arange(n_params)
        
        error_p1 = np.abs(MAP_phase1 - theta_true_subset)
        error_p2 = np.abs(MAP_phase2 - theta_true_subset)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bar chart of errors
        width = 0.35
        axes[0].bar(idx - width/2, error_p1, width, label="Phase 1 (before update)", alpha=0.8, color="coral")
        axes[0].bar(idx + width/2, error_p2, width, label="Phase 2 (after update)", alpha=0.8, color="seagreen")
        axes[0].set_xticks(idx)
        axes[0].set_xticklabels(param_names)
        axes[0].set_ylabel("|MAP - True|", fontsize=12)
        axes[0].set_title("MAP Error by Parameter", fontsize=14)
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3, axis="y")
        
        # Summary improvement
        total_error_p1 = np.linalg.norm(MAP_phase1 - theta_true_subset)
        total_error_p2 = np.linalg.norm(MAP_phase2 - theta_true_subset)
        improvement = (total_error_p1 - total_error_p2) / total_error_p1 * 100 if total_error_p1 > 0 else 0
        
        axes[1].bar(["Phase 1", "Phase 2"], [total_error_p1, total_error_p2], 
                    color=["coral", "seagreen"], alpha=0.8)
        axes[1].set_ylabel("||MAP - True||", fontsize=12)
        axes[1].set_title(f"Total MAP Error\n(Improvement: {improvement:.1f}%)", fontsize=14)
        axes[1].grid(True, alpha=0.3, axis="y")
        
        plt.tight_layout()
        self.save_figure("linearization_improvement_M3.png")
    
    def save_manifest(self, filename: str = "FIGURES_MANIFEST.json"):
        """Save manifest of all generated figures."""
        manifest = {
            "output_dir": str(self.output_dir),
            "n_figures": len(self.generated_figs),
            "figures": [p.name for p in self.generated_figs],
            "generated_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        path = self.output_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        logger.info("Saved manifest: %s (%s figures)", path.name, len(self.generated_figs))
    
    def plot_beta_schedule(self, beta_schedules: List[List[float]], name: str):
        """Plot TMCMC beta schedule (tempering progression)."""
        plt.figure(figsize=(10, 5))
        for c, beta in enumerate(beta_schedules):
            stages = range(len(beta))
            plt.plot(stages, beta, marker="o", markersize=4, label=f"Chain {c+1}", linewidth=1.5, alpha=0.7)
        
        plt.axhline(1.0, color="red", linestyle="--", linewidth=1, label="β=1.0 (target)", alpha=0.5)
        plt.xlabel("Stage", fontsize=12)
        plt.ylabel(r"$\beta$ (tempering parameter)", fontsize=12)
        plt.title(f"TMCMC Beta Schedule: {name}", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()
        self.save_figure(f"TMCMC_beta_schedule_{name}.png", use_paper_naming=True)
    
    def plot_linearization_history(
        self,
        theta0_histories: List[List[np.ndarray]],
        name: str,
        active_indices: Optional[List[int]] = None,
    ):
        """Plot linearization point update history."""
        if not theta0_histories or all(h is None or len(h) == 0 for h in theta0_histories):
            logger.warning("No linearization history for %s", name)
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Update step norm ||θ₀^{k+1} - θ₀^k||
        for chain_idx, theta0_history in enumerate(theta0_histories):
            if theta0_history is None or len(theta0_history) < 2:
                continue
            
            step_norms = []
            for k in range(1, len(theta0_history)):
                if active_indices is not None:
                    # Only active parameters
                    theta0_k = np.array([theta0_history[k][i] for i in active_indices])
                    theta0_km1 = np.array([theta0_history[k-1][i] for i in active_indices])
                else:
                    theta0_k = theta0_history[k]
                    theta0_km1 = theta0_history[k-1]
                
                step_norm = np.linalg.norm(theta0_k - theta0_km1)
                step_norms.append(step_norm)
            
            if step_norms:
                update_indices = range(1, len(step_norms) + 1)
                axes[0].plot(update_indices, step_norms, marker="o", label=f"Chain {chain_idx+1}", linewidth=1.5)
        
        axes[0].axhline(1e-3, color="red", linestyle="--", linewidth=1, label="Threshold (1e-3)", alpha=0.5)
        axes[0].set_xlabel("Update #", fontsize=12)
        axes[0].set_ylabel(r"$||\theta_0^{k+1} - \theta_0^k||$", fontsize=12)
        axes[0].set_title(f"Linearization Point Update Step Norm: {name}", fontsize=14)
        axes[0].set_yscale("log")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(fontsize=10)
        
        # Plot 2: Total number of updates per chain
        n_updates = [len(h) if h is not None else 0 for h in theta0_histories]
        chain_labels = [f"Chain {i+1}" for i in range(len(n_updates))]
        axes[1].bar(chain_labels, n_updates, alpha=0.7, color="steelblue")
        axes[1].set_ylabel("Number of Updates", fontsize=12)
        axes[1].set_title(f"Total Linearization Updates: {name}", fontsize=14)
        axes[1].grid(True, alpha=0.3, axis="y")
        
        plt.tight_layout()
        self.save_figure(f"TMCMC_linearization_history_{name}.png", use_paper_naming=True)
    
    def plot_rom_error_history(
        self,
        rom_error_history: List[float],
        name: str,
        threshold: float = 0.01,
    ):
        """Plot ROM error history during linearization updates."""
        if not rom_error_history or len(rom_error_history) == 0:
            logger.warning("No ROM error history for %s", name)
            return
        
        plt.figure(figsize=(10, 5))
        update_indices = range(1, len(rom_error_history) + 1)
        plt.plot(update_indices, rom_error_history, marker="o", linewidth=2, markersize=6, label="ROM error")
        plt.axhline(threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold ({threshold})", alpha=0.7)
        plt.xlabel("Linearization Update #", fontsize=12)
        plt.ylabel(r"$\varepsilon_{ROM}$ (relative error)", fontsize=12)
        plt.title(f"ROM Error History: {name}\n" + r"$\varepsilon_{ROM} = ||\bar{\phi}_{ROM}(t_{obs}) - \bar{\phi}_{FOM}(t_{obs})||_2 / ||\bar{\phi}_{FOM}(t_{obs})||_2$", fontsize=14)
        plt.yscale("log")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()
        self.save_figure(f"TMCMC_rom_error_history_{name}.png", use_paper_naming=True)
    
    def plot_map_error_comparison(
        self,
        map_errors_tmcmc: Dict[str, float],
        map_errors_2phase: Optional[Dict[str, float]] = None,
        name: str = "All_Models",
    ):
        """Plot MAP error comparison (TMCMC vs 2-phase MCMC)."""
        models = list(map_errors_tmcmc.keys())
        errors_tmcmc = [map_errors_tmcmc[m] for m in models]
        
        plt.figure(figsize=(10, 6))
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = plt.bar(x - width/2, errors_tmcmc, width, label="TMCMC", alpha=0.8, color="steelblue")
        
        if map_errors_2phase is not None:
            errors_2phase = [map_errors_2phase.get(m, 0) for m in models]
            bars2 = plt.bar(x + width/2, errors_2phase, width, label="2-phase MCMC", alpha=0.8, color="coral")
        
        plt.xlabel("Model", fontsize=12)
        plt.ylabel(r"$||MAP - \theta_{true}||$", fontsize=12)
        plt.title("MAP Error Comparison: TMCMC vs 2-phase MCMC", fontsize=14)
        plt.xticks(x, models, fontsize=11)
        plt.yscale("log")
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        self.save_figure(f"MAP_error_comparison_{name}.png", use_paper_naming=True)
    
    def plot_cost_accuracy_comparison(
        self,
        cost_tmcmc: Dict[str, float],
        map_errors_tmcmc: Dict[str, float],
        cost_2phase: Optional[Dict[str, float]] = None,
        map_errors_2phase: Optional[Dict[str, float]] = None,
        cost_unit: str = "FOM evaluations",
        name: str = "All_Models",
    ):
        """
        Plot cost-accuracy tradeoff (TMCMC vs 2-phase MCMC).
        
        Parameters
        ----------
        cost_tmcmc : Dict[str, float]
            Cost (FOM evaluations, wall time, etc.) for TMCMC per model
        map_errors_tmcmc : Dict[str, float]
            MAP errors for TMCMC per model
        cost_2phase : Optional[Dict[str, float]]
            Cost for 2-phase MCMC per model (if available)
        map_errors_2phase : Optional[Dict[str, float]]
            MAP errors for 2-phase MCMC per model (if available)
        cost_unit : str
            Unit label for cost axis (e.g., "FOM evaluations", "Wall time (s)")
        name : str
            Figure name tag
        """
        models = list(map_errors_tmcmc.keys())
        costs_tmcmc = [cost_tmcmc.get(m, 0) for m in models]
        errors_tmcmc = [map_errors_tmcmc[m] for m in models]
        
        plt.figure(figsize=(10, 6))
        
        # Plot TMCMC
        plt.scatter(costs_tmcmc, errors_tmcmc, s=150, marker="o", label="TMCMC", 
                   color="steelblue", alpha=0.8, zorder=5, edgecolors="black", linewidth=1.5)
        for i, m in enumerate(models):
            plt.annotate(m, (costs_tmcmc[i], errors_tmcmc[i]), 
                        xytext=(5, 5), textcoords="offset points", fontsize=10)
        
        # Plot 2-phase MCMC if available
        if cost_2phase is not None and map_errors_2phase is not None:
            costs_2phase = [cost_2phase.get(m, 0) for m in models]
            errors_2phase = [map_errors_2phase.get(m, 0) for m in models]
            plt.scatter(costs_2phase, errors_2phase, s=150, marker="s", label="2-phase MCMC",
                       color="coral", alpha=0.8, zorder=5, edgecolors="black", linewidth=1.5)
            for i, m in enumerate(models):
                plt.annotate(m, (costs_2phase[i], errors_2phase[i]),
                            xytext=(5, 5), textcoords="offset points", fontsize=10)
        
        plt.xlabel(f"Computational Cost ({cost_unit})", fontsize=12)
        plt.ylabel(r"$||MAP - \theta_{true}||$", fontsize=12)
        plt.title("Cost-Accuracy Tradeoff: TMCMC vs 2-phase MCMC", fontsize=14)
        plt.yscale("log")
        plt.xscale("log")
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3, which="both")
        plt.tight_layout()
        self.save_figure(f"Cost_accuracy_comparison_{name}.png", use_paper_naming=True)
    
    def plot_rom_subset_tradeoff(
        self,
        subset_sizes: List[int],
        map_errors: List[float],
        rom_errors: Optional[List[float]] = None,
        name: str = "M3",
    ):
        """
        Plot cost-accuracy tradeoff for ROM error subset size.
        
        Parameters
        ----------
        subset_sizes : List[int]
            Subset sizes tested (e.g., [5, 10, 20, 50, 100])
        map_errors : List[float]
            MAP errors for each subset size
        rom_errors : Optional[List[float]]
            ROM errors for each subset size (if available)
        name : str
            Model name tag
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: MAP error vs subset size
        axes[0].plot(subset_sizes, map_errors, marker="o", linewidth=2, markersize=8, 
                    color="steelblue", label="MAP error")
        axes[0].axvline(20, color="red", linestyle="--", linewidth=1.5, 
                       label="Selected (20)", alpha=0.7)
        axes[0].set_xlabel("Subset Size (number of particles)", fontsize=12)
        axes[0].set_ylabel(r"$||MAP - \theta_{true}||$", fontsize=12)
        axes[0].set_title(f"MAP Error vs Subset Size: {name}", fontsize=14)
        axes[0].set_yscale("log")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(fontsize=11)
        
        # Plot 2: ROM error vs subset size (if available)
        if rom_errors is not None:
            axes[1].plot(subset_sizes, rom_errors, marker="s", linewidth=2, markersize=8,
                        color="coral", label="ROM error")
            axes[1].axvline(20, color="red", linestyle="--", linewidth=1.5,
                           label="Selected (20)", alpha=0.7)
            axes[1].axhline(0.01, color="gray", linestyle=":", linewidth=1,
                           label="Threshold (0.01)", alpha=0.5)
            axes[1].set_xlabel("Subset Size (number of particles)", fontsize=12)
            axes[1].set_ylabel(r"$\varepsilon_{ROM}$ (relative error)", fontsize=12)
            axes[1].set_title(f"ROM Error vs Subset Size: {name}", fontsize=14)
            axes[1].set_yscale("log")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend(fontsize=11)
        else:
            axes[1].axis("off")
            axes[1].text(0.5, 0.5, "ROM error data not available", 
                        ha="center", va="center", fontsize=12, transform=axes[1].transAxes)
        
        plt.suptitle(f"Cost-Accuracy Tradeoff: ROM Subset Size Selection ({name})", fontsize=16)
        plt.tight_layout()
        self.save_figure(f"ROM_subset_tradeoff_{name}.png", use_paper_naming=True)
