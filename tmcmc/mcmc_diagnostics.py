"""
mcmc_diagnostics.py - MCMC Convergence Diagnostics

Provides Gelman-Rubin R-hat, effective sample size (ESS),
and summary statistics for MCMC chains.

Author: Keisuke (keisuke58)
Date: December 2024
"""

import numpy as np
import json
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class MCMCDiagnostics:
    """
    MCMC diagnostics including R-hat, ESS, and summary statistics.
    
    Parameters
    ----------
    chains : List[np.ndarray]
        List of MCMC chains, each (n_samples, n_params)
    param_names : List[str]
        Parameter names
    """
    
    def __init__(self, chains: List[np.ndarray], param_names: List[str]):
        self.chains = [np.asarray(c) for c in chains]
        self.param_names = param_names
        self.n_chains = len(chains)
        self.n_samples = self.chains[0].shape[0]
        self.n_params = self.chains[0].shape[1]
        
        # Results storage
        self.Rhat = None
        self.ESS = None
        self.mean = None
        self.std = None
        self.quantiles = None
    
    def compute_rhat(self) -> np.ndarray:
        """
        Compute Gelman-Rubin R-hat statistic.
        Requires at least 2 chains; otherwise undefined.
        """
        m = self.n_chains
        p = self.n_params

        if m < 2:
            # R-hat undefined for a single chain
            self.Rhat = np.full(p, np.nan, dtype=float)
            return self.Rhat

        n = self.n_samples

        # Stack chains: (m, n, p)
        arr = np.stack(self.chains, axis=0)

        # Chain means: (m, p)
        chain_means = arr.mean(axis=1)

        # Overall mean: (p,)
        overall_mean = arr.mean(axis=(0, 1))

        # Between-chain variance
        B = n / (m - 1) * np.sum((chain_means - overall_mean) ** 2, axis=0)

        # Within-chain variance
        W = np.mean(np.var(arr, axis=1, ddof=1), axis=0)

        # Pooled variance estimate
        var_hat = (n - 1) / n * W + B / n

        # R-hat
        self.Rhat = np.sqrt(var_hat / (W + 1e-10))
        return self.Rhat

    
    def compute_ess(self, max_lag: int = 200) -> np.ndarray:
        """
        Compute effective sample size (ESS).
        
        Parameters
        ----------
        max_lag : int
            Maximum lag for autocorrelation
            
        Returns
        -------
        ESS : ndarray (n_params,)
            Effective sample size per parameter
        """
        # Concatenate all chains
        arr = np.concatenate(self.chains, axis=0)
        N, p = arr.shape
        
        self.ESS = np.zeros(p)
        
        for j in range(p):
            x = arr[:, j] - arr[:, j].mean()
            
            # Autocorrelation
            acov = np.correlate(x, x, mode='full')
            acov = acov[acov.size // 2:]
            acov = acov[:max_lag + 1]
            acov /= acov[0] + 1e-10
            
            rho = acov
            
            # Sum pairs until negative (Geyer's truncation)
            t = 1
            s = 0.0
            while t + 1 <= max_lag:
                if rho[t] + rho[t + 1] < 0:
                    break
                s += rho[t] + rho[t + 1]
                t += 2
            
            self.ESS[j] = N / (1 + 2 * s)
        
        return self.ESS
    
    def compute_summary(self) -> dict:
        """
        Compute summary statistics.
        
        Returns
        -------
        summary : dict
            Dictionary with mean, std, quantiles
        """
        arr = np.concatenate(self.chains, axis=0)
        
        self.mean = arr.mean(axis=0)
        self.std = arr.std(axis=0, ddof=1)
        self.quantiles = {
            '2.5%': np.percentile(arr, 2.5, axis=0),
            '25%': np.percentile(arr, 25, axis=0),
            '50%': np.percentile(arr, 50, axis=0),
            '75%': np.percentile(arr, 75, axis=0),
            '97.5%': np.percentile(arr, 97.5, axis=0),
        }
        
        return {
            'mean': self.mean,
            'std': self.std,
            'quantiles': self.quantiles,
        }
    
    def compute_all(self):
        """Compute all diagnostics."""
        self.compute_rhat()
        self.compute_ess()
        self.compute_summary()
    
    def print_summary(self, log: Optional[logging.Logger] = None):
        """
        Log formatted summary of diagnostics.

        Kept for backward compatibility (legacy name), but it does not use `print`.
        """
        log = log or logger
        if self.Rhat is None:
            self.compute_all()

        lines: List[str] = []
        lines.append("=" * 70)
        lines.append("MCMC Diagnostics Summary")
        lines.append("=" * 70)
        lines.append(f"Number of chains: {self.n_chains}")
        lines.append(f"Samples per chain: {self.n_samples}")
        lines.append(f"Total samples: {self.n_chains * self.n_samples}")
        lines.append("")
        lines.append(f"{'Param':<10} {'Mean':>10} {'Std':>10} {'R-hat':>8} {'ESS':>8}")
        lines.append("-" * 50)
        
        for i, name in enumerate(self.param_names):

            if np.isnan(self.Rhat[i]):
                rhat_str = "  N/A "
                rhat_flag = " "
            else:
                rhat_str = f"{self.Rhat[i]:7.3f}"
                rhat_flag = "ok" if self.Rhat[i] < 1.1 else "bad"

            ess_flag = "ok" if self.ESS[i] > 100 else "bad"

            lines.append(
                f"{name:<10} {self.mean[i]:>10.4f} {self.std[i]:>10.4f} "
                f"{rhat_str}{rhat_flag:>3} {self.ESS[i]:>7.0f}{ess_flag:>4}"
            )

        
        # Summary
        lines.append("-" * 50)
        # ★ 5) NaN処理: single chain のとき R-hat は NaN になるため nanmax/nanmin を使用
        max_rhat = np.nanmax(self.Rhat)
        min_ess = np.nanmin(self.ESS)
        
        if max_rhat < 1.1:
            lines.append(f"R-hat: All < 1.1 (max = {max_rhat:.3f}) - CONVERGED")
        else:
            lines.append(f"R-hat: Max = {max_rhat:.3f} > 1.1 - NOT CONVERGED")
        
        if min_ess > 100:
            lines.append(f"ESS: All > 100 (min = {min_ess:.0f}) - ADEQUATE")
        else:
            lines.append(f"ESS: Min = {min_ess:.0f} < 100 - MORE SAMPLES NEEDED")

        for ln in lines:
            log.info("%s", ln)
    
    def save_report(self, filepath: Path):
        """
        Save diagnostics report to JSON.
        
        Parameters
        ----------
        filepath : Path or str
            Output file path
        """
        if self.Rhat is None:
            self.compute_all()
        
        report = {
            'n_chains': self.n_chains,
            'n_samples': self.n_samples,
            'param_names': self.param_names,
            'Rhat': self.Rhat.tolist(),
            'ESS': self.ESS.tolist(),
            'mean': self.mean.tolist(),
            'std': self.std.tolist(),
            'quantiles': {k: v.tolist() for k, v in self.quantiles.items()},
            # ★ 5) NaN処理: single chain のとき R-hat は NaN になるため nanmax/nanmin を使用
            'converged': bool(np.nanmax(self.Rhat) < 1.1) if self.Rhat is not None else False,
            'adequate_ess': bool(np.nanmin(self.ESS) > 100) if self.ESS is not None else False,
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)


def compute_rhat_simple(chains: List[np.ndarray]) -> np.ndarray:
    """
    Standalone R-hat computation (same as MCMCDiagnostics.compute_rhat).
    
    Parameters
    ----------
    chains : List[np.ndarray]
        List of MCMC chains
        
    Returns
    -------
    Rhat : ndarray
        R-hat values
    """
    chains = [np.asarray(c) for c in chains]
    m = len(chains)
    n = chains[0].shape[0]
    p = chains[0].shape[1]
    
    arr = np.stack(chains, axis=0)
    chain_means = arr.mean(axis=1)
    overall_mean = arr.mean(axis=(0, 1))
    
    B = n / (m - 1) * np.sum((chain_means - overall_mean) ** 2, axis=0)
    W = np.mean(np.var(arr, axis=1, ddof=1), axis=0)
    var_hat = (n - 1) / n * W + B / n
    
    return np.sqrt(var_hat / (W + 1e-10))


def compute_ess_simple(chains: List[np.ndarray], max_lag: int = 200) -> np.ndarray:
    """
    Standalone ESS computation.
    
    Parameters
    ----------
    chains : List[np.ndarray]
        List of MCMC chains
    max_lag : int
        Maximum lag for autocorrelation
        
    Returns
    -------
    ESS : ndarray
        Effective sample size
    """
    arr = np.concatenate(chains, axis=0)
    N, p = arr.shape
    ess = np.zeros(p)
    
    for j in range(p):
        x = arr[:, j] - arr[:, j].mean()
        acov = np.correlate(x, x, mode='full')
        acov = acov[acov.size // 2:][:max_lag + 1]
        acov /= acov[0] + 1e-10
        
        t, s = 1, 0.0
        while t + 1 <= max_lag:
            if acov[t] + acov[t + 1] < 0:
                break
            s += acov[t] + acov[t + 1]
            t += 2
        
        ess[j] = N / (1 + 2 * s)
    
    return ess


__all__ = [
    'MCMCDiagnostics',
    'compute_rhat_simple',
    'compute_ess_simple',
]
