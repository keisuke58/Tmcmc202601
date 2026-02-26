"""
tmcmc/pbox.py

Helper module for p-box (probability box) computation and visualization.
Focuses on comparing:
1. Prior box (bounds)
2. Posterior box (min-max range or credible interval of samples)
3. True value (if known)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


def compute_pbox_bounds(
    samples: np.ndarray, quantiles: Optional[Tuple[float, float]] = None
) -> np.ndarray:
    """
    Compute bounds for p-box.
    If quantiles is None, returns [min, max].
    If quantiles is (q_low, q_high) e.g. (0.025, 0.975), returns [q_low, q_high].

    Parameters
    ----------
    samples : (N, D) or (N,) array
    quantiles : tuple or None

    Returns
    -------
    bounds : (D, 2) array of [low, high]
    """
    if samples.ndim == 1:
        samples = samples[:, None]

    if quantiles is None:
        low = np.min(samples, axis=0)
        high = np.max(samples, axis=0)
    else:
        q_low, q_high = quantiles
        low = np.percentile(samples, q_low * 100, axis=0)
        high = np.percentile(samples, q_high * 100, axis=0)

    return np.stack([low, high], axis=1)


def plot_pbox_comparison(
    pbox_posterior: np.ndarray,
    prior_bounds: List[Tuple[float, float]],
    param_names: List[str],
    theta_true: Optional[np.ndarray] = None,
    theta_map: Optional[np.ndarray] = None,
    filename: Optional[str] = None,
    title: str = "Parameter p-box Comparison",
    label_posterior: str = "Posterior Box (min-max)",
):
    """
    Plot p-box comparison (Prior Interval vs Posterior Interval).

    Parameters
    ----------
    pbox_posterior : (D, 2) array
        [min, max] for each parameter from posterior samples.
    prior_bounds : List[(min, max)]
        Prior bounds for each parameter.
    param_names : List[str]
        Parameter names.
    theta_true : (D,) array, optional
        True parameter values.
    theta_map : (D,) array, optional
        MAP parameter values.
    filename : str, optional
        If provided, save figure to this path.
    """
    n_params = len(param_names)

    fig, ax = plt.subplots(figsize=(8, n_params * 0.6 + 1.0))

    y = np.arange(n_params)[::-1]  # Top to bottom
    height = 0.3

    for i, (name, y_pos) in enumerate(zip(param_names, y)):
        # Prior box (Light gray)
        prior_min, prior_max = prior_bounds[i] if i < len(prior_bounds) else (0, 0)
        ax.barh(
            y_pos,
            prior_max - prior_min,
            left=prior_min,
            height=height,
            color="lightgray",
            edgecolor="gray",
            alpha=0.5,
            label="Prior Bounds" if i == 0 else "",
        )

        # Posterior box (Blue / Color)
        post_min, post_max = pbox_posterior[i]
        ax.barh(
            y_pos,
            post_max - post_min,
            left=post_min,
            height=height * 0.6,
            color="skyblue",
            edgecolor="blue",
            alpha=0.9,
            label=label_posterior if i == 0 else "",
        )

        # True value (Red line)
        if theta_true is not None:
            val = theta_true[i]
            ax.vlines(
                val,
                y_pos - height / 2,
                y_pos + height / 2,
                colors="red",
                linestyles="-",
                linewidth=2,
                label="True Value" if i == 0 else "",
            )

        # MAP value (Green line or similar, optional)
        if theta_map is not None:
            val = theta_map[i]
            ax.plot(val, y_pos, "o", color="green", markersize=5, label="MAP" if i == 0 else "")

    ax.set_yticks(y)
    ax.set_yticklabels(param_names)
    ax.set_xlabel("Parameter Value")
    ax.set_title(title)
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)

    # Unique legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="best")

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
