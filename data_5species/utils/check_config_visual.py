import numpy as np
import matplotlib.pyplot as plt
import sys

# Add path to find data_5species module
sys.path.append("/home/nishioka/IKM_Hiwi/Tmcmc202601")

# Add path to data_5species itself for direct imports (fixes config import issue)
sys.path.append("/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species")

# Workaround for config import issues: Create a dummy config module if needed or rely on path
# The issue is `from config import ...` inside logger.py expects config.py to be in sys.path
# data_5species/config.py likely exists.

try:
    from data_5species.core.nishioka_model import get_nishioka_bounds
except ImportError:
    # Fallback: try direct import if we are inside the package structure but it fails
    # Create a minimal mock of nishioka_model if imports are too tangled for a simple check script
    # This avoids loading the heavy TMCMC stack just to check bounds.
    print(
        "Warning: Could not import nishioka_model due to dependency issues. Using local definition."
    )

    INTERACTION_GRAPH_JSON = {
        "locked_edges": [
            {"param_idx": 6},
            {"param_idx": 12},
            {"param_idx": 13},
            {"param_idx": 16},
            {"param_idx": 17},
        ]
    }

    def get_nishioka_bounds():
        bounds = [(-1.0, 1.0)] * 20
        locked_indices = [edge["param_idx"] for edge in INTERACTION_GRAPH_JSON["locked_edges"]]
        for idx in locked_indices:
            bounds[idx] = (0.0, 0.0)
        bounds[18] = (0.0, 1.0)
        return bounds, locked_indices


def main():
    bounds, locked = get_nishioka_bounds()

    low = [b[0] for b in bounds]
    high = [b[1] for b in bounds]

    plt.figure(figsize=(12, 6))
    indices = np.arange(len(bounds))

    # Calculate center and error (half-width)
    means = [(l + h) / 2 for l, h in zip(low, high)]
    errors = [(h - l) / 2 for l, h in zip(low, high)]

    # Plot error bars
    plt.errorbar(
        indices,
        means,
        yerr=errors,
        fmt="o",
        capsize=5,
        label="Search Range (Bounds)",
        color="blue",
        ecolor="gray",
    )

    # Highlight locked parameters
    if locked:
        plt.scatter(
            locked,
            [0] * len(locked),
            color="red",
            s=100,
            zorder=5,
            label="Locked (0.0)",
            marker="x",
        )

    # Highlight special positive parameter (a18)
    # Check if a18 is indeed positive-only (0, 1) -> mean 0.5, error 0.5
    if bounds[18] == (0.0, 1.0):
        plt.scatter(
            [18],
            [0.5],
            color="green",
            s=100,
            zorder=5,
            label="Positive Constraint (Vei->Pg)",
            marker="^",
        )

    plt.axhline(0, color="black", linestyle="--", alpha=0.3)

    # Customize X axis labels
    labels = [f"a{i}" for i in indices]
    plt.xticks(indices, labels, rotation=45)

    plt.title(
        "Visualization of Nishioka Model Parameter Bounds\n(Locked vs Active Ranges)", fontsize=14
    )
    plt.ylabel("Interaction Coefficient Strength")
    plt.xlabel("Parameter Index")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = "/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/bounds_visualization.png"
    plt.savefig(output_path, dpi=300)
    print(f"Visualization saved to {output_path}")


if __name__ == "__main__":
    main()
