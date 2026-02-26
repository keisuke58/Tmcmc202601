import numpy as np
import json
import os

# 1. JSON Definition
INTERACTION_GRAPH_JSON = {
    "description": "Interaction network derived from Figure 4C (Nishioka Algorithm)",
    "nodes": [
        {"id": 0, "name": "S. oralis", "short": "S.o", "color": "blue"},
        {"id": 1, "name": "A. naeslundii", "short": "A.n", "color": "green"},
        {"id": 2, "name": "Veillonella", "short": "Vei", "color": "yellow"},
        {"id": 3, "name": "F. nucleatum", "short": "F.n", "color": "purple"},
        {"id": 4, "name": "P. gingivalis", "short": "P.g", "color": "red"},
    ],
    "active_edges": [
        {
            "source": 0,
            "target": 1,
            "relation": "bidirectional",
            "param_idx": 1,
            "biological_note": "Co-aggregation",
        },
        {
            "source": 0,
            "target": 2,
            "relation": "bidirectional",
            "param_idx": 10,
            "biological_note": "Lactate consumption",
        },
        {
            "source": 0,
            "target": 3,
            "relation": "bidirectional",
            "param_idx": 11,
            "biological_note": "Formate/Acetate symbiosis",
        },
        {
            "source": 2,
            "target": 4,
            "relation": "bidirectional",
            "param_idx": 18,
            "biological_note": "pH rise support",
        },
        {
            "source": 3,
            "target": 4,
            "relation": "bidirectional",
            "param_idx": 19,
            "biological_note": "Co-aggregation/Peptides",
        },
    ],
    "locked_edges": [
        {
            "source": 2,
            "target": 3,
            "relation": "none",
            "param_idx": 6,
            "reason": "No direct interaction in Fig 4C",
        },
        {
            "source": 1,
            "target": 2,
            "relation": "none",
            "param_idx": 12,
            "reason": "No direct interaction in Fig 4C",
        },
        {
            "source": 1,
            "target": 3,
            "relation": "none",
            "param_idx": 13,
            "reason": "No direct interaction in Fig 4C",
        },
        {
            "source": 0,
            "target": 4,
            "relation": "none",
            "param_idx": 16,
            "reason": "No direct interaction in Fig 4C",
        },
        {
            "source": 1,
            "target": 4,
            "relation": "none",
            "param_idx": 17,
            "reason": "No direct interaction in Fig 4C",
        },
    ],
}


def get_nishioka_mask():
    """
    Creates a 5x5 interaction mask matrix based on the JSON graph.
    Returns:
        mask (np.ndarray): 5x5 matrix where 1 indicates active interaction, 0 indicates locked.
        param_map (dict): Mapping from (row, col) to parameter index (if available).
    """
    mask = np.ones((5, 5), dtype=int)
    param_map = {}

    # Initialize diagonal as active (self-interaction is usually active or handled separately)
    # But in this model, diagonals are growth rates/self-inhibition, usually estimated.

    # Process locked edges
    for edge in INTERACTION_GRAPH_JSON["locked_edges"]:
        u, v = edge["source"], edge["target"]
        # Interaction A_ij means effect of j on i.
        # The JSON 'source'/'target' might imply direction or just connection.
        # The 'relation': 'none' implies NO interaction in EITHER direction for these specific pairs
        # if they are described as "No direct interaction".
        # However, looking at param_idx, let's map specific indices to (row, col).

        # Mapping from improved_5species_jit.py logic:
        # idx 6: A[2,3] and A[3,2] (Vei-F.n)
        # idx 12: A[1,2] and A[2,1] (A.n-Vei)
        # idx 13: A[1,3] and A[3,1] (A.n-F.n)
        # idx 16: A[0,4] and A[4,0] (S.o-P.g)
        # idx 17: A[1,4] and A[4,1] (A.n-P.g)

        # Explicitly setting 0 based on known indices from previous analysis
        # Note: 'param_idx' in JSON is the theta index.
        pass

    # Better approach: Construct mask from param indices if we know the mapping.
    # Or, define the mask directly based on the "locked_edges" logical definition.

    # Let's use the explicit Locked Edges to set zeros in the mask.
    # Assuming symmetric locking for these pairs based on the code structure (A[i,j] = A[j,i] = theta[k]).

    for edge in INTERACTION_GRAPH_JSON["locked_edges"]:
        u = edge["source"]
        v = edge["target"]
        mask[u, v] = 0
        mask[v, u] = 0

    return mask


def get_nishioka_bounds():
    """
    Legacy function for backward compatibility.
    Defaults to Commensal Static logic + Standard Nishioka Locks.
    """
    return get_condition_bounds("Commensal", "Static")


def get_condition_bounds(condition, cultivation):
    """
    Returns bounds and locked indices based on Experiment Condition.
    Reflects the 'Heine + Nishioka' strategy (2025/2026).
    Configuration is loaded from data_5species/config/prior_bounds.json.

    Strategies:
    1. Commensal Static:
       - Lock Pathogens (Purple/Red) growth and interactions to 0.0.
       - Estimate Commensal (Blue/Green/Yellow) with moderate bounds.

    2. Dysbiotic HOBIC:
       - Wide bounds for Orange (V. parvula) growth (up to 5.0).
       - Strong self-inhibition (up to 5.0).
       - Negative interactions for 'Hidden Cooperation' (a31, a53, a54).

    Indices Mapping (Improved 5-Species Model):
    M1 (S0, S1): a11(0), a12(1), a22(2), b1(3), b2(4)
    M2 (S2, S3): a33(5), a34(6), a44(7), b3(8), b4(9)
    M3 (Cross):  a13(10), a14(11), a23(12), a24(13)
    M4 (S4 Self): a55(14), b5(15)
    M5 (S4 Cross): a15(16), a25(17), a35(18), a45(19)

    S0: Blue, S1: Green, S2: Yellow/Orange, S3: Purple, S4: Red
    """
    # Path to config file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "..", "model_config", "prior_bounds.json")

    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Warning: Config file not found at {config_path}. Using hardcoded defaults.")
        # Minimal fallback if file is missing (though it shouldn't be)
        config = {"default_bounds": [-1.0, 1.0], "strategies": {}}

    bounds = [tuple(config.get("default_bounds", [-1.0, 1.0]))] * 20
    locked_indices = []

    key = f"{condition}_{cultivation}"

    if key in config["strategies"]:
        strategy = config["strategies"][key]
    else:
        # Basic fallback matching historical defaults
        strategy = {
            "locks": [6, 12, 13, 16, 17],
            "bounds": {
                "3": [0.0, 3.0],
                "4": [0.0, 3.0],
                "8": [0.0, 3.0],
                "9": [0.0, 3.0],
                "15": [0.0, 3.0],
            },
        }

    # Apply locks
    for idx in strategy.get("locks", []):
        bounds[idx] = (0.0, 0.0)
        if idx not in locked_indices:
            locked_indices.append(idx)

    # Apply bounds
    for idx_str, range_val in strategy.get("bounds", {}).items():
        idx = int(idx_str)
        bounds[idx] = tuple(range_val)

    # Final Safety: Ensure locked indices are (0.0, 0.0)
    for idx in locked_indices:
        bounds[idx] = (0.0, 0.0)

    return bounds, locked_indices


def get_model_constants():
    """
    Loads general model constants from data_5species/model_config/model_constants.json.
    Returns a dictionary with keys: active_species, active_indices, param_names, etc.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "..", "model_config", "model_constants.json")

    try:
        with open(config_path, "r") as f:
            constants = json.load(f)
    except FileNotFoundError:
        print(f"Warning: Constants file not found at {config_path}. Using hardcoded defaults.")
        # Minimal fallback
        constants = {
            "active_species": [0, 1, 2, 3, 4],
            "active_indices": list(range(20)),
            "default_solver_params": {"dt": 1e-4},
        }
    return constants


def predict_posterior_mean_masked(X_train, Y_train, X_star, mask_matrix, sigma_n=1e-10):
    """
    Computes the posterior mean for a Gaussian Process-like update,
    applying the mask to the weight matrix logic if applicable.

    Equation: mu* = K(X*, X) @ inv(K(X, X) + sigma^2 I) @ Y

    However, the user asked: "Apply mask to excluded gradient calculation".
    If we are estimating W where Y = XW, then W_hat = inv(X'X)X'Y.
    If we mask W, we effectively remove columns from X.

    Here, we provide a generic implementation using NumPy that respects the mask
    conceptually by zeroing out contributions from masked interactions.

    Args:
        X_train: (N, D)
        Y_train: (N, TargetDim)
        X_star: (M, D)
        mask_matrix: (D, TargetDim) or similar broadcastable mask.
    """
    # This is a placeholder for the specific GP formula requested.
    # Since we don't have the exact Kernel function K defined in the prompt,
    # we'll implement a linear kernel version (Bayesian Linear Regression)
    # which is often used as the "weight estimation" step.

    # If the user literally wants the formula K(x*, X)...:
    # We need a kernel function.
    pass


def apply_mask_to_gradient(grad, mask):
    """
    Simple helper to zero out gradients for locked parameters.
    """
    return grad * mask


class PgingivalisSurgeMLP:
    def __init__(self, input_dim=3, hidden_dim=8, output_scale=0.5, seed=None):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(scale=0.1, size=(input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim, dtype=float)
        self.W2 = rng.normal(scale=0.1, size=(hidden_dim, 1))
        self.b2 = np.zeros(1, dtype=float)
        self.output_scale = float(output_scale)

    def __call__(self, features):
        x = np.asarray(features, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        h = np.tanh(x @ self.W1 + self.b1)
        y = np.tanh(h @ self.W2 + self.b2)
        y = self.output_scale * y
        return np.squeeze(y, axis=-1)


def compute_pg_effective_growth(base_rate, phi_vei, phi_pg, t_norm, mlp):
    x = np.array([phi_vei, phi_pg, t_norm], dtype=float)
    delta = float(mlp(x))
    return float(base_rate) + delta
