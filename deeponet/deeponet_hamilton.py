#!/usr/bin/env python3
"""
DeepONet surrogate for 5-species Hamilton ODE.

Architecture:
  Branch net: θ (20) → hidden → R^p  (encodes parameter dependence)
  Trunk net:  t (1)  → hidden → R^p  (encodes time dependence)
  Output: φ_i(t; θ) = sum_k branch_k(θ) * trunk_k(t) + bias  (for each species i)

We train 5 independent output heads (one per species) sharing the trunk.

Usage:
  # Train
  python deeponet_hamilton.py train --data data/train_Dysbiotic_HOBIC_N10000.npz

  # Evaluate
  python deeponet_hamilton.py eval --checkpoint checkpoints/best.eqx --data data/train_Dysbiotic_HOBIC_N10000.npz

  # Benchmark vs ODE solver
  python deeponet_hamilton.py benchmark --checkpoint checkpoints/best.eqx --data data/train_Dysbiotic_HOBIC_N10000.npz
"""

import argparse
import time
from pathlib import Path
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax


# ============================================================
# Model
# ============================================================

class BranchNet(eqx.Module):
    """Branch network: θ (20) → R^(p * n_species)."""
    layers: list

    def __init__(self, in_dim: int, hidden: int, p: int, n_species: int, *, key):
        keys = jr.split(key, 4)
        out_dim = p * n_species
        self.layers = [
            eqx.nn.Linear(in_dim, hidden, key=keys[0]),
            eqx.nn.Linear(hidden, hidden, key=keys[1]),
            eqx.nn.Linear(hidden, hidden, key=keys[2]),
            eqx.nn.Linear(hidden, out_dim, key=keys[3]),
        ]

    def __call__(self, theta):
        x = theta
        for layer in self.layers[:-1]:
            x = jax.nn.gelu(layer(x))
        return self.layers[-1](x)


class TrunkNet(eqx.Module):
    """Trunk network: t (1) → R^(p * n_species)."""
    layers: list

    def __init__(self, p: int, n_species: int, hidden: int, *, key):
        keys = jr.split(key, 4)
        out_dim = p * n_species
        self.layers = [
            eqx.nn.Linear(1, hidden, key=keys[0]),
            eqx.nn.Linear(hidden, hidden, key=keys[1]),
            eqx.nn.Linear(hidden, hidden, key=keys[2]),
            eqx.nn.Linear(hidden, out_dim, key=keys[3]),
        ]

    def __call__(self, t):
        x = jnp.atleast_1d(t)
        for layer in self.layers[:-1]:
            x = jax.nn.gelu(layer(x))
        return self.layers[-1](x)


class DeepONet(eqx.Module):
    """DeepONet for multi-output (5 species)."""
    branch: BranchNet
    trunk: TrunkNet
    bias: jnp.ndarray
    p: int
    n_species: int

    def __init__(self, theta_dim: int = 20, n_species: int = 5,
                 p: int = 64, hidden: int = 128, *, key):
        k1, k2 = jr.split(key)
        self.branch = BranchNet(theta_dim, hidden, p, n_species, key=k1)
        self.trunk = TrunkNet(p, n_species, hidden, key=k2)
        self.bias = jnp.zeros(n_species)
        self.p = p
        self.n_species = n_species

    def __call__(self, theta, t):
        """
        Predict φ(t; θ) for a single (θ, t) pair.

        Args:
            theta: (20,) parameter vector
            t: scalar, time point

        Returns:
            (5,) predicted species fractions
        """
        b = self.branch(theta)  # (p * n_species,)
        tr = self.trunk(t)      # (p * n_species,)

        b = b.reshape(self.n_species, self.p)   # (5, p)
        tr = tr.reshape(self.n_species, self.p)  # (5, p)

        # Dot product per species
        out = jnp.sum(b * tr, axis=1) + self.bias  # (5,)
        return out

    def predict_trajectory(self, theta, t_grid):
        """
        Predict full trajectory φ(t; θ) for all time points.

        Args:
            theta: (20,)
            t_grid: (T,)

        Returns:
            (T, 5) predicted species fractions
        """
        return jax.vmap(lambda t: self(theta, t))(t_grid)


# ============================================================
# Data loading & normalization
# ============================================================

def load_data(path: str):
    """Load training data and compute normalization stats."""
    data = np.load(path)
    theta = data["theta"]   # (N, 20)
    phi = data["phi"]       # (N, T, 5)
    t = data["t"]           # (T,)
    bounds = data["bounds"]  # (20, 2)

    # Normalize theta to [0, 1]
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    width = hi - lo
    width = np.where(width < 1e-12, 1.0, width)  # avoid div by 0 for locked params
    theta_norm = (theta - lo) / width

    # Normalize t to [0, 1]
    t_min, t_max = t.min(), t.max()
    t_norm = (t - t_min) / (t_max - t_min + 1e-12)

    stats = {
        "theta_lo": lo, "theta_hi": hi, "theta_width": width,
        "t_min": t_min, "t_max": t_max,
    }

    return (
        jnp.array(theta_norm, dtype=jnp.float32),
        jnp.array(phi, dtype=jnp.float32),
        jnp.array(t_norm, dtype=jnp.float32),
        stats,
    )


# ============================================================
# Training
# ============================================================

@eqx.filter_jit
def loss_fn(model, theta_batch, phi_batch, t_grid):
    """
    MSE loss over a batch of trajectories.

    Args:
        theta_batch: (B, 20)
        phi_batch: (B, T, 5)
        t_grid: (T,)
    """
    def single_loss(theta, phi_true):
        phi_pred = model.predict_trajectory(theta, t_grid)  # (T, 5)
        return jnp.mean((phi_pred - phi_true) ** 2)

    losses = jax.vmap(single_loss)(theta_batch, phi_batch)
    return jnp.mean(losses)


@eqx.filter_jit
def train_step(model, opt_state, theta_batch, phi_batch, t_grid, optimizer):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(
        model, theta_batch, phi_batch, t_grid
    )
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


def train(
    data_path: str,
    n_epochs: int = 500,
    batch_size: int = 256,
    lr: float = 1e-3,
    p: int = 64,
    hidden: int = 128,
    val_frac: float = 0.1,
    seed: int = 0,
    checkpoint_dir: str = "checkpoints",
):
    print(f"Loading data from {data_path}...")
    theta, phi, t_grid, stats = load_data(data_path)
    N = theta.shape[0]
    T = t_grid.shape[0]
    print(f"  N={N}, T={T}, theta_dim={theta.shape[1]}, n_species={phi.shape[2]}")

    # Train/val split
    n_val = max(1, int(N * val_frac))
    n_train = N - n_val
    idx = jr.permutation(jr.PRNGKey(seed), N)
    train_idx, val_idx = idx[:n_train], idx[n_train:]

    theta_train, phi_train = theta[train_idx], phi[train_idx]
    theta_val, phi_val = theta[val_idx], phi[val_idx]

    print(f"  Train: {n_train}, Val: {n_val}")

    # Model
    key = jr.PRNGKey(seed)
    model = DeepONet(
        theta_dim=theta.shape[1],
        n_species=phi.shape[2],
        p=p,
        hidden=hidden,
        key=key,
    )

    n_params = sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
    print(f"  Model params: {n_params:,}")

    # Optimizer
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-5,
        peak_value=lr,
        warmup_steps=100,
        decay_steps=n_epochs * (n_train // batch_size + 1),
        end_value=1e-6,
    )
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # Checkpoint
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    print(f"\nTraining for {n_epochs} epochs...")
    t0 = time.time()

    for epoch in range(n_epochs):
        # Shuffle
        perm = jr.permutation(jr.PRNGKey(epoch), n_train)
        n_batches = n_train // batch_size

        epoch_loss = 0.0
        for i in range(n_batches):
            batch_idx = perm[i * batch_size : (i + 1) * batch_size]
            tb = theta_train[batch_idx]
            pb = phi_train[batch_idx]
            model, opt_state, loss = train_step(
                model, opt_state, tb, pb, t_grid, optimizer
            )
            epoch_loss += float(loss)

        epoch_loss /= max(n_batches, 1)
        train_losses.append(epoch_loss)

        # Validation (full batch)
        val_loss = float(loss_fn(model, theta_val, phi_val, t_grid))
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            eqx.tree_serialise_leaves(str(ckpt_dir / "best.eqx"), model)
            marker = " *"
        else:
            marker = ""

        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch+1:4d}/{n_epochs}  "
                  f"train={epoch_loss:.6f}  val={val_loss:.6f}  "
                  f"best={best_val_loss:.6f}  [{elapsed:.0f}s]{marker}")

    # Save final + losses
    eqx.tree_serialise_leaves(str(ckpt_dir / "final.eqx"), model)
    np.savez(
        ckpt_dir / "training_history.npz",
        train_losses=np.array(train_losses),
        val_losses=np.array(val_losses),
    )

    # Save normalization stats
    np.savez(ckpt_dir / "norm_stats.npz", **{k: np.array(v) for k, v in stats.items()})

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s. Best val loss: {best_val_loss:.6f}")
    print(f"Checkpoints saved to {ckpt_dir}/")

    return model, stats


# ============================================================
# Evaluation
# ============================================================

def evaluate(checkpoint: str, data_path: str):
    """Evaluate model on test data."""
    theta, phi, t_grid, stats = load_data(data_path)
    N = theta.shape[0]

    key = jr.PRNGKey(0)
    model = DeepONet(theta_dim=20, n_species=5, key=key)
    model = eqx.tree_deserialise_leaves(checkpoint, model)

    # Per-species error
    @jax.jit
    def compute_errors(theta_all, phi_all, t_grid):
        def single(theta, phi_true):
            phi_pred = model.predict_trajectory(theta, t_grid)
            mse = jnp.mean((phi_pred - phi_true) ** 2, axis=0)  # (5,)
            mae = jnp.mean(jnp.abs(phi_pred - phi_true), axis=0)
            rel = jnp.mean(jnp.abs(phi_pred - phi_true) / (jnp.abs(phi_true) + 1e-6), axis=0)
            return mse, mae, rel
        return jax.vmap(single)(theta_all, phi_all)

    mse, mae, rel = compute_errors(theta, phi, t_grid)

    species = ["S.oralis", "A.naeslundii", "V.dispar", "F.nucleatum", "P.gingivalis"]
    print("\nPer-species errors (mean over dataset):")
    print(f"{'Species':<15} {'MSE':>10} {'MAE':>10} {'Rel.Err':>10}")
    print("-" * 47)
    for i, sp in enumerate(species):
        print(f"{sp:<15} {float(mse[:, i].mean()):>10.6f} "
              f"{float(mae[:, i].mean()):>10.6f} "
              f"{float(rel[:, i].mean()):>10.4f}")

    print(f"\n{'Overall':<15} {float(mse.mean()):>10.6f} "
          f"{float(mae.mean()):>10.6f} "
          f"{float(rel.mean()):>10.4f}")


# ============================================================
# Benchmark: DeepONet vs ODE solver speed
# ============================================================

def benchmark(checkpoint: str, data_path: str, n_bench: int = 1000):
    """Compare inference speed: DeepONet vs numba ODE solver."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "tmcmc" / "program2602"))
    sys.path.insert(0, str(Path(__file__).parent.parent / "tmcmc"))
    from improved_5species_jit import BiofilmNewtonSolver5S

    data = np.load(data_path)
    theta_raw = data["theta"][:n_bench]
    bounds = data["bounds"]

    # Load DeepONet
    key = jr.PRNGKey(0)
    model = DeepONet(theta_dim=20, n_species=5, key=key)
    model = eqx.tree_deserialise_leaves(checkpoint, model)

    # Normalize for DeepONet
    lo, hi = bounds[:, 0], bounds[:, 1]
    width = np.where(hi - lo < 1e-12, 1.0, hi - lo)
    theta_norm = jnp.array((theta_raw - lo) / width, dtype=jnp.float32)
    t_norm = jnp.linspace(0, 1, 100, dtype=jnp.float32)

    # Warmup DeepONet
    _ = model.predict_trajectory(theta_norm[0], t_norm)

    # Batch predict with vmap
    @jax.jit
    def batch_predict(thetas):
        return jax.vmap(lambda th: model.predict_trajectory(th, t_norm))(thetas)

    _ = batch_predict(theta_norm[:10])  # JIT warmup

    # Benchmark DeepONet
    t0 = time.time()
    _ = batch_predict(theta_norm)
    jax.block_until_ready(_)
    t_don = time.time() - t0

    # Benchmark ODE solver
    solver = BiofilmNewtonSolver5S(
        dt=1e-5, maxtimestep=500, eps=1e-6, Kp1=1e-4,
        c_const=100.0, alpha_const=100.0, phi_init=0.2,
        K_hill=0.05, n_hill=4.0, max_newton_iter=50, use_numba=True,
    )
    # Warmup
    solver.run_deterministic(theta_raw[0])

    n_ode = min(n_bench, 200)  # ODE is slow, limit
    t0 = time.time()
    for i in range(n_ode):
        solver.run_deterministic(theta_raw[i])
    t_ode = time.time() - t0

    don_per = t_don / n_bench
    ode_per = t_ode / n_ode
    speedup = ode_per / don_per

    print(f"\nBenchmark ({n_bench} DeepONet / {n_ode} ODE samples):")
    print(f"  DeepONet:  {t_don:.3f}s total, {don_per*1000:.3f} ms/sample")
    print(f"  ODE solver: {t_ode:.3f}s total, {ode_per*1000:.3f} ms/sample")
    print(f"  Speedup: {speedup:.0f}x")


# ============================================================
# Visualization
# ============================================================

def plot_results(checkpoint: str, data_path: str, n_examples: int = 6):
    """Plot predicted vs true trajectories."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    theta, phi, t_grid, stats = load_data(data_path)

    key = jr.PRNGKey(0)
    model = DeepONet(theta_dim=20, n_species=5, key=key)
    model = eqx.tree_deserialise_leaves(checkpoint, model)

    species = ["S.o", "A.n", "V.d", "F.n", "P.g"]
    colors = ["#1f77b4", "#2ca02c", "#d4a017", "#9467bd", "#d62728"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()

    rng = np.random.default_rng(42)
    idx = rng.choice(len(theta), n_examples, replace=False)

    for ax_i, sample_i in enumerate(idx):
        ax = axes[ax_i]
        th = theta[sample_i]
        phi_true = np.array(phi[sample_i])  # (T, 5)
        phi_pred = np.array(model.predict_trajectory(th, t_grid))  # (T, 5)
        t_plot = np.array(t_grid)

        for s in range(5):
            ax.plot(t_plot, phi_true[:, s], "-", color=colors[s], alpha=0.7)
            ax.plot(t_plot, phi_pred[:, s], "--", color=colors[s], alpha=0.9)
        ax.set_title(f"Sample {sample_i}", fontsize=10)
        ax.set_xlabel("t (normalized)")
        ax.set_ylabel("φ")

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = []
    for s in range(5):
        legend_elements.append(Line2D([0], [0], color=colors[s], label=species[s]))
    legend_elements.append(Line2D([0], [0], color="gray", ls="-", label="True"))
    legend_elements.append(Line2D([0], [0], color="gray", ls="--", label="DeepONet"))
    fig.legend(handles=legend_elements, loc="lower center", ncol=7, fontsize=9)

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    out_path = Path(checkpoint).parent / "prediction_examples.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_path}")
    plt.close()

    # Training curve
    history_path = Path(checkpoint).parent / "training_history.npz"
    if history_path.exists():
        h = np.load(history_path)
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.semilogy(h["train_losses"], label="Train")
        ax2.semilogy(h["val_losses"], label="Val")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("MSE Loss")
        ax2.legend()
        ax2.set_title("Training History")
        fig2.savefig(Path(checkpoint).parent / "training_curve.png", dpi=150)
        print(f"Saved training curve")
        plt.close()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="DeepONet for Hamilton ODE")
    sub = parser.add_subparsers(dest="command")

    # Train
    p_train = sub.add_parser("train")
    p_train.add_argument("--data", required=True)
    p_train.add_argument("--epochs", type=int, default=500)
    p_train.add_argument("--batch-size", type=int, default=256)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--p", type=int, default=64, help="DeepONet basis dim")
    p_train.add_argument("--hidden", type=int, default=128)
    p_train.add_argument("--seed", type=int, default=0)
    p_train.add_argument("--checkpoint-dir", default="checkpoints")

    # Eval
    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--checkpoint", required=True)
    p_eval.add_argument("--data", required=True)

    # Benchmark
    p_bench = sub.add_parser("benchmark")
    p_bench.add_argument("--checkpoint", required=True)
    p_bench.add_argument("--data", required=True)
    p_bench.add_argument("--n-bench", type=int, default=1000)

    # Plot
    p_plot = sub.add_parser("plot")
    p_plot.add_argument("--checkpoint", required=True)
    p_plot.add_argument("--data", required=True)

    args = parser.parse_args()

    if args.command == "train":
        train(
            data_path=args.data,
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            p=args.p,
            hidden=args.hidden,
            seed=args.seed,
            checkpoint_dir=args.checkpoint_dir,
        )
    elif args.command == "eval":
        evaluate(args.checkpoint, args.data)
    elif args.command == "benchmark":
        benchmark(args.checkpoint, args.data, args.n_bench)
    elif args.command == "plot":
        plot_results(args.checkpoint, args.data)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
