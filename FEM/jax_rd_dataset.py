import jax
import jax.numpy as jnp
from pathlib import Path

from jax_rd_nd_demo import solve_0d, solve_nd


jax.config.update("jax_enable_x64", True)


def make_dataset_0d(out_dir: Path):
    uT, t_arr, traj = solve_0d()
    out_dir.mkdir(parents=True, exist_ok=True)
    jnp.save(out_dir / "t_0d.npy", t_arr)
    jnp.save(out_dir / "u_0d_traj.npy", traj)


def make_dataset_1d(out_dir: Path, N=64):
    T = 0.1
    dt = 1e-4
    D = 0.01
    k = 1.0
    s0 = 10.0
    uT, traj = solve_nd(dim=1, N=N, T=T, dt=dt, D=D, k=k, s0=s0)
    x = jnp.linspace(0.0, 1.0, N)
    n_steps = traj.shape[0]
    t = jnp.linspace(0.0, T, n_steps)
    out_dir.mkdir(parents=True, exist_ok=True)
    jnp.save(out_dir / "x_1d.npy", x)
    jnp.save(out_dir / "t_1d.npy", t)
    jnp.save(out_dir / "u_1d_traj.npy", traj)


def make_dataset_2d(out_dir: Path, N=32):
    T = 0.05
    dt = 5e-5
    D = 0.01
    k = 1.0
    s0 = 10.0
    uT, traj = solve_nd(dim=2, N=N, T=T, dt=dt, D=D, k=k, s0=s0)
    x = jnp.linspace(0.0, 1.0, N)
    y = jnp.linspace(0.0, 1.0, N)
    n_steps = traj.shape[0]
    t = jnp.linspace(0.0, T, n_steps)
    out_dir.mkdir(parents=True, exist_ok=True)
    jnp.save(out_dir / "x_2d.npy", x)
    jnp.save(out_dir / "y_2d.npy", y)
    jnp.save(out_dir / "t_2d.npy", t)
    jnp.save(out_dir / "u_2d_traj.npy", traj)


def make_dataset_3d(out_dir: Path, N=16):
    T = 0.03
    dt = 5e-5
    D = 0.01
    k = 1.0
    s0 = 10.0
    uT, traj = solve_nd(dim=3, N=N, T=T, dt=dt, D=D, k=k, s0=s0)
    x = jnp.linspace(0.0, 1.0, N)
    y = jnp.linspace(0.0, 1.0, N)
    z = jnp.linspace(0.0, 1.0, N)
    n_steps = traj.shape[0]
    t = jnp.linspace(0.0, T, n_steps)
    out_dir.mkdir(parents=True, exist_ok=True)
    jnp.save(out_dir / "x_3d.npy", x)
    jnp.save(out_dir / "y_3d.npy", y)
    jnp.save(out_dir / "z_3d.npy", z)
    jnp.save(out_dir / "t_3d.npy", t)
    jnp.save(out_dir / "u_3d_traj.npy", traj)


def main():
    base = Path("_jax_rd_datasets")
    make_dataset_0d(base)
    make_dataset_1d(base)
    make_dataset_2d(base)
    make_dataset_3d(base)
    print("Saved datasets to", base)


if __name__ == "__main__":
    main()

