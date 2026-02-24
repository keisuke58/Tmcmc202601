import jax
import jax.numpy as jnp

from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.generate_mesh import rectangle_mesh, Mesh, get_meshio_cell_type
from jax_fem.utils import save_sol


class ReactionDiffusion(Problem):
    def __init__(self, diffusion_coeff, reaction_coeff, source_strength, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.diffusion_coeff = diffusion_coeff
        self.reaction_coeff = reaction_coeff
        self.source_strength = source_strength

    def get_tensor_map(self):
        D = self.diffusion_coeff

        def tensor_map(x):
            return jnp.eye(self.dim) * D

        return tensor_map

    def get_mass_map(self):
        k = self.reaction_coeff
        s = self.source_strength

        def mass_map(u, x):
            r2 = (x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2
            src = s * jnp.exp(-r2 / 0.02)
            val = k * u[0] - src
            return jnp.array([val])

        return mass_map


def build_problem(D, k, s, nx=40, ny=40):
    ele_type = "QUAD4"
    cell_type = get_meshio_cell_type(ele_type)
    Lx = 1.0
    Ly = 1.0
    nodes, cells = rectangle_mesh(Lx, Ly, nx, ny, ele_type)
    mesh = Mesh(nodes, cells, cell_type)
    num_state_vars = 1
    boundary_nodes = mesh.boundary_nodes
    dirichlet_val = jnp.zeros((boundary_nodes.shape[0], num_state_vars))
    problem = ReactionDiffusion(
        diffusion_coeff=D,
        reaction_coeff=k,
        source_strength=s,
        mesh=mesh,
        dirichlet_bc=[(boundary_nodes, dirichlet_val)],
        num_state_vars=num_state_vars,
    )
    return problem


def solve_forward(D, k, s, nx=40, ny=40):
    problem = build_problem(D, k, s, nx=nx, ny=ny)
    u0 = jnp.zeros((problem.num_total_nodes, problem.num_state_vars))
    u, _ = solver(problem, u0, jit=True)
    return problem, u


def loss_fn(D, k, s, target_points, target_values):
    problem, u = solve_forward(D, k, s)
    coords = problem.mesh.points
    losses = []
    for (x_tar, y_tar), v_tar in zip(target_points, target_values):
        d2 = jnp.sum((coords - jnp.array([x_tar, y_tar])) ** 2, axis=1)
        idx = jnp.argmin(d2)
        u_val = u[idx, 0]
        losses.append((u_val - v_tar) ** 2)
    return jnp.sum(jnp.stack(losses))


def main():
    D0 = 0.01
    k0 = 1.0
    s0 = 10.0
    target_points = jnp.array([[0.25, 0.25], [0.75, 0.75]])
    target_values = jnp.array([0.2, 0.1])
    grad_loss_D = jax.grad(lambda D: loss_fn(D, k0, s0, target_points, target_values))
    loss_value = loss_fn(D0, k0, s0, target_points, target_values)
    grad_value = grad_loss_D(D0)
    print("JAX-FEM reaction-diffusion demo")
    print(f"D = {float(D0):.6f}")
    print(f"loss(D) = {float(loss_value):.6f}")
    print(f"d loss / d D = {float(grad_value):.6f}")
    problem, u = solve_forward(D0, k0, s0)
    save_sol(problem, u, "jax_fem_reaction_diffusion")


if __name__ == "__main__":
    main()

