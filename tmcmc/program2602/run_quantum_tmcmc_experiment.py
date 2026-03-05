import numpy as np
import time
import logging
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tmcmc_5species_tsm import TMCMC_5Species_TSM

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MockDataLoader:
    def __init__(self):
        self.t_measure = np.linspace(0, 1.0, 10)
        self.y_measure = np.zeros((10, 12))  # 12D state
        self.y_measure[:, 0] = np.linspace(0, 1.0, 10)  # Dummy linear growth for component 0
        self.sigma_measure = np.ones_like(self.y_measure) * 0.1


def log_likelihood(y_pred, y_obs, sigma_obs):
    """Simple Gaussian Log-Likelihood"""
    diff = y_pred - y_obs
    ll = -0.5 * np.sum((diff / sigma_obs) ** 2)
    return ll


def run_experiment():
    logger.info("=== Starting Quantum TMCMC Experiment ===")

    # 1. Setup
    n_particles = 1000  # Simulate a population of 1000 particles
    n_params_active = 6  # Input dimension (Quantum circuit takes 6)
    n_params_total = 20  # Solver needs 20

    # Generate random particles (thetas)
    # Uniform prior [-1, 1] for demonstration
    # We create full 20D vectors, but only vary first 6
    thetas = np.zeros((n_particles, n_params_total))

    # Randomize active params
    thetas[:, :n_params_active] = np.random.uniform(-1.0, 1.0, (n_particles, n_params_active))

    # Set reasonable defaults for others (to avoid singular matrices in ODE)
    thetas[:, 3] = 0.1
    thetas[:, 4] = 0.1
    thetas[:, 8] = 0.1
    thetas[:, 9] = 0.1
    thetas[:, 15] = 0.1

    data_loader = MockDataLoader()

    # 2. Classical Run (Benchmark)
    logger.info(f"--- Running Classical Solver for {n_particles} particles ---")
    tsm_classical = TMCMC_5Species_TSM(data_loader, use_quantum_surrogate=False)

    start_time_c = time.time()

    # To save time, we might reduce n_particles for classical if it's too slow
    # But let's try 50 (ODE is ~1.2s, so 1000 would take 20 mins)
    # We will estimate full time based on a smaller sample
    n_sample_c = 10
    logger.info(
        f"Running partial Classical benchmark ({n_sample_c} samples) to estimate total time..."
    )

    for i in range(n_sample_c):
        t, y, s = tsm_classical.solve_tsm(thetas[i])

        # Interpolate classical result to match observation time points
        # y shape is (time, 12), t is (time,)
        # We need y at data_loader.t_measure
        y_interp = np.zeros((len(data_loader.t_measure), 12))
        for k in range(12):
            y_interp[:, k] = np.interp(data_loader.t_measure, t, y[:, k])

        _ = log_likelihood(y_interp, data_loader.y_measure, data_loader.sigma_measure)

    end_time_c = time.time()
    avg_time_c = (end_time_c - start_time_c) / n_sample_c
    estimated_total_c = avg_time_c * n_particles
    logger.info(f"Classical Avg Time: {avg_time_c:.4f} s/sample")
    logger.info(
        f"Estimated Total Time for {n_particles} particles: {estimated_total_c:.2f} s ({estimated_total_c/60:.2f} min)"
    )

    # 3. Quantum Run (Full Population)
    logger.info(f"--- Running Quantum Surrogate for {n_particles} particles ---")
    tsm_quantum = TMCMC_5Species_TSM(data_loader, use_quantum_surrogate=True)

    if not tsm_quantum.use_quantum_surrogate:
        logger.error(
            "Quantum Surrogate failed to initialize (missing weights?). Aborting quantum run."
        )
        return

    start_time_q = time.time()

    # Run ALL particles
    log_likelihoods = []
    for i in range(n_particles):
        t, y, s = tsm_quantum.solve_tsm(thetas[i])
        ll = log_likelihood(y, data_loader.y_measure, data_loader.sigma_measure)
        log_likelihoods.append(ll)

    end_time_q = time.time()
    total_time_q = end_time_q - start_time_q
    avg_time_q = total_time_q / n_particles

    logger.info(f"Quantum Total Time: {total_time_q:.4f} s")
    logger.info(f"Quantum Avg Time:   {avg_time_q:.6f} s/sample")

    # 4. Comparison
    speedup = avg_time_c / avg_time_q
    logger.info(f"=== RESULT: Quantum Speedup Factor: {speedup:.1f}x ===")

    # Verify meaningful output
    logger.info(f"Sample Log-Likelihoods (First 5): {log_likelihoods[:5]}")
    logger.info("Experiment Completed Successfully.")


if __name__ == "__main__":
    run_experiment()
