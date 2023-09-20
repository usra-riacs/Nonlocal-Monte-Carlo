import numpy as np
from scipy.sparse import csr_matrix
import sys

sys.path.append('../')  # Add path to access NMC_Github folder modules
from nmc import NMC

def generate_random_J_h(N):
    """
    Generate random J (adjacency) and h matrices for a given size.

    :param N: Size of the square matrix.
    :return: Tuple containing J and h matrices.
    """
    # Generate h as a random vector of size N
    h = np.random.randn(N, 1)

    # Generate a symmetric random matrix for J
    upper_triangle_indices = np.triu_indices(N, 1)
    upper_triangle_values = np.random.randn(len(upper_triangle_indices[0]))
    J = np.zeros((N, N))
    J[upper_triangle_indices] = upper_triangle_values
    J += J.T  # Make it symmetric

    return csr_matrix(J), h


def main():
    N = 10  # Size of the random J matrix
    J, h = generate_random_J_h(N)

    # Create an instance of NMC class
    nmc_instance = NMC(J.toarray(), h)

    # Set NMC hyperparameters (check the run method for detailed docstring of the parameters)
    num_sweeps_initial = int(1e4)
    num_sweeps_per_NMC_phase = int(1e4)
    num_NMC_cycles = 10
    full_update_frequency = 1
    M_skip = 1
    temp_x = 20
    global_beta = 3
    lambda_start = 3
    lambda_end = 0.01
    lambda_reduction_factor = 0.9
    threshold_initial = 0.9999999
    threshold_cutoff = 0.999999
    max_iterations = 100
    tolerance = np.finfo(float).eps
    use_hash_table = False  # Flag to decide whether to use hash table

    # Initiate the main NMC run
    print("\n[INFO] Starting main NMC process...")
    M_overall, energy_overall, min_energy = nmc_instance.run(num_sweeps_initial, num_sweeps_per_NMC_phase,
                                                             num_NMC_cycles, full_update_frequency, M_skip, temp_x,
                                                             global_beta, lambda_start, lambda_end,
                                                             lambda_reduction_factor, threshold_initial,
                                                             threshold_cutoff,
                                                             max_iterations, tolerance, use_hash_table=use_hash_table)

    print(f"Minimum Energy: {min_energy:.8f}")
    print("\n[INFO] NMC process complete.")


if __name__ == '__main__':
    main()
