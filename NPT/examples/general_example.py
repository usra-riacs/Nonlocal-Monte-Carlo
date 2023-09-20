import numpy as np
from scipy.sparse import csr_matrix
import sys

sys.path.append('../')  # Add path to access NPT_Github folder modules
from npt import NPT
from apt_preprocessor import APT_preprocessor  # Assuming the name of the file is apt_preprocessor.py


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

    # Begin preprocessing with APT
    print("\n[INFO] Starting APT preprocessing...")

    # create an APT_preprocessor instance
    apt_prep = APT_preprocessor(J.copy(), h.copy())

    # run Adaptive Parallel Tempering preprocessing
    """
    :param num_sweeps_MCMC: An integer representing the number of MCMC sweeps in each RNG chain (default: 1000).
    :param num_sweeps_read: An integer representing the number of last sweeps to read (default: 1000).
    :param num_rng: An integer representing the number of independent MCMC chains  (default: 100).
    :param beta_start: Initial beta value (default: 0.5). (chosen such that the largest spins can flip often enough)
    :param alpha: alpha value (default: 1.25). (defines the separation of beta in the schedule)
    :param sigma_E_val: initial energy STD value (default: 1000). (set it to large value)
    :param beta_max: Maximum beta value (default: 30). (maximum beta allowed in the schedule)
    :param use_hash_table: Whether to use a hash table or not (default =0).
    :param num_cores: How many CPU cores to use in parallel (default =8).
    """
    beta, sigma = apt_prep.run(num_sweeps_MCMC=1000, num_sweeps_read=1000, num_rng=100,
                               beta_start=0.5, alpha=1.25, sigma_E_val=1000, beta_max=64, use_hash_table=0, num_cores=8)

    print("\n[INFO] APT preprocessing complete.")

    beta_list = np.array(beta)
    print(f"[INFO] Beta List: {beta_list}")

    num_replicas = beta_list.shape[0]
    print(f"[INFO] Number of replicas: {num_replicas}")

    # Define APT parameters (see run method for detailed docstring)
    num_sweeps_MCMC = int(1e4)
    num_sweeps_read = int(1e2)
    num_swap_attempts = int(1e1)
    num_swapping_pairs = round(0.3 * num_replicas)
    use_hash_table = False
    num_cores = 8

    # Define NMC parameters  (see run method for detailed docstring)
    doNMC = [False] * (num_replicas - 5) + [True] * 5
    num_cycles = 10
    full_update_frequency = 1
    M_skip = 1
    temp_x = 20
    global_beta = 1 / 0.366838 * 5
    lambda_start = 3
    lambda_end = 0.01
    lambda_reduction_factor = 0.9
    threshold_initial = 0.9999999
    threshold_cutoff = 0.999999
    max_iterations = 100
    tolerance = np.finfo(float).eps

    # Create an NPT instance
    npt = NPT(J.toarray(), h)

    # Initiate the main NPT run
    print("\n[INFO] Starting main NPT process...")

    # Run NPT process
    M, Energy = npt.run(
        beta_list=beta_list,
        num_replicas=num_replicas,
        doNMC=doNMC,
        num_sweeps_MCMC=num_sweeps_MCMC,
        num_sweeps_read=num_sweeps_read,
        num_swap_attempts=num_swap_attempts,
        num_swapping_pairs=num_swapping_pairs,
        num_cycles=num_cycles,
        full_update_frequency=full_update_frequency,
        M_skip=M_skip,
        temp_x=temp_x,
        global_beta=global_beta,
        lambda_start=lambda_start,
        lambda_end=lambda_end,
        lambda_reduction_factor=lambda_reduction_factor,
        threshold_initial=threshold_initial,
        threshold_cutoff=threshold_cutoff,
        max_iterations=max_iterations,
        tolerance=tolerance,
        use_hash_table=use_hash_table,
        num_cores=num_cores
    )

    print(Energy)

    print("\n[INFO] NPT process complete.")


if __name__ == '__main__':
    main()
