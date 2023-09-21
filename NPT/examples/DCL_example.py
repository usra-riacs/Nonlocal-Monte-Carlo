import numpy as np
from scipy.sparse import csr_matrix
import sys

sys.path.append('../')  # Add path to access NPT_Github folder modules
from npt import NPT
from apt_preprocessor import APT_preprocessor  # Assuming the name of the file is apt_preprocessor.py

def txt_to_A_DCL(txtfile):
    """
    Convert a txt file into matrices J and h.

    The txt file should contain data in the format:
    spin1 spin2 value

    If spin1 not equals spin2, the value is assigned to J (interaction).
    There is no h bias for DCL instances, so h is assigned to 0.

    Parameters:
    - txtfile (str): Path to the txt file.

    Returns:
    - tuple: Tuple containing sparse matrix J and h matrices.
    """
    W = {}

    with open(txtfile, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            x = list(map(float, line.split()))
            if int(x[0]) == int(x[1]):
                continue
            W[(int(x[0]), int(x[1]))] = x[2]
            W[(int(x[1]), int(x[0]))] = x[2]

    N = max(max(W.keys())) + 1
    h = np.zeros((N, 1))
    W_matrix = np.zeros((N, N))

    for (i, j), value in W.items():
        W_matrix[i, j] = value

    W_sparse = csr_matrix(W_matrix)

    return W_sparse, h


def main():
    size_DCL = 8  # chimera instance size
    instance = 1  # index of chimera instance
    txtfile = f'./DCL_instances/C{size_DCL}/{instance:02}.txt'
    J, h = txt_to_A_DCL(txtfile)
    J = -J  # match the sign of Hamiltonian
    h = -h  # match the sign of Hamiltonian

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
