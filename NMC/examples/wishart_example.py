import numpy as np
from scipy.sparse import csr_matrix
import sys

sys.path.append('../')  # Add path to access NMC_Github folder modules
from nmc import NMC

def txt_to_A_wishart(txtfile):
    """
    Convert a txt file into matrices J and h.

    The txt file should contain data in the format:
    spin1 spin2 value

    If spin1 not equals spin2, the value is assigned to J (interaction).
    There is no h bias for wishart instances, so h is assigned to 0.

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
    size_wishart = 10  # wishart instance size
    alpha_wishart = 0.5  # alpha parameter for wishart instance
    instance = 1  # index of wishart instance
    txtfile = f'./wishart_small/wishart_planting_N_{size_wishart}_alpha_{alpha_wishart:.2f}/wishart_planting_N_{size_wishart}_alpha_{alpha_wishart:.2f}_inst_{instance}.txt'
    J, h = txt_to_A_wishart(txtfile)
    J = -J  # match the sign of Hamiltonian

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
