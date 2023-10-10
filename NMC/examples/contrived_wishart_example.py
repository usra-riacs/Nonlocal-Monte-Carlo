import numpy as np
from scipy.sparse import csr_matrix
import sys

sys.path.append('../')  # Add path to access NMC_Github folder modules
from nmc import NMC

def txt_to_A_wishart_contrived_tree(txtfile):
    """
    Convert a txt file into matrices J and h.

    The txt file should contain data in the format:
    spin1 spin2 value

    If spin1 equals spin2, the value is assigned to h (bias).
    Otherwise, it's assigned to J (interaction).

    Parameters:
    - txtfile (str): Path to the txt file.

    Returns:
    - tuple: Tuple containing the matrix J and the vector h.
    """

    W = {}
    h_values = {}

    # Parse the txt file and populate the dictionary with interaction values
    with open(txtfile, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            x = list(map(float, line.split()))
            if int(x[0]) == int(x[1]):
                h_values[int(x[0])] = x[2]
            else:
                W[(int(x[0]), int(x[1]))] = x[2]
                W[(int(x[1]), int(x[0]))] = x[2]

    N = max(max(W.keys())) + 1
    h = np.zeros((N, 1))
    W_matrix = np.zeros((N, N))

    # Assign the values from the h_values dictionary to the h vector
    for i, value in h_values.items():
        h[i] = value

    # Convert the dictionary into a 2D matrix for J
    for (i, j), value in W.items():
        W_matrix[i, j] = value

    # Convert the matrix to a sparse format for efficient storage and computations
    W_sparse = csr_matrix(W_matrix)

    return W_sparse, h



def main():
    size_wishart = 10  # wishart instance size
    alpha_wishart = 0.2  # alpha parameter for wishart instance
    instance = 1  # index of wishart instance
    txtfile = f'./contrived_wishart_backbone/wishart_contrived_trees/wishart_planting_N_{size_wishart}_alpha_{alpha_wishart:.2f}_contrived_tree/wishart_planting_N_{size_wishart}_alpha_{alpha_wishart:.2f}_inst_{instance}_contrived_tree.txt'
    J, h = txt_to_A_wishart_contrived_tree(txtfile)
    J = -J  # match the sign of Hamiltonian
    h = -h

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
    threshold_initial = 0.9999
    threshold_cutoff = 0.999
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
