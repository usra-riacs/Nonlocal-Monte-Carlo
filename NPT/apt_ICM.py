import random
import time
from concurrent.futures import ProcessPoolExecutor
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix
from cachetools import LRUCache
from random import randint


# np.random.seed(12624755)  # Set the seed to an arbitrary number

class APT_ICM:
    """
    The apt_ICM class is used to implement the Adaptive Parallel Tempering algorithm with Iso-cluster Move (ICM or Houdayer's move).
    """

    def __init__(self, J, h):
        """
        Initialize an AdaptiveParallelTempering object.
        :param J: A 2D numpy array representing the coupling matrix (weights J).
        :param h: A 1D numpy array or list representing the external field (biases h).
        """
        self.J = J

        # Convert h to a numpy array if it's a list
        if isinstance(h, list):
            h = np.array(h)

        # If h is a 1D array, reshape it to be a 2D column vector
        if len(h.shape) == 1:
            h = h[:, np.newaxis]
        self.h = h


    def replica_energy(self, M, num_sweeps):
        """
        Calculate the energy of a given replica over a number of sweeps.

        :param M: A 2D numpy array representing the MCMC state after each sweep.
        :param num_sweeps: An integer representing the number of sweeps.

        :return: A tuple where the first element is the minimum energy and the second element is an array of energies.
        """
        EE1 = np.zeros(num_sweeps)
        for ii in range(num_sweeps):
            m1 = M[:, ii]
            EE1[ii] = -1 * (m1.T @ self.J @ m1 / 2 + m1.T @ self.h)
        minEnergy = np.min(EE1)
        return minEnergy, EE1


    def MCMC(self, num_sweeps, m_start, beta, hash_table=None, use_hash_table=False):
        """
        Implements the Markov Chain Monte Carlo (MCMC) method using Gibbs sampling.

        Parameters:
        - num_sweeps (int): Number of MCMC sweeps to be performed.
        - m_start (np.array[N,]): Initial seed value of the states, where N is the size of the graph.
        - beta (float): Inverse temperature.
        - hash_table (LRUCache, optional): An LRUCache object for storing previously computed dE values.
        - use_hash_table (bool, optional, default=False): If set to True, utilizes the hash table for caching results.

        Returns:
        - M (np.array[N, num_sweeps]): Matrix containing all the sweeps in bipolar form.
        """

        N = self.J.shape[0]
        m = np.asarray(m_start).copy().reshape(-1, 1)
        M = np.zeros((N, num_sweeps))

        for jj in range(num_sweeps):
            spin_state = tuple(m.ravel())

            for kk in np.random.permutation(N):
                if use_hash_table:
                    if not isinstance(hash_table, LRUCache):
                        raise ValueError("hash_table must be an instance of LRUCache")

                    if spin_state in hash_table:
                        dE = hash_table[spin_state]
                    else:
                        dE = self.J.dot(m) + self.h
                        hash_table[spin_state] = dE

                    m[kk] = np.sign(np.tanh(beta * dE[kk]) - 2 * np.random.rand() + 1)
                else:
                    x = self.J.dot(m) + self.h
                    m[kk] = np.sign(np.tanh(beta * x[kk]) - 2 * np.random.rand() + 1)

            M[:, jj] = m.ravel()

        return M

    def select_non_overlapping_pairs(self, all_pairs):
        """
        Select non-overlapping pairs from a list of all possible consecutive pairs.

        :param all_pairs: A list of tuples where each tuple is a consecutive pair.

        :return: A list of non-overlapping pairs (a total of num_swapping_pairs).
        """
        available_pairs = all_pairs.copy()
        selected_pairs = []
        for _ in range(self.num_swapping_pairs):
            if not available_pairs:
                raise ValueError("Cannot find non-overlapping pairs.")
            i_pair = randint(0, len(available_pairs) - 1)
            pair = available_pairs[i_pair]
            selected_pairs.append(pair)
            # Remove pairs that overlap with the chosen pair
            available_pairs = [p for p in available_pairs if
                               p[0] != pair[0] and p[0] != pair[1] and p[1] != pair[0] and p[1] != pair[1]]
        return selected_pairs

    def find_disagreement_clusters(self, state_1, state_2, J):
        N = len(state_1)
        differing_spins = [i for i in range(N) if state_1[i] * state_2[i] == -1]
        visited = [False] * N
        clusters = []

        for spin in differing_spins:
            if not visited[spin]:
                cluster = [spin]
                stack = [spin]

                while stack:
                    current_spin = stack.pop(0)
                    neighbors = [i for i, val in enumerate(J[current_spin]) if val != 0]
                    disagree_neighbors = list(set(neighbors).intersection(differing_spins))

                    for neighbor in disagree_neighbors:
                        if not visited[neighbor]:
                            visited[neighbor] = True
                            cluster.append(neighbor)
                            stack.append(neighbor)

                if cluster:
                    clusters.append(list(set(cluster)))  # Ensure the cluster contains unique elements

                visited[spin] = True  # Mark the current spin as visited

        return clusters

    def run(self, beta_list, num_replicas, num_sweeps_MCMC=1000, num_sweeps_read=1000, num_swap_attempts=100,
            num_swapping_pairs=1, use_hash_table=0, num_cores=8):
        """
        Run the adaptive parallel tempering algorithm.
        :param beta_list: A 1D numpy array representing the inverse temperatures for the replicas.
        :param num_replicas: An integer, the number of replicas (parallel chains) to use in the algorithm.
        :param num_sweeps_MCMC: An integer, the number of Monte Carlo sweeps to perform (default =1000)
        :param num_sweeps_read: An integer, the number of last sweeps to read from the chains (default =1000)
        :param num_swap_attempts: An integer, the number of swap attempts between chains (default = 100).
        :param num_swapping_pairs: An integer, the number of non-overlapping replica pairs per swap attempt (default =1).
        :param use_hash_table: Whether to use a hash table or not (default =0).
        :param num_cores: How many CPU cores to use in parallel (default= 8).

        :return: Tuple containing:
        - M (2D numpy array): Spin states for each replica. Rows correspond to replicas and columns to states.
        - Energy (1D numpy array): Energy values corresponding to each replica.
        """
        self.num_replicas = num_replicas
        self.num_sweeps_MCMC = num_sweeps_MCMC
        self.num_sweeps_read = num_sweeps_read
        self.num_swap_attempts = num_swap_attempts
        self.num_sweeps_MCMC_per_swap = self.num_sweeps_MCMC // self.num_swap_attempts
        self.num_sweeps_read_per_swap = self.num_sweeps_read // self.num_swap_attempts
        self.num_swapping_pairs = num_swapping_pairs
        self.use_hash_table = use_hash_table

        # If use_hash_table is True, create a new hash table for this process
        if use_hash_table:
            hash_table = LRUCache(maxsize=10000)
        else:
            hash_table = None

        num_subreplicas = 10
        useKatzgraber = True
        num_spins = self.J.shape[0]
        count = np.zeros(self.num_swap_attempts)
        swap_attempted_replicas = np.zeros((self.num_swap_attempts * self.num_swapping_pairs*num_subreplicas, 2))
        swap_accepted_replicas = np.zeros((self.num_swap_attempts * self.num_swapping_pairs*num_subreplicas, 2))

        # Generate all possible consecutive pairs of replicas
        all_pairs = [(i, i + 1) for i in range(1, self.num_replicas)]

        M = np.zeros((len(self.J) * self.num_replicas, self.num_sweeps_MCMC_per_swap * num_subreplicas))
        m_start_matrix = np.sign(2 * np.random.rand(len(self.J) * self.num_replicas, num_subreplicas) - 1)

        swap_index = 0

        for ii in range(int(self.num_swap_attempts)):
            print(f"\nRunning swap attempt = {ii + 1}")
            start_time = time.time()

            # Generate MCMC sweeps for replicas and sub-replicas
            for replica_i in range(self.num_replicas):
                for sub_replica_j in range(num_subreplicas):
                    start_index = replica_i * num_spins
                    end_index = (replica_i + 1) * num_spins

                    current_m_start = m_start_matrix[start_index:end_index, sub_replica_j]

                    # Generate new state with MCMC
                    M_temp = self.MCMC(self.num_sweeps_MCMC_per_swap, current_m_start, beta_list[replica_i],
                                          hash_table=hash_table, use_hash_table=self.use_hash_table)

                    # Store the MCMC-generated states in M
                    M[start_index:end_index, sub_replica_j * self.num_sweeps_MCMC_per_swap:(sub_replica_j + 1) * self.num_sweeps_MCMC_per_swap] = M_temp

                    # Update the starting matrix for next iteration
                    m_start_matrix[start_index:end_index, sub_replica_j] = M_temp[:, -1]

            # Houdayer's Move
            for replica_i in range(num_replicas):
                shuffled_indices = np.random.permutation(num_subreplicas)
                num_pairs = num_subreplicas // 2

                for pair_idx in range(num_pairs):
                    sub_replica_j = shuffled_indices[2 * pair_idx]
                    sub_replica_k = shuffled_indices[2 * pair_idx + 1]

                    # Extract states for clusters
                    state_1 = M[replica_i * num_spins:(replica_i + 1) * num_spins,
                              sub_replica_j * self.num_sweeps_MCMC_per_swap]
                    state_2 = M[replica_i * num_spins:(replica_i + 1) * num_spins,
                              sub_replica_k * self.num_sweeps_MCMC_per_swap]

                    clusters = self.find_disagreement_clusters(state_1, state_2, self.J)

                    if clusters:
                        selected_cluster = clusters[np.random.randint(len(clusters))]

                        # Katzgraber's modification
                        if useKatzgraber and len(selected_cluster) > num_spins // 2:
                            state_1 = -state_1
                        else:
                            state_1[selected_cluster], state_2[selected_cluster] = state_2[selected_cluster], state_1[
                                selected_cluster]

                        # Update the M matrix with swapped states
                        M[replica_i * num_spins:(replica_i + 1) * num_spins,
                        sub_replica_j * self.num_sweeps_MCMC_per_swap] = state_1
                        M[replica_i * num_spins:(replica_i + 1) * num_spins,
                        sub_replica_k * self.num_sweeps_MCMC_per_swap] = state_2

            selected_pairs = self.select_non_overlapping_pairs(all_pairs)

            # PT Swap for each sub-replica
            for chosen_subreplica in range(num_subreplicas):
                mm = M[:, chosen_subreplica * self.num_sweeps_MCMC_per_swap:(chosen_subreplica + 1) * self.num_sweeps_MCMC_per_swap].T

                # Attempt to swap states of each selected pair of replicas
                for pair in selected_pairs:
                    sel, next = pair

                    m_sel = mm[-1, (sel - 1) * num_spins:sel * num_spins].T
                    m_next = mm[-1, (next - 1) * num_spins:next * num_spins].T

                    E_sel = -m_sel.T @ self.J @ m_sel / 2 - m_sel.T @ self.h
                    E_next = -m_next.T @ self.J @ m_next / 2 - m_next.T @ self.h
                    beta_sel = beta_list[sel - 1]
                    beta_next = beta_list[next - 1]

                    print(f"\nSelected pair indices: {sel}, {next}")
                    print(f"β values: {beta_sel}, {beta_next}")
                    print(f"Energies: {E_sel}, {E_next}")

                    swap_attempted_replicas[swap_index] = [sel, next]

                    DeltaE = E_next - E_sel
                    DeltaB = beta_next - beta_sel

                    if np.random.rand() < min(1, np.exp(DeltaB * DeltaE)):
                        count[ii] += 1
                        swap_accepted_replicas[swap_index] = [sel, next]
                        print(f"swapping {np.sum(count)}th time")

                        # Swap the states of the selected replicas
                        m_start_matrix[(sel - 1) * num_spins:sel * num_spins, chosen_subreplica] = m_next
                        m_start_matrix[(next - 1) * num_spins:next * num_spins, chosen_subreplica] = m_sel

                    swap_index += 1

            elapsed_time = time.time() - start_time
            print(f"Elapsed time for swap attempt {ii + 1}: {elapsed_time}")

        # Calculate the final energies of the replicas
        Energy = np.zeros(self.num_replicas)
        EE1_list = []
        for look_replica in range(1, self.num_replicas + 1):
            M_replica = M[(look_replica - 1) * self.J.shape[1]:look_replica * self.J.shape[1], :]
            minEnergy, EE1 = self.replica_energy(M_replica, self.num_sweeps_read_per_swap)
            Energy[look_replica - 1] = minEnergy
            EE1_list.append(EE1)

        # Output the results
        print(f"\nLatest energy from each replica = {Energy}")
        print(f"Swap acceptance rate = {np.count_nonzero(count) / count.size * 100:.2f} per cent\n")

        # Plot the energy traces
        self.plot_energies(EE1_list, beta_list)
        return M, Energy

    def plot_energies(self, EE1_list, beta_list):
        """
        Plot the energy traces of all replicas.

        :param EE1_list: A list of 1D numpy arrays representing the energy traces of the replicas.
        :param beta_list: A 1D numpy array representing the inverse temperatures for the replicas.
        """
        plt.figure()
        for i in range(self.num_replicas):
            plt.plot(EE1_list[i], label=f"Replica {i + 1} (β={beta_list[i]:.2f})")
        plt.xlabel('Sweeps')
        plt.ylabel('Energy')
        plt.title('Energy traces for different replicas')
        plt.legend()
        # plt.show()
        plt.savefig('APT_ICM_energy..png')


def main():
    # Load the coupling and external field matrices, and the list of inverse temperatures
    J = np.load('J.npy')
    h = np.load('h.npy')

    beta_list = np.load('beta_list_python.npy')
    print(f"[INFO] Beta List: {beta_list}")

    # # uncomment if you want to manually select the inverse temperatures for the replicas from beta_list
    # startingBeta = 18
    # num_replicas = 12
    # selectedBeta = range(startingBeta, startingBeta + num_replicas)
    # beta_list = beta_list[selectedBeta]

    num_replicas = beta_list.shape[0]
    print(f"[INFO] Number of replicas: {num_replicas}")

    norm_factor = np.max(np.abs(J))
    J = J / norm_factor
    h = h / norm_factor

    # Initiate the main APT_ICM run
    print("\n[INFO] Starting main Adaptive Parallel Tempering process with ICM moves...")

    # Create an apt_ICM instance
    apt_ICM = APT_ICM(J.copy(), h.copy())

    # run Adaptive Parallel Tempering with ICM move
    M, Energy = apt_ICM.run(beta_list, num_replicas=num_replicas,
                            num_sweeps_MCMC=int(1e4),
                            num_sweeps_read=int(1e3),
                            num_swap_attempts=int(1e2),
                            num_swapping_pairs=1, use_hash_table=0, num_cores=8)

    # print(M)
    print(Energy)


if __name__ == '__main__':
    main()
