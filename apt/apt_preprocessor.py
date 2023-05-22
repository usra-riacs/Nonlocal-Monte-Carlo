import numpy as np
import os
import time
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import networkx as nx
from cachetools import LRUCache
from concurrent.futures import ProcessPoolExecutor, as_completed

# np.random.seed(12624755)  # Set the seed to an arbitrary number


class APT_preprocessor:
    def __init__(self, J, h):
        """
        Initialize the APT_preprocessor class with J and h.

        :param J: A 2D numpy array representing the coupling matrix (weights J).
        :param h: A 1D numpy array representing the external field (biases h).
        """
        self.J = J
        self.h = h
        self.N = J.shape[0]  # number of spins
        self.colorMap = self.greedy_coloring_saturation_largest_first()

    def greedy_coloring_saturation_largest_first(self):
        """
        Perform greedy coloring using the saturation largest first strategy.

        :return colorMap: A 1D numpy array containing the colormap of the graph.
        """
        # Create a NetworkX graph from the J matrix
        G = nx.Graph(self.J)

        # Perform greedy coloring with the saturation largest first strategy
        color_map = nx.coloring.greedy_color(G, strategy='saturation_largest_first')

        # Convert the color map to a 1D numpy array
        colorMap = np.array([color_map[node] for node in G.nodes])

        return colorMap

    def MCMC_GC(self, num_sweeps_MCMC, m_start, beta, hash_table, use_hash_table=0):
        """
        Perform MCMC with graph coloring.

        :param num_sweeps_MCMC: An integer representing the number of MCMC sweeps.
        :param m_start: A 1D numpy array representing the initial state.
        :param beta: A float representing the inverse temperature.
        :param hash_table: A LRUCache object used to store previously calculated dE values.
        :param use_hash_table: A boolean flag. If True, a hash table will be used for caching results. (default = 0)

        :return M: A 2D numpy array representing the MCMC state after each sweep.
        """
        N = self.J.shape[0]
        m = m_start
        M = np.zeros((N, num_sweeps_MCMC))
        J = csr_matrix(self.J)

        if self.h.shape[0] == 1:
            self.h = self.h.T

        # Group spins by color
        required_colors = len(np.unique(self.colorMap))
        Groups = [None] * required_colors
        for k in range(required_colors):
            Groups[k] = np.where(self.colorMap == k)[0]

        # Create a list of grouped J and h matrices
        J_grouped = [J[Groups[k], :] for k in range(required_colors)]
        h_grouped = [self.h[Groups[k]] for k in range(required_colors)]

        for jj in range(num_sweeps_MCMC):
            for ijk in range(required_colors):
                group = Groups[ijk]
                spin_state = tuple(m.ravel())

                if use_hash_table:
                    if not isinstance(hash_table, LRUCache):
                        raise ValueError("hash_table must be an instance of cachetools.LRUCache")

                    if spin_state in hash_table:
                        dE = hash_table[spin_state]
                    else:
                        dE = J.dot(m) + self.h
                        hash_table[spin_state] = dE

                    m[group] = np.sign(np.tanh(beta * dE[group]) - 2 * np.random.rand(len(group), 1) + 1)
                else:
                    x = J_grouped[ijk].dot(m) + h_grouped[ijk]
                    m[group] = np.sign(np.tanh(beta * x) - 2 * np.random.rand(len(group), 1) + 1)

            M[:, jj] = m.ravel()

        return M

    def MCMC_task(self, m_start, beta, num_sweeps_MCMC, num_sweeps_read, use_hash_table=0):
        """
        Perform a Monte Carlo simulation for a single task.

        This method is designed to be run in a separate parallel process.

        :param m_start: A 1D numpy array representing the initial state.
        :param beta: A float representing the inverse temperature.
        :param num_sweeps_MCMC: An integer representing the number of MCMC sweeps.
        :param num_sweeps_read: An integer representing the number of last sweeps to read.
        :param use_hash_table: A boolean flag. If True, a hash table will be used for caching results. (default = 0)
        :return: A tuple of (Energy, m), where Energy is a 1D numpy array representing the energy at each sweep,
                 and m is the final state after the sweeps.
        """

        # If use_hash_table is True, create a new hash table for this process
        if use_hash_table:
            hash_table = LRUCache(maxsize=10000)
        else:
            hash_table = None

        # Run the MCMC algorithm with graph coloring
        M = self.MCMC_GC(num_sweeps_MCMC, m_start, beta, hash_table, use_hash_table)

        # Only keep the last num_sweeps_read sweeps
        mm = M[:, -num_sweeps_read:]

        # Initialize an array to store the energy at each sweep
        Energy = np.zeros(num_sweeps_read)

        # Calculate the energy at each sweep
        for kk in range(num_sweeps_read):
            m = mm[:, kk].reshape(1, -1)
            result = -(m @ (self.J / 2) @ m.T + m @ self.h)
            Energy[kk] = result.item()

        # Return the energy and the final state
        return Energy, m

    def run(self, num_sweeps_MCMC=1000, num_sweeps_read=1000, num_rng=100):
        """
        Run the Adaptive Parallel Tempering (APT) preprocessing algorithm.

        :param num_sweeps_MCMC: An integer representing the number of MCMC sweeps (default: 1000).
        :param num_sweeps_read: An integer representing the number of last sweeps to read (default: 1000).
        :param num_rng: An integer representing the number of independent MCMC chains  (default: 100).
        """
        foldername = 'data'
        os.makedirs(os.path.join('Results', foldername), exist_ok=True)

        # 1) Normalize energy with |J_ij| ~ 1
        norm_factor = np.max(np.abs(self.J))
        self.J = self.J / norm_factor
        self.h = self.h / norm_factor

        if self.h.shape[0] == 1:
            self.h = self.h.T

        # 2) Initialize β_0 ~ 0.5 (chosen such that the largest spins can flip often enough)
        beta = [0.5]
        alpha = 1.25
        iter = 1
        sigma_E = 1000
        sigma_E_min = 0.5 * np.min(np.abs(self.J[self.J != 0]))  # avg_std < σ_min ~ 0.5*(smallest J_ij)

        beta_max = 30
        sigma = []
        use_hash_table = 1

        # 4) APT loop until freezeout, typically avg_std < σ_min ~ 0.5*(smallest J_ij)
        while sigma_E > sigma_E_min:
            start_time = time.time()

            if iter != 1:
                # c) Compute new β_i+1 = β_i +(0.85-1.25)/avg_std
                beta.append(beta[-1] + alpha / sigma_E)

            Energy = np.zeros((num_rng, num_sweeps_read))
            saved_state = np.zeros((num_rng, self.N))

            with ProcessPoolExecutor(max_workers=8) as executor:
                futures = {}
                for j in range(num_rng):
                    if iter == 1:
                        m_start = np.sign(2. * np.random.rand(self.J.shape[0], 1) - 1)
                    else:
                        m_start = saved_state[j, :].reshape(-1, 1)

                    futures[executor.submit(self.MCMC_task, m_start, beta[-1], num_sweeps_MCMC,
                                            num_sweeps_read, use_hash_table)] = j

            for future in as_completed(futures):
                j = futures[future]
                Energy[j, :], saved_state[j, :] = future.result()

            # b) Compute avg_std = <E_j> over index j
            sigma_E = np.mean(np.std(Energy, axis=1))
            print(f'\ncurrent iteration = {iter}, β = {beta[-1]:.3f}, and average σ = {sigma_E:.3f}\n')

            if beta[-1] > beta_max:
                print('Did not converge but hit the max beta limit\n')
                break

            sigma.append(sigma_E)

            # Save energy and sigma for each iteration
            np.save(os.path.join('Results', foldername, f'Energy_iter_{iter}.npy'), Energy)
            np.save(os.path.join('Results', foldername, f'sigma_iter_{iter}.npy'), sigma_E)

            iter += 1

            end_time = time.time()
            print(f"Elapsed time: {end_time - start_time:.2f} seconds")

        print('beta =')
        print(beta)
        print('sigma =')
        print(sigma)
        np.save('beta_list_python.npy', beta)
        np.save('sigma_list_python.npy', sigma)
        self.plot_results(beta, sigma)

    def plot_results(self, beta, sigma):
        """
        Plot the beta schedule.
        :param beta: A list of beta values.
        :param sigma: A list of sigma values.
        """
        fig, ax1 = plt.subplots()
        ax1.plot(beta, marker='*', linewidth=2, markersize=6, label='beta')
        ax1.set_ylabel('beta')
        ax1.tick_params(axis='y')

        ax2 = ax1.twinx()
        ax2.plot(sigma, marker='>', linewidth=2, markersize=6, color='tab:orange', label='sigma')
        ax2.set_ylabel('sigma')
        ax2.tick_params(axis='y')

        ax1.set_xlabel('iteration')
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        ax1.tick_params(axis='both', which='major', labelsize=18)
        ax2.tick_params(axis='both', which='major', labelsize=18)
        for label in ax1.get_xticklabels() + ax1.get_yticklabels() + ax2.get_yticklabels():
            label.set_weight('bold')
        plt.show()
        # plt.savefig('beta_sigma.png')


def main():
    # Load J and h from .npy files
    J = np.load('J.npy')
    h = np.load('h.npy')
    J = csr_matrix(J)  # Convert the dense matrix to a sparse one
    # print(J)
    # print(h)

    apt_prep = APT_preprocessor(J, h)
    apt_prep.run(num_sweeps_MCMC=10000, num_sweeps_read=1000, num_rng=100)


if __name__ == '__main__':
    main()
