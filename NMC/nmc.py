import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from cachetools import LRUCache
from collections import defaultdict

# Setting random seed for reproducibility
np.random.seed(0)


class NMC:
    """
    The NMC class is used to implement the Non-equilibrium Monte Carlo (NMC) algorithm.
    """

    def __init__(self, J, h):
        """
        Initialize an NMC object.
        :param J: A 2D numpy array representing the coupling matrix (weights J).
        :param h: A 1D numpy array or list representing the external field (biases h).
        """
        self.J = J
        self.h = h
        self.h = np.asarray(h).reshape(-1)  # Reshape h into a 1D array

        self.colorMap = self.greedy_coloring_saturation_largest_first()

    def greedy_coloring_saturation_largest_first(self):
        """
        Perform greedy coloring using the saturation largest first strategy.

        :return colorMap: A 1D numpy array containing the colorMap of the graph.
        """
        # Create a NetworkX graph from the J matrix
        G = nx.Graph(self.J)

        # Perform greedy coloring with the saturation largest first strategy
        color_map = nx.coloring.greedy_color(G, strategy='saturation_largest_first')

        # Convert the color map to a 1D numpy array
        colorMap = np.array([color_map[node] for node in G.nodes])

        return colorMap

    def MCMC_GC(self, num_sweeps, m_start, beta, J, h, colorMap, anneal=False, sweeps_per_beta=1, initial_beta=0,
                hash_table=None, use_hash_table=False):
        """
        Implements the Markov Chain Monte Carlo (MCMC) method using Gibbs sampling and Graph coloring.

        Parameters:
        - num_sweeps (int): Number of MCMC sweeps to be performed.
        - m_start (np.array[N,]): Initial seed value of the states, where N is the size of the graph.
        - beta (float): Inverse temperature. Use the maximum value if anneal is set to True.
        - J (np.array[N, N]): Weight matrix where N is the size of the graph.
        - h (np.array[N,]): Bias values where N is the size of the graph.
        - colorMap (np.array[N,]): Color assignments for the nodes where N is the size of the graph.
        - anneal (bool, default=False): Set to True for annealing, else False.
        - sweeps_per_beta (int, optional, default=1): Number of sweeps to perform at each beta level during annealing.
        - initial_beta (float, optional, default=0): Initial value for beta when annealing.
        - hash_table (LRUCache, optional): A LRUCache object for storing previously computed dE values.
        - use_hash_table (bool, optional, default=False): If set to True, utilizes the hash table for caching results.

        Returns:
        - M (np.array[N, num_sweeps]): Matrix containing all the sweeps in bipolar form.
        """

        N = J.shape[0]
        m = np.asarray(m_start).copy().reshape(-1,1)  # Make sure m_star is a numpy array and has shape (N, 1) and also create a copy
        M = np.zeros((N, num_sweeps))
        J = csr_matrix(J)
        h = np.asarray(h).copy().reshape(-1, 1)  # Make sure h is a numpy array and has shape (N, 1)

        required_colors = len(np.unique(colorMap))
        Groups = [None] * required_colors
        for k in range(required_colors):
            Groups[k] = np.where(colorMap == k)[0]

        J_grouped = [J[Groups[k], :] for k in range(required_colors)]
        h_grouped = [h[Groups[k]] for k in range(required_colors)]

        num_betas = num_sweeps // sweeps_per_beta
        if anneal:
            beta_vals = np.linspace(initial_beta, beta, num_betas)
        beta_run = np.zeros(num_sweeps)
        beta_idx = 0

        for jj in range(num_sweeps):
            if anneal:
                if jj % sweeps_per_beta == 0 and beta_idx < num_betas - 1:
                    beta_idx += 1
                beta_run[jj] = beta_vals[beta_idx]
            else:
                beta_run[jj] = beta

            for ijk in range(required_colors):
                group = Groups[ijk]
                spin_state = tuple(m.ravel())

                if use_hash_table:
                    if not isinstance(hash_table, LRUCache):
                        raise ValueError("hash_table must be an instance of cachetools.LRUCache")

                    if spin_state in hash_table:
                        dE = hash_table[spin_state]
                    else:
                        dE = J.dot(m) + h
                        hash_table[spin_state] = dE

                    m[group] = np.sign(np.tanh(beta_run[jj] * dE[group]) - 2 * np.random.rand(len(group), 1) + 1)
                else:
                    x = J_grouped[ijk].dot(m) + h_grouped[ijk]
                    m[group] = np.sign(np.tanh(beta_run[jj] * x) - 2 * np.random.rand(len(group), 1) + 1)

                M[:, jj] = m.ravel()

        return M

    def LBP_convexified(self, lambda_start, lambda_end, lambda_reduction_factor, m_star, epsilon, tolerance,
                        max_iterations, threshold_initial, threshold_cutoff, global_beta):
        """
        Implements the Loopy Belief Propagation (LBP) with a convexification procedure.

        Parameters:
        - lambda_start (float): Initial lambda value for the convexification procedure.
        - lambda_end (float): Final lambda value for the convexification procedure.
        - lambda_reduction_factor (float): Reduction factor by which lambda is reduced at each step.
        - m_star (numpy.ndarray, size (N,)): Initial good state spins configuration, where N is the number of spins.
        - epsilon (np.array, shape (N,)): Epsilon values for convexification.
        - tolerance (float): Tolerance level for convergence of LBP.
        - max_iterations (int): Maximum number of iterations allowed in LBP.
        - threshold_initial (float): Initial threshold for identifying seed spins for the backbone.
        - threshold_cutoff (float): Cutoff threshold below which clusters (backbones) won't be expanded.
        - global_beta (float): Inverse temperature value for the system.

        Returns:
        - clusters (list of np.arrays): List of clusters found by the function. Each cluster is an array of node indices.
        - marginals_all_lambdas (defaultdict of np.arrays): Marginal probabilities for all lambda values. Each array has shape (N,).
        - mean_marginals_all_lambdas (defaultdict of floats): Mean marginal probabilities for all lambda values.
        - h_tilde_all_lambdas (defaultdict of np.arrays): Values of h_tilde for all lambda values. Each array has shape (N,).
        - J_tilde_all_lambdas (defaultdict of np.arrays): Values of J_tilde for all lambda values. Each array has shape (N, N).
        """
        # Reshape h into a 1D array
        h = np.asarray(self.h).copy().reshape(-1)
        m_star = np.asarray(m_star).copy().reshape(-1)

        lambda_val = lambda_start
        marginals_all_lambdas = defaultdict(list)
        mean_marginals_all_lambdas = defaultdict(list)
        h_tilde_all_lambdas = defaultdict(list)
        J_tilde_all_lambdas = defaultdict(list)

        # Initialize h and u messages for LBP
        h_msgs = np.zeros((self.J.shape[0], self.J.shape[0]))
        u_msgs = self.J * m_star.reshape(1, -1)

        while lambda_val >= lambda_end:
            # implement the soft clamping at m_star (convexify)
            hClamp = lambda_val * m_star * epsilon
            h_lambda = h + hClamp

            # Run LBP
            marginal, _, h_tilde, J_tilde, iteration_LBP, h_msgs, u_msgs = self.LoopyBeliefPropagation(
                self.J, h_lambda.copy(), global_beta, h_msgs.copy(), u_msgs.copy(), tolerance, max_iterations
            )

            # Handle the case when LBP diverges
            if iteration_LBP == max_iterations - 1 and lambda_val == lambda_start:
                raise ValueError(
                    'LBP diverged at initial lambda, please try a larger lambda_start or increase max_iterations or beta')
            elif iteration_LBP == max_iterations - 1 and lambda_val != lambda_start:
                lambda_end = lambda_val
                marginal = marginal_LBP_prev
            else:
                marginal_LBP_prev = marginal

            # Store marginal probabilities, their mean, h_tilde, and J_tilde for the current lambda
            marginals_all_lambdas[lambda_val] = marginal
            mean_marginals_all_lambdas[lambda_val] = np.mean(marginal)
            h_tilde_all_lambdas[lambda_val] = h_tilde
            J_tilde_all_lambdas[lambda_val] = J_tilde

            # Reduce lambda for the next iteration
            lambda_val = lambda_val * lambda_reduction_factor

            if round(lambda_val, 6) == 0:
                break

        clusters = self.find_clusters(marginal, threshold_initial, threshold_cutoff, 0.01)
        print(f"\ncluster size = {sum(len(cluster) for cluster in clusters)}\n")

        return clusters, marginals_all_lambdas, mean_marginals_all_lambdas, h_tilde_all_lambdas, J_tilde_all_lambdas

    def LoopyBeliefPropagation(self, J, h, beta, h_msgs, u_msgs, tolerance, max_iterations):

        """
        Implements the Loopy Belief Propagation (LBP) algorithm.

        Parameters:
        - J (np.array, shape (N, N)): Interaction matrix, where N is the size of the graph.
        - h (np.array, shape (N,)): External magnetic field, reshaped into a 1D numpy array.
        - beta (float): Inverse temperature value for the system.
        - h_msgs (np.array, shape (N, N)): Initial h messages for LBP.
        - u_msgs (np.array, shape (N, N)): Initial u messages for LBP.
        - tolerance (float): Tolerance level for convergence of LBP.
        - max_iterations (int): Maximum number of iterations for LBP.

        Returns:
        - magnetizations (np.array, shape (N,)): Estimated magnetizations.
        - correlations (np.array, shape (N, N)): Estimated correlations.
        - h_tilde (np.array, shape (N,)): Effective h from LBP.
        - J_tilde (np.array, shape (N, N)): Effective J from LBP.
        - iteration (int): Number of iterations run in the LBP.
        - h_msgs (np.array, shape (N, N)): Final h messages.
        - u_msgs (np.array, shape (N, N)): Final u messages.
        """

        # Reshape h into a 1D array
        h = np.asarray(h).reshape(-1)

        for iteration in range(max_iterations):
            h_old = h_msgs.copy()
            u_old = u_msgs.copy()

            # vectorized method (faster)
            for i in range(h_msgs.shape[0]):
                total_u_msgs = h[i] + np.sum(u_msgs[:, i])
                h_msgs[i, :] = total_u_msgs - u_msgs[:, i]
                h_msgs[i, i] = 0  # Set the diagonal to 0

            u_msgs = (1 / beta) * self.atanh_saturated(np.tanh(beta * J) * np.tanh(beta * h_msgs))

            # Calculate relative changes
            u_change = np.max(np.abs(u_msgs - u_old)) / np.max(np.abs(u_msgs) + np.abs(u_old))
            h_change = np.max(np.abs(h_msgs - h_old)) / np.max(np.abs(h_msgs) + np.abs(h_old))

            # Check for convergence using relative changes
            if u_change < tolerance and h_change < tolerance:
                break

        # Compute magnetizations and correlations
        magnetizations = np.tanh(beta * (h + np.sum(u_msgs, axis=0)))
        correlations = (np.tanh(beta * J) + np.tanh(beta * h_msgs) * np.tanh(beta * h_msgs.T)) / \
                       (1 + np.tanh(beta * J) * np.tanh(beta * h_msgs) * np.tanh(beta * h_msgs.T) + 1e-10)

        # Remove self-correlations
        correlations = correlations - np.diag(np.diag(correlations))

        # Compute h_tilde using the magnetizations
        h_tilde = (1 / beta) * self.atanh_saturated(magnetizations)
        # Compute J_tilde using the correlations
        J_tilde = (1 / beta) * self.atanh_saturated(correlations)

        return magnetizations, correlations, h_tilde, J_tilde, iteration, h_msgs, u_msgs

    def atanh_saturated(self, x):
        """
        Compute the arctanh of x with saturation.

        The function saturates the input at values for which np.tanh reaches its asymptotic limits.
        This is done to prevent numerical issues when using np.arctanh with values outside its domain
        (which is between -1 and 1).

        Parameters:
        x (np.array or float): Input array or value to compute arctanh.

        Returns:
        np.array or float: The arctanh of the input with saturation.
        """

        epsilon = np.finfo(float).eps
        pos_sat = np.tanh(19.06)
        neg_sat = np.tanh(-19.06)

        # Saturate input values to be strictly within the domain of arctanh
        x_clipped = np.clip(x, neg_sat + epsilon, pos_sat - epsilon)

        # Compute atanh for all elements. It's now safe due to the previous saturation.
        out = np.arctanh(x_clipped)

        return out

    def find_clusters(self, magnetizations, threshold_initial, threshold_cutoff, threshold_step):
        """
        Find clusters based on magnetizations and given thresholds.

        This method identifies clusters of spins by starting with a high threshold to determine initial
        seed spins and then gradually decreasing the threshold to expand each cluster until a specified
        cutoff threshold is reached.

        Parameters:
        - magnetizations (np.array): Estimated magnetizations of shape (N, ), where N is the number of spins.
        - threshold_initial (float): Initial threshold for identifying seed spins for the backbone.
        - threshold_cutoff (float): Cutoff threshold below which clusters (backbones) won't be expanded.
        - threshold_step (float): Value by which the threshold is decreased in each iteration.

        Returns:
        - clusters (list of np.array): Each np.array in the list represents a cluster of spins, specified by their indices.
        """
        # Find the seed spins (indices)
        seed_indices = np.where(np.abs(magnetizations) >= threshold_initial)[0]

        # Initialize an empty list to store the clusters
        clusters = []

        # Loop through each seed spin to initialize the clusters
        for seed in seed_indices:
            # Check if the seed is already part of a cluster
            if any(seed in cluster for cluster in clusters):
                continue

            # Find the direct neighbors of the seed
            neighbors = np.where(self.J[seed, :] != 0)[0]

            # Exclude neighbors that are already part of a cluster
            neighbors = np.setdiff1d(neighbors, np.hstack(clusters) if clusters else [])

            # Find the common spins between neighbors and seed_indices
            common_spins = np.intersect1d(neighbors, seed_indices)

            # Add the common neighbor spins to the clusters
            clusters.append(np.append(seed, common_spins))

        # Reduce the threshold step by step and grow the clusters
        current_threshold = threshold_initial - threshold_step
        while current_threshold > threshold_cutoff:

            for i, cluster in enumerate(clusters):
                # Find the direct neighbors of all nodes in the current cluster
                neighbors = np.unique(np.where(self.J[cluster, :] != 0)[1])

                # Exclude neighbors that are already part of any cluster
                neighbors = np.setdiff1d(neighbors, np.hstack(clusters) if clusters else [])

                # Find the spins among the neighbors that are above the current threshold
                spins_above_threshold = np.abs(magnetizations[neighbors]) >= current_threshold

                # Add these spins to the current cluster
                clusters[i] = np.append(clusters[i], neighbors[spins_above_threshold])

            # Decrement the current threshold
            current_threshold -= threshold_step

        return clusters

    def NMC_subroutine(self, m_star, num_cycles, num_sweeps_per_NMC_phase, full_update_frequency, M_skip, global_beta,
                       temp_x, lambda_start, lambda_end, lambda_reduction_factor, threshold_initial, threshold_cutoff,
                       max_iterations, tolerance, all_clusters=None, hash_table=None, use_hash_table=False):
        """
        Implements a Non-equilibrium Monte Carlo method for energy minimization.

        Parameters:
        - m_star (numpy.ndarray, size (N,)): Initial good state spins configuration, where N is the number of spins.
        - num_cycles (int): Number of complete NMC cycles to be performed.
        - num_sweeps_per_NMC_phase (int): Number of MCMC sweeps to be performed in each NMC phase.
        - full_update_frequency (int): Frequency of full spin updates.
        - M_skip (int): Specifies that only every M_skip-th sweep's results are saved.
        - global_beta (float): Inverse temperature.
        - temp_x (float): Temperature scaling factor for the clusters (backbones), their global_beta will be divided by this.
        - lambda_start (float): Initial value for lambda to start convexified LBP.
        - lambda_end (float): Final value for lambda to end  convexified  LBP.
        - lambda_reduction_factor (float): Factor by which lambda is reduced in each iteration of LBP_convexified.
        - threshold_initial (float): Initial threshold for identifying seed spins for the backbone.
        - threshold_cutoff (float): Cutoff threshold below which clusters (backbones) won't be expanded.
        - max_iterations (int): Maximum number of iterations allowed in LBP.
        - tolerance (float): Tolerance value for convergence in LBP.
        - all_clusters (numpy.ndarray, optional): Precomputed clusters if available, otherwise None.
        - hash_table (LRUCache, optional): A LRUCache object used to store previously calculated dE values, should be initialized before being passed here.
        - use_hash_table (bool, optional): If True, the hash table will be used for caching results, defaults to False.

        Returns:
        - M_overall (numpy.ndarray, size (N, num_sweeps_per_NMC_phase * num_cycles * 3 // M_skip)): Array storing the magnetization values over the simulation.
        - energy_overall (numpy.ndarray, size (num_sweeps_per_NMC_phase * num_cycles * 3 // M_skip,)): Array storing the energy values over the simulation.
        - min_energy (float): The minimum energy observed during the simulation.
        - all_clusters (numpy.ndarray): The clusters used or calculated during the simulation.
        """

        N = len(self.h)
        epsilon = np.abs(self.h) + np.sum(np.abs(self.J),
                                          axis=1)  # Compute epsilon as absolute of h + sum of absolute J's row-wise
        all_spins = np.arange(len(self.h))  # Get all possible spins
        m_init = m_star  # Initialize m_init with m_star
        clusters_provided = all_clusters is not None

        # Create containers for storing results
        M_overall = np.zeros((N, num_sweeps_per_NMC_phase * num_cycles * 3 // M_skip))
        energy_overall = np.zeros(num_sweeps_per_NMC_phase * num_cycles * 3 // M_skip)

        M_index = 0  # Index to keep track of position in M_overall and energy_overall

        for cycle in range(num_cycles):
            print(f'\nCurrent iteration = {cycle + 1}')
            # Get clusters if not provided, using LBP_convexified
            if not clusters_provided:
                clusters, _, _, _, _ = self.LBP_convexified(
                    lambda_start, lambda_end, lambda_reduction_factor, m_star.copy(),
                    epsilon, tolerance, max_iterations, threshold_initial, threshold_cutoff, global_beta
                )
                all_clusters = np.concatenate(clusters).astype(int) if clusters else np.array([], dtype=int)
            non_clusters = np.setdiff1d(all_spins, all_clusters)  # Get spins that are not in clusters

            # Modify J and h for clusters
            J_c = self.J.copy()
            h_c = self.h.copy()
            J_c[all_clusters, :] = J_c[all_clusters, :] / temp_x # clusters run at higher temperature
            h_c[all_clusters] /= temp_x
            h_c[non_clusters] = m_init[non_clusters] * 10000  # Strongly bias the non-cluster spins to keep them frozen
            # caution: hash_table is not used as J and h are being scaled by temp_x.
            M = self.MCMC_GC(num_sweeps_per_NMC_phase, m_init.copy(), global_beta, J_c, h_c, self.colorMap,
                             anneal=False, hash_table=hash_table,
                             use_hash_table=False)  # Run MCMC for clusters
            energies = [- (M[:, i].T @ self.J @ M[:, i] / 2 + M[:, i].T @ self.h) for i in
                        range(M.shape[1])]  # Compute energies

            # Store results
            M_overall[:, M_index:M_index + num_sweeps_per_NMC_phase // M_skip] = M[:, ::M_skip]
            energy_overall[M_index:M_index + num_sweeps_per_NMC_phase // M_skip] = energies[::M_skip]

            M_index += num_sweeps_per_NMC_phase // M_skip
            min_energy_idx = np.argmin(energies)
            m_init = M[:, min_energy_idx]

            # Modify J and h for non-clusters
            J_nc = self.J.copy()  # non-clusters run at normal temperature
            h_nc = self.h.copy()
            h_nc[all_clusters] = m_init[all_clusters] * 10000  # Strongly bias the cluster (backbones) spins to keep them frozen

            # caution: hash_table is not used as h_nc is not same as h
            M = self.MCMC_GC(num_sweeps_per_NMC_phase, m_init.copy(), global_beta, J_nc, h_nc, self.colorMap,
                             anneal=False, hash_table=hash_table,
                             use_hash_table=False)  # Run MCMC for non-clusters
            energies = [- (M[:, i].T @ self.J @ M[:, i] / 2 + M[:, i].T @ self.h) for i in
                        range(M.shape[1])]  # Compute energies

            # Store results
            M_overall[:, M_index:M_index + num_sweeps_per_NMC_phase // M_skip] = M[:, ::M_skip]
            energy_overall[M_index:M_index + num_sweeps_per_NMC_phase // M_skip] = energies[::M_skip]

            M_index += num_sweeps_per_NMC_phase // M_skip
            min_energy_idx = np.argmin(energies)
            m_init = M[:, min_energy_idx]

            # Full update after every full_update_frequency cycles
            if cycle % full_update_frequency == 0:
                M = self.MCMC_GC(num_sweeps_per_NMC_phase, m_init.copy(), global_beta, self.J, self.h, self.colorMap,
                                 anneal=False, hash_table=hash_table, use_hash_table=use_hash_table)
                energies = [- (M[:, i].T @ self.J @ M[:, i] / 2 + M[:, i].T @ self.h) for i in range(M.shape[1])]

                # Store results
                M_overall[:, M_index:M_index + num_sweeps_per_NMC_phase // M_skip] = M[:, ::M_skip]
                energy_overall[M_index:M_index + num_sweeps_per_NMC_phase // M_skip] = energies[::M_skip]

                M_index += num_sweeps_per_NMC_phase // M_skip
                Energy_star = np.min(energies)
                min_energy_idx = np.argmin(energies)
                m_init = M[:, min_energy_idx]

                m_star = m_init.copy()
                print(f'\ncurrent m_star energy = {Energy_star:.8f}')

        M_overall = M_overall[:, :M_index]
        energy_overall = energy_overall[:M_index]
        min_energy = np.min(energy_overall)

        return M_overall, energy_overall, min_energy, all_clusters

    def run(self, num_sweeps_initial=int(1e4), num_sweeps_per_NMC_phase=int(1e4),
            num_NMC_cycles=10, full_update_frequency=1, M_skip=1, temp_x=20,
            global_beta=2.5, lambda_start=0.5, lambda_end=0.01, lambda_reduction_factor=0.9,
            threshold_initial=0.999999, threshold_cutoff=0.99999, max_iterations=100, tolerance=np.finfo(float).eps,
            use_hash_table=False):
        """
        Execute the NMC  algorithm to solve for optimal states and energy.

        Parameters:
        - num_sweeps_initial (int): Number of sweeps for initial MCMC run to find good m_star.
        - num_sweeps_per_NMC_phase (int): Number of MCMC sweeps per NMC phase.
        - num_NMC_cycles (int): Total number of NMC cycles.
        - full_update_frequency (int): Frequency of all spin updates in NMC.
        - M_skip (int): Interval for storing results.
        - temp_x (int): Heated temperature factor for NMC (scales beta by dividing by temp_x)
        - global_beta (float): Global inverse temperature for MCMC and NMC.
        - lambda_start (float): Initial lambda value for convexified LBP.
        - lambda_end (float): Ending lambda value for for convexified LBP.
        - lambda_reduction_factor (float): Factor by which lambda is reduced at each LBP run.
        - threshold_initial (float): Initial threshold of marginals for growing backbones seeds
        - threshold_cutoff (float): Ending threshold of marginals for backbones.
        - max_iterations (int): Maximum iterations for LBP to converge
        - tolerance (float): Tolerance for convergence in LBP.
        - use_hash_table (bool, optional): If True, the hash table will be used for caching results.

        Returns:
        - Tuple: M_overall, energy_overall, min_energy
        """

        # Boolean flag to decide normalization
        normalize = True
        # Normalize energy with |J_ij| ~ 1
        norm_factor = np.max(np.abs(self.J)) if normalize else 1
        self.J = self.J / norm_factor
        self.h = self.h / norm_factor

        N = len(self.h)

        # If use_hash_table is True, create a new hash table for this process
        if use_hash_table:
            hash_table = LRUCache(maxsize=10000)
        else:
            hash_table = None

        # initial MCMC to find m_star
        m_init = np.sign(2 * np.random.rand(N) - 1)

        # Running MCMC to find the initial m_star
        M = self.MCMC_GC(num_sweeps_initial, m_init.copy(), global_beta, self.J, self.h, self.colorMap, anneal=True,
                         sweeps_per_beta=1, initial_beta=0,
                         hash_table=hash_table,
                         use_hash_table=use_hash_table)

        # Calculate initial energies from MCMC
        initial_energies = [- (M[:, i].T @ self.J @ M[:, i] / 2 + M[:, i].T @ self.h) for i in range(M.shape[1])]

        Energy_star = min(initial_energies)
        min_energy_idx = np.argmin(initial_energies)

        m_init = M[:, min_energy_idx]
        m_star = m_init.copy()
        print(f'\ninitial m_star energy = {Energy_star:.8f}')

        M_overall, energy_overall, min_energy, all_clusters = self.NMC_subroutine(m_star, num_NMC_cycles,
                                                                                  num_sweeps_per_NMC_phase,
                                                                                  full_update_frequency, M_skip,
                                                                                  global_beta, temp_x,
                                                                                  lambda_start, lambda_end,
                                                                                  lambda_reduction_factor,
                                                                                  threshold_initial, threshold_cutoff,
                                                                                  max_iterations,
                                                                                  tolerance, hash_table=hash_table,
                                                                                  use_hash_table=use_hash_table)

        # Call to the plot method
        self.plot_results(M_overall, energy_overall, all_clusters, M_skip,
                          num_NMC_cycles, full_update_frequency, num_sweeps_per_NMC_phase)

        return M_overall, energy_overall, min_energy

    def plot_results(self, M_overall, energy_overall, all_clusters, M_skip, num_NMC_cycles, full_update_frequency,
                     num_sweeps_per_NMC_phase):
        """
        Plot the spin configurations and energy evolution.

        Parameters:
        - M_overall (numpy.ndarray): The overall magnetization/spin data.
        - energy_overall (numpy.ndarray): Energy data for each sweep.
        - all_clusters (numpy.ndarray): Indices of all clusters.
        - M_skip (int): Number of sweeps to skip for the magnetization plot.
        - num_NMC_cycles (int): Total number of NMC cycles.
        - full_update_frequency (int): Frequency of full updates in the NMC cycles.
        - num_sweeps_per_NMC_phase (int): Number of sweeps per NMC phase.

        Returns:
        None. This method shows plots as output.
        """

        N = len(self.h)  # Get the size of the spin system

        # Creating the first figure to visualize the cluster spins
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))

        # Plotting cluster spins
        axes[0].imshow(M_overall[all_clusters, ::M_skip], aspect='auto', cmap='viridis')
        axes[0].set_xlabel('number of sweeps', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('cluster index', fontsize=14, fontweight='bold')
        axes[0].tick_params(axis='both', which='major', labelsize=14)

        # Counter for plotting vertical lines and labels at appropriate positions
        counter = 1

        # Vertical lines and labels for the NMC cycles
        for i in range(num_NMC_cycles):
            axes[0].axvline(x=counter * num_sweeps_per_NMC_phase, color='k', linewidth=2)
            axes[0].text(counter * num_sweeps_per_NMC_phase - num_sweeps_per_NMC_phase / 2, -5, 'C', fontsize=14,
                         ha='center', color='red', fontweight='bold')
            counter += 1

            axes[0].axvline(x=counter * num_sweeps_per_NMC_phase, color='k', linewidth=2)
            axes[0].text(counter * num_sweeps_per_NMC_phase - num_sweeps_per_NMC_phase / 2, -5, 'NC', fontsize=14,
                         ha='center', color=(0, 0.5, 0), fontweight='bold')
            counter += 1

            if i % full_update_frequency == 0:
                axes[0].axvline(x=counter * num_sweeps_per_NMC_phase, color='k', linewidth=2)
                axes[0].text(counter * num_sweeps_per_NMC_phase - num_sweeps_per_NMC_phase / 2, -5, 'ALL', fontsize=14,
                             ha='center', color='blue', fontweight='bold')
                counter += 1

        non_cluster_indices = np.setdiff1d(np.arange(N), all_clusters)

        # Plotting non-cluster spins
        axes[1].imshow(M_overall[non_cluster_indices, ::M_skip], aspect='auto', cmap='viridis')
        axes[1].set_xlabel('number of sweeps', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('non-cluster index', fontsize=14, fontweight='bold')
        axes[1].tick_params(axis='both', which='major', labelsize=14)

        # Resetting the counter for the non-cluster plot
        counter = 1

        # Vertical lines and labels for the NMC cycles (similar to above)
        for i in range(num_NMC_cycles):
            axes[1].axvline(x=counter * num_sweeps_per_NMC_phase, color='k', linewidth=2)
            axes[1].text(counter * num_sweeps_per_NMC_phase - num_sweeps_per_NMC_phase / 2, -5, 'C', fontsize=14,
                         ha='center', color='red', fontweight='bold')
            counter += 1

            axes[1].axvline(x=counter * num_sweeps_per_NMC_phase, color='k', linewidth=2)
            axes[1].text(counter * num_sweeps_per_NMC_phase - num_sweeps_per_NMC_phase / 2, -5, 'NC', fontsize=14,
                         ha='center', color=(0, 0.5, 0), fontweight='bold')
            counter += 1

            if i % full_update_frequency == 0:
                axes[1].axvline(x=counter * num_sweeps_per_NMC_phase, color='k', linewidth=2)
                axes[1].text(counter * num_sweeps_per_NMC_phase - num_sweeps_per_NMC_phase / 2, -5, 'ALL', fontsize=14,
                             ha='center', color='blue', fontweight='bold')
                counter += 1

        plt.tight_layout()
        # plt.show()
        plt.savefig('NMC_spins.png')

        # Creating a separate figure for the energy evolution
        ymax = np.percentile(energy_overall, 100)  # Adjust the percentile as needed
        ymin = np.min(energy_overall)
        fig, ax = plt.subplots(figsize=(10, 5))

        # Plotting the energy values
        ax.plot(np.arange(0, len(energy_overall) * M_skip, M_skip), energy_overall)
        ax.set_xlabel('number of sweeps', fontsize=14, fontweight='bold')
        ax.set_ylabel('energy', fontsize=14, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_ylim([ymin, ymax])

        # Resetting the counter for the energy plot
        counter = 1
        label_position = ymin + 0.05 * (ymax - ymin)  # Position labels 5% above the minimum

        # Vertical lines and labels for the NMC cycles (similar to above)
        for i in range(num_NMC_cycles):
            ax.axvline(x=counter * num_sweeps_per_NMC_phase, color='k', linewidth=2)
            ax.text(counter * num_sweeps_per_NMC_phase - num_sweeps_per_NMC_phase / 2, label_position, 'C', fontsize=14,
                    ha='center', color='red', fontweight='bold')
            counter += 1

            ax.axvline(x=counter * num_sweeps_per_NMC_phase, color='k', linewidth=2)
            ax.text(counter * num_sweeps_per_NMC_phase - num_sweeps_per_NMC_phase / 2, label_position, 'NC',
                    fontsize=14, ha='center', color=(0, 0.5, 0), fontweight='bold')
            counter += 1

            if i % full_update_frequency == 0:
                ax.axvline(x=counter * num_sweeps_per_NMC_phase, color='k', linewidth=2)
                ax.text(counter * num_sweeps_per_NMC_phase - num_sweeps_per_NMC_phase / 2, label_position, 'ALL',
                        fontsize=14, ha='center', color='blue', fontweight='bold')
                counter += 1

        plt.tight_layout()
        # plt.show()
        plt.savefig('NMC_energy.png')


def main():
    # Load the coupling and external field matrices, and the list of inverse temperatures
    J = np.load('J.npy')
    h = np.load('h.npy') # 1D array

    # Create an instance of NMC class
    nmc_instance = NMC(J, h)

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


if __name__ == "__main__":
    main()
