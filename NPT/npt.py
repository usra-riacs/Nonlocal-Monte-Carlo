import random
import time
from concurrent.futures import ProcessPoolExecutor
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix
from cachetools import LRUCache
from random import randint
from collections import defaultdict


# np.random.seed(12624755)  # Set the seed for reproducibility

class NPT:
    """
    The NPT class is used to implement the Non-equilibrium Monte Carlo (NMC) + Adaptive Parallel Tempering (APT)
    algorithm.
    """

    def __init__(self, J, h):
        """
        Initialize an NPT object.
        :param J: A 2D numpy array representing the coupling matrix (weights J).
        :param h: A 1D numpy array or list representing the external field (biases h).
        """
        self.J = J
        self.h = h
        self.h = np.asarray(h).reshape(-1)  # Reshape h into a 1D array

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

    def MCMC(self, num_sweeps, m_start, beta, J, h, anneal=False, sweeps_per_beta=1, initial_beta=0,
             hash_table=None, use_hash_table=False):
        """
        Implements the Markov Chain Monte Carlo (MCMC) method using Gibbs sampling.

        Parameters:
        - num_sweeps (int): Number of MCMC sweeps to be performed.
        - m_start (np.array[N,]): Initial seed value of the states, where N is the size of the graph.
        - beta (float): Inverse temperature. Use the maximum value if anneal is set to True.
        - J (np.array[N, N]): Weight matrix where N is the size of the graph.
        - h (np.array[N,]): Bias values where N is the size of the graph.
        - anneal (bool, default=False): Set to True for annealing, else False.
        - sweeps_per_beta (int, optional, default=1): Number of sweeps to perform at each beta level during annealing.
        - initial_beta (float, optional, default=0): Initial value for beta when annealing.
        - hash_table (cachetools.LRUCache, optional): An LRUCache object for storing previously computed dE values.
        - use_hash_table (bool, optional, default=False): If set to True, utilizes the hash table for caching results.

        Returns:
        - M (np.array[N, num_sweeps]): Matrix containing all the sweeps in bipolar form.
        """

        N = J.shape[0]
        m = np.asarray(m_start).copy().reshape(-1,
                                               1)  # Make sure m_star is a numpy array and has shape (N, 1) and also create a copy
        M = np.zeros((N, num_sweeps))
        J = csr_matrix(J)
        h = np.asarray(h).copy().reshape(-1, 1)  # Make sure h is a numpy array and has shape (N, 1)

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

            spin_state = tuple(m.ravel())

            for kk in np.random.permutation(N):
                if use_hash_table:
                    if not isinstance(hash_table, LRUCache):
                        raise ValueError("hash_table must be an instance of cachetools.LRUCache")

                    if spin_state in hash_table:
                        dE = hash_table[spin_state]
                    else:
                        dE = J.dot(m) + h
                        hash_table[spin_state] = dE

                    m[kk] = np.sign(np.tanh(beta_run[jj] * dE[kk]) - 2 * np.random.rand() + 1)
                else:
                    x = J.dot(m) + h
                    m[kk] = np.sign(np.tanh(beta_run[jj] * x[kk]) - 2 * np.random.rand() + 1)

            M[:, jj] = m.ravel()

        return M


    def MCMC_task(self, replica_i, num_sweeps_MCMC, m_start, beta_list, use_hash_table=False, hash_table=None):
        """
        Perform a Monte Carlo simulation for a single task.

        This method is designed to be run in a separate process.

        :param replica_i: An integer representing the replica index.
        :param m_start: A 1D numpy array representing the initial state.
        :param beta_list: A 1D numpy array representing the inverse temperatures for the replicas.
        :param num_sweeps_MCMC: An integer representing the number of MCMC sweeps.
        :param use_hash_table: A boolean flag. If True, a hash table will be used for caching results. (default = 0)
        :param hash_table (LRUCache, optional): A LRUCache object for storing previously computed dE values.
        """

        return self.MCMC(num_sweeps_MCMC, m_start.copy(), beta_list[replica_i - 1], self.J, self.h,
                         hash_table=hash_table, use_hash_table=use_hash_table)

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
        Find clusters based on magnetizations and given thresholds. Can be easily modified to use h_tilde or J_tilde
        from LBP for thresholding.

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
        J_c[all_clusters, :] = J_c[all_clusters, :] / temp_x  # clusters run at higher temperature
        h_c[all_clusters] /= temp_x

        # Modify J and h for non-clusters
        J_nc = self.J.copy()  # non-clusters run at normal temperature
        h_nc = self.h.copy()

        # Create containers for storing results
        M_overall = np.zeros((N, num_sweeps_per_NMC_phase * num_cycles * 3 // M_skip))
        energy_overall = np.zeros(num_sweeps_per_NMC_phase * num_cycles * 3 // M_skip)

        M_index = 0  # Index to keep track of position in M_overall and energy_overall

        for cycle in range(num_cycles):
            # print(f'\nCurrent iteration = {cycle + 1}')

            h_c[non_clusters] = m_init[non_clusters] * 10000  # Strongly bias the non-cluster spins to keep them frozen
            # caution: hash_table is not used as J and h are being scaled by temp_x.
            M = self.MCMC(num_sweeps_per_NMC_phase, m_init.copy(), global_beta, J_c, h_c,
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

            h_nc[all_clusters] = m_init[
                                     all_clusters] * 10000  # Strongly bias the cluster (backbones) spins to keep them frozen

            # caution: hash_table is not used as h_nc is not same as h
            M = self.MCMC(num_sweeps_per_NMC_phase, m_init.copy(), global_beta, J_nc, h_nc,
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
                M = self.MCMC(num_sweeps_per_NMC_phase, m_init.copy(), global_beta, self.J, self.h,
                              anneal=False, hash_table=hash_table, use_hash_table=use_hash_table)
                energies = [- (M[:, i].T @ self.J @ M[:, i] / 2 + M[:, i].T @ self.h) for i in range(M.shape[1])]

                # Store results
                M_overall[:, M_index:M_index + num_sweeps_per_NMC_phase // M_skip] = M[:, ::M_skip]
                energy_overall[M_index:M_index + num_sweeps_per_NMC_phase // M_skip] = energies[::M_skip]

                M_index += num_sweeps_per_NMC_phase // M_skip
                min_energy_idx = np.argmin(energies)
                m_init = M[:, min_energy_idx]

        M_overall = M_overall[:, :M_index]
        energy_overall = energy_overall[:M_index]
        min_energy = np.min(energy_overall)

        return M_overall, energy_overall, min_energy, all_clusters

    def NMC_task(self, m_start, num_cycles, num_sweeps_per_NMC_phase, full_update_frequency, M_skip, global_beta,
                 temp_x, lambda_start, lambda_end, lambda_reduction_factor, threshold_initial, threshold_cutoff,
                 max_iterations, tolerance, use_hash_table=False, hash_table=None):
        """
        This is a wrapper function to call NMC_subroutine for parallel processing.
        This method is designed to be run in a separate process.

        :param m_start: np.ndarray - A 1D numpy array representing the initial state.
        :param num_cycles: int - Number of NMC cycles on each chain.
        :param num_sweeps_per_NMC_phase: int - Number of sweeps per phase in the Non-Local Monte Carlo.
        :param full_update_frequency: int - Frequency of all spin updates in NMC.
        :param M_skip: int - Interval for storing results during the simulation.
        :param global_beta: float - Global inverse temperature for MCMC and NMC.
        :param temp_x: float - Heated temperature factor for NMC (scales beta by dividing by temp_x).
        :param lambda_start: float - Initial lambda value for convexified LBP.
        :param lambda_end: float - Ending lambda value for for convexified LBP.
        :param lambda_reduction_factor: float - Factor by which lambda is reduced at each LBP run.
        :param threshold_initial: float - Initial threshold of marginals for growing backbones seeds.
        :param threshold_cutoff: float - Ending threshold of marginals for backbones.
        :param max_iterations: int - Maximum iterations for LBP to converge.
        :param tolerance: float - Tolerance for convergence in LBP.
        :param use_hash_table: bool - Whether to use a hash table for caching results (default = False).
        :param hash_table: LRUCache, optional - A LRUCache object for caching previously computed dE values (default = None).

        :return: Returns the result of the NMC_subroutine.
        """
        M_overall, energy_overall, min_energy, all_clusters = self.NMC_subroutine(
            m_start, num_cycles, num_sweeps_per_NMC_phase, full_update_frequency, M_skip, global_beta,
            temp_x, lambda_start, lambda_end, lambda_reduction_factor, threshold_initial, threshold_cutoff,
            max_iterations, tolerance, hash_table=hash_table, use_hash_table=use_hash_table
        )

        # Return the result, or modify as needed.
        return M_overall

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

    def run(self, beta_list, num_replicas, doNMC, num_sweeps_MCMC=1000, num_sweeps_read=1000, num_swap_attempts=100,
            num_swapping_pairs=1, num_cycles=10, full_update_frequency=1, M_skip=1, temp_x=20,
            global_beta=2.5, lambda_start=0.5, lambda_end=0.01, lambda_reduction_factor=0.9,
            threshold_initial=0.999999, threshold_cutoff=0.99999, max_iterations=100, tolerance=np.finfo(float).eps,
            use_hash_table=False, num_cores=8):
        """
        Run the NPT algorithm.
        APT parameters
        :param beta_list: A 1D numpy array representing the inverse temperatures for the replicas.
        :param num_replicas: An integer, the number of replicas (parallel chains) to use in the algorithm.
        :param num_sweeps_MCMC: An integer, the number of Monte Carlo sweeps to perform (default =1000) before a swap.
        :param num_sweeps_read: An integer, the number of last sweeps to read from the chains (default =1000) before a swap.
        :param num_swap_attempts: An integer, the number of swap attempts between chains (default = 100).
        :param num_swapping_pairs: An integer, the number of non-overlapping replica pairs per swap attempt (default =1).

        NMC parameters
        :param doNMC (list of bool): List of booleans indicating if a specific replica should use the NMC task.
                    True to use NMC, False to use MCMC.
                    The length of doNMC should match num_replicas.
        :param num_cycles (int): number of NMC cycles on each chain.
        :param full_update_frequency (int): Frequency of all spin updates in NMC.
        :param M_skip (int): Interval for storing results.
        :param temp_x (int): Heated temperature factor for NMC (scales beta by dividing by temp_x)
        :param global_beta (float): Global inverse temperature for MCMC and NMC.
        :param lambda_start (float): Initial lambda value for convexified LBP.
        :param lambda_end (float): Ending lambda value for for convexified LBP.
        :param lambda_reduction_factor (float): Factor by which lambda is reduced at each LBP run.
        :param threshold_initial (float): Initial threshold of marginals for growing backbones seeds
        :param threshold_cutoff (float): Ending threshold of marginals for backbones.
        :param max_iterations (int): Maximum iterations for LBP to converge
        :param tolerance (float): Tolerance for convergence in LBP.
        :param use_hash_table: Whether to use a hash table or not (default =False).
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
        self.num_sweeps_per_NMC_phase_per_swap = int(
            np.ceil(self.num_sweeps_MCMC / self.num_swap_attempts / 3 / num_cycles))
        self.num_swapping_pairs = num_swapping_pairs
        self.use_hash_table = use_hash_table
        self.doNMC = doNMC

        # Boolean flag to decide normalization
        normalize = True
        # Normalize energy with |J_ij| ~ 1
        norm_factor = np.max(np.abs(self.J)) if normalize else 1
        self.J = self.J / norm_factor
        self.h = self.h / norm_factor

        # Check if doNMC is of the correct length
        if len(self.doNMC) != self.num_replicas:
            raise ValueError("The length of doNMC does not match the number of replicas.")

        # If use_hash_table is True, create a new hash table for this process
        if self.use_hash_table:
            self.hash_table = LRUCache(maxsize=10000)
        else:
            self.hash_table = None

        num_spins = self.J.shape[0]
        count = np.zeros(self.num_swap_attempts)
        swap_attempted_replicas = np.zeros((self.num_swap_attempts * self.num_swapping_pairs, 2))
        swap_accepted_replicas = np.zeros((self.num_swap_attempts * self.num_swapping_pairs, 2))

        # Generate all possible consecutive pairs of replicas
        all_pairs = [(i, i + 1) for i in range(1, self.num_replicas)]

        # Initialize states for all replicas
        M = np.zeros((self.num_replicas * num_spins, self.num_sweeps_MCMC_per_swap))
        m_start = np.sign(2 * np.random.rand(self.num_replicas * num_spins, 1) - 1)

        swap_index = 0

        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            for ii in range(self.num_swap_attempts):
                print(f"\nRunning swap attempt = {ii + 1}")
                start_time = time.time()

                futures = []
                for replica_i in range(1, self.num_replicas + 1):
                    if not self.doNMC[replica_i - 1]:
                        print(f"\nReplica {replica_i}/{self.num_replicas} is running MCMC\n")
                        future = executor.submit(self.MCMC_task, replica_i, self.num_sweeps_MCMC_per_swap,
                                                 m_start[(replica_i - 1) * num_spins:replica_i * num_spins].copy(),
                                                 beta_list, self.use_hash_table, self.hash_table)
                    else:
                        print(f"\nReplica {replica_i}/{self.num_replicas} is running NMC\n")
                        future = executor.submit(self.NMC_task,
                                                 m_start[
                                                 (replica_i - 1) * num_spins:replica_i * num_spins].copy().flatten(),
                                                 num_cycles, self.num_sweeps_per_NMC_phase_per_swap,
                                                 full_update_frequency,
                                                 M_skip, global_beta, temp_x, lambda_start, lambda_end,
                                                 lambda_reduction_factor, threshold_initial, threshold_cutoff,
                                                 max_iterations, tolerance, self.use_hash_table, self.hash_table)
                    futures.append(future)

                M_results = [future.result() for future in futures]

                for replica_i, M_replica in enumerate(M_results, start=1):
                    M_replica = M_replica[:, -self.num_sweeps_MCMC_per_swap:]
                    M[(replica_i - 1) * num_spins:replica_i * num_spins, :] = M_replica.copy()

                mm = M[:, -self.num_sweeps_read_per_swap:].copy().T
                m_start = M[:, -1].copy().reshape(-1, 1)

                selected_pairs = self.select_non_overlapping_pairs(all_pairs)

                # Attempt to swap states of each selected pair of replicas
                for pair in selected_pairs:
                    sel, next = pair
                    m_sel = mm[-1, (sel - 1) * num_spins:sel * num_spins].copy().T
                    m_next = mm[-1, (next - 1) * num_spins:next * num_spins].copy().T

                    E_sel = -m_sel.T @ self.J @ m_sel / 2 - m_sel.T @ self.h
                    E_next = -m_next.T @ self.J @ m_next / 2 - m_next.T @ self.h
                    beta_sel = beta_list[sel - 1]
                    beta_next = beta_list[next - 1]

                    print(f"\nSelected pair indices: {sel}, {next}")
                    print(f"β values: {beta_sel}, {beta_next}")
                    print(f"Energies: {E_sel}, {E_next}")

                    swap_attempted_replicas[swap_index, :] = [sel, next]

                    DeltaE = E_next - E_sel
                    DeltaB = beta_next - beta_sel

                    if np.random.rand() < min(1, np.exp(DeltaB * DeltaE)):
                        count[ii] += 1
                        swap_accepted_replicas[swap_index, :] = [sel, next]
                        print(f"Swapping {int(sum(count))}th time")

                        # Swap the states of the selected replicas
                        m_start[(sel - 1) * num_spins:sel * num_spins] = m_next.copy().reshape(-1, 1)
                        m_start[(next - 1) * num_spins:next * num_spins] = m_sel.copy().reshape(-1, 1)

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
        plt.savefig('NPT_energy.png')


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
    npt = NPT(J, h)

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


if __name__ == '__main__':
    main()
