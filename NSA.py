import networkx as nx
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from cachetools import LRUCache
from collections import defaultdict

# Setting random seed for reproducibility
np.random.seed(123)


class NSA:
    """
    The NSA class is used to implement the Non-equilibrium Monte Carlo (NMC) + Simulated Annealing hybrid (NSA) algorithm.
    """

    def __init__(self, J, h):
        """
        Initialize an NSA object.
        :param J: A 2D numpy array representing the coupling matrix (weights J).
        :param h: A 1D numpy array or list representing the external field (biases h).
        """
        self.J = J
        self.h = h

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
        m = np.asarray(m_start).copy().reshape(-1,
                                               1)  # Make sure m_star is a numpy array and has shape (N, 1) and also create a copy
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
            if u_change < tolerance:
                break

        # Compute magnetizations and correlations
        magnetizations = np.tanh(beta * (h + np.sum(u_msgs, axis=0)))
        correlations = (np.tanh(beta * J) + np.tanh(beta * h_msgs) * np.tanh(beta * h_msgs.T)) / \
                       (1 + np.tanh(beta * J) * np.tanh(beta * h_msgs) * np.tanh(beta * h_msgs.T))

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

    def NMC_subroutine_annealed(self, m_star, num_cycles, num_sweeps_per_NMC_phase, full_update_frequency, M_skip,
                                global_beta,
                                temp_x, lambda_start, lambda_end, lambda_reduction_factor, threshold_initial,
                                threshold_cutoff,
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
            print(f'\nCurrent NMC iteration = {cycle + 1}')
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
            # Note: the following two lines are not needed anymore since we are annealing in backbones heating starting at global_beta/temp_x
            # J_c[all_clusters, :] = J_c[all_clusters, :] / temp_x
            # h_c[all_clusters] /= temp_x
            h_c[non_clusters] = m_init[
                                    non_clusters] * 10000 * temp_x  # Strongly bias the non-cluster spins to keep
            # caution: hash_table is not used as J and h are being scaled by temp_x.
            M = self.MCMC_GC(num_sweeps_per_NMC_phase, m_init.copy(), global_beta, J_c, h_c, self.colorMap,
                             anneal=True, sweeps_per_beta=round(num_sweeps_per_NMC_phase / 20),
                             initial_beta=global_beta / temp_x, hash_table=hash_table,
                             use_hash_table=False)  # anneal for clusters starting at temp_x higher than global_beta
            energies = [- (M[:, i].T @ self.J @ M[:, i] / 2 + M[:, i].T @ self.h) for i in
                        range(M.shape[1])]  # Compute energies

            # Store results
            M_overall[:, M_index:M_index + num_sweeps_per_NMC_phase // M_skip] = M[:, ::M_skip]
            energy_overall[M_index:M_index + num_sweeps_per_NMC_phase // M_skip] = energies[::M_skip]

            M_index += num_sweeps_per_NMC_phase // M_skip
            min_energy_idx = np.argmin(energies)
            m_init = M[:, min_energy_idx]

            # Modify J and h for non-clusters
            J_nc = self.J.copy()
            h_nc = self.h.copy()
            h_nc[all_clusters] = m_init[
                                     all_clusters] * 10000  # Strongly bias the cluster (backbones) spins to keep them frozen

            # caution: hash_table is not used as h_nc is not same as h
            M = self.MCMC_GC(num_sweeps_per_NMC_phase, m_init.copy(), global_beta, J_nc, h_nc, self.colorMap,
                             anneal=True, sweeps_per_beta=round(num_sweeps_per_NMC_phase / 20),
                             initial_beta=global_beta * 0.8, hash_table=hash_table,
                             use_hash_table=False)  # anneal for non-clusters starting at 80% higher than global_beta

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
                                 anneal=True, sweeps_per_beta=round(num_sweeps_per_NMC_phase / 20),
                                 initial_beta=global_beta * 0.8, hash_table=hash_table,
                                 use_hash_table=use_hash_table)  # anneal for all spins starting at 80% higher than global_beta
                energies = [- (M[:, i].T @ self.J @ M[:, i] / 2 + M[:, i].T @ self.h) for i in range(M.shape[1])]

                # Store results
                M_overall[:, M_index:M_index + num_sweeps_per_NMC_phase // M_skip] = M[:, ::M_skip]
                energy_overall[M_index:M_index + num_sweeps_per_NMC_phase // M_skip] = energies[::M_skip]

                M_index += num_sweeps_per_NMC_phase // M_skip
                Energy_star = np.min(energies)
                min_energy_idx = np.argmin(energies)
                m_init = M[:, min_energy_idx]

                m_star = m_init.copy()
                # print(f'\ncurrent m_star energy = {Energy_star:.8f}')

        M_overall = M_overall[:, :M_index]
        energy_overall = energy_overall[:M_index]
        min_energy = np.min(energy_overall)

        return M_overall, energy_overall, min_energy, all_clusters

    def run(self, num_cycles=2, full_update_frequency=1, M_skip=1,
            lambda_start=0.5, lambda_end=0.01, lambda_reduction_factor=0.9, temp_x=30,
            threshold_initial=0.999999, threshold_cutoff=1, tolerance=np.finfo(float).eps, max_iterations=1000,
            global_beta=3, pure_SA_portion=1 / 3, SA_portion_in_NMC=0.25, num_SA_betas=45, num_NMC_betas=5,
            total_num_sweeps=int(3e4), use_hash_table=False):
        """
        Execute the NSA algorithm to solve for optimal states and energy.

        Parameters:
        - num_cycles (int, default=2): Total number of NMC cycles at each NMC beta.
        - full_update_frequency (int, default=1): Frequency at which all spins are updated in NMC.
        - M_skip (int, default=1): Interval between results storage.
        - temp_x (int, default=30): Heated temperature factor for NSA (scales beta by dividing by temp_x).
        - global_beta (float, default=3): Global inverse temperature for SA and NMC.
        - lambda_start (float): Initial lambda value for convexified LBP.
        - lambda_end (float): Ending lambda value for for convexified LBP.
        - lambda_reduction_factor (float): Factor by which lambda is reduced at each LBP run.
        - threshold_initial (float): Initial threshold of marginals for growing backbones seeds
        - threshold_cutoff (float): Ending threshold of marginals for backbones.
        - max_iterations (int, default=1000): Maximum iterations for LBP to converge.
        - tolerance (float, default=np.finfo(float).eps): Tolerance for convergence in LBP.
        - pure_SA_portion (float, default=1/3): Proportion of the run that's pure simulated annealing.
        - SA_portion_in_NMC (float, default=0.25): Proportion of initial SA in each NMC cycle.
        - num_SA_betas (int, default=45): Number of betas in the SA portion.
        - num_NMC_betas (int, default=5): Number of betas in the NMC portion.
        - total_num_sweeps (int, default=3e4): Total number of sweeps performed (SA+NMC).
        - use_hash_table (bool, optional): If True, the hash table will be used for caching results.

        Returns:
        - Tuple: M_overall, energy_overall
        """

        # Boolean flag to decide normalization
        normalize = True
        # Normalize energy with |J_ij| ~ 1
        norm_factor = np.max(np.abs(self.J)) if normalize else 1
        self.J = self.J / norm_factor
        self.h = self.h / norm_factor

        # Setup hyperparameters for the simulated annealing and NMC process
        NMC_portion_in_NMC = 1 - SA_portion_in_NMC
        total_num_sweeps_SA = int(pure_SA_portion * total_num_sweeps)
        total_num_sweeps_NMC = total_num_sweeps - total_num_sweeps_SA

        sweeps_per_beta_SA = round(total_num_sweeps_SA / num_SA_betas)
        sweeps_per_beta_SA_b4_NMC = round(SA_portion_in_NMC * total_num_sweeps_NMC / num_NMC_betas)
        sweeps_per_beta_NMC = round(NMC_portion_in_NMC * total_num_sweeps_NMC / num_NMC_betas)
        sweeps_per_beta_NMC_per_phase = round(sweeps_per_beta_NMC / 3 / num_cycles)

        num_betas = num_SA_betas + num_NMC_betas
        beta_vals = np.linspace(1 / total_num_sweeps, global_beta, num_betas)

        # Initialize random initial state for spins
        m_init = np.sign(2 * np.random.rand(len(self.J)) - 1)

        # Arrays to store magnetization and energy values
        N = len(self.h)
        M_overall = np.zeros((N, num_SA_betas * sweeps_per_beta_SA + num_NMC_betas * (
                sweeps_per_beta_SA_b4_NMC + 3 * num_cycles * sweeps_per_beta_NMC_per_phase)))
        energy_overall = np.zeros(num_SA_betas * sweeps_per_beta_SA + num_NMC_betas * (
                sweeps_per_beta_SA_b4_NMC + 3 * num_cycles * sweeps_per_beta_NMC_per_phase))

        sweep_index = 0

        # If use_hash_table is True, create a new hash table for this process
        if use_hash_table:
            hash_table = LRUCache(maxsize=10000)
        else:
            hash_table = None

        # Main loop iterating over beta values
        for beta_idx in range(num_betas):
            print(f'\nProgress: Beta {beta_idx + 1}/{num_betas}. Current beta = {beta_vals[beta_idx]:.8f}')

            beta = beta_vals[beta_idx]

            # If still in the SA-only phase
            if beta_idx < num_SA_betas:
                # Perform pure SA and save results
                M_SA = self.MCMC_GC(sweeps_per_beta_SA, m_init, beta, self.J, self.h, self.colorMap, anneal=False,
                                    hash_table=hash_table,
                                    use_hash_table=use_hash_table)
                current_sweep_count = M_SA.shape[1]
                M_overall[:, sweep_index: sweep_index + current_sweep_count] = M_SA
                energy_SA = np.array([-np.dot(m.T, self.J).dot(m) / 2 - np.dot(m.T, self.h) for m in M_SA.T])
                energy_overall[sweep_index: sweep_index + current_sweep_count] = energy_SA
                sweep_index += current_sweep_count
                m_init = M_SA[:, -1]
                print(f'\nm_star energy after full SA = {energy_SA[-1]:.8f}')

            else:
                # Perform partial SA before NMC and then NMC and save results
                M_SA = self.MCMC_GC(sweeps_per_beta_SA_b4_NMC, m_init, beta, self.J, self.h, self.colorMap,
                                    anneal=False,
                                    hash_table=hash_table,
                                    use_hash_table=use_hash_table)
                current_sweep_count = M_SA.shape[1]
                M_overall[:, sweep_index: sweep_index + current_sweep_count] = M_SA
                energy_SA = np.array([-np.dot(m.T, self.J).dot(m) / 2 - np.dot(m.T, self.h) for m in M_SA.T])
                energy_overall[sweep_index: sweep_index + current_sweep_count] = energy_SA
                sweep_index += current_sweep_count
                m_init = M_SA[:, np.argmin(energy_SA)]
                print(f'\nm_star energy after partial SA before NMC = {min(energy_SA):.8f}')

                # annealing the hyperparameters for backbones (to find larger to smaller backbones as we cool down in SA)
                if beta_idx != num_SA_betas:
                    temp_x *= 0.9
                    threshold_cutoff += (1 - threshold_cutoff) * 0.8

                M_NMC, _, _, _ = self.NMC_subroutine_annealed(m_init, num_cycles,
                                                              sweeps_per_beta_NMC_per_phase, full_update_frequency,
                                                              M_skip, beta, temp_x, lambda_start, lambda_end,
                                                              lambda_reduction_factor, threshold_initial,
                                                              threshold_cutoff, max_iterations, tolerance,
                                                              hash_table=hash_table, use_hash_table=use_hash_table)
                current_sweep_count = M_NMC.shape[1]
                M_overall[:, sweep_index: sweep_index + current_sweep_count] = M_NMC
                energy_NMC = np.array([-np.dot(m.T, self.J).dot(m) / 2 - np.dot(m.T, self.h) for m in M_NMC.T])
                energy_overall[sweep_index: sweep_index + current_sweep_count] = energy_NMC
                sweep_index += current_sweep_count
                m_init = M_NMC[:, np.argmin(energy_NMC)]
                print(f'\nm_star energy after NMC = {min(energy_NMC):.8f}')

        self.plot_results(energy_overall, beta_vals, sweeps_per_beta_SA, num_SA_betas, sweeps_per_beta_SA_b4_NMC,
                          sweeps_per_beta_NMC_per_phase, num_NMC_betas, num_cycles, total_num_sweeps)

        return M_overall, energy_overall

    def plot_results(self, energy_overall, beta_vals, sweeps_per_beta_SA, num_SA_betas, sweeps_per_beta_SA_b4_NMC,
                     sweeps_per_beta_NMC_per_phase, num_NMC_betas, num_cycles, total_num_sweeps):
        """
        Plot the results of the NSA algorithm.

        Parameters:
        - energy_overall (array): Energy data for each sweep.
        - beta_vals (array): NSA annealing schedule.
        - sweeps_per_beta_SA (int): Number of sweeps for each beta during Simulated Annealing phase.
        - num_SA_betas (int): Number of betas for the Simulated Annealing phase.
        - sweeps_per_beta_SA_b4_NMC (int): Number of sweeps for each beta ahead of each NMC phase.
        - sweeps_per_beta_NMC_per_phase (int): Number of sweeps for each NMC phase at each beta.
        - num_NMC_betas (int): Number of betas for the NMC phase.
        - num_cycles (int): Number of NMC cycles at each beta.
        - total_num_sweeps (int): Total number of sweeps (SA+NMC).

        Returns:
        None. This method shows plots as output.
        """

        # Calculate the upper y limit
        ymax = np.percentile(energy_overall, 95)  # Adjust the percentile as per your needs
        ymin = np.min(energy_overall)

        # Prepare x-axis labels
        beta_axis = np.zeros(len(energy_overall))
        sweeps_cumulative = np.cumsum(
            [val for sublist in
             [sweeps_per_beta_SA * np.ones(num_SA_betas),
              (sweeps_per_beta_SA_b4_NMC + 3 * num_cycles * sweeps_per_beta_NMC_per_phase) * np.ones(num_NMC_betas)]
             for val in sublist]
        )

        for i in range(len(beta_vals)):
            if i == 0:
                beta_axis[0:int(sweeps_cumulative[i])] = beta_vals[i]
            else:
                beta_axis[int(sweeps_cumulative[i - 1]):int(sweeps_cumulative[i])] = beta_vals[i]

        tickskip = int(total_num_sweeps / 10)

        # Plotting Energy
        plt.figure()
        plt.plot(energy_overall)
        plt.xlabel('Beta', fontsize=14, fontweight='bold')
        plt.ylabel('Energy', fontsize=14, fontweight='bold')
        plt.xticks(ticks=np.arange(0, len(energy_overall), tickskip), labels=np.round(beta_axis[::tickskip], 2))
        plt.ylim([ymin, ymax])  # set the y limit
        plt.xlim([0, len(energy_overall)])  # set the x limit
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def main():
    # Load J and h from .npy files
    # J = np.load('J.npy')
    # h = np.load('h.npy')

    # Assumes that the keys in mat files are 'J' and 'h'
    J = loadmat('JJ.mat')['J']
    h = loadmat('h.mat')['h']
    h = np.asarray(h).copy().reshape(-1)  # Reshape h into a 1D array

    # Define parameters as variables for easy access
    num_cycles = 2
    full_update_frequency = 1
    M_skip = 1
    lambda_start = 0.5
    lambda_end = 0.01
    lambda_reduction_factor = 0.9
    temp_x = 30
    threshold_initial = 0.999999
    threshold_cutoff = 1
    tolerance = np.finfo(float).eps
    max_iterations = 1000
    global_beta = 3
    pure_SA_portion = 1 / 3
    SA_portion_in_NMC = 0.25
    num_SA_betas = 45
    num_NMC_betas = 5
    total_num_sweeps = int(3e4)
    use_hash_table = True  # Flag to decide whether to use hash table

    # Create an instance of the NSA class
    nsa_instance = NSA(J, h)

    # Call the run method with parameters
    nsa_instance.run(num_cycles=num_cycles,
                     full_update_frequency=full_update_frequency,
                     M_skip=M_skip,
                     lambda_start=lambda_start,
                     lambda_end=lambda_end,
                     lambda_reduction_factor=lambda_reduction_factor,
                     temp_x=temp_x,
                     threshold_initial=threshold_initial,
                     threshold_cutoff=threshold_cutoff,
                     tolerance=tolerance,
                     max_iterations=max_iterations,
                     global_beta=global_beta,
                     pure_SA_portion=pure_SA_portion,
                     SA_portion_in_NMC=SA_portion_in_NMC,
                     num_SA_betas=num_SA_betas,
                     num_NMC_betas=num_NMC_betas,
                     total_num_sweeps=total_num_sweeps,
                     use_hash_table=use_hash_table)


if __name__ == "__main__":
    main()
