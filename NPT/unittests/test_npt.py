import unittest
import numpy as np
from ..npt import NPT
import os

class TestNPT(unittest.TestCase):

    def generate_random_J_h(self, N):
        """Generate random J (adjacency) and h matrices for a given size."""
        h = np.random.randn(N, 1)
        upper_triangle_indices = np.triu_indices(N, 1)
        upper_triangle_values = np.random.randn(len(upper_triangle_indices[0]))
        J = np.zeros((N, N))
        J[upper_triangle_indices] = upper_triangle_values
        J += J.T  # Make it symmetric
        return J, h

    def setUp(self):
        """Setup mock data for tests."""
        self.N = 10
        self.J, self.h = self.generate_random_J_h(self.N)
        self.npt = NPT(self.J, self.h)

    def test_initialization(self):
        """Test that the class initializes correctly."""
        self.assertIsNotNone(self.npt)
        self.assertTrue(np.array_equal(self.npt.J, self.J))
        self.assertTrue(np.array_equal(self.npt.h, self.h.reshape(-1)))

    def test_run_method(self):
        """Test if the run method generates M and Energy."""
        beta_list = np.array([0.5, 1.0, 1.5, 2.0])  # A basic list; you can adjust
        num_replicas = 4  # Matches the length of beta_list

        # APT parameters
        num_sweeps_MCMC = int(1e2)
        num_sweeps_read = int(1e2)
        num_swap_attempts = int(1e1)
        num_swapping_pairs = round(0.3 * num_replicas)
        use_hash_table = False
        num_cores = 1

        # NMC parameters
        doNMC = [False] * (num_replicas - 2) + [True] * 2
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
        max_iterations = 10
        tolerance = np.finfo(float).eps

        M, Energy = self.npt.run(
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

        # Basic check if the method runs without exceptions and returns expected shapes
        self.assertEqual(M.shape, (self.N * num_replicas, num_sweeps_MCMC // num_swap_attempts))
        self.assertEqual(Energy.shape, (num_replicas,))

    def tearDown(self):
        """Cleanup after tests."""
        if os.path.exists('NPT_energy.png'):
            os.remove('NPT_energy.png')

if __name__ == "__main__":
    unittest.main()
