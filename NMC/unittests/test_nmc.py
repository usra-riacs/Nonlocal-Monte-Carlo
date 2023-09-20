import unittest
import numpy as np
from nmc import NMC
import os


class TestNMC(unittest.TestCase):

    def generate_random_J_h(self, N):
        """Generate random J (adjacency) and h matrices for a given size."""
        h = np.random.randn(N)
        upper_triangle_indices = np.triu_indices(N, 1)
        upper_triangle_values = np.random.randn(len(upper_triangle_indices[0]))
        J = np.zeros((N, N))
        J[upper_triangle_indices] = upper_triangle_values
        J += J.T  # Make it symmetric
        return J, h

    def setUp(self):
        """Setup mock data for tests."""
        self.J, self.h = self.generate_random_J_h(10)  # or any appropriate size
        # Create an instance of NMC class
        self.nmc_instance = NMC(self.J, self.h)

    def test_initialization(self):
        """Test that the class initializes correctly."""
        self.assertIsNotNone(self.nmc_instance)
        self.assertTrue(np.array_equal(self.nmc_instance.J, self.J))
        self.assertTrue(np.array_equal(self.nmc_instance.h, self.h.reshape(-1)))

    def test_run_method(self):
        """Test if the run method generates correct outputs."""
        num_sweeps_initial = int(1e2)
        num_sweeps_per_NMC_phase = int(1e1)
        num_NMC_cycles = 2
        full_update_frequency = 1
        M_skip = 1
        temp_x = 20
        global_beta = 3
        lambda_start = 3
        lambda_end = 0.01
        lambda_reduction_factor = 0.9
        threshold_initial = 0.9999999
        threshold_cutoff = 0.999999
        max_iterations = 10
        tolerance = np.finfo(float).eps
        use_hash_table = False  # Flag to decide whether to use hash table

        M_overall, energy_overall, min_energy = self.nmc_instance.run(num_sweeps_initial, num_sweeps_per_NMC_phase,
                                                                      num_NMC_cycles, full_update_frequency, M_skip,
                                                                      temp_x,
                                                                      global_beta, lambda_start, lambda_end,
                                                                      lambda_reduction_factor, threshold_initial,
                                                                      threshold_cutoff,
                                                                      max_iterations, tolerance,
                                                                      use_hash_table=use_hash_table)
        self.assertTrue(isinstance(M_overall, np.ndarray))
        self.assertTrue(isinstance(energy_overall, (list, np.ndarray)))
        self.assertTrue(isinstance(min_energy, (float, np.float64)))


    def tearDown(self):
        """Cleanup after tests."""
        if os.path.exists('NMC_energy.png'):
            os.remove('NMC_energy.png')
        if os.path.exists('NMC_spins.png'):
            os.remove('NMC_spins.png')


if __name__ == "__main__":
    unittest.main()
