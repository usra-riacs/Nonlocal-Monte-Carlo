# Nonlocal Monte-Carlo Methods
Non-Equilibrium Monte Carlo Methods, including Adaptive Parallel Tempering Solver

This repository contains the Python implementations of Non-Local Monte-Carlo (NMC) and its enhanced version, Non-Local Monte-Carlo with Adaptive Parallel Tempering (NPT).

This repository contains:

1. The primary non-local [NMC script](https://github.com/usra-riacs/Nonlocal-Monte-Carlo/blob/main/NMC/nmc.py)
2. A [preprocessing script for NPT](https://github.com/usra-riacs/Nonlocal-Monte-Carlo/blob/main/NPT/apt_preprocessor.py)
3. The hybrid APT+NMC [NPT script](https://github.com/usra-riacs/Nonlocal-Monte-Carlo/blob/main/NPT/npt.py)
4. An additional script [apt_ICM](https://github.com/usra-riacs/Nonlocal-Monte-Carlo/blob/main/NPT/apt_ICM.py) for benchmarking NPT with traditional Iso-cluster Move (ICM) coupled with APT.

## Table of Contents

- [Background](#background)
- [Installation](#installation)
- [Usage](#usage)
- [Related Efforts](#related-efforts)
- [Contributors](#contributors)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Background

The Non-Local Monte-Carlo (NMC) method is a powerful algorithmic approach designed for optimization and sampling in complex landscapes such as found in Ising models. The further refined Non-Local Monte-Carlo with Adaptive Parallel Tempering (NPT) enhances this approach, incorporating Adaptive Parallel Tempering (APT). For more details, please see [Non-Equilibrium_Monte_Carlo.pdf](https://github.com/usra-riacs/Nonlocal-Monte-Carlo/blob/main/Non_equilibrium_Monte_Carlo.pdf).



## Installation

### Method 1: Cloning the Repository

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/usra-riacs/Nonlocal-Monte-Carlo.git
    cd Nonlocal-Monte-Carlo
    ```

2. **Set up a Virtual Environment (Recommended)**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `.\venv\Scripts\activate`
    ```

3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### Method 2: Downloading as a Zip Archive

1. **Download the Repository**:
    - Navigate to the [Nonlocal-Monte-Carlo GitHub page](https://github.com/usra-riacs/Nonlocal-Monte-Carlo).
    - Click on the `Code` button.
    - Choose `Download ZIP`.
    - Once downloaded, extract the ZIP archive and navigate to the extracted folder in your terminal or command prompt.

2. **Set up a Virtual Environment (Recommended)**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `.\venv\Scripts\activate`
    ```

3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To employ the provided optimization methods effectively, follow the instructions for each method:

### 1. Using the NMC Method

**Setting Up**

Ensure you've properly installed the necessary dependencies as highlighted in the [Installation](#installation) section.

**Running Non-Local Monte Carlo (NMC)**

To use NMC, first instantiate the `NMC` class with weights and biases `J` and `h`. Here's an example:

```python
from nmc import NMC

# Assuming your J and h are loaded or generated elsewhere
nmc_instance = NMC(J, h)

# Initiate the main NMC run
print("\n[INFO] Starting main NMC process...")
M_overall, energy_overall, min_energy = nmc_instance.run(num_sweeps_initial=int(1e4),
                                                         num_sweeps_per_NMC_phase=int(1e4),
                                                         num_NMC_cycles=10,
                                                         full_update_frequency=1,
                                                         M_skip=1,
                                                         temp_x=20,
                                                         global_beta=3,
                                                         lambda_start=3,
                                                         lambda_end=0.01,
                                                         lambda_reduction_factor=0.9,
                                                         threshold_initial=0.9999999,
                                                         threshold_cutoff=0.999999,
                                                         max_iterations=100,
                                                         tolerance=np.finfo(float).eps,
                                                         use_hash_table=False)

print(f"Minimum Energy: {min_energy:.8f}")
print("\n[INFO] NMC process complete.")
```

### 2. Using the NPT Method

**Setting Up**

Ensure you've properly installed the necessary dependencies as highlighted in the [Installation](#installation) section.


**Preprocessing: generate the inverse temperature (beta) schedule**

Prepare your data for NPT by running the preprocessing code to generate the inverse temperature (beta) schedule and ascertain the number of required replicas:

```python
from apt_preprocessor import APT_preprocessor

# Assuming your J and h are loaded or generated elsewhere
apt_prep = APT_preprocessor(J, h)

apt_prep.run(num_sweeps_MCMC=1000, num_sweeps_read=1000, num_rng=100,
             beta_start=0.5, alpha=1.25, sigma_E_val=1000, beta_max=64, use_hash_table=0, num_cores=8)
```

**Running Non-Local Monte Carlo with Adaptive Parallel Tempering (NPT)**

To use NPT, first instantiate the `NPT` class with weights and biases `J` and `h`. Here's an example:

```python
from npt import NPT

# Assuming your J and h are loaded or generated elsewhere
# Create an NPT instance
npt = NPT(J, h)

# Initiate the main NPT run
print("\n[INFO] Starting main NPT process...")
M, Energy = npt.run(
    beta_list=beta_list,
    num_replicas=beta_list.shape[0],
    doNMC=[False] * (beta_list.shape[0] - 5) + [True] * 5,
    num_sweeps_MCMC=int(1e4),
    num_sweeps_read=int(1e2),
    num_swap_attempts=int(1e1),
    num_swapping_pairs=round(0.3 * beta_list.shape[0]),
    num_cycles=10,
    full_update_frequency=1,
    M_skip=1,
    temp_x=20,
    global_beta=1 / 0.366838 * 5,
    lambda_start=3,
    lambda_end=0.01,
    lambda_reduction_factor=0.9,
    threshold_initial=0.9999999,
    threshold_cutoff=0.999999,
    max_iterations=100,
    tolerance=np.finfo(float).eps,
    use_hash_table=False,
    num_cores=8
)

print(Energy)
print("\n[INFO] NPT process complete.")

```
### 3. Using the APT+ICM method for benchmarking:

After preprocessing same as NPT mentioned above, proceed with the main Adaptive Parallel Tempering + ICM moves:

```python
from apt_ICM import APT_ICM

# Normalize your beta list
beta_list = np.load('beta_list_python.npy')
norm_factor = np.max(np.abs(J))
beta_list = beta_list / norm_factor

apt_ICM = APT_ICM(J, h)
M, Energy = apt_ICM.run(beta_list, num_replicas=beta_list.shape[0],
                    num_sweeps_MCMC=int(1e4),
                    num_sweeps_read=int(1e3),
                    num_swap_attempts=int(1e2),
                    num_swapping_pairs=1, use_hash_table=0, num_cores=8)
```

### 4. Example Script

For full demonstrations of both NMC and NPT in action, refer to the example scripts located in the examples folder of this repository.

## Related Efforts
- [PySA](https://github.com/nasa/PySA) - High Performance NASA QuAIL Python Simulated Annealing code
- [APT-solver](https://github.com/usra-riacs/APT-solver) - High Performance Adaptive Parallel Tempering (APT) code

## Contributors
- [@NavidAadit](https://github.com/navidaadit) Navid Aadit
- [@PAaronLott](https://github.com/PAaronLott) Aaron Lott
- [@Mohseni](https://github.com/mohseni7) Masoud Mohseni

## Acknowledgements

This code was developed under the NSF Expeditions Program NSF award CCF-1918549 on [Coherent Ising Machines](https://cohesing.org/)

## License

[Apache2](LICENSE)
