from mpi4py import MPI
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from functools import partial
from numba import jit, config
import logging

from stochastic_pgfs.laub_xia_algo import l_x_algo, is_real, in_bounds, _solve_self_consistent_equation, G1_prime, get_outbreak_size
from stochastic_pgfs.random_graphs import poisson_degree_sequence, powerlaw_degree_sequence


# Enable Numba logging
config.DEBUG = False
logging.basicConfig(level=logging.DEBUG)

# Add a test function with Numba
@jit(nopython=True)
def test_numba():
    x = 0
    for i in range(100):
        x += i
    return x

date = datetime.today().strftime('%m-%d-%Y')
logging.basicConfig(level=logging.INFO)

def calculate_critical_transition(my_degree_sequence):
    pk = np.vstack((np.arange(0, my_degree_sequence.shape[0], 1), my_degree_sequence)).T
    return 1 / G1_prime(1, pk, 1)

def process_data(lmbd, T, degree_sequence_func, lx_func):
    my_degree_sequence = degree_sequence_func(lmbd)
    outbreak_size = get_outbreak_size(my_degree_sequence, T)
    return {'lmbd': lmbd, 'T': T, 'sce': lx_func(my_degree_sequence, T=T), 'outbreak_size': outbreak_size}

N_max = 500  # Maximum value for N in the distribution
my_K = int(1e5)  # number of samples per SCE estimate
max_iter = int(1e5)
tol = 1e-5

# params to sweep over
T_vals = np.linspace(0.001, 1, 60)
alpha_vals = np.linspace(1.8, 4, 30)
lmbd_vals = np.linspace(0.001, 2, 30)

lx_additive = partial(l_x_algo, K=my_K, conditions=[is_real, in_bounds], is_pgf=True, 
                      perturbation_type='additive', max_iter=max_iter, tol=tol)
lx_multiplicative = partial(l_x_algo, K=my_K, conditions=[is_real, in_bounds], is_pgf=True, 
                            perturbation_type='multiplicative', max_iter=max_iter, tol=tol)

poisson_degree_sequence_partial = partial(poisson_degree_sequence, N_max=N_max)
powerlaw_degree_sequence_partial = partial(powerlaw_degree_sequence, N_max=N_max)

dist_dict = {'poisson': poisson_degree_sequence_partial, 'powerlaw': powerlaw_degree_sequence_partial}
noise_dict = {'additive': lx_additive}
control_params_dict = {'poisson': lmbd_vals, 'powerlaw': alpha_vals}

def worker_task(control_param, T_vals_plus_crit, dist_func, noise_func):
    # Processes a single control_param for all T in T_vals_plus_crit
    results = []
    for T in T_vals_plus_crit:
        results.append(process_data(control_param, T, dist_func, noise_func))
    return results

if __name__ == '__main__':
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    print("Testing Numba compilation on rank", rank, "...")
    # First call compiles the function
    result1 = test_numba()
    # Second call uses compiled version
    result2 = test_numba()
    if rank == 0:
        print("Numba test complete")

    data_dict_list = []

    # We will loop over distributions and noises. Each rank will process a subset of control parameters
    for dist_name, dist_func in dist_dict.items():
        for noise_name, noise_func in noise_dict.items():
            control_param_vals = control_params_dict[dist_name]

            # Distribute control_param_vals among MPI ranks
            # np.array_split returns a list of arrays, one per rank
            params_chunks = np.array_split(control_param_vals, size)
            params_for_this_rank = params_chunks[rank]

            local_results = []
            for control_param in params_for_this_rank:
                my_dist = dist_func(control_param)
                critical_value = calculate_critical_transition(my_dist)
                T_vals_plus_crit = np.concatenate([T_vals, np.array([critical_value])])
                T_vals_plus_crit = np.sort(T_vals_plus_crit)
                # Filter out invalid values
                T_vals_plus_crit = T_vals_plus_crit[(T_vals_plus_crit > 0) & (T_vals_plus_crit < 1)]
                logging.info(f"Rank {rank}, Control Param: {control_param}, T_vals_plus_crit: {T_vals_plus_crit}")

                # Directly call worker_task on this rank
                local_results.extend(worker_task(control_param, T_vals_plus_crit, dist_func, noise_func))

            # Gather all results at rank 0
            all_results = comm.gather(local_results, root=0)

            if rank == 0:
                # Flatten list of lists
                all_results = [item for sublist in all_results for item in sublist]
                data_dict_list.extend(all_results)

                # After processing all control_param_vals for this dist_name and noise_name pair
                df = pd.DataFrame(data_dict_list)
                df = df.explode(['lmbd', 'sce', 'outbreak_size'])
                df.outbreak_size = df.outbreak_size.apply(lambda x: x.real)
                df.to_csv(f"data/random_graphs/random_graphs_sweep_{dist_name}_{noise_name}.csv")
