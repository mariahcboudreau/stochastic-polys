from multiprocessing import Pool
import multiprocessing as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from functools import partial
from numba import jit, config
import logging
from stochastic_pgfs.laub_xia_algo import (
    l_x_algo,
    G1_prime,
    get_outbreak_size
)
from stochastic_pgfs.random_graphs import (
    poisson_degree_sequence, 
    powerlaw_degree_sequence
)


# Enable Numba logging
#config.LOGNAME = "numba.jitclass"
config.DEBUG = False
logging.basicConfig(level=logging.INFO)

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
    logging.info(f"T: {T}")
    logging.info(f"lmbd: {lmbd}")
    my_degree_sequence = degree_sequence_func(lmbd)
    outbreak_size = get_outbreak_size(my_degree_sequence, T)
    return {'lmbd': lmbd, 'T': T, 'sce': lx_func(my_degree_sequence, T=T), 'outbreak_size': outbreak_size}

#N_max = 1000  # Maximum value for N in the distribution
N_max = 100  # Maximum value for N in the distribution
my_K = int(1e3)  # number of samples per SCE estimate
max_iter = int(1e8)
tol = 1e-10


#params to sweep over
# T_vals = np.linspace(0.001, 1, 60)
# alpha_vals = np.linspace(2.1, 3.4,30)
# lmbd_vals = np.linspace(1.1, 2,30)

T_vals = np.linspace(0.001, 1, 10)
alpha_vals = np.linspace(2.1, 3.4,3)
lmbd_vals = np.linspace(1.1, 2,3)


# alpha_vals = np.linspace(2.1, 3.4,1)
# lmbd_vals = np.linspace(1.1, 2,1)


lx_additive = partial(l_x_algo, 
                      max_iter=max_iter,
                      tol = tol,
                      K=my_K, 
                      acceleration_method = 'steffensen',
                      sampling_method = 'orthogonal'
                    )


poisson_degree_sequence_partial = partial(poisson_degree_sequence, N_max=N_max)
powerlaw_degree_sequence_partial = partial(powerlaw_degree_sequence, N_max=N_max)

dist_dict = {'poisson': poisson_degree_sequence_partial, 'powerlaw': powerlaw_degree_sequence_partial}
noise_dict = {'additive': lx_additive}
control_params_dict = {'poisson': lmbd_vals, 'powerlaw': alpha_vals}

def worker_task(control_param, T, dist_func, noise_func):
    logging.info(f"Process {mp.current_process().pid} started task")
    result = process_data(control_param, T, dist_func, noise_func)
    return result

if __name__ == '__main__':
    
    data_dict_list = []

    for dist_name, dist_func in dist_dict.items():
        for noise_name, noise_func in noise_dict.items():
            control_param_vals = control_params_dict[dist_name]
            tasks = []

            for control_param in control_param_vals:
                my_dist = dist_func(control_param)
                critical_value = calculate_critical_transition(my_dist)
                print("Control Param: ", control_param)
                print(f"Critical Value: {critical_value}")
                T_vals_plus_crit = np.concatenate([T_vals, np.array([critical_value])])
                T_vals_plus_crit = np.sort(T_vals_plus_crit)
                T_vals_plus_crit = T_vals_plus_crit[(T_vals_plus_crit > 0) & (T_vals_plus_crit < 1)]
                logging.info(f"Control Param: {control_param}, T_vals_plus_crit: {T_vals_plus_crit}")

                # Prepare tasks for each T value
                for T in T_vals_plus_crit:
                    tasks.append((control_param, T, dist_func, noise_func))

            # Create the pool once
            with Pool() as pool:
                # Map the worker_task over all tasks
                results = pool.starmap(worker_task, tasks)
            data_dict_list.extend(results)

            df = pd.DataFrame(data_dict_list)
            df = df.explode(['lmbd', 'sce', 'outbreak_size'])
            df.outbreak_size = df.outbreak_size.apply(lambda x: x.real)
            df.to_csv(f"data/random_graphs/random_graphs_sweep_{dist_name}_{noise_name}.csv")