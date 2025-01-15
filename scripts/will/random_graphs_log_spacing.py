import numpy as np
import timeit
from stochastic_pgfs.pgfs import PGF,make_G_u_minus_u,numerical_inversion,percolated_pgf
from stochastic_pgfs.laub_xia_algo import l_x_algo,G1_prime,get_outbreak_size
from stochastic_pgfs.random_graphs import powerlaw_degree_sequence,poisson_degree_sequence
import numpy as np
from scipy.stats import nbinom
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from functools import partial
import scipy.stats as stats

import matplotlib as mpl
import copy
date = datetime.today().strftime('%m-%d-%Y')

from mpi4py import MPI
import logging

logging.basicConfig(level=logging.INFO)
        
def calculate_critical_transition(my_degree_sequence):
    pk = np.vstack((np.arange(0, my_degree_sequence.shape[0], 1), my_degree_sequence)).T
    return 1/G1_prime(1, pk, 1)
        
        
def process_data(lmbd, T,degree_sequence_func,lx_func):
    logging.info(f"T {T}")
    logging.info(f"lmbd: {lmbd}")
    my_degree_sequence = degree_sequence_func(lmbd)
    outbreak_size = get_outbreak_size(my_degree_sequence,T)
    
    return {'lmbd': lmbd, 'T': T, 'sce': lx_func(my_degree_sequence,T = T), 'outbreak_size': outbreak_size}


N_max = 100  # Maximum value for N in the distribution
my_K = int(1e6)#number of samples per SCE estimte
max_iter = int(1e8)
tol = 1e-10

# #params to sweep over
T_vals = np.linspace(0.001,1,60)
alpha_vals = np.linspace(3.1,4,10)
lmbd_vals = np.linspace(0.001,2,10)


#create partial function for the condition number heatmap for addative and multiplicative noise
lx_additive = partial(l_x_algo, 
                      max_iter=max_iter,
                      tol = tol,
                      K=my_K, 
                      acceleration_method = 'steffensen',
                      sampling_method = 'orthogonal'
                    )

#partial functions for degree distriubtions
poisson_degree_sequence_partial = partial(poisson_degree_sequence,N_max = N_max)
powerlaw_degree_sequence_partial = partial(powerlaw_degree_sequence,N_max = N_max)

#data structures for sweep
data_dict_list = []
dist_dict = {'poisson': poisson_degree_sequence_partial,'powerlaw': powerlaw_degree_sequence_partial}
noise_dict = {'addative': lx_additive}

control_params_dict = {'poisson': lmbd_vals,'powerlaw': alpha_vals}

def worker_task(control_param, T, dist_func, noise_func):
    logging.info(f"Process {mp.current_process().pid} started task")
    result = process_data(control_param, T, dist_func, noise_func)
    return result

if __name__ == '__main__':
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    data_dict_list = []
    
    # Only rank 0 initializes parameters
    if rank == 0:
        dist_dict = {'poisson': poisson_degree_sequence_partial,'powerlaw': powerlaw_degree_sequence_partial}
        noise_dict = {'addative': lx_additive}
        control_params_dict = {'poisson': lmbd_vals,'powerlaw': alpha_vals}
    else:
        dist_dict = None
        noise_dict = None
        control_params_dict = None

    # Broadcast dictionaries to all processes
    dist_dict = comm.bcast(dist_dict, root=0)
    noise_dict = comm.bcast(noise_dict, root=0)
    control_params_dict = comm.bcast(control_params_dict, root=0)

    for dist_name, dist_func in dist_dict.items():
        for noise_name, noise_func in noise_dict.items():
            control_param_vals = control_params_dict[dist_name]
            tasks = []

            if rank == 0:
                # Generate all tasks
                for control_param in control_param_vals:
                    my_dist = dist_func(control_param)
                    critical_value = calculate_critical_transition(my_dist)
                    T_vals_plus_crit = np.logspace(np.log10(critical_value), np.log10(critical_value + 1e-4), 30)
                    
                    for T in T_vals_plus_crit:
                        tasks.append((control_param, T))
                
                # Distribute tasks among processes
                chunks = np.array_split(tasks, size)
            else:
                chunks = None

            # Scatter tasks to all processes
            my_tasks = comm.scatter(chunks, root=0)
            
            # Process local tasks
            local_results = []
            for control_param, T in my_tasks:
                result = process_data(control_param, T, dist_func, noise_func)
                local_results.append(result)

            # Gather results from all processes
            all_results = comm.gather(local_results, root=0)

            # Only rank 0 saves results
            if rank == 0:
                # Flatten results
                data_dict_list = [item for sublist in all_results for item in sublist]
                
                # Save results
                df = pd.DataFrame(data_dict_list)
                df = df.explode(['lmbd', 'sce', 'outbreak_size'])
                df.outbreak_size = df.outbreak_size.apply(lambda x: x.real)
                df.to_csv(f"data/random_graphs/critical_scaling_log_sweep_{dist_name}_{noise_name}.csv")

    MPI.Finalize()




