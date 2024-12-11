import numpy as np
import timeit
from stochastic_pgfs.pgfs import PGF,make_G_u_minus_u,numerical_inversion,percolated_pgf
from stochastic_pgfs.viz_utils import condition_number_heatmap
from stochastic_pgfs.laub_xia_algo import l_x_algo, is_real, in_bounds,_solve_self_consistent_equation,G1_prime,get_outbreak_size,iterate_until_convergence
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

from multiprocessing import Pool
import logging

logging.basicConfig(level=logging.INFO)
        
def calculate_critical_transition(my_degree_sequence):
    pk = np.vstack((np.arange(0, my_degree_sequence.shape[0], 1), my_degree_sequence)).T
    return 1/G1_prime(1, pk, 1)
        
        
def process_data(lmbd, T,degree_sequence_func,lx_func):
    print("T: ",T)
    print("lmbd: ",lmbd)
    sce_list = []
    outbreak_size_list = []
    my_degree_sequence = degree_sequence_func(lmbd)
    outbreak_size = get_outbreak_size(my_degree_sequence,T)
    
    return {'lmbd': lmbd, 'T': T, 'sce': lx_func(my_degree_sequence,T = T), 'outbreak_size': outbreak_size}


N_max = 100  # Maximum value for N in the distribution
my_K = int(1e5)#number of samples per SCE estimte
max_iter = int(1e5)
tol = 1e-10

#params to sweep over
# T_vals = np.linspace(0.001,1,60)
# alpha_vals = np.linspace(3.1,4,10)
# lmbd_vals = np.linspace(0.001,2,10)

T_vals = np.linspace(0.001, 1, 30)
alpha_vals = [3.0]
lmbd_vals = [2.0]


#create partial function for the condition number heatmap for addative and multiplicative noise
lx_addative = partial(l_x_algo, K=my_K, conditions=[is_real, in_bounds],is_pgf=True,perturbation_type='additive',max_iter = max_iter,tol = tol)
lx_multiplicative = partial(l_x_algo, K=my_K, conditions=[is_real, in_bounds],is_pgf=True,perturbation_type='multiplicative',max_iter = max_iter)

#partial functions for degree distriubtions
poisson_degree_sequence_partial = partial(poisson_degree_sequence,N_max = N_max)
powerlaw_degree_sequence_partial = partial(powerlaw_degree_sequence,N_max = N_max)

#data structures for sweep
data_dict_list = []
dist_dict = {'poisson': poisson_degree_sequence_partial,'powerlaw': powerlaw_degree_sequence_partial}
noise_dict = {'addative': lx_addative}

control_params_dict = {'poisson': lmbd_vals,'powerlaw': alpha_vals}


if __name__ == '__main__':
    for dist_name,dist_func in dist_dict.items(): 
        print(dist_name)
        for noise_name,noise_func in noise_dict.items():
            print(noise_name)
            control_param_vals = control_params_dict[dist_name]
            with Pool() as pool:
                results = []
                for control_param in control_param_vals:
                    #add critical value to lise of T values
                    my_dist = dist_func(control_param)
                    critical_value = calculate_critical_transition(my_dist)
                    T_vals_plus_crit = np.logspace(critical_value,critical_value+0.1,30)
                    logging.info(f"Control Param: {control_param}")
                    for T in T_vals_plus_crit:
                        results.append(pool.apply_async(process_data, (control_param, T,dist_func,noise_func)))
                        
                data_dict_list = [result.get() for result in results]
                #save results
                df = pd.DataFrame(data_dict_list)
                df = df.explode(['lmbd','sce','outbreak_size'])
                df.outbreak_size = df.outbreak_size.apply(lambda x: x.real)
                df.to_csv(f"data/random_graphs/critical_scaling_log_sweep_{dist_name}_{noise_name}.csv")




