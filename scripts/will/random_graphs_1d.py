import numpy as np
import timeit
from stochastic_pgfs.pgfs import PGF,make_G_u_minus_u,numerical_inversion,percolated_pgf
from stochastic_pgfs.viz_utils import condition_number_heatmap
from stochastic_pgfs.laub_xia_algo import l_x_algo, is_real, in_bounds,_solve_self_consistent_equation
from stochastic_pgfs.random_graphs import er_degree_sequence,powerlaw_degree_sequence,poisson_degree_sequence,make_config_model,power_law_variance,mean_power_law
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






N_max = 100  # Maximum value for N in the distribution

#create partial function for the condition number heatmap
#my_K = 1000
my_K = 10000

#create partial function for the condition number heatmap for addative and multiplicative noise
lx_addative = partial(l_x_algo, K=my_K, conditions=[is_real, in_bounds],is_pgf=True,perturbation_type='additive')
lx_multiplicative = partial(l_x_algo, K=my_K, conditions=[is_real, in_bounds],is_pgf=True,perturbation_type='multiplicative')


mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams["font.family"] = "Times New Roman"


poisson_degree_sequence_partial = partial(poisson_degree_sequence,N_max = N_max)
powerlaw_degree_sequence_partial = partial(powerlaw_degree_sequence,N_max = N_max)

lx_func = lx_addative
#degree_sequence_func = poisson_degree_sequence
data_dict_list = []

#create the pgf for the degree sequence
#my_pgf = PGF(my_degree_sequence)
#perform percolation on the degree sequece by composing the pgf with the (1-T)+Tx 
#my_percolated_pgf = partial(percolated_pgf,my_pgf,T = T)
#invert the percolated pgf to get the coefficients
#percolated_coef = numerical_inversion(my_percolated_pgf)


def generate_T_values(lmbd,dist_name,k_min = 1,k_max = N_max):
    if dist_name == 'poisson':
        return 1/lmbd
    if dist_name == 'powerlaw':
        return mean_power_law(k_min,k_max,lmbd)/power_law_variance(k_min,k_max,lmbd)
    
   

    

lmbd_vals = np.linspace(0,2,60)
alpha_vals = np.linspace(2.1,4,60)
T_vals = np.linspace(0.001,1,60)
T_vals = np.array([1])
lmbd_vals = np.array([1])
alpha_vals = np.array([2])


#poisson_critical_values = generate_T_values(lmbd_vals,dist_name='poisson') 
poisson_critical_values = [generate_T_values(lmbd_i,dist_name='poisson')  for lmbd_i in lmbd_vals]
powerlaw_critical_values = [generate_T_values(alpha_i,dist_name='powerlaw')  for alpha_i in alpha_vals]
# Concatenate the arrays
all_critical_values = np.concatenate((poisson_critical_values, powerlaw_critical_values))
# Filter points between 0 and 1
filtered_critical_values = all_critical_values[(all_critical_values > 0) & (all_critical_values < 1)]


T_vals= np.concatenate([filtered_critical_values,T_vals])
T_vals = np.sort(T_vals)



def process_data(lmbd, T,degree_sequence_func,lx_func):
    print("T: ",T)
    print("lmbd: ",lmbd)
    sce_list = []
    outbreak_size_list = []
    my_degree_sequence = degree_sequence_func(lmbd)
    #create the pgf for the degree sequence
    my_pgf = PGF(my_degree_sequence)
    #perform percolation on the degree sequece by composing the pgf with the (1-T)+Tx 
    my_percolated_pgf = partial(percolated_pgf,my_pgf,T = T)
    #invert the percolated pgf to get the coefficients
    percolated_coef = numerical_inversion(my_percolated_pgf)
    degree_sequence = percolated_coef
    outbreak_size = 1-np.min(_solve_self_consistent_equation(degree_sequence))

    return {'lmbd': lmbd, 'T': T, 'sce': lx_func(degree_sequence), 'outbreak_size': outbreak_size}

#dist_dict = {'poisson': poisson_degree_sequence_partial,'powerlaw': powerlaw_degree_sequence_partial}
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
                print(noise_func)
                results = [pool.apply_async(process_data, (control_param, T, dist_func,noise_func)) for control_param in control_param_vals for T in T_vals]
                data_dict_list = [result.get() for result in results]

            df = pd.DataFrame(data_dict_list)
            df = df.explode(['lmbd','sce','outbreak_size'])
            df.outbreak_size = df.outbreak_size.apply(lambda x: x.real)
    
            #df.to_csv(f"data/random_graphs/random_graphs_sweep_{dist_name}_{noise_name}.csv")
    
    
    # with Pool() as pool:
    #     results = [pool.apply_async(process_data, (alpha, T, powerlaw_degree_sequence_partial)) for alpha in alpha_vals for T in T_vals]
    #     data_dict_list = [result.get() for result in results]

    #     df = pd.DataFrame(data_dict_list)
    #     df = df.explode(['lmbd','sce','outbreak_size'])
    #     df.outbreak_size = df.outbreak_size.apply(lambda x: x.real)
    
        #df.to_csv("data/random_graphs/random_graphs_sweep_powerlaw.csv")




