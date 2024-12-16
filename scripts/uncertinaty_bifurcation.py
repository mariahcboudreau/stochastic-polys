from stochastic_pgfs.pgfs import PGF,make_G_u_minus_u,numerical_inversion,percolated_pgf
from stochastic_pgfs.viz_utils import condition_number_heatmap
from stochastic_pgfs.laub_xia_algo import l_x_algo, is_real, in_bounds,_solve_self_consistent_equation
import numpy as np
from scipy.stats import nbinom
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from functools import partial
import scipy.stats as stats
import copy
import seaborn as sns
date = datetime.today().strftime('%m-%d-%Y')





N_max = 10  # Maximum value for N in the distribution

#create partial function for the condition number heatmap

#given a pmf compute the mean 
def calc_mean_degree(degree_sequence):
    nums = np.arange(0,len(degree_sequence))
    return np.sum(nums*degree_sequence)


def bifurcation_diagram_with_uncertainty(my_degree_sequence,lx_func,T):
    #define the degree sequence for a given value of degree_squence_vals
    print("mea degree unper",np.mean(my_degree_sequence))
    #create the pgf for the degree sequence
    my_pgf = PGF(my_degree_sequence)
    #perform percolation on the degree sequece by composing the pgf with the (1-T)+Tx 
    my_percolated_pgf = partial(percolated_pgf,my_pgf,T = T)
    #invert the percolated pgf to get the coefficients
    percolated_coef = numerical_inversion(my_percolated_pgf)

    mean_degree= calc_mean_degree(my_degree_sequence)
    print("mean_degree",mean_degree)

    roots = _solve_self_consistent_equation(percolated_coef)
    s = 1-np.min(roots)
    s = np.real(s)
    #s = 1-np.max(roots)

    #compute the condition number 
    perturbed_roots = lx_func(percolated_coef)
    return {'s':s,'perturbed_roots':perturbed_roots,'mean_degree':mean_degree,'T':T}


my_N = 100
my_K = 10



    
    

degree_sequence_vals = np.linspace(0.1,1.8,20)
T_vals = [0.7,0.8,0.9]
lx_addative = partial(l_x_algo, K=my_K, conditions=[is_real,in_bounds],is_pgf=True,perturbation_type='additive',bifurcation = True,delta = 1e5)
perturbation_dict_list = []

for lmbd in degree_sequence_vals:
    print(lmbd)
    for T in T_vals:
        alpha = 1#the dispersion parameter
        degree_sequence = stats.poisson.pmf(np.arange(0,my_N),lmbd)
        print("my mean",np.mean(degree_sequence))
        perturbation_dict_list.append(bifurcation_diagram_with_uncertainty(degree_sequence,lx_addative,T))

df= pd.DataFrame(perturbation_dict_list)
df = df.explode('perturbed_roots')
df['roots'] = df['s']+ df['perturbed_roots']


fig,ax = plt.subplots()
sns.lineplot(data = df, x = 'mean_degree', y = 'roots',hue = 'T',ax = ax)
plt.show()







