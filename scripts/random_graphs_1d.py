import numpy as np
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
date = datetime.today().strftime('%m-%d-%Y')


#define degree sequence for ER graph 
def er_degree_sequence(p,n,support = np.arange(0,100)):
    #poisosn distribution
    degree_sequence = stats.poisson.pmf(support,n*p)
    return degree_sequence

def powerlaw_degree_sequence(alpha,N_max):
    #power law
    degree_sequence = np.zeros(N_max+1)
    for i in range(1,N_max+1):
        degree_sequence[i] = i**(-alpha)
    return degree_sequence/sum(degree_sequence)

def poisson_degree_sequence(lambd,n = 100):
    #poisson distribution
    degree_sequence = stats.poisson.pmf(np.arange(0,n),lambd)
    return degree_sequence





lmbd_vals = np.linspace(0,1,20)
t_vals = np.linspace(0.1,1,10)

N_max = 10  # Maximum value for N in the distribution

#create partial function for the condition number heatmap
#my_K = 1000
my_K = 1

#create partial function for the condition number heatmap for additive and multiplicative noise
lx_addative = partial(l_x_algo, K=my_K, conditions=[is_real, in_bounds],is_pgf=True,perturbation_type='additive')
lx_multiplicative = partial(l_x_algo, K=my_K, conditions=[is_real, in_bounds],is_pgf=True,perturbation_type='multiplicative')


poisson_degree_sequence_partial = partial(poisson_degree_sequence,n = N_max)
powerlaw_degree_sequence_partial = partial(powerlaw_degree_sequence,N_max = N_max)

lx_func = lx_addative
degree_sequence_func = poisson_degree_sequence
sce_list = []
outbreak_size_list = []
t_list = []
row_list = []

for T in t_vals:
    for lmbd in lmbd_vals: 
        degree_sequence = degree_sequence_func(lmbd,N_max)
        my_pgf = PGF(degree_sequence)
        #perform percolation on the degree sequece by composing the pgf with the (1-T)+Tx 
        my_percolated_pgf = partial(percolated_pgf,my_pgf,T = T)
        #invert the percolated pgf to get the coefficients
        percolated_coef = numerical_inversion(my_percolated_pgf)
        lx_func(percolated_coef)
        outbreak_size = 1- np.min(_solve_self_consistent_equation(percolated_coef))
        condition_number = _solve_self_consistent_equation(percolated_coef)
        row = {"T":T,"lmbd":lmbd,"outbreak_size":outbreak_size,'condition_number': condition_number}
        row_list.append(row)

df = pd.DataFrame(row_list)

fig,ax = plt.subplots(nrows = 2,sharex = True)
sns.scatterplot(data = df,x = 'lmbd',y = 'outbreak_size',hue = 'T',ax = ax[0])
sns.scatterplot(data = df,x = 'lmbd',y = 'condition_number',hue = 'T',ax = ax[0])
plt.show()



