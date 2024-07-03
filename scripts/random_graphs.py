from stochastic_pgfs.laub_xia_algo import l_x_algo, is_real, in_bounds
from stochastic_pgfs.pgfs import PGF, make_G_u_minus_u,numerical_inversion,percolated_pgf
from stochastic_pgfs.viz_utils import condition_number_heatmap
import numpy as np
from scipy.stats import nbinom
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from functools import partial
import scipy.stats as stats
date = datetime.today().strftime('%m-%d-%Y')

#define degree sequence for ER graph 
def er_degree_sequence(n,p,support = np.arange(0,100)):
    #poisosn distribution
    degree_sequence = stats.poisson.pmf(support,n*p)
    return degree_sequence


#generate ER graph
n = 10#the size of the graph
p = 0.214#density
my_er_degree_sequence = er_degree_sequence(n = n, p = p)
er_pgf = PGF(my_er_degree_sequence)

T =  0.5#infection rate
percolated_er_pgf =  partial(percolated_pgf,er_pgf,T = T)


percolated_coef = numerical_inversion(percolated_er_pgf)

lx_addative = partial(l_x_algo, K=1000, conditions=[is_real, in_bounds],is_pgf=True,perturbation_type='additive')

lx_addative(percolated_coef)


#initialize the laub xia algorithm for addative and multiplicated noise

lx_addative = partial(l_x_algo, K=1000, conditions=[is_real, in_bounds],is_pgf=True,perturbation_type='additive')
lx_multiplicative = partial(l_x_algo, K=1000, conditions=[is_real, in_bounds],is_pgf=True,perturbation_type='multiplicative')



lx_addative = partial(l_x_algo, K=1000, conditions=[is_real, in_bounds],is_pgf=True,perturbation_type='additive')
lx_multiplicative = partial(l_x_algo, K=1000, conditions=[is_real, in_bounds],is_pgf=True,perturbation_type='multiplicative')

p_vals = np.linspace(0.1,0.9,15)
R0_vals = np.linspace(1,5,15)

N_max = 100  # Maximum value for N in the distribution

#create partial function for the condition number heatmap
my_K = 1000

def er_condition_number_heatmap(lx_func, p_vals, R0_vals, N_max):
    condition_nums = np.zeros((len(p_vals), len(R0_vals)))
    for i in range(len(p_vals)):
        for j in range(len(R0_vals)):
            print(i,j)
            alpha = 1
            R0 = R0_vals[j]
            T = alpha / (R0 + alpha) #the infection rate
            print("T",T)
            print("P",p_vals[i])
            my_er_degree_sequence = er_degree_sequence(n = 10, p = p_vals[i])
            er_pgf = PGF(my_er_degree_sequence)
            percolated_er_pgf = partial(percolated_pgf,er_pgf,T = T)
            percolated_coef = numerical_inversion(percolated_er_pgf)
            condition_nums[i,j] = lx_func(percolated_coef)
    return condition_nums



condition_nums_addative= er_condition_number_heatmap(lx_addative, p_vals, R0_vals, N_max)
condition_nums_multiplicative = er_condition_number_heatmap(lx_multiplicative, p_vals, R0_vals, N_max)



### Heat Map
X, Y = np.meshgrid(R0_vals, p_vals)
fig, ax = plt.subplots(ncols = 2,figsize = (10,5))
ax[0].imshow(condition_nums_addative,
            extent = (min(R0_vals),max(R0_vals),min(p_vals),max(p_vals)),
             aspect = 'auto'
            )
ax[0].set(xlabel='R0 Values', ylabel='Alpha Values', title='Additive Condition Numbers')

ax[1].imshow(condition_nums_multiplicative,
            extent = (min(R0_vals),max(R0_vals),min(p_vals),max(p_vals)),
             aspect = 'auto'
            )
ax[1].set(xlabel='R0 Values', ylabel='Alpha Values', title='Multiplicative Condition Numbers')
plt.tight_layout()
plt.show()
















