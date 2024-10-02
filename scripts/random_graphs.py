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


def condition_num_heatmap(lx_func,degree_sequence_func,degree_sequence_vals, R0_vals, N_max):
    condition_nums = np.zeros((len(degree_sequence_vals), len(R0_vals)))
    for i in range(len(degree_sequence_vals)):
        for j in range(len(R0_vals)):
            print(i,j)
            alpha = 1#the dispersion parameter
            #R0 = R0_vals[j]
            #T = alpha / (R0 + alpha) #the infection rate
            #print("T",T)
            #print("alpha",degree_sequence_vals[i])
            T = R0_vals[j]

            #define the degree sequence for a given value of degree_squence_vals
            my_degree_sequence = degree_sequence_func(degree_sequence_vals[i])
            #create the pgf for the degree sequence
            my_pgf = PGF(my_degree_sequence)
            #perform percolation on the degree sequece by composing the pgf with the (1-T)+Tx 
            my_percolated_pgf = partial(percolated_pgf,my_pgf,T = T)
            #invert the percolated pgf to get the coefficients
            percolated_coef = numerical_inversion(my_percolated_pgf)
            #compute the condition number 
            condition_nums[i,j] = lx_func(percolated_coef)
    return condition_nums



N,M = 8,8#resolution of the heatmap

R0_vals = np.linspace(0.1,1,N)#
#p_vals = np.linspace(0,2,M)#density for the ER graph
lmbd_vals = np.linspace(0.1,2,M)#density for the ER graph
alpha_vals  = np.linspace(1,4,M)#power law exponent
N_max = 10  # Maximum value for N in the distribution

#create partial function for the condition number heatmap
my_K = 500

#create partial function for the condition number heatmap for additive and multiplicative noise
lx_addative = partial(l_x_algo, K=my_K, conditions=[is_real, in_bounds],is_pgf=True,perturbation_type='additive')
lx_multiplicative = partial(l_x_algo, K=my_K, conditions=[is_real, in_bounds],is_pgf=True,perturbation_type='multiplicative')




poisson_degree_sequence_partial = partial(poisson_degree_sequence,n = N_max)
powerlaw_degree_sequence_partial = partial(powerlaw_degree_sequence,N_max = N_max)


er_condition_nums_addative= condition_num_heatmap(lx_addative,poisson_degree_sequence_partial,lmbd_vals, R0_vals, N_max)
powerlaw_condition_nums_addative= condition_num_heatmap(lx_addative,powerlaw_degree_sequence_partial,alpha_vals, R0_vals, N_max)


er_condition_nums_multiplicative= condition_num_heatmap(lx_multiplicative,poisson_degree_sequence_partial,lmbd_vals, R0_vals, N_max)
powerlaw_condition_nums_multiplicative= condition_num_heatmap(lx_multiplicative,powerlaw_degree_sequence_partial,alpha_vals, R0_vals, N_max)



#### Heat Map
X, Y = np.meshgrid(R0_vals, lmbd_vals)
fig, ax = plt.subplots(ncols = 2,nrows =2,figsize = (10,5))

im = ax[0,0].imshow((er_condition_nums_addative),
            extent = (min(R0_vals),max(R0_vals),min(lmbd_vals),max(lmbd_vals)),
             aspect = 'auto'
            )

ax[0,0].set(xlabel='T Values', ylabel='p Values', title='ER Additive Condition Numbers')
#add a colorbar 
fig.colorbar(im, ax=ax[0,0])

ax[1,0].imshow(powerlaw_condition_nums_addative,
           extent = (min(R0_vals),max(R0_vals),min(alpha_vals),max(alpha_vals)),
            aspect = 'auto'
           )
ax[1,0].set(xlabel='T Values', ylabel=r'$\alpha$ Values', title='Powerlaw Additive Condition Numbers')



im = ax[0,1].imshow((er_condition_nums_multiplicative),
            extent = (min(R0_vals),max(R0_vals),min(lmbd_vals),max(lmbd_vals)),
             aspect = 'auto'
            )

ax[0,1].set(xlabel='T Values', ylabel='p Values', title='ER Multiplicative Condition Numbers')
#add a colorbar 
fig.colorbar(im, ax=ax[0,1])

ax[1,1].imshow(powerlaw_condition_nums_multiplicative,
           extent = (min(R0_vals),max(R0_vals),min(alpha_vals),max(alpha_vals)),
            aspect = 'auto'
           )
ax[1,1].set(xlabel='T Values', ylabel=r'$\alpha$ Values', title='Powerlaw Multiplicative Condition Numbers')



plt.tight_layout()
plt.show()
















