from stochastic_pgfs.pgfs import PGF,make_G_u_minus_u,numerical_inversion,percolated_pgf
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





#generate ER graph
n = 10#the size of the graph
p = 0.214#density
my_er_degree_sequence = er_degree_sequence(n = n, p = p)
er_pgf = PGF(my_er_degree_sequence)

T =  1.0#infection rate
percolated_er_pgf =  partial(percolated_pgf,er_pgf,T = T)


percolated_coef = numerical_inversion(percolated_er_pgf)

lx_addative = partial(l_x_algo, K=1000, conditions=[is_real, in_bounds],is_pgf=True,perturbation_type='additive')
lx_addative(percolated_coef)




def condition_num_heatmap(lx_func,degree_sequence_func,degree_sequence_vals, R0_vals, N_max):
    condition_nums = np.zeros((len(degree_sequence_vals), len(R0_vals)))
    for i in range(len(degree_sequence_vals)):
        for j in range(len(R0_vals)):
            print(i,j)
            alpha = 1#the dispersion parameter
            R0 = R0_vals[j]
            T = alpha / (R0 + alpha) #the infection rate
            print("T",T)
            print("alpha",degree_sequence_vals[i])
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


#create partial function for the condition number heatmap for additive and multiplicative noise
lx_addative = partial(l_x_algo, K=1000, conditions=[is_real, in_bounds],is_pgf=True,perturbation_type='additive')
lx_multiplicative = partial(l_x_algo, K=1000, conditions=[is_real, in_bounds],is_pgf=True,perturbation_type='multiplicative')

N,M = 5,5#resolution of the heatmap

R0_vals = np.linspace(1,5,N)#
p_vals = np.linspace(0.1,0.9,M)#density for the ER graph
alpha_vals  = np.linspace(1,4,M)#power law exponent
N_max = 100  # Maximum value for N in the distribution

#create partial function for the condition number heatmap
my_K = 1000

er_degree_sequence_partial = partial(er_degree_sequence,n = N_max)
powerlaw_degree_sequence_partial = partial(powerlaw_degree_sequence,N_max = N_max)

er_condition_nums_addative= condition_num_heatmap(lx_addative,er_degree_sequence_partial,p_vals, R0_vals, N_max)
powerlaw_condition_nums_addative= condition_num_heatmap(lx_addative,powerlaw_degree_sequence_partial,p_vals, R0_vals, N_max)

#### Heat Map
X, Y = np.meshgrid(R0_vals, p_vals)
fig, ax = plt.subplots(ncols = 2,nrows =2,figsize = (10,5))
ax[0,0].imshow(er_condition_nums_addative,
            extent = (min(R0_vals),max(R0_vals),min(p_vals),max(p_vals)),
             aspect = 'auto'
            )
ax[0,0].set(xlabel='R0 Values', ylabel='p Values', title='ER Additive Condition Numbers')


ax[1,0].imshow(powerlaw_condition_nums_addative,
           extent = (min(R0_vals),max(R0_vals),min(alpha_vals),max(alpha_vals)),
            aspect = 'auto'
           )
ax[1,0].set(xlabel='R0 Values', ylabel=r'$\alpha$ Values', title='Powerlaw Additive Condition Numbers')

#
#ax[1].imshow(condition_nums_multiplicative,
#            extent = (min(R0_vals),max(R0_vals),min(p_vals),max(p_vals)),
#             aspect = 'auto'
#            )
#ax[1].set(xlabel='R0 Values', ylabel='Alpha Values', title='Multiplicative Condition Numbers')
plt.tight_layout()
plt.show()
















