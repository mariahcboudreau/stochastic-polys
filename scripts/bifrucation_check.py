
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





#generate ER graph
n = 10#the size of the graph
lmbd = 2.#density
alpha = 1.5

def calculate_giant_component_size(degree_sequence):
        roots = _solve_self_consistent_equation(degree_sequence,derivative_test=True)
        s = 1-np.min(roots)
        T =  0.5#infection rate

        return s


T_vals = np.linspace(0.1,0.99,4)

lmbd_vals = np.linspace(0.5,2,20)


s_list_list = []
for i,T in enumerate(T_vals):
    s_list = []
    for j,lmbd in enumerate(lmbd_vals):
        n = 15
        degree_sequence = np.zeros(n+1)
        degree_sequence = stats.poisson.pmf(np.arange(0,n+1),lmbd)
        pgf = PGF(degree_sequence)
        percolated_er_pgf =  partial(percolated_pgf,pgf,T = T)
        percolated_coef = numerical_inversion(percolated_er_pgf)

        roots = _solve_self_consistent_equation(percolated_coef)
        s = 1-np.min(roots)
        s_list.append(s)
    s_list_list.append(copy.deepcopy(s_list))


fig,ax = plt.subplots()
for i in range(len(T_vals)):
    ax.plot(lmbd_vals,s_list_list[i],label = f"T = {T_vals[i]}")
plt.legend()
plt.show()





