import numpy as np
import timeit
from stochastic_pgfs.pgfs import PGF,make_G_u_minus_u,numerical_inversion,percolated_pgf
from stochastic_pgfs.viz_utils import condition_number_heatmap
from stochastic_pgfs.laub_xia_algo import l_x_algo, is_real, in_bounds,_solve_self_consistent_equation
from stochastic_pgfs.random_graphs import er_degree_sequence,powerlaw_degree_sequence,poisson_degree_sequence
import numpy as np
from scipy.stats import nbinom
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from functools import partial
import scipy.stats as stats


import numpy as np
import timeit
from stochastic_pgfs.pgfs import PGF,make_G_u_minus_u,numerical_inversion,percolated_pgf
from stochastic_pgfs.viz_utils import condition_number_heatmap
from stochastic_pgfs.laub_xia_algo import l_x_algo, is_real, in_bounds,_solve_self_consistent_equation
from stochastic_pgfs.random_graphs import er_degree_sequence,powerlaw_degree_sequence,poisson_degree_sequence,make_config_model
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






degree_sequence_func = powerlaw_degree_sequence
#degree_sequence_func = poisson_degree_sequence

N_max = 20 # Maximum value for N in the distribution

lmbd_vals = np.linspace(1,4,10)
#T_vals = np.linspace(0,1,30)
T_vals = np.array([1])

# df = pd.read_csv("data/random_graphs/random_graphs_sweep_poisson_multiplicative.csv")
# df_pl = pd.read_csv("data/random_graphs/random_graphs_sweep_powerlaw_multiplicative.csv")

df = pd.read_csv("data/random_graphs/random_graphs_sweep_poisson_addative.csv")
df_pl = pd.read_csv("data/random_graphs/random_graphs_sweep_powerlaw_addative.csv")


df_no_perc = df_pl.query("T == 1")
sns.lineplot(data = df_no_perc,x = 'lmbd',y = 'outbreak_size')
plt.show()



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
lmbd_vals = np.linspace(0,2,60)
alpha_vals = np.linspace(2,4,60)
#T_vals = np.linspace(0.001,1,60)
T_vals = np.array([0.1])


degree_sequence_func =  powerlaw_degree_sequence 
lmbd = 2.1
T = 0.99

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
a = lx_func(degree_sequence)







# def melloy_reed_criterion(alpha,k_min,k_max):
#     k_max = k_min*N**(1/(alpha+1))
#     if alpha > 3:
#         A = k_min
#     elif 3 > alpha > 2:
#         A = k_max**(3-alpha)*k_min**(alpha-2)
#     elif 2 > alpha > 1:
#         A = k_max
#     else: 
#         raise(ValueError)
        
#     kappa = np.abs((2-alpha)/(3-alpha))*A
   
#     f_c = 1 - 1/(kappa -1) 
    
#     return f_c


def calculate_fc(gamma, k_min, k_max=None):
    if 2 < gamma < 3:
        if k_max is None:
            raise ValueError("k_max is required for 2 < gamma < 3")
        numerator = gamma - 2
        denominator = 3 - gamma
        return 1 - (1 / ((numerator / denominator) * (k_min ** (gamma - 2)) * (k_max ** (3 - gamma)) - 1))
    elif gamma > 3:
        numerator = gamma - 2
        denominator = gamma - 3
        return 1 - (1 / ((numerator / denominator) * (k_min ** (gamma - 2)) - 1))
    else:
        raise ValueError("Gamma must be greater than 2")




N = 1000
k_min = 1


# alpha_vals = np.linspace(2.1,4,100)    
# f_c_vals = [calculate_fc(alpha_i,1,100000) for alpha_i in alpha_vals]
# plt.plot(alpha_vals,f_c_vals)

for lmbd in lmbd_vals:
    outbreak_size_list = []
    for T in T_vals :
        my_degree_sequence = degree_sequence_func(lmbd,N_max)
        #plt.plot(my_degree_sequence)
        #create the pgf for the degree sequence
        my_pgf = PGF(my_degree_sequence)
        #perform percolation on the degree sequece by composing the pgf with the (1-T)+Tx 
        my_percolated_pgf = partial(percolated_pgf,my_pgf,T = T)
        #invert the percolated pgf to get the coefficients
        percolated_coef = numerical_inversion(my_percolated_pgf)
        perc_degree_sequence = percolated_coef
        G_u_minus_u = make_G_u_minus_u(perc_degree_sequence)    
        outbreak_size = 1-np.min(_solve_self_consistent_equation(perc_degree_sequence))
        outbreak_size_list.append(outbreak_size)
        
plt.plot(1-T_vals,outbreak_size_list,label=f"{lmbd:.2f}")
    #f_c = calculate_fc(lmbd,1,N_max)
    #plt.axvline(f_c)
    

plt.ylabel("Outbreak Size")
plt.xlabel("T")
plt.legend()
plt.tight_layout()