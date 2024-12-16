from functools import partial
import numpy as np

#from stochastic_pgfs.pgfs import PGF, make_G_u_minus_u, numerical_inversion, percolated_pgf,make_G_1

from stochastic_pgfs.laub_xia_algo import _solve_self_consistent_equation,G0,iterate_until_convergence,G1_prime,get_outbreak_size

from stochastic_pgfs.random_graphs import (
    poisson_degree_sequence,
    powerlaw_degree_sequence,
    make_config_model,
    plot_graph,
    mean_power_law,
    second_moment_power_law
)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Specify the degree sequence function here
N_max = 1000
#degree_sequence_func = poisson_degree_sequence 
degree_sequence_func = powerlaw_degree_sequence
degree_sequence_partial = partial(degree_sequence_func,N_max = N_max)


def process_data(lmbd, T, degree_sequence_func):
    print("T: ", T)
    print("lmbd: ", lmbd)
    my_degree_sequence = degree_sequence_func(lmbd)
    pk = np.vstack((np.arange(0, my_degree_sequence.shape[0], 1), my_degree_sequence)).T
    u1,u2 =iterate_until_convergence(pk,T = T )
    outbreak_size = get_outbreak_size(my_degree_sequence, T) 
    return {'lmbd': lmbd, 'T': T, 'outbreak_size': outbreak_size}

T_vals = np.linspace(0.001, 1, 60)
#lambda_vals = np.array([2.,3.01,4.])#np.linspace(3.1, 3, 4)
lambda_vals = np.linspace(1.8, 4, 8)
#lambda_vals = np.linspace(0.1, 2, 8)

processed_data = []

for T in T_vals:
    for lmbd in lambda_vals:
        processed_data.append(process_data(lmbd, T, degree_sequence_partial))
        print("Finished lambda: ", lmbd, " T: ", T)
    print("Finished T: ", T)

df = pd.DataFrame(processed_data)

def calculate_critical_transition(my_degree_sequence):
    pk = np.vstack((np.arange(0, my_degree_sequence.shape[0], 1), my_degree_sequence)).T
    return 1/G1_prime(1, pk, T)


def critical_trans_barabasi(alpha,k_min,k_max):
    if alpha <= 2:
        return 0.0
    elif 2 < alpha <= 3: 
        denom = (alpha -2)/(3-alpha)* k_min**(alpha - 2)*k_max**(3 - alpha)-1
    if alpha > 3:
        denom = ((alpha -2)/(alpha - 3))*k_min-1
    return 1/denom



# #%% 
# N = 1e6
lambda_vals = df['lmbd'].unique()
critical_transition = np.array([calculate_critical_transition(degree_sequence_func(lam,N_max)) for lam in lambda_vals])
#replace valeus greateer than 1 wieth None
critical_transition = np.where(critical_transition > 1, None, critical_transition)
critical_trans_barabasi_list = [critical_trans_barabasi(alpha, 1, N_max) for alpha in lambda_vals]


df_crit_trans = pd.DataFrame({'lambda': lambda_vals, 'p_c': critical_transition})

# print(critical_transition)
# print(critical_trans_barabasi_list)

# #%%
# critical_trans_barabasi(4, 1, 1000)   

#%%

sns.set_palette("tab10")

fig, ax = plt.subplots()
for i, row in df_crit_trans.iterrows():
    if row['p_c'] is not None:
        ax.axvline(row['p_c'], color=f'C{i}', linestyle='--', label=f'lambda={row["lambda"]}')
sns.scatterplot(data=df, x='T', y='outbreak_size', hue='lmbd', palette='tab10', ax=ax)
plt.title('Critical Transition for Lambda Values')
plt.xlabel('T')
plt.ylabel('Critical Transition')
plt.xlim(0.0, 1)
ax.legend().set_visible(True)
plt.show()

empirical_transition = df.query('outbreak_size > 1e-8').groupby('lmbd').agg(critical_trans=('T', 'min'))
#empirical_transition['PGF_value'] = critical_transition

# for i, lmbd in enumerate(lambda_vals):
#     print("Lambda: ", lmbd)
#     print("Critical Transition Empirical: ", empirical_transition.values[i])
    
   
    
calculate_critical_transition(degree_sequence_func(4., 100))
# %%
#plt.plot(lambda_vals, critical_trans_barabasi_list, label='Barabasi Critical Transition Values')
sns.scatterplot(data = df_crit_trans,x = 'lambda',y = 'p_c',label = 'PGF Critical Transition',alpha = 0.5)

plt.scatter(empirical_transition.index, empirical_transition.critical_trans, label='Empirical Critical Transition',alpha=0.5)
#plt.plot(lambda_vals, critical_PGF_list, label='PGF Critical Transition')   
plt.legend()

# %%
