from functools import partial
import numpy as np

from stochastic_pgfs.pgfs import PGF, make_G_u_minus_u, numerical_inversion, percolated_pgf

from stochastic_pgfs.laub_xia_algo import _solve_self_consistent_equation

from stochastic_pgfs.random_graphs import (
    er_degree_sequence,
    powerlaw_degree_sequence,
    make_config_model,
    plot_graph,
    mean_power_law,
    second_moment_power_law,
    poisson_degree_sequence
)

import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Specify the degree sequence function here
#degree_sequence_func = powerlaw_degree_sequence
degree_sequence_func = poisson_degree_sequence
degree_sequence_partial = partial(degree_sequence_func, N_max=1000)

T_vals = np.linspace(0.001, 1, 60)
lambda_vals = np.linspace(2, 4, 5)


def calculate_critical_transition(degree_sequence):
    # degree_sequence = np.array(degree_sequence, dtype=float)
    # degree_sequence /= np.sum(degree_sequence)  # Ensure normalization
    G_0 = PGF(degree_sequence)
    G_0_prime = G_0.derivative()
    G_0_double_prime = G_0_prime.derivative()
    G_0_prime_at_1 = G_0_prime(1)
    G_0_double_prime_at_1 = G_0_double_prime(1)
    p_c = G_0_prime_at_1 / G_0_double_prime_at_1
    return p_c

df = pd.read_csv("data/random_graphs/critical_scaling_log_sweep_poisson_addative.csv")
df_pl = pd.read_csv("data/random_graphs/critical_scaling_log_sweep_powerlaw_addative.csv")


#df = df_pl
# N = 1e6
# lambda_vals = np.sort(df['lmbd'].unique())
# critical_transition = [1/lam for lam in lambda_vals]

# df_crit_trans = pd.DataFrame({'lmbd': lambda_vals, 'critical_transition': critical_transition})
# #merge crit trans with df
# df = df.merge(df_crit_trans, on='lmbd', how='left')

#%% 
df['T'] = df['T'] - 1 / df['lmbd']#resacale for critical transition
df = df.query('T > 0 & lmbd > 1.0')
df['T'] = df['T'] 
fig,ax = plt.subplots()
sns.lineplot(data=df, x='T', y='sce', hue='lmbd',ax = ax,palette = 'rocket')


my_lmbd = df.lmbd.unique()[-1]

#%% 
cut_df = df.query("lmbd == @my_lmbd")

fig,ax = plt.subplots()
sns.scatterplot(data=df, x='T', y='sce', hue='lmbd',ax = ax,palette = 'tab10')

#set xlim 
#ax.set_xlim(0,1)
ax.set(yscale= 'log',xscale = 'log', xlabel = 'T', ylabel = 'SCE')

#log transform data
df_log_data = df.copy()
df_log_data['sce'] = np.log10(df_log_data['sce'])
df_log_data['T'] = np.log10(df_log_data['T'])   

scaling_exponent_list =  []
for lmbd, group in df.groupby('lmbd'):
    X = group[['T']].values.reshape(-1, 1)
    y = group['sce'].values

    model = LinearRegression()
    model.fit(X, y)
    slope = model.coef_[0]
    intercept = model.intercept_
    #get the R^2 value
    r_squared = model.score(X,y)
    scaling_exponent_list.append({'lmbd': lmbd, 'slope': slope, 'intercept': intercept,'r_squared': r_squared}) 
    
scaling_exponent_df = pd.DataFrame(scaling_exponent_list)

fig,(ax1,ax2) = plt.subplots(nrows = 2, figsize = (10,10))
sns.scatterplot(data=scaling_exponent_df, x='lmbd', y='slope',color='red',ax = ax1)
sns.scatterplot(data=scaling_exponent_df, x='lmbd', y='r_squared',color='red',ax = ax2)
ax1.set(xlabel = 'Lambda', ylabel = 'SCE Critical Exponent)', title = 'SCE Critical Exponent vs. Lambda')
ax2.set(xlabel = 'Lambda', ylabel = 'SCE Critical Exponent R^2)', title = 'SCE Critical Exponent vs. Lambda')
plt.suptitle('SCE Critical Exponent vs. Lambda(T > T_c)')



# %%
