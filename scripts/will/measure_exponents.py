from functools import partial
import numpy as np

from stochastic_pgfs.pgfs import PGF, make_G_u_minus_u, numerical_inversion, percolated_pgf


from stochastic_pgfs.random_graphs import (
    er_degree_sequence,
    powerlaw_degree_sequence,
    make_config_model,
    plot_graph,
    mean_power_law,
    poisson_degree_sequence
)

import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy import stats

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


df.query('sce > 0', inplace=True)


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
    
    # Fit the model
    model = LinearRegression()
    model.fit(X, y)
    
    # Get parameters
    n = len(X)
    p = X.shape[1]  # number of predictors (1 in this case)
    
    # Calculate standard errors
    mse = np.sum((y - model.predict(X)) ** 2) / (n - p - 1)
    var_coef = mse * np.linalg.inv(X.T @ X)[0][0]
    se = np.sqrt(var_coef)
    
    # Calculate confidence intervals (95%)
    t_stat = stats.t.ppf(0.975, n - p - 1)
    ci_lower = model.coef_[0] - t_stat * se
    ci_upper = model.coef_[0] + t_stat * se
    
    scaling_exponent_list.append({
        'lmbd': lmbd,
        'slope': model.coef_[0],
        'intercept': model.intercept_,
        'r_squared': model.score(X, y),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'std_err': se
    })

scaling_exponent_df = pd.DataFrame(scaling_exponent_list)

fig,(ax1,ax2) = plt.subplots(nrows = 2, figsize = (10,10))
sns.scatterplot(data=scaling_exponent_df, x='lmbd', y='slope',color='C0',ax = ax1,lw = 50)

# Add confidence intervals as error bars
ax1.errorbar(scaling_exponent_df['lmbd'], 
            scaling_exponent_df['slope'],
            yerr=[scaling_exponent_df['slope'] - scaling_exponent_df['ci_lower'],
                  scaling_exponent_df['ci_upper'] - scaling_exponent_df['slope']],
            fmt='none', color='gray', alpha=0.3)

sns.lineplot(data=scaling_exponent_df, x='lmbd', y='r_squared',color='red',ax = ax2,lw = 2)
ax1.set(xlabel = 'Lambda', ylabel = 'SCE Critical Exponent)', title = 'SCE Critical Exponent vs. Lambda')
ax2.set(xlabel = 'Lambda', ylabel = 'SCE Critical Exponent R^2)', title = 'SCE Critical Exponent vs. Lambda')
plt.suptitle('SCE Critical Exponent vs. Lambda(T > T_c)')

# Log-transform the data
df['log_T'] = np.log10(df['T'])
df['log_sce'] = np.log10(df['sce'])


# Plot log-log data with regression lines
fig, ax = plt.subplots()
sns.scatterplot(data=df, x='log_T', y='log_sce', hue='lmbd', ax=ax, palette='tab10')

# For each lmbd, plot the line of best fit
for lmbd, group in df.groupby('lmbd'):
    print(lmbd)
    X = group['log_T'].values.reshape(-1, 1)
    y = group['log_sce'].values
    model = LinearRegression()
    model.fit(X, y)
    x_range = np.linspace(X.min(), X.max(), 100)
    y_pred = model.predict(x_range.reshape(-1, 1))
    ax.plot(x_range, y_pred, linestyle='--', label=f'Fit λ={lmbd}')
    

ax.set(xlabel='log(T-T_c)', ylabel='log(SCE)', title='Log-Log SCE vs T with Regression Lines')

# Move the legend outside the chart
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Process power-law data
df_pl = pd.read_csv("data/random_graphs/critical_scaling_log_sweep_powerlaw_addative.csv")
df_pl.query('sce > 0', inplace=True)
df_pl['T'] = df_pl['T'] - 1 / df_pl['lmbd']  # Rescale for critical transition
df_pl = df_pl.query('T > 0 & lmbd > 1.0')
df_pl['log_T'] = np.log10(df_pl['T'])
df_pl['log_sce'] = np.log10(df_pl['sce'])

# Calculate scaling exponents for power-law data
scaling_exponent_list_pl = []
for lmbd, group in df_pl.groupby('lmbd'):
    X = group['log_T'].values.reshape(-1, 1)
    y = group['log_sce'].values
    model = LinearRegression()
    model.fit(X, y)
    slope = model.coef_[0]
    intercept = model.intercept_
    r_squared = model.score(X, y)
    scaling_exponent_list_pl.append({'lmbd': lmbd, 'slope': slope, 'intercept': intercept, 'r_squared': r_squared})

scaling_exponent_df_pl = pd.DataFrame(scaling_exponent_list_pl)

# Plot scaling exponents for power-law data
fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10, 10))
sns.scatterplot(data=scaling_exponent_df_pl, x='lmbd', y='slope', color='blue', ax=ax1)
sns.scatterplot(data=scaling_exponent_df_pl, x='lmbd', y='r_squared', color='blue', ax=ax2)
ax1.set(xlabel='Alpha', ylabel='SCE Critical Exponent', title='Power-Law SCE Critical Exponent vs. Alpha')
ax2.set(xlabel='Alpha', ylabel='SCE Critical Exponent R²', title='Power-Law SCE R² vs. Alpha')
plt.suptitle('Power-Law SCE Critical Exponent Analysis')

# %%
