from stochastic_pgfs.laub_xia_algo import l_x_algo, is_real, in_bounds
from stochastic_pgfs.pgfs import PGF, make_G_u_minus_u
from stochastic_pgfs.sim import *
import numpy as np
from scipy.stats import nbinom
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
date = datetime.today().strftime('%m-%d-%Y')


# Define the matrix size or retrieve it from existing data
alpha_vals = np.arange(0.1,0.3,0.01)  # Example values
R0_vals = np.arange(1,5, 0.1)  # Example values
N_max = 10  # Maximum value for N in the distribution
condition_nums = np.zeros((len(alpha_vals), len(R0_vals)))
variances = np.zeros((len(alpha_vals), len(R0_vals)))

for i in range(len(alpha_vals)):
    for j in range(len(R0_vals)):
        alpha = alpha_vals[i]
        R0 = R0_vals[j]
        r = 1 / alpha #dispersion parameter
        p = alpha / (R0 + alpha) #Changed this parameter
        dist = nbinom(n=r, p=p)  # Scipy parameterizes differently, check this
        my_pdf = dist.pmf(range(N_max + 1))
        # G = PGF(my_pdf)  # Generating function for the degree distribution
        # # Assuming giant_component_size and make_G_u_minus_u are properly implemented
        # G_u_minus_u = make_G_u_minus_u(G)
        condition_nums[i,j] = l_x_algo_multiplicative(my_pdf, K=1000, conditions=[is_real, in_bounds], is_pgf = True)
        variances[i,j] = variance_sim_multiplicative(my_pdf, trials = 1000, conditions=[is_real, in_bounds])


print('Stop')
df_cond_nums = pd.DataFrame(condition_nums) 
df_cond_nums.rename(columns=lambda s:R0_vals[s], index=lambda s: alpha_vals[s], inplace = True)

fig, (ax1, ax2) = plt.subplots(1,2, sharey=True)

ax1 = sns.heatmap(df_cond_nums, cmap = "Blues", ax = ax1)
ax1.invert_yaxis()

df_variances = pd.DataFrame(variances) 
df_variances.rename(columns=lambda s:R0_vals[s], index=lambda s: alpha_vals[s], inplace = True)
ax2 = sns.heatmap(df_variances, cmap = "Greens", ax = ax2)

fmt = '{:0.2f}'
xticklabels = []
for item in ax1.get_xticklabels():
    item.set_text(fmt.format(float(item.get_text())))
    xticklabels += [item]
yticklabels = []
for item in ax1.get_yticklabels():
    item.set_text(fmt.format(float(item.get_text())))
    yticklabels += [item]

ax1.set_xticklabels(xticklabels)
ax1.set_yticklabels(yticklabels)

xticklabels = []
for item in ax2.get_xticklabels():
    item.set_text(fmt.format(float(item.get_text())))
    xticklabels += [item]

ax2.set_xticklabels(xticklabels)
ax2.invert_yaxis()
ax1.set_title("Condition numbers")
ax2.set_title("Variances")
plt.show()
# plt.savefig("Users/mcboudre/Documents/LSD_Lab/stochastic_polys/stochastic_pgfs/plots/condition_number_heatmap"+ date +"_1000trials.pdf", format = "pdf")

### Heat Map
X, Y = np.meshgrid(R0_vals, alpha_vals)
fig, (ax1, ax2) = plt.subplots(2, sharex=True)
levels = np.arange(0, np.max(condition_nums), 750)
CS = ax1.contour(X, Y, condition_nums, levels = levels)
# ax.clabel(CS, inline=True, fontsize=8)
ax1.set_title('Condition numbers')
# plt.show()


### Heat Map
# X, Y = np.meshgrid(R0_vals, alpha_vals)
# fig, ax = plt.subplots()
levels = np.arange(0, np.max(variances), 0.00045)
CS = ax2.contour(X, Y, variances, levels = levels)
# ax.clabel(CS, inline=True, fontsize=10)
ax2.set_title('Variances')
plt.show()
print("stop")