from stochastic_pgfs.laub_xia_algo import l_x_algo, is_real, in_bounds
from stochastic_pgfs.pgfs import PGF, make_G_u_minus_u
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
        condition_nums[i,j] = l_x_algo(my_pdf, K=1000, conditions=[is_real, in_bounds], is_pgf = True)

# df_variances = pd.DataFrame(condition_nums) 
# sns.heatmap(df_variances, yticklabels= alpha_vals, xticklabels= R0_vals, cmap = "Blues")
# plt.savefig("Users/mcboudre/Documents/LSD_Lab/stochastic_polys/stochastic_pgfs/plots/condition_number_heatmap"+ date +"_1000trials.pdf", format = "pdf")

### Heat Map
X, Y = np.meshgrid(R0_vals, alpha_vals)
fig, ax = plt.subplots()
levels = np.arange(0, 14000, 500)
CS = ax.contourf(X, Y, condition_nums, levels = levels)
ax.clabel(CS, inline=True, fontsize=10)
ax.set_title('Condition numbers')
plt.show()
print("stop")