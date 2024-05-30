from stochastic_pgfs.laub_xia_algo import l_x_algo, is_real, in_bounds
from stochastic_pgfs.pgfs import PGF, make_G_u_minus_u
import numpy as numpy
from stochastic_pgfs.sim import *
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import nbinom
from datetime import datetime
date = datetime.today().strftime('%m-%d-%Y')


# Define the matrix size or retrieve it from existing data
alpha_vals = np.arange(0.1,0.3,0.01)  # Example values
R0_vals = np.arange(1,5, 0.1)  # Example values
N_max = 10  # Maximum value for N in the distribution
variances = np.zeros((len(alpha_vals), len(R0_vals)))

for i in range(len(alpha_vals)):
    for j in range(len(R0_vals)):
        # k = alpha_vals[i]
        # R0 = R0_vals[j]
        # coeffs = neg_binom(R0, k, N_max)
        alpha = alpha_vals[i]
        R0 = R0_vals[j]
        r = 1 / alpha #dispersion parameter
        p = alpha / (R0 + alpha) #Changed this parameter
        dist = nbinom(n=r, p=p)  # Scipy parameterizes differently, check this
        my_pdf = dist.pmf(range(N_max + 1))
        variances[i,j] = variance_sim(my_pdf, trials = 1000, conditions=[is_real, in_bounds])

### Heat Map
X, Y = np.meshgrid(R0_vals, alpha_vals)
fig, ax = plt.subplots()
levels = np.arange(0, 0.4, 0.01)
CS = ax.contourf(X, Y, variances, levels = levels)
ax.clabel(CS, inline=True, fontsize=10)
ax.set_title('Variances')
plt.show()
print("stop")



df_variances = pd.DataFrame(variances) 
sns.heatmap(df_variances, yticklabels= alpha_vals, xticklabels= R0_vals, cmap = "Greens")
plt.savefig("Users/mcboudre/Documents/LSD_Lab/stochastic_polys/stochastic_pgfs/plots/variance_heatmap"+ date +"_1000trials.pdf", format = "pdf")
print(variances)