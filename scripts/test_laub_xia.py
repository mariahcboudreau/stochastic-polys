from stochastic_pgfs.laub_xia_algo import l_x_algo, is_real, in_bounds
from stochastic_pgfs.pgfs import PGF, make_G_u_minus_u
import numpy as np
from scipy.stats import nbinom
from scipy import special

# Define the matrix size or retrieve it from existing data
alpha_vals = np.linspace(0.1,0.2,10)  # Example values
R0_vals = np.arange(1,5)  # Example values
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
        condition_nums[i,j] = l_x_algo(my_pdf, K=10, conditions=[is_real, in_bounds], is_pgf = True)

# print(true_roots)
# print(perturbed_roots)
print(condition_nums)
