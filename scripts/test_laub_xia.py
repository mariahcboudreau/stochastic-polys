from stochastic_pgfs.laub_xia_algo import kappa_SCE, is_real, in_bounds
from stochastic_pgfs.pgfs import PGF, make_G_u_minus_u
import numpy as np
from scipy.stats import nbinom
from scipy import special

# Define the matrix size or retrieve it from existing data
alpha_vals = np.linspace(0.1,1.0,10)  # Example values
R0_vals = np.arange(1,5)  # Example values
N_max = 10  # Maximum value for N in the distribution
results = np.zeros((len(alpha_vals), len(R0_vals)))

for i in range(len(alpha_vals)):
    for j in range(len(R0_vals)):
        alpha = alpha_vals[i]
        R0 = R0_vals[j]
        r = 1 / alpha
        p = R0 / (R0 + alpha**(-1))
        dist = nbinom(n=r, p=p)  # Scipy parameterizes differently, check this
        my_pdf = dist.pmf(range(N_max + 1))
        G = PGF(my_pdf)  # Generating function for the degree distribution
        # Assuming giant_component_size and make_G_u_minus_u are properly implemented
        G_u_minus_u = make_G_u_minus_u(G)
        results[i, j] = kappa_SCE(G_u_minus_u, K=100, conditions=[is_real, in_bounds])

print(results)