from .laub_xia_algo import l_x_algo, is_real, in_bounds
from .pgfs import PGF, make_G_u_minus_u
import numpy as np
from scipy.stats import nbinom

def condition_number_heatmap(lx_func, alpha_vals, R0_vals, N_max):
    condition_nums = np.zeros((len(alpha_vals), len(R0_vals)))
    for i in range(len(alpha_vals)):
        for j in range(len(R0_vals)):
            alpha = alpha_vals[i]
            R0 = R0_vals[j]
            r = 1 / alpha #dispersion parameter
            p = alpha / (R0 + alpha) #Changed this parameter
            dist = nbinom(n=r, p=p)  # Scipy parameterizes differently, check this
            my_pdf = dist.pmf(range(N_max + 1))
            condition_nums[i,j] = lx_func(my_pdf)
    return condition_nums
