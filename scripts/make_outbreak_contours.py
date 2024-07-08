from stochastic_pgfs.laub_xia_algo import polynomial_roots, is_real, in_bounds, _solve_self_consistent_equation
from stochastic_pgfs.pgfs import PGF, make_G_u_minus_u
from stochastic_pgfs.viz_utils import outbreak_contours
import numpy as np
from stochastic_pgfs.sim import *
from scipy.stats import nbinom
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
from functools import partial
date = datetime.today().strftime('%m-%d-%Y')

def pgf_solver(my_poly_coef, conditions, solve_root = True):

    if np.sum(derivative(my_poly_coef)) > 1:
        my_pgf_coef = make_G_u_minus_u(my_poly_coef)
        all_og_roots = polynomial_roots(np.flip(my_pgf_coef))
        all_conditions = np.array([True] * len(all_og_roots))
        if conditions:  # Double checks they are roots
            all_conditions = np.logical_and.reduce(
                [cond(all_og_roots) for cond in conditions]
            )
        if len(all_og_roots[all_conditions]) > 1:
            og_roots = np.max(all_og_roots[all_conditions])
        elif len(all_og_roots[all_conditions]) == 0:
            og_roots = 1
        else:
            og_roots = all_og_roots[all_conditions][0]
    else:
        og_roots = 1
        my_pgf_coef = make_G_u_minus_u(my_poly_coef)
        all_og_roots = polynomial_roots(np.flip(my_pgf_coef))
        all_conditions = np.array([True] * len(all_og_roots))
        if conditions:  # Double checks they are roots
            all_conditions = np.logical_and.reduce(
                [cond(all_og_roots) for cond in conditions]
            )

    if solve_root:
        return og_roots
    else:
        return 1 - og_roots

# alpha_vals = np.linspace(0.1,0.9,15)
# R0_vals = np.linspace(1,4,15)


# N_max = 10  # Maximum value for N in the distribution

# #create partial function for the condition number heatmap
# my_K = 1000

# roots = partial(_solve_self_consistent_equation, conditions=[is_real, in_bounds], solve_root = True)
# outbreaks = partial(_solve_self_consistent_equation, conditions=[is_real, in_bounds], solve_root = False)


# roots_lines = outbreak_contours(roots, alpha_vals, R0_vals, N_max)
# outbreak_lines = outbreak_contours(outbreaks, alpha_vals, R0_vals, N_max)



alpha_vals = np.linspace(0.1,0.9,80)
R0_vals = np.linspace(1,4,80)
N_max = 10  # Maximum value for N in the distribution

date = "07-06-2024"

with open('/Users/mcboudre/Documents/LSD_Lab/stochastic-polys/data/additive_condition_nums_80res_'+date+'.npy', 'rb') as f:
    condition_nums_addative = np.load(f)
with open('/Users/mcboudre/Documents/LSD_Lab/stochastic-polys/data/additive_variances_80res_'+date+'.npy', 'rb') as f:
    var_addative = np.load(f, allow_pickle = True)
with open('/Users/mcboudre/Documents/LSD_Lab/stochastic-polys/data/root_lines_80res_'+date+'.npy', 'rb') as f:
    roots_lines = np.load(f)
with open('/Users/mcboudre/Documents/LSD_Lab/stochastic-polys/data/outbreak_lines_80res_'+date+'.npy', 'rb') as f:
    outbreak_lines = np.load(f)


#### Contour plot
# import matplotlib as mpl
# mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams["font.family"] = "Times New Roman"

X, Y = np.meshgrid(R0_vals, alpha_vals)
fig, (ax1, ax2) = plt.subplots(ncols = 2,figsize = (10,5), sharey=True)
levels = np.arange(0, np.max(roots_lines), 0.025)
CS = ax1.contour(X, Y, roots_lines, levels=levels, cmap = 'copper')
ax1.clabel(CS, inline=True, fontsize=8)
ax1.set(ylabel = "Dispersion parameter", xlabel= r'Average secondary cases, $R_0$', title=r'Polynomial root, $u$')


levels = np.arange(0.025, np.max(outbreak_lines), 0.025)
CS = ax2.contour(X, Y, outbreak_lines, levels=levels, cmap = "copper")
ax2.clabel(CS, inline=True, fontsize=8)
ax2.set(xlabel= r'Average secondary cases, $R_0$', title=r'Outbreak size, $1-u$')

plt.show()
#plt.savefig("/Users/mcboudre/Documents/LSD_Lab/stochastic-polys/figures/outbreak_contours_correctdist_r0_1-4_alpha_01-09_copper_80res_"+date+".pdf", format = "pdf")

