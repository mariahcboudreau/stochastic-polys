import os
import sys
sys.path.insert(1,os.getcwd())

from stochastic_pgfs.laub_xia_algo import l_x_algo
from stochastic_pgfs.pgfs import PGF, make_G_u_minus_u
from stochastic_pgfs.viz_utils import outbreak_contours, condition_number_heatmap 
import numpy as np 
from stochastic_pgfs.sim import *
from scipy.stats import nbinom
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from functools import partial
date = datetime.today().strftime('%m-%d-%Y')

# def pgf_solver(my_poly_coef, conditions, solve_root = True):

#     if np.sum(derivative(my_poly_coef)) > 1:
#         my_pgf_coef = make_G_u_minus_u(my_poly_coef)
#         all_og_roots = polynomial_roots(np.flip(my_pgf_coef))
#         all_conditions = np.array([True] * len(all_og_roots))
#         if conditions:  # Double checks they are roots
#             all_conditions = np.logical_and.reduce(
#                 [cond(all_og_roots) for cond in conditions]
#             )
#         if len(all_og_roots[all_conditions]) > 1:
#             og_roots = np.max(all_og_roots[all_conditions])
#         elif len(all_og_roots[all_conditions]) == 0:
#             og_roots = 1
#         else:
#             og_roots = all_og_roots[all_conditions][0]
#     else:
#         og_roots = 1
#         my_pgf_coef = make_G_u_minus_u(my_poly_coef)
#         all_og_roots = polynomial_roots(np.flip(my_pgf_coef))
#         all_conditions = np.array([True] * len(all_og_roots))
#         if conditions:  # Double checks they are roots
#             all_conditions = np.logical_and.reduce(
#                 [cond(all_og_roots) for cond in conditions]
#             )

#     if solve_root:
#         return og_roots
#     else:
#         return 1 - og_roots


#alpha_vals = np.linspace(0.01,1,10)
alpha_vals = np.linspace(0.01,1,10)

# R0_vals = np.linspace(0.5,4,10)
#R0_vals = np.arange(0.8, 4.05, 0.05)
R0_vals = np.arange(1.1, 4.05, 0.05)

N_max = 200 # Maximum value for N in the distribution
max_iter = int(1e8)
tol = 1e-10

#create partial function for the condition number heatmap
my_K = 1000

lx_additive = partial(l_x_algo, K=my_K, tol = tol, max_iter = max_iter)
# var_addative = partial(variance_sim, K=1000, conditions=[is_real, in_bounds],is_pgf=True,perturbation_type='additive')


# variance_sim_addative = condition_number_heatmap(var_addative, alpha_vals, R0_vals, N_max)
condition_nums_additive= condition_number_heatmap(lx_additive, alpha_vals, R0_vals, N_max)

# roots = partial(fast_polynomial_roots, solve_root = True)
# outbreaks = partial(fast_polynomial_roots, solve_root = False)
# roots = partial(pgf_solver, conditions=[is_real, in_bounds], solve_root = True)
# outbreaks = partial(pgf_solver, conditions=[is_real, in_bounds], solve_root = False)
# roots_lines = outbreak_contours(roots, alpha_vals, R0_vals, N_max)
# outbreak_lines = outbreak_contours(outbreaks, alpha_vals, R0_vals, N_max)
### Heat Map for Condition number and variances 

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams["font.family"] = "Times New Roman"
X, Y = np.meshgrid(R0_vals, alpha_vals)
# fig, ax = plt.subplots(ncols = 2,figsize = (10,5), sharey=True)

plt.imshow(condition_nums_additive,
            extent = (min(R0_vals),max(R0_vals),min(alpha_vals),max(alpha_vals)),
             origin = 'lower', aspect = 'auto', cmap = 'pink'
            )
plt.xlabel( r'Average secondary cases, $R_0$')
plt.ylabel(ylabel=r'Dispersion parameter')
plt.title('Condition Numbers')

plt.show()

# ax[1].imshow(variance_sim_addative,
#             extent = (min(R0_vals),max(R0_vals),min(alpha_vals),max(alpha_vals)),
#              origin = 'lower', aspect = 'auto', cmap = 'Greens'
#             )
# ax[1].set(xlabel= r'Average secondary cases, $R_0$', title='Variances of polynomial roots')
# plt.tight_layout()
<<<<<<< Updated upstream
plt.show()
# plt.savefig("/Users/mar.boudreau/Documents/UVM/stochastic-polys/figures/additive_conds_correctdist_r0_05-4_alpha_001-1_viridis_50res_"+date+".pdf", format = "pdf")
=======
# plt.show()
#plt.savefig("/Users/mar.boudreau/Documents/UVM/stochastic-polys/figures/additive_conds_correctdist_r0_05-4_alpha_001-1_viridis_50res_"+date+".pdf", format = "pdf")

>>>>>>> Stashed changes

print(condition_nums_additive)

# # Saving all files

# with open('/Users/mar.boudreau/Documents/UVM/stochastic-polys/data/additive_condition_nums_50res_'+date+'.npy', 'wb') as f:
#     np.save(f, condition_nums_additive)
# # with open('/Users/mcboudre/Documents/LSD_Lab/stochastic-polys/data/additive_variances_60res_'+date+'.npy', 'wb') as f:
# #     np.save(f, var_addative)
# # with open('/Users/mcboudre/Documents/LSD_Lab/stochastic-polys/data/root_lines_80res_'+date+'.npy', 'wb') as f:
# #     np.save(f, roots_lines) 
# # with open('/Users/mcboudre/Documents/LSD_Lab/stochastic-polys/data/outbreak_lines_80res_'+date+'.npy', 'wb') as f:
# #     np.save(f, outbreak_lines)   

# import matplotlib as mpl
# mpl.rcParams.update(mpl.rcParamsDefault)
# plt.rcParams["font.family"] = "Times New Roman"
# X, Y = np.meshgrid(R0_vals, alpha_vals)
# fig, ax = plt.subplots(ncols = 2,figsize = (10,5), sharey=True)

# #### Contour plot
# # X, Y = np.meshgrid(R0_vals, alpha_vals)
# # fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize = (10,5))
# levels = np.arange(0, np.max(roots_lines), 0.025)
# CS = ax[0].contour(X, Y, roots_lines, levels = levels, cmap = "copper")
# ax[0].clabel(CS, inline=True, fontsize=8)

# levels = np.arange(0.025, np.max(outbreak_lines), 0.025)
# CS = ax[1].contour(X, Y, outbreak_lines, levels = levels, cmap = "copper")
# ax[1].clabel(CS, inline=True, fontsize=8)


# # ax[0].imshow(condition_nums_addative,
# #             extent = (min(R0_vals),max(R0_vals),min(alpha_vals),max(alpha_vals)),
# #              origin = 'lower',aspect = 'auto', cmap = 'Greens'
# #             )
# # ax[0].set(xlabel= r'Average secondary cases, $R_0$', ylabel=r'Dispersion parameter', title='Condition numbers with polynomial roots')

# # ax[1].imshow(condition_nums_addative,
# #             extent = (min(R0_vals),max(R0_vals),min(alpha_vals),max(alpha_vals)),
# #              origin = 'lower', aspect = 'auto', cmap = 'Greens'
# #             )
# # ax[1].set(xlabel= r'Average secondary cases, $R_0$', title='Condition numbers with outbreak sizes')


# plt.tight_layout()
# plt.show()
# plt.savefig("/Users/mcboudre/Documents/LSD_Lab/stochastic-polys/figures/outbreak_contours_correctdist_r0_1-4_alpha_005-09_greens_80res_"+date+".pdf", format = "pdf")

# plt.savefig("/Users/mcboudre/Documents/LSD_Lab/stochastic-polys/figures/additive_conds_withoutbreak_correctdist_r0_1-4_alpha_005-09_greens_80res_"+date+".pdf", format = "pdf")

