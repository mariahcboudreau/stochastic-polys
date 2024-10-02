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

def G(x, r, p):
    return ((1 - p) / (1 - p*x))**r


def solve_for_root(r, p):
    u_old = 0        # Dummy value
    u_new = 0.31416  # Dummy value
    while not np.allclose(u_old, u_new, rtol=1e-03, atol=1e-05):
        u_old = u_new
        u_new = G(u_old, r, p)
    return u_new

alpha_vals = np.linspace(0.01,1,60)
R0_vals = np.linspace(1,4,60)


R0_vals = np.linspace(1, 4.0, 1000)
alpha_vals = np.linspace(0.01, 1, 999)
X, Y = np.meshgrid(R0_vals, alpha_vals)
p = X/(Y+X)
r = Y
root_lines = solve_for_root(r, p)
outbreak_lines = 1 - root_lines


# alpha_vals = np.linspace(0.1,0.9,80)
# R0_vals = np.linspace(1,4,80)
# N_max = 10  # Maximum value for N in the distribution

date = "08-29-2024"


with open('/Users/mcboudre/Documents/LSD_Lab/stochastic-polys/data/root_lines_60res_'+date+'.npy', 'wb') as f:
    np.save(f, root_lines) 
with open('/Users/mcboudre/Documents/LSD_Lab/stochastic-polys/data/outbreak_lines_60res_'+date+'.npy', 'wb') as f:
    np.save(f, outbreak_lines)   
# with open('/Users/mcboudre/Documents/LSD_Lab/stochastic-polys/data/additive_condition_nums_60res_'+date+'.npy', 'rb') as f:
#     condition_nums_addative = np.load(f)
# with open('/Users/mcboudre/Documents/LSD_Lab/stochastic-polys/data/additive_variances_60res_'+date+'.npy', 'rb') as f:
#     var_addative = np.load(f, allow_pickle = True)
# with open('/Users/mcboudre/Documents/LSD_Lab/stochastic-polys/data/root_lines_80res_'+date+'.npy', 'rb') as f:
#     roots_lines = np.load(f)
# with open('/Users/mcboudre/Documents/LSD_Lab/stochastic-polys/data/outbreak_lines_80res_'+date+'.npy', 'rb') as f:
#     outbreak_lines = np.load(f)


#### Contour plot
# import matplotlib as mpl
# mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams["font.family"] = "Times New Roman"

X, Y = np.meshgrid(R0_vals, alpha_vals)
fig, (ax1, ax2) = plt.subplots(ncols = 2,figsize = (10,5), sharey=True)
levels = np.arange(0, np.max(root_lines), 0.025)
CS = ax1.contour(X, Y, root_lines, levels=levels, cmap = 'copper')
ax1.clabel(CS, inline=True, fontsize=8)
ax1.set(ylabel = "Dispersion parameter", xlabel= r'Average secondary cases, $R_0$', title=r'Polynomial root, $u$')


levels = np.arange(0.025, np.max(outbreak_lines), 0.025)
CS = ax2.contour(X, Y, outbreak_lines, levels=levels, cmap = "copper")
ax2.clabel(CS, inline=True, fontsize=8)
ax2.set(xlabel= r'Average secondary cases, $R_0$', title=r'Outbreak size, $1-u$')

# plt.show()
plt.savefig("/Users/mcboudre/Documents/LSD_Lab/stochastic-polys/figures/outbreak_contours_correctdist_r0_1-4_alpha_001-1_copper_"+date+".pdf", format = "pdf")

