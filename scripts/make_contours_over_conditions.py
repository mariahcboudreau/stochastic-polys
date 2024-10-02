from stochastic_pgfs.laub_xia_algo import polynomial_roots, is_real, in_bounds, _solve_self_consistent_equation
from stochastic_pgfs.pgfs import PGF, make_G_u_minus_u
from stochastic_pgfs.viz_utils import outbreak_contours
import numpy as np
from stochastic_pgfs.sim import *
from scipy.stats import nbinom
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from functools import partial


alpha_vals_cond = np.linspace(0.01,1,60)
R0_vals_cond = np.linspace(1,4,60)
R0_vals_inf = np.linspace(1, 4.0, 1000)
alpha_vals_inf = np.linspace(0.01, 1, 999)




with open('/Users/mcboudre/Documents/LSD_Lab/stochastic-polys/data/additive_condition_nums_60res_08-26-2024.npy', 'rb') as f:
    condition_nums_addative = np.load(f)
# with open('/Users/mcboudre/Documents/LSD_Lab/stochastic-polys/data/additive_variances_60res_'+date+'.npy', 'rb') as f:
#     var_addative = np.load(f, allow_pickle = True)
date = "08-29-2024"
with open('/Users/mcboudre/Documents/LSD_Lab/stochastic-polys/data/root_lines_60res_'+date+'.npy', 'rb') as f:
    roots_lines = np.load(f)
with open('/Users/mcboudre/Documents/LSD_Lab/stochastic-polys/data/outbreak_lines_60res_'+date+'.npy', 'rb') as f:
    outbreak_lines = np.load(f)

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams["font.family"] = "Times New Roman"
X_inf, Y_inf = np.meshgrid(R0_vals_inf, alpha_vals_inf)
fig, ax = plt.subplots(ncols = 2,figsize = (10,5), sharey=True)

#### Contour plot
# X, Y = np.meshgrid(R0_vals, alpha_vals)
# fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize = (10,5))
levels = np.arange(0, np.max(roots_lines), 0.025)
CS = ax[0].contour(X_inf, Y_inf, roots_lines, levels = levels, cmap = "copper")
ax[0].clabel(CS, inline=True, fontsize=8)

levels = np.arange(0.025, np.max(outbreak_lines), 0.025)
CS = ax[1].contour(X_inf, Y_inf, outbreak_lines, levels = levels, cmap = "copper")
ax[1].clabel(CS, inline=True, fontsize=8)


ax[0].imshow(condition_nums_addative,
            extent = (min(R0_vals_cond),max(R0_vals_cond),min(alpha_vals_cond),max(alpha_vals_cond)),
             origin = 'lower',aspect = 'auto', cmap = 'Greens'
            )
ax[0].set(xlabel= r'Average secondary cases, $R_0$', ylabel=r'Dispersion parameter', title='Condition numbers with polynomial roots')

ax[1].imshow(condition_nums_addative,
            extent = (min(R0_vals_cond),max(R0_vals_cond),min(alpha_vals_cond),max(alpha_vals_cond)),
             origin = 'lower', aspect = 'auto', cmap = 'Greens'
            )
ax[1].set(xlabel= r'Average secondary cases, $R_0$', title='Condition numbers with outbreak sizes')


plt.tight_layout()
# plt.show()
plt.savefig("/Users/mcboudre/Documents/LSD_Lab/stochastic-polys/figures/additive_conds_withoutbreak_correctdist_r0_1-4_alpha_05-1_greens_60res_"+date+".pdf", format = "pdf")

