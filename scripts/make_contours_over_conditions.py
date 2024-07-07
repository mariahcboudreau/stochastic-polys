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


alpha_vals = np.linspace(0.1,0.9,80)
R0_vals = np.linspace(1,4,80)
N_max = 10  # Maximum value for N in the distribution

date = "07-06-2024"

with open('/Users/mcboudre/Documents/LSD_Lab/stochastic-polys/data/additive_condition_nums_80res_'+date+'.npy', 'rb') as f:
    condition_nums_addative = np.load(f)
with open('/Users/mcboudre/Documents/LSD_Lab/stochastic-polys/data/additive_variances_80res_'+date+'.npy', 'rb') as f:
    var_addative = np.load(f)
with open('/Users/mcboudre/Documents/LSD_Lab/stochastic-polys/data/root_lines_80res_'+date+'.npy', 'rb') as f:
    roots_lines = np.load(f)
with open('/Users/mcboudre/Documents/LSD_Lab/stochastic-polys/data/outbreak_lines_80res_'+date+'.npy', 'rb') as f:
    outbreak_lines = np.load(f)

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams["font.family"] = "Times New Roman"
X, Y = np.meshgrid(R0_vals, alpha_vals)
fig, ax = plt.subplots(ncols = 2,figsize = (10,5), sharey=True)

#### Contour plot
# X, Y = np.meshgrid(R0_vals, alpha_vals)
# fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize = (10,5))
levels = np.arange(0, np.max(roots_lines), 0.025)
CS = ax[0].contour(X, Y, roots_lines, levels = levels, cmap = "copper")
ax[0].clabel(CS, inline=True, fontsize=8)

levels = np.arange(0.025, np.max(outbreak_lines), 0.025)
CS = ax[1].contour(X, Y, outbreak_lines, levels = levels, cmap = "copper")
ax[1].clabel(CS, inline=True, fontsize=8)


ax[0].imshow(condition_nums_addative,
            extent = (min(R0_vals),max(R0_vals),min(alpha_vals),max(alpha_vals)),
             origin = 'lower',aspect = 'auto', cmap = 'Greens'
            )
ax[0].set(xlabel= r'Average secondary cases, $R_0$', ylabel=r'Dispersion parameter', title='Condition numbers with polynomial roots')

ax[1].imshow(condition_nums_addative,
            extent = (min(R0_vals),max(R0_vals),min(alpha_vals),max(alpha_vals)),
             origin = 'lower', aspect = 'auto', cmap = 'Greens'
            )
ax[1].set(xlabel= r'Average secondary cases, $R_0$', title='Condition numbers with outbreak sizes')


plt.tight_layout()
plt.show()
#plt.savefig("/Users/mcboudre/Documents/LSD_Lab/stochastic-polys/figures/additive_conds_withoutbreak_correctdist_r0_1-4_alpha_01-09_greens_80res_"+date+".pdf", format = "pdf")

