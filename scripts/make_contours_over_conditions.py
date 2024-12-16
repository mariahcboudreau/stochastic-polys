
import numpy as np
from scipy.stats import nbinom
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from functools import partial


alpha_vals_cond = np.linspace(0.01,1,50)
R0_vals_cond = np.arange(0.8, 4.05, 0.05)
R0_vals_inf = np.linspace(0.8, 4.0, 1000)
alpha_vals_inf = np.linspace(0.01, 1, 50)




date = "11-21-2024"

with open('/Users/mar.boudreau/Documents/UVM/stochastic-polys/data/additive_condition_nums_50res_'+date+'.npy', 'rb') as f:
    condition_nums_additive = np.load(f) # R_0 values are columns and alpha are rows 

date = "11-22-2024"

with open('/Users/mar.boudreau/Documents/UVM/stochastic-polys/data/outbreak_lines_infinite_'+date+'.npy', 'rb') as f:
    outbreaks = np.load(f) 

roots = 1-outbreaks

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams["font.family"] = "Times New Roman"
X_inf, Y_inf = np.meshgrid(R0_vals_inf, alpha_vals_inf)
fig, ax = plt.subplots(ncols = 2,figsize = (10,5), sharey=True)

# ### Contour plot
# X, Y = np.meshgrid(R0_vals, alpha_vals)
# fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize = (10,5))
levels = np.arange(0, 1, 0.025)
CS = ax[0].contour(X_inf, Y_inf, roots, levels = levels, cmap = "cividis")
ax[0].clabel(CS, inline=True, fontsize=8)


levels = np.arange(0, 1, 0.025)
CS = ax[1].contour(X_inf, Y_inf, outbreaks, levels = levels, cmap = "cividis")
ax[1].clabel(CS, inline=True, fontsize=8)



ax[0].imshow(condition_nums_additive,
            extent = (0.8,4 ,min(alpha_vals_cond),max(alpha_vals_cond)),
             origin = 'lower',aspect = 'auto', cmap = 'pink_r')
ax[0].set(xlabel= r'Average secondary cases, $R_0$', ylabel=r'Dispersion parameter', title='Condition numbers')

im = plt.imshow(condition_nums_additive,
            extent = (0.8,4,min(alpha_vals_cond),max(alpha_vals_cond)),
             origin = 'lower', aspect = 'auto', cmap = 'pink_r',
            )
ax[1].set(xlabel= r'Average secondary cases, $R_0$', title=r'Outbreak sizes, $1-u$')
ax[0].set(ylabel = r'Disperion parameter, $\alpha$', xlabel= r'Average secondary cases, $R_0$', title=r'Polynomial roots, $u$')
# plt.xlabel(r'Average secondary cases, $R_0$')
# plt.ylabel('Dispersion parameter')
cbar = plt.colorbar(im)
cbar.set_label('Condition number', rotation=270, labelpad=15)
plt.tight_layout()
plt.show()
# plt.savefig("/Users/mar.boudreau/Documents/UVM/stochastic-polys/figures/additive_conds_onlyOutbreaks_r0_08-4_alpha_001-1_greens_50res_"+date+".pdf", format = "pdf")

# plt.savefig("/Users/mar.boudreau/Documents/UVM/stochastic-polys/figures/outbreaks_roots_r0_08-4_alpha_001-1_cividis_50res_"+date+".pdf", format = "pdf")

# X_cond, Y_cond = np.meshgrid(R0_vals_cond, alpha_vals_cond)

# levels = np.arange(np.min(condition_nums_additive), np.max(condition_nums_additive), 0.25)
# CS = plt.contour(X_cond, Y_cond, condition_nums_additive, cmap = "Greens")
# plt.clabel(CS, inline=True, fontsize=8)

# plt.show()
