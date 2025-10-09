import os
import sys
sys.path.insert(1,os.getcwd())



from stochastic_pgfs.pgfs import PGF, make_G_u_minus_u
from stochastic_pgfs.viz_utils import outbreak_contours
import numpy as np
from stochastic_pgfs.sim import *
from scipy.stats import nbinom
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime
from functools import partial


alpha_vals_cond = np.linspace(0.01,10 ,1000)
R0_vals_cond = np.arange(0.8, 4.025, 0.025)
R0_vals_inf = np.linspace(0.8, 4.0, 1000)
alpha_vals_inf = np.linspace(0.01, 10, 100)

# cmap = plt.cm.cividis
# cmap_r0 = cmap(np.linspace(0.5,1, cmap.N // 2))
# cmap_alpha = cmap(np.linspace(0,0.5, cmap.N // 2))

# Create a new colormap from the sliced colors
GREEN="#519E3E"
ORANGE="#EF8636"
TEAL="#37B1AB"
SALMON="#F77B61"
BLUE="#3B75AF"
GRAY="#CCCCCC"
# Get the two endpoints of the cividis colormap as hex values
cividis = mpl.cm.get_cmap('cividis')
CIVIDIS_START = mpl.colors.rgb2hex(cividis(0))
CIVIDIS_END = mpl.colors.rgb2hex(cividis(230))
#crete a custom colormap from these
n_bins = 100
colors_blue = [(0, GRAY), (1, CIVIDIS_START)]  # Define start (0) and end (1) points
cmap_blue = mpl.colors.LinearSegmentedColormap.from_list("custom_blue", colors_blue, N=n_bins)
# Create colormap from gray to salmon (changed from green)
colors_yellow = [(0, GRAY), (1, CIVIDIS_END)]  # Define start (0) and end (1) points
cmap_yellow = mpl.colors.LinearSegmentedColormap.from_list("custom_yellow", colors_yellow, N=n_bins)

date = "03-13-2025"

with open('./stochastic-polys/data//additive_condition_nums_1000res_lo_001-10_025r0res_'+date+'.npy', 'rb') as f:
    condition_nums_additive = np.load(f) # R_0 values are columns and alpha are rows 

date = "01-27-2025"

with open('./stochastic-polys/data/outbreak_lines_infinite_'+date+'.npy', 'rb') as f:
    outbreaks = np.load(f) 


import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams["font.family"] = "Times New Roman"
date = datetime.today().strftime('%m-%d-%Y')


vals = np.arange(0.8, 4.025, 0.025)
len_vals = np.arange(len(vals))

indices = np.array(vals)
for i in len_vals:
    indices[i] = np.where(R0_vals_cond >= vals[i])[0][0]

# Find the maximum value for alpha
max_alpha = np.max(condition_nums_additive, axis=1)
max_alpha_indices = np.argmax(condition_nums_additive, axis=1) # indices for alpha values

# Find the maximum value for R0
max_r0 = np.max(condition_nums_additive, axis=0)
max_r0_indices = np.argmax(condition_nums_additive, axis=0) # indices for alpha values


# Plot for alpha value lines

colors_r0_cond = cmap_yellow(np.linspace(0,1,len(vals) // 2))
colors_r0_inf = cmap_yellow(np.linspace(0,1,len(R0_vals_inf) // 2))

fig, axs = plt.subplots(2,2, sharey = 'row', sharex = 'col', figsize = (10,7))


for i in range(0, len(R0_vals_inf), 40):
    axs[0,1].plot(alpha_vals_inf, outbreaks[:, i], color = colors_r0_inf[i //2])

axs[0,1].text(0.015, 0.95, '(b)', transform=axs[0,1].transAxes, fontsize=14, fontweight='bold', va='top')
for ind, j in zip(indices[::5], len_vals[::5]): 
    axs[1,1].plot(alpha_vals_cond, condition_nums_additive[:,int(ind)], color = colors_r0_cond[j//2])
axs[1,1].text(0.015, 0.95, '(d)', transform=axs[1,1].transAxes, fontsize=14, fontweight='bold', va='top')

# axs[2,1].plot(alpha_vals_cond, max_alpha, color = 'black')

sm = plt.cm.ScalarMappable(cmap=cmap_yellow, norm=plt.Normalize(0.8, 4))
sm.set_array([])

cbar = plt.colorbar(sm, ax = [axs[0,1], axs[1,1]])
cbar.set_label(r'$R_0$', rotation = 0, labelpad=15)

axs[1,1].set_xlabel(r"Dispersion parameter, $k$", fontsize = 14)
axs[1,1].set_xscale('log')
axs[0,1].set_xscale('log')

# Annotate the maximum points for R0


# Plot for R_0 value lines

# colors_cond = plt.cm.cividis(np.linspace(0,0.5,len(alpha_vals_cond)))

# colors_inf = plt.cm.cividis(np.linspace(0,0.5,len(alpha_vals_inf)))
colors_alpha_cond = cmap_blue(np.linspace(0,1,len(alpha_vals_cond) // 2))
colors_alpha_inf = cmap_blue(np.linspace(0,1,len(alpha_vals_inf) // 2))

for i in range(0, len(alpha_vals_inf), 2):
    axs[0,0].plot(R0_vals_inf, outbreaks[i], color = colors_alpha_inf[i//2])

axs[0,0].text(0.015, 0.95, '(a)', transform=axs[0,0].transAxes, fontsize=14, fontweight='bold', va='top')
for i in range(0, len(alpha_vals_cond), 25):
    axs[1,0].plot(R0_vals_cond, condition_nums_additive[i], color = colors_alpha_cond[i//2])

axs[1,0].text(0.015, 0.95, '(c)', transform=axs[1,0].transAxes, fontsize=14, fontweight='bold', va='top')
# axs[2,0].plot(R0_vals_cond, max_r0, color = 'black')

axs[0,0].axvline(x = 1.0, color = 'black', linestyle = 'dashed', alpha = 0.5)
axs[1,0].axvline(x = 1.0, color = 'black', linestyle = 'dashed', alpha = 0.5)


axs[0,0].set_ylabel("Epidemic probability", fontsize = 14)
axs[1,0].set_ylabel(r"Statistical conditon estimate, $\kappa_{\mathrm{SCE}}$", fontsize = 14)
axs[1,0].set_xlabel(r'Average secondary cases, $R_0$', fontsize = 14)
axs[1,0].tick_params(axis = 'both', which = 'major', labelsize = 12)
axs[0,0].tick_params(axis = 'y', which = 'major', labelsize = 12)
axs[1,1].tick_params(axis = 'x', which = 'major', labelsize = 12)


sm = plt.cm.ScalarMappable(cmap=cmap_blue, norm=plt.Normalize(0.01, 10))
sm.set_array([])
cbar = plt.colorbar(sm, ax=[axs[0,0], axs[1,0]])
cbar.set_label(r'$k$', rotation = 0, labelpad=15)

from datetime import datetime
date = datetime.today().strftime('%m-%d-%Y')

plt.show()
# plt.savefig("./stochastic-polys/figures/peaks_plots_negativebinom_final_"+date+".pdf", format = "pdf")


