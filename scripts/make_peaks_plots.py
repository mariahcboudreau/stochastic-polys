import os
import sys
sys.path.insert(1,os.getcwd())


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


import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams["font.family"] = "Times New Roman"
date = datetime.today().strftime('%m-%d-%Y')




vals = np.arange(0.8, 4.05, 0.05)
len_vals = np.arange(len(vals))

indices = np.array(vals)
for i in len_vals:
    indices[i] = np.where(R0_vals_cond >= vals[i])[0][0]

colors_r0_cond = plt.cm.magma(np.linspace(0.35,1,len(vals)))
colors_r0_inf = plt.cm.magma(np.linspace(0.35,1,len(R0_vals_inf)))

fig, axs = plt.subplots(2,2, sharey = 'row', sharex = 'col', figsize = (10,7))

for i in range(len(R0_vals_inf)):
    axs[0,1].plot(alpha_vals_inf, outbreaks[:, i], color = colors_r0_inf[i])

for ind, j in zip(indices, len_vals): 
    axs[1,1].plot(alpha_vals_cond, condition_nums_additive[:,int(ind)], color = colors_r0_cond[j])


colors_r0_cond = plt.cm.ScalarMappable(cmap=plt.cm.magma, norm=plt.Normalize(0.8, 4))
colors_r0_cond.set_array([])

cbar = plt.colorbar(colors_r0_cond, ax = [axs[0,1], axs[1,1]])
cbar.set_label(r'$R_0$ value', rotation=270, labelpad=15)

axs[1,1].set_xlabel(r"Dispersion parameter, $\alpha$")
axs[1,1].set_xlim(0,1)
# axs[1,1].set_ylabel("Condition number")
# axs[0,1].set_ylabel("Outbreak size")
# plt.show()



colors_cond = plt.cm.cividis(np.linspace(0,1,len(alpha_vals_cond)))

colors_inf = plt.cm.cividis(np.linspace(0,1,len(alpha_vals_inf)))

for i in range(len(alpha_vals_inf)):
    axs[0,0].plot(R0_vals_inf, outbreaks[i], color = colors_inf[i])

for i in range(len(alpha_vals_cond)):
    axs[1,0].plot(R0_vals_cond, condition_nums_additive[i], color = colors_cond[i])

axs[0,0].axvline(x = 1.0, color = 'grey', linestyle = 'dashed')
axs[1,0].axvline(x = 1.05, color = 'grey', linestyle = 'dashed')

axs[0,0].set_ylabel("Outbreak size")
axs[1,0].set_ylabel("Condition number")
axs[1,0].set_xlabel(r'Average secondary cases, $R_0$')


colors_alpha_cond = plt.cm.ScalarMappable(cmap=plt.cm.cividis, norm=plt.Normalize(0.01, 1))
colors_alpha_cond.set_array([])
cbar = plt.colorbar(colors_alpha_cond, ax=[axs[0,0], axs[1,0]])
cbar.set_label('Dispersion parameter', rotation=270, labelpad=15)

# plt.show()
plt.savefig("/Users/mar.boudreau/Documents/UVM/stochastic-polys/figures/peaks_plots_negativebinom_r0_08-4_alpha_001-1_cividis_magma_R_50res_"+date+".pdf", format = "pdf")


