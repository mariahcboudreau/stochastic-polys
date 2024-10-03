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



N,M = 16,16#resolution of the heatmap


R0_vals = np.linspace(0.3,1,N)#
#p_vals = np.linspace(0,2,M)#density for the ER graph
lmbd_vals = np.linspace(1,3,M)#density for the ER graph
alpha_vals  = np.linspace(1,4,M)#power law exponent
N_max = 10  # Maximum value for N in the distribution



#load data 

save_dir = 'data/'
er_condition_nums_addative = np.load(save_dir+'er_condition_nums_addative.npy')
powerlaw_condition_nums_addative = np.load(save_dir+'powerlaw_condition_nums_addative.npy')
er_condition_nums_multiplicative =  np.load(save_dir+'er_condition_nums_multiplicative.npy')
powerlaw_condition_nums_multiplicative = np.load(save_dir+'powerlaw_condition_nums_multiplicative.npy')


import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams["font.family"] = "Times New Roman"
X, Y = np.meshgrid(R0_vals, alpha_vals)
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams["font.family"] = "Times New Roman"



#### Contour plot
# X, Y = np.meshgrid(R0_vals, alpha_vals)
pl_extent = (min(R0_vals),max(R0_vals),min(alpha_vals),max(alpha_vals))
er_extent = (min(R0_vals),max(R0_vals),min(lmbd_vals),max(lmbd_vals))

def make_levels(data,extent,my_ax,X,Y,interval = 1):
    levels = np.arange(np.min(data), np.max(data), interval)
    #breakpoint()
    #CS = my_ax.contour(X, Y, data, levels = levels, cmap = "copper")
    #my_ax.clabel(CS, inline=True, fontsize=8)
    im = my_ax.imshow(data,
                extent = extent,
                 origin = 'lower',aspect = 'auto', cmap = 'Greens'
                )

    cbar = fig.colorbar(im, ax=my_ax)
    cbar.set_label("Condition Number")


fig, axes = plt.subplots(nrows = 2,ncols = 2, sharex=True, figsize = (10,5))

make_levels(er_condition_nums_addative,er_extent,axes[0,0],X,Y,interval = 1)
axes[0,0].set(xlabel = 'Percolation Threshold(T)', ylabel = r'Mean Degree $\langle k \rangle$',title = 'Erdős–Rényi Additive Noise')

make_levels(powerlaw_condition_nums_addative,pl_extent,axes[1,0],X,Y,interval = 100)
axes[1,0].set(xlabel = 'Percolation Threshold(T)', ylabel = r'Power Law Exponent $\alpha$',title = 'Power Law Additive Noise')


make_levels(er_condition_nums_multiplicative,er_extent,axes[0,1],X,Y,interval = 100)
axes[0,1].set(xlabel = 'Percolation Threshold(T)', ylabel = r'Mean Degree $\langle k \rangle$',title = 'Erdős–Rényi Multiplicative Noise')

make_levels(powerlaw_condition_nums_multiplicative,pl_extent,axes[1,1],X,Y,interval = 100)
axes[1,1].set(xlabel = 'Percolation Threshold(T)', ylabel = r'Power Law Exponent $ \alpha $',title = 'Power Law Multiplicative Noise')

plt.tight_layout()
plt.show()


#my_ax = ax[0]
#levels = np.arange(0.025, np.max(outbreak_lines), 0.025)
#CS = my_ax.contour(X, Y, outbreak_lines, levels = levels, cmap = "copper")
#my_ax.clabel(CS, inline=True, fontsize=8)
