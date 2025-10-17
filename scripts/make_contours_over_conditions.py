
import numpy as np
from scipy.stats import nbinom
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from functools import partial


alpha_vals_cond = np.linspace(0.01,10,1000)
R0_vals_cond = np.arange(0.8, 4.025, 0.025)
R0_vals_inf = np.linspace(0.8, 4.0, 1000)
alpha_vals_inf = np.linspace(0.01, 10, 100)




date = "01-27-2025"

# with open('./data/additive_condition_nums_1000res_lo_001-10_025r0res_'+date+'.npy', 'rb') as f:
#     condition_nums_additive = np.load(f) # R_0 values are columns and alpha are rows 



with open('./data/outbreak_lines_infinite_'+date+'.npy', 'rb') as f:
    outbreaks = np.load(f) 

roots = 1-outbreaks

def find_indices_in_interval(arr, lower_bound, upper_bound):
  """
  Finds the indices of elements in an array that fall within a specified interval (inclusive).

  Args:
    arr: The input array.
    lower_bound: The lower bound of the interval.
    upper_bound: The upper bound of the interval.

  Returns:
    A NumPy array containing the indices of the elements within the interval.
  """
  indices = np.where((arr >= lower_bound) & (arr <= upper_bound))[0]
  return indices


result_indices_r0 = find_indices_in_interval(R0_vals_inf, 1.66, 4.62)
result_indices_k = find_indices_in_interval(alpha_vals_inf, 0.26, 0.69)

x = outbreaks[result_indices_k[0]:result_indices_k[-1], result_indices_r0[0]:result_indices_r0[-1]]

print( "Min:", np.min(x), "Max:", np.max(x))

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams["font.family"] = "Times New Roman"
from datetime import datetime
date = datetime.today().strftime('%m-%d-%Y')
X_inf, Y_inf = np.meshgrid(R0_vals_inf, alpha_vals_inf)
# fig, ax = plt.subplots(ncols = 2,figsize = (10,5), sharey=True)

# ### Contour plot
# X, Y = np.meshgrid(R0_vals, alpha_vals)
# fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize = (10,5))
# levels = np.arange(0, 1, 0.025)
# CS = ax[0].contour(X_inf, Y_inf, roots, levels = levels, cmap = 'cividis')
# ax[0].clabel(CS, inline=True, fontsize=8)


levels = np.arange(0.025, 1, 0.025)
CS = plt.contour(X_inf, Y_inf, outbreaks, levels = levels, cmap="cividis")
plt.clabel(CS, inline=True, fontsize=10)



# ax[0].imshow(condition_nums_additive,
#             extent = (0.8,4 ,min(alpha_vals_cond),max(alpha_vals_cond)),
#              origin = 'lower',aspect = 'auto', cmap = 'pink_r')
# ax[0].set(xlabel= r'Average secondary cases, $R_0$', ylabel=r'Dispersion parameter', title='Condition numbers')

# im = plt.imshow(condition_nums_additive,
#             extent = (0.8,4,min(alpha_vals_cond),max(alpha_vals_cond)),
#              origin = 'lower', aspect = 'auto', cmap = 'pink_r',
#             )
# ax[1].set(xlabel= r'Average secondary cases, $R_0$', title=r'Outbreak sizes, $1-u$')
# ax[0].set(ylabel = r'Disperion parameter, $k$', xlabel= r'Average secondary cases, $R_0$', title=r'Polynomial roots, $u$')
plt.title('Epidemic probability', fontsize = 18)
plt.xlabel(r'Average secondary cases, $R_0$', fontsize = 18)
plt.xticks(fontsize = 16)
plt.ylabel(r'Dispersion parameter, $k$', fontsize = 18)
plt.yticks(fontsize = 16)
# cbar = plt.colorbar(im)
# cbar.set_label('Condition number', rotation=270, labelpad=15)
# ax[1].set_yscale('log')
# ax[0].set_yscale('log')
plt.yscale('log')
plt.tight_layout()
plt.show()
# plt.savefig("./stochastic-polys/figures/epidemic_prob_contourscividis_100res_"+date+".pdf", format = "pdf")


# X_inf, Y_inf = np.meshgrid(R0_vals_inf, alpha_vals_inf)

# # ### Contour plot



# levels = np.arange(0.025, 1, 0.025)
# CS = plt.contour(X_inf, Y_inf, outbreaks, levels = levels, cmap="cividis")
# plt.clabel(CS, inline=True, fontsize=8)



# # ax[0].imshow(condition_nums_additive,
# #             extent = (0.8,4 ,min(alpha_vals_cond),max(alpha_vals_cond)),
# #              origin = 'lower',aspect = 'auto', cmap = 'pink_r')
# # ax[0].set(xlabel= r'Average secondary cases, $R_0$', ylabel=r'Dispersion parameter', title='Condition numbers')

# im = plt.imshow(condition_nums_additive,
#             extent = (0.8,4,min(alpha_vals_cond),max(alpha_vals_cond)),
#              origin = 'lower', aspect = 'auto', cmap = 'pink_r',
#             )
# # ax[1].set(xlabel= r'Average secondary cases, $R_0$', title=r'Outbreak sizes, $1-u$')
# # ax[0].set(ylabel = r'Disperion parameter, $k$', xlabel= r'Average secondary cases, $R_0$', title=r'Polynomial roots, $u$')
# plt.xlabel(r'Average secondary cases, $R_0$')
# plt.ylabel(r'Dispersion parameter, $k$')
# cbar = plt.colorbar(im)
# cbar.set_label('Condition number', rotation=270, labelpad=15)
# plt.yscale('log')
# plt.tight_layout()
# # plt.show()
# plt.savefig("./stochastic-polys/figures/outbreaks_condition_overlay_r0_08-4_alpha_001-1_cividis_100res_"+date+".pdf", format = "pdf")
