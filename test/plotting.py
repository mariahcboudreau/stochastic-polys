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
import pyarrow
date = datetime.today().strftime('%m-%d-%Y')



df = pd.read_parquet('./test/sce_implcit_results.pq') 

sce_rel_lx = df.pivot(index='k', columns='r0', values='sce_rel_lx').to_numpy()
sce_abs_lx = df.pivot(index='k', columns='r0', values='sce_abs_lx').to_numpy()
implicit_abs = df.pivot(index='k', columns='r0', values='sce_abs_impl_nonparam').to_numpy()
implicit_rel = df.pivot(index='k', columns='r0', values='sce_rel_impl_nonparam').to_numpy()





def extract_log_spaced_indices(values, num_points=20):
    """
    Extract indices that represent evenly spaced points on a logarithmic scale.
    
    Parameters:
    -----------
    values : array-like
        The input vector of values
    num_points : int, optional
        Number of points to extract (default is 10)
    
    Returns:
    --------
    numpy.ndarray
        Indices of selected points
    """
    # Convert to numpy array for consistency
    values = np.array(values)
    
    # Take log of absolute values (avoiding log(0))
    log_values = np.log10(np.abs(values) + 1)
    
    # Calculate logarithmically spaced points
    log_points = np.linspace(
        log_values.min(), 
        log_values.max(), 
        num=num_points
    )
    
    # Find indices closest to log-spaced points
    indices = []
    for point in log_points:
        # Find the index of the value closest to the log-spaced point
        closest_index = np.argmin(np.abs(log_values - point))
        
        # Ensure we don't select duplicate indices
        if closest_index not in indices:
            indices.append(closest_index)
    
    return np.array(indices)



k_vals = np.linspace(0.01, 10, 1000)
R0_vals = np.arange(0.8, 4.025, 0.025)


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


result_indices_r0 = find_indices_in_interval(R0_vals, 1.4, 3.825)
result_indices_k = find_indices_in_interval(k_vals, 0.1, 0.12)

x = condition_nums_additive[result_indices_k[0]:result_indices_k[-1]+1, result_indices_r0[0]:result_indices_r0[-1]+1]

# Define parameter values
k_vals = np.linspace(0.01, 10, 1000)
R0_vals = np.arange(0.8, 4.025, 0.025)   

import matplotlib as mpl
import matplotlib.colors as mcolors
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams["font.family"] = "Times New Roman"
X, Y = np.meshgrid(R0_vals, k_vals)
# fig, ax = plt.subplots(ncols = 2,figsize = (10,5), sharey=True)

# Plot the heatmap
im = plt.imshow(condition_nums_additive,
        extent = (min(R0_vals),max(R0_vals),min(k_vals),max(k_vals)),
            origin = 'lower', aspect = 'auto', cmap = mcolors.LinearSegmentedColormap.from_list("white_to_{}".format("#00234F"), ["white", "#00234F"])
        )



# Add colorbar at bottom
plt.colorbar(im, label=r'Statistical condition estimate, $\kappa_{SCE}$', location = 'top')


# Plot column patterns (top)
r0_maxs = np.max(condition_nums_additive, axis=0) # R_0 maxes 
r0_maxs_indices = np.argmax(condition_nums_additive, axis=0)

plt.scatter(R0_vals[20::5], k_vals[r0_maxs_indices[20::5]], linewidths= 1.5, facecolors='none', edgecolors="#2CD1C1")


# Plot row patterns (right)
k_maxs = np.max(condition_nums_additive, axis=1)
k_maxs_indices = np.argmax(condition_nums_additive, axis = 1)

# Extract log-spaced indices
selected_indices = extract_log_spaced_indices(k_maxs, 60)

# Select corresponding values and create x-positions
selected_values = k_maxs[selected_indices]
x_positions = np.linspace(0, 1, len(selected_values))

plt.scatter(R0_vals[k_maxs_indices[selected_indices[15:]]], k_vals[selected_indices[15:]], linewidths= 1.5, facecolors='none', edgecolors="#E5CF53")

plt.scatter(1.2, 8.05, marker='s', facecolors='none', edgecolors='black', linewidths=1.5)
plt.scatter(1.4, 0.12, marker='^', facecolors='none', edgecolors='black', linewidths=1.5)
plt.scatter(3.8, 0.12, marker='D', facecolors='none', edgecolors='black', linewidths=1.5)


# Add labels
plt.xlabel( r'Average secondary cases, $R_0$')
plt.ylabel(ylabel=r'Dispersion parameter, $k$')


# plt.legend(bbox_to_anchor=(1.5, 1.175), loc='upper right', frameon = False)
plt.tight_layout()
# plt.show()