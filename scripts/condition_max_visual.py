import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
import seaborn as sns
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams["font.family"] = "Times New Roman"
from matplotlib.patches import Rectangle



date = "03-13-2025"

with open('./data/additive_condition_nums_1000res_lo_001-10_025r0res_'+date+'.npy', 'rb') as f:
    condition_nums_additive = np.load(f) # R_0 values are columns and alpha are rows 


from datetime import datetime
date = datetime.today().strftime('%m-%d-%Y')



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



alpha_vals = np.linspace(0.01, 10, 1000)

# R0_vals = np.linspace(0.5,4,10)
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


result_indices_r0 = find_indices_in_interval(R0_vals, 1.52, 2.18)
result_indices_k = find_indices_in_interval(alpha_vals, 0.18, 0.54)

x = condition_nums_additive[result_indices_k[0]:result_indices_k[-1]+1, result_indices_r0[0]:result_indices_r0[-1]+1]
print( "Min:", np.min(x), "Max:", np.max(x))
print('stop')

# Create subplots
# ax_main = fig.add_subplot(gs[1, 0])    # Main heatmap
# ax_right = fig.add_subplot(gs[1, 1])   # Right margin
# ax_top = fig.add_subplot(gs[0, 0])     # Top margin
# ax_cbar = fig.add_subplot(gs[-1, :])    # Colorbar space

import matplotlib.colors as mcolors

# Plot the heatmap
im = plt.imshow(condition_nums_additive,
        extent = (min(R0_vals),max(R0_vals),min(alpha_vals),max(alpha_vals)),
            origin = 'lower', aspect = 'auto', cmap = mcolors.LinearSegmentedColormap.from_list("white_to_{}".format("#00234F"), ["white", "#00234F"])
        )



# Add colorbar at bottom
plt.colorbar(im, label=r'Statistical condition estimate, $\kappa_{SCE}$', location = 'top')


# Plot column patterns (top)
r0_maxs = np.max(condition_nums_additive, axis=0) # R_0 maxes 
r0_maxs_indices = np.argmax(condition_nums_additive, axis=0)

plt.scatter(R0_vals[20::5], alpha_vals[r0_maxs_indices[20::5]], linewidths= 1.5, facecolors='none', edgecolors="#2CD1C1")


# ax_top.plot(R0_vals, col_max, 'black')

# ax_top.set_title(r'Maximum $R_0$ condition number')

# Plot row patterns (right)
k_maxs = np.max(condition_nums_additive, axis=1)
k_maxs_indices = np.argmax(condition_nums_additive, axis = 1)

# Extract log-spaced indices
selected_indices = extract_log_spaced_indices(k_maxs, 60)

# Select corresponding values and create x-positions
selected_values = k_maxs[selected_indices]
x_positions = np.linspace(0, 1, len(selected_values))

plt.scatter(R0_vals[k_maxs_indices[selected_indices[15:]]], alpha_vals[selected_indices[15:]], linewidths= 1.5, facecolors='none', edgecolors="#E5CF53")

plt.scatter(1.2, 8.05, marker='s', facecolors='none', edgecolors='black', linewidths=1.5)
plt.scatter(1.4, 0.12, marker='^', facecolors='none', edgecolors='black', linewidths=1.5)
plt.scatter(3.8, 0.12, marker='D', facecolors='none', edgecolors='black', linewidths=1.5)


# Add labels
plt.xlabel( r'Average secondary cases, $R_0$')
plt.ylabel(ylabel=r'Dispersion parameter, $k$')
plt.yscale('log')

# plt.legend(bbox_to_anchor=(1.5, 1.175), loc='upper right', frameon = False)
plt.tight_layout()
plt.show()

# plt.savefig("./stochastic-polys/figures/heatmap_max_variancemarkers_blues_logscale_"+date+".pdf", format = "pdf")
