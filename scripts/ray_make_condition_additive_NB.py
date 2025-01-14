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
date = datetime.today().strftime('%m-%d-%Y')

import ray



# Initialize Ray
ray.init()

# Define your simulation function
@ray.remote
def run_simulation(alpha, R0, N_max, K):
    lx_additive = partial(l_x_algo, K=K, conditions=[is_real, in_bounds], perturbation_type='additive')
    return condition_number_heatmap(lx_additive, [alpha], [R0], N_max)

alpha_vals = np.linspace(0.01, 1, 100)
R0_vals = np.arange(0.8, 4.05, 0.05)
N_max = 200
my_K = 100

# Create a list of tasks to run in parallel
tasks = [run_simulation.remote(alpha, R0, N_max, my_K) for alpha in alpha_vals for R0 in R0_vals]

# Execute the tasks in parallel and gather the results
results = ray.get(tasks)

# Process the results as needed
# For example, you can reshape the results into a 2D array if necessary
condition_nums_additive = np.array(results).reshape(len(alpha_vals), len(R0_vals))




### Heat Map for Condition number and variances 

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams["font.family"] = "Times New Roman"
X, Y = np.meshgrid(R0_vals, alpha_vals)
# fig, ax = plt.subplots(ncols = 2,figsize = (10,5), sharey=True)

plt.imshow(condition_nums_additive,
            extent = (min(R0_vals),max(R0_vals),min(alpha_vals),max(alpha_vals)),
             origin = 'lower', aspect = 'auto', cmap = 'viridis'
            )
plt.xlabel( r'Average secondary cases, $R_0$')
plt.ylabel(ylabel=r'Dispersion parameter')
plt.title('Condition Numbers')
plt.plot()

print('stop')


# # Saving all files

with open('/Users/mar.boudreau/Documents/UVM/stochastic-polys/data/additive_condition_nums_100res_'+date+'.npy', 'wb') as f:
    np.save(f, condition_nums_additive)