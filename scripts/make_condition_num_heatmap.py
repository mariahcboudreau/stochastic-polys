from stochastic_pgfs.laub_xia_algo import l_x_algo, is_real, in_bounds
from stochastic_pgfs.pgfs import PGF, make_G_u_minus_u
from stochastic_pgfs.viz_utils import condition_number_heatmap
import numpy as np
from scipy.stats import nbinom
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from functools import partial
date = datetime.today().strftime('%m-%d-%Y')



alpha_vals = np.linspace(0.1,0.3,20)
R0_vals = np.linspace(1,5,20)


N_max = 10  # Maximum value for N in the distribution

#create partial function for the condition number heatmap
my_K = 1000

lx_addative = partial(l_x_algo, K=10000, conditions=[is_real, in_bounds],is_pgf=True,perturbation_type='additive')
lx_multiplicative = partial(l_x_algo, K=10000, conditions=[is_real, in_bounds],is_pgf=True,perturbation_type='multiplicative')


condition_nums_addative= condition_number_heatmap(lx_addative, alpha_vals, R0_vals, N_max)
condition_nums_multiplicative = condition_number_heatmap(lx_multiplicative, alpha_vals, R0_vals, N_max)

### Heat Map
X, Y = np.meshgrid(R0_vals, alpha_vals)
fig, ax = plt.subplots(ncols = 2,figsize = (10,5))
ax[0].imshow(condition_nums_addative,
            extent = (min(R0_vals),max(R0_vals),min(alpha_vals),max(alpha_vals)),
             aspect = 'auto'
            )
ax[0].set(xlabel='R0 Values', ylabel='Alpha Values', title='Additive Condition Numbers')

ax[1].imshow(condition_nums_multiplicative,
            extent = (min(R0_vals),max(R0_vals),min(alpha_vals),max(alpha_vals)),
             aspect = 'auto'
            )
ax[1].set(xlabel='R0 Values', ylabel='Alpha Values', title='Multiplicative Condition Numbers')
plt.tight_layout()
plt.show()




