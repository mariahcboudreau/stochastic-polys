import numpy as np
import timeit
from stochastic_pgfs.pgfs import PGF,make_G_u_minus_u,numerical_inversion,percolated_pgf
from stochastic_pgfs.viz_utils import condition_number_heatmap
from stochastic_pgfs.laub_xia_algo import l_x_algo, is_real, in_bounds,_solve_self_consistent_equation,fast_polynomial_roots
import numpy as np
from scipy.stats import nbinom
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from functools import partial
import scipy.stats as stats

import matplotlib as mpl
import copy
date = datetime.today().strftime('%m-%d-%Y')


N = 1000
#test_coefs = np.array([1/2,-1/2])
test_coefs = np.random.rand(10)
test_coefs /= np.sum(test_coefs)#normalize 
test_coefs[1] -= 1



timer = timeit.Timer(partial(fast_polynomial_roots,test_coefs))
execution_time = timer.timeit(number=5)  # Run the function once
print(f"np.roots execution time: {execution_time} seconds")

timer = timeit.Timer(partial(np.roots,test_coefs))
execution_time = timer.timeit(number=5)  # Run the function once
print(f"fast poly roots time: {execution_time} seconds")